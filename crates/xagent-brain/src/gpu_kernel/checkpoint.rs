//! In-memory capture and restore of a [`GpuKernel`]'s persistent state.
//!
//! Capture records one command encoder that copies every captured buffer into
//! one staging buffer, with exactly one counted submit and no compute pass.
//! Restore validates the whole checkpoint before the first write.
//!
//! # Derived buffers (not captured)
//!
//! | Buffer | Reason |
//! | --- | --- |
//! | `heightmap`, `biome` | Rebuilt from the world seed by `upload_world`. |
//! | `world_config_bufs` | Rewritten by `upload_world_config` before every dispatch. |
//! | `dispatch_args` | Written by the prepare pass before every indirect dispatch. |
//! | `food_grid`, `agent_grid` | Rebuilt by the global pass every cycle. |
//! | `collision_scratch` | Written by the collision pass before it is read. |
//! | `state_staging`, `food_staging`, `trail_staging`, telemetry staging | Readback staging only. |
//! | `global_credit` features buffer | Rewritten by the main kernel every cycle. |
//! | packed encoder buffer | Private copy of `brain_state`, refreshed after invalidation. |
//!
//! # Cache invalidation on restore
//!
//! `restore_checkpoint` clears every CPU-side cache `GpuKernel` owns:
//! `state_cache` (replaced), `food_cache_valid`, `cached_telemetry`,
//! `trail_cache` (rebuilt from `trail_ring`) and the packed encoder (its
//! `valid` flag through `invalidate_packed_encoder`, and its scratch prefix
//! through `packed_encoder::Cache::reset_for_restore`).
//!
//! The context, food-grid, sparse-grid, projection and encoder no-op cache
//! helpers named by review are not production state, and none can go stale:
//!
//! | Helper | Why restore has nothing to call |
//! | --- | --- |
//! | `food_grid_cache_validation::invalidate_metadata` | `#[cfg(test)]` module; writes zeros to the food-grid header in the packed cache prefix, which `reset_for_restore` zeroes. |
//! | `sparse_grid_validation` `invalidate_metadata` | `#[cfg(test)]` module; method on a test-local candidate struct that zeroes its header in the packed cache prefix. |
//! | `context_cache_validation` | `#[cfg(test)]` module; it only rewrites shader source and writes no buffer and no `GpuKernel` field. |
//! | `gathered_context_cache_validation` | `#[cfg(test)]` module; its header lives in the packed cache prefix and its candidate state is test-local, not a `GpuKernel` field. |
//! | `agent_projection_validation`, `fresh_projection_validation` `invalidate` | `#[cfg(test)]` modules; per-agent headers in the packed cache prefix, state held by the test-local arm struct. |
//! | `encoder_noop_cache_validation` `invalidate_all` | `#[cfg(test)]` module; zeroes a buffer owned by a test-local `CacheVariant`, never by `GpuKernel`. |
//!
//! None of these modules exists in a non-test build, and none holds a
//! `GpuKernel` field, so no CPU-side copy of their state survives a restore.
//! `kernel_checkpoint_restore_over_dispatched_packed_kernel` proves restore
//! over a dispatched kernel with the packed encoder enabled.

use super::{FoodKind, GpuKernel, TrunkLesions};

use std::fmt;

/// Section name of the agent physics rows.
pub const SECTION_AGENT_PHYS: &str = "agent_phys";
/// Section name of the decision buffer.
pub const SECTION_DECISION: &str = "decision";
/// Section name of the food state rows.
pub const SECTION_FOOD_STATE: &str = "food_state";
/// Section name of the food flags and claim slots.
pub const SECTION_FOOD_FLAGS: &str = "food_flags";
/// Section name of the sensory buffer.
pub const SECTION_SENSORY: &str = "sensory";
/// Section name of the next-cycle sensory buffer.
pub const SECTION_SENSORY_NEXT: &str = "sensory_next";
/// Section name of the brain state buffer.
pub const SECTION_BRAIN_STATE: &str = "brain_state";
/// Section name of the brain scratch buffer.
pub const SECTION_BRAIN_SCRATCH: &str = "brain_scratch";
/// Section name of the pattern memory buffer.
pub const SECTION_PATTERN: &str = "pattern";
/// Section name of the brain config uniform.
pub const SECTION_BRAIN_CONFIG: &str = "brain_config";
/// Section name of the trail ring.
pub const SECTION_TRAIL_RING: &str = "trail_ring";

const F32_BYTES: usize = std::mem::size_of::<f32>();
/// Width of an encoded `u64`.
const U64_BYTES: usize = std::mem::size_of::<u64>();
/// Width of an encoded `u32`.
const U32_BYTES: usize = std::mem::size_of::<u32>();
/// Width of an encoded `bool`.
const FLAG_BYTES: usize = 1;

/// Offset of `food_kind_draws`.
const CPU_OFFSET_DRAWS: usize = 0;
/// Offset of the nourishing-kind present flag.
const CPU_OFFSET_NOURISHING_PRESENT: usize = CPU_OFFSET_DRAWS + U64_BYTES;
/// Offset of the nourishing-kind second flag.
const CPU_OFFSET_NOURISHING_SECOND: usize = CPU_OFFSET_NOURISHING_PRESENT + FLAG_BYTES;
/// Offset of `active_config_index`.
const CPU_OFFSET_ACTIVE_CONFIG_INDEX: usize = CPU_OFFSET_NOURISHING_SECOND + FLAG_BYTES;
/// Offset of `freeze_steering_weights`.
const CPU_OFFSET_FREEZE_STEERING: usize = CPU_OFFSET_ACTIVE_CONFIG_INDEX + U32_BYTES;
/// Offset of the turn-weights lesion flag.
const CPU_OFFSET_LESION_TURN: usize = CPU_OFFSET_FREEZE_STEERING + FLAG_BYTES;
/// Offset of the forward-weights lesion flag.
const CPU_OFFSET_LESION_FORWARD: usize = CPU_OFFSET_LESION_TURN + FLAG_BYTES;
/// Offset of the memory-blend lesion flag.
const CPU_OFFSET_LESION_MEMORY_BLEND: usize = CPU_OFFSET_LESION_FORWARD + FLAG_BYTES;
/// Offset of the critic lesion flag.
const CPU_OFFSET_LESION_CRITIC: usize = CPU_OFFSET_LESION_MEMORY_BLEND + FLAG_BYTES;
/// Offset of `danger_percept_enabled`.
const CPU_OFFSET_DANGER_ENABLED: usize = CPU_OFFSET_LESION_CRITIC + FLAG_BYTES;
/// Offset of `danger_percept_blinded`.
const CPU_OFFSET_DANGER_BLINDED: usize = CPU_OFFSET_DANGER_ENABLED + FLAG_BYTES;
/// Offset of `brain_tick_probe`.
const CPU_OFFSET_TICK_PROBE: usize = CPU_OFFSET_DANGER_BLINDED + FLAG_BYTES;
/// Offset of `speed_cost_exponent`.
const CPU_OFFSET_SPEED_COST_EXPONENT: usize = CPU_OFFSET_TICK_PROBE + FLAG_BYTES;

/// Byte length of [`KernelCpuState::to_bytes`].
pub const CPU_STATE_LEN: usize = CPU_OFFSET_SPEED_COST_EXPONENT + F32_BYTES;

/// One captured buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelSection {
    pub name: &'static str,
    pub bytes: Vec<u8>,
}

/// Non-cache CPU fields that alter dispatch output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KernelCpuState {
    pub food_kind_draws: u64,
    pub nourishing_food_kind: Option<FoodKind>,
    pub active_config_index: u32,
    pub freeze_steering_weights: bool,
    pub trunk_lesions: TrunkLesions,
    pub danger_percept_enabled: bool,
    pub danger_percept_blinded: bool,
    pub brain_tick_probe: bool,
    pub speed_cost_exponent: f32,
}

fn push_bool(out: &mut Vec<u8>, value: bool) {
    out.push(u8::from(value));
}

impl KernelCpuState {
    /// Fixed little-endian layout, fields in declaration order. The nourishing
    /// kind is two bytes: present, then second.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(CPU_STATE_LEN);
        out.extend_from_slice(&self.food_kind_draws.to_le_bytes());
        push_bool(&mut out, self.nourishing_food_kind.is_some());
        push_bool(
            &mut out,
            matches!(self.nourishing_food_kind, Some(FoodKind::Second)),
        );
        out.extend_from_slice(&self.active_config_index.to_le_bytes());
        push_bool(&mut out, self.freeze_steering_weights);
        push_bool(&mut out, self.trunk_lesions.turn_weights);
        push_bool(&mut out, self.trunk_lesions.forward_weights);
        push_bool(&mut out, self.trunk_lesions.memory_blend);
        push_bool(&mut out, self.trunk_lesions.critic);
        push_bool(&mut out, self.danger_percept_enabled);
        push_bool(&mut out, self.danger_percept_blinded);
        push_bool(&mut out, self.brain_tick_probe);
        out.extend_from_slice(&self.speed_cost_exponent.to_bits().to_le_bytes());
        out
    }

    /// Inverse of [`to_bytes`](Self::to_bytes); any other length is refused.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, KernelRestoreError> {
        if bytes.len() != CPU_STATE_LEN {
            return Err(KernelRestoreError::CpuStateLength {
                expected: CPU_STATE_LEN,
                actual: bytes.len(),
            });
        }
        let flag = |offset: usize| bytes[offset] != 0;
        let mut draws = [0_u8; U64_BYTES];
        draws.copy_from_slice(&bytes[CPU_OFFSET_DRAWS..CPU_OFFSET_DRAWS + U64_BYTES]);
        let mut index = [0_u8; U32_BYTES];
        index.copy_from_slice(
            &bytes[CPU_OFFSET_ACTIVE_CONFIG_INDEX..CPU_OFFSET_ACTIVE_CONFIG_INDEX + U32_BYTES],
        );
        let mut exponent = [0_u8; F32_BYTES];
        exponent.copy_from_slice(
            &bytes[CPU_OFFSET_SPEED_COST_EXPONENT..CPU_OFFSET_SPEED_COST_EXPONENT + F32_BYTES],
        );
        let nourishing_food_kind = if flag(CPU_OFFSET_NOURISHING_PRESENT) {
            Some(if flag(CPU_OFFSET_NOURISHING_SECOND) {
                FoodKind::Second
            } else {
                FoodKind::First
            })
        } else {
            None
        };
        Ok(Self {
            food_kind_draws: u64::from_le_bytes(draws),
            nourishing_food_kind,
            active_config_index: u32::from_le_bytes(index),
            freeze_steering_weights: flag(CPU_OFFSET_FREEZE_STEERING),
            trunk_lesions: TrunkLesions {
                turn_weights: flag(CPU_OFFSET_LESION_TURN),
                forward_weights: flag(CPU_OFFSET_LESION_FORWARD),
                memory_blend: flag(CPU_OFFSET_LESION_MEMORY_BLEND),
                critic: flag(CPU_OFFSET_LESION_CRITIC),
            },
            danger_percept_enabled: flag(CPU_OFFSET_DANGER_ENABLED),
            danger_percept_blinded: flag(CPU_OFFSET_DANGER_BLINDED),
            brain_tick_probe: flag(CPU_OFFSET_TICK_PROBE),
            speed_cost_exponent: f32::from_bits(u32::from_le_bytes(exponent)),
        })
    }
}

/// Why a checkpoint was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelRestoreError {
    UnknownSection(String),
    MissingSection(&'static str),
    LengthMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    CpuStateLength {
        expected: usize,
        actual: usize,
    },
    DuplicateSection(&'static str),
    ConfigIndexOutOfRange(u32),
    /// A captured buffer's byte size does not fit `usize` on this target.
    SectionSizeOutOfRange(&'static str),
}

impl fmt::Display for KernelRestoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSection(name) => write!(f, "unknown checkpoint section {name:?}"),
            Self::MissingSection(name) => write!(f, "missing checkpoint section {name:?}"),
            Self::LengthMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "checkpoint section {name:?} has {actual} bytes, expected {expected}"
            ),
            Self::DuplicateSection(name) => write!(f, "duplicate checkpoint section {name:?}"),
            Self::SectionSizeOutOfRange(name) => {
                write!(f, "checkpoint section {name} size does not fit usize")
            }
            Self::ConfigIndexOutOfRange(index) => {
                write!(f, "checkpoint config index {index} does not fit usize")
            }
            Self::CpuStateLength { expected, actual } => write!(
                f,
                "checkpoint CPU state has {actual} bytes, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for KernelRestoreError {}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<F32_BYTES>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Persistent kernel state captured in one submit.
#[derive(Clone, Debug)]
pub struct KernelCheckpoint {
    sections: Vec<KernelSection>,
    cpu: KernelCpuState,
    physics: Vec<f32>,
    food: Vec<f32>,
}

impl KernelCheckpoint {
    /// Build a checkpoint from raw sections and the CPU state.
    pub fn new(sections: Vec<KernelSection>, cpu: KernelCpuState) -> Self {
        let find = |name: &str| {
            sections
                .iter()
                .find(|s| s.name == name)
                .map(|s| decode_f32(&s.bytes))
                .unwrap_or_default()
        };
        let physics = find(SECTION_AGENT_PHYS);
        let food = find(SECTION_FOOD_STATE);
        Self {
            sections,
            cpu,
            physics,
            food,
        }
    }

    pub fn sections(&self) -> &[KernelSection] {
        &self.sections
    }

    /// The captured CPU state.
    pub fn cpu(&self) -> &KernelCpuState {
        &self.cpu
    }

    /// The `agent_phys` rows.
    pub fn physics(&self) -> &[f32] {
        &self.physics
    }

    /// The `food_state` rows; `None` when that section is empty.
    pub fn food(&self) -> Option<&[f32]> {
        if self.food.is_empty() {
            None
        } else {
            Some(&self.food)
        }
    }
}

impl GpuKernel {
    /// Captured buffers with the byte length each section carries.
    fn captured_buffers(&self) -> Vec<(&'static str, &wgpu::Buffer, u64)> {
        let food = self.food_count > 0;
        let len = |b: &wgpu::Buffer, on: bool| if on { b.size() } else { 0 };
        vec![
            (
                SECTION_AGENT_PHYS,
                &self.agent_phys_buffer,
                self.agent_phys_buffer.size(),
            ),
            (
                SECTION_DECISION,
                &self.decision_buffer,
                self.decision_buffer.size(),
            ),
            (
                SECTION_FOOD_STATE,
                &self.food_state_buffer,
                len(&self.food_state_buffer, food),
            ),
            (
                SECTION_FOOD_FLAGS,
                &self.food_flags_buffer,
                len(&self.food_flags_buffer, food),
            ),
            (
                SECTION_SENSORY,
                &self.sensory_buffer,
                self.sensory_buffer.size(),
            ),
            (
                SECTION_SENSORY_NEXT,
                &self._sensory_next_buffer,
                self._sensory_next_buffer.size(),
            ),
            (
                SECTION_BRAIN_STATE,
                &self.brain_state_buffer,
                self.brain_state_buffer.size(),
            ),
            (
                SECTION_BRAIN_SCRATCH,
                &self.brain_scratch_buffer,
                self.brain_scratch_buffer.size(),
            ),
            (
                SECTION_PATTERN,
                &self.pattern_buffer,
                self.pattern_buffer.size(),
            ),
            (
                SECTION_BRAIN_CONFIG,
                &self.brain_config_buffer,
                self.brain_config_buffer.size(),
            ),
            (
                SECTION_TRAIL_RING,
                &self.trail_ring_buffer,
                self.trail_ring_buffer.size(),
            ),
        ]
    }

    fn cpu_state(&self) -> KernelCpuState {
        KernelCpuState {
            food_kind_draws: self.food_kind_draws,
            nourishing_food_kind: self.nourishing_food_kind,
            active_config_index: u32::try_from(self.active_config_index).unwrap_or_else(|_| {
                log::error!(
                    "[GpuKernel] active_config_index {} does not fit u32",
                    self.active_config_index
                );
                panic!("active_config_index does not fit u32");
            }),
            freeze_steering_weights: self.freeze_steering_weights,
            trunk_lesions: self.trunk_lesions,
            danger_percept_enabled: self.danger_percept_enabled,
            danger_percept_blinded: self.danger_percept_blinded,
            brain_tick_probe: self.brain_tick_probe,
            speed_cost_exponent: self.speed_cost_exponent,
        }
    }

    /// Capture every persistent buffer and CPU field with one command encoder,
    /// one staging buffer and exactly one counted submit.
    pub fn read_checkpoint_blocking(&mut self) -> KernelCheckpoint {
        let _vulkan = super::vulkan_gate::enter();
        let table = self.captured_buffers();
        let total: u64 = table.iter().map(|(_, _, len)| *len).sum();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_checkpoint_readback"),
            size: total.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut offsets = Vec::with_capacity(table.len());
        let mut offset = 0_u64;
        for (_, buffer, len) in &table {
            offsets.push(offset);
            if *len > 0 {
                encoder.copy_buffer_to_buffer(buffer, 0, &staging, offset, *len);
            }
            offset += *len;
        }
        self.submit_counted(encoder.finish());

        let slice = staging.slice(..);
        let (map_tx, map_rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = map_tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        match map_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                log::error!("[GpuKernel] checkpoint readback map failed: {err}");
                panic!("checkpoint readback map failed: {err}");
            }
            Err(err) => {
                log::error!("[GpuKernel] checkpoint readback callback dropped: {err}");
                panic!("checkpoint readback callback dropped: {err}");
            }
        }
        let data = slice.get_mapped_range();
        let sections: Vec<KernelSection> = table
            .iter()
            .zip(&offsets)
            .map(|((name, _, len), start)| {
                let convert = |value: u64| {
                    usize::try_from(value).unwrap_or_else(|_| {
                        log::error!("[GpuKernel] checkpoint offset {value} does not fit usize");
                        panic!("checkpoint offset does not fit usize");
                    })
                };
                let from = convert(*start);
                let to = convert(*start + *len);
                KernelSection {
                    name,
                    bytes: data[from..to].to_vec(),
                }
            })
            .collect();
        drop(data);
        staging.unmap();

        let checkpoint = KernelCheckpoint::new(sections, self.cpu_state());
        self.state_cache.clear();
        self.state_cache.extend_from_slice(checkpoint.physics());
        if let Some(food) = checkpoint.food() {
            self.food_cache.clear();
            self.food_cache.extend_from_slice(food);
            self.food_cache_valid = true;
        }
        self.collected_staging_sequence = self.next_staging_sequence - 1;
        checkpoint
    }

    /// Restore a checkpoint. Every section name and length and the CPU state
    /// length are validated before the first write.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &KernelCheckpoint,
    ) -> Result<(), KernelRestoreError> {
        let _vulkan = super::vulkan_gate::enter();
        let table = self.captured_buffers();
        for (i, section) in checkpoint.sections().iter().enumerate() {
            if let Some((name, _, _)) = table.iter().find(|(n, _, _)| *n == section.name) {
                if checkpoint.sections()[..i].iter().any(|s| s.name == *name) {
                    return Err(KernelRestoreError::DuplicateSection(name));
                }
            }
            let Some((name, _, expected)) = table.iter().find(|(n, _, _)| *n == section.name)
            else {
                return Err(KernelRestoreError::UnknownSection(section.name.to_string()));
            };
            let expected = usize::try_from(*expected)
                .map_err(|_| KernelRestoreError::SectionSizeOutOfRange(name))?;
            if section.bytes.len() != expected {
                return Err(KernelRestoreError::LengthMismatch {
                    name,
                    expected,
                    actual: section.bytes.len(),
                });
            }
        }
        for (name, _, _) in &table {
            if !checkpoint.sections().iter().any(|s| s.name == *name) {
                return Err(KernelRestoreError::MissingSection(name));
            }
        }
        let cpu = *checkpoint.cpu();
        let config_index = usize::try_from(cpu.active_config_index)
            .map_err(|_| KernelRestoreError::ConfigIndexOutOfRange(cpu.active_config_index))?;

        for section in checkpoint.sections() {
            if section.bytes.is_empty() {
                continue;
            }
            if let Some((_, buffer, _)) = table.iter().find(|(n, _, _)| *n == section.name) {
                self.queue.write_buffer(buffer, 0, &section.bytes);
            }
        }
        self.food_kind_draws = cpu.food_kind_draws;
        self.nourishing_food_kind = cpu.nourishing_food_kind;
        self.active_config_index = config_index;
        self.freeze_steering_weights = cpu.freeze_steering_weights;
        self.trunk_lesions = cpu.trunk_lesions;
        self.danger_percept_enabled = cpu.danger_percept_enabled;
        self.danger_percept_blinded = cpu.danger_percept_blinded;
        self.brain_tick_probe = cpu.brain_tick_probe;
        self.speed_cost_exponent = cpu.speed_cost_exponent;

        // The context, food-grid, gathered-context and physics cache headers
        // live in the packed cache scratch prefix; zero them and invalidate the
        // packed encoder. The CPU-side mirrors are invalidated or replaced here.
        // The only CPU-side validity state those caches have in production is
        // the packed cache `valid` flag, cleared by `reset_for_restore`; see
        // "Cache invalidation on restore" in the module docs.
        self.invalidate_packed_encoder();
        if let Some(cache) = self
            .global_credit
            .as_ref()
            .and_then(|credit| credit.packed_encoder.as_ref())
        {
            cache.reset_for_restore(&self.queue);
        }
        self.food_cache_valid = false;
        self.cached_telemetry = None;
        if let Some(trail) = checkpoint
            .sections()
            .iter()
            .find(|s| s.name == SECTION_TRAIL_RING)
        {
            self.trail_cache = decode_f32(&trail.bytes);
        }
        self.state_cache.clear();
        self.state_cache.resize(checkpoint.physics().len(), 0.0);
        self.state_cache.copy_from_slice(checkpoint.physics());
        Ok(())
    }
}
