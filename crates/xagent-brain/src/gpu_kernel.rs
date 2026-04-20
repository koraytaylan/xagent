//! Fused kernel: single-dispatch GPU simulation.
//!
//! Composes all phase WGSL fragments into one shader, creates a unified
//! buffer set and pipeline, and runs N ticks per dispatch(1,1,1).
//!
//! # Shader composition contract
//!
//! All pipelines start with `common.wgsl`, then concatenate phase-specific
//! fragments in a fixed order:
//!
//! | Pipeline | Fragments after `common.wgsl`                                         |
//! |----------|-----------------------------------------------------------------------|
//! | physics  | phase_clear, phase_food_grid, phase_physics, phase_death,             |
//! |          | phase_food_detect, phase_food_respawn, phase_agent_grid,              |
//! |          | phase_collision, physics_tick                                         |
//! | vision   | phase_vision, vision_tick                                             |
//! | brain    | brain_passes, brain_tick                                              |
//! | kernel   | brain_passes, kernel_tick                                             |
//! | global   | phase_clear, phase_food_grid, phase_food_respawn, phase_agent_grid,   |
//! |          | phase_collision, global_tick                                          |
//! | prepare  | phase_prepare_dispatch                                                |
//!
//! ## Pipeline-overridable constants
//!
//! `VISION_W` and `VISION_H` are declared as `override` in `common.wgsl` and
//! supplied by [`vision_override_constants`] at pipeline creation. All derived
//! sizes (`VISION_RAYS`, `FEATURE_COUNT`, the `O_*` offset cascade,
//! `BRAIN_STRIDE`, `FEATURES_STRIDE`) are also `override`-expressions chained
//! from these two inputs. No string replacement is required.
//!
//! ## Subgroup markers (stability contract)
//!
//! When `wgpu::Features::SUBGROUP` is available, [`apply_subgroup_markers`]
//! splices in the subgroup builtin and the subgroup-accelerated bitonic sort.
//! When subgroup is absent, the same function collapses the placeholders to
//! no-ops and the workgroup-memory fallback in `brain_passes.wgsl` is used.
//!
//! | Marker                                    | Site                                        |
//! |-------------------------------------------|---------------------------------------------|
//! | `// SUBGROUP_ENTRY_PARAMS`                | brain_tick.wgsl entry params                |
//! | `// KERNEL_SUBGROUP_ENTRY_PARAMS`         | kernel_tick.wgsl entry params               |
//! | `/* SUBGROUP_TOPK_PARAMS */`              | brain_passes.wgsl `coop_recall_topk`        |
//! | `/* SUBGROUP_TOPK_ARGS */`                | call sites of `coop_recall_topk`            |
//! | `/* KERNEL_SUBGROUP_TOPK_PARAMS */`       | kernel_tick.wgsl `brain_tick_inner`         |
//! | `/* KERNEL_SUBGROUP_TOPK_ARGS */`         | `brain_tick_inner` -> `coop_recall_topk`    |
//! | `/* KERNEL_SUBGROUP_TOPK_INNER_ARGS */`   | `kernel_tick` -> `brain_tick_inner`         |
//! | `// BEGIN_BITONIC_SORT`                   | fallback bitonic sort start (brain_passes)  |
//! | `// END_BITONIC_SORT`                     | fallback bitonic sort end (brain_passes)    |
//!
//! Tests in `gpu_kernel::tests::shader_composition` assert presence and
//! symmetry of every marker across the files that reference them. Do not
//! rename or move markers without updating both the helper and the tests.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

use wgpu;
use xagent_shared::{BrainConfig, WorldConfig};

use crate::buffers::*;

/// Replacement for the subgroup-accelerated bitonic sort. Loaded from a
/// dedicated WGSL file so the fragment stays validator-friendly.
const BITONIC_SORT_SUBGROUP_SRC: &str = include_str!("shaders/kernel/bitonic_sort_subgroup.wgsl");

/// Build the pipeline override map that feeds `VISION_W` / `VISION_H`
/// into the WGSL override cascade. The returned map is the single source
/// of truth for the vision-grid dimensions at pipeline creation time.
fn vision_override_constants(layout: &BrainLayout) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    map.insert("VISION_W".to_string(), f64::from(layout.vision_width));
    map.insert("VISION_H".to_string(), f64::from(layout.vision_height));
    map
}

/// Splice subgroup markers into a composed shader source.
///
/// `src` is expected to contain every marker listed in the module-level
/// doc block. When `has_subgroup` is true, markers are replaced with their
/// subgroup-enabled substitutions (including the bitonic-sort fragment from
/// `bitonic_sort_subgroup.wgsl`). When false, markers collapse to no-ops and
/// the workgroup-memory fallback in `brain_passes.wgsl` is retained.
fn apply_subgroup_markers(src: &str, has_subgroup: bool) -> String {
    let mut out = src.to_string();
    if has_subgroup {
        // Entry-point builtins
        let builtin = "@builtin(subgroup_invocation_id) sgid: u32,";
        out = out.replace("// SUBGROUP_ENTRY_PARAMS", builtin);
        out = out.replace("// KERNEL_SUBGROUP_ENTRY_PARAMS", builtin);

        // Function signatures and call sites
        out = out.replace("/* SUBGROUP_TOPK_PARAMS */", ", sgid: u32");
        out = out.replace("/* SUBGROUP_TOPK_ARGS */", ", sgid");
        out = out.replace("/* KERNEL_SUBGROUP_TOPK_PARAMS */", ", sgid: u32");
        out = out.replace("/* KERNEL_SUBGROUP_TOPK_ARGS */", ", sgid");
        out = out.replace("/* KERNEL_SUBGROUP_TOPK_INNER_ARGS */", ", sgid");

        // Replace the fallback bitonic-sort body with the subgroup variant.
        // `replace_fenced` reports an error if the fence markers disappear.
        out = replace_fenced(
            &out,
            "// BEGIN_BITONIC_SORT",
            "// END_BITONIC_SORT",
            BITONIC_SORT_SUBGROUP_SRC,
        );
    } else {
        // Strip placeholder comments so the fallback code compiles cleanly.
        out = out.replace("// SUBGROUP_ENTRY_PARAMS\n", "");
        out = out.replace("// KERNEL_SUBGROUP_ENTRY_PARAMS\n", "");
        out = out.replace(" /* SUBGROUP_TOPK_PARAMS */", "");
        out = out.replace(" /* SUBGROUP_TOPK_ARGS */", "");
        out = out.replace(" /* KERNEL_SUBGROUP_TOPK_PARAMS */", "");
        out = out.replace(" /* KERNEL_SUBGROUP_TOPK_ARGS */", "");
        out = out.replace(" /* KERNEL_SUBGROUP_TOPK_INNER_ARGS */", "");
    }
    out
}

/// Replace every region enclosed by `begin`/`end` (inclusive) with
/// `replacement`. Logs an error and returns the input untouched if either
/// marker is missing — callers treat this as a loud signal that the
/// composition contract has drifted.
fn replace_fenced(src: &str, begin: &str, end: &str, replacement: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut cursor = 0;
    let mut found_any = false;
    while let Some(begin_rel) = src[cursor..].find(begin) {
        let begin_abs = cursor + begin_rel;
        let Some(end_rel) = src[begin_abs..].find(end) else {
            log::error!("[GpuKernel] Unmatched composition fence: {begin} without trailing {end}");
            return src.to_string();
        };
        let end_abs = begin_abs + end_rel + end.len();
        out.push_str(&src[cursor..begin_abs]);
        out.push_str(replacement);
        cursor = end_abs;
        found_any = true;
    }
    if !found_any {
        log::error!("[GpuKernel] Composition fence {begin}..{end} not found — subgroup sort was not spliced in");
        return src.to_string();
    }
    out.push_str(&src[cursor..]);
    out
}

/// Number of async staging slots for state readback.
/// Six slots give async readback additional headroom while keeping the GPU
/// queue shallow enough that `queue.submit()` never blocks.
const STAGING_SLOTS: usize = 6;

/// Pending async readback of 4 staging buffers for telemetry.
struct TelemetryReadback {
    /// Number of map_async callbacks that have fired (need all 4).
    completed: Arc<AtomicU32>,
    /// Set if any map_async callback returned an error.
    had_error: Arc<AtomicBool>,
    agent_index: u32,
}

/// Pre-allocated staging buffers reused across telemetry readback requests.
struct TelemetryStagingBuffers {
    sensory: wgpu::Buffer,
    decision: wgpu::Buffer,
    brain: wgpu::Buffer,
    phys: wgpu::Buffer,
}

/// Lightweight telemetry for one agent's vision, motor, and brain-state readback.
#[derive(Clone, Debug)]
pub struct AgentTelemetry {
    pub vision_color: Vec<f32>,
    pub motor_fwd: f32,
    pub motor_turn: f32,
    pub mean_attenuation: f32,
    pub curiosity_bonus: f32,
    pub fatigue_factor: f32,
    pub staleness: f32,
    pub urgency: f32,
    pub gradient: f32,
    pub prediction_error: f32,
    pub exploration_rate: f32,
}

/// In-flight async readback of a single agent's brain state.
struct AgentStateReadback {
    brain_staging: wgpu::Buffer,
    pattern_staging: wgpu::Buffer,
    history_staging: wgpu::Buffer,
    brain_size: u64,
    pattern_size: u64,
    history_size: u64,
    /// Counts how many of the 3 map_async callbacks have fired successfully.
    mapped_count: Arc<AtomicU8>,
    /// Set if any map_async callback reports an error.
    had_error: Arc<AtomicBool>,
    brain_stride: usize,
}

/// Terrain heightmap vertices per side.
const TERRAIN_VPS: usize = 129;
/// Biome grid resolution (cells per side).
const BIOME_GRID_RES: usize = 256;

#[allow(dead_code)] // GPU buffers are read via bind groups, not Rust field access
pub struct GpuKernel {
    device: wgpu::Device,
    queue: wgpu::Queue,
    agent_count: u32,
    food_count: usize,

    // ── Shared buffers (14 storage + 2 uniform + 1 indirect) ──
    agent_phys_buffer: wgpu::Buffer,
    decision_buffer: wgpu::Buffer,
    heightmap_buffer: wgpu::Buffer,
    biome_buffer: wgpu::Buffer,
    world_config_bufs: [wgpu::Buffer; 2],
    food_state_buffer: wgpu::Buffer,
    food_flags_buffer: wgpu::Buffer,
    food_grid_buffer: wgpu::Buffer,
    agent_grid_buffer: wgpu::Buffer,
    collision_scratch_buffer: wgpu::Buffer,
    sensory_buffer: wgpu::Buffer,
    brain_state_buffer: wgpu::Buffer,
    pattern_buffer: wgpu::Buffer,
    history_buffer: wgpu::Buffer,
    brain_config_buffer: wgpu::Buffer,

    // ── Indirect dispatch ──
    dispatch_args_buffer: wgpu::Buffer,

    // ── Pipelines ──
    prepare_pipeline: wgpu::ComputePipeline,
    physics_pipeline: wgpu::ComputePipeline,
    vision_pipeline: wgpu::ComputePipeline,
    brain_pipeline: wgpu::ComputePipeline,
    kernel_pipeline: wgpu::ComputePipeline,
    global_pipeline: wgpu::ComputePipeline,
    vision_stride: u32,
    bind_groups: [wgpu::BindGroup; 2],
    active_config_index: usize,

    // ── Async state readback (STAGING_SLOTS ring, non-blocking) ──
    // Staging is decoupled from dispatch: in-flight readbacks do not
    // block new dispatches. Each slot contains state + food staging
    // buffers for one readback.
    // staging_ready counts completed map_async callbacks (need expected_staging_callbacks).
    state_staging: [wgpu::Buffer; STAGING_SLOTS],
    food_staging: [wgpu::Buffer; STAGING_SLOTS],
    staging_index: usize,                           // which buffer to write NEXT
    staging_in_flight: [bool; STAGING_SLOTS],       // submitted, not yet collected
    staging_ready: [Arc<AtomicU32>; STAGING_SLOTS], // map_async callbacks completed
    staging_had_error: [Arc<AtomicBool>; STAGING_SLOTS], // set if any map_async callback errored
    state_cache: Vec<f32>,
    food_cache: Vec<f32>,
    food_cache_valid: bool,

    // ── Async telemetry readback ──
    telemetry_staging: TelemetryStagingBuffers,
    pending_telemetry: Option<TelemetryReadback>,
    cached_telemetry: Option<AgentTelemetry>,

    // ── Async agent-state readback (for non-blocking generation transitions) ──
    agent_state_staging: Option<AgentStateReadback>,

    // ── Config ──
    world_config: WorldConfig,
    layout: BrainLayout,
    brain_tick_stride: u32,
    has_subgroup: bool, // retained for runtime diagnostics
}

impl GpuKernel {
    /// Check whether a GPU (or fallback CPU) adapter is available.
    pub fn is_available() -> bool {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .or_else(|| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            }))
        })
        .is_some()
    }

    /// Expose the wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Expose the wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Number of agents this kernel was sized for.
    pub fn agent_count(&self) -> u32 {
        self.agent_count
    }

    /// Re-initialize all per-agent GPU state for a new generation without
    /// recreating the device, pipelines, or buffers.  The caller must also
    /// call `upload_agents` afterwards to set physics positions.
    /// Reset all agent state using a deterministic RNG seed.
    /// Produces identical brain weights for the same seed + config,
    /// enabling reproducible test runs across resets.
    pub fn reset_agents_seeded(&mut self, brain_config: &BrainConfig, seed: u64) {
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        self.reset_agents_with_rng(brain_config, &mut rng);
    }

    pub fn reset_agents(&mut self, brain_config: &BrainConfig) {
        let mut rng = rand::rng();
        self.reset_agents_with_rng(brain_config, &mut rng);
    }

    fn reset_agents_with_rng(&mut self, brain_config: &BrainConfig, rng: &mut impl rand::Rng) {
        // Drain any pending async readback so staging buffers are clean.
        let expected = self.expected_staging_callbacks();
        for i in 0..STAGING_SLOTS {
            if self.staging_in_flight[i] {
                while self.staging_ready[i].load(Ordering::Acquire) < expected {
                    self.device.poll(wgpu::Maintain::Poll);
                }

                if self.staging_had_error[i].load(Ordering::Acquire) {
                    // map_async errored — buffer isn't mapped, just unmap and clear.
                    self.state_staging[i].unmap();
                    if self.food_count > 0 {
                        self.food_staging[i].unmap();
                    }
                } else {
                    let buf_size = (self.agent_count as usize * PHYS_STRIDE * 4) as u64;
                    let slice = self.state_staging[i].slice(..buf_size);
                    let _data = slice.get_mapped_range();
                    drop(_data);
                    self.state_staging[i].unmap();

                    if self.food_count > 0 {
                        let food_size = (self.food_count * FOOD_STATE_STRIDE * 4) as u64;
                        let food_slice = self.food_staging[i].slice(..food_size);
                        let _food_data = food_slice.get_mapped_range();
                        drop(_food_data);
                        self.food_staging[i].unmap();
                    }
                }

                self.staging_in_flight[i] = false;
            }
            self.staging_ready[i].store(0, Ordering::Release);
        }
        self.staging_index = 0;
        self.food_cache_valid = false;

        // Clear async telemetry state so stale readbacks from the
        // previous generation don't leak into the new one.
        self.unmap_telemetry_staging();
        self.pending_telemetry = None;
        self.cached_telemetry = None;

        let n = self.agent_count as usize;

        // Fresh brain state, pattern memory, and action history.
        let mut brain_data = Vec::with_capacity(n * self.layout.brain_stride);
        let mut pattern_data = Vec::with_capacity(n * PATTERN_STRIDE);
        let mut history_data = Vec::with_capacity(n * HISTORY_STRIDE);
        for _ in 0..n {
            brain_data.extend_from_slice(&init_brain_state_for(brain_config, &self.layout, rng));
            pattern_data.extend_from_slice(&init_pattern_memory());
            history_data.extend_from_slice(&init_action_history());
        }
        self.queue.write_buffer(
            &self.brain_state_buffer,
            0,
            bytemuck::cast_slice(&brain_data),
        );
        self.queue
            .write_buffer(&self.pattern_buffer, 0, bytemuck::cast_slice(&pattern_data));
        self.queue
            .write_buffer(&self.history_buffer, 0, bytemuck::cast_slice(&history_data));
        self.queue.write_buffer(
            &self.brain_config_buffer,
            0,
            bytemuck::cast_slice(&build_config_for(brain_config, &self.layout)),
        );
    }

    /// Create the fused kernel: device, buffers, composed shader, pipeline, bind group.
    pub fn new(
        agent_count: u32,
        food_count: usize,
        brain_config: &BrainConfig,
        world_config: &WorldConfig,
    ) -> Self {
        let n = agent_count as usize;
        let f = food_count;
        let gw = grid_width(world_config.world_size);
        let grid_cells = gw * gw;

        // ── wgpu device ──
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .or_else(|| {
            log::warn!("[GpuKernel] No GPU adapter found, trying fallback (CPU) adapter");
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            }))
        })
        .expect("No GPU or fallback adapter found");

        log::info!("[GpuKernel] Adapter: {:?}", adapter.get_info());

        let has_subgroup = adapter.features().contains(wgpu::Features::SUBGROUP);
        if has_subgroup {
            log::info!("[GpuKernel] Subgroup support detected — enabling subgroup intrinsics for brain shader");
        } else {
            log::info!("[GpuKernel] No subgroup support — using shared-memory-only bitonic sort");
        }

        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::default();
        required_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        required_limits.max_storage_buffers_per_shader_stage =
            adapter_limits.max_storage_buffers_per_shader_stage.min(16);

        let required_features = if has_subgroup {
            wgpu::Features::PUSH_CONSTANTS | wgpu::Features::SUBGROUP
        } else {
            wgpu::Features::PUSH_CONSTANTS
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gpu-kernel"),
                required_features,
                required_limits: {
                    required_limits.max_push_constant_size = 8;
                    required_limits
                },
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("Failed to create GPU device");

        let layout = BrainLayout::new(brain_config.vision_width, brain_config.vision_height);
        let brain_tick_stride = brain_config.brain_tick_stride;

        let storage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        // ── Storage buffers (13 read-write + 2 read-only) ──

        let agent_phys_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_agent_phys"),
            size: (n * PHYS_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let decision_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_decision"),
            size: (n * DECISION_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let heightmap_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_heightmap"),
            size: (TERRAIN_VPS * TERRAIN_VPS * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let biome_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_biome"),
            size: (BIOME_GRID_RES * BIOME_GRID_RES * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let world_config_bufs = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kernel_world_config_0"),
                size: (WORLD_CONFIG_SIZE * 4) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kernel_world_config_1"),
                size: (WORLD_CONFIG_SIZE * 4) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        ];
        let food_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_food_state"),
            size: ((f * FOOD_STATE_STRIDE * 4) as u64).max(4),
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let food_flags_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_food_flags"),
            size: ((f * 4) as u64).max(4),
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let food_grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_food_grid"),
            size: ((grid_cells * FOOD_GRID_CELL_STRIDE * 4) as u64).max(4),
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let agent_grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_agent_grid"),
            size: (grid_cells * AGENT_GRID_CELL_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let collision_scratch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_collision_scratch"),
            size: (n * 3 * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let sensory_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_sensory"),
            size: (n * layout.sensory_stride * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let brain_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_brain_state"),
            size: (n * layout.brain_stride * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let pattern_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_pattern"),
            size: (n * PATTERN_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let history_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_history"),
            size: (n * HISTORY_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let brain_config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_brain_config"),
            size: (CONFIG_SIZE * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dispatch_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_dispatch_args"),
            size: 6 * 4, // 2 × (x, y, z) u32 triplets
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Async state readback staging (STAGING_SLOTS ring buffer) ──
        let state_size = (n * PHYS_STRIDE * 4) as u64;
        let food_state_size = ((f * FOOD_STATE_STRIDE * 4) as u64).max(4);
        let staging_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let state_staging = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("kernel_state_staging_{i}")),
                size: state_size,
                usage: staging_usage,
                mapped_at_creation: false,
            })
        });
        let food_staging = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("kernel_food_staging_{i}")),
                size: food_state_size,
                usage: staging_usage,
                mapped_at_creation: false,
            })
        });

        // ── Pre-allocated telemetry staging buffers ──
        let tel_sensory_size = (layout.sensory_stride * 4) as u64;
        let tel_decision_size = (DECISION_STRIDE * 4) as u64;
        let tel_brain_size = (layout.brain_stride * 4) as u64;
        let tel_phys_size = (PHYS_STRIDE * 4) as u64;
        let make_tel_staging = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let telemetry_staging = TelemetryStagingBuffers {
            sensory: make_tel_staging("telemetry_sensory", tel_sensory_size),
            decision: make_tel_staging("telemetry_decision", tel_decision_size),
            brain: make_tel_staging("telemetry_brain", tel_brain_size),
            phys: make_tel_staging("telemetry_phys", tel_phys_size),
        };

        // ── Initialize persistent brain state ──
        let mut rng = rand::rng();
        let mut brain_data = Vec::with_capacity(n * layout.brain_stride);
        let mut pattern_data = Vec::with_capacity(n * PATTERN_STRIDE);
        let mut history_data = Vec::with_capacity(n * HISTORY_STRIDE);
        for _ in 0..n {
            brain_data.extend_from_slice(&init_brain_state_for(brain_config, &layout, &mut rng));
            pattern_data.extend_from_slice(&init_pattern_memory());
            history_data.extend_from_slice(&init_action_history());
        }
        queue.write_buffer(&brain_state_buffer, 0, bytemuck::cast_slice(&brain_data));
        queue.write_buffer(&pattern_buffer, 0, bytemuck::cast_slice(&pattern_data));
        queue.write_buffer(&history_buffer, 0, bytemuck::cast_slice(&history_data));
        queue.write_buffer(
            &brain_config_buffer,
            0,
            bytemuck::cast_slice(&build_config_for(brain_config, &layout)),
        );

        // ── Compose shader sources (see module-level composition contract) ──
        let common_src = include_str!("shaders/kernel/common.wgsl");

        // Physics pipeline: common + phase fragments + physics entry
        let physics_source = [
            common_src,
            include_str!("shaders/kernel/phase_clear.wgsl"),
            include_str!("shaders/kernel/phase_food_grid.wgsl"),
            include_str!("shaders/kernel/phase_physics.wgsl"),
            include_str!("shaders/kernel/phase_death.wgsl"),
            include_str!("shaders/kernel/phase_food_detect.wgsl"),
            include_str!("shaders/kernel/phase_food_respawn.wgsl"),
            include_str!("shaders/kernel/phase_agent_grid.wgsl"),
            include_str!("shaders/kernel/phase_collision.wgsl"),
            include_str!("shaders/kernel/physics_tick.wgsl"),
        ]
        .join("\n");

        // Vision pipeline: common + vision fragments + vision entry
        let vision_source = [
            common_src,
            include_str!("shaders/kernel/phase_vision.wgsl"),
            include_str!("shaders/kernel/vision_tick.wgsl"),
        ]
        .join("\n");

        // Brain pipeline: common + brain_passes + brain entry
        let brain_source = apply_subgroup_markers(
            &[
                common_src,
                include_str!("shaders/kernel/brain_passes.wgsl"),
                include_str!("shaders/kernel/brain_tick.wgsl"),
            ]
            .join("\n"),
            has_subgroup,
        );

        // Fused kernel pipeline: common + brain_passes + kernel entry
        let kernel_source = apply_subgroup_markers(
            &[
                common_src,
                include_str!("shaders/kernel/brain_passes.wgsl"),
                include_str!("shaders/kernel/kernel_tick.wgsl"),
            ]
            .join("\n"),
            has_subgroup,
        );

        // Global pipeline: common + global fragments + global entry
        let global_source = [
            common_src,
            include_str!("shaders/kernel/phase_clear.wgsl"),
            include_str!("shaders/kernel/phase_food_grid.wgsl"),
            include_str!("shaders/kernel/phase_food_respawn.wgsl"),
            include_str!("shaders/kernel/phase_agent_grid.wgsl"),
            include_str!("shaders/kernel/phase_collision.wgsl"),
            include_str!("shaders/kernel/global_tick.wgsl"),
        ]
        .join("\n");

        // Pipeline-overridable constants: VISION_W and VISION_H drive the
        // entire vision/feature/brain-offset cascade via override-expressions
        // in common.wgsl. All derived overrides evaluate at pipeline creation.
        let vision_overrides = vision_override_constants(&layout);
        let override_options = wgpu::PipelineCompilationOptions {
            constants: &vision_overrides,
            zero_initialize_workgroup_memory: true,
        };

        // ── Explicit bind group layout (all 16 bindings) ──
        // Each pipeline entry point only references a subset of bindings, but we
        // need a single shared layout so one bind group works for all 3 pipelines.
        use wgpu::{BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages};
        let storage_rw_entry = |binding: u32| -> BindGroupLayoutEntry {
            BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };
        let storage_ro_entry = |binding: u32| -> BindGroupLayoutEntry {
            BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };
        let uniform_entry = |binding: u32| -> BindGroupLayoutEntry {
            BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kernel_bgl"),
            entries: &[
                storage_rw_entry(0),  // agent_phys
                storage_rw_entry(1),  // decision
                storage_ro_entry(2),  // heightmap
                storage_ro_entry(3),  // biome
                uniform_entry(4),     // world_config
                storage_rw_entry(5),  // food_state
                storage_rw_entry(6),  // food_flags
                storage_rw_entry(7),  // food_grid
                storage_rw_entry(8),  // agent_grid
                storage_rw_entry(9),  // collision_scratch
                storage_rw_entry(10), // sensory
                storage_rw_entry(11), // brain_state
                storage_rw_entry(12), // pattern
                storage_rw_entry(13), // history
                uniform_entry(14),    // brain_config
                storage_rw_entry(15), // dispatch_args
            ],
        });
        // ── Physics pipeline layout (has push constants) ──
        let physics_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("physics_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..8,
            }],
        });

        // ── Brain pipeline layout (no push constants) ──
        let brain_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brain_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // ── Create physics pipeline ──
        let physics_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("physics_tick"),
            source: wgpu::ShaderSource::Wgsl(physics_source.into()),
        });
        let physics_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("physics_tick"),
            layout: Some(&physics_layout),
            module: &physics_module,
            entry_point: Some("physics_tick"),
            compilation_options: override_options.clone(),
            cache: None,
        });

        // ── Create vision pipeline (no push constants, like brain) ──
        let vision_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vision_tick"),
            source: wgpu::ShaderSource::Wgsl(vision_source.into()),
        });
        let vision_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vision_tick"),
            layout: Some(&brain_layout),
            module: &vision_module,
            entry_point: Some("vision_tick"),
            compilation_options: override_options.clone(),
            cache: None,
        });

        // ── Create brain pipeline ──
        let brain_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brain_tick"),
            source: wgpu::ShaderSource::Wgsl(brain_source.into()),
        });
        let brain_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brain_tick"),
            layout: Some(&brain_layout),
            module: &brain_module,
            entry_point: Some("brain_tick"),
            compilation_options: override_options.clone(),
            cache: None,
        });

        // ── Create prepare_dispatch pipeline (indirect dispatch args) ──
        let prepare_source = [
            common_src,
            include_str!("shaders/kernel/phase_prepare_dispatch.wgsl"),
        ]
        .join("\n");
        let prepare_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prepare_dispatch"),
            source: wgpu::ShaderSource::Wgsl(prepare_source.into()),
        });
        let prepare_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prepare_dispatch"),
            layout: Some(&brain_layout),
            module: &prepare_module,
            entry_point: Some("prepare_dispatch"),
            compilation_options: override_options.clone(),
            cache: None,
        });

        // ── Global pipeline layout (push constants for tick) ──
        let global_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("global_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..8,
            }],
        });

        // ── Create fused kernel pipeline (no push constants) ──
        let kernel_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kernel_tick"),
            source: wgpu::ShaderSource::Wgsl(kernel_source.into()),
        });
        let kernel_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kernel_tick"),
            layout: Some(&brain_layout),
            module: &kernel_module,
            entry_point: Some("kernel_tick"),
            compilation_options: override_options.clone(),
            cache: None,
        });

        // ── Create global pipeline ──
        let global_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("global_tick"),
            source: wgpu::ShaderSource::Wgsl(global_source.into()),
        });
        let global_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("global_tick"),
            layout: Some(&global_layout),
            module: &global_module,
            entry_point: Some("global_tick"),
            compilation_options: override_options.clone(),
            cache: None,
        });

        // ── Create bind groups (double-buffered on world_config) ──
        let make_bind_group = |wc_buf: &wgpu::Buffer, label: &str| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: agent_phys_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: decision_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: heightmap_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: biome_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wc_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: food_state_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: food_flags_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: food_grid_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: agent_grid_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: collision_scratch_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: sensory_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: brain_state_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: pattern_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: history_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: brain_config_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: dispatch_args_buffer.as_entire_binding(),
                    },
                ],
            })
        };
        let bind_groups = [
            make_bind_group(&world_config_bufs[0], "kernel_tick_bg_0"),
            make_bind_group(&world_config_bufs[1], "kernel_tick_bg_1"),
        ];

        Self {
            device,
            queue,
            agent_count,
            food_count,
            agent_phys_buffer,
            decision_buffer,
            heightmap_buffer,
            biome_buffer,
            world_config_bufs,
            food_state_buffer,
            food_flags_buffer,
            food_grid_buffer,
            agent_grid_buffer,
            collision_scratch_buffer,
            sensory_buffer,
            brain_state_buffer,
            pattern_buffer,
            history_buffer,
            brain_config_buffer,
            dispatch_args_buffer,
            prepare_pipeline,
            physics_pipeline,
            vision_pipeline,
            brain_pipeline,
            kernel_pipeline,
            global_pipeline,
            vision_stride: brain_config.vision_stride,
            bind_groups,
            active_config_index: 0,
            state_staging,
            food_staging,
            staging_index: 0,
            staging_in_flight: [false; STAGING_SLOTS],
            staging_ready: std::array::from_fn(|_| Arc::new(AtomicU32::new(0))),
            staging_had_error: std::array::from_fn(|_| Arc::new(AtomicBool::new(false))),
            state_cache: vec![0.0; n * PHYS_STRIDE],
            food_cache: vec![0.0; f * FOOD_STATE_STRIDE],
            food_cache_valid: false,
            telemetry_staging,
            pending_telemetry: None,
            cached_telemetry: None,
            world_config: world_config.clone(),
            layout,
            agent_state_staging: None,
            brain_tick_stride,
            has_subgroup,
        }
    }

    /// Upload one-time world data: terrain, biomes, food positions.
    pub fn upload_world(
        &self,
        terrain_heights: &[f32],
        biome_grid: &[u32],
        food_positions: &[(f32, f32, f32)],
        food_consumed: &[bool],
        food_timers: &[f32],
    ) {
        self.queue.write_buffer(
            &self.heightmap_buffer,
            0,
            bytemuck::cast_slice(terrain_heights),
        );
        self.queue
            .write_buffer(&self.biome_buffer, 0, bytemuck::cast_slice(biome_grid));

        let mut food_data = Vec::with_capacity(food_positions.len() * FOOD_STATE_STRIDE);
        for (i, &(x, y, z)) in food_positions.iter().enumerate() {
            food_data.push(x);
            food_data.push(y);
            food_data.push(z);
            food_data.push(food_timers[i]);
        }
        self.queue
            .write_buffer(&self.food_state_buffer, 0, bytemuck::cast_slice(&food_data));

        let flags: Vec<u32> = food_consumed
            .iter()
            .map(|&c| if c { 1 } else { 0 })
            .collect();
        self.queue
            .write_buffer(&self.food_flags_buffer, 0, bytemuck::cast_slice(&flags));
    }

    /// Upload initial agent physics state.
    pub fn upload_agents(
        &self,
        agents: &[(glam::Vec3, f32, f32, usize, usize)], // (pos, max_energy, max_integrity, mem_cap, proc_slots)
    ) {
        let mut data = vec![0.0f32; self.agent_count as usize * PHYS_STRIDE];
        for (i, &(pos, max_e, max_i, mem_cap, proc_slots)) in agents.iter().enumerate() {
            let base = i * PHYS_STRIDE;
            data[base + P_POS_X] = pos.x;
            data[base + P_POS_Y] = pos.y;
            data[base + P_POS_Z] = pos.z;
            data[base + P_FACING_Z] = 1.0;
            data[base + P_ENERGY] = max_e;
            data[base + P_MAX_ENERGY] = max_e;
            data[base + P_INTEGRITY] = max_i;
            data[base + P_MAX_INTEGRITY] = max_i;
            data[base + P_PREV_ENERGY] = max_e;
            data[base + P_PREV_INTEGRITY] = max_i;
            data[base + P_ALIVE] = 1.0;
            data[base + P_MEMORY_CAP] = mem_cap as f32;
            data[base + P_PROCESSING_SLOTS] = proc_slots as f32;
        }
        self.queue
            .write_buffer(&self.agent_phys_buffer, 0, bytemuck::cast_slice(&data));
    }

    /// Minimum ticks per dispatch to guarantee at least one brain cycle.
    pub fn brain_tick_stride(&self) -> u32 {
        self.brain_tick_stride
    }

    /// The number of physics ticks in one full kernel batch
    /// (`vision_stride * brain_tick_stride`).  Dispatching in exact
    /// multiples of this value guarantees deterministic global-pass and
    /// vision-pass frequency regardless of how the total is decomposed.
    pub fn kernel_batch_size(&self) -> u32 {
        self.vision_stride * self.brain_tick_stride
    }

    /// Write world config uniform with batch parameters.
    pub fn upload_world_config(&self, start_tick: u64, ticks_to_run: u32) {
        self.upload_world_config_masked(start_tick, ticks_to_run, 0x7);
    }

    /// Write world config with explicit phase mask.
    /// Bit 0 = physics, bit 1 = vision, bit 2 = brain.
    pub fn upload_world_config_masked(&self, start_tick: u64, ticks_to_run: u32, phase_mask: u32) {
        let mut wc = build_world_config(
            &self.world_config,
            self.food_count,
            self.agent_count as usize,
            start_tick,
            ticks_to_run,
            self.vision_stride,
            self.brain_tick_stride,
        );
        wc[WC_PHASE_MASK] = phase_mask as f32;
        self.queue.write_buffer(
            &self.world_config_bufs[self.active_config_index],
            0,
            bytemuck::cast_slice(&wc),
        );
    }

    /// Write world config with explicit vision_stride override.
    /// Used by kernel dispatch to set the GPU loop count per batch.
    fn upload_world_config_with_cycles(
        &self,
        start_tick: u64,
        ticks_to_run: u32,
        phase_mask: u32,
        vision_stride_override: u32,
    ) {
        let mut wc = build_world_config(
            &self.world_config,
            self.food_count,
            self.agent_count as usize,
            start_tick,
            ticks_to_run,
            vision_stride_override,
            self.brain_tick_stride,
        );
        wc[WC_PHASE_MASK] = phase_mask as f32;
        self.queue.write_buffer(
            &self.world_config_bufs[self.active_config_index],
            0,
            bytemuck::cast_slice(&wc),
        );
    }

    /// Dispatch with explicit phase mask (for profiling).
    /// Bit 0 = physics, bit 1 = vision, bit 2 = brain.
    pub fn dispatch_batch_masked(&mut self, start_tick: u64, ticks_to_run: u32, phase_mask: u32) {
        let n = self.agent_count as usize;
        let buf_size = (n * PHYS_STRIDE * 4) as u64;

        let phys_mask = phase_mask & 0x1;
        let run_vision = (phase_mask & 0x2) != 0;
        let run_brain = (phase_mask & 0x4) != 0;

        self.upload_world_config_masked(start_tick, self.brain_tick_stride, phys_mask);

        let num_cycles = ticks_to_run / self.brain_tick_stride;
        let remainder = ticks_to_run % self.brain_tick_stride;

        // Chunk cycles to avoid Metal command buffer deadlock
        const CYCLES_PER_CHUNK: u32 = 100;
        let mut cycle = 0u32;
        while cycle < num_cycles {
            let chunk_end = (cycle + CYCLES_PER_CHUNK).min(num_cycles);
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dispatch_masked"),
                });
            // Prepare indirect dispatch args (once per chunk)
            if run_vision || run_brain {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.prepare_pipeline);
                pass.set_bind_group(0, &self.bind_groups[self.active_config_index], &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            for c in cycle..chunk_end {
                let base_tick = start_tick + (c as u64) * (self.brain_tick_stride as u64);
                let pc: [u32; 2] = [base_tick as u32, self.brain_tick_stride];
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.physics_pipeline);
                    pass.set_bind_group(0, &self.bind_groups[self.active_config_index], &[]);
                    pass.set_push_constants(0, bytemuck::cast_slice(&pc));
                    pass.dispatch_workgroups(1, 1, 1);
                }
                if run_vision {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.vision_pipeline);
                    pass.set_bind_group(0, &self.bind_groups[self.active_config_index], &[]);
                    pass.dispatch_workgroups_indirect(&self.dispatch_args_buffer, 0);
                }
                if run_brain {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.brain_pipeline);
                    pass.set_bind_group(0, &self.bind_groups[self.active_config_index], &[]);
                    pass.dispatch_workgroups_indirect(&self.dispatch_args_buffer, 12);
                }
            }
            self.queue.submit(std::iter::once(encoder.finish()));
            cycle = chunk_end;
        }

        // Remainder: physics only (no vision/brain)
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dispatch_masked_final"),
            });
        if remainder > 0 {
            let rem_base = start_tick + (num_cycles as u64) * (self.brain_tick_stride as u64);
            let pc: [u32; 2] = [rem_base as u32, remainder];
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.physics_pipeline);
                pass.set_bind_group(0, &self.bind_groups[self.active_config_index], &[]);
                pass.set_push_constants(0, bytemuck::cast_slice(&pc));
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        encoder.copy_buffer_to_buffer(
            &self.agent_phys_buffer,
            0,
            &self.state_staging[self.staging_index],
            0,
            buf_size,
        );
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();

        self.active_config_index = 1 - self.active_config_index;
    }

    /// Number of map_async callbacks expected per staging slot: 1 (phys) + 1 (food)
    /// when food exists, or just 1 when food_count is 0.
    fn expected_staging_callbacks(&self) -> u32 {
        if self.food_count > 0 {
            2
        } else {
            1
        }
    }

    /// Non-blocking: collect any ready staging buffers into state_cache + food_cache.
    /// Returns true if caches were updated.
    fn try_collect_staging(&mut self) -> bool {
        let n = self.agent_count as usize;
        let buf_size = (n * PHYS_STRIDE * 4) as u64;
        let expected = self.expected_staging_callbacks();
        let mut collected = false;
        for i in 0..STAGING_SLOTS {
            if !self.staging_in_flight[i] {
                continue;
            }
            if self.staging_ready[i].load(Ordering::Acquire) < expected {
                continue;
            }

            if self.staging_had_error[i].load(Ordering::Acquire) {
                // A map_async callback errored — abandon this slot.
                self.state_staging[i].unmap();
                if self.food_count > 0 {
                    self.food_staging[i].unmap();
                }
                self.staging_in_flight[i] = false;
                continue;
            }

            // Collect agent phys state
            let slice = self.state_staging[i].slice(..buf_size);
            let data = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&data);
            self.state_cache.clear();
            self.state_cache.extend_from_slice(floats);
            drop(data);
            self.state_staging[i].unmap();

            // Collect food state (only when food exists)
            if self.food_count > 0 {
                let food_size = (self.food_count * FOOD_STATE_STRIDE * 4) as u64;
                let food_slice = self.food_staging[i].slice(..food_size);
                let food_data = food_slice.get_mapped_range();
                let food_floats: &[f32] = bytemuck::cast_slice(&food_data);
                self.food_cache.clear();
                self.food_cache.extend_from_slice(food_floats);
                drop(food_data);
                self.food_staging[i].unmap();
                self.food_cache_valid = true;
            }

            self.staging_in_flight[i] = false;
            collected = true;
        }
        collected
    }

    /// Dispatch ticks via fused kernel + global + vision passes.
    ///
    /// Each kernel-batch runs `vision_stride` brain cycles in a single dispatch,
    /// followed by one global (grid+collisions) and one vision (raycasting) pass.
    /// Always dispatches — compute is decoupled from staging readback.
    /// Staging copy is opportunistic (skipped when all slots are in-flight).
    /// Always returns `true` (kept for API compatibility).
    ///
    /// # Vision ordering guarantee
    ///
    /// Within a single batch the pass order is:
    ///   1. `prepare`  — set up indirect dispatch args
    ///   2. `kernel`   — fused physics → food → death → brain (loops `vision_stride` times)
    ///   3. `global`   — grid rebuild + collisions
    ///   4. `vision`   — raycasting writes `sensory_buf`
    ///
    /// The brain work in step 2 consumes features from `sensory_buf`, so it reads
    /// the values produced by the vision pass of the **previous** batch. That
    /// one-batch sensory lag is intentional and consistent regardless of stride
    /// settings.
    ///
    /// Within each kernel inner cycle, the fused shader executes its physics
    /// phases before its brain phase in program order. However, this comment
    /// should not be read as a claim that `workgroupBarrier()` alone makes
    /// `agent_phys` storage-buffer writes visible across invocations; the brain's
    /// feature inputs come from `sensory_buf`, not from same-batch vision output.
    ///
    /// When `brain_tick_stride == vision_stride` there is exactly one vision
    /// pass per batch, i.e. one vision pass per `vision_stride` brain cycles
    /// (at the end of the batch). The batch covers
    /// `vision_stride * brain_tick_stride` physics ticks and the sensory lag
    /// is one batch = `vision_stride * brain_tick_stride` physics ticks.
    pub fn dispatch_batch(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {
        let n = self.agent_count as usize;
        let buf_size = (n * PHYS_STRIDE * 4) as u64;

        let brain_cycles = ticks_to_run / self.brain_tick_stride;
        let kernel_batches = brain_cycles / self.vision_stride;
        let remainder_cycles = brain_cycles % self.vision_stride;

        let mut tick_cursor = start_tick;

        // Each kernel-batch gets its own encoder+submit to ensure the uniform
        // write (world config with tick/vision_stride) is visible to its dispatches.
        // Each batch is only 4 passes (prepare + kernel + global + vision).
        let total_batches = kernel_batches + if remainder_cycles > 0 { 1 } else { 0 };

        for m in 0..total_batches {
            let is_remainder = m == kernel_batches;
            let cycles_this_batch = if is_remainder {
                remainder_cycles
            } else {
                self.vision_stride
            };
            let ticks_this_batch = cycles_this_batch * self.brain_tick_stride;

            // Upload world config with vision_stride override for this batch
            self.upload_world_config_with_cycles(
                tick_cursor,
                ticks_this_batch,
                0x7,
                cycles_this_batch,
            );

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dispatch_kernel"),
                });

            // Prepare indirect dispatch args (for vision)
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.prepare_pipeline);
                pass.set_bind_group(0, &self.bind_groups[self.active_config_index], &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            // Fused kernel: 1 pass, cycles_this_batch brain cycles
            // SAFETY: dispatch(agent_count, 1, 1) — one workgroup per agent.
            // Multi-thread passes may conditionally call cooperative helpers
            // that contain internal workgroup/storage barriers, so barrier
            // uniformity relies on the `alive` guard being workgroup-uniform.
            // The kernel makes this uniform by construction: thread 0 (the sole
            // writer of `P_ALIVE`) broadcasts the post-write value into a
            // `var<workgroup> s_alive` before each workgroupBarrier(), and all
            // other threads read `s_alive` rather than `physics_state[P_ALIVE]`.
            // See the top-of-file SAFETY INVARIANT in kernel_tick.wgsl before
            // changing the dispatch shape or the alive-uniformity contract.
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.kernel_pipeline);
                pass.set_bind_group(0, &self.bind_groups[self.active_config_index], &[]);
                pass.dispatch_workgroups(self.agent_count, 1, 1);
            }

            // Global pass: grid rebuild + collisions
            {
                let tick_for_global = tick_cursor + ticks_this_batch as u64;
                let gpc: [u32; 2] = [tick_for_global as u32, 0];
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.global_pipeline);
                pass.set_bind_group(0, &self.bind_groups[self.active_config_index], &[]);
                pass.set_push_constants(0, bytemuck::cast_slice(&gpc));
                pass.dispatch_workgroups(1, 1, 1);
            }

            // Vision pass: raycasting
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.vision_pipeline);
                pass.set_bind_group(0, &self.bind_groups[self.active_config_index], &[]);
                pass.dispatch_workgroups_indirect(&self.dispatch_args_buffer, 0);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
            tick_cursor += ticks_this_batch as u64;
        }

        // Physics-only remainder (ticks that don't fill a brain cycle)
        let physics_remainder = ticks_to_run % self.brain_tick_stride;
        if physics_remainder > 0 {
            self.upload_world_config_masked(tick_cursor, physics_remainder, 0x1);
            let pc: [u32; 2] = [tick_cursor as u32, physics_remainder];
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dispatch_kernel_physics_remainder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.physics_pipeline);
                pass.set_bind_group(0, &self.bind_groups[self.active_config_index], &[]);
                pass.set_push_constants(0, bytemuck::cast_slice(&pc));
                pass.dispatch_workgroups(1, 1, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Opportunistic staging copy: scan for any free slot.
        // Compute dispatch above always runs — staging readback is
        // decoupled so dispatch is never blocked by readback latency.
        let write_slot = (0..STAGING_SLOTS)
            .map(|offset| (self.staging_index + offset) % STAGING_SLOTS)
            .find(|&slot| !self.staging_in_flight[slot]);
        if let Some(slot_index) = write_slot {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dispatch_staging_copy"),
                });
            encoder.copy_buffer_to_buffer(
                &self.agent_phys_buffer,
                0,
                &self.state_staging[slot_index],
                0,
                buf_size,
            );
            if self.food_count > 0 {
                let food_size = (self.food_count * FOOD_STATE_STRIDE * 4) as u64;
                encoder.copy_buffer_to_buffer(
                    &self.food_state_buffer,
                    0,
                    &self.food_staging[slot_index],
                    0,
                    food_size,
                );
            }
            self.queue.submit(std::iter::once(encoder.finish()));

            self.staging_ready[slot_index].store(0, Ordering::Release);
            self.staging_had_error[slot_index].store(false, Ordering::Release);

            let phys_flag = self.staging_ready[slot_index].clone();
            let phys_err = self.staging_had_error[slot_index].clone();
            self.state_staging[slot_index].slice(..buf_size).map_async(
                wgpu::MapMode::Read,
                move |result| {
                    if result.is_err() {
                        phys_err.store(true, Ordering::Release);
                    }
                    phys_flag.fetch_add(1, Ordering::Release);
                },
            );
            if self.food_count > 0 {
                let food_size = (self.food_count * FOOD_STATE_STRIDE * 4) as u64;
                let food_flag = self.staging_ready[slot_index].clone();
                let food_err = self.staging_had_error[slot_index].clone();
                self.food_staging[slot_index].slice(..food_size).map_async(
                    wgpu::MapMode::Read,
                    move |result| {
                        if result.is_err() {
                            food_err.store(true, Ordering::Release);
                        }
                        food_flag.fetch_add(1, Ordering::Release);
                    },
                );
            }
            self.staging_in_flight[slot_index] = true;
            self.staging_index = (slot_index + 1) % STAGING_SLOTS;
        }

        self.active_config_index = 1 - self.active_config_index;
        true
    }

    /// Non-blocking poll + read staging. Returns true if new data was collected.
    pub fn try_collect_state(&mut self) -> bool {
        self.device.poll(wgpu::Maintain::Poll);
        self.try_collect_staging()
    }

    /// Last collected agent physics state.
    pub fn cached_state(&self) -> &[f32] {
        &self.state_cache
    }

    /// Last collected food state (pos_x, pos_y, pos_z, respawn_timer per food).
    /// Returns `None` until the first successful GPU readback.
    pub fn cached_food_state(&self) -> Option<&[f32]> {
        if self.food_cache_valid {
            Some(&self.food_cache)
        } else {
            None
        }
    }

    /// Blocking readback of full agent physics state.
    pub fn read_full_state_blocking(&mut self) -> &[f32] {
        let n = self.agent_count as usize;
        let buf_size = (n * PHYS_STRIDE * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_blocking_readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.agent_phys_buffer, 0, &staging, 0, buf_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        self.state_cache.clear();
        self.state_cache.extend_from_slice(floats);
        drop(data);
        staging.unmap();
        &self.state_cache
    }

    /// Read back full brain state for one agent (blocking GPU readback).
    pub fn read_agent_state(&self, index: u32) -> AgentBrainState {
        let i = index as usize;
        let bs = self.layout.brain_stride;
        let mut state = AgentBrainState::new_for(bs);

        let brain_offset = (i * bs * 4) as u64;
        let brain_size = (bs * 4) as u64;
        self.read_buffer_range(
            &self.brain_state_buffer,
            brain_offset,
            brain_size,
            &mut state.brain_state,
        );

        let pat_offset = (i * PATTERN_STRIDE * 4) as u64;
        let pat_size = (PATTERN_STRIDE * 4) as u64;
        self.read_buffer_range(
            &self.pattern_buffer,
            pat_offset,
            pat_size,
            &mut state.patterns,
        );

        let hist_offset = (i * HISTORY_STRIDE * 4) as u64;
        let hist_size = (HISTORY_STRIDE * 4) as u64;
        self.read_buffer_range(
            &self.history_buffer,
            hist_offset,
            hist_size,
            &mut state.history,
        );

        state
    }

    /// Write full brain state for one agent (blocking GPU upload).
    pub fn write_agent_state(&self, index: u32, state: &AgentBrainState) {
        let i = index as usize;
        let bs = self.layout.brain_stride;

        let brain_offset = (i * bs * 4) as u64;
        self.queue.write_buffer(
            &self.brain_state_buffer,
            brain_offset,
            bytemuck::cast_slice(&state.brain_state),
        );

        let pat_offset = (i * PATTERN_STRIDE * 4) as u64;
        self.queue.write_buffer(
            &self.pattern_buffer,
            pat_offset,
            bytemuck::cast_slice(&state.patterns),
        );

        let hist_offset = (i * HISTORY_STRIDE * 4) as u64;
        self.queue.write_buffer(
            &self.history_buffer,
            hist_offset,
            bytemuck::cast_slice(&state.history),
        );
    }

    /// Patch per-agent heritable config values in brain_state buffer.
    ///
    /// Writes habituation_sensitivity, max_curiosity_bonus, fatigue_floor,
    /// and movement_speed from the given BrainConfig into the agent's
    /// brain_state slots. Use this after `reset_agents()` to apply
    /// per-agent config variation.
    pub fn write_agent_heritable_config(&self, index: u32, config: &BrainConfig) {
        let i = index as usize;
        let bs = self.layout.brain_stride;

        // Heritable slots are contiguous in the fixed tail of brain_state.
        // Use dynamic base so this works with any BrainLayout, not
        // just the default FEATURE_COUNT (see init_brain_state_for).
        let tail_base = bs - FIXED_TAIL_SIZE;
        let first_delta = O_HAB_SENSITIVITY - O_PREDICTOR_CONTEXT_WEIGHT;
        debug_assert_eq!(
            O_HAB_MAX_CURIOSITY - O_PREDICTOR_CONTEXT_WEIGHT,
            first_delta + 1
        );
        debug_assert_eq!(
            O_FATIGUE_FLOOR - O_PREDICTOR_CONTEXT_WEIGHT,
            first_delta + 2
        );
        debug_assert_eq!(
            O_MOVEMENT_SPEED - O_PREDICTOR_CONTEXT_WEIGHT,
            first_delta + 3
        );

        let values = [
            config.habituation_sensitivity,
            config.max_curiosity_bonus,
            config.fatigue_floor,
            config.movement_speed,
        ];
        let byte_offset = ((i * bs + tail_base + first_delta) * 4) as u64;
        self.queue.write_buffer(
            &self.brain_state_buffer,
            byte_offset,
            bytemuck::cast_slice(&values),
        );
    }

    /// Non-blocking: kick off async readback of one agent's brain state.
    /// Results are collected via `try_collect_agent_state`.
    pub fn request_agent_state(&mut self, index: u32) -> bool {
        if index >= self.agent_count {
            log::warn!(
                "[GPU] request_agent_state: index {index} out of bounds (agent_count={})",
                self.agent_count
            );
            return false;
        }
        if self.agent_state_staging.is_some() {
            log::warn!("[GPU] request_agent_state: previous readback still in flight — skipping");
            return false;
        }
        let i = index as usize;
        let bs = self.layout.brain_stride;

        let brain_size = (bs * 4) as u64;
        let pat_size = (PATTERN_STRIDE * 4) as u64;
        let hist_size = (HISTORY_STRIDE * 4) as u64;

        let make_staging = |label, size| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let brain_staging = make_staging("agent_state_brain_staging", brain_size);
        let pattern_staging = make_staging("agent_state_pattern_staging", pat_size);
        let history_staging = make_staging("agent_state_history_staging", hist_size);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("agent_state_readback"),
            });

        let brain_offset = (i * bs * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &self.brain_state_buffer,
            brain_offset,
            &brain_staging,
            0,
            brain_size,
        );
        let pat_offset = (i * PATTERN_STRIDE * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &self.pattern_buffer,
            pat_offset,
            &pattern_staging,
            0,
            pat_size,
        );
        let hist_offset = (i * HISTORY_STRIDE * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &self.history_buffer,
            hist_offset,
            &history_staging,
            0,
            hist_size,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Set up map_async on all three staging buffers. Each callback
        // increments a shared counter; we only consider the readback ready
        // when all 3 have completed. If any fails, we set an error flag so
        // the caller can abandon the readback instead of hanging forever.
        let mapped_count = Arc::new(AtomicU8::new(0));
        let had_error = Arc::new(AtomicBool::new(false));

        let make_callback = |count: Arc<AtomicU8>, err: Arc<AtomicBool>| {
            move |result: Result<(), wgpu::BufferAsyncError>| {
                if result.is_ok() {
                    count.fetch_add(1, Ordering::Release);
                } else {
                    err.store(true, Ordering::Release);
                }
            }
        };

        brain_staging.slice(..).map_async(
            wgpu::MapMode::Read,
            make_callback(mapped_count.clone(), had_error.clone()),
        );
        pattern_staging.slice(..).map_async(
            wgpu::MapMode::Read,
            make_callback(mapped_count.clone(), had_error.clone()),
        );
        history_staging.slice(..).map_async(
            wgpu::MapMode::Read,
            make_callback(mapped_count.clone(), had_error.clone()),
        );

        self.agent_state_staging = Some(AgentStateReadback {
            brain_staging,
            pattern_staging,
            history_staging,
            brain_size,
            pattern_size: pat_size,
            history_size: hist_size,
            mapped_count,
            had_error,
            brain_stride: bs,
        });
        true
    }

    /// Non-blocking poll: returns `Some(Some(state))` when ready,
    /// `Some(None)` if a map_async error occurred (readback abandoned),
    /// or `None` if still in flight or nothing was requested.
    pub fn try_collect_agent_state(&mut self) -> Option<Option<AgentBrainState>> {
        let readback = self.agent_state_staging.as_ref()?;
        self.device.poll(wgpu::Maintain::Poll);

        // If any mapping failed, abandon the readback.
        if readback.had_error.load(Ordering::Acquire) {
            log::error!("[GPU] agent-state map_async failed — abandoning readback");
            let rb = self.agent_state_staging.take().unwrap();
            // Unmap any buffers that did succeed to avoid leaking.
            let _ =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rb.brain_staging.unmap()));
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                rb.pattern_staging.unmap()
            }));
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                rb.history_staging.unmap()
            }));
            return Some(None);
        }

        if readback.mapped_count.load(Ordering::Acquire) < 3 {
            return None; // not all 3 buffers mapped yet
        }

        let readback = self.agent_state_staging.take().unwrap();
        let mut state = AgentBrainState::new_for(readback.brain_stride);

        let brain_data = readback.brain_staging.slice(..readback.brain_size);
        let mapped = brain_data.get_mapped_range();
        state.brain_state.clear();
        state
            .brain_state
            .extend_from_slice(bytemuck::cast_slice(&mapped));
        drop(mapped);
        readback.brain_staging.unmap();

        let pat_data = readback.pattern_staging.slice(..readback.pattern_size);
        let mapped = pat_data.get_mapped_range();
        state.patterns.clear();
        state
            .patterns
            .extend_from_slice(bytemuck::cast_slice(&mapped));
        drop(mapped);
        readback.pattern_staging.unmap();

        let hist_data = readback.history_staging.slice(..readback.history_size);
        let mapped = hist_data.get_mapped_range();
        state.history.clear();
        state
            .history
            .extend_from_slice(bytemuck::cast_slice(&mapped));
        drop(mapped);
        readback.history_staging.unmap();

        Some(Some(state))
    }

    /// Batch-upload brain states for all agents in a single write per buffer.
    /// `states` is a closure that returns the `AgentBrainState` for agent index `i`.
    pub fn batch_write_agent_states<F>(&self, count: usize, states: F)
    where
        F: Fn(usize) -> AgentBrainState,
    {
        debug_assert_eq!(
            count, self.agent_count as usize,
            "batch_write_agent_states: count ({}) != agent_count ({})",
            count, self.agent_count
        );
        let bs = self.layout.brain_stride;
        let mut brain_data = Vec::with_capacity(count * bs);
        let mut pattern_data = Vec::with_capacity(count * PATTERN_STRIDE);
        let mut history_data = Vec::with_capacity(count * HISTORY_STRIDE);

        for i in 0..count {
            let s = states(i);
            debug_assert_eq!(
                s.brain_state.len(),
                bs,
                "agent {}: brain_state length {} != brain_stride {}",
                i,
                s.brain_state.len(),
                bs
            );
            debug_assert_eq!(
                s.patterns.len(),
                PATTERN_STRIDE,
                "agent {}: patterns length {} != PATTERN_STRIDE {}",
                i,
                s.patterns.len(),
                PATTERN_STRIDE
            );
            debug_assert_eq!(
                s.history.len(),
                HISTORY_STRIDE,
                "agent {}: history length {} != HISTORY_STRIDE {}",
                i,
                s.history.len(),
                HISTORY_STRIDE
            );
            brain_data.extend_from_slice(&s.brain_state);
            pattern_data.extend_from_slice(&s.patterns);
            history_data.extend_from_slice(&s.history);
        }

        debug_assert_eq!(brain_data.len(), count * bs);
        debug_assert_eq!(pattern_data.len(), count * PATTERN_STRIDE);
        debug_assert_eq!(history_data.len(), count * HISTORY_STRIDE);

        self.queue.write_buffer(
            &self.brain_state_buffer,
            0,
            bytemuck::cast_slice(&brain_data),
        );
        self.queue
            .write_buffer(&self.pattern_buffer, 0, bytemuck::cast_slice(&pattern_data));
        self.queue
            .write_buffer(&self.history_buffer, 0, bytemuck::cast_slice(&history_data));
    }

    /// Non-blocking version of `reset_agents`. Returns false if async staging
    /// buffers are still in flight (caller should retry next frame).
    pub fn try_reset_agents(&mut self, brain_config: &BrainConfig) -> bool {
        // Check if any staging buffer is still in flight.
        let expected = self.expected_staging_callbacks();
        for i in 0..STAGING_SLOTS {
            if self.staging_in_flight[i] {
                self.device.poll(wgpu::Maintain::Poll);
                if self.staging_ready[i].load(Ordering::Acquire) < expected {
                    return false; // not ready yet — caller retries next frame
                }

                if self.staging_had_error[i].load(Ordering::Acquire) {
                    self.state_staging[i].unmap();
                    if self.food_count > 0 {
                        self.food_staging[i].unmap();
                    }
                } else {
                    let buf_size = (self.agent_count as usize * PHYS_STRIDE * 4) as u64;
                    let slice = self.state_staging[i].slice(..buf_size);
                    let _data = slice.get_mapped_range();
                    drop(_data);
                    self.state_staging[i].unmap();

                    if self.food_count > 0 {
                        let food_size = (self.food_count * FOOD_STATE_STRIDE * 4) as u64;
                        let food_slice = self.food_staging[i].slice(..food_size);
                        let _food_data = food_slice.get_mapped_range();
                        drop(_food_data);
                        self.food_staging[i].unmap();
                    }
                }

                self.staging_in_flight[i] = false;
            }
            self.staging_ready[i].store(0, Ordering::Release);
        }
        self.staging_index = 0;

        let n = self.agent_count as usize;

        // Fresh brain state, pattern memory, and action history.
        let mut rng = rand::rng();
        let mut brain_data = Vec::with_capacity(n * self.layout.brain_stride);
        let mut pattern_data = Vec::with_capacity(n * PATTERN_STRIDE);
        let mut history_data = Vec::with_capacity(n * HISTORY_STRIDE);
        for _ in 0..n {
            brain_data.extend_from_slice(&init_brain_state_for(
                brain_config,
                &self.layout,
                &mut rng,
            ));
            pattern_data.extend_from_slice(&init_pattern_memory());
            history_data.extend_from_slice(&init_action_history());
        }
        self.queue.write_buffer(
            &self.brain_state_buffer,
            0,
            bytemuck::cast_slice(&brain_data),
        );
        self.queue
            .write_buffer(&self.pattern_buffer, 0, bytemuck::cast_slice(&pattern_data));
        self.queue
            .write_buffer(&self.history_buffer, 0, bytemuck::cast_slice(&history_data));
        self.queue.write_buffer(
            &self.brain_config_buffer,
            0,
            bytemuck::cast_slice(&build_config_for(brain_config, &self.layout)),
        );
        true
    }

    /// Helper: blocking read of a buffer range into a pre-sized Vec<f32>.
    fn read_buffer_range(&self, buffer: &wgpu::Buffer, offset: u64, size: u64, out: &mut Vec<f32>) {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel_read_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kernel_read_copy"),
            });
        encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        out.clear();
        out.extend_from_slice(floats);
        drop(data);
        staging.unmap();
    }

    /// Blocking readback of sensory, decision, and key brain-state data for one agent.
    /// Returns (vision_rgba, motor_fwd, motor_turn, habituation_mean, curiosity, fatigue, staleness, urgency, gradient).
    pub fn read_agent_telemetry_blocking(&self, index: u32) -> AgentTelemetry {
        let i = index as usize;
        let ss = self.layout.sensory_stride;
        let fc = self.layout.feature_count;

        // Sensory: read vision color portion (RGBA for each ray)
        let sensory_offset = (i * ss * 4) as u64;
        let sensory_size = (ss * 4) as u64;
        let mut sensory = Vec::with_capacity(ss);
        self.read_buffer_range(
            &self.sensory_buffer,
            sensory_offset,
            sensory_size,
            &mut sensory,
        );
        let vision_color: Vec<f32> = sensory[..self.layout.vision_color_count].to_vec();

        // Decision: read motor outputs (last 4 floats of DECISION_STRIDE)
        let dec_offset = (i * DECISION_STRIDE * 4) as u64;
        let dec_size = (DECISION_STRIDE * 4) as u64;
        let mut decision = Vec::with_capacity(DECISION_STRIDE);
        self.read_buffer_range(&self.decision_buffer, dec_offset, dec_size, &mut decision);
        let motor_base = DECISION_MOTOR;
        let motor_fwd = decision[motor_base];
        let motor_turn = decision[motor_base + 1];

        // Brain state: read key fields
        let bs = self.layout.brain_stride;
        let brain_offset = (i * bs * 4) as u64;
        let brain_size = (bs * 4) as u64;
        let mut brain = Vec::with_capacity(bs);
        self.read_buffer_range(
            &self.brain_state_buffer,
            brain_offset,
            brain_size,
            &mut brain,
        );

        // Compute dynamic brain-state offsets from layout's feature_count
        let dyn_pred_ctx_wt =
            fc * ENCODED_DIMENSION + ENCODED_DIMENSION + PREDICTOR_DIMENSION * ENCODED_DIMENSION;
        let dyn_hab_atten = dyn_pred_ctx_wt + (O_HAB_ATTEN - O_PREDICTOR_CONTEXT_WEIGHT);
        let dyn_fatigue_factor = dyn_pred_ctx_wt + (O_FATIGUE_FACTOR - O_PREDICTOR_CONTEXT_WEIGHT);
        let dyn_fatigue_floor = dyn_pred_ctx_wt + (O_FATIGUE_FLOOR - O_PREDICTOR_CONTEXT_WEIGHT);
        // Habituation: mean of attenuation values (ENCODED_DIMENSION floats)
        let atten_sum: f32 = brain[dyn_hab_atten..dyn_hab_atten + ENCODED_DIMENSION]
            .iter()
            .sum();
        let mean_attenuation = atten_sum / ENCODED_DIMENSION as f32;

        // Curiosity bonus: 1.0 - mean_attenuation (higher attenuation = less curious)
        let curiosity_bonus = (1.0 - mean_attenuation).max(0.0);

        // Fatigue factor and floor
        let fatigue_factor = brain[dyn_fatigue_factor];
        let fatigue_floor = brain[dyn_fatigue_floor];

        // Staleness: raw spatial stagnation [0.0, 1.0] recovered by inverting the
        // shader formula: fatigue_factor = 1.0 - staleness * (1.0 - fatigue_floor).
        // Dividing by max_penalty undoes the clamping so this spans the full [0,1]
        // range regardless of fatigue_floor.
        let max_penalty = (1.0 - fatigue_floor).max(1e-6);
        let staleness = ((1.0 - fatigue_factor) / max_penalty).clamp(0.0, 1.0);

        // Physics buffer: gradient, urgency, prediction_error, exploration_rate
        let phys_offset = (i * PHYS_STRIDE * 4) as u64;
        let phys_size = (PHYS_STRIDE * 4) as u64;
        let mut phys = Vec::with_capacity(PHYS_STRIDE);
        self.read_buffer_range(&self.agent_phys_buffer, phys_offset, phys_size, &mut phys);
        let gradient = phys[P_GRADIENT_OUT];
        let urgency = phys[P_URGENCY_OUT];
        let prediction_error = phys[P_PREDICTION_ERROR];
        let exploration_rate = phys[P_EXPLORATION_RATE_OUT];

        AgentTelemetry {
            vision_color,
            motor_fwd,
            motor_turn,
            mean_attenuation,
            curiosity_bonus,
            fatigue_factor,
            staleness,
            urgency,
            gradient,
            prediction_error,
            exploration_rate,
        }
    }

    /// Unmap all pre-allocated telemetry staging buffers.
    /// Only unmaps when a telemetry readback was pending.
    fn unmap_telemetry_staging(&self) {
        if self.pending_telemetry.is_some() {
            self.telemetry_staging.sensory.unmap();
            self.telemetry_staging.decision.unmap();
            self.telemetry_staging.brain.unmap();
            self.telemetry_staging.phys.unmap();
        }
    }

    /// Kick off non-blocking telemetry readback for one agent.
    ///
    /// Reuses pre-allocated staging buffers. If a readback for the same agent
    /// is already pending, this is a no-op. If a different agent is requested,
    /// the old pending readback is cleared first.
    pub fn request_agent_telemetry(&mut self, index: u32) {
        // Gate: skip if a readback for the same agent is already in flight.
        if let Some(pending) = &self.pending_telemetry {
            if pending.agent_index == index {
                return; // same agent already pending — wait for it
            }
            // Different agent — unmap staging buffers before reusing them.
            self.unmap_telemetry_staging();
            self.pending_telemetry = None;
        }

        let i = index as usize;
        let bs = self.layout.brain_stride;
        let ss = self.layout.sensory_stride;

        let sensory_size = (ss * 4) as u64;
        let decision_size = (DECISION_STRIDE * 4) as u64;
        let brain_size = (bs * 4) as u64;
        let phys_size = (PHYS_STRIDE * 4) as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("telemetry_readback_copy"),
            });

        let sensory_offset = (i * ss * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &self.sensory_buffer,
            sensory_offset,
            &self.telemetry_staging.sensory,
            0,
            sensory_size,
        );

        let dec_offset = (i * DECISION_STRIDE * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &self.decision_buffer,
            dec_offset,
            &self.telemetry_staging.decision,
            0,
            decision_size,
        );

        let brain_offset = (i * bs * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &self.brain_state_buffer,
            brain_offset,
            &self.telemetry_staging.brain,
            0,
            brain_size,
        );

        let phys_offset = (i * PHYS_STRIDE * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &self.agent_phys_buffer,
            phys_offset,
            &self.telemetry_staging.phys,
            0,
            phys_size,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Completion counter increments unconditionally (Ok or Err);
        // error flag records whether any mapping failed.
        let completed = Arc::new(AtomicU32::new(0));
        let had_error = Arc::new(AtomicBool::new(false));

        for staging in [
            &self.telemetry_staging.sensory,
            &self.telemetry_staging.decision,
            &self.telemetry_staging.brain,
            &self.telemetry_staging.phys,
        ] {
            let cnt = completed.clone();
            let err = had_error.clone();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_err() {
                        err.store(true, Ordering::Release);
                    }
                    cnt.fetch_add(1, Ordering::AcqRel);
                });
        }

        self.pending_telemetry = Some(TelemetryReadback {
            completed,
            had_error,
            agent_index: index,
        });
    }

    /// Non-blocking poll: if telemetry readback is complete, parse the results
    /// into `AgentTelemetry`, cache it, and return `Some`. Otherwise `None`.
    ///
    /// If any map_async callback reported an error, clears the pending state
    /// so the next frame can retry.
    pub fn try_collect_telemetry(&mut self) -> Option<AgentTelemetry> {
        self.device.poll(wgpu::Maintain::Poll);

        let pending = self.pending_telemetry.as_ref()?;
        let completed = pending.completed.load(Ordering::Acquire);
        if completed < 4 {
            return None;
        }

        // All 4 callbacks fired. Check for errors.
        if pending.had_error.load(Ordering::Acquire) {
            // At least one mapping failed — unmap staging buffers and allow retry.
            self.unmap_telemetry_staging();
            self.pending_telemetry = None;
            return None;
        }

        // All 4 mappings succeeded — consume pending state and read data.
        let _pending = self.pending_telemetry.take().unwrap();

        // Compute dynamic brain-state offsets from layout's feature_count.
        // All offsets past O_PREDICTOR_CONTEXT_WEIGHT have a fixed delta from that anchor.
        let fc = self.layout.feature_count;
        let dyn_pred_ctx_wt =
            fc * ENCODED_DIMENSION + ENCODED_DIMENSION + PREDICTOR_DIMENSION * ENCODED_DIMENSION;
        let dyn_hab_atten = dyn_pred_ctx_wt + (O_HAB_ATTEN - O_PREDICTOR_CONTEXT_WEIGHT);
        let dyn_fatigue_factor = dyn_pred_ctx_wt + (O_FATIGUE_FACTOR - O_PREDICTOR_CONTEXT_WEIGHT);
        let dyn_fatigue_floor = dyn_pred_ctx_wt + (O_FATIGUE_FLOOR - O_PREDICTOR_CONTEXT_WEIGHT);
        // Sensory
        let sensory_data = self.telemetry_staging.sensory.slice(..).get_mapped_range();
        let sensory: &[f32] = bytemuck::cast_slice(&sensory_data);
        let vision_color: Vec<f32> = sensory[..self.layout.vision_color_count].to_vec();
        drop(sensory_data);
        self.telemetry_staging.sensory.unmap();

        // Decision
        let decision_data = self.telemetry_staging.decision.slice(..).get_mapped_range();
        let decision: &[f32] = bytemuck::cast_slice(&decision_data);
        let motor_base = DECISION_MOTOR;
        let motor_fwd = decision[motor_base];
        let motor_turn = decision[motor_base + 1];
        drop(decision_data);
        self.telemetry_staging.decision.unmap();

        // Brain state
        let brain_data = self.telemetry_staging.brain.slice(..).get_mapped_range();
        let brain: &[f32] = bytemuck::cast_slice(&brain_data);

        let atten_sum: f32 = brain[dyn_hab_atten..dyn_hab_atten + ENCODED_DIMENSION]
            .iter()
            .sum();
        let mean_attenuation = atten_sum / ENCODED_DIMENSION as f32;
        let curiosity_bonus = (1.0 - mean_attenuation).max(0.0);
        let fatigue_factor = brain[dyn_fatigue_factor];
        let fatigue_floor = brain[dyn_fatigue_floor];
        // Staleness: raw spatial stagnation [0.0, 1.0] recovered by inverting the
        // shader formula: fatigue_factor = 1.0 - staleness * (1.0 - fatigue_floor).
        // Dividing by max_penalty undoes the clamping so this spans the full [0,1]
        // range regardless of fatigue_floor.
        let max_penalty = (1.0 - fatigue_floor).max(1e-6);
        let staleness = ((1.0 - fatigue_factor) / max_penalty).clamp(0.0, 1.0);

        drop(brain_data);
        self.telemetry_staging.brain.unmap();

        // Physics — gradient and urgency sourced from phys buffer (authoritative)
        let phys_data = self.telemetry_staging.phys.slice(..).get_mapped_range();
        let phys: &[f32] = bytemuck::cast_slice(&phys_data);
        let gradient = phys[P_GRADIENT_OUT];
        let urgency = phys[P_URGENCY_OUT];
        let prediction_error = phys[P_PREDICTION_ERROR];
        let exploration_rate = phys[P_EXPLORATION_RATE_OUT];
        drop(phys_data);
        self.telemetry_staging.phys.unmap();

        let tel = AgentTelemetry {
            vision_color,
            motor_fwd,
            motor_turn,
            mean_attenuation,
            curiosity_bonus,
            fatigue_factor,
            staleness,
            urgency,
            gradient,
            prediction_error,
            exploration_rate,
        };
        self.cached_telemetry = Some(tel.clone());
        Some(tel)
    }

    /// Returns the most recently collected telemetry, if any.
    pub fn cached_telemetry(&self) -> Option<&AgentTelemetry> {
        self.cached_telemetry.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BRAIN_PASSES_SRC: &str = include_str!("shaders/kernel/brain_passes.wgsl");
    const BRAIN_TICK_SRC: &str = include_str!("shaders/kernel/brain_tick.wgsl");
    const KERNEL_TICK_SRC: &str = include_str!("shaders/kernel/kernel_tick.wgsl");
    const COMMON_SRC: &str = include_str!("shaders/kernel/common.wgsl");
    const BITONIC_SUBGROUP_SRC: &str = include_str!("shaders/kernel/bitonic_sort_subgroup.wgsl");

    // ────────────────────────────────────────────────────────────────────────
    // Pipeline-overridable constants — verify the cascade is declared with
    // `override` and the host override map lists the canonical roots.
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn vision_dimensions_are_pipeline_overridable() {
        assert!(
            COMMON_SRC.contains("override VISION_W:"),
            "common.wgsl must declare VISION_W as `override` for pipeline-level overrides"
        );
        assert!(
            COMMON_SRC.contains("override VISION_H:"),
            "common.wgsl must declare VISION_H as `override` for pipeline-level overrides"
        );
        assert!(
            !COMMON_SRC.contains("const VISION_W:"),
            "VISION_W must not have a stale const declaration — only `override`"
        );
        assert!(
            !COMMON_SRC.contains("const VISION_H:"),
            "VISION_H must not have a stale const declaration — only `override`"
        );
    }

    #[test]
    fn derived_vision_constants_are_override_expressions() {
        // Any constant that transitively references VISION_W/VISION_H must be
        // `override`, otherwise WGSL validation will reject mixing const and
        // override in the same expression.
        for name in [
            "VISION_RAYS",
            "VISION_COLOR_COUNT",
            "VISION_DEPTH_COUNT",
            "SENSORY_STRIDE",
            "FEATURE_COUNT",
            "O_ENC_BIASES",
            "O_PREDICTOR_WEIGHTS",
            "O_MOVEMENT_SPEED",
            "BRAIN_STRIDE",
            "FEATURES_STRIDE",
        ] {
            let needle = format!("override {name}:");
            assert!(
                COMMON_SRC.contains(&needle),
                "{name} must be declared `override` because it depends on VISION_W/VISION_H"
            );
        }
    }

    #[test]
    fn vision_override_map_covers_root_constants() {
        let layout = BrainLayout::new(10, 7);
        let map = vision_override_constants(&layout);
        assert_eq!(map.get("VISION_W").copied(), Some(10.0));
        assert_eq!(map.get("VISION_H").copied(), Some(7.0));
        // Derived values chain through WGSL `override` expressions — they
        // must NOT be set from Rust or the chain becomes two sources of truth.
        assert!(!map.contains_key("FEATURE_COUNT"));
        assert!(!map.contains_key("BRAIN_STRIDE"));
    }

    // ────────────────────────────────────────────────────────────────────────
    // Subgroup marker stability contract — every marker must appear where
    // the composition helper expects it, and placeholder pairs must match.
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn brain_tick_has_expected_subgroup_markers() {
        assert!(
            BRAIN_TICK_SRC.contains("// SUBGROUP_ENTRY_PARAMS"),
            "brain_tick.wgsl must carry `// SUBGROUP_ENTRY_PARAMS` for the entry point"
        );
        assert!(
            BRAIN_TICK_SRC.contains("/* SUBGROUP_TOPK_ARGS */"),
            "brain_tick.wgsl must carry `/* SUBGROUP_TOPK_ARGS */` at the coop_recall_topk call"
        );
    }

    #[test]
    fn brain_passes_has_topk_param_marker_and_bitonic_fence() {
        assert_eq!(
            BRAIN_PASSES_SRC
                .matches("/* SUBGROUP_TOPK_PARAMS */")
                .count(),
            1,
            "brain_passes.wgsl must declare exactly one SUBGROUP_TOPK_PARAMS marker"
        );
        assert!(
            BRAIN_PASSES_SRC.contains("// BEGIN_BITONIC_SORT"),
            "brain_passes.wgsl must open the BEGIN_BITONIC_SORT fence"
        );
        assert!(
            BRAIN_PASSES_SRC.contains("// END_BITONIC_SORT"),
            "brain_passes.wgsl must close the END_BITONIC_SORT fence"
        );
    }

    #[test]
    fn kernel_tick_has_expected_subgroup_markers() {
        for marker in [
            "// KERNEL_SUBGROUP_ENTRY_PARAMS",
            "/* KERNEL_SUBGROUP_TOPK_PARAMS */",
            "/* KERNEL_SUBGROUP_TOPK_ARGS */",
            "/* KERNEL_SUBGROUP_TOPK_INNER_ARGS */",
        ] {
            assert!(
                KERNEL_TICK_SRC.contains(marker),
                "kernel_tick.wgsl must declare the `{marker}` marker"
            );
        }
    }

    #[test]
    fn top_k_param_and_arg_counts_match_for_composed_brain_shader() {
        // After concatenation, every SUBGROUP_TOPK_PARAMS site must have a
        // matching ARGS site at the call, otherwise substitution leaves the
        // shader with unbalanced function signatures.
        let composed = [COMMON_SRC, BRAIN_PASSES_SRC, BRAIN_TICK_SRC].join("\n");
        let params = composed.matches("/* SUBGROUP_TOPK_PARAMS */").count();
        let args = composed.matches("/* SUBGROUP_TOPK_ARGS */").count();
        assert_eq!(
            params, args,
            "SUBGROUP_TOPK_PARAMS and SUBGROUP_TOPK_ARGS must be balanced in the brain shader"
        );
    }

    #[test]
    fn top_k_param_and_arg_counts_match_for_composed_kernel_shader() {
        let composed = [COMMON_SRC, BRAIN_PASSES_SRC, KERNEL_TICK_SRC].join("\n");
        let k_params = composed
            .matches("/* KERNEL_SUBGROUP_TOPK_PARAMS */")
            .count();
        let k_args = composed
            .matches("/* KERNEL_SUBGROUP_TOPK_INNER_ARGS */")
            .count();
        assert_eq!(
            k_params, k_args,
            "KERNEL_SUBGROUP_TOPK_PARAMS and KERNEL_SUBGROUP_TOPK_INNER_ARGS must be balanced"
        );
        let delegate_args = composed.matches("/* KERNEL_SUBGROUP_TOPK_ARGS */").count();
        assert_eq!(
            delegate_args, 1,
            "KERNEL_SUBGROUP_TOPK_ARGS must appear exactly once at brain_tick_inner's delegate call"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // apply_subgroup_markers — both branches produce markerless output.
    // ────────────────────────────────────────────────────────────────────────

    fn composed_brain_shader() -> String {
        [COMMON_SRC, BRAIN_PASSES_SRC, BRAIN_TICK_SRC].join("\n")
    }

    fn composed_kernel_shader() -> String {
        [COMMON_SRC, BRAIN_PASSES_SRC, KERNEL_TICK_SRC].join("\n")
    }

    fn assert_no_markers_remain(src: &str) {
        for marker in [
            "// SUBGROUP_ENTRY_PARAMS",
            "// KERNEL_SUBGROUP_ENTRY_PARAMS",
            "/* SUBGROUP_TOPK_PARAMS */",
            "/* SUBGROUP_TOPK_ARGS */",
            "/* KERNEL_SUBGROUP_TOPK_PARAMS */",
            "/* KERNEL_SUBGROUP_TOPK_ARGS */",
            "/* KERNEL_SUBGROUP_TOPK_INNER_ARGS */",
        ] {
            assert!(
                !src.contains(marker),
                "Marker `{marker}` should be substituted, found in output:\n{src}"
            );
        }
    }

    #[test]
    fn subgroup_path_substitutes_all_brain_markers() {
        let out = apply_subgroup_markers(&composed_brain_shader(), true);
        assert_no_markers_remain(&out);
        assert!(
            out.contains("subgroup_invocation_id"),
            "subgroup path must splice in the subgroup builtin"
        );
        assert!(
            out.contains("subgroupShuffle"),
            "subgroup path must splice in the subgroup bitonic sort"
        );
        assert!(
            !out.contains("// BEGIN_BITONIC_SORT"),
            "fence markers must be removed after substitution"
        );
    }

    #[test]
    fn non_subgroup_path_strips_all_brain_markers() {
        let out = apply_subgroup_markers(&composed_brain_shader(), false);
        assert_no_markers_remain(&out);
        assert!(
            !out.contains("subgroup_invocation_id"),
            "non-subgroup path must not reference subgroup builtins"
        );
        // Fences stay — the fallback sort body lives between them.
        assert!(out.contains("// BEGIN_BITONIC_SORT"));
        assert!(out.contains("// END_BITONIC_SORT"));
    }

    #[test]
    fn subgroup_path_substitutes_all_kernel_markers() {
        let out = apply_subgroup_markers(&composed_kernel_shader(), true);
        assert_no_markers_remain(&out);
        // Kernel shader concatenates brain_passes, so the subgroup sort must
        // have been spliced there too.
        assert!(out.contains("subgroupShuffle"));
    }

    #[test]
    fn non_subgroup_path_strips_all_kernel_markers() {
        let out = apply_subgroup_markers(&composed_kernel_shader(), false);
        assert_no_markers_remain(&out);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Source-shape contract — brain_passes must NOT carry an entry point,
    // so that kernel_tick can concatenate it without duplicate functions.
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn brain_passes_has_no_entry_point() {
        assert!(
            !BRAIN_PASSES_SRC.contains("@compute"),
            "brain_passes.wgsl must contain only helper functions, not an entry point"
        );
    }

    #[test]
    fn brain_tick_entry_point_is_present() {
        assert!(
            BRAIN_TICK_SRC.contains("@compute @workgroup_size(256)"),
            "brain_tick.wgsl must declare the @compute entry point"
        );
        assert!(
            BRAIN_TICK_SRC.contains("fn brain_tick("),
            "brain_tick.wgsl must declare the brain_tick entry function"
        );
    }

    #[test]
    fn kernel_tick_entry_point_is_present() {
        assert!(
            KERNEL_TICK_SRC.contains("@compute @workgroup_size(256)"),
            "kernel_tick.wgsl must declare the @compute entry point"
        );
        assert!(
            KERNEL_TICK_SRC.contains("fn kernel_tick("),
            "kernel_tick.wgsl must declare the kernel_tick entry function"
        );
    }

    #[test]
    fn bitonic_sort_subgroup_file_has_subgroup_shuffle() {
        assert!(
            BITONIC_SUBGROUP_SRC.contains("subgroupShuffle"),
            "bitonic_sort_subgroup.wgsl must rely on subgroupShuffle — otherwise why live in a separate file"
        );
    }
}
