//! Mega-kernel: single-dispatch GPU simulation.
//!
//! Composes all phase WGSL fragments into one shader, creates a unified
//! buffer set and pipeline, and runs N ticks per dispatch(1,1,1).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use wgpu;
use xagent_shared::{BrainConfig, WorldConfig};

use crate::buffers::*;

/// Terrain heightmap vertices per side.
const TERRAIN_VPS: usize = 129;
/// Biome grid resolution (cells per side).
const BIOME_GRID_RES: usize = 256;

#[allow(dead_code)] // GPU buffers are read via bind groups, not Rust field access
pub struct GpuMegaKernel {
    device: wgpu::Device,
    queue: wgpu::Queue,
    agent_count: u32,
    food_count: usize,

    // ── Shared buffers (15 storage + 2 uniform) ──
    agent_phys_buf: wgpu::Buffer,
    decision_buf: wgpu::Buffer,
    heightmap_buf: wgpu::Buffer,
    biome_buf: wgpu::Buffer,
    world_config_buf: wgpu::Buffer,
    food_state_buf: wgpu::Buffer,
    food_flags_buf: wgpu::Buffer,
    food_grid_buf: wgpu::Buffer,
    agent_grid_buf: wgpu::Buffer,
    collision_scratch_buf: wgpu::Buffer,
    sensory_buf: wgpu::Buffer,
    brain_state_buf: wgpu::Buffer,
    pattern_buf: wgpu::Buffer,
    history_buf: wgpu::Buffer,
    brain_config_buf: wgpu::Buffer,

    // ── Pipelines ──
    physics_pipeline: wgpu::ComputePipeline,
    vision_pipeline: wgpu::ComputePipeline,
    brain_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    // ── Async state readback (double-buffered, non-blocking) ──
    state_staging: [wgpu::Buffer; 2],
    staging_idx: usize,                      // which buffer to write NEXT
    staging_in_flight: [bool; 2],            // submitted, not yet collected
    staging_ready: [Arc<AtomicBool>; 2],     // map_async callback fired
    state_cache: Vec<f32>,

    // ── Config ──
    world_config: WorldConfig,
    layout: BrainLayout,
    brain_tick_stride: u32,
}

impl GpuMegaKernel {
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
    pub fn reset_agents(&mut self, brain_config: &BrainConfig) {
        // Drain any pending async readback so staging buffers are clean.
        for i in 0..2 {
            if self.staging_in_flight[i] {
                while !self.staging_ready[i].load(Ordering::Acquire) {
                    self.device.poll(wgpu::Maintain::Poll);
                }
                let buf_size = (self.agent_count as usize * PHYS_STRIDE * 4) as u64;
                let slice = self.state_staging[i].slice(..buf_size);
                let _data = slice.get_mapped_range();
                drop(_data);
                self.state_staging[i].unmap();
                self.staging_in_flight[i] = false;
            }
            self.staging_ready[i].store(false, Ordering::Release);
        }
        self.staging_idx = 0;

        let n = self.agent_count as usize;

        // Fresh brain state, pattern memory, and action history.
        let mut rng = rand::rng();
        let mut brain_data = Vec::with_capacity(n * self.layout.brain_stride);
        let mut pattern_data = Vec::with_capacity(n * PATTERN_STRIDE);
        let mut history_data = Vec::with_capacity(n * HISTORY_STRIDE);
        for _ in 0..n {
            brain_data.extend_from_slice(&init_brain_state_for(brain_config, &self.layout, &mut rng));
            pattern_data.extend_from_slice(&init_pattern_memory());
            history_data.extend_from_slice(&init_action_history());
        }
        self.queue.write_buffer(&self.brain_state_buf, 0, bytemuck::cast_slice(&brain_data));
        self.queue.write_buffer(&self.pattern_buf, 0, bytemuck::cast_slice(&pattern_data));
        self.queue.write_buffer(&self.history_buf, 0, bytemuck::cast_slice(&history_data));
        self.queue.write_buffer(
            &self.brain_config_buf, 0,
            bytemuck::cast_slice(&build_config_for(brain_config, &self.layout)),
        );
    }

    /// Create the mega-kernel: device, buffers, composed shader, pipeline, bind group.
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
            log::warn!("[GpuMegaKernel] No GPU adapter found, trying fallback (CPU) adapter");
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            }))
        })
        .expect("No GPU or fallback adapter found");

        log::info!("[GpuMegaKernel] Adapter: {:?}", adapter.get_info());

        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::default();
        required_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        required_limits.max_storage_buffers_per_shader_stage =
            adapter_limits.max_storage_buffers_per_shader_stage.min(16);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gpu-mega-kernel"),
                required_features: wgpu::Features::PUSH_CONSTANTS,
                required_limits: {
                    required_limits.max_push_constant_size = 8;
                    required_limits
                },
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("Failed to create GPU device");

        let layout = BrainLayout::new(brain_config.vision_rays);
        let brain_tick_stride = brain_config.brain_tick_stride;

        let storage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        // ── Storage buffers (13 read-write + 2 read-only) ──

        let agent_phys_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_agent_phys"),
            size: (n * PHYS_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let decision_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_decision"),
            size: (n * DECISION_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let heightmap_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_heightmap"),
            size: (TERRAIN_VPS * TERRAIN_VPS * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let biome_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_biome"),
            size: (BIOME_GRID_RES * BIOME_GRID_RES * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let world_config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_world_config"),
            size: (WORLD_CONFIG_SIZE * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let food_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_food_state"),
            size: ((f * FOOD_STATE_STRIDE * 4) as u64).max(4),
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let food_flags_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_food_flags"),
            size: ((f * 4) as u64).max(4),
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let food_grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_food_grid"),
            size: ((grid_cells * FOOD_GRID_CELL_STRIDE * 4) as u64).max(4),
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let agent_grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_agent_grid"),
            size: (grid_cells * AGENT_GRID_CELL_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let collision_scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_collision_scratch"),
            size: (n * 3 * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let sensory_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_sensory"),
            size: (n * layout.sensory_stride * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let brain_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_brain_state"),
            size: (n * layout.brain_stride * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let pattern_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_pattern"),
            size: (n * PATTERN_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let history_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_history"),
            size: (n * HISTORY_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let brain_config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_brain_config"),
            size: (CONFIG_SIZE * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Async state readback staging (double-buffered) ──
        let state_size = (n * PHYS_STRIDE * 4) as u64;
        let state_staging = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mega_state_staging_0"),
                size: state_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mega_state_staging_1"),
                size: state_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        ];

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
        queue.write_buffer(&brain_state_buf, 0, bytemuck::cast_slice(&brain_data));
        queue.write_buffer(&pattern_buf, 0, bytemuck::cast_slice(&pattern_data));
        queue.write_buffer(&history_buf, 0, bytemuck::cast_slice(&history_data));
        queue.write_buffer(&brain_config_buf, 0, bytemuck::cast_slice(&build_config_for(brain_config, &layout)));

        // ── String-replace vision grid dimensions in common shader ──
        let common_src = include_str!("shaders/mega/common.wgsl")
            .replace(
                "const VISION_W: u32 = 8u;",
                &format!("const VISION_W: u32 = {}u;", layout.vision_w),
            )
            .replace(
                "const VISION_H: u32 = 6u;",
                &format!("const VISION_H: u32 = {}u;", layout.vision_h),
            );

        // ── Compose physics shader ──
        let physics_source = [
            &common_src,
            include_str!("shaders/mega/phase_clear.wgsl"),
            include_str!("shaders/mega/phase_food_grid.wgsl"),
            include_str!("shaders/mega/phase_physics.wgsl"),
            include_str!("shaders/mega/phase_death.wgsl"),
            include_str!("shaders/mega/phase_food_detect.wgsl"),
            include_str!("shaders/mega/phase_food_respawn.wgsl"),
            include_str!("shaders/mega/phase_agent_grid.wgsl"),
            include_str!("shaders/mega/phase_collision.wgsl"),
            include_str!("shaders/mega/physics_tick.wgsl"),
        ]
        .join("\n");

        // ── Compose vision shader ──
        let vision_source = [
            &common_src,
            include_str!("shaders/mega/phase_vision.wgsl"),
            include_str!("shaders/mega/vision_tick.wgsl"),
        ]
        .join("\n");

        // ── Compose brain shader ──
        let brain_source = [
            &common_src,
            include_str!("shaders/mega/brain_tick.wgsl"),
        ]
        .join("\n");

        // ── Explicit bind group layout (all 15 bindings) ──
        // Each pipeline entry point only references a subset of bindings, but we
        // need a single shared layout so one bind group works for all 3 pipelines.
        use wgpu::{BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages};
        let storage_rw_entry = |binding: u32| -> BindGroupLayoutEntry {
            BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }
        };
        let storage_ro_entry = |binding: u32| -> BindGroupLayoutEntry {
            BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }
        };
        let uniform_entry = |binding: u32| -> BindGroupLayoutEntry {
            BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mega_bgl"),
            entries: &[
                storage_rw_entry(0),   // agent_phys
                storage_rw_entry(1),   // decision
                storage_ro_entry(2),   // heightmap
                storage_ro_entry(3),   // biome
                uniform_entry(4),      // world_config
                storage_rw_entry(5),   // food_state
                storage_rw_entry(6),   // food_flags
                storage_rw_entry(7),   // food_grid
                storage_rw_entry(8),   // agent_grid
                storage_rw_entry(9),   // collision_scratch
                storage_rw_entry(10),  // sensory
                storage_rw_entry(11),  // brain_state
                storage_rw_entry(12),  // pattern
                storage_rw_entry(13),  // history
                uniform_entry(14),     // brain_config
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
            compilation_options: Default::default(),
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
            compilation_options: Default::default(),
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
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Create bind group ──
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mega_tick_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0,  resource: agent_phys_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1,  resource: decision_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2,  resource: heightmap_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3,  resource: biome_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4,  resource: world_config_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5,  resource: food_state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6,  resource: food_flags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7,  resource: food_grid_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8,  resource: agent_grid_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9,  resource: collision_scratch_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: sensory_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: brain_state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: pattern_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: history_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 14, resource: brain_config_buf.as_entire_binding() },
            ],
        });

        Self {
            device,
            queue,
            agent_count,
            food_count,
            agent_phys_buf,
            decision_buf,
            heightmap_buf,
            biome_buf,
            world_config_buf,
            food_state_buf,
            food_flags_buf,
            food_grid_buf,
            agent_grid_buf,
            collision_scratch_buf,
            sensory_buf,
            brain_state_buf,
            pattern_buf,
            history_buf,
            brain_config_buf,
            physics_pipeline,
            vision_pipeline,
            brain_pipeline,
            bind_group,
            state_staging,
            staging_idx: 0,
            staging_in_flight: [false, false],
            staging_ready: [Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false))],
            state_cache: vec![0.0; n * PHYS_STRIDE],
            world_config: world_config.clone(),
            layout,
            brain_tick_stride,
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
        self.queue.write_buffer(&self.heightmap_buf, 0, bytemuck::cast_slice(terrain_heights));
        self.queue.write_buffer(&self.biome_buf, 0, bytemuck::cast_slice(biome_grid));

        let mut food_data = Vec::with_capacity(food_positions.len() * FOOD_STATE_STRIDE);
        for (i, &(x, y, z)) in food_positions.iter().enumerate() {
            food_data.push(x);
            food_data.push(y);
            food_data.push(z);
            food_data.push(food_timers[i]);
        }
        self.queue.write_buffer(&self.food_state_buf, 0, bytemuck::cast_slice(&food_data));

        let flags: Vec<u32> = food_consumed.iter().map(|&c| if c { 1 } else { 0 }).collect();
        self.queue.write_buffer(&self.food_flags_buf, 0, bytemuck::cast_slice(&flags));
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
        self.queue.write_buffer(&self.agent_phys_buf, 0, bytemuck::cast_slice(&data));
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
        );
        wc[WC_PHASE_MASK] = phase_mask as f32;
        self.queue.write_buffer(&self.world_config_buf, 0, bytemuck::cast_slice(&wc));
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
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dispatch_masked"),
            });
            for c in cycle..chunk_end {
                let base_tick = start_tick + (c as u64) * (self.brain_tick_stride as u64);
                let pc: [u32; 2] = [base_tick as u32, self.brain_tick_stride];
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.physics_pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.set_push_constants(0, bytemuck::cast_slice(&pc));
                    pass.dispatch_workgroups(1, 1, 1);
                }
                if run_vision {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.vision_pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.dispatch_workgroups(self.agent_count, 1, 1);
                }
                if run_brain {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.brain_pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.dispatch_workgroups(self.agent_count, 1, 1);
                }
            }
            self.queue.submit(std::iter::once(encoder.finish()));
            cycle = chunk_end;
        }

        // Remainder: physics only (no vision/brain)
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dispatch_masked_final"),
        });
        if remainder > 0 {
            let rem_base = start_tick + (num_cycles as u64) * (self.brain_tick_stride as u64);
            let pc: [u32; 2] = [rem_base as u32, remainder];
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.physics_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_push_constants(0, bytemuck::cast_slice(&pc));
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        encoder.copy_buffer_to_buffer(&self.agent_phys_buf, 0, &self.state_staging[self.staging_idx], 0, buf_size);
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
    }

    /// Non-blocking: collect any ready staging buffers into state_cache.
    /// Returns true if state_cache was updated.
    fn try_collect_staging(&mut self) -> bool {
        let n = self.agent_count as usize;
        let buf_size = (n * PHYS_STRIDE * 4) as u64;
        let mut collected = false;
        for i in 0..2 {
            if self.staging_in_flight[i] && self.staging_ready[i].load(Ordering::Acquire) {
                let slice = self.state_staging[i].slice(..buf_size);
                let data = slice.get_mapped_range();
                let floats: &[f32] = bytemuck::cast_slice(&data);
                self.state_cache.clear();
                self.state_cache.extend_from_slice(floats);
                drop(data);
                self.state_staging[i].unmap();
                self.staging_in_flight[i] = false;
                collected = true;
            }
        }
        collected
    }

    /// Dispatch all ticks via alternating physics → vision → brain passes.
    /// Fully non-blocking: skips dispatch if the write-target staging buffer
    /// is still in flight (GPU backpressure).
    /// Returns `(dispatched, state_updated)`.
    pub fn dispatch_batch(&mut self, start_tick: u64, ticks_to_run: u32) -> (bool, bool) {
        // Single non-blocking poll to advance GPU callbacks.
        self.device.poll(wgpu::Maintain::Poll);

        let collected = self.try_collect_staging();

        // Check if the write-target staging buffer is free.
        let widx = self.staging_idx;
        if self.staging_in_flight[widx] {
            // GPU still busy — skip this dispatch to avoid blocking.
            return (false, collected);
        }

        let n = self.agent_count as usize;
        let buf_size = (n * PHYS_STRIDE * 4) as u64;

        // Upload world config once (mask=1: physics enabled)
        self.upload_world_config_masked(start_tick, self.brain_tick_stride, 0x1);

        let num_cycles = ticks_to_run / self.brain_tick_stride;
        let remainder = ticks_to_run % self.brain_tick_stride;

        // Chunk cycles to avoid Metal command buffer deadlock
        const CYCLES_PER_CHUNK: u32 = 100;
        let mut cycle = 0u32;
        while cycle < num_cycles {
            let chunk_end = (cycle + CYCLES_PER_CHUNK).min(num_cycles);
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dispatch_batch"),
            });
            for c in cycle..chunk_end {
                let base_tick = start_tick + (c as u64) * (self.brain_tick_stride as u64);
                let pc: [u32; 2] = [base_tick as u32, self.brain_tick_stride];

                // Physics
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.physics_pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.set_push_constants(0, bytemuck::cast_slice(&pc));
                    pass.dispatch_workgroups(1, 1, 1);
                }

                // Vision (agent_count workgroups)
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.vision_pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.dispatch_workgroups(self.agent_count, 1, 1);
                }

                // Brain (agent_count workgroups, 256 cooperative threads)
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.brain_pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.dispatch_workgroups(self.agent_count, 1, 1);
                }
            }
            self.queue.submit(std::iter::once(encoder.finish()));
            cycle = chunk_end;
        }

        // Final submit: remainder + async state readback
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dispatch_batch_final"),
        });
        if remainder > 0 {
            let rem_base = start_tick + (num_cycles as u64) * (self.brain_tick_stride as u64);
            let pc: [u32; 2] = [rem_base as u32, remainder];
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.physics_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_push_constants(0, bytemuck::cast_slice(&pc));
                pass.dispatch_workgroups(1, 1, 1);
            }
        }

        // Async state readback into staging[widx]
        encoder.copy_buffer_to_buffer(&self.agent_phys_buf, 0, &self.state_staging[widx], 0, buf_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        self.staging_ready[widx].store(false, Ordering::Release);
        let flag = self.staging_ready[widx].clone();
        self.state_staging[widx]
            .slice(..buf_size)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    flag.store(true, Ordering::Release);
                }
            });
        self.staging_in_flight[widx] = true;
        self.staging_idx = 1 - self.staging_idx;
        (true, collected)
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

    /// Blocking readback of full agent physics state.
    pub fn read_full_state_blocking(&mut self) -> &[f32] {
        let n = self.agent_count as usize;
        let buf_size = (n * PHYS_STRIDE * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_blocking_readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.agent_phys_buf, 0, &staging, 0, buf_size);
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
        self.read_buffer_range(&self.brain_state_buf, brain_offset, brain_size, &mut state.brain_state);

        let pat_offset = (i * PATTERN_STRIDE * 4) as u64;
        let pat_size = (PATTERN_STRIDE * 4) as u64;
        self.read_buffer_range(&self.pattern_buf, pat_offset, pat_size, &mut state.patterns);

        let hist_offset = (i * HISTORY_STRIDE * 4) as u64;
        let hist_size = (HISTORY_STRIDE * 4) as u64;
        self.read_buffer_range(&self.history_buf, hist_offset, hist_size, &mut state.history);

        state
    }

    /// Write full brain state for one agent (blocking GPU upload).
    pub fn write_agent_state(&self, index: u32, state: &AgentBrainState) {
        let i = index as usize;
        let bs = self.layout.brain_stride;

        let brain_offset = (i * bs * 4) as u64;
        self.queue.write_buffer(&self.brain_state_buf, brain_offset, bytemuck::cast_slice(&state.brain_state));

        let pat_offset = (i * PATTERN_STRIDE * 4) as u64;
        self.queue.write_buffer(&self.pattern_buf, pat_offset, bytemuck::cast_slice(&state.patterns));

        let hist_offset = (i * HISTORY_STRIDE * 4) as u64;
        self.queue.write_buffer(&self.history_buf, hist_offset, bytemuck::cast_slice(&state.history));
    }

    /// Helper: blocking read of a buffer range into a pre-sized Vec<f32>.
    fn read_buffer_range(&self, buffer: &wgpu::Buffer, offset: u64, size: u64, out: &mut Vec<f32>) {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_read_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mega_read_copy"),
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
}
