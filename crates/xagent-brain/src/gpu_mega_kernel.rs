//! Mega-kernel: single-dispatch GPU simulation.
//!
//! Composes all phase WGSL fragments into one shader, creates a unified
//! buffer set and pipeline, and runs N ticks per dispatch(1,1,1).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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

    // ── Pipeline ──
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    // ── Async state readback (double-buffered) ──
    state_staging: [wgpu::Buffer; 2],
    staging_idx: usize,
    state_submit_seq: u64,
    state_mapped_seq: Arc<AtomicU64>,
    state_cache: Vec<f32>,

    // ── Config ──
    world_config: WorldConfig,
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
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("Failed to create GPU device");

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
            size: (n * SENSORY_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let brain_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega_brain_state"),
            size: (n * BRAIN_STRIDE * 4) as u64,
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
        let mut brain_data = Vec::with_capacity(n * BRAIN_STRIDE);
        let mut pattern_data = Vec::with_capacity(n * PATTERN_STRIDE);
        let mut history_data = Vec::with_capacity(n * HISTORY_STRIDE);
        for _ in 0..n {
            brain_data.extend_from_slice(&init_brain_state(brain_config, &mut rng));
            pattern_data.extend_from_slice(&init_pattern_memory());
            history_data.extend_from_slice(&init_action_history());
        }
        queue.write_buffer(&brain_state_buf, 0, bytemuck::cast_slice(&brain_data));
        queue.write_buffer(&pattern_buf, 0, bytemuck::cast_slice(&pattern_data));
        queue.write_buffer(&history_buf, 0, bytemuck::cast_slice(&history_data));
        queue.write_buffer(&brain_config_buf, 0, bytemuck::cast_slice(&build_config(brain_config)));

        // ── Compose WGSL source from fragments ──
        let source = [
            include_str!("shaders/mega/common.wgsl"),
            include_str!("shaders/mega/phase_clear.wgsl"),
            include_str!("shaders/mega/phase_food_grid.wgsl"),
            include_str!("shaders/mega/phase_physics.wgsl"),
            include_str!("shaders/mega/phase_death.wgsl"),
            include_str!("shaders/mega/phase_food_detect.wgsl"),
            include_str!("shaders/mega/phase_food_respawn.wgsl"),
            include_str!("shaders/mega/phase_agent_grid.wgsl"),
            include_str!("shaders/mega/phase_collision.wgsl"),
            include_str!("shaders/mega/phase_vision.wgsl"),
            include_str!("shaders/mega/phase_brain.wgsl"),
            include_str!("shaders/mega/mega_tick.wgsl"),
        ]
        .join("\n");

        // ── Create pipeline ──
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mega_tick"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mega_tick"),
            layout: None,
            module: &module,
            entry_point: Some("mega_tick"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Create bind group (binding order must match common.wgsl) ──
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mega_tick_bg"),
            layout: &pipeline.get_bind_group_layout(0),
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
            pipeline,
            bind_group,
            state_staging,
            staging_idx: 0,
            state_submit_seq: 0,
            state_mapped_seq: Arc::new(AtomicU64::new(0)),
            state_cache: vec![0.0; n * PHYS_STRIDE],
            world_config: world_config.clone(),
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
        let wc = build_world_config(
            &self.world_config,
            self.food_count,
            self.agent_count as usize,
            start_tick,
            ticks_to_run,
        );
        self.queue.write_buffer(&self.world_config_buf, 0, bytemuck::cast_slice(&wc));
    }

    /// Dispatch a batch of N ticks. Uploads config, dispatches (1,1,1),
    /// copies agent_phys to staging, submits, and maps staging for async read.
    pub fn dispatch_batch(&mut self, start_tick: u64, ticks_to_run: u32) {
        self.upload_world_config(start_tick, ticks_to_run);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mega_batch"),
        });

        // Dispatch the mega-kernel: single workgroup of 256 threads
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Copy agent_phys to staging for async readback
        let widx = self.staging_idx;
        let n = self.agent_count as usize;
        let buf_size = (n * PHYS_STRIDE * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.agent_phys_buf, 0, &self.state_staging[widx], 0, buf_size);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map staging for async read
        let seq = self.state_submit_seq + 1;
        self.state_submit_seq = seq;
        let flag = self.state_mapped_seq.clone();
        self.state_staging[widx]
            .slice(..buf_size)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    flag.store(seq, Ordering::Release);
                }
            });
        self.staging_idx = 1 - self.staging_idx;
    }

    /// Non-blocking poll + read staging. Returns true if new data was collected.
    pub fn try_collect_state(&mut self) -> bool {
        self.device.poll(wgpu::Maintain::Poll);

        if self.state_mapped_seq.load(Ordering::Acquire) < self.state_submit_seq {
            return false;
        }

        let read_idx = 1 - self.staging_idx;
        let n = self.agent_count as usize;
        let buf_size = (n * PHYS_STRIDE * 4) as u64;
        let slice = self.state_staging[read_idx].slice(..buf_size);
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        self.state_cache.clear();
        self.state_cache.extend_from_slice(floats);
        drop(data);
        self.state_staging[read_idx].unmap();
        true
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
        let mut state = AgentBrainState::new_blank();

        let brain_offset = (i * BRAIN_STRIDE * 4) as u64;
        let brain_size = (BRAIN_STRIDE * 4) as u64;
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

        let brain_offset = (i * BRAIN_STRIDE * 4) as u64;
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
