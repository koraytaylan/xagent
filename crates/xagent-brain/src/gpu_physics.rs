//! GPU-resident physics: all world/physics computation runs on GPU.
//!
//! Owns world buffers (terrain, biome, food, grids), agent physics state,
//! and 8 compute pipelines. Shares sensory/decision buffers with GpuBrain.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use wgpu;
use xagent_shared::WorldConfig;

use crate::buffers::*;
use crate::gpu_brain::GpuBrain;

/// Terrain heightmap vertices per side.
const TERRAIN_VPS: usize = 129;
/// Biome grid resolution (cells per side).
const BIOME_GRID_RES: usize = 256;
/// How many ticks between death-flag readback submissions.
/// Higher values = fewer CPU↔GPU round-trips but slower death detection.
const DEATH_CHECK_INTERVAL: u64 = 50;

#[allow(dead_code)] // GPU buffers are read via bind groups, not Rust field access
pub struct GpuPhysics {
    pub(crate) agent_count: u32,
    pub(crate) food_count: usize,
    pub(crate) world_config: WorldConfig,

    // ── World buffers ──
    pub(crate) heightmap_buf: wgpu::Buffer,
    pub(crate) biome_buf: wgpu::Buffer,
    pub(crate) food_state_buf: wgpu::Buffer,
    pub(crate) food_flags_buf: wgpu::Buffer,
    pub(crate) food_grid_buf: wgpu::Buffer,
    pub(crate) world_config_buf: wgpu::Buffer,

    // ── Agent physics ──
    pub(crate) agent_phys_buf: wgpu::Buffer,

    // ── Spatial grids ──
    pub(crate) agent_grid_buf: wgpu::Buffer,

    // ── Collision scratch ──
    pub(crate) collision_scratch_buf: wgpu::Buffer,

    // ── Death readback staging (double-buffered) ──
    pub(crate) death_staging: [wgpu::Buffer; 2],
    pub(crate) death_staging_idx: usize,
    pub(crate) death_staging_mapped: [bool; 2],
    pub(crate) death_submit_seq: u64,
    pub(crate) death_mapped_seq: Arc<AtomicU64>,

    // ── Async state readback (double-buffered) ──
    pub(crate) state_staging: [wgpu::Buffer; 2],
    pub(crate) state_staging_idx: usize,
    pub(crate) state_submit_seq: u64,
    pub(crate) state_mapped_seq: Arc<AtomicU64>,
    /// Cached state from last completed readback.
    pub(crate) state_cache: Vec<f32>,

    // ── Compute pipelines + bind groups ──
    pub(crate) physics_pipeline: wgpu::ComputePipeline,
    pub(crate) physics_bind_group: wgpu::BindGroup,

    pub(crate) food_grid_build_pipeline: wgpu::ComputePipeline,
    pub(crate) food_grid_build_bind_group: wgpu::BindGroup,

    pub(crate) food_detect_pipeline: wgpu::ComputePipeline,
    pub(crate) food_detect_bind_group: wgpu::BindGroup,

    pub(crate) food_respawn_pipeline: wgpu::ComputePipeline,
    pub(crate) food_respawn_bind_group: wgpu::BindGroup,

    pub(crate) agent_grid_build_pipeline: wgpu::ComputePipeline,
    pub(crate) agent_grid_build_bind_group: wgpu::BindGroup,

    pub(crate) collision_accumulate_pipeline: wgpu::ComputePipeline,
    pub(crate) collision_accumulate_bind_group: wgpu::BindGroup,

    pub(crate) collision_apply_pipeline: wgpu::ComputePipeline,
    pub(crate) collision_apply_bind_group: wgpu::BindGroup,

    pub(crate) vision_pipeline: wgpu::ComputePipeline,
    pub(crate) vision_bind_group: wgpu::BindGroup,
}

impl GpuPhysics {
    /// Create a new GpuPhysics instance, sharing device/queue/buffers from an existing GpuBrain.
    pub fn new(
        brain: &GpuBrain,
        agent_count: u32,
        food_count: usize,
        world_config: &WorldConfig,
    ) -> Self {
        let device = brain.device();
        let n = agent_count as usize;
        let f = food_count;
        let gw = grid_width(world_config.world_size);
        let grid_cells = gw * gw;

        let storage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        // ── World buffers ──
        let heightmap_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phys_heightmap"),
            size: (TERRAIN_VPS * TERRAIN_VPS * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let biome_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phys_biome"),
            size: (BIOME_GRID_RES * BIOME_GRID_RES * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let food_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phys_food_state"),
            size: ((f * FOOD_STATE_STRIDE * 4) as u64).max(4),
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let food_flags_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phys_food_flags"),
            size: ((f * 4) as u64).max(4),
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let food_grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phys_food_grid"),
            size: ((grid_cells * FOOD_GRID_CELL_STRIDE * 4) as u64).max(4),
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let world_config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phys_world_config"),
            size: (WORLD_CONFIG_SIZE * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Agent physics buffer ──
        let agent_phys_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phys_agent_phys"),
            size: (n * PHYS_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        // ── Spatial grids ──
        let agent_grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phys_agent_grid"),
            size: (grid_cells * AGENT_GRID_CELL_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        // ── Collision scratch (3 i32 per agent: push_x, push_y, push_z) ──
        let collision_scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phys_collision_scratch"),
            size: (n * 3 * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        // ── Death readback staging (double-buffered) ──
        // One f32 per agent (the died_flag): n * 4 bytes.
        let death_size = (n * 4) as u64;
        let death_staging = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("death_staging_0"),
                size: death_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("death_staging_1"),
                size: death_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        ];

        // ── Async state readback staging (double-buffered) ──
        let state_size = (n * PHYS_STRIDE * 4) as u64;
        let state_staging = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("state_staging_0"),
                size: state_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("state_staging_1"),
                size: state_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        ];

        // ── Shader constants ──
        let brain_consts = wgsl_constants();
        let phys_consts = wgsl_physics_constants(world_config.world_size, f, n);

        // ── Physics pipeline ──
        let physics_source = format!(
            "{}\n{}\n{}",
            brain_consts,
            phys_consts,
            include_str!("shaders/physics.wgsl"),
        );
        let physics_pipeline = GpuBrain::create_pipeline(device, "physics", &physics_source);
        let physics_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("physics_bg"),
            layout: &physics_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: agent_phys_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: brain.decision_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: heightmap_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: biome_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: world_config_buf.as_entire_binding() },
            ],
        });

        // ── Food grid build pipeline ──
        let food_grid_build_source = format!(
            "{}\n{}",
            phys_consts,
            include_str!("shaders/food_grid_build.wgsl"),
        );
        let food_grid_build_pipeline = GpuBrain::create_pipeline(device, "food_grid_build", &food_grid_build_source);
        let food_grid_build_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("food_grid_build_bg"),
            layout: &food_grid_build_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: food_state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: food_flags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: food_grid_buf.as_entire_binding() },
            ],
        });

        // ── Food detect pipeline ──
        let food_detect_source = format!(
            "{}\n{}",
            phys_consts,
            include_str!("shaders/food_detect.wgsl"),
        );
        let food_detect_pipeline = GpuBrain::create_pipeline(device, "food_detect", &food_detect_source);
        let food_detect_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("food_detect_bg"),
            layout: &food_detect_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: agent_phys_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: food_state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: food_flags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: food_grid_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: world_config_buf.as_entire_binding() },
            ],
        });

        // ── Food respawn pipeline (needs brain consts for pcg_hash) ──
        let food_respawn_source = format!(
            "{}\n{}\n{}",
            brain_consts,
            phys_consts,
            include_str!("shaders/food_respawn.wgsl"),
        );
        let food_respawn_pipeline = GpuBrain::create_pipeline(device, "food_respawn", &food_respawn_source);
        let food_respawn_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("food_respawn_bg"),
            layout: &food_respawn_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: food_state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: food_flags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: food_grid_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: heightmap_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: biome_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: world_config_buf.as_entire_binding() },
            ],
        });

        // ── Agent grid build pipeline ──
        let agent_grid_build_source = format!(
            "{}\n{}",
            phys_consts,
            include_str!("shaders/agent_grid_build.wgsl"),
        );
        let agent_grid_build_pipeline = GpuBrain::create_pipeline(device, "agent_grid_build", &agent_grid_build_source);
        let agent_grid_build_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("agent_grid_build_bg"),
            layout: &agent_grid_build_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: agent_phys_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: agent_grid_buf.as_entire_binding() },
            ],
        });

        // ── Collision accumulate pipeline ──
        let collision_accumulate_source = format!(
            "{}\n{}",
            phys_consts,
            include_str!("shaders/collision_accumulate.wgsl"),
        );
        let collision_accumulate_pipeline = GpuBrain::create_pipeline(device, "collision_accumulate", &collision_accumulate_source);
        let collision_accumulate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("collision_accumulate_bg"),
            layout: &collision_accumulate_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: agent_phys_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: agent_grid_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: collision_scratch_buf.as_entire_binding() },
            ],
        });

        // ── Collision apply pipeline ──
        let collision_apply_source = format!(
            "{}\n{}",
            phys_consts,
            include_str!("shaders/collision_apply.wgsl"),
        );
        let collision_apply_pipeline = GpuBrain::create_pipeline(device, "collision_apply", &collision_apply_source);
        let collision_apply_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("collision_apply_bg"),
            layout: &collision_apply_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: agent_phys_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: collision_scratch_buf.as_entire_binding() },
            ],
        });

        // ── Vision pipeline (needs brain consts for SENSORY_STRIDE) ──
        let vision_source = format!(
            "{}\n{}\n{}",
            brain_consts,
            phys_consts,
            include_str!("shaders/vision.wgsl"),
        );
        let vision_pipeline = GpuBrain::create_pipeline(device, "vision", &vision_source);
        let vision_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vision_bg"),
            layout: &vision_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: agent_phys_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: heightmap_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: biome_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: food_state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: food_flags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: food_grid_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: agent_grid_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: brain.sensory_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: world_config_buf.as_entire_binding() },
            ],
        });

        Self {
            agent_count,
            food_count,
            world_config: world_config.clone(),
            heightmap_buf,
            biome_buf,
            food_state_buf,
            food_flags_buf,
            food_grid_buf,
            world_config_buf,
            agent_phys_buf,
            agent_grid_buf,
            collision_scratch_buf,
            death_staging,
            death_staging_idx: 0,
            death_staging_mapped: [false, false],
            death_submit_seq: 0,
            death_mapped_seq: Arc::new(AtomicU64::new(0)),
            state_staging,
            state_staging_idx: 0,
            state_submit_seq: 0,
            state_mapped_seq: Arc::new(AtomicU64::new(0)),
            state_cache: vec![0.0; n * PHYS_STRIDE],
            physics_pipeline,
            physics_bind_group,
            food_grid_build_pipeline,
            food_grid_build_bind_group,
            food_detect_pipeline,
            food_detect_bind_group,
            food_respawn_pipeline,
            food_respawn_bind_group,
            agent_grid_build_pipeline,
            agent_grid_build_bind_group,
            collision_accumulate_pipeline,
            collision_accumulate_bind_group,
            collision_apply_pipeline,
            collision_apply_bind_group,
            vision_pipeline,
            vision_bind_group,
        }
    }

    /// Upload all world state to GPU. Call once per generation.
    pub fn upload_world(
        &self,
        queue: &wgpu::Queue,
        terrain_heights: &[f32],
        biome_grid: &[u32],
        food_positions: &[(f32, f32, f32)],
        food_consumed: &[bool],
        food_timers: &[f32],
    ) {
        queue.write_buffer(&self.heightmap_buf, 0, bytemuck::cast_slice(terrain_heights));
        queue.write_buffer(&self.biome_buf, 0, bytemuck::cast_slice(biome_grid));
        let mut food_data = Vec::with_capacity(food_positions.len() * FOOD_STATE_STRIDE);
        for (i, &(x, y, z)) in food_positions.iter().enumerate() {
            food_data.push(x);
            food_data.push(y);
            food_data.push(z);
            food_data.push(food_timers[i]);
        }
        queue.write_buffer(&self.food_state_buf, 0, bytemuck::cast_slice(&food_data));
        let flags: Vec<u32> = food_consumed.iter().map(|&c| if c { 1 } else { 0 }).collect();
        queue.write_buffer(&self.food_flags_buf, 0, bytemuck::cast_slice(&flags));
    }

    /// Upload initial agent physics state to GPU. Call once per generation.
    pub fn upload_agents(
        &self,
        queue: &wgpu::Queue,
        agents: &[(glam::Vec3, f32, f32, usize, usize)], // (pos, max_energy, max_integrity, mem_cap, proc_slots)
    ) {
        let mut data = vec![0.0f32; self.agent_count as usize * PHYS_STRIDE];
        for (i, &(pos, max_e, max_i, mem_cap, proc_slots)) in agents.iter().enumerate() {
            let base = i * PHYS_STRIDE;
            data[base + P_POS_X] = pos.x;
            data[base + P_POS_Y] = pos.y;
            data[base + P_POS_Z] = pos.z;
            data[base + P_FACING_Z] = 1.0; // default facing forward (+Z)
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
        queue.write_buffer(&self.agent_phys_buf, 0, bytemuck::cast_slice(&data));
    }

    /// Update the tick counter in the world config uniform. Call every tick.
    pub fn update_tick(&self, queue: &wgpu::Queue, tick: u64) {
        let tick_offset = (WC_TICK * 4) as u64;
        let tick_val = [tick as f32];
        queue.write_buffer(&self.world_config_buf, tick_offset, bytemuck::cast_slice(&tick_val));
    }

    /// Upload the full world config uniform. Call once per generation.
    pub fn upload_world_config(
        &self,
        queue: &wgpu::Queue,
        config: &WorldConfig,
        food_count: usize,
        agent_count: usize,
        tick: u64,
    ) {
        let wc = build_world_config(config, food_count, agent_count, tick, 1);
        queue.write_buffer(&self.world_config_buf, 0, bytemuck::cast_slice(&wc));
    }

    /// Encode all physics dispatches for one tick into the command encoder.
    /// Does NOT include vision or brain passes.
    pub fn encode_tick(&self, encoder: &mut wgpu::CommandEncoder) {
        // Clear grids
        encoder.clear_buffer(&self.food_grid_buf, 0, None);
        encoder.clear_buffer(&self.agent_grid_buf, 0, None);
        encoder.clear_buffer(&self.collision_scratch_buf, 0, None);

        // 1. Rebuild food grid
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.food_grid_build_pipeline);
            pass.set_bind_group(0, &self.food_grid_build_bind_group, &[]);
            pass.dispatch_workgroups(self.food_count as u32, 1, 1);
        }

        // 2. Physics
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.physics_pipeline);
            pass.set_bind_group(0, &self.physics_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }

        // 3. Food detect
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.food_detect_pipeline);
            pass.set_bind_group(0, &self.food_detect_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }

        // 4. Food respawn
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.food_respawn_pipeline);
            pass.set_bind_group(0, &self.food_respawn_bind_group, &[]);
            pass.dispatch_workgroups(self.food_count as u32, 1, 1);
        }

        // 5. Agent grid build
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.agent_grid_build_pipeline);
            pass.set_bind_group(0, &self.agent_grid_build_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }

        // 6. Collision: 3 iterations × (accumulate + apply)
        for _ in 0..3 {
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.collision_accumulate_pipeline);
                pass.set_bind_group(0, &self.collision_accumulate_bind_group, &[]);
                pass.dispatch_workgroups(self.agent_count, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.collision_apply_pipeline);
                pass.set_bind_group(0, &self.collision_apply_bind_group, &[]);
                pass.dispatch_workgroups(self.agent_count, 1, 1);
            }
        }
    }

    /// Encode vision dispatch (fills sensory_buf). Call on brain ticks only.
    pub fn encode_vision(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.vision_pipeline);
        pass.set_bind_group(0, &self.vision_bind_group, &[]);
        pass.dispatch_workgroups(self.agent_count, 1, 1);
    }

    /// Returns the death check interval.
    pub fn death_check_interval(&self) -> u64 {
        DEATH_CHECK_INTERVAL
    }

    /// Encode death flag copies into the given encoder. Call before the batch submit.
    pub fn encode_death_readback(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let n = self.agent_count as usize;
        let widx = self.death_staging_idx;

        if self.death_staging_mapped[widx] {
            self.death_staging[widx].unmap();
            self.death_staging_mapped[widx] = false;
        }

        for i in 0..n {
            let src_offset = ((i * PHYS_STRIDE + P_DIED_FLAG) * 4) as u64;
            let dst_offset = (i * 4) as u64;
            encoder.copy_buffer_to_buffer(
                &self.agent_phys_buf, src_offset,
                &self.death_staging[widx], dst_offset,
                4,
            );
        }
    }

    /// Map the death staging buffer after submit. Must be called AFTER the
    /// encoder containing encode_death_readback is submitted.
    pub fn map_death_readback(&mut self, device: &wgpu::Device) {
        let n = self.agent_count as usize;
        let widx = self.death_staging_idx;
        let death_size = (n * 4) as u64;
        let seq = self.death_submit_seq + 1;
        self.death_submit_seq = seq;
        let flag = self.death_mapped_seq.clone();
        self.death_staging[widx]
            .slice(..death_size)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    flag.store(seq, Ordering::Release);
                }
            });
        self.death_staging_mapped[widx] = true;
        self.death_staging_idx = 1 - self.death_staging_idx;
        device.poll(wgpu::Maintain::Poll);
    }

    /// Legacy: initiate async readback with its own submit. Used by callers
    /// that don't have an open encoder.
    pub fn submit_death_readback(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&Default::default());
        self.encode_death_readback(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));
        self.map_death_readback(device);
    }

    /// Try to read back death flags (non-blocking). Returns indices of dead agents.
    pub fn try_collect_deaths(&mut self, device: &wgpu::Device) -> Option<Vec<u32>> {
        device.poll(wgpu::Maintain::Poll);

        let read_idx = 1 - self.death_staging_idx;
        if self.death_mapped_seq.load(Ordering::Acquire) < self.death_submit_seq {
            return None;
        }

        let n = self.agent_count as usize;
        let death_size = (n * 4) as u64;
        let slice = self.death_staging[read_idx].slice(..death_size);
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);

        let mut dead = Vec::new();
        for i in 0..n {
            if floats[i] > 0.5 {
                dead.push(i as u32);
            }
        }
        drop(data);
        self.death_staging[read_idx].unmap();
        self.death_staging_mapped[read_idx] = false;

        Some(dead)
    }

    /// Respawn a dead agent at a new position. Resets physics state on GPU.
    pub fn respawn_agent(
        &self,
        queue: &wgpu::Queue,
        index: u32,
        pos: glam::Vec3,
        max_energy: f32,
        max_integrity: f32,
        memory_cap: usize,
        processing_slots: usize,
    ) {
        let mut data = vec![0.0f32; PHYS_STRIDE];
        data[P_POS_X] = pos.x;
        data[P_POS_Y] = pos.y;
        data[P_POS_Z] = pos.z;
        data[P_FACING_Z] = 1.0;
        data[P_ENERGY] = max_energy;
        data[P_MAX_ENERGY] = max_energy;
        data[P_INTEGRITY] = max_integrity;
        data[P_MAX_INTEGRITY] = max_integrity;
        data[P_PREV_ENERGY] = max_energy;
        data[P_PREV_INTEGRITY] = max_integrity;
        data[P_ALIVE] = 1.0;
        data[P_DIED_FLAG] = 0.0;
        data[P_MEMORY_CAP] = memory_cap as f32;
        data[P_PROCESSING_SLOTS] = processing_slots as f32;
        // food_count and ticks_alive are NOT reset (for fitness tracking)

        let offset = (index as usize * PHYS_STRIDE * 4) as u64;
        queue.write_buffer(&self.agent_phys_buf, offset, bytemuck::cast_slice(&data));
    }

    /// Initiate async readback of full agent physics state.
    /// Call once per frame after the last queue.submit(). The copy runs on GPU;
    /// collect results next frame with `try_collect_state()`.
    pub fn submit_state_readback(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let n = self.agent_count as usize;
        let buf_size = (n * PHYS_STRIDE * 4) as u64;
        let widx = self.state_staging_idx;
        encoder.copy_buffer_to_buffer(&self.agent_phys_buf, 0, &self.state_staging[widx], 0, buf_size);
        self.state_staging_idx = 1 - self.state_staging_idx;
        self.state_submit_seq += 1;
    }

    /// Finalize the async state readback: map the staging buffer after submit.
    /// Must be called AFTER the encoder containing submit_state_readback is submitted.
    pub fn map_state_readback(&mut self, device: &wgpu::Device) {
        let read_idx = 1 - self.state_staging_idx; // the one we just copied into
        let n = self.agent_count as usize;
        let buf_size = (n * PHYS_STRIDE * 4) as u64;
        let seq = self.state_submit_seq;
        let flag = self.state_mapped_seq.clone();
        self.state_staging[read_idx].slice(..buf_size).map_async(
            wgpu::MapMode::Read,
            move |result| {
                if result.is_ok() {
                    flag.store(seq, Ordering::Release);
                }
            },
        );
        device.poll(wgpu::Maintain::Poll);
    }

    /// Try to collect async state readback (non-blocking). Updates internal cache.
    /// Returns true if new data was collected.
    pub fn try_collect_state(&mut self, device: &wgpu::Device) -> bool {
        device.poll(wgpu::Maintain::Poll);
        if self.state_mapped_seq.load(Ordering::Acquire) < self.state_submit_seq {
            return false;
        }
        let read_idx = 1 - self.state_staging_idx;
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

    /// Get the cached state from the last successful async readback.
    pub fn cached_state(&self) -> &[f32] {
        &self.state_cache
    }

    /// Read back full agent physics state. Returns data for all agents as flat f32 slice.
    /// Call once per frame in windowed mode for rendering/UI.
    pub fn read_full_state_blocking(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        let n = self.agent_count as usize;
        let buf_size = (n * PHYS_STRIDE * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state_readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.agent_phys_buf, 0, &staging, 0, buf_size);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, f32>(&data).to_vec()
    }

    /// Blocking readback of agent stats for fitness evaluation. Call at generation end.
    pub fn read_agent_stats(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<(u32, u64)> {
        let n = self.agent_count as usize;
        let buf_size = (n * PHYS_STRIDE * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stats_readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.agent_phys_buf, 0, &staging, 0, buf_size);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);

        let mut stats = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * PHYS_STRIDE;
            let food = floats[base + P_FOOD_COUNT] as u32;
            let ticks = floats[base + P_TICKS_ALIVE] as u64;
            stats.push((food, ticks));
        }
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physics_shader_moves_agent_forward() {
        if !GpuBrain::is_available() { return; }
        let brain_config = xagent_shared::BrainConfig::default();
        let world_config = xagent_shared::WorldConfig::default();
        let brain = GpuBrain::new(1, &brain_config);

        // Write a forward motor command into decision_buf
        let mut decision_data = vec![0.0f32; DECISION_STRIDE];
        decision_data[DIM + DIM] = 1.0; // forward = 1.0
        brain.queue().write_buffer(brain.decision_buf(), 0, bytemuck::cast_slice(&decision_data));

        let phys = GpuPhysics::new(&brain, 1, 0, &world_config);

        // Upload a flat terrain (all zeros)
        let heights = vec![0.0f32; 129 * 129];
        let biomes = vec![0u32; 256 * 256]; // all FoodRich
        phys.upload_world(brain.queue(), &heights, &biomes, &[], &[], &[]);

        // Place agent at origin, facing +Z
        let agents = vec![(glam::Vec3::new(0.0, 1.0, 0.0), 100.0, 100.0, 128, 16)];
        phys.upload_agents(brain.queue(), &agents);
        phys.upload_world_config(brain.queue(), &world_config, 0, 1, 0);

        // Dispatch physics
        let mut encoder = brain.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&phys.physics_pipeline);
            pass.set_bind_group(0, &phys.physics_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        brain.queue().submit(std::iter::once(encoder.finish()));

        // Read back agent position
        let readback = brain.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (PHYS_STRIDE * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc2 = brain.device().create_command_encoder(&Default::default());
        enc2.copy_buffer_to_buffer(&phys.agent_phys_buf, 0, &readback, 0, (PHYS_STRIDE * 4) as u64);
        brain.queue().submit(std::iter::once(enc2.finish()));
        brain.device().poll(wgpu::Maintain::Wait);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        brain.device().poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);

        // Agent should have moved in +Z direction (facing +Z, forward=1)
        let new_z = floats[P_POS_Z];
        assert!(new_z > 0.0, "agent should move forward in Z: got {}", new_z);
        assert!(floats[P_ALIVE] > 0.5, "agent should still be alive");
    }

    #[test]
    fn collision_pushes_overlapping_agents_apart() {
        if !GpuBrain::is_available() { return; }
        let brain_config = xagent_shared::BrainConfig::default();
        let world_config = xagent_shared::WorldConfig::default();
        let brain = GpuBrain::new(2, &brain_config);
        let phys = GpuPhysics::new(&brain, 2, 0, &world_config);

        let heights = vec![0.0f32; 129 * 129];
        let biomes = vec![0u32; 256 * 256];
        phys.upload_world(brain.queue(), &heights, &biomes, &[], &[], &[]);

        // Two agents overlapping at nearly the same position
        let agents = vec![
            (glam::Vec3::new(0.0, 1.0, 0.0), 100.0, 100.0, 128, 16),
            (glam::Vec3::new(0.5, 1.0, 0.0), 100.0, 100.0, 128, 16),
        ];
        phys.upload_agents(brain.queue(), &agents);
        phys.upload_world_config(brain.queue(), &world_config, 0, 2, 0);

        let mut encoder = brain.device().create_command_encoder(&Default::default());
        encoder.clear_buffer(&phys.agent_grid_buf, 0, None);
        encoder.clear_buffer(&phys.collision_scratch_buf, 0, None);
        // Build agent grid
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&phys.agent_grid_build_pipeline);
            pass.set_bind_group(0, &phys.agent_grid_build_bind_group, &[]);
            pass.dispatch_workgroups(2, 1, 1);
        }
        // Collision accumulate + apply
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&phys.collision_accumulate_pipeline);
            pass.set_bind_group(0, &phys.collision_accumulate_bind_group, &[]);
            pass.dispatch_workgroups(2, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&phys.collision_apply_pipeline);
            pass.set_bind_group(0, &phys.collision_apply_bind_group, &[]);
            pass.dispatch_workgroups(2, 1, 1);
        }
        brain.queue().submit(std::iter::once(encoder.finish()));

        // Read back both agents
        let buf_size = (2 * PHYS_STRIDE * 4) as u64;
        let readback = brain.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc2 = brain.device().create_command_encoder(&Default::default());
        enc2.copy_buffer_to_buffer(&phys.agent_phys_buf, 0, &readback, 0, buf_size);
        brain.queue().submit(std::iter::once(enc2.finish()));
        brain.device().poll(wgpu::Maintain::Wait);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        brain.device().poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);

        let x0 = floats[P_POS_X];
        let x1 = floats[PHYS_STRIDE + P_POS_X];
        let dist = (x1 - x0).abs();
        assert!(dist > 0.5, "agents should be pushed further apart: dist={}", dist);
    }

    #[test]
    fn food_detect_awards_energy_on_claim() {
        if !GpuBrain::is_available() { return; }
        let brain_config = xagent_shared::BrainConfig::default();
        let world_config = xagent_shared::WorldConfig::default();
        let brain = GpuBrain::new(1, &brain_config);
        let phys = GpuPhysics::new(&brain, 1, 1, &world_config);

        let heights = vec![0.0f32; 129 * 129];
        let biomes = vec![0u32; 256 * 256];
        phys.upload_world(brain.queue(), &heights, &biomes,
            &[(0.0, 1.0, 0.0)], &[false], &[0.0]);
        phys.upload_agents(brain.queue(), &[(glam::Vec3::new(0.0, 1.0, 0.0), 50.0, 100.0, 128, 16)]);
        phys.upload_world_config(brain.queue(), &world_config, 1, 1, 0);

        // Build food grid, then detect
        let mut encoder = brain.device().create_command_encoder(&Default::default());
        encoder.clear_buffer(&phys.food_grid_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&phys.food_grid_build_pipeline);
            pass.set_bind_group(0, &phys.food_grid_build_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&phys.food_detect_pipeline);
            pass.set_bind_group(0, &phys.food_detect_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        brain.queue().submit(std::iter::once(encoder.finish()));

        // Read back agent energy
        let readback = brain.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (PHYS_STRIDE * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc2 = brain.device().create_command_encoder(&Default::default());
        enc2.copy_buffer_to_buffer(&phys.agent_phys_buf, 0, &readback, 0, (PHYS_STRIDE * 4) as u64);
        brain.queue().submit(std::iter::once(enc2.finish()));
        brain.device().poll(wgpu::Maintain::Wait);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        brain.device().poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);

        let energy = floats[P_ENERGY];
        assert!(energy > 60.0, "energy should have increased from food: got {}", energy);
    }
}
