//! Headless benchmark runner for measuring raw simulation throughput.
//!
//! Runs the full tick loop via GPU dispatch (GpuPhysics + GpuBrain) without
//! any UI, database, recording, or evolution overhead. Dead agents respawn
//! immediately to maintain constant population pressure.

use std::time::Instant;

use xagent_shared::{BrainConfig, WorldConfig};

use xagent_brain::GpuBrain;

use crate::world::WorldState;

/// Brain decimation stride: senses + brain run every Nth tick; on other ticks,
/// agents reuse their cached motor command. Physics runs every tick.
/// N=1 means no decimation (brain fires every tick).
/// N=2 halves the brain/vision compute cost.
const BRAIN_STRIDE: u64 = 4;

/// Benchmark result returned by [`run_bench`].
pub struct BenchResult {
    pub total_ticks: u64,
    pub agent_count: usize,
    pub elapsed_secs: f64,
    pub ticks_per_sec: f64,
    /// Final agent positions for determinism validation.
    pub final_positions: Vec<[f32; 3]>,
}

/// Run a headless benchmark: `total_ticks` simulation ticks with
/// `agent_count` agents. Returns timing statistics.
///
/// Uses GpuPhysics + GpuBrain for all computation.
pub fn run_bench(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) -> BenchResult {
    println!("[bench] Using GpuPhysics + GpuBrain ({} agents)", agent_count);
    run_bench_inner(brain, world_config, agent_count, total_ticks)
}

fn run_bench_inner(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) -> BenchResult {
    use xagent_brain::gpu_physics::GpuPhysics;

    let world = WorldState::new(world_config.clone());
    let mut gpu_brain = GpuBrain::new(agent_count as u32, &brain);

    // Gather food data for GPU upload
    let food_positions: Vec<(f32, f32, f32)> = world.food_items.iter()
        .map(|f| (f.position.x, f.position.y, f.position.z))
        .collect();
    let food_consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
    let food_timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
    let food_count = world.food_items.len();

    let mut gpu_physics = GpuPhysics::new(&gpu_brain, agent_count as u32, food_count, &world_config);

    // Upload world data
    let heights: Vec<f32> = world.terrain.heights.clone();
    let biomes: Vec<u32> = world.biome_map.grid_as_u32();
    gpu_physics.upload_world(gpu_brain.queue(), &heights, &biomes,
        &food_positions, &food_consumed, &food_timers);

    // Upload agent initial state
    let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = (0..agent_count)
        .map(|_| {
            let pos = world.safe_spawn_position();
            (pos, 100.0, 100.0, brain.memory_capacity, brain.processing_slots)
        })
        .collect();
    gpu_physics.upload_agents(gpu_brain.queue(), &agent_data);
    gpu_physics.upload_world_config(gpu_brain.queue(), &world_config, food_count, agent_count, 0);

    let start = Instant::now();
    let death_interval = gpu_physics.death_check_interval();
    let mut encoder: Option<wgpu::CommandEncoder> = None;

    for tick in 0..total_ticks {
        let brain_tick = (tick % BRAIN_STRIDE) == 0;

        let enc = encoder.get_or_insert_with(|| {
            gpu_brain.device().create_command_encoder(&Default::default())
        });

        gpu_physics.update_tick(gpu_brain.queue(), tick);
        gpu_physics.encode_tick(enc);
        if brain_tick {
            gpu_physics.encode_vision(enc);
            gpu_brain.encode_brain_passes(enc);
        }

        // At death-check boundaries: encode death copy, submit batch, handle deaths
        if (tick + 1) % death_interval == 0 {
            // Encode death flag copy into the same batch
            {
                let enc = encoder.get_or_insert_with(|| {
                    gpu_brain.device().create_command_encoder(&Default::default())
                });
                gpu_physics.encode_death_readback(enc);
            }
            if let Some(enc) = encoder.take() {
                gpu_brain.queue().submit(std::iter::once(enc.finish()));
            }
            gpu_physics.map_death_readback(gpu_brain.device());
            if tick >= death_interval {
                gpu_brain.device().poll(wgpu::Maintain::Wait);
                if let Some(dead) = gpu_physics.try_collect_deaths(gpu_brain.device()) {
                    for idx in dead {
                        let pos = world.safe_spawn_position();
                        gpu_physics.respawn_agent(
                            gpu_brain.queue(), idx, pos, 100.0, 100.0,
                            brain.memory_capacity, brain.processing_slots,
                        );
                        gpu_brain.death_signal(idx);
                    }
                }
            }
        }
    }

    // Submit any remaining ticks
    if let Some(enc) = encoder.take() {
        gpu_brain.queue().submit(std::iter::once(enc.finish()));
    }

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let ticks_per_sec = total_ticks as f64 / elapsed_secs;

    // Read back final agent positions from GPU for determinism validation
    gpu_brain.device().poll(wgpu::Maintain::Wait);
    let state = gpu_physics.read_full_state_blocking(gpu_brain.device(), gpu_brain.queue());
    let final_positions: Vec<[f32; 3]> = (0..agent_count)
        .map(|i| {
            let base = i * xagent_brain::buffers::PHYS_STRIDE;
            [
                state[base + xagent_brain::buffers::P_POS_X],
                state[base + xagent_brain::buffers::P_POS_Y],
                state[base + xagent_brain::buffers::P_POS_Z],
            ]
        })
        .collect();

    BenchResult {
        total_ticks,
        agent_count,
        elapsed_secs,
        ticks_per_sec,
        final_positions,
    }
}
