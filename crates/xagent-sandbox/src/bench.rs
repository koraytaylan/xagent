//! Headless benchmark runner for measuring raw simulation throughput.
//!
//! Runs the full tick loop via a single GPU mega-kernel dispatch without
//! any UI, database, recording, or evolution overhead.

use std::time::Instant;

use xagent_shared::{BrainConfig, WorldConfig};
use xagent_brain::GpuMegaKernel;
use xagent_brain::buffers::{PHYS_STRIDE, P_POS_X, P_POS_Y, P_POS_Z};

use crate::world::WorldState;

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
pub fn run_bench(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) -> BenchResult {
    println!("[bench] Using GpuMegaKernel ({} agents)", agent_count);

    let world = WorldState::new(world_config.clone());
    let food_count = world.food_items.len();

    let mut mk = GpuMegaKernel::new(
        agent_count as u32, food_count, &brain, &world_config,
    );

    // Upload world data
    let heights = world.terrain.heights.clone();
    let biomes = world.biome_map.grid_as_u32();
    let food_pos: Vec<(f32, f32, f32)> = world.food_items.iter()
        .map(|f| (f.position.x, f.position.y, f.position.z)).collect();
    let food_consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
    let food_timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
    mk.upload_world(&heights, &biomes, &food_pos, &food_consumed, &food_timers);

    // Upload agents
    let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = (0..agent_count)
        .map(|_| {
            let pos = world.safe_spawn_position();
            (pos, 100.0, 100.0, brain.memory_capacity, brain.processing_slots)
        })
        .collect();
    mk.upload_agents(&agent_data);

    let start = Instant::now();

    // Single dispatch for all ticks
    mk.dispatch_batch(0, total_ticks as u32);
    let state = mk.read_full_state_blocking();

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let ticks_per_sec = total_ticks as f64 / elapsed_secs;

    let final_positions: Vec<[f32; 3]> = (0..agent_count)
        .map(|i| {
            let base = i * PHYS_STRIDE;
            [state[base + P_POS_X], state[base + P_POS_Y], state[base + P_POS_Z]]
        })
        .collect();

    BenchResult { total_ticks, agent_count, elapsed_secs, ticks_per_sec, final_positions }
}
