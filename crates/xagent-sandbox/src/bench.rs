//! Headless benchmark runner for measuring raw simulation throughput.
//!
//! Runs the full tick loop via a single GPU fused kernel dispatch without
//! any UI, database, recording, or evolution overhead.

use std::time::Instant;

use xagent_brain::buffers::{PHYS_STRIDE, P_POS_X, P_POS_Y, P_POS_Z};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

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
    println!("[bench] Using GpuKernel ({} agents)", agent_count);

    let (mut kernel, _world) = create_kernel(&brain, &world_config, agent_count);

    let start = Instant::now();

    // Single dispatch for all ticks
    kernel.dispatch_batch(0, total_ticks as u32);
    let state = kernel.read_full_state_blocking();

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let ticks_per_sec = total_ticks as f64 / elapsed_secs;

    let final_positions: Vec<[f32; 3]> = (0..agent_count)
        .map(|i| {
            let base = i * PHYS_STRIDE;
            [
                state[base + P_POS_X],
                state[base + P_POS_Y],
                state[base + P_POS_Z],
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

/// Profile phase costs by running with different phase masks.
/// Prints a breakdown: barriers-only, physics, physics+vision, full.
pub fn run_profile(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) {
    println!(
        "[profile] {} agents, {} ticks — phase cost breakdown:",
        agent_count, total_ticks
    );

    let (mut kernel, _world) = create_kernel(&brain, &world_config, agent_count);

    // mask=0: barriers only (no compute, same barrier structure)
    let t0 = Instant::now();
    kernel.dispatch_batch_masked(0, total_ticks as u32, 0);
    let barriers_only = t0.elapsed().as_secs_f64();

    // mask=1: physics only (barriers + physics compute)
    let (mut kernel, _) = create_kernel(&brain, &world_config, agent_count);
    let t1 = Instant::now();
    kernel.dispatch_batch_masked(0, total_ticks as u32, 1);
    let physics = t1.elapsed().as_secs_f64();

    // mask=3: physics + vision
    let (mut kernel, _) = create_kernel(&brain, &world_config, agent_count);
    let t2 = Instant::now();
    kernel.dispatch_batch_masked(0, total_ticks as u32, 3);
    let phys_vision = t2.elapsed().as_secs_f64();

    // mask=7: full (physics + vision + brain)
    let (mut kernel, _) = create_kernel(&brain, &world_config, agent_count);
    let t3 = Instant::now();
    kernel.dispatch_batch_masked(0, total_ticks as u32, 7);
    let full = t3.elapsed().as_secs_f64();

    println!("  barriers only:     {:.3}s", barriers_only);
    println!(
        "  + physics:         {:.3}s  (physics compute: {:.3}s)",
        physics,
        physics - barriers_only
    );
    println!(
        "  + vision:          {:.3}s  (vision compute:  {:.3}s)",
        phys_vision,
        phys_vision - physics
    );
    println!(
        "  + brain (full):    {:.3}s  (brain compute:   {:.3}s)",
        full,
        full - phys_vision
    );
    println!("  total tps (full):  {:.0}", total_ticks as f64 / full);
}

/// Simulate the real tick loop with accumulator and per-frame dispatch —
/// no rendering. Prints DIAG lines every second and returns the result.
pub fn run_tick_loop_bench(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
    speed_multiplier: f32,
    timeout_secs: f64,
) -> BenchResult {
    const SIM_DT: f64 = 1.0 / 60.0;

    let (mut kernel, _world) = create_kernel(&brain, &world_config, agent_count);

    let start = Instant::now();
    let mut tick: u64 = 0;
    let mut accumulator: f64 = 0.0;
    let mut gpu_tick_budget: u32 = 32;
    let mut last_diag = Instant::now();
    let mut diag_ticks_since: u64 = 0;
    let mut dispatch_count: u64 = 0;

    // Simulate 120fps frame rate (8.33ms per frame)
    let frame_delta_time: f64 = 1.0 / 120.0;

    while tick < total_ticks {
        let wall_elapsed = start.elapsed().as_secs_f64();
        if wall_elapsed > timeout_secs {
            eprintln!(
                "[BENCH] TIMEOUT after {:.1}s — only {}/{} ticks ({:.0} tps)",
                wall_elapsed,
                tick,
                total_ticks,
                tick as f64 / wall_elapsed
            );
            break;
        }

        // Accumulate
        accumulator += frame_delta_time * speed_multiplier as f64;
        let max_accumulator = SIM_DT * speed_multiplier as f64 * 3.0;
        accumulator = accumulator.min(max_accumulator);

        let remaining = (total_ticks - tick) as u32;
        let ticks_to_run = ((accumulator / SIM_DT) as u32)
            .min(gpu_tick_budget)
            .min(500)
            .min(remaining);

        if ticks_to_run > 0 {
            kernel.try_collect_state();
            kernel.dispatch_batch(tick, ticks_to_run);

            accumulator -= ticks_to_run as f64 * SIM_DT;
            gpu_tick_budget = (gpu_tick_budget + gpu_tick_budget / 4 + 1).min(64_000);
            tick += ticks_to_run as u64;
            diag_ticks_since += ticks_to_run as u64;
            dispatch_count += 1;
        }

        // DIAG every second
        if last_diag.elapsed().as_secs_f64() >= 1.0 {
            let tps = diag_ticks_since as f64 / last_diag.elapsed().as_secs_f64();
            eprintln!(
                "[BENCH-DIAG] tick={}/{} tps={:.0} budget={} dispatches={} accumulator={:.4}",
                tick, total_ticks, tps, gpu_tick_budget, dispatch_count, accumulator
            );
            diag_ticks_since = 0;
            dispatch_count = 0;
            last_diag = Instant::now();
        }
    }

    // Final readback
    let state = kernel.read_full_state_blocking();
    let elapsed_secs = start.elapsed().as_secs_f64();
    let ticks_per_sec = tick as f64 / elapsed_secs;

    eprintln!(
        "[BENCH] Done: {} ticks in {:.2}s = {:.0} tps",
        tick, elapsed_secs, ticks_per_sec
    );

    let final_positions: Vec<[f32; 3]> = (0..agent_count)
        .map(|i| {
            let base = i * PHYS_STRIDE;
            [
                state[base + P_POS_X],
                state[base + P_POS_Y],
                state[base + P_POS_Z],
            ]
        })
        .collect();

    BenchResult {
        total_ticks: tick,
        agent_count,
        elapsed_secs,
        ticks_per_sec,
        final_positions,
    }
}

fn create_kernel(
    brain: &BrainConfig,
    world_config: &WorldConfig,
    agent_count: usize,
) -> (GpuKernel, WorldState) {
    let world = WorldState::new(world_config.clone());
    let food_count = world.food_items.len();

    let kernel = GpuKernel::new(agent_count as u32, food_count, brain, world_config);

    // Upload world data
    let heights = world.terrain.heights.clone();
    let biomes = world.biome_map.grid_as_u32();
    let food_pos: Vec<(f32, f32, f32)> = world
        .food_items
        .iter()
        .map(|f| (f.position.x, f.position.y, f.position.z))
        .collect();
    let food_consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
    let food_timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
    kernel.upload_world(&heights, &biomes, &food_pos, &food_consumed, &food_timers);

    // Upload agents
    let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = (0..agent_count)
        .map(|_| {
            let pos = world.safe_spawn_position();
            (
                pos,
                100.0,
                100.0,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agent_data);

    (kernel, world)
}
