//! Headless benchmark runner for measuring raw simulation throughput.
//!
//! Runs the full tick loop (senses, brain, physics, collision) without
//! any UI, database, recording, or evolution overhead. Dead agents respawn
//! immediately to maintain constant population pressure.
//!
//! Automatically selects GPU or CPU path based on `ComputeBackend::probe()`.
//! GPU path offloads vision raycasting to the GPU (the biggest bottleneck,
//! ~40-50% of frame time). Brain and physics remain on CPU.

use std::time::Instant;

use glam::Vec3;
use rayon::prelude::*;
use xagent_shared::{BrainConfig, WorldConfig};

use crate::agent::{senses, Agent, AgentBody};
use crate::world::WorldState;

/// Agent count at or below which we skip rayon and run sequentially.
/// For very small populations, rayon dispatch overhead exceeds the parallel benefit.
const RAYON_THRESHOLD: usize = 4;

/// Brain decimation stride: senses + brain run every Nth tick; on other ticks,
/// agents reuse their cached motor command. Physics runs every tick.
/// N=1 means no decimation (brain fires every tick).
/// N=2 halves the brain/vision compute cost.
const BRAIN_STRIDE: u64 = 2;

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
/// Probes for GPU at startup and dispatches to the GPU or CPU path
/// accordingly. The GPU path offloads vision raycasting; everything
/// else (touch, brain, physics, collision) stays on CPU.
pub fn run_bench(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) -> BenchResult {
    let backend = crate::compute_backend::ComputeBackend::probe();

    // GPU vision is synchronous (blocking collect per tick), so CPU is
    // faster for small agent counts where CPU rayon saturates the cores.
    // Only use GPU when the per-agent vision cost outweighs the roundtrip.
    const GPU_AGENT_THRESHOLD: usize = 50;

    match backend {
        crate::compute_backend::ComputeBackend::GpuAccelerated {
            device,
            queue,
            adapter_name,
        } if agent_count >= GPU_AGENT_THRESHOLD => {
            println!("[bench] Using GPU: {} ({} agents)", adapter_name, agent_count);
            run_bench_gpu(device, queue, brain, world_config, agent_count, total_ticks)
        }
        _ => {
            println!("[bench] Using CPU ({} agents)", agent_count);
            run_bench_cpu(brain, world_config, agent_count, total_ticks)
        }
    }
}

/// CPU-only benchmark path.
///
/// Merges senses + brain + physics into a single parallel pass to minimize
/// rayon dispatch overhead. Brain decimation (BRAIN_STRIDE) skips senses +
/// brain on non-stride ticks, reusing the cached motor command.
pub fn run_bench_cpu(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) -> BenchResult {
    // Limit rayon thread count based on agent count and available P-cores.
    let p_cores = detect_perf_core_count();
    let max_threads = if p_cores > 0 { p_cores } else { rayon::current_num_threads() };
    let threads = agent_count.min(max_threads);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();

    run_bench_cpu_inner(brain, world_config, agent_count, total_ticks)
}

fn run_bench_cpu_inner(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) -> BenchResult {
    let dt = 1.0 / world_config.tick_rate;
    let mut world = WorldState::new(world_config);

    let mut agents: Vec<Agent> = (0..agent_count)
        .map(|i| {
            let pos = world.safe_spawn_position();
            Agent::new(i as u32, pos, brain.clone(), 0)
        })
        .collect();

    let mut all_positions: Vec<(Vec3, bool)> = Vec::with_capacity(agent_count);
    let mut agent_grid = crate::world::spatial::AgentGrid::new(world.config.world_size);
    let mut results: Vec<(Option<usize>, bool)> = vec![(None, false); agent_count];
    let start = Instant::now();

    for tick in 0..total_ticks {
        // Phase 1: snapshot positions (in-place)
        if all_positions.len() != agents.len() {
            all_positions.resize(agents.len(), (Vec3::ZERO, false));
        }
        for (i, agent) in agents.iter().enumerate() {
            all_positions[i] = (agent.body.body.position, agent.body.body.alive);
        }

        // Phase 2+3: senses + brain + physics (merged into one pass)
        agent_grid.rebuild(&all_positions);
        if agent_count <= RAYON_THRESHOLD {
            for (i, agent) in agents.iter_mut().enumerate() {
                results[i] = tick_one_agent(i, agent, &world, &all_positions, &agent_grid, tick, dt);
            }
        } else {
            let world_ref: &WorldState = &world;
            let pos = &all_positions;
            let grid_ref = &agent_grid;
            agents.iter_mut().enumerate()
                .zip(results.iter_mut())
                .collect::<Vec<_>>()
                .into_par_iter()
                .for_each(|((i, agent), result)| {
                    *result = tick_one_agent(i, agent, world_ref, pos, grid_ref, tick, dt);
                });
        }

        // Deferred food consumption (sequential — mutates world)
        for (i, (consumed, _)) in results.iter().enumerate() {
            if let Some(idx) = consumed {
                let food = &mut world.food_items[*idx];
                if !food.consumed {
                    let fx = food.position.x;
                    let fz = food.position.z;
                    food.consumed = true;
                    food.respawn_timer = 10.0;
                    world.food_grid.remove(*idx, fx, fz);
                    agents[i].body.body.internal.energy =
                        (agents[i].body.body.internal.energy + world.config.food_energy_value)
                            .min(agents[i].body.body.internal.max_energy);
                }
            }
        }

        // Respawn dead agents
        for (i, (_, died)) in results.iter().enumerate() {
            if *died || !agents[i].body.body.alive {
                let pos = world.safe_spawn_position();
                agents[i].body = AgentBody::new(pos);
                agents[i].body.body.internal.integrity =
                    agents[i].body.body.internal.max_integrity;
                agents[i].brain.death_signal();
                agents[i].brain.trauma(0.5);
            }
        }

        // Phase 4: collision resolution (O(n^2) pairwise)
        run_collision(&mut agents);

        world.update(dt);
    }

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let ticks_per_sec = total_ticks as f64 / elapsed_secs;

    let final_positions: Vec<[f32; 3]> = agents
        .iter()
        .map(|a| [a.body.body.position.x, a.body.body.position.y, a.body.body.position.z])
        .collect();

    BenchResult {
        total_ticks,
        agent_count,
        elapsed_secs,
        ticks_per_sec,
        final_positions,
    }
}

/// GPU-accelerated benchmark path.
///
/// Offloads vision raycasting to the GPU while keeping brain, physics,
/// and collision on CPU. The tick loop:
/// 1. Snapshot positions
/// 2a. Compute ray params on CPU (position + yaw -> origins + dirs)
/// 2b. Pack food/agent positions for GPU
/// 2c. Submit vision to GPU
/// 2d. Collect vision results
/// 2e. Build SensoryFrame from GPU vision + CPU touch/proprioception
/// 2f. brain.tick() on CPU
/// 3. Physics (same as CPU path)
/// 4. Collision (same as CPU path)
fn run_bench_gpu(
    device: wgpu::Device,
    queue: wgpu::Queue,
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) -> BenchResult {
    let dt = 1.0 / world_config.tick_rate;
    let mut world = WorldState::new(world_config);

    let mut agents: Vec<Agent> = (0..agent_count)
        .map(|i| {
            let pos = world.safe_spawn_position();
            Agent::new(i as u32, pos, brain.clone(), 0)
        })
        .collect();

    // Build terrain/biome data for GPU
    let terrain_vps = world.terrain.subdivisions + 1;
    let biome_types = world.biome_map.grid_as_u32();
    let biome_res = world.biome_map.grid_res() as u32;

    let mut gpu_vision = crate::gpu_compute::GpuVisionCompute::new(
        device,
        queue,
        agent_count as u32,
        &world.terrain.heights,
        &biome_types,
        terrain_vps,
        world.terrain.size,
        biome_res,
        world.food_items.len(),
        agent_count,
    );

    let mut all_positions: Vec<(Vec3, bool)> = Vec::with_capacity(agent_count);
    let mut agent_grid = crate::world::spatial::AgentGrid::new(world.config.world_size);
    let num_rays = 48usize;
    // Pre-allocate GPU data buffers outside the loop to avoid per-tick allocation
    let mut agent_poses: Vec<(Vec3, f32)> = Vec::with_capacity(agent_count);
    let mut food_positions: Vec<f32> = Vec::with_capacity(world.food_items.len() * 4);
    let mut agent_positions: Vec<f32> = Vec::with_capacity(agent_count * 4);
    let start = Instant::now();

    for tick in 0..total_ticks {
        // Phase 1: snapshot positions
        if all_positions.len() != agents.len() {
            all_positions.resize(agents.len(), (Vec3::ZERO, false));
        }
        for (i, agent) in agents.iter().enumerate() {
            all_positions[i] = (agent.body.body.position, agent.body.body.alive);
        }

        // Phase 2a: compute ray params on CPU
        agent_poses.clear();
        agent_poses.extend(agents.iter().map(|a| (a.body.body.position, a.body.yaw)));
        let (ray_origins, ray_dirs) = crate::gpu_compute::compute_ray_params(&agent_poses);

        // Phase 2b: pack food positions [x, y, z, consumed_as_f32]
        food_positions.clear();
        for f in &world.food_items {
            food_positions.push(f.position.x);
            food_positions.push(f.position.y);
            food_positions.push(f.position.z);
            food_positions.push(if f.consumed { 1.0 } else { 0.0 });
        }

        // Pack agent positions [x, y, z, alive_as_f32]
        agent_positions.clear();
        for (pos, alive) in &all_positions {
            agent_positions.push(pos.x);
            agent_positions.push(pos.y);
            agent_positions.push(pos.z);
            agent_positions.push(if *alive { 1.0 } else { 0.0 });
        }

        // Phase 2c: submit vision to GPU
        gpu_vision.submit(&food_positions, &agent_positions, &ray_origins, &ray_dirs);

        // Phase 2d: collect vision results (blocking)
        let vision_data = gpu_vision
            .collect_blocking()
            .expect("[bench] GPU vision collect failed");

        // Phase 2e+2f: build frames from GPU vision + CPU touch, then brain.tick
        agent_grid.rebuild(&all_positions);
        {
            let world_ref: &WorldState = &world;
            let pos = &all_positions;
            let vdata = &vision_data;
            agents.par_iter_mut().enumerate().for_each(|(i, agent)| {
                if !agent.body.body.alive {
                    return;
                }
                // Fill vision from GPU output: 5 floats per ray (RGBA + depth)
                let ray_offset = i * num_rays * 5;
                let vf = &mut agent.cached_frame.vision;
                for r in 0..num_rays {
                    let src = ray_offset + r * 5;
                    let ci = r * 4;
                    vf.color[ci] = vdata[src];
                    vf.color[ci + 1] = vdata[src + 1];
                    vf.color[ci + 2] = vdata[src + 2];
                    vf.color[ci + 3] = vdata[src + 3];
                    vf.depth[r] = vdata[src + 4];
                }
                // Fill proprioception + touch on CPU
                senses::fill_frame_non_vision(
                    &agent.body,
                    world_ref,
                    tick,
                    pos,
                    i,
                    &agent_grid,
                    &mut agent.cached_frame,
                );
                agent.cached_motor = agent.brain.tick(&agent.cached_frame);
            });
        }

        // Phase 3: physics + respawn
        run_physics_and_respawn(&mut agents, &mut world, dt);

        // Phase 4: collision resolution
        run_collision(&mut agents);

        world.update(dt);
    }

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let ticks_per_sec = total_ticks as f64 / elapsed_secs;

    let final_positions: Vec<[f32; 3]> = agents
        .iter()
        .map(|a| [a.body.body.position.x, a.body.body.position.y, a.body.body.position.z])
        .collect();

    BenchResult {
        total_ticks,
        agent_count,
        elapsed_secs,
        ticks_per_sec,
        final_positions,
    }
}

// ── Shared helpers ────────────────────────────────────────────────────

/// Process one agent: senses + brain + physics. Returns (consumed_food_idx, died).
/// Brain decimation: senses + brain only run on stride ticks; physics runs every tick.
#[inline(always)]
fn tick_one_agent(
    i: usize,
    agent: &mut Agent,
    world: &WorldState,
    positions: &[(Vec3, bool)],
    agent_grid: &crate::world::spatial::AgentGrid,
    tick: u64,
    dt: f32,
) -> (Option<usize>, bool) {
    if !agent.body.body.alive {
        return (None, false);
    }

    if (tick % BRAIN_STRIDE) == 0 {
        senses::extract_senses_with_positions(
            &agent.body, world, tick, positions, i,
            agent_grid,
            &mut agent.cached_frame,
        );
        agent.cached_motor = agent.brain.tick(&agent.cached_frame);
    }

    let motor = agent.cached_motor.clone();
    let (consumed, died) =
        crate::physics::step_pure(&mut agent.body, &motor, world, dt);

    let brain_drain = crate::physics::metabolic_drain_per_tick(
        agent.brain.config.memory_capacity,
        agent.brain.config.processing_slots,
    );
    agent.body.body.internal.energy -= brain_drain;
    if agent.body.body.internal.energy <= 0.0 {
        agent.body.body.internal.energy = 0.0;
        agent.body.body.alive = false;
    }

    (consumed, died || !agent.body.body.alive)
}

/// Run physics (parallel), deferred food consumption, and dead-agent respawn.
fn run_physics_and_respawn(agents: &mut [Agent], world: &mut WorldState, dt: f32) {
    let results: Vec<(Option<usize>, bool)> = {
        let world_ref: &WorldState = world;
        agents
            .par_iter_mut()
            .map(|agent| {
                if !agent.body.body.alive {
                    return (None, false);
                }
                let motor = agent.cached_motor.clone();
                let (consumed, died) =
                    crate::physics::step_pure(&mut agent.body, &motor, world_ref, dt);

                let brain_drain = crate::physics::metabolic_drain_per_tick(
                    agent.brain.config.memory_capacity,
                    agent.brain.config.processing_slots,
                );
                agent.body.body.internal.energy -= brain_drain;
                if agent.body.body.internal.energy <= 0.0 {
                    agent.body.body.internal.energy = 0.0;
                    agent.body.body.alive = false;
                    return (consumed, true);
                }

                (consumed, died)
            })
            .collect()
    };

    for (i, (consumed, _)) in results.iter().enumerate() {
        if let Some(idx) = consumed {
            let food = &mut world.food_items[*idx];
            if !food.consumed {
                let fx = food.position.x;
                let fz = food.position.z;
                food.consumed = true;
                food.respawn_timer = 10.0;
                world.food_grid.remove(*idx, fx, fz);
                agents[i].body.body.internal.energy =
                    (agents[i].body.body.internal.energy + world.config.food_energy_value)
                        .min(agents[i].body.body.internal.max_energy);
            }
        }
    }

    for (i, (_, died)) in results.iter().enumerate() {
        if *died || !agents[i].body.body.alive {
            let pos = world.safe_spawn_position();
            agents[i].body = AgentBody::new(pos);
            agents[i].body.body.internal.integrity =
                agents[i].body.body.internal.max_integrity;
            agents[i].brain.death_signal();
            agents[i].brain.trauma(0.5);
        }
    }
}

/// Detect the number of performance (P) cores on the current CPU.
/// Returns 0 on platforms where this is unavailable (caller uses default).
fn detect_perf_core_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl")
            .args(["-n", "hw.perflevel0.logicalcpu"])
            .output()
        {
            if output.status.success() {
                if let Ok(s) = std::str::from_utf8(&output.stdout) {
                    if let Ok(n) = s.trim().parse::<usize>() {
                        return n;
                    }
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// O(n^2) pairwise collision resolution.
fn run_collision(agents: &mut [Agent]) {
    let min_dist: f32 = 2.0;
    let min_dist_sq = min_dist * min_dist;
    let n = agents.len();
    for i in 0..n {
        if !agents[i].body.body.alive {
            continue;
        }
        for j in (i + 1)..n {
            if !agents[j].body.body.alive {
                continue;
            }
            let diff =
                agents[j].body.body.position - agents[i].body.body.position;
            let dist_sq = diff.length_squared();
            if dist_sq < min_dist_sq && dist_sq > 0.001 {
                let dist = dist_sq.sqrt();
                let overlap = min_dist - dist;
                let push = diff.normalize() * (overlap * 0.5);
                let (left, right) = agents.split_at_mut(j);
                left[i].body.body.position -= push;
                right[0].body.body.position += push;
            }
        }
    }
}
