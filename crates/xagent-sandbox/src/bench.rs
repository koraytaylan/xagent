//! Headless benchmark runner for measuring raw simulation throughput.
//!
//! Runs the full tick loop (senses, brain, physics, collision) without
//! any UI, database, recording, or evolution overhead. Dead agents respawn
//! immediately to maintain constant population pressure.
//!
//! Uses GpuBrain for all brain computation.

use std::time::Instant;

use glam::Vec3;
use xagent_shared::{BrainConfig, WorldConfig};

use xagent_brain::GpuBrain;

use crate::agent::{senses, Agent, AgentBody};
use crate::world::WorldState;

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
/// Uses GpuBrain for all brain computation.
pub fn run_bench(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) -> BenchResult {
    println!("[bench] Using GpuBrain ({} agents)", agent_count);
    run_bench_inner(brain, world_config, agent_count, total_ticks)
}

/// CPU-only benchmark path (still uses GpuBrain, kept for CLI compat).
pub fn run_bench_cpu(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) -> BenchResult {
    println!("[bench] Using GpuBrain/CPU-compat ({} agents)", agent_count);
    run_bench_inner(brain, world_config, agent_count, total_ticks)
}

fn run_bench_inner(
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
            Agent::new(i as u32, pos, i as u32, brain.clone(), 0)
        })
        .collect();

    let mut gpu_brain = GpuBrain::new(agent_count as u32, &brain);

    let mut all_positions: Vec<(Vec3, bool)> = Vec::with_capacity(agent_count);
    let mut agent_grid = crate::world::spatial::AgentGrid::new(world.config.world_size);
    let start = Instant::now();

    for tick in 0..total_ticks {
        // Phase 1: snapshot positions (in-place)
        if all_positions.len() != agents.len() {
            all_positions.resize(agents.len(), (Vec3::ZERO, false));
        }
        for (i, agent) in agents.iter().enumerate() {
            all_positions[i] = (agent.body.body.position, agent.body.body.alive);
        }

        let brain_tick = (tick % BRAIN_STRIDE) == 0;

        if brain_tick {
            // Senses extraction
            agent_grid.rebuild(&all_positions);
            {
                let world_ref: &WorldState = &world;
                let pos = &all_positions;
                let grid_ref = &agent_grid;
                for (i, agent) in agents.iter_mut().enumerate() {
                    if !agent.body.body.alive { continue; }
                    senses::extract_senses_with_positions(
                        &agent.body, world_ref, tick, pos, i, grid_ref,
                        &mut agent.cached_frame,
                    );
                }
            }

            // Collect previous tick's results (non-blocking)
            if let Some(motors) = gpu_brain.try_collect() {
                for (i, motor) in motors.into_iter().enumerate() {
                    agents[i].cached_motor = motor;
                }
            }

            // Submit this tick's brain work (non-blocking)
            let frames: Vec<xagent_shared::SensoryFrame> =
                agents.iter().map(|a| a.cached_frame.clone()).collect();
            gpu_brain.submit(&frames);
        }

        // Physics
        let mut results: Vec<(Option<usize>, bool)> = Vec::with_capacity(agent_count);
        for agent in agents.iter_mut() {
            if !agent.body.body.alive {
                results.push((None, false));
                continue;
            }
            let motor = agent.cached_motor.clone();
            let (consumed, died) =
                crate::physics::step_pure(&mut agent.body, &motor, &world, dt);

            let brain_drain = crate::physics::metabolic_drain_per_tick(
                agent.brain_config.memory_capacity,
                agent.brain_config.processing_slots,
            );
            agent.body.body.internal.energy -= brain_drain;
            if agent.body.body.internal.energy <= 0.0 {
                agent.body.body.internal.energy = 0.0;
                agent.body.body.alive = false;
                results.push((consumed, true));
                continue;
            }

            results.push((consumed, died));
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
                gpu_brain.death_signal(agents[i].brain_idx);
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
