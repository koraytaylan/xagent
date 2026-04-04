//! Headless benchmark runner for measuring raw simulation throughput.
//!
//! Runs the full tick loop (senses, brain, physics, collision) without
//! any UI, database, recording, or evolution overhead. Dead agents respawn
//! immediately to maintain constant population pressure.

use std::time::Instant;

use glam::Vec3;
use rayon::prelude::*;
use xagent_shared::{BrainConfig, WorldConfig};

use crate::agent::{senses, Agent, AgentBody};
use crate::world::WorldState;

/// Benchmark result returned by [`run_bench`].
pub struct BenchResult {
    pub total_ticks: u64,
    pub agent_count: usize,
    pub elapsed_secs: f64,
    pub ticks_per_sec: f64,
}

/// Run a headless benchmark: `total_ticks` simulation ticks with
/// `agent_count` agents. Returns timing statistics.
///
/// The loop replicates the four core phases:
/// 1. Snapshot agent positions
/// 2. Brain ticks via rayon (senses + brain.tick)
/// 3. Sequential physics (step + metabolic drain + death/respawn)
/// 4. Collision resolution (O(n^2) pairwise)
pub fn run_bench(
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
    let start = Instant::now();

    for tick in 0..total_ticks {
        // Phase 1: snapshot positions (in-place)
        if all_positions.len() != agents.len() {
            all_positions.resize(agents.len(), (Vec3::ZERO, false));
        }
        for (i, agent) in agents.iter().enumerate() {
            all_positions[i] = (agent.body.body.position, agent.body.body.alive);
        }

        // Phase 2: brain ticks (rayon parallel)
        let agent_grid = crate::world::spatial::AgentGrid::from_positions(&all_positions);
        {
            let world_ref: &WorldState = &world;
            let pos = &all_positions;
            agents.par_iter_mut().enumerate().for_each(|(i, agent)| {
                if !agent.body.body.alive {
                    return;
                }
                senses::extract_senses_with_positions(
                    &agent.body, world_ref, tick, pos, i,
                    &agent_grid,
                    &mut agent.cached_frame,
                );
                agent.cached_motor = agent.brain.tick(&agent.cached_frame);
            });
        }

        // Phase 3: parallel physics via step_pure + deferred food consumption + respawns
        //
        // step_pure takes &WorldState (immutable) so all agents can be
        // physics-stepped in parallel. Food consumption and respawns
        // require mutable world access, so they happen sequentially after.

        // 3a: parallel physics (rayon)
        let results: Vec<(Option<usize>, bool)> = {
            let world_ref: &WorldState = &world;
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

        // 3b: deferred food consumption (sequential — mutates world)
        for (consumed, _) in &results {
            if let Some(idx) = consumed {
                let food = &mut world.food_items[*idx];
                if !food.consumed {
                    let fx = food.position.x;
                    let fz = food.position.z;
                    food.consumed = true;
                    food.respawn_timer = 10.0;
                    world.food_grid.remove(*idx, fx, fz);
                }
            }
        }

        // 3c: respawn dead agents (sequential — needs mutable world for safe_spawn_position)
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
        {
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
                    let diff = agents[j].body.body.position
                        - agents[i].body.body.position;
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

        world.update(dt);
    }

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let ticks_per_sec = total_ticks as f64 / elapsed_secs;

    BenchResult {
        total_ticks,
        agent_count,
        elapsed_secs,
        ticks_per_sec,
    }
}
