//! Headless (no-window) evolution loop and tree dump utilities.
//!
//! Extracted from main.rs to keep the windowed application logic separate
//! from the pure-simulation evolution runner.

use std::time::Instant;

use glam::Vec3;
use log::info;
use rayon::prelude::*;
use xagent_shared::{BrainConfig, FullConfig};

use xagent_brain::GpuBrain;
use xagent_brain::AgentBrainState;

use crate::agent::{mutate_brain_state, mutate_config, senses, Agent, AgentBody};
use crate::governor::{AdvanceResult, Governor};
use crate::world::WorldState;

/// Run the headless evolution loop: no window, no rendering, max speed.
/// Creates a Governor, runs generations until complete or interrupted.
pub fn run_headless(config: FullConfig, db_path: &str, resume: bool, _has_gpu: bool) {
    info!("Running headless evolution");
    let dt = 1.0 / config.world.tick_rate;
    let world_json = serde_json::to_string(&config.world).unwrap_or_default();
    let start_time = Instant::now();

    let mut governor = if resume {
        println!("Resuming from {}", db_path);
        Governor::resume(db_path).expect("Failed to resume from database")
    } else {
        Governor::new(db_path, config.governor.clone(), &config.brain, &world_json)
            .expect("Failed to initialize governor database")
    };

    let seed_config = governor.current_config().unwrap_or(config.brain.clone());
    println!(
        "Population: {} | Tick budget: {} | Elitism: {} | Patience: {}",
        governor.config.population_size,
        governor.config.tick_budget,
        governor.config.elitism_count,
        governor.config.patience,
    );

    let mut current_configs: Vec<BrainConfig> = {
        let repeats = governor.config.eval_repeats.max(1);
        let unique_count = (governor.config.population_size / repeats).max(1);
        let mut unique_configs = vec![seed_config.clone()];
        for _ in 1..unique_count {
            unique_configs.push(mutate_config(&seed_config));
        }
        let mut configs = Vec::with_capacity(governor.config.population_size);
        for uc in &unique_configs {
            for _ in 0..repeats {
                if configs.len() >= governor.config.population_size {
                    break;
                }
                configs.push(uc.clone());
            }
        }
        configs
    };

    // Brain state inherited from the previous generation's best performer.
    // Champions (first eval_repeats agents) receive these weights so learning
    // accumulates across generations instead of restarting from scratch.
    let mut inherited_state: Option<AgentBrainState> = None;
    let mut inherited_mutation_strength: f32 = 0.0;
    let repeats = governor.config.eval_repeats.max(1);

    // Create GpuBrain once — shared across all generations.
    let pop_size = governor.config.population_size;
    let mut gpu_brain = GpuBrain::new(pop_size as u32, &seed_config);

    loop {
        if governor.evolution_complete() {
            println!("\n✓ Evolution complete after {} generations", governor.generation);
            break;
        }

        // Initialize world and agents for this generation
        let mut world = WorldState::new(config.world.clone());
        let mut agents: Vec<Agent> = current_configs
            .iter()
            .enumerate()
            .map(|(i, cfg)| {
                let pos = world.safe_spawn_position();
                Agent::new(i as u32, pos, i as u32, cfg.clone(), 0)
            })
            .collect();

        // Inherit learned weights: champions get exact weights,
        // mutants get perturbed weights for neuroevolution.
        if let Some(ref state) = inherited_state {
            for (i, agent) in agents.iter_mut().enumerate() {
                if i < repeats {
                    // Champion: exact inherited weights
                    gpu_brain.write_agent_state(agent.brain_idx, state);
                } else {
                    // Mutant: inherited weights + small perturbation
                    let mutated = mutate_brain_state(state, inherited_mutation_strength);
                    gpu_brain.write_agent_state(agent.brain_idx, &mutated);
                }
            }
        }

        governor.gen_tick = 0;

        // Run generation
        let gen_start = Instant::now();
        let mut tick: u64 = 0;
        let mut positions: Vec<(Vec3, bool)> = Vec::with_capacity(agents.len());
        let mut agent_grid = crate::world::spatial::AgentGrid::new(world.config.world_size);

        while !governor.generation_complete() {
            if positions.len() != agents.len() {
                positions.resize(agents.len(), (Vec3::ZERO, false));
            }
            for (i, a) in agents.iter().enumerate() {
                positions[i] = (a.body.body.position, a.body.body.alive);
            }

            agent_grid.rebuild(&positions);
            let brain_tick = (tick % 2) == 0;

            // Senses extraction (parallel)
            if brain_tick {
                let world_ref: &WorldState = &world;
                let pos = &positions;
                let grid_ref = &agent_grid;
                agents.par_iter_mut().enumerate().for_each(|(i, agent)| {
                    if !agent.body.body.alive { return; }
                    senses::extract_senses_with_positions(
                        &agent.body, world_ref, tick, pos, i, grid_ref,
                        &mut agent.cached_frame,
                    );
                });

                // Batched GPU brain tick
                let frames: Vec<xagent_shared::SensoryFrame> =
                    agents.iter().map(|a| a.cached_frame.clone()).collect();
                let motors = gpu_brain.tick(&frames);
                for (i, motor) in motors.into_iter().enumerate() {
                    agents[i].cached_motor = motor;
                }
            }

            // Physics (sequential — small populations don't benefit from rayon here)
            let mut results: Vec<(Option<usize>, bool)> = Vec::with_capacity(agents.len());
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

                agent.total_ticks_alive += 1;
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
                        agents[i].food_consumed += 1;
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
                    agents[i].death_count += 1;
                    agents[i].life_start_tick = tick;
                }
            }

            world.update(dt);
            tick += 1;
            governor.tick();

            if governor.gen_tick % (governor.config.tick_budget / 10).max(1) == 0 {
                let pct = (governor.gen_tick as f32 / governor.config.tick_budget as f32 * 100.0) as u32;
                print!("\rGen {} [{:>3}%]", governor.generation, pct);
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
        }

        println!();
        let gen_elapsed = gen_start.elapsed();

        let fitness = governor.evaluate(&agents);

        // Capture best agent's brain state using actual composite fitness.
        // fitness is sorted descending — first entry is the best performer.
        let best_idx = fitness.first().map(|f| f.agent_index).unwrap_or(0);
        inherited_state = agents.get(best_idx).map(|a| gpu_brain.read_agent_state(a.brain_idx));
        governor.log_generation(&fitness);
        println!(
            "  Time: {:.1}s | {:.0} ticks/sec",
            gen_elapsed.as_secs_f64(),
            governor.config.tick_budget as f64 / gen_elapsed.as_secs_f64(),
        );

        governor.update_wall_time(start_time.elapsed().as_secs_f64());

        match governor.advance(&fitness) {
            AdvanceResult::Continue { configs, messages, mutation_strength } => {
                for msg in &messages {
                    println!("{}", msg);
                }
                current_configs = configs;
                inherited_mutation_strength = mutation_strength;
            }
            AdvanceResult::Finished { messages } => {
                for msg in &messages {
                    println!("{}", msg);
                }
                break;
            }
        }
    }

    let total_time = start_time.elapsed();
    println!(
        "\nTotal wall time: {:.1}s | {} generations",
        total_time.as_secs_f64(),
        governor.generation,
    );
}

/// Print evolution tree from database and exit.
pub fn dump_tree(db_path: &str) {
    let mut gov = match Governor::resume(db_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to open database '{}': {}", db_path, e);
            std::process::exit(1);
        }
    };

    let nodes = gov.tree_nodes();
    if nodes.is_empty() {
        println!("No evolution nodes found in {}", db_path);
        return;
    }

    println!("Evolution Tree ({} nodes):", nodes.len());
    println!("─────────────────────────────────────────────");
    for node in &nodes {
        let indent = "  ".repeat(node.generation as usize);
        let fitness_str = node
            .best_fitness
            .map(|f| format!("{:.4}", f))
            .unwrap_or_else(|| "—".into());
        let mutation_str = if node.mutations.is_empty() {
            String::new()
        } else {
            let parts: Vec<String> = node
                .mutations
                .iter()
                .map(|(p, d)| format!("{}{}", p, if *d > 0.0 { "↑" } else { "↓" }))
                .collect();
            format!(" ({})", parts.join(" "))
        };
        let status_marker = match node.status.as_str() {
            "failed" => " ✗",
            "exhausted" => " ⊘",
            "successful" => " ✓",
            "active" if Some(node.id) == gov.current_node_id => " ★",
            _ => "",
        };
        println!(
            "{}Gen {:>3} [{}] fitness={}{}{} ",
            indent, node.generation, node.id, fitness_str, mutation_str, status_marker,
        );
    }
}
