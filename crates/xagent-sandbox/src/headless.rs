//! Headless (no-window) evolution loop and tree dump utilities.
//!
//! Extracted from main.rs to keep the windowed application logic separate
//! from the pure-simulation evolution runner.

use std::time::Instant;

use log::info;
use xagent_shared::{BrainConfig, FullConfig};

use xagent_brain::GpuBrain;
use xagent_brain::AgentBrainState;

use crate::agent::{mutate_brain_state, mutate_config, Agent};
use crate::governor::{AdvanceResult, Governor};
use crate::world::WorldState;

/// Run the headless evolution loop: no window, no rendering, max speed.
/// Creates a Governor, runs generations until complete or interrupted.
pub fn run_headless(config: FullConfig, db_path: &str, resume: bool, _has_gpu: bool) {
    info!("Running headless evolution");
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
        let world = WorldState::new(config.world.clone());
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

        // ── GPU physics setup for this generation ──
        let food_positions: Vec<(f32, f32, f32)> = world.food_items.iter()
            .map(|f| (f.position.x, f.position.y, f.position.z))
            .collect();
        let food_consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
        let food_timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
        let food_count = world.food_items.len();

        let mut gpu_physics = xagent_brain::gpu_physics::GpuPhysics::new(
            &gpu_brain, pop_size as u32, food_count, &config.world,
        );

        let heights: Vec<f32> = world.terrain.heights.clone();
        let biomes: Vec<u32> = world.biome_map.grid_as_u32();
        gpu_physics.upload_world(gpu_brain.queue(), &heights, &biomes,
            &food_positions, &food_consumed, &food_timers);

        let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = agents.iter()
            .map(|a| (
                a.body.body.position,
                a.body.body.internal.max_energy,
                a.body.body.internal.max_integrity,
                a.brain_config.memory_capacity,
                a.brain_config.processing_slots,
            ))
            .collect();
        gpu_physics.upload_agents(gpu_brain.queue(), &agent_data);
        gpu_physics.upload_world_config(gpu_brain.queue(), &config.world, food_count, pop_size, 0);

        let death_interval = gpu_physics.death_check_interval();
        let mut encoder: Option<wgpu::CommandEncoder> = None;

        while !governor.generation_complete() {
            let brain_tick = (tick % config.brain.brain_tick_stride as u64) == 0;

            let enc = encoder.get_or_insert_with(|| {
                gpu_brain.device().create_command_encoder(&Default::default())
            });

            gpu_physics.update_tick(gpu_brain.queue(), tick);
            gpu_physics.encode_tick(enc);
            if brain_tick {
                gpu_physics.encode_vision(enc);
                gpu_brain.encode_brain_passes(enc);
            }

            tick += 1;
            governor.tick();

            // At death-check boundaries: encode death copy, submit batch, handle deaths
            if tick % death_interval == 0 {
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
                if tick > death_interval {
                    gpu_brain.device().poll(wgpu::Maintain::Wait);
                    if let Some(dead) = gpu_physics.try_collect_deaths(gpu_brain.device()) {
                        for idx in dead {
                            let pos = world.safe_spawn_position();
                            gpu_physics.respawn_agent(
                                gpu_brain.queue(), idx, pos,
                                agents[idx as usize].body.body.internal.max_energy,
                                agents[idx as usize].body.body.internal.max_integrity,
                                agents[idx as usize].brain_config.memory_capacity,
                                agents[idx as usize].brain_config.processing_slots,
                            );
                            gpu_brain.death_signal(agents[idx as usize].brain_idx);
                            agents[idx as usize].death_count += 1;
                        }
                    }
                }
            }

            // Periodic heatmap update: read back positions every 100 ticks
            if tick % 100 == 0 {
                // Flush any pending work before blocking readback
                if let Some(enc) = encoder.take() {
                    gpu_brain.queue().submit(std::iter::once(enc.finish()));
                }
                let state = gpu_physics.read_full_state_blocking(gpu_brain.device(), gpu_brain.queue());
                for i in 0..agents.len() {
                    let base = i * xagent_brain::buffers::PHYS_STRIDE;
                    let alive = state[base + xagent_brain::buffers::P_ALIVE] > 0.5;
                    if alive {
                        agents[i].body.body.position = glam::Vec3::new(
                            state[base + xagent_brain::buffers::P_POS_X],
                            state[base + xagent_brain::buffers::P_POS_Y],
                            state[base + xagent_brain::buffers::P_POS_Z],
                        );
                        agents[i].record_heatmap(config.world.world_size);
                    }
                }
            }

            if governor.gen_tick % (governor.config.tick_budget / 10).max(1) == 0 {
                let pct = (governor.gen_tick as f32 / governor.config.tick_budget as f32 * 100.0) as u32;
                print!("\rGen {} [{:>3}%]", governor.generation, pct);
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
        }

        // Submit any remaining ticks
        if let Some(enc) = encoder.take() {
            gpu_brain.queue().submit(std::iter::once(enc.finish()));
        }

        // Read back stats from GPU for fitness evaluation
        let stats = gpu_physics.read_agent_stats(gpu_brain.device(), gpu_brain.queue());
        for (i, (food, ticks)) in stats.iter().enumerate() {
            agents[i].food_consumed = *food;
            agents[i].total_ticks_alive = *ticks;
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
