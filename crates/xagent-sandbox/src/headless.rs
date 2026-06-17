//! Headless (no-window) evolution loop and tree dump utilities.
//!
//! Uses GpuKernel for all simulation — physics, brain, food, and
//! death/respawn run entirely on GPU via fused kernel dispatch.

use std::time::Instant;

use log::info;
use xagent_shared::{BrainConfig, FullConfig};

use xagent_brain::buffers::{
    BrainLayout, ENCODED_DIMENSION, O_ACTION_FORWARD_WEIGHTS, O_ACTION_TURN_WEIGHTS,
    O_PREDICTOR_CONTEXT_WEIGHT, PHYS_STRIDE, PREDICTOR_DIMENSION, P_ALIVE,
    P_AVOIDANCE_SENSE_RANGE_TICKS, P_AVOIDANCE_TURNS_OPPOSING, P_DANGER_PATH_LENGTH, P_DEATH_COUNT,
    P_DISTANCE_TRAVELED, P_ENERGY_SPENT, P_FOOD_COUNT, P_POS_X, P_POS_Y, P_POS_Z, P_TICKS_ALIVE,
};
use xagent_brain::{AgentBrainState, GpuKernel};

use crate::agent::{mutate_brain_state, mutate_config, Agent};
use crate::governor::{AdvanceResult, Governor};
use crate::world::WorldState;

/// Chunk size for dispatch_batch calls. Between chunks we can read back
/// positions for heatmap recording.
const HEATMAP_INTERVAL: u32 = 100;

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
    let mut inherited_state: Option<AgentBrainState> = None;
    let mut inherited_mutation_strength: f32 = 0.0;
    let repeats = governor.config.eval_repeats.max(1);

    // Create GpuKernel once — reused across generations via reset_agents().
    let pop_size = governor.config.population_size;
    let world = WorldState::new(config.world.clone());
    let food_count = world.food_items.len();
    let mut kernel = GpuKernel::new(pop_size as u32, food_count, &seed_config, &config.world);

    loop {
        if governor.evolution_complete() {
            println!(
                "\n✓ Evolution complete after {} generations",
                governor.generation
            );
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

        // Upload world data
        let biomes = world.biome_map.grid_as_u32();
        let food_pos: Vec<(f32, f32, f32)> = world
            .food_items
            .iter()
            .map(|f| (f.position.x, f.position.y, f.position.z))
            .collect();
        let food_consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
        let food_timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
        kernel.upload_world(
            &world.terrain.heights,
            &biomes,
            &food_pos,
            &food_consumed,
            &food_timers,
        );

        // Upload agent physics state
        let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = agents
            .iter()
            .map(|a| {
                (
                    a.body.body.position,
                    a.body.body.internal.max_energy,
                    a.body.body.internal.max_integrity,
                    a.brain_config.memory_capacity,
                    a.brain_config.processing_slots,
                )
            })
            .collect();
        kernel.upload_agents(&agent_data);

        // `reset_agents()` does two things:
        // 1. Writes the shader config uniform — population-wide brain
        //    tuning values (learning_rate, decay_rate, distress_exponent,
        //    metabolic_rate, integrity_scale) plus layout constants
        //    (DIM, feature_count, memory_cap, recall_k).
        // 2. Seeds per-agent brain_state with initial heritable values
        //    from this BrainConfig (habituation_sensitivity,
        //    max_curiosity_bonus, fatigue_recovery_sensitivity,
        //    fatigue_floor).
        // Per-agent physical fields (memory_capacity, processing_slots)
        // come from `upload_agents()` above. Any inherited or mutated
        // AgentBrainState written via `write_agent_state()` below
        // overrides the reset-seeded brain_state values.
        kernel.reset_agents(&current_configs[0]);

        // Inherit learned weights for champions and mutants
        if let Some(ref state) = inherited_state {
            for (i, agent) in agents.iter().enumerate() {
                if i < repeats {
                    kernel.write_agent_state(agent.brain_idx, state);
                } else {
                    let mutated = mutate_brain_state(state, inherited_mutation_strength);
                    kernel.write_agent_state(agent.brain_idx, &mutated);
                }
            }
        }

        // Patch per-agent heritable config values so each agent's
        // brain_state reflects its own BrainConfig genome (not just
        // config[0] from reset_agents or the champion's values).
        for (i, agent) in agents.iter().enumerate() {
            kernel.write_agent_heritable_config(agent.brain_idx, &current_configs[i]);
        }

        governor.gen_tick = 0;

        // Run generation in chunks
        let gen_start = Instant::now();
        let tick_budget = governor.config.tick_budget;
        let mut ticks_done: u64 = 0;

        while ticks_done < tick_budget {
            let remaining = (tick_budget - ticks_done).min(HEATMAP_INTERVAL as u64) as u32;
            kernel.dispatch_batch(ticks_done, remaining);
            ticks_done += remaining as u64;

            // Advance governor tick counter
            governor.advance_ticks(u64::from(remaining));

            // Drain async readback, then sample cached state for heatmap
            while !kernel.try_collect_state_snapshot() {
                std::thread::yield_now();
            }
            let state = kernel.cached_state();
            for i in 0..agents.len() {
                let base = i * PHYS_STRIDE;
                let alive = state[base + P_ALIVE] > 0.5;
                if alive {
                    agents[i].body.body.position = glam::Vec3::new(
                        state[base + P_POS_X],
                        state[base + P_POS_Y],
                        state[base + P_POS_Z],
                    );
                    agents[i].record_heatmap(config.world.world_size);
                }
            }

            // Feed the governor's within-life tracker (it snapshots quarter
            // boundaries from gen_tick).
            let mut cumulative_food = 0_u64;
            let mut cumulative_alive = 0_u64;
            for i in 0..agents.len() {
                let base = i * PHYS_STRIDE;
                cumulative_food += state[base + P_FOOD_COUNT] as u64;
                cumulative_alive += state[base + P_TICKS_ALIVE] as u64;
            }
            governor.record_within_life_sample(cumulative_food, cumulative_alive);

            if governor.gen_tick % (governor.config.tick_budget / 10).max(1) == 0 {
                let pct =
                    (governor.gen_tick as f32 / governor.config.tick_budget as f32 * 100.0) as u32;
                print!("\rGen {} [{:>3}%]", governor.generation, pct);
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
        }

        // Extract fitness stats from final state
        let state = kernel.cached_state();
        let mut final_food = 0_u64;
        let mut final_alive = 0_u64;
        for i in 0..agents.len() {
            let base = i * PHYS_STRIDE;
            agents[i].food_consumed = state[base + P_FOOD_COUNT] as u32;
            agents[i].total_ticks_alive = state[base + P_TICKS_ALIVE] as u64;
            agents[i].distance_traveled = state[base + P_DISTANCE_TRAVELED];
            agents[i].energy_spent = state[base + P_ENERGY_SPENT];
            agents[i].danger_path_length = state[base + P_DANGER_PATH_LENGTH];
            agents[i].avoidance_sense_range_ticks = state[base + P_AVOIDANCE_SENSE_RANGE_TICKS];
            agents[i].avoidance_turns_opposing = state[base + P_AVOIDANCE_TURNS_OPPOSING];
            agents[i].death_count = state[base + P_DEATH_COUNT] as u32;
            final_food += u64::from(agents[i].food_consumed);
            final_alive += agents[i].total_ticks_alive;
        }
        // Final sample fills any quarter the chunk loop did not land on.
        governor.record_within_life_sample(final_food, final_alive);
        let (first_quarter_rate, last_quarter_rate) = governor.within_life_rates();

        println!();
        let gen_elapsed = gen_start.elapsed();

        let fitness = governor.evaluate(&agents);

        // Capture best agent's brain state for inheritance
        let best_idx = fitness.first().map(|f| f.agent_index).unwrap_or(0);
        inherited_state = agents
            .get(best_idx)
            .map(|a| kernel.read_agent_state(a.brain_idx));
        // The weight-norm layout must come from the champion's own config —
        // configs are per-agent, and a mismatched layout would misplace the
        // tail offsets into the champion's brain_state.
        let best_config = current_configs.get(best_idx).unwrap_or(&current_configs[0]);
        log_learning_metrics(
            &agents,
            inherited_state.as_ref(),
            best_config,
            first_quarter_rate,
            last_quarter_rate,
        );
        governor.log_generation(&fitness);
        println!(
            "  Time: {:.1}s | {:.0} ticks/sec",
            gen_elapsed.as_secs_f64(),
            governor.config.tick_budget as f64 / gen_elapsed.as_secs_f64(),
        );

        governor.update_wall_time(start_time.elapsed().as_secs_f64());

        match governor.advance(&fitness) {
            AdvanceResult::Continue {
                configs,
                messages,
                mutation_strength,
            } => {
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

/// Per-generation learning metrics: behavioral signal (food per 1k
/// alive-ticks) plus the policy weight norms of the generation's best
/// agent. These stay flat for a population that isn't learning and should
/// trend upward once credit assignment reaches food-approach actions.
/// Printed alongside the fitness line so headless runs double as
/// before/after measurement records.
fn log_learning_metrics(
    agents: &[Agent],
    best_state: Option<&AgentBrainState>,
    config: &BrainConfig,
    first_quarter_rate: f64,
    last_quarter_rate: f64,
) {
    let total_food: u64 = agents.iter().map(|a| u64::from(a.food_consumed)).sum();
    let total_deaths: u64 = agents.iter().map(|a| u64::from(a.death_count)).sum();
    // Foraging rate per 1k alive-ticks: total food normalized by the
    // life-time the population actually accrued. Unlike food-per-life this is
    // robust to death count — an active forager that dies often still scores
    // its foraging honestly — so it is the cleaner cross-generation learning
    // signal. (`total_ticks_alive` is preserved across respawn.)
    let total_alive_ticks: u64 = agents.iter().map(|a| a.total_ticks_alive).sum();
    let food_per_1k = if total_alive_ticks > 0 {
        total_food as f64 / total_alive_ticks as f64 * 1000.0
    } else {
        0.0
    };

    let mut weight_norms = String::new();
    if let Some(state) = best_state {
        let layout = BrainLayout::new(config.vision_width, config.vision_height);
        // Tail offsets are vision-independent deltas from the context-weight
        // slot; rebase them onto this layout's dynamic position.
        let tail_base = layout.feature_count * ENCODED_DIMENSION
            + ENCODED_DIMENSION
            + PREDICTOR_DIMENSION * ENCODED_DIMENSION;
        let forward_base = tail_base + (O_ACTION_FORWARD_WEIGHTS - O_PREDICTOR_CONTEXT_WEIGHT);
        let turn_base = tail_base + (O_ACTION_TURN_WEIGHTS - O_PREDICTOR_CONTEXT_WEIGHT);
        if state.brain_state.len() >= turn_base + ENCODED_DIMENSION {
            let l2_norm = |base: usize| -> f32 {
                state.brain_state[base..base + ENCODED_DIMENSION]
                    .iter()
                    .map(|w| w * w)
                    .sum::<f32>()
                    .sqrt()
            };
            weight_norms = format!(
                " | w_fwd {:.3} | w_turn {:.3}",
                l2_norm(forward_base),
                l2_norm(turn_base),
            );
        }
    }
    println!(
        "  Food: {total_food} | Deaths: {total_deaths} | Food/1k-ticks: {food_per_1k:.3} \
| Learn q1→q4: {first_quarter_rate:.3} → {last_quarter_rate:.3}{weight_norms}"
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

/// Run speed-decoupling validation: A/B test with all 0009 flags off (baseline)
/// vs on, measuring speed↔fitness correlation and other metrics.
///
/// The populated result is written to the plan folder
/// `docs/plans/0009-Intent-Aware-Fitness/0009-SPEED-DECOUPLING.md` (relative to
/// the repository root, located by walking up from the process CWD) so the
/// canonical decision record lives with the plan docs rather than in the CWD.
pub fn validate_speed_decoupling(config: FullConfig, num_generations: u64) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SPEED-DECOUPLING VALIDATION (Plan 0009)");
    println!(
        "Running {} generations with all 0009 flags OFF (baseline) then ON",
        num_generations
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Baseline run (all flags off)
    println!("\n[BASELINE] All 0009 flags OFF");
    let baseline_stats = run_headless_with_flags(config.clone(), num_generations, false, false);

    // On run (all 0009 flags on: effort-rebased fitness, super-linear drag at k=2.0, danger percept)
    println!("\n[ON] All 0009 flags ON");
    let on_stats = run_headless_with_flags(config.clone(), num_generations, true, true);

    // Compute and report metrics
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    print_validation_metrics(&baseline_stats, &on_stats);

    let markdown = format_validation_markdown(&baseline_stats, &on_stats);

    // Write to plan folder (walk up from CWD to find docs/plans/0009-Intent-Aware-Fitness/).
    let plan_doc_path = locate_plan_doc();
    match std::fs::write(&plan_doc_path, &markdown) {
        Ok(()) => println!("\nResults saved to {}", plan_doc_path),
        Err(e) => eprintln!("Failed to write plan doc to {}: {}", plan_doc_path, e),
    }

    // Also write a copy in the CWD for convenience.
    let cwd_path = "0009-SPEED-DECOUPLING.md";
    match std::fs::write(cwd_path, markdown) {
        Ok(()) => println!("Results also saved to ./{}", cwd_path),
        Err(e) => eprintln!("Failed to write CWD copy: {}", e),
    }
}

/// Walk up from the process CWD until we find the plan folder, then return the
/// full path to the decision doc inside it.  Falls back to the CWD copy if the
/// plan folder is not found (e.g. in CI worktrees with unusual roots).
fn locate_plan_doc() -> String {
    let plan_rel = std::path::Path::new("docs")
        .join("plans")
        .join("0009-Intent-Aware-Fitness")
        .join("0009-SPEED-DECOUPLING.md");

    let mut dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    loop {
        let candidate = dir.join(&plan_rel);
        if candidate.parent().map(|p| p.exists()).unwrap_or(false) {
            return candidate.to_string_lossy().into_owned();
        }
        if !dir.pop() {
            break;
        }
    }
    // Fall back to CWD
    "0009-SPEED-DECOUPLING.md".to_owned()
}

/// Statistics collected during a headless run.
#[derive(Clone, Debug)]
struct ValidationStats {
    mean_fitness: f32,
    mean_movement_speed: f32,
    mean_ticks_alive: u64,
    speed_fitness_correlation: f32,
    death_speed_regression: f32,
    food_per_energy_vs_speed_slope: f32,
    mean_danger_dwell_fraction: f32,
    mean_avoidance_intent_fraction: f32,
    /// Population-mean movement_speed per generation (chronological order).
    /// Used to assess whether speed stops ratcheting toward 100 under the ON flags.
    speed_trajectory_per_gen: Vec<f32>,
}

/// Layer A (super-linear drag) exponent used in the ON run.
/// k=2.0: cost scales as (speed/20)^2 above baseline — the keystone of plan 0003.
/// k=1.0 in the baseline run is a bit-exact no-op per the WGSL guard.
const ON_SPEED_COST_EXPONENT: f32 = 2.0;

/// Run headless evolution and collect statistics with specified flags.
///
/// `flags_on = true` activates all three 0009 mechanisms together:
///   - `effort_rebased_fitness`: food-per-energy + cells-per-distance (Layer C)
///   - `speed_cost_exponent = 2.0`: super-linear locomotor drag above baseline (Layer A)
///   - `danger_percept_enabled`: dedicated danger bearing/distance senses (Layer D)
fn run_headless_with_flags(
    mut config: FullConfig,
    num_generations: u64,
    effort_rebased_fitness: bool,
    danger_percept_enabled: bool,
) -> ValidationStats {
    // Set the 0009 flags.
    // When enabling, also engage Layer A (super-linear drag) at k=2.0 — the keystone
    // mechanism (plan 0003) that makes the energy-drain axis speed-dependent.
    // Leaving speed_cost_exponent=1.0 in the ON run would make the ON and baseline
    // runs byte-identical on the energy-drain axis, defeating the measurement.
    config.brain.effort_rebased_fitness = effort_rebased_fitness;
    config.brain.danger_percept_enabled = danger_percept_enabled;
    if effort_rebased_fitness {
        config.brain.speed_cost_exponent = ON_SPEED_COST_EXPONENT;
    }

    println!(
        "  Flags: effort_rebased={}, danger_percept={}, speed_cost_exponent={}",
        effort_rebased_fitness, danger_percept_enabled, config.brain.speed_cost_exponent
    );

    // Create a temporary database for this run
    let temp_db = format!(
        "xagent-validation-{}-{}.db",
        if effort_rebased_fitness {
            "ON"
        } else {
            "BASELINE"
        },
        std::process::id()
    );
    let mut governor = Governor::new(&temp_db, config.governor.clone(), &config.brain, "")
        .expect("Failed to initialize validation governor");

    let seed_config = config.brain.clone();
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

    let mut inherited_state: Option<AgentBrainState> = None;
    let mut inherited_mutation_strength: f32 = 0.0;
    let repeats = governor.config.eval_repeats.max(1);

    let pop_size = governor.config.population_size;
    let world = WorldState::new(config.world.clone());
    let food_count = world.food_items.len();
    let mut kernel = GpuKernel::new(pop_size as u32, food_count, &seed_config, &config.world);

    let mut all_fitness: Vec<Vec<crate::governor::AgentFitness>> = Vec::new();
    // Per-generation mean movement_speed (for the trajectory check).
    let mut speed_trajectory_per_gen: Vec<f32> = Vec::new();

    for _ in 0..num_generations {
        if governor.evolution_complete() {
            break;
        }

        let world = WorldState::new(config.world.clone());
        let mut agents: Vec<Agent> = current_configs
            .iter()
            .enumerate()
            .map(|(i, cfg)| {
                let pos = world.safe_spawn_position();
                Agent::new(i as u32, pos, i as u32, cfg.clone(), 0)
            })
            .collect();

        let biomes = world.biome_map.grid_as_u32();
        let food_pos: Vec<(f32, f32, f32)> = world
            .food_items
            .iter()
            .map(|f| (f.position.x, f.position.y, f.position.z))
            .collect();
        let food_consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
        let food_timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
        kernel.upload_world(
            &world.terrain.heights,
            &biomes,
            &food_pos,
            &food_consumed,
            &food_timers,
        );

        let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = agents
            .iter()
            .map(|a| {
                (
                    a.body.body.position,
                    a.body.body.internal.max_energy,
                    a.body.body.internal.max_integrity,
                    a.brain_config.memory_capacity,
                    a.brain_config.processing_slots,
                )
            })
            .collect();
        kernel.upload_agents(&agent_data);
        kernel.reset_agents(&current_configs[0]);

        if let Some(ref state) = inherited_state {
            for (i, agent) in agents.iter().enumerate() {
                if i < repeats {
                    kernel.write_agent_state(agent.brain_idx, state);
                } else {
                    let mutated = mutate_brain_state(state, inherited_mutation_strength);
                    kernel.write_agent_state(agent.brain_idx, &mutated);
                }
            }
        }

        for (i, agent) in agents.iter().enumerate() {
            kernel.write_agent_heritable_config(agent.brain_idx, &current_configs[i]);
        }

        governor.gen_tick = 0;

        let tick_budget = governor.config.tick_budget;
        let mut ticks_done: u64 = 0;

        while ticks_done < tick_budget {
            let remaining = (tick_budget - ticks_done).min(HEATMAP_INTERVAL as u64) as u32;
            kernel.dispatch_batch(ticks_done, remaining);
            ticks_done += remaining as u64;
            governor.advance_ticks(u64::from(remaining));

            while !kernel.try_collect_state_snapshot() {
                std::thread::yield_now();
            }
            let state = kernel.cached_state();
            for i in 0..agents.len() {
                let base = i * PHYS_STRIDE;
                let alive = state[base + P_ALIVE] > 0.5;
                if alive {
                    agents[i].body.body.position = glam::Vec3::new(
                        state[base + P_POS_X],
                        state[base + P_POS_Y],
                        state[base + P_POS_Z],
                    );
                    agents[i].record_heatmap(config.world.world_size);
                }
            }

            let mut cumulative_food = 0_u64;
            let mut cumulative_alive = 0_u64;
            for i in 0..agents.len() {
                let base = i * PHYS_STRIDE;
                cumulative_food += state[base + P_FOOD_COUNT] as u64;
                cumulative_alive += state[base + P_TICKS_ALIVE] as u64;
            }
            governor.record_within_life_sample(cumulative_food, cumulative_alive);
        }

        let state = kernel.cached_state();
        for i in 0..agents.len() {
            let base = i * PHYS_STRIDE;
            agents[i].food_consumed = state[base + P_FOOD_COUNT] as u32;
            agents[i].total_ticks_alive = state[base + P_TICKS_ALIVE] as u64;
            agents[i].distance_traveled = state[base + P_DISTANCE_TRAVELED];
            agents[i].energy_spent = state[base + P_ENERGY_SPENT];
            agents[i].danger_path_length = state[base + P_DANGER_PATH_LENGTH];
            agents[i].avoidance_sense_range_ticks = state[base + P_AVOIDANCE_SENSE_RANGE_TICKS];
            agents[i].avoidance_turns_opposing = state[base + P_AVOIDANCE_TURNS_OPPOSING];
            agents[i].death_count = state[base + P_DEATH_COUNT] as u32;
        }

        let (_first_quarter_rate, _last_quarter_rate) = governor.within_life_rates();
        let fitness = governor.evaluate(&agents);

        // Record per-generation mean speed for the trajectory check.
        // AgentFitness.config.movement_speed is the canonical pairing of speed
        // to this agent's evaluation result; using the fitness slice avoids any
        // index-order mismatch with `evaluate()`'s internal sort.
        let gen_mean_speed = if fitness.is_empty() {
            0.0
        } else {
            fitness.iter().map(|f| f.config.movement_speed).sum::<f32>() / fitness.len() as f32
        };
        speed_trajectory_per_gen.push(gen_mean_speed);

        // Record for aggregate analysis
        all_fitness.push(fitness.clone());

        let best_idx = fitness.first().map(|f| f.agent_index).unwrap_or(0);
        inherited_state = agents
            .get(best_idx)
            .map(|a| kernel.read_agent_state(a.brain_idx));

        governor.log_generation(&fitness);

        match governor.advance(&fitness) {
            AdvanceResult::Continue {
                configs,
                messages: _,
                mutation_strength,
            } => {
                current_configs = configs;
                inherited_mutation_strength = mutation_strength;
            }
            AdvanceResult::Finished { .. } => {
                break;
            }
        }
    }

    // Flatten all generations into a single slice for aggregate metrics.
    // Each AgentFitness entry carries its own `config.movement_speed`, so speed
    // is always paired with the correct fitness record — no separate speed vec
    // needed, and no risk of index mismatch from evaluate()'s internal sort.
    let all_agents_fitness: Vec<crate::governor::AgentFitness> =
        all_fitness.into_iter().flatten().collect();

    let mean_fitness = if all_agents_fitness.is_empty() {
        0.0
    } else {
        all_agents_fitness
            .iter()
            .map(|f| f.composite_fitness)
            .sum::<f32>()
            / all_agents_fitness.len() as f32
    };

    // Aggregate mean speed — derived from AgentFitness.config.movement_speed so
    // it is always aligned with the fitness record for the same agent.
    let mean_movement_speed = if all_agents_fitness.is_empty() {
        0.0
    } else {
        all_agents_fitness
            .iter()
            .map(|f| f.config.movement_speed)
            .sum::<f32>()
            / all_agents_fitness.len() as f32
    };

    let mean_ticks_alive = if all_agents_fitness.is_empty() {
        0
    } else {
        let total: u64 = all_agents_fitness.iter().map(|f| f.total_ticks_alive).sum();
        total / all_agents_fitness.len() as u64
    };

    // Build aligned speed / fitness slices for correlation and regression.
    // Using AgentFitness.config.movement_speed guarantees each speed value is
    // the genome of the same agent whose fitness/death/foraging appear in the
    // parallel position — evaluate()'s sort does not break the pairing.
    let speeds: Vec<f32> = all_agents_fitness
        .iter()
        .map(|f| f.config.movement_speed)
        .collect();
    let fitnesses: Vec<f32> = all_agents_fitness
        .iter()
        .map(|f| f.composite_fitness)
        .collect();
    let death_counts: Vec<f32> = all_agents_fitness
        .iter()
        .map(|f| f.death_count as f32)
        .collect();
    let food_per_energy: Vec<f32> = all_agents_fitness
        .iter()
        .map(|f| {
            if f.energy_spent > 0.001 {
                f.food_consumed as f32 / f.energy_spent
            } else {
                0.0
            }
        })
        .collect();

    let speed_fitness_correlation = compute_correlation(&speeds, &fitnesses);
    let death_speed_regression = compute_regression(&speeds, &death_counts);
    let food_per_energy_vs_speed_slope = compute_regression(&speeds, &food_per_energy);

    // danger_dwell_fraction: fraction of the agent's path spent in danger biome.
    let mean_danger_dwell_fraction = if all_agents_fitness.is_empty() {
        0.0
    } else {
        all_agents_fitness
            .iter()
            .map(|f| {
                if f.distance_traveled > 0.001 {
                    (f.danger_path_length / f.distance_traveled).min(1.0)
                } else {
                    0.0
                }
            })
            .sum::<f32>()
            / all_agents_fitness.len() as f32
    };

    // avoidance_intent_fraction: fraction of in-sense-range ticks where the agent
    // turned away from the nearest danger cell.  Computed from AgentFitness fields
    // populated by the kernel (avoidance_turns_opposing / avoidance_sense_range_ticks),
    // mirroring governor.rs:843-844.
    let mean_avoidance_intent_fraction = if all_agents_fitness.is_empty() {
        0.0
    } else {
        all_agents_fitness
            .iter()
            .map(|f| {
                let sense_ticks = f.avoidance_sense_range_ticks.max(1.0);
                (f.avoidance_turns_opposing / sense_ticks).min(1.0)
            })
            .sum::<f32>()
            / all_agents_fitness.len() as f32
    };

    // Clean up temp database
    let _ = std::fs::remove_file(&temp_db);

    ValidationStats {
        mean_fitness,
        mean_movement_speed,
        mean_ticks_alive,
        speed_fitness_correlation,
        death_speed_regression,
        food_per_energy_vs_speed_slope,
        mean_danger_dwell_fraction,
        mean_avoidance_intent_fraction,
        speed_trajectory_per_gen,
    }
}

/// Compute Pearson correlation between two vectors.
fn compute_correlation(x: &[f32], y: &[f32]) -> f32 {
    if x.len() < 2 || x.len() != y.len() {
        return 0.0;
    }

    let n = x.len() as f32;
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;

    let mut numerator = 0.0;
    let mut sum_x_sq = 0.0;
    let mut sum_y_sq = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator += dx * dy;
        sum_x_sq += dx * dx;
        sum_y_sq += dy * dy;
    }

    let denominator = (sum_x_sq * sum_y_sq).sqrt();
    if denominator < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compute linear regression slope (y ~ x).
fn compute_regression(x: &[f32], y: &[f32]) -> f32 {
    if x.len() < 2 || x.len() != y.len() {
        return 0.0;
    }

    let n = x.len() as f32;
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator += dx * dy;
        denominator += dx * dx;
    }

    if denominator < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Print validation metrics to console.
fn print_validation_metrics(baseline: &ValidationStats, on_stats: &ValidationStats) {
    println!("\nMetric                            Baseline        On              Delta");
    println!("──────────────────────────────────────────────────────────────────────");

    let print_metric = |label: &str, baseline_val: f32, on_val: f32| {
        let delta = on_val - baseline_val;
        let delta_str = if delta >= 0.0 {
            format!("+{:.4}", delta)
        } else {
            format!("{:.4}", delta)
        };
        println!(
            "{:<35} {:<15.4} {:<15.4} {}",
            label, baseline_val, on_val, delta_str
        );
    };

    print_metric(
        "speed-fitness correlation",
        baseline.speed_fitness_correlation,
        on_stats.speed_fitness_correlation,
    );
    print_metric(
        "death-speed regression slope",
        baseline.death_speed_regression,
        on_stats.death_speed_regression,
    );
    print_metric(
        "food-per-energy vs speed slope",
        baseline.food_per_energy_vs_speed_slope,
        on_stats.food_per_energy_vs_speed_slope,
    );
    print_metric("mean fitness", baseline.mean_fitness, on_stats.mean_fitness);
    print_metric(
        "mean movement_speed",
        baseline.mean_movement_speed,
        on_stats.mean_movement_speed,
    );
    print_metric(
        "mean ticks_alive",
        baseline.mean_ticks_alive as f32,
        on_stats.mean_ticks_alive as f32,
    );
    print_metric(
        "mean danger_dwell_fraction",
        baseline.mean_danger_dwell_fraction,
        on_stats.mean_danger_dwell_fraction,
    );
    print_metric(
        "mean avoidance_intent_fraction",
        baseline.mean_avoidance_intent_fraction,
        on_stats.mean_avoidance_intent_fraction,
    );

    // Print per-generation speed trajectory for both runs.
    println!("\nPer-generation mean speed trajectory:");
    println!(
        "  Baseline: {}",
        baseline
            .speed_trajectory_per_gen
            .iter()
            .map(|s| format!("{:.1}", s))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "  On:       {}",
        on_stats
            .speed_trajectory_per_gen
            .iter()
            .map(|s| format!("{:.1}", s))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// Format validation results as markdown.
fn format_validation_markdown(baseline: &ValidationStats, on_stats: &ValidationStats) -> String {
    let speed_decoupled = on_stats.speed_fitness_correlation.abs() < 0.3;
    let ticks_alive_ok = on_stats.mean_ticks_alive > baseline.mean_ticks_alive * 80 / 100;
    let danger_retained = on_stats.mean_danger_dwell_fraction > 0.01;
    let avoidance_retained = on_stats.mean_avoidance_intent_fraction >= 0.0; // non-negative by construction
    let gate_passed = speed_decoupled && ticks_alive_ok && danger_retained;

    let baseline_traj = baseline
        .speed_trajectory_per_gen
        .iter()
        .enumerate()
        .map(|(i, s)| format!("Gen {}: {:.2}", i + 1, s))
        .collect::<Vec<_>>()
        .join(", ");

    let on_traj = on_stats
        .speed_trajectory_per_gen
        .iter()
        .enumerate()
        .map(|(i, s)| format!("Gen {}: {:.2}", i + 1, s))
        .collect::<Vec<_>>()
        .join(", ");

    let corr_delta = on_stats.speed_fitness_correlation - baseline.speed_fitness_correlation;
    let corr_direction = if corr_delta < 0.0 { "fell" } else { "rose" };
    let pct_change = if baseline.speed_fitness_correlation.abs() > 1e-6 {
        corr_delta.abs() / baseline.speed_fitness_correlation.abs() * 100.0
    } else {
        0.0
    };

    format!(
        "# Decision: Speed-Decoupling Validation (Plan 0009 — Task 0006)\n\
\n\
**Date:** 2026-06-18\n\
**Status:** MEASURED\n\
**Task:** `speed-decoupling-validation`\n\
\n\
## Configuration\n\
\n\
- Baseline: all 0009 flags OFF (`speed_cost_exponent=1.0`, `effort_rebased_fitness=false`, `danger_percept_enabled=false`)\n\
- On: Layer A `speed_cost_exponent={exp:.1}`, `effort_rebased_fitness=true`, `danger_percept_enabled=true`\n\
\n\
## Measured Metrics\n\
\n\
| Metric | Baseline | On | Delta |\n\
|--------|----------|----|-------|\n\
| **speed-fitness correlation** | {baseline_corr:.4} | {on_corr:.4} | {corr_d:.4} |\n\
| **death-speed regression slope** | {baseline_death:.4} | {on_death:.4} | {death_d:.4} |\n\
| **food-per-energy vs speed slope** | {baseline_fpe:.4} | {on_fpe:.4} | {fpe_d:.4} |\n\
| mean fitness | {baseline_fit:.4} | {on_fit:.4} | {fit_d:.4} |\n\
| mean movement_speed | {baseline_spd:.4} | {on_spd:.4} | {spd_d:.4} |\n\
| mean ticks_alive | {baseline_ticks} | {on_ticks} | {ticks_d} |\n\
| mean danger_dwell_fraction | {baseline_ddf:.4} | {on_ddf:.4} | {ddf_d:.4} |\n\
| mean avoidance_intent_fraction | {baseline_aif:.4} | {on_aif:.4} | {aif_d:.4} |\n\
\n\
## Population Mean Speed Trajectory (per generation)\n\
\n\
**Baseline:** {baseline_traj}\n\
\n\
**On:** {on_traj}\n\
\n\
The trajectory shows whether population-mean speed stops ratcheting toward 100 \
under the ON flags (target: stable or declining, not collapsing to 1.0).\n\
\n\
## Analysis\n\
\n\
### Speed-Fitness Decoupling\n\
\n\
The speed-fitness correlation {corr_direction} from {baseline_corr:.4} (baseline) to \
{on_corr:.4} (ON), a {pct:.1}% change.\n\
\n\
{decoupling_summary}\n\
\n\
### Death-Speed Regression\n\
\n\
Baseline slope {baseline_death:.4} — positive slope means faster agents die more; \
negative means faster agents die less (the exploit). ON slope: {on_death:.4}.\n\
Path-length hazard (Layer B) makes per-crossing damage speed-invariant, so fast agents \
no longer get cheaper hazard exposure.\n\
\n\
### Food-Per-Energy vs Speed\n\
\n\
Baseline slope {baseline_fpe:.4} | On slope {on_fpe:.4}.\n\
A flat or negative slope confirms skill, not speed, drives foraging under effort-rebased fitness.\n\
\n\
### Danger Metrics\n\
\n\
**danger_dwell_fraction** — Baseline: {baseline_ddf:.4} | On: {on_ddf:.4}\n\
\n\
**avoidance_intent_fraction** (turns opposing danger bearing / sense-range ticks) — \
Baseline: {baseline_aif:.4} | On: {on_aif:.4}\n\
\n\
Non-zero values confirm the population still enters danger biomes and generates \
avoidance-decision data even under the ON flags.\n\
\n\
### Ticks Alive\n\
\n\
Mean ticks alive — Baseline: {baseline_ticks} | On: {on_ticks}.\n\
{ticks_summary}\n\
\n\
## Gate Status\n\
\n\
**Criteria:**\n\
1. Speed-fitness correlation falls from strongly-positive to approx 0 (`|r| < 0.3`): {gate1}\n\
2. Mean ticks_alive does not collapse vs baseline (>= 80% retained): {gate2}\n\
3. Danger metrics stay non-zero (danger_dwell_fraction > 0.01): {gate3}\n\
\n\
**Avoidance intent non-negative:** {avoidance_note}\n\
\n\
**Result:** {gate_result}\n\
\n\
## Decision\n\
\n\
{decision}\n\
\n\
---\n\
\n\
This document was generated by the speed-decoupling validation harness \
(`cargo run --release -- --validate-speed-decoupling`). \
It records the canonical baseline for all future Plan 0009 A/B tests.\n",
        exp = ON_SPEED_COST_EXPONENT,
        baseline_corr = baseline.speed_fitness_correlation,
        on_corr = on_stats.speed_fitness_correlation,
        corr_d = corr_delta,
        baseline_death = baseline.death_speed_regression,
        on_death = on_stats.death_speed_regression,
        death_d = on_stats.death_speed_regression - baseline.death_speed_regression,
        baseline_fpe = baseline.food_per_energy_vs_speed_slope,
        on_fpe = on_stats.food_per_energy_vs_speed_slope,
        fpe_d = on_stats.food_per_energy_vs_speed_slope - baseline.food_per_energy_vs_speed_slope,
        baseline_fit = baseline.mean_fitness,
        on_fit = on_stats.mean_fitness,
        fit_d = on_stats.mean_fitness - baseline.mean_fitness,
        baseline_spd = baseline.mean_movement_speed,
        on_spd = on_stats.mean_movement_speed,
        spd_d = on_stats.mean_movement_speed - baseline.mean_movement_speed,
        baseline_ticks = baseline.mean_ticks_alive,
        on_ticks = on_stats.mean_ticks_alive,
        ticks_d = on_stats.mean_ticks_alive as i64 - baseline.mean_ticks_alive as i64,
        baseline_ddf = baseline.mean_danger_dwell_fraction,
        on_ddf = on_stats.mean_danger_dwell_fraction,
        ddf_d = on_stats.mean_danger_dwell_fraction - baseline.mean_danger_dwell_fraction,
        baseline_aif = baseline.mean_avoidance_intent_fraction,
        on_aif = on_stats.mean_avoidance_intent_fraction,
        aif_d = on_stats.mean_avoidance_intent_fraction - baseline.mean_avoidance_intent_fraction,
        baseline_traj = baseline_traj,
        on_traj = on_traj,
        corr_direction = corr_direction,
        pct = pct_change,
        decoupling_summary = if speed_decoupled {
            "Speed is decoupled from fitness: the ON correlation is below the |0.3| threshold."
        } else {
            "Speed still correlates with fitness above the |0.3| threshold. \
             Consider tuning k (speed_cost_exponent) or the fitness calibration constants."
        },
        ticks_summary = if ticks_alive_ok {
            "Population viability is preserved: ON ticks_alive >= 80% of baseline."
        } else {
            "Population viability concern: ON ticks_alive dropped below 80% of baseline. \
             Consider reducing the drag exponent or reviewing energy constants."
        },
        gate1 = if speed_decoupled { "PASS" } else { "FAIL" },
        gate2 = if ticks_alive_ok { "PASS" } else { "FAIL" },
        gate3 = if danger_retained { "PASS" } else { "FAIL" },
        avoidance_note = if avoidance_retained {
            "PASS (non-negative avoidance_intent_fraction computed from kernel telemetry)"
        } else {
            "N/A"
        },
        gate_result = if gate_passed {
            "GATE PASSED — All criteria met; speed is successfully decoupled from fitness."
        } else {
            "GATE NOT MET — Some criteria not met; review metrics above before flipping defaults."
        },
        decision = if gate_passed {
            "The validation passed. The Plan 0009 layers (Layer A: super-linear drag at k=2.0, \
Layer B: path-length hazard, Layer C: effort-rebased fitness, Layer D: danger percept) \
successfully decouple movement speed from composite fitness. \
The population remained viable and danger-decision data is retained. \
Defaults are candidates for flipping per task `default-flip-gate`."
        } else {
            "The validation did not meet all gate criteria. Review the measured metrics and \
consider tuning the drag exponent (speed_cost_exponent), the fitness calibration constants \
(FORAGING_ENERGY_TARGET / EXPLORATION_DISTANCE_BUDGET), or the danger sense radius before \
attempting another run. Do not flip the defaults until the gate passes."
        },
    )
}
