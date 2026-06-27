//! Headless (no-window) evolution loop and tree dump utilities.
//!
//! Uses GpuKernel for all simulation — physics, brain, food, and
//! death/respawn run entirely on GPU via fused kernel dispatch.

use std::time::Instant;

use log::info;
use serde::{Deserialize, Serialize};
use xagent_shared::{BrainConfig, FullConfig};

use xagent_brain::buffers::{
    BrainLayout, ENCODED_DIMENSION, O_ACTION_FORWARD_WEIGHTS, O_ACTION_TURN_WEIGHTS,
    O_PREDICTOR_CONTEXT_WEIGHT, PHYS_STRIDE, PREDICTOR_DIMENSION, P_ALIVE,
    P_APPROACH_SENSE_RANGE_TICKS, P_APPROACH_TURNS_TOWARD, P_AVOIDANCE_SENSE_RANGE_TICKS,
    P_AVOIDANCE_TURNS_OPPOSING, P_DANGER_PATH_LENGTH, P_DEATH_COUNT, P_DISTANCE_TRAVELED,
    P_ENERGY_SPENT, P_FOOD_COUNT, P_POS_X, P_POS_Y, P_POS_Z, P_TICKS_ALIVE,
};
use xagent_brain::{AgentBrainState, GpuKernel};

use crate::agent::{
    mutate_brain_state, mutate_brain_state_seeded, mutate_config, mutate_config_seeded, Agent,
};
use crate::governor::{
    compute_approach_intent_fraction, compute_avoidance_intent_fraction,
    compute_danger_dwell_fraction, AdvanceResult, Governor,
};
use crate::world::WorldState;

/// Chunk size for dispatch_batch calls in the standard headless run.
/// Between chunks we read back positions for heatmap recording.
const HEATMAP_INTERVAL: u32 = 100;

/// Number of position samples taken per generation in the validation dispatch loop.
/// 4 samples (one per quarter of the tick budget) satisfies the within-life tracker's
/// quarter-boundary requirements while keeping the GPU readback count low.  The
/// actual dispatch batch size is `tick_budget / VALIDATION_HEATMAP_SAMPLES`, clamped
/// to at least `HEATMAP_INTERVAL` (100 ticks) so it is always a valid batch.
const VALIDATION_HEATMAP_SAMPLES: u32 = 4;

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
            agents[i].approach_sense_range_ticks = state[base + P_APPROACH_SENSE_RANGE_TICKS];
            agents[i].approach_turns_toward = state[base + P_APPROACH_TURNS_TOWARD];
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

/// Run the paired baseline-vs-ON speed-decoupling A/B at a fixed envelope.
///
/// Wraps `num_replicates` bootstrap replicates (baseline: all flags OFF; ON: effort-rebased
/// fitness + super-linear locomotor drag at `speed_cost_exponent=2.0`, danger percept OFF) and
/// computes 95% CI for four metrics: `mean_ticks_alive`, `speed_fitness_correlation`,
/// `mean_fitness`, `danger_dwell_fraction`. Outputs JSON with point/CI/effect and a
/// machine-readable decision rule.
///
/// Production default: 100 replicates, population 100, 50 generations. Pass smaller values via
/// `--validation-replicates` / `--validation-population` / `--validation-generations` for quick
/// hardware-limited checks.
///
/// `tick_budget_override`: if non-zero, replaces the governor's default 1 M-tick budget per
/// generation. Use a smaller value (e.g. 10_000) to make N=100 production-scale bootstrap
/// feasible on hardware where 1 M ticks × 50 gen × 100 pop × 200 arm-calls is prohibitive;
/// 0 means keep the governor's configured value.
pub fn validate_speed_decoupling(
    config: FullConfig,
    num_generations: u64,
    population: u32,
    num_replicates: usize,
    tick_budget_override: u64,
) {
    let effective_tick_budget = if tick_budget_override > 0 {
        tick_budget_override
    } else {
        config.governor.tick_budget
    };

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SPEED-DECOUPLING PRODUCTION A/B VALIDATION");
    println!(
        "Running N={} bootstrap replicates at population {} × {} generations × {} ticks/gen",
        num_replicates, population, num_generations, effective_tick_budget
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Collect baseline and ON metrics across N replicates.
    let mut baseline_ticks_alive: Vec<f32> = Vec::new();
    let mut baseline_speed_correlation: Vec<f32> = Vec::new();
    let mut baseline_fitness: Vec<f32> = Vec::new();
    let mut baseline_danger_dwell: Vec<f32> = Vec::new();

    let mut on_ticks_alive: Vec<f32> = Vec::new();
    let mut on_speed_correlation: Vec<f32> = Vec::new();
    let mut on_fitness: Vec<f32> = Vec::new();
    let mut on_danger_dwell: Vec<f32> = Vec::new();

    for replicate in 0..num_replicates {
        println!(
            "\n[Replicate {}/{}] Running baseline and ON arms...",
            replicate + 1,
            num_replicates
        );

        // Generate a seeded but independent world for this replicate.
        let mut replicate_config = config.clone();
        replicate_config.governor.population_size = population as usize;
        replicate_config.world.seed = config.world.seed.wrapping_add(replicate as u64);
        // Apply tick_budget_override when set: both arms use the same budget for a fair A/B.
        if tick_budget_override > 0 {
            replicate_config.governor.tick_budget = tick_budget_override;
        }

        // Baseline run (all flags off)
        let baseline_stats = run_headless_with_flags(
            replicate_config.clone(),
            num_generations,
            false,
            false,
            false,
        );

        // ON run: effort-rebased fitness, super-linear drag at k=2.0, danger percept OFF.
        // danger_percept_enabled=false isolates the effort-fitness + speed-cost axis from the
        // danger-percept signal, so the two mechanisms are measured independently.
        let on_stats =
            run_headless_with_flags(replicate_config, num_generations, true, false, false);

        // Record metrics for this replicate.
        baseline_ticks_alive.push(baseline_stats.mean_ticks_alive as f32);
        baseline_speed_correlation.push(baseline_stats.speed_fitness_correlation);
        baseline_fitness.push(baseline_stats.mean_fitness);
        baseline_danger_dwell.push(baseline_stats.mean_danger_dwell_fraction);

        on_ticks_alive.push(on_stats.mean_ticks_alive as f32);
        on_speed_correlation.push(on_stats.speed_fitness_correlation);
        on_fitness.push(on_stats.mean_fitness);
        on_danger_dwell.push(on_stats.mean_danger_dwell_fraction);
    }

    // Compute bootstrap metrics: point estimate (mean), lower/upper 95% CI, effect.
    let ticks_alive_baseline = compute_bootstrap_metric(&baseline_ticks_alive, None);
    let ticks_alive_on =
        compute_bootstrap_metric(&on_ticks_alive, Some(ticks_alive_baseline.point));

    let speed_correlation_baseline = compute_bootstrap_metric(&baseline_speed_correlation, None);
    let speed_correlation_on = compute_bootstrap_metric(
        &on_speed_correlation,
        Some(speed_correlation_baseline.point),
    );

    let fitness_baseline = compute_bootstrap_metric(&baseline_fitness, None);
    let fitness_on = compute_bootstrap_metric(&on_fitness, Some(fitness_baseline.point));

    let danger_dwell_baseline = compute_bootstrap_metric(&baseline_danger_dwell, None);
    let danger_dwell_on =
        compute_bootstrap_metric(&on_danger_dwell, Some(danger_dwell_baseline.point));

    // Output JSON — include run parameters alongside metrics so the artifact is self-describing.
    let json_output = serde_json::json!({
        "run_parameters": {
            "num_replicates": num_replicates,
            "population": population,
            "num_generations": num_generations,
            "tick_budget_per_generation": effective_tick_budget,
        },
        "mean_ticks_alive_baseline": ticks_alive_baseline,
        "mean_ticks_alive_on": ticks_alive_on,
        "speed_correlation_baseline": speed_correlation_baseline,
        "speed_correlation_on": speed_correlation_on,
        "mean_fitness_baseline": fitness_baseline,
        "mean_fitness_on": fitness_on,
        "danger_dwell_fraction_baseline": danger_dwell_baseline,
        "danger_dwell_fraction_on": danger_dwell_on,
    });

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BOOTSTRAP RESULTS (JSON)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{}",
        serde_json::to_string_pretty(&json_output).unwrap_or_default()
    );

    // Save JSON output.
    let json_path = "speed_decoupling_bootstrap.json";
    match std::fs::write(
        json_path,
        serde_json::to_string_pretty(&json_output).unwrap_or_default(),
    ) {
        Ok(()) => println!("\nJSON output saved to ./{}", json_path),
        Err(e) => eprintln!("Failed to write {}: {}", json_path, e),
    }

    // Print decision rule.
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DECISION RULE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    print_speed_decoupling_decision_rule(
        &ticks_alive_baseline,
        &ticks_alive_on,
        &speed_correlation_baseline,
        &speed_correlation_on,
        &danger_dwell_baseline,
        &danger_dwell_on,
    );
}

/// Run danger-percept production A/B validation with bootstrap 95% confidence intervals.
/// Baseline: danger_percept_enabled=false (all other flags off).
/// ON: danger_percept_enabled=true (all other flags off).
/// Measures avoidance-intent, approach-intent, survival (ticks_alive), and steering_alignment
/// across N bootstrap replicates at production scale.
///
/// `tick_budget_override`: if non-zero, replaces the governor's default 1 M-tick budget per
/// generation. Use a smaller value (e.g. 10_000) to make N=100 production-scale bootstrap
/// feasible on hardware where 1 M ticks × 50 gen × 100 pop × 200 arm-calls is prohibitive;
/// 0 means keep the governor's configured value.
pub fn validate_danger_percept(
    config: FullConfig,
    num_generations: u64,
    population: u32,
    num_replicates: usize,
    tick_budget_override: u64,
) {
    let effective_tick_budget = if tick_budget_override > 0 {
        tick_budget_override
    } else {
        config.governor.tick_budget
    };

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DANGER-PERCEPT PRODUCTION A/B VALIDATION");
    println!(
        "Running N={} bootstrap replicates at population {} × {} generations × {} ticks/gen",
        num_replicates, population, num_generations, effective_tick_budget
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Collect metrics across N replicates: baseline (danger_percept OFF) and ON.
    let mut baseline_avoidance_intent: Vec<f32> = Vec::new();
    let mut baseline_approach_intent: Vec<f32> = Vec::new();
    let mut baseline_ticks_alive: Vec<f32> = Vec::new();
    let mut baseline_steering_alignment: Vec<f32> = Vec::new();

    let mut on_avoidance_intent: Vec<f32> = Vec::new();
    let mut on_approach_intent: Vec<f32> = Vec::new();
    let mut on_ticks_alive: Vec<f32> = Vec::new();
    let mut on_steering_alignment: Vec<f32> = Vec::new();

    for replicate in 0..num_replicates {
        println!(
            "\n[Replicate {}/{}] Running baseline (danger_percept OFF) and ON arms...",
            replicate + 1,
            num_replicates
        );

        // Generate a seeded but independent world for this replicate.
        let mut replicate_config = config.clone();
        replicate_config.governor.population_size = population as usize;
        replicate_config.world.seed = config.world.seed.wrapping_add(replicate as u64);
        // Apply tick_budget_override when set: both arms use the same budget for a fair A/B.
        if tick_budget_override > 0 {
            replicate_config.governor.tick_budget = tick_budget_override;
        }

        // Baseline run: all flags off, danger_percept_enabled=false
        let baseline_stats = run_headless_with_flags(
            replicate_config.clone(),
            num_generations,
            false,
            false,
            false,
        );

        // ON run: danger_percept_enabled=true, all other flags off
        let on_stats =
            run_headless_with_flags(replicate_config, num_generations, false, true, false);

        // Record metrics for this replicate.
        baseline_avoidance_intent.push(baseline_stats.mean_avoidance_intent_fraction);
        baseline_approach_intent.push(baseline_stats.mean_approach_intent_fraction);
        baseline_ticks_alive.push(baseline_stats.mean_ticks_alive as f32);
        baseline_steering_alignment.push(baseline_stats.mean_steering_alignment);

        on_avoidance_intent.push(on_stats.mean_avoidance_intent_fraction);
        on_approach_intent.push(on_stats.mean_approach_intent_fraction);
        on_ticks_alive.push(on_stats.mean_ticks_alive as f32);
        on_steering_alignment.push(on_stats.mean_steering_alignment);
    }

    // Compute bootstrap metrics: point estimate (mean), lower/upper 95% CI, effect.
    let avoidance_baseline = compute_bootstrap_metric(&baseline_avoidance_intent, None);
    let avoidance_on =
        compute_bootstrap_metric(&on_avoidance_intent, Some(avoidance_baseline.point));

    let approach_baseline = compute_bootstrap_metric(&baseline_approach_intent, None);
    let approach_on = compute_bootstrap_metric(&on_approach_intent, Some(approach_baseline.point));

    let ticks_alive_baseline = compute_bootstrap_metric(&baseline_ticks_alive, None);
    let ticks_alive_on =
        compute_bootstrap_metric(&on_ticks_alive, Some(ticks_alive_baseline.point));

    let steering_baseline = compute_bootstrap_metric(&baseline_steering_alignment, None);
    let steering_on =
        compute_bootstrap_metric(&on_steering_alignment, Some(steering_baseline.point));

    // Output JSON — include run parameters alongside metrics so the artifact is self-describing.
    let json_output = serde_json::json!({
        "run_parameters": {
            "num_replicates": num_replicates,
            "population": population,
            "num_generations": num_generations,
            "tick_budget_per_generation": effective_tick_budget,
        },
        "avoidance_intent_baseline": avoidance_baseline,
        "avoidance_intent_on": avoidance_on,
        "approach_intent_baseline": approach_baseline,
        "approach_intent_on": approach_on,
        "mean_ticks_alive_baseline": ticks_alive_baseline,
        "mean_ticks_alive_on": ticks_alive_on,
        "steering_alignment_baseline": steering_baseline,
        "steering_alignment_on": steering_on,
    });

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BOOTSTRAP RESULTS (JSON)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{}",
        serde_json::to_string_pretty(&json_output).unwrap_or_default()
    );

    // Save JSON output.
    let json_path = "danger_percept_bootstrap.json";
    match std::fs::write(
        json_path,
        serde_json::to_string_pretty(&json_output).unwrap_or_default(),
    ) {
        Ok(()) => println!("\nJSON output saved to ./{}", json_path),
        Err(e) => eprintln!("Failed to write {}: {}", json_path, e),
    }

    // Print decision rule.
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DECISION RULE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    print_danger_percept_decision_rule(
        &avoidance_baseline,
        &avoidance_on,
        &approach_baseline,
        &approach_on,
        &ticks_alive_baseline,
        &ticks_alive_on,
        &steering_baseline,
        &steering_on,
    );
}

/// Run innate-instinct prove-or-kill A/B benchmark.
/// Baseline: innate_instincts_enabled=false (blank slate, learning from scratch).
/// ON: innate_instincts_enabled=true (seeded instinct patterns + learning).
/// Both arms use identical seeded populations and worlds (deterministic mutations from config.world.seed).
/// Evaluates three gates: survival (+10%), alignment (>=0.4), food-per-death (>=2.0, real food/deaths).
pub fn validate_innate_instincts(config: FullConfig, num_generations: u64) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("INNATE-INSTINCT VALIDATION");
    println!(
        "Running {} generations with innate_instincts_enabled OFF (baseline) then ON",
        num_generations
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let (_baseline_stats, _on_stats, passed) = run_innate_instinct_ab(config, num_generations);

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!(
        "\nFinal Decision: {}",
        if passed {
            "✓✓✓ ALL GATES PASS"
        } else {
            "✗✗✗ GATE FAILURE"
        }
    );
}

/// Run innate-instinct A/B comparison and evaluate gates.
/// Returns (baseline_stats, on_stats, passed: bool).
pub fn run_innate_instinct_ab(
    config: FullConfig,
    num_generations: u64,
) -> (ValidationStats, ValidationStats, bool) {
    // Baseline: innate_instincts_enabled=false (blank slate, learning from scratch).
    // All speed-decoupling flags off — isolates the innate-instinct flag as the only variable.
    println!("\n=== Baseline (innate_instincts_enabled=false) ===");
    let baseline_stats =
        run_headless_with_flags(config.clone(), num_generations, false, false, false);

    // ON: innate_instincts_enabled=true (seeded instincts).
    // All speed-decoupling flags remain off — same single-variable isolation.
    println!("\n=== ON (innate_instincts_enabled=true) ===");
    let on_stats = run_headless_with_flags(config.clone(), num_generations, false, false, true);

    // Apply gates.
    println!("\n=== Gate Evaluation ===");

    let survival_gate = on_stats.mean_ticks_alive
        >= ((baseline_stats.mean_ticks_alive as f32) * (1.0 + INSTINCT_SURVIVAL_MARGIN)) as u64;
    println!(
        "Survival gate (ON >= baseline × {:.1}%): baseline={}, ON={} → {}",
        INSTINCT_SURVIVAL_MARGIN * 100.0,
        baseline_stats.mean_ticks_alive,
        on_stats.mean_ticks_alive,
        if survival_gate {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );

    let alignment_gate = on_stats.mean_avoidance_intent_fraction >= INSTINCT_ALIGNMENT_FLOOR;
    println!(
        "Alignment gate (avoidance-intent >= {:.1}): ON={:.3} → {}",
        INSTINCT_ALIGNMENT_FLOOR,
        on_stats.mean_avoidance_intent_fraction,
        if alignment_gate {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );

    // REAL food consumed (population mean), never mean_fitness (a composite).
    let food_per_death_on = if on_stats.mean_death_count > FOOD_PER_DEATH_ZERO_GUARD {
        on_stats.mean_food_consumed / on_stats.mean_death_count
    } else {
        f32::INFINITY
    };
    let food_per_death_gate = food_per_death_on >= INSTINCT_FOOD_PER_DEATH_MIN;
    println!(
        "Food-per-death gate (ratio >= {:.1}): ON={:.2} → {}",
        INSTINCT_FOOD_PER_DEATH_MIN,
        food_per_death_on,
        if food_per_death_gate {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );

    let passed = survival_gate && alignment_gate && food_per_death_gate;

    (baseline_stats, on_stats, passed)
}

/// One metric's bootstrap summary over N replicates: the point estimate,
/// its 95% CI bounds (2.5th/97.5th percentile), and the ON−baseline effect.
/// Recorded for ticks_alive, speed_correlation, fitness, and danger_dwell so the
/// flip/retire/defer rule is backed by precision, not point-estimate noise.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BootstrapMetric {
    pub point: f32,
    pub lower_ci: f32,
    pub upper_ci: f32,
    pub effect: f32,
}

/// Statistics collected during a headless run.
#[derive(Clone, Debug)]
pub struct ValidationStats {
    pub mean_fitness: f32,
    pub mean_movement_speed: f32,
    pub mean_ticks_alive: u64,
    /// Mean deaths per agent across all agents and all generations in this run.
    /// Uncapped: unlike `mean_ticks_alive`, this is never pinned to `tick_budget`.
    /// A value of 0.0 means no agent ever died — a population-viability red flag.
    pub mean_death_count: f32,
    /// Mean food consumed per agent across all agents and all generations.
    /// Used for food-per-death ratio calculation in instinct validation.
    pub mean_food_consumed: f32,
    pub speed_fitness_correlation: f32,
    pub death_speed_regression: f32,
    pub food_per_energy_vs_speed_slope: f32,
    pub mean_danger_dwell_fraction: f32,
    pub mean_avoidance_intent_fraction: f32,
    /// Mean approach-intent fraction: sum of approach turns toward food divided by sum of
    /// approach sense-range ticks, population aggregate. Paired with avoidance-intent
    /// for the danger-percept A/B evaluation.
    pub mean_approach_intent_fraction: f32,
    /// Mean per-agent steering alignment: how well the chosen turn tracks the danger/food
    /// gradient. Recorded as a deferral signal in danger-percept A/B to explain why intent
    /// improves but steering stays in the chance band (credit-path bottleneck).
    pub mean_steering_alignment: f32,
    /// Population-mean movement_speed per generation (chronological order).
    /// Used to assess whether speed stops ratcheting toward 100 under the ON flags.
    pub speed_trajectory_per_gen: Vec<f32>,
}

/// Super-linear locomotor-drag exponent used in the ON run.
/// k=2.0: energy cost scales as (speed/20)^2 above the baseline speed, so the
/// energy-drain axis becomes speed-dependent and faster movement costs
/// disproportionately more. k=1.0 in the baseline run is a bit-exact no-op per
/// the WGSL guard.
const ON_SPEED_COST_EXPONENT: f32 = 2.0;

/// Guard value for the food-per-death denominator.
/// Treats mean_death_count values below this threshold as effectively zero, returning
/// f32::INFINITY instead of dividing. Chosen at 1e-4 to absorb floating-point imprecision
/// near zero (e.g., a sub-1-per-10000 death rate is meaninglessly small as a divisor).
const FOOD_PER_DEATH_ZERO_GUARD: f32 = 1e-4;

/// Gate: ON (instincts seeded) must improve survival over baseline by at least this margin.
/// Tuned at 0.10 (10%) to require meaningful benefit while tolerating natural variance.
const INSTINCT_SURVIVAL_MARGIN: f32 = 0.10;

/// Gate: mean avoidance-intent fraction (turns opposing danger bearing) in ON run
/// must exceed this floor. Tuned at 0.4 to require substantial steering-alignment.
const INSTINCT_ALIGNMENT_FLOOR: f32 = 0.4;

/// Gate: ON run's food-per-death ratio (mean food count / mean death count) must
/// exceed this threshold. Tuned at 2.0 to require at least 2 food consumed per death.
const INSTINCT_FOOD_PER_DEATH_MIN: f32 = 2.0;

/// Run headless evolution and collect statistics with specified flags.
///
/// `effort_rebased_fitness = true` activates effort-rebased fitness + super-linear drag:
///   - `effort_rebased_fitness`: food-per-energy + cells-per-distance
///   - `speed_cost_exponent = 2.0`: super-linear locomotor drag above baseline
///   - `danger_percept_enabled`: dedicated danger bearing/distance senses
///   - `innate_instincts_enabled`: seeded instinct patterns for the innate-instinct A/B harness.
///     Pass `false` for the speed-decoupling harness (baseline and ON arms both use `false`).
fn run_headless_with_flags(
    mut config: FullConfig,
    num_generations: u64,
    effort_rebased_fitness: bool,
    danger_percept_enabled: bool,
    innate_instincts_enabled: bool,
) -> ValidationStats {
    // Set the validation flags.
    // When enabling effort-rebased fitness, also engage the super-linear drag at k=2.0 — the
    // keystone mechanism that makes the energy-drain axis speed-dependent.
    // Leaving speed_cost_exponent=1.0 in the ON run would make the ON and baseline
    // runs byte-identical on the energy-drain axis, defeating the measurement.
    config.brain.effort_rebased_fitness = effort_rebased_fitness;
    config.brain.danger_percept_enabled = danger_percept_enabled;
    config.brain.innate_instincts_enabled = innate_instincts_enabled;
    if effort_rebased_fitness {
        config.brain.speed_cost_exponent = ON_SPEED_COST_EXPONENT;
    }

    println!(
        "  Flags: effort_rebased={}, danger_percept={}, speed_cost_exponent={}, innate_instincts={}",
        effort_rebased_fitness,
        danger_percept_enabled,
        config.brain.speed_cost_exponent,
        innate_instincts_enabled
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
    // Derive a stable seed for population initialization from the world seed, ensuring both arms
    // (baseline and ON) get identical initial genomes when run with the same world seed.
    let pop_init_seed = config.world.seed;
    let mut current_configs: Vec<BrainConfig> = {
        let repeats = governor.config.eval_repeats.max(1);
        let unique_count = (governor.config.population_size / repeats).max(1);
        let mut unique_configs = vec![seed_config.clone()];
        for i in 1..unique_count {
            // Each mutation gets a deterministic seed derived from the world seed and the index.
            let mutation_seed = pop_init_seed.wrapping_add(i as u64);
            unique_configs.push(mutate_config_seeded(&seed_config, mutation_seed));
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
        kernel.reset_agents_seeded(&current_configs[0], config.world.seed);

        if let Some(ref state) = inherited_state {
            for (i, agent) in agents.iter().enumerate() {
                if i < repeats {
                    kernel.write_agent_state(agent.brain_idx, state);
                } else {
                    // Derive a seeded mutation for this agent: world seed wrapping_add the agent index.
                    // This ensures both baseline and ON arms draw identical brain-state mutations.
                    let mutation_seed = config.world.seed.wrapping_add(i as u64);
                    let mutated = mutate_brain_state_seeded(
                        state,
                        inherited_mutation_strength,
                        mutation_seed,
                    );
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

        // Adaptive dispatch interval: tick_budget / VALIDATION_HEATMAP_SAMPLES gives exactly
        // VALIDATION_HEATMAP_SAMPLES readbacks per generation, clamped to at least
        // HEATMAP_INTERVAL (100 ticks) to avoid sub-stride batches.  This balances heatmap
        // coverage (cells_explored) against GPU round-trip overhead regardless of tick_budget.
        let validation_interval = ((tick_budget / u64::from(VALIDATION_HEATMAP_SAMPLES))
            .max(u64::from(HEATMAP_INTERVAL))) as u32;

        while ticks_done < tick_budget {
            // Use the adaptive dispatch interval to keep Metal/Vulkan round-trip overhead low
            // while still sampling enough positions for meaningful heatmap coverage.  Both arms
            // see the same interval, so the relative A/B comparison is unbiased.
            let remaining = (tick_budget - ticks_done).min(validation_interval as u64) as u32;
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
            agents[i].approach_sense_range_ticks = state[base + P_APPROACH_SENSE_RANGE_TICKS];
            agents[i].approach_turns_toward = state[base + P_APPROACH_TURNS_TOWARD];
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
    // Mean deaths per agent: uncapped metric not pinned to tick_budget.
    // If agents never die, this is 0.0 — a population-viability red flag.
    let mean_death_count = if death_counts.is_empty() {
        0.0
    } else {
        death_counts.iter().sum::<f32>() / death_counts.len() as f32
    };
    let mean_food_consumed = if all_agents_fitness.is_empty() {
        0.0
    } else {
        all_agents_fitness
            .iter()
            .map(|f| f.food_consumed as f32)
            .sum::<f32>()
            / all_agents_fitness.len() as f32
    };
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

    // danger_dwell_fraction and avoidance_intent_fraction: population-aggregate
    // statistics computed through the shared reducer, identical to production.
    let mean_danger_dwell_fraction = if all_agents_fitness.is_empty() {
        0.0
    } else {
        compute_danger_dwell_fraction(&all_agents_fitness)
    };

    let mean_avoidance_intent_fraction = if all_agents_fitness.is_empty() {
        0.0
    } else {
        compute_avoidance_intent_fraction(&all_agents_fitness)
    };

    let mean_approach_intent_fraction = if all_agents_fitness.is_empty() {
        0.0
    } else {
        compute_approach_intent_fraction(&all_agents_fitness)
    };

    // Steering alignment: average of approach and avoidance intent fractions.
    // Both are bounded [0, 1]; their mean reflects overall directional steering effectiveness.
    // At chance, this is ~0.46-0.5 (the empirical baseline for vision-conditional steering).
    let mean_steering_alignment =
        (mean_approach_intent_fraction + mean_avoidance_intent_fraction) / 2.0;

    // Clean up temp database and its sidecars
    let _ = std::fs::remove_file(&temp_db);
    let _ = std::fs::remove_file(format!("{}-wal", &temp_db));
    let _ = std::fs::remove_file(format!("{}-shm", &temp_db));

    ValidationStats {
        mean_fitness,
        mean_movement_speed,
        mean_ticks_alive,
        mean_death_count,
        mean_food_consumed,
        speed_fitness_correlation,
        death_speed_regression,
        food_per_energy_vs_speed_slope,
        mean_danger_dwell_fraction,
        mean_avoidance_intent_fraction,
        mean_approach_intent_fraction,
        mean_steering_alignment,
        speed_trajectory_per_gen,
    }
}

/// Lower percentile bound for the 95% bootstrap confidence interval (2.5th percentile).
const CI_LOWER_PERCENTILE: f32 = 0.025;
/// Upper percentile bound for the 95% bootstrap confidence interval (97.5th percentile).
const CI_UPPER_PERCENTILE: f32 = 0.975;

/// Compute bootstrap metric: point estimate, 95% CI bounds, and effect size.
/// If baseline_point is None, effect is computed as zero (no comparison).
fn compute_bootstrap_metric(values: &[f32], baseline_point: Option<f32>) -> BootstrapMetric {
    if values.is_empty() {
        return BootstrapMetric {
            point: 0.0,
            lower_ci: 0.0,
            upper_ci: 0.0,
            effect: 0.0,
        };
    }

    // Point estimate: mean of replicates.
    let point = values.iter().sum::<f32>() / values.len() as f32;

    // 95% CI: 2.5th and 97.5th percentile.
    // Use floor for the lower index and ceil for the upper index, then clamp so that
    // lower_idx <= upper_idx regardless of N (necessary for N < 40 where the two
    // percentile indices would otherwise cross).
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let last = values.len() - 1;
    let lower_idx = ((CI_LOWER_PERCENTILE * last as f32).floor() as usize).min(last);
    let upper_idx = ((CI_UPPER_PERCENTILE * last as f32).ceil() as usize)
        .min(last)
        .max(lower_idx);

    let lower_ci = sorted[lower_idx];
    let upper_ci = sorted[upper_idx];

    // Effect: ON − baseline.
    let effect = baseline_point.map_or(0.0, |b| point - b);

    BootstrapMetric {
        point,
        lower_ci,
        upper_ci,
        effect,
    }
}

/// Print speed-decoupling decision rule and verdict.
/// Upper ticks-alive multiplier defining the +5% stability band above baseline.
/// ON arm may not exceed this without indicating unexpected survival inflation.
const TICKS_ALIVE_UPPER_BAND: f32 = 1.05;
/// Minimum fraction of baseline danger-dwell the ON arm must retain.
/// Agents must still enter hazard zones at ≥80% of the baseline rate.
const DANGER_DWELL_RETENTION: f32 = 0.8;

fn print_speed_decoupling_decision_rule(
    ticks_alive_baseline: &BootstrapMetric,
    ticks_alive_on: &BootstrapMetric,
    speed_correlation_baseline: &BootstrapMetric,
    speed_correlation_on: &BootstrapMetric,
    danger_dwell_baseline: &BootstrapMetric,
    danger_dwell_on: &BootstrapMetric,
) {
    // Thresholds (locked decision).
    const SPEED_CORR_EXPLOITABLE: f32 = 0.5;
    const SPEED_CORR_DECOUPLED: f32 = 0.2;
    const TICKS_ALIVE_BAND: f32 = 0.1; // ±10% of baseline.

    println!("\nThresholds:");
    println!(
        "  (a) speed_correlation BASELINE strongly positive (>= {:.1})",
        SPEED_CORR_EXPLOITABLE
    );
    println!(
        "  (b) speed_correlation ON near zero (in [-{:.1}, {:.1}])",
        SPEED_CORR_DECOUPLED, SPEED_CORR_DECOUPLED
    );
    println!(
        "  (c) ticks_alive ON within [-10%, +{:.0}%] of baseline",
        (TICKS_ALIVE_UPPER_BAND - 1.0) * 100.0
    );
    println!(
        "  (d) danger_dwell_fraction ON >= baseline * {:.1}",
        DANGER_DWELL_RETENTION
    );

    // Check gates.
    let gate_a = speed_correlation_baseline.point >= SPEED_CORR_EXPLOITABLE;
    let gate_b = speed_correlation_on.point.abs() <= SPEED_CORR_DECOUPLED;
    let lower_band = ticks_alive_baseline.point * (1.0 - TICKS_ALIVE_BAND);
    let upper_band = ticks_alive_baseline.point * TICKS_ALIVE_UPPER_BAND;
    let gate_c = ticks_alive_on.point >= lower_band && ticks_alive_on.point <= upper_band;
    let gate_d = danger_dwell_on.point >= danger_dwell_baseline.point * DANGER_DWELL_RETENTION;

    println!("\nGate Evaluation:");
    println!(
        "  (a) baseline speed_correlation {:.4} >= {:.1}? → {}",
        speed_correlation_baseline.point,
        SPEED_CORR_EXPLOITABLE,
        if gate_a { "✓" } else { "✗" }
    );
    println!(
        "  (b) ON speed_correlation {:.4} in [-{:.1}, {:.1}]? → {}",
        speed_correlation_on.point,
        SPEED_CORR_DECOUPLED,
        SPEED_CORR_DECOUPLED,
        if gate_b { "✓" } else { "✗" }
    );
    println!(
        "  (c) ON ticks_alive {:.0} in [{:.0}, {:.0}]? → {}",
        ticks_alive_on.point,
        lower_band,
        upper_band,
        if gate_c { "✓" } else { "✗" }
    );
    println!(
        "  (d) ON danger_dwell {:.4} >= {:.4}? → {}",
        danger_dwell_on.point,
        danger_dwell_baseline.point * DANGER_DWELL_RETENTION,
        if gate_d { "✓" } else { "✗" }
    );

    // FLIP: all gates pass.
    // RETIRE: ON is clearly negative on any axis (mechanism failure):
    //   - speed_correlation_on <= -SPEED_CORR_DECOUPLED: correlation inverted beyond the
    //     "near zero" band (< -0.2), meaning ON arm actively drives a negative speed-fitness
    //     link — the mechanism is working in the wrong direction.
    //   - ticks_alive_on < lower_band: survival drops >10% below baseline (harmful).
    // DEFER: thresholds do not align — gate (a) baseline not exploitable at this scale,
    //        gate (b) or gate (d) borderline, or gate (c) within band. Gate (a) failure
    //        is a scale artifact (speed-ratchet requires sufficient generations to emerge),
    //        not a mechanism failure, so it routes to DEFER rather than RETIRE.
    let verdict = if gate_a && gate_b && gate_c && gate_d {
        "FLIP: All gates pass. Effort-fitness is ready to ship."
    } else if speed_correlation_on.point < -SPEED_CORR_DECOUPLED
        || ticks_alive_on.point < lower_band
    {
        "RETIRE: ON speed_correlation is inverted beyond the near-zero band (<-0.2, mechanism working backwards) OR ticks_alive ON drops >10% below baseline."
    } else {
        "DEFER: Thresholds do not align; more data or refinement needed (see gate evaluation above)."
    };

    println!("\n>>> VERDICT: {}", verdict);
}

/// Lower bound of the "steering at chance" band. A value below this is considered regressed.
/// 0.38 mirrors the upper bound (0.62) symmetrically around 0.5 (random chance).
const STEERING_CHANCE_FLOOR: f32 = 0.38;
/// Upper bound of the chance band; also the minimum threshold for "good" steering.
/// Using a single constant avoids having two names for the same boundary (0.62).
const STEERING_GOOD_THRESHOLD: f32 = 0.62;
/// Acceptable ±band for ticks_alive stability between baseline and ON arms.
/// ±5% keeps the survival gate sensitive enough to catch real regressions while
/// tolerating the stochastic noise typical of 50-generation evolutionary runs.
const SURVIVAL_STABILITY_BAND: f32 = 0.05;

/// Print danger-percept decision rule and verdict.
/// Thresholds define flip (intent up, survival stable, steering good), retire (intent down,
/// survival down), or defer (intent/survival pass but steering at chance, unlock when credit
/// path improves).
fn print_danger_percept_decision_rule(
    avoidance_baseline: &BootstrapMetric,
    avoidance_on: &BootstrapMetric,
    approach_baseline: &BootstrapMetric,
    approach_on: &BootstrapMetric,
    ticks_alive_baseline: &BootstrapMetric,
    ticks_alive_on: &BootstrapMetric,
    steering_baseline: &BootstrapMetric,
    steering_on: &BootstrapMetric,
) {
    println!("\nThresholds:");
    println!("  (a) avoidance-intent ON > baseline (lower CI of ON > point of baseline)");
    println!(
        "  (b) survival ON within ±{:.0}% of baseline (CI overlap)",
        SURVIVAL_STABILITY_BAND * 100.0
    );
    println!(
        "  (c) steering_alignment ON: good if > {:.2}, chance if in [{:.2}, {:.2}]",
        STEERING_GOOD_THRESHOLD, STEERING_CHANCE_FLOOR, STEERING_GOOD_THRESHOLD
    );

    // Check gates.
    let lower_survival_band = ticks_alive_baseline.point * (1.0 - SURVIVAL_STABILITY_BAND);
    let upper_survival_band = ticks_alive_baseline.point * (1.0 + SURVIVAL_STABILITY_BAND);
    let gate_a = avoidance_on.lower_ci > avoidance_baseline.point;
    let gate_b =
        ticks_alive_on.point >= lower_survival_band && ticks_alive_on.point <= upper_survival_band;
    let steering_is_good = steering_on.point > STEERING_GOOD_THRESHOLD;
    let steering_is_chance =
        steering_on.point >= STEERING_CHANCE_FLOOR && steering_on.point <= STEERING_GOOD_THRESHOLD;

    println!("\nGate Evaluation:");
    println!(
        "  (a) ON avoidance_intent {:.4} (CI: [{:.4}, {:.4}]) > baseline {:.4}? → {}",
        avoidance_on.point,
        avoidance_on.lower_ci,
        avoidance_on.upper_ci,
        avoidance_baseline.point,
        if gate_a { "✓" } else { "✗" }
    );
    println!(
        "  (b) ON ticks_alive {:.0} in [{:.0}, {:.0}]? → {}",
        ticks_alive_on.point,
        lower_survival_band,
        upper_survival_band,
        if gate_b { "✓" } else { "✗" }
    );
    let steering_label = if steering_is_good {
        format!("GOOD (> {:.2})", STEERING_GOOD_THRESHOLD)
    } else if steering_is_chance {
        format!(
            "CHANCE-BAND ({:.2}-{:.2}, credit path bottleneck)",
            STEERING_CHANCE_FLOOR, STEERING_GOOD_THRESHOLD
        )
    } else {
        format!("REGRESSED (< {:.2})", STEERING_CHANCE_FLOOR)
    };
    println!(
        "  (c) ON steering_alignment {:.4} (baseline {:.4}): {}",
        steering_on.point, steering_baseline.point, steering_label
    );

    // Print auxiliary metrics for context.
    println!("\nAuxiliary Metrics:");
    println!(
        "  approach_intent ON: {:.4} (baseline {:.4}, Δ {:.4})",
        approach_on.point, approach_baseline.point, approach_on.effect
    );
    println!(
        "  avoidance_intent ON: {:.4} (baseline {:.4}, Δ {:.4})",
        avoidance_on.point, avoidance_baseline.point, avoidance_on.effect
    );

    // Decision logic:
    // FLIP: avoidance intent up AND survival stable.
    // RETIRE: avoidance intent down OR survival regressed.
    // DEFER: avoidance intent and survival pass BUT steering stays at chance,
    //        document unlock condition (steering must exceed 0.62).
    let verdict = if gate_a && gate_b && steering_is_good {
        "FLIP: Avoidance-intent up, survival stable, steering good. Danger-percept is ready to ship."
    } else if gate_a && gate_b && steering_is_chance {
        "DEFER: Avoidance-intent up, survival stable, but steering remains in the chance band [0.38, 0.62]. \
         Unlock condition: steering_alignment must exceed 0.62 on the default learning path (requires credit-path improvements) before danger-percept-enabled contributes independently. \
         The percept is wired and improves intent, but the credit path (not the percept) limits steering effectiveness."
    } else if !gate_a || !gate_b {
        "RETIRE: Avoidance-intent did not improve beyond baseline OR survival regressed. Mechanism does not show benefit at production scale."
    } else {
        "DEFER: Inconclusive. Steering regressed beyond the chance band, requiring investigation before integration."
    };

    println!("\n>>> VERDICT: {}", verdict);
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
#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_correlation_extremes() {
        // Perfect positive correlation: x = y
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = compute_correlation(&x, &y);
        assert!(
            (corr - 1.0).abs() < 1e-6,
            "perfect correlation should be ~1.0"
        );

        // Perfect negative correlation: x = -y
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
        let corr = compute_correlation(&x, &y);
        assert!(
            (corr - (-1.0)).abs() < 1e-6,
            "perfect negative correlation should be ~-1.0"
        );

        // Zero correlation: constant x
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = compute_correlation(&x, &y);
        assert!(
            corr.abs() < 1e-6,
            "constant x should give ~0 correlation (not NaN)"
        );

        // Insufficient data
        let x = vec![1.0];
        let y = vec![1.0];
        let corr = compute_correlation(&x, &y);
        assert!(corr == 0.0, "insufficient data should return 0.0");
    }

    #[test]
    fn compute_regression_slope() {
        // Known slope: y = 2*x
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let slope = compute_regression(&x, &y);
        assert!((slope - 2.0).abs() < 1e-6, "slope of y=2x should be 2.0");

        // Known slope: y = -0.5*x
        let x = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let y = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
        let slope = compute_regression(&x, &y);
        assert!(
            (slope - (-0.5)).abs() < 1e-6,
            "slope of y=-0.5x should be -0.5"
        );

        // Insufficient data
        let x = vec![1.0];
        let y = vec![1.0];
        let slope = compute_regression(&x, &y);
        assert!(slope == 0.0, "insufficient data should return 0.0");
    }

    // ── innate-instinct validation gate predicates ──────────────────────

    #[test]
    fn innate_gate_survival_requires_10_percent_improvement() {
        // Test the survival gate predicate: ON ticks must exceed baseline by 10%.
        let baseline_ticks = 100000u64;
        let on_ticks_below = 109999u64; // Just below 110% threshold
        let on_ticks_at = 110000u64; // Exactly at 110%
        let on_ticks_above = 110001u64; // Above 110%

        let threshold = (baseline_ticks as f32 * (1.0 + INSTINCT_SURVIVAL_MARGIN)) as u64;

        // Below threshold should fail
        assert!(
            on_ticks_below < threshold,
            "test setup: on_ticks_below should be below threshold"
        );

        // At or above threshold should pass
        assert!(
            on_ticks_at >= threshold,
            "test setup: on_ticks_at should be at threshold"
        );
        assert!(
            on_ticks_above >= threshold,
            "test setup: on_ticks_above should be above threshold"
        );
    }

    #[test]
    fn innate_gate_alignment_requires_0_4_avoidance_intent() {
        // Test alignment gate predicate: avoidance-intent fraction must be >= 0.4.
        let below_floor = 0.39999_f32;
        let at_floor = 0.4_f32;
        let above_floor = 0.40001_f32;

        // Below floor should fail gate
        assert!(
            below_floor < INSTINCT_ALIGNMENT_FLOOR,
            "below_floor should fail alignment gate"
        );

        // At or above floor should pass gate
        assert!(
            at_floor >= INSTINCT_ALIGNMENT_FLOOR,
            "at_floor should pass alignment gate"
        );
        assert!(
            above_floor >= INSTINCT_ALIGNMENT_FLOOR,
            "above_floor should pass alignment gate"
        );
    }

    #[test]
    fn innate_gate_food_per_death_requires_2_0_ratio() {
        // Test food-per-death gate predicate: ratio must be >= 2.0.
        let mean_food = 10.0_f32;
        let death_count_below = 5.00001_f32; // ratio = 10 / 5.00001 ≈ 1.9999 < 2.0
        let death_count_at = 5.0_f32; // ratio = 10 / 5.0 = 2.0
        let death_count_above = 4.99999_f32; // ratio = 10 / 4.99999 ≈ 2.00001 > 2.0

        let ratio_below = mean_food / death_count_below;
        let ratio_at = mean_food / death_count_at;
        let ratio_above = mean_food / death_count_above;

        // Below threshold should fail gate
        assert!(
            ratio_below < INSTINCT_FOOD_PER_DEATH_MIN,
            "ratio_below should fail food-per-death gate"
        );

        // At or above threshold should pass gate
        assert!(
            ratio_at >= INSTINCT_FOOD_PER_DEATH_MIN,
            "ratio_at should pass food-per-death gate"
        );
        assert!(
            ratio_above >= INSTINCT_FOOD_PER_DEATH_MIN,
            "ratio_above should pass food-per-death gate"
        );
    }

    #[test]
    fn innate_gate_food_per_death_infinity_with_zero_deaths() {
        // Edge case: zero deaths yields infinity, which always passes food-per-death gate.
        let mean_food = 10.0_f32;
        let zero_deaths = 0.0_f32;

        let food_per_death = if zero_deaths > FOOD_PER_DEATH_ZERO_GUARD {
            mean_food / zero_deaths
        } else {
            f32::INFINITY
        };

        // Infinity should pass the gate
        assert!(
            food_per_death >= INSTINCT_FOOD_PER_DEATH_MIN,
            "food_per_death=infinity should pass the threshold"
        );
    }

    #[test]
    fn innate_gate_all_three_predicates_required() {
        // Verify that all three gates are conjunctive (all must be true).
        // Test a scenario where two pass but one fails.

        // Survival gate passes (ON = baseline + 15%)
        let baseline_ticks = 100000u64;
        let on_ticks = 115000u64;

        // Alignment gate passes (ON avoidance = 0.5 > 0.4)
        let on_avoidance = 0.5_f32;

        // Food-per-death gate FAILS (ratio = 1.5 < 2.0)
        let on_food = 7.5_f32;
        let on_death = 5.0_f32;
        let on_food_per_death = on_food / on_death;

        // Individual gate evaluations
        let survival_gate =
            on_ticks >= ((baseline_ticks as f32) * (1.0 + INSTINCT_SURVIVAL_MARGIN)) as u64;
        let alignment_gate = on_avoidance >= INSTINCT_ALIGNMENT_FLOOR;
        let food_per_death_gate = on_food_per_death >= INSTINCT_FOOD_PER_DEATH_MIN;

        // Verify individual gate states
        assert!(survival_gate, "survival gate should pass");
        assert!(alignment_gate, "alignment gate should pass");
        assert!(!food_per_death_gate, "food-per-death gate should fail");

        // Combined gate must fail
        let combined_gate = survival_gate && alignment_gate && food_per_death_gate;
        assert!(
            !combined_gate,
            "combined gate must fail when any individual gate fails"
        );
    }
}
