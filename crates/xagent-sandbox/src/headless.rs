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
    P_APPROACH_SENSE_RANGE_TICKS, P_APPROACH_TURNS_TOWARD, P_AVOIDANCE_SENSE_RANGE_TICKS,
    P_AVOIDANCE_TURNS_OPPOSING, P_DANGER_PATH_LENGTH, P_DEATH_COUNT, P_DISTANCE_TRAVELED,
    P_ENERGY_SPENT, P_FOOD_COUNT, P_POS_X, P_POS_Y, P_POS_Z, P_TICKS_ALIVE,
};
use xagent_brain::{AgentBrainState, GpuKernel};

use crate::agent::{
    mutate_brain_state, mutate_brain_state_seeded, mutate_config, mutate_config_seeded, Agent,
};
use crate::governor::{
    compute_avoidance_intent_fraction, compute_danger_dwell_fraction, AdvanceResult, Governor,
};
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

/// Run speed-decoupling validation: an A/B test with effort-rebased fitness,
/// super-linear locomotor drag, and the danger percept OFF (baseline) vs ON,
/// measuring the speed↔fitness correlation and supporting metrics.
///
/// The populated result is written to `speed-decoupling-validation.md` in the
/// process working directory.
pub fn validate_speed_decoupling(config: FullConfig, num_generations: u64) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SPEED-DECOUPLING VALIDATION");
    println!(
        "Running {} generations with the effort/drag/danger flags OFF (baseline) then ON",
        num_generations
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Baseline run (all flags off)
    println!("\n[BASELINE] effort-rebased fitness / super-linear drag / danger percept OFF");
    let baseline_stats =
        run_headless_with_flags(config.clone(), num_generations, false, false, false);

    // On run: effort-rebased fitness, super-linear drag at k=2.0, and danger percept all on
    println!("\n[ON] effort-rebased fitness / super-linear drag / danger percept ON");
    let on_stats = run_headless_with_flags(config.clone(), num_generations, true, true, false);

    // Compute and report metrics
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    print_validation_metrics(&baseline_stats, &on_stats);

    let markdown = format_validation_markdown(
        num_generations,
        config.world.seed,
        config.governor.population_size,
        &baseline_stats,
        &on_stats,
    );

    // Save the report in the process working directory.
    let report_path = "speed-decoupling-validation.md";
    match std::fs::write(report_path, markdown) {
        Ok(()) => println!("\nResults saved to ./{}", report_path),
        Err(e) => eprintln!("Failed to write {}: {}", report_path, e),
    }
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

/// Gate: baseline must have an exploitable speed-fitness correlation.
/// The gate only certifies decoupling when the baseline actually had a speed
/// exploit to remove (strongly-positive correlation). A baseline already below
/// threshold is reported "inconclusive", never PASS.
const BASELINE_CORR_MIN: f32 = 0.3;

/// Gate: ON arm's speed-fitness correlation must fall below this threshold.
const DECOUPLE_CORR_MAX: f32 = 0.3;

/// Gate: ON arm must strictly improve below baseline by this margin.
/// Prevents claiming success when ON is only marginally better.
const DECOUPLE_MARGIN: f32 = 0.02;

/// Gate: minimum mean avoidance intent fraction (turns opposing danger bearing
/// per sense-range tick). Below this, the population is not engaging with danger
/// decisions meaningfully. Tuned at 0.05 to require some non-zero avoidance
/// measurement while tolerating natural variance.
const AVOIDANCE_FLOOR: f32 = 0.05;

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
fn format_validation_markdown(
    num_generations: u64,
    world_seed: u64,
    population_size: usize,
    baseline: &ValidationStats,
    on_stats: &ValidationStats,
) -> String {
    // Gate logic: require strongly-positive baseline AND strict improvement AND uncapped viability.
    let baseline_corr_exploitable = baseline.speed_fitness_correlation.abs() >= BASELINE_CORR_MIN;
    let on_decoupled = on_stats.speed_fitness_correlation.abs() < DECOUPLE_CORR_MAX;
    let strict_improvement = on_stats.speed_fitness_correlation.abs()
        <= baseline.speed_fitness_correlation.abs() - DECOUPLE_MARGIN;
    let speed_decoupled = baseline_corr_exploitable && on_decoupled && strict_improvement;

    // Viability: uncapped metric — mean death-count per agent.
    // Unlike mean_ticks_alive, this cannot be pinned to tick_budget:
    // a value of 0.0 means no agent ever died, indicating the population is not
    // actually cycling through life/death/respawn as expected.
    let viability_ok = on_stats.mean_death_count > 0.0;

    let danger_retained = on_stats.mean_danger_dwell_fraction > 0.01;
    let avoidance_above_chance = on_stats.mean_avoidance_intent_fraction >= AVOIDANCE_FLOOR;
    let gate_passed = speed_decoupled && viability_ok && danger_retained && avoidance_above_chance;

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
        "# Speed-Decoupling Validation Report\n\
\n\
**Date:** 2026-06-18\n\
**Status:** MEASURED\n\
**Generations:** {num_gens}\n\
**World Seed:** {seed}\n\
**Population Size:** {pop_size}\n\
\n\
## Configuration\n\
\n\
- Baseline: effort/drag/danger flags OFF (`speed_cost_exponent=1.0`, `effort_rebased_fitness=false`, `danger_percept_enabled=false`)\n\
- On: super-linear drag `speed_cost_exponent={exp:.1}`, `effort_rebased_fitness=true`, `danger_percept_enabled=true`\n\
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
Path-length hazard makes per-crossing damage speed-invariant, so fast agents \
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
1. Baseline correlation is strongly-positive (|r| >= {baseline_min:.2}): {gate_baseline}\n\
2. Speed-fitness correlation {corr_direction} from strongly-positive to below threshold (|r| < {decouple_max:.2}): {gate_decoupling}\n\
3. ON strictly improves below baseline by margin (>= {margin:.3}): {gate_margin}\n\
4. Population viability maintained (mean death-count > 0): {gate_viability}\n\
5. Danger metrics stay non-zero (danger_dwell_fraction > 0.01): {gate_danger}\n\
6. Avoidance intent above floor (>= {avoidance_floor:.3}): {gate_avoidance}\n\
\n\
**Overall Result:** {gate_overall}\n\
\n\
## Decision\n\
\n\
{decision}\n\
\n\
---\n\
\n\
This document was generated by the speed-decoupling validation harness \
(`cargo run --release -- --validate-speed-decoupling`). \
It records the baseline for future speed-decoupling A/B tests.\n",
        num_gens = num_generations,
        seed = world_seed,
        pop_size = population_size,
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
        ticks_summary = if viability_ok {
            "Population viability is maintained: mean death-count per generation > 0."
        } else {
            "Population viability concern: no deaths per generation recorded. \
             Consider reducing the drag exponent or reviewing energy constants."
        },
        baseline_min = BASELINE_CORR_MIN,
        decouple_max = DECOUPLE_CORR_MAX,
        margin = DECOUPLE_MARGIN,
        avoidance_floor = AVOIDANCE_FLOOR,
        gate_baseline = if baseline_corr_exploitable { "PASS" } else { "FAIL" },
        gate_decoupling = if on_decoupled { "PASS" } else { "FAIL" },
        gate_margin = if strict_improvement { "PASS" } else { "FAIL" },
        gate_viability = if viability_ok { "PASS" } else { "FAIL" },
        gate_danger = if danger_retained { "PASS" } else { "FAIL" },
        gate_avoidance = if avoidance_above_chance { "PASS" } else { "FAIL" },
        gate_overall = if !baseline_corr_exploitable {
            "INCONCLUSIVE — Baseline correlation not strongly-positive; cannot assess improvement."
        } else if gate_passed {
            "GATE PASSED — All criteria met; speed is successfully decoupled from fitness."
        } else {
            "GATE FAILED — Some criteria not met; review metrics above before flipping defaults."
        },
        decision = if !baseline_corr_exploitable {
            format!(
                "The baseline correlation is not strongly-positive (below |{:.2}| threshold). \
The gate cannot assess decoupling improvement without a clear baseline exploit. \
Baseline variants may need tuning or the validation may need retrying with different configurations.",
                BASELINE_CORR_MIN
            )
        } else if gate_passed {
            "The validation passed. The four mechanisms (super-linear drag at k=2.0, \
path-length hazard, effort-rebased fitness, and the danger percept) \
successfully decouple movement speed from composite fitness. \
The population remained viable and danger-decision data is retained. \
The defaults are candidates for flipping now that this gate has passed."
                .to_string()
        } else {
            "The validation did not meet all gate criteria. Review the measured metrics and \
consider tuning the drag exponent (speed_cost_exponent), the fitness calibration constants \
(FORAGING_ENERGY_TARGET / EXPLORATION_RATE_TARGET), or the danger sense radius before \
attempting another run. Do not flip the defaults until the gate passes."
                .to_string()
        },
    )
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

    #[test]
    fn gate_rejects_rising_correlation() {
        // The original real-world failure: baseline corr = 0.31 (exploitable, >=0.3)
        // and ON corr = 0.295 (below DECOUPLE_CORR_MAX so on_decoupled is true), but
        // the drop of 0.015 is less than DECOUPLE_MARGIN = 0.02, so strict_improvement
        // is false. This mirrors the 0.2238 → 0.2569 bug where the gate printed PASS
        // despite a rising correlation. The baseline_corr_exploitable conjunct is true
        // here so this test independently falsifies strict_improvement.
        let baseline = ValidationStats {
            mean_fitness: 0.5,
            mean_movement_speed: 50.0,
            mean_ticks_alive: 900000,
            mean_death_count: 5.0,
            mean_food_consumed: 10.0,
            speed_fitness_correlation: 0.31,
            death_speed_regression: 0.0,
            food_per_energy_vs_speed_slope: 0.0,
            mean_danger_dwell_fraction: 0.05,
            mean_avoidance_intent_fraction: 0.1,
            speed_trajectory_per_gen: vec![50.0],
        };
        let on = ValidationStats {
            mean_fitness: 0.5,
            mean_movement_speed: 45.0,
            mean_ticks_alive: 900000,
            mean_death_count: 5.0,
            mean_food_consumed: 10.0,
            speed_fitness_correlation: 0.295, // dropped only 0.015, below DECOUPLE_MARGIN=0.02
            death_speed_regression: 0.0,
            food_per_energy_vs_speed_slope: 0.0,
            mean_danger_dwell_fraction: 0.05,
            mean_avoidance_intent_fraction: 0.1,
            speed_trajectory_per_gen: vec![45.0],
        };

        let baseline_corr_exploitable =
            baseline.speed_fitness_correlation.abs() >= BASELINE_CORR_MIN;
        let on_decoupled = on.speed_fitness_correlation.abs() < DECOUPLE_CORR_MAX;
        let strict_improvement = on.speed_fitness_correlation.abs()
            <= baseline.speed_fitness_correlation.abs() - DECOUPLE_MARGIN;
        let speed_decoupled = baseline_corr_exploitable && on_decoupled && strict_improvement;

        assert!(
            baseline_corr_exploitable,
            "baseline must be exploitable so this test isolates strict_improvement"
        );
        assert!(
            on_decoupled,
            "on_decoupled must be true so this test isolates strict_improvement"
        );
        assert!(
            !strict_improvement,
            "strict_improvement must be false: ON dropped less than DECOUPLE_MARGIN"
        );
        assert!(
            !speed_decoupled,
            "gate should reject when ON fails strict improvement margin"
        );
    }

    #[test]
    fn gate_rejects_weak_baseline() {
        // Baseline corr = 0.25 (below BASELINE_CORR_MIN = 0.3)
        let baseline = ValidationStats {
            mean_fitness: 0.5,
            mean_movement_speed: 50.0,
            mean_ticks_alive: 900000,
            mean_death_count: 5.0,
            mean_food_consumed: 10.0,
            speed_fitness_correlation: 0.25,
            death_speed_regression: 0.0,
            food_per_energy_vs_speed_slope: 0.0,
            mean_danger_dwell_fraction: 0.05,
            mean_avoidance_intent_fraction: 0.1,
            speed_trajectory_per_gen: vec![50.0],
        };

        let baseline_corr_exploitable =
            baseline.speed_fitness_correlation.abs() >= BASELINE_CORR_MIN;
        assert!(
            !baseline_corr_exploitable,
            "gate should reject weak baseline"
        );
    }

    #[test]
    fn gate_rejects_tick_collapse() {
        // Baseline has good corr, ON reduces it strictly enough, but the ON arm
        // shows zero deaths (mean_death_count = 0.0): viability failure.
        // An ON population that never dies is budget-saturating — not genuinely
        // cycling through respawn — so the gate must reject it.
        let baseline = ValidationStats {
            mean_fitness: 0.5,
            mean_movement_speed: 50.0,
            mean_ticks_alive: 900000,
            mean_death_count: 5.0,
            mean_food_consumed: 10.0,
            speed_fitness_correlation: 0.5,
            death_speed_regression: 0.0,
            food_per_energy_vs_speed_slope: 0.0,
            mean_danger_dwell_fraction: 0.05,
            mean_avoidance_intent_fraction: 0.1,
            speed_trajectory_per_gen: vec![50.0],
        };
        let on = ValidationStats {
            mean_fitness: 0.5,
            mean_movement_speed: 45.0,
            mean_ticks_alive: 900000,
            mean_death_count: 0.0, // Zero deaths: viability collapse
            mean_food_consumed: 10.0,
            speed_fitness_correlation: 0.15,
            death_speed_regression: 0.0,
            food_per_energy_vs_speed_slope: 0.0,
            mean_danger_dwell_fraction: 0.05,
            mean_avoidance_intent_fraction: 0.1,
            speed_trajectory_per_gen: vec![45.0],
        };

        let baseline_corr_exploitable =
            baseline.speed_fitness_correlation.abs() >= BASELINE_CORR_MIN;
        let on_decoupled = on.speed_fitness_correlation.abs() < DECOUPLE_CORR_MAX;
        let strict_improvement = on.speed_fitness_correlation.abs()
            <= baseline.speed_fitness_correlation.abs() - DECOUPLE_MARGIN;
        let speed_decoupled = baseline_corr_exploitable && on_decoupled && strict_improvement;

        // Viability check: zero deaths must cause the gate to fail.
        let viability_ok = on.mean_death_count > 0.0;
        let danger_retained = on.mean_danger_dwell_fraction > 0.01;
        let avoidance_above_chance = on.mean_avoidance_intent_fraction >= AVOIDANCE_FLOOR;
        let gate_passed =
            speed_decoupled && viability_ok && danger_retained && avoidance_above_chance;

        assert!(
            !gate_passed,
            "gate should fail when mean_death_count is zero (viability collapse)"
        );
        assert!(
            !viability_ok,
            "viability_ok must be false when mean_death_count is 0.0"
        );
    }

    #[test]
    fn gate_rejects_danger_zero() {
        // ON has low danger_dwell_fraction
        let on = ValidationStats {
            mean_fitness: 0.5,
            mean_movement_speed: 45.0,
            mean_ticks_alive: 900000,
            mean_death_count: 5.0,
            mean_food_consumed: 10.0,
            speed_fitness_correlation: 0.15,
            death_speed_regression: 0.0,
            food_per_energy_vs_speed_slope: 0.0,
            mean_danger_dwell_fraction: 0.005, // Below 0.01 threshold
            mean_avoidance_intent_fraction: 0.1,
            speed_trajectory_per_gen: vec![45.0],
        };

        let danger_retained = on.mean_danger_dwell_fraction > 0.01;
        assert!(
            !danger_retained,
            "gate should reject when danger_dwell_fraction drops below 0.01"
        );
    }

    #[test]
    fn gate_rejects_avoidance_below_floor() {
        // All other conjuncts pass, but ON avoidance is below AVOIDANCE_FLOOR.
        // This independently falsifies the avoidance_above_chance conjunct: removing
        // it from gate_passed would flip the result to true, proving the conjunct
        // is load-bearing and not redundant with another check.
        let baseline = ValidationStats {
            mean_fitness: 0.5,
            mean_movement_speed: 50.0,
            mean_ticks_alive: 900000,
            mean_death_count: 5.0,
            mean_food_consumed: 10.0,
            speed_fitness_correlation: 0.5, // exploitable baseline
            death_speed_regression: 0.0,
            food_per_energy_vs_speed_slope: 0.0,
            mean_danger_dwell_fraction: 0.05,
            mean_avoidance_intent_fraction: 0.1,
            speed_trajectory_per_gen: vec![50.0],
        };
        let on = ValidationStats {
            mean_fitness: 0.5,
            mean_movement_speed: 45.0,
            mean_ticks_alive: 900000,
            mean_death_count: 3.0, // deaths > 0 so viability_ok is true
            mean_food_consumed: 10.0,
            speed_fitness_correlation: 0.15, // < DECOUPLE_CORR_MAX and 0.5-0.15=0.35 >= DECOUPLE_MARGIN
            death_speed_regression: 0.0,
            food_per_energy_vs_speed_slope: 0.0,
            mean_danger_dwell_fraction: 0.05, // > 0.01 so danger_retained is true
            mean_avoidance_intent_fraction: 0.02, // below AVOIDANCE_FLOOR = 0.05
            speed_trajectory_per_gen: vec![45.0],
        };

        let baseline_corr_exploitable =
            baseline.speed_fitness_correlation.abs() >= BASELINE_CORR_MIN;
        let on_decoupled = on.speed_fitness_correlation.abs() < DECOUPLE_CORR_MAX;
        let strict_improvement = on.speed_fitness_correlation.abs()
            <= baseline.speed_fitness_correlation.abs() - DECOUPLE_MARGIN;
        let speed_decoupled = baseline_corr_exploitable && on_decoupled && strict_improvement;
        let viability_ok = on.mean_death_count > 0.0;
        let danger_retained = on.mean_danger_dwell_fraction > 0.01;
        let avoidance_above_chance = on.mean_avoidance_intent_fraction >= AVOIDANCE_FLOOR;
        let gate_passed =
            speed_decoupled && viability_ok && danger_retained && avoidance_above_chance;

        // Verify all other conjuncts are true so only avoidance drives the result.
        assert!(
            speed_decoupled,
            "speed_decoupled must be true to isolate avoidance conjunct"
        );
        assert!(
            viability_ok,
            "viability_ok must be true to isolate avoidance conjunct"
        );
        assert!(
            danger_retained,
            "danger_retained must be true to isolate avoidance conjunct"
        );
        assert!(
            !avoidance_above_chance,
            "avoidance_above_chance must be false: ON avoidance below AVOIDANCE_FLOOR"
        );
        assert!(
            !gate_passed,
            "gate should reject when avoidance intent fraction is below AVOIDANCE_FLOOR"
        );
    }

    #[test]
    fn gate_passes_when_all_criteria_met() {
        // All criteria satisfied: baseline exploitable, ON decouples, improves strictly,
        // viability ok (deaths > 0), danger retained, avoidance above floor
        let baseline = ValidationStats {
            mean_fitness: 0.5,
            mean_movement_speed: 50.0,
            mean_ticks_alive: 900000,
            mean_death_count: 5.0,
            mean_food_consumed: 10.0,
            speed_fitness_correlation: 0.5,
            death_speed_regression: 0.0,
            food_per_energy_vs_speed_slope: 0.0,
            mean_danger_dwell_fraction: 0.05,
            mean_avoidance_intent_fraction: 0.12,
            speed_trajectory_per_gen: vec![50.0],
        };
        let on = ValidationStats {
            mean_fitness: 0.5,
            mean_movement_speed: 45.0,
            mean_ticks_alive: 900000,
            mean_death_count: 3.0,
            mean_food_consumed: 10.0,
            speed_fitness_correlation: 0.25, // 0.5 - 0.25 = 0.25 >= DECOUPLE_MARGIN
            death_speed_regression: 0.0,
            food_per_energy_vs_speed_slope: 0.0,
            mean_danger_dwell_fraction: 0.05,
            mean_avoidance_intent_fraction: 0.12,
            speed_trajectory_per_gen: vec![45.0],
        };

        let baseline_corr_exploitable =
            baseline.speed_fitness_correlation.abs() >= BASELINE_CORR_MIN;
        let on_decoupled = on.speed_fitness_correlation.abs() < DECOUPLE_CORR_MAX;
        let strict_improvement = on.speed_fitness_correlation.abs()
            <= baseline.speed_fitness_correlation.abs() - DECOUPLE_MARGIN;
        let speed_decoupled = baseline_corr_exploitable && on_decoupled && strict_improvement;
        let viability_ok = on.mean_death_count > 0.0;
        let danger_retained = on.mean_danger_dwell_fraction > 0.01;
        let avoidance_above_chance = on.mean_avoidance_intent_fraction >= AVOIDANCE_FLOOR;
        let gate_passed =
            speed_decoupled && viability_ok && danger_retained && avoidance_above_chance;

        assert!(gate_passed, "gate should pass when all criteria are met");
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
