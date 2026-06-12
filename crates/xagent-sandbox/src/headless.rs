//! Headless (no-window) evolution loop and tree dump utilities.
//!
//! Uses GpuKernel for all simulation — physics, brain, food, and
//! death/respawn run entirely on GPU via fused kernel dispatch.

use std::time::Instant;

use log::info;
use xagent_shared::{BrainConfig, FullConfig};

use xagent_brain::buffers::{
    BrainLayout, ENCODED_DIMENSION, O_ACTION_FORWARD_WEIGHTS, O_ACTION_TURN_WEIGHTS,
    O_PREDICTOR_CONTEXT_WEIGHT, PHYS_STRIDE, PREDICTOR_DIMENSION, P_ALIVE, P_DEATH_COUNT,
    P_FOOD_COUNT, P_POS_X, P_POS_Y, P_POS_Z, P_TICKS_ALIVE,
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

        let quarter_length = (tick_budget / 4).max(1);
        let mut quarter_samples: [(u64, u64); 4] = [(0, 0); 4];
        let mut next_quarter: usize = 0;

        while ticks_done < tick_budget {
            let remaining = (tick_budget - ticks_done).min(HEATMAP_INTERVAL as u64) as u32;
            kernel.dispatch_batch(ticks_done, remaining);
            ticks_done += remaining as u64;

            // Advance governor tick counter
            for _ in 0..remaining {
                governor.tick();
            }

            // Drain async readback, then sample cached state for heatmap
            while !kernel.try_collect_state() {
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

            while next_quarter < 4 && ticks_done >= quarter_length * (next_quarter as u64 + 1) {
                let mut cumulative_food = 0_u64;
                let mut cumulative_alive = 0_u64;
                for i in 0..agents.len() {
                    let base = i * PHYS_STRIDE;
                    cumulative_food += state[base + P_FOOD_COUNT] as u64;
                    cumulative_alive += state[base + P_TICKS_ALIVE] as u64;
                }
                quarter_samples[next_quarter] = (cumulative_food, cumulative_alive);
                next_quarter += 1;
            }

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
        for i in 0..agents.len() {
            let base = i * PHYS_STRIDE;
            agents[i].food_consumed = state[base + P_FOOD_COUNT] as u32;
            agents[i].total_ticks_alive = state[base + P_TICKS_ALIVE] as u64;
            agents[i].death_count = state[base + P_DEATH_COUNT] as u32;
        }

        while next_quarter < 4 {
            let mut cumulative_food = 0_u64;
            let mut cumulative_alive = 0_u64;
            for i in 0..agents.len() {
                let base = i * PHYS_STRIDE;
                cumulative_food += state[base + P_FOOD_COUNT] as u64;
                cumulative_alive += state[base + P_TICKS_ALIVE] as u64;
            }
            quarter_samples[next_quarter] = (cumulative_food, cumulative_alive);
            next_quarter += 1;
        }
        let (first_quarter_rate, last_quarter_rate) = quarter_rates(&quarter_samples);

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

/// Food rates (per 1k alive-ticks) of the first and last generation
/// quarters, from cumulative (food, alive_ticks) samples taken at the
/// four quarter boundaries. The last-quarter rate uses the deltas
/// between the third and fourth samples. Rising last-over-first is the
/// direct signal that the population improves within a lifetime instead
/// of only across generations.
fn quarter_rates(samples: &[(u64, u64); 4]) -> (f64, f64) {
    let (first_food, first_alive) = samples[0];
    let first_quarter_rate = if first_alive > 0 {
        first_food as f64 / first_alive as f64 * 1000.0
    } else {
        0.0
    };
    let last_food = samples[3].0.saturating_sub(samples[2].0);
    let last_alive = samples[3].1.saturating_sub(samples[2].1);
    let last_quarter_rate = if last_alive > 0 {
        last_food as f64 / last_alive as f64 * 1000.0
    } else {
        0.0
    };
    (first_quarter_rate, last_quarter_rate)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_rates_computes_first_and_last_quarter_food_rates() {
        // Cumulative (food, alive_ticks) samples at the four quarter
        // boundaries: q1 ate 4 in 1000 alive-ticks (rate 4.0/1k); the
        // last quarter ate 12−8 = 4 in 4000−3200 = 800 alive-ticks
        // (rate 5.0/1k).
        let samples = [(4_u64, 1000_u64), (6, 2100), (8, 3200), (12, 4000)];
        let (first_quarter_rate, last_quarter_rate) = quarter_rates(&samples);
        assert!((first_quarter_rate - 4.0).abs() < 1e-9);
        assert!((last_quarter_rate - 5.0).abs() < 1e-9);
    }

    #[test]
    fn quarter_rates_handles_zero_alive_ticks() {
        let samples = [(0_u64, 0_u64), (0, 0), (0, 0), (0, 0)];
        let (first_quarter_rate, last_quarter_rate) = quarter_rates(&samples);
        assert_eq!(first_quarter_rate, 0.0);
        assert_eq!(last_quarter_rate, 0.0);
    }
}
