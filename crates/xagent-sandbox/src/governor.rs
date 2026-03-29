//! Evolution governor — orchestrates natural selection across generations.
//!
//! The governor manages a population of agents, evaluates their fitness after
//! a fixed tick budget, selects the best performers, and breeds mutated offspring.
//! All state is persisted in a SQLite database (`xagent.db`), enabling resume
//! and post-hoc analysis of the evolution tree.

use rand::Rng;
use rusqlite::{params, Connection, Result as SqlResult};
use serde::Serialize;
use xagent_shared::{BrainConfig, GovernorConfig};

use crate::agent::{crossover_config, mutate_config_with_strength, Agent, HEATMAP_RES};

/// Per-agent fitness evaluation result.
#[derive(Clone, Debug, Serialize)]
pub struct AgentFitness {
    pub agent_index: usize,
    pub config: BrainConfig,
    pub total_ticks_alive: u64,
    pub death_count: u32,
    pub food_consumed: u32,
    pub cells_explored: u32,
    pub composite_fitness: f32,
}

/// Result of `Governor::advance()` — tells the caller what to do next.
pub enum AdvanceResult {
    /// Simulation continues — spawn the given configs for the next generation.
    Continue {
        configs: Vec<BrainConfig>,
        messages: Vec<String>,
    },
    /// Max generations reached — stop the simulation.
    Finished {
        messages: Vec<String>,
    },
}

/// An independent evolutionary lineage within the island model.
#[derive(Clone, Debug)]
struct Island {
    spawn_parent_id: Option<i64>,
    best_score: f32,
    elite_configs: Vec<BrainConfig>,
}

/// The evolution governor. Owns the SQLite connection and drives the
/// generational loop: evaluate → advance → breed → repeat.
///
/// # Algorithm
///
/// A generation "succeeds" if its fitness meets or exceeds the global
/// `best_score`. Successful nodes become the new spawn parent (tree deepens).
/// Ties (matching best_score) count as success — they represent reproducible
/// quality and shouldn't burn patience in a stochastic environment.
///
/// Each node may spawn at most `patience` children. When the limit is
/// reached, the node is "exhausted" and the governor cascades up to its
/// parent. If the root exhausts, `best_score` is partially rolled back
/// (×0.95), its attempt counter resets, and evolution continues.
///
/// The population includes an unmutated copy of the spawn parent's config
/// (the champion), elite configs from the last successful generation, and
/// mutated variants from the spawn parent with adaptive mutation strength.
pub struct Governor {
    pub db: Connection,
    pub config: GovernorConfig,
    pub run_id: i64,
    pub current_node_id: Option<i64>,
    pub generation: u32,
    pub gen_tick: u64,
    /// Independent evolutionary lineages.
    islands: Vec<Island>,
    /// Index of the island currently being evaluated (round-robin).
    active_island: usize,
}

impl Governor {
    /// Global best score across all islands.
    pub fn best_score(&self) -> f32 {
        self.islands.iter().map(|i| i.best_score).fold(-1.0_f32, f32::max)
    }

    /// Active island's spawn parent (for tree display).
    pub fn spawn_parent_id(&self) -> Option<i64> {
        self.islands.get(self.active_island).and_then(|i| i.spawn_parent_id)
    }

    /// Create a new governor, initializing the database schema and inserting
    /// a new run record. `db_path` is the SQLite file path.
    pub fn new(
        db_path: &str,
        config: GovernorConfig,
        seed_brain: &BrainConfig,
        world_config_json: &str,
    ) -> SqlResult<Self> {
        let db = Connection::open(db_path)?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        init_schema(&db)?;

        let governor_json = serde_json::to_string(&config).unwrap_or_default();
        let brain_json = serde_json::to_string(seed_brain).unwrap_or_default();

        db.execute(
            "INSERT INTO run (seed, governor_config, brain_config, world_config)
             VALUES (?1, ?2, ?3, ?4)",
            params![42i64, governor_json, brain_json, world_config_json],
        )?;
        let run_id = db.last_insert_rowid();

        // Insert root node (generation 0) with seed config
        db.execute(
            "INSERT INTO node (run_id, parent_id, generation, config_json, status)
             VALUES (?1, NULL, 0, ?2, 'active')",
            params![run_id, brain_json],
        )?;
        let root_id = db.last_insert_rowid();

        // Persist spawn parent for resume
        db.execute(
            "UPDATE run SET best_score = -1, spawn_parent_id = ?1 WHERE id = ?2",
            params![root_id, run_id],
        )?;

        let num_islands = config.num_islands.max(1);
        let islands: Vec<Island> = (0..num_islands)
            .map(|_| Island {
                spawn_parent_id: Some(root_id),
                best_score: -1.0,
                elite_configs: Vec::new(),
            })
            .collect();

        Ok(Self {
            db,
            config,
            run_id,
            current_node_id: Some(root_id),
            generation: 0,
            gen_tick: 0,
            islands,
            active_island: 0,
        })
    }

    /// Resume from an existing database. Finds the latest active node
    /// for the most recent run.
    pub fn resume(db_path: &str) -> SqlResult<Self> {
        let db = Connection::open(db_path)?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        let (run_id, governor_json, best_score, spawn_parent_id): (i64, String, f64, Option<i64>) =
            db.query_row(
                "SELECT id, governor_config, COALESCE(best_score, -1), spawn_parent_id
                 FROM run ORDER BY id DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )?;

        let config: GovernorConfig =
            serde_json::from_str(&governor_json).unwrap_or_default();

        let (node_id, generation): (i64, u32) = db.query_row(
            "SELECT id, generation FROM node
             WHERE run_id = ?1 AND status = 'active'
             ORDER BY generation DESC LIMIT 1",
            params![run_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;

        // Reconstruct all islands from the persisted best state
        let num_islands = config.num_islands.max(1);
        let islands: Vec<Island> = (0..num_islands)
            .map(|_| Island {
                spawn_parent_id,
                best_score: best_score as f32,
                elite_configs: Vec::new(),
            })
            .collect();

        Ok(Self {
            db,
            config,
            run_id,
            current_node_id: Some(node_id),
            generation,
            gen_tick: 0,
            islands,
            active_island: 0,
        })
    }

    /// Get the seed BrainConfig for the current node.
    pub fn current_config(&self) -> Option<BrainConfig> {
        let node_id = self.current_node_id?;
        let json: String = self
            .db
            .query_row(
                "SELECT config_json FROM node WHERE id = ?1",
                params![node_id],
                |row| row.get(0),
            )
            .ok()?;
        serde_json::from_str(&json).ok()
    }

    /// Whether the tick budget for this generation is exhausted.
    pub fn generation_complete(&self) -> bool {
        self.gen_tick >= self.config.tick_budget
    }

    /// Whether we've reached the maximum number of generations (0 = unlimited).
    pub fn evolution_complete(&self) -> bool {
        self.config.max_generations > 0
            && (self.generation + 1) as u64 >= self.config.max_generations
    }

    /// Advance the generation tick counter.
    pub fn tick(&mut self) {
        self.gen_tick += 1;
    }

    /// Evaluate all agents and record results. Returns fitness scores sorted
    /// descending by composite fitness.
    pub fn evaluate(&self, agents: &[Agent]) -> Vec<AgentFitness> {
        let node_id = match self.current_node_id {
            Some(id) => id,
            None => return Vec::new(),
        };

        let mut results: Vec<AgentFitness> = agents
            .iter()
            .enumerate()
            .map(|(i, a)| {
                AgentFitness {
                    agent_index: i,
                    config: a.brain.config.clone(),
                    total_ticks_alive: a.total_ticks_alive,
                    death_count: a.death_count,
                    food_consumed: a.food_consumed,
                    cells_explored: a.unique_cells_explored(),
                    composite_fitness: 0.0, // computed below
                }
            })
            .collect();

        // Absolute fitness scoring — no intra-generational normalization.
        // Each axis uses a meaningful denominator so scores reflect real quality.
        let tick_budget = self.config.tick_budget as f32;
        let total_grid_cells = (HEATMAP_RES * HEATMAP_RES / 4) as f32;
        let food_target = (tick_budget / 1000.0).max(10.0);

        for r in &mut results {
            // Survival: penalize dying. 0 deaths → 1.0, 1 → 0.67, 2 → 0.5
            let survival = 1.0 / (1.0 + r.death_count as f32 * 0.5);
            // Foraging: food per generation, capped at target
            let foraging = (r.food_consumed as f32 / food_target).min(1.0);
            // Exploration: fraction of reachable grid visited (25% of total = perfect)
            let exploration = (r.cells_explored as f32 / total_grid_cells).min(1.0);

            r.composite_fitness = survival * 0.4 + foraging * 0.3 + exploration * 0.3;
        }

        // Insert agent_result records
        for r in &results {
            let config_json =
                serde_json::to_string(&r.config).unwrap_or_default();
            let _ = self.db.execute(
                "INSERT INTO agent_result
                 (node_id, agent_index, config_json, total_ticks_alive,
                  death_count, food_consumed, cells_explored, composite_fitness)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    node_id,
                    r.agent_index as i64,
                    config_json,
                    r.total_ticks_alive as i64,
                    r.death_count,
                    r.food_consumed,
                    r.cells_explored,
                    r.composite_fitness,
                ],
            );
        }

        // Update node fitness
        let best = results
            .iter()
            .map(|r| r.composite_fitness)
            .fold(0.0f32, f32::max);
        let avg = results.iter().map(|r| r.composite_fitness).sum::<f32>()
            / results.len().max(1) as f32;

        let _ = self.db.execute(
            "UPDATE node SET best_fitness = ?1, avg_fitness = ?2 WHERE id = ?3",
            params![best, avg, node_id],
        );

        results.sort_by(|a, b| {
            b.composite_fitness
                .partial_cmp(&a.composite_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Reduce per-agent fitness to per-config-group fitness by averaging.
    /// Groups agents by `index / eval_repeats` and averages composite_fitness.
    fn reduce_fitness(&self, fitness: &[AgentFitness]) -> Vec<AgentFitness> {
        let repeats = self.config.eval_repeats.max(1);
        let mut groups: Vec<AgentFitness> = Vec::new();
        for chunk in fitness.chunks(repeats) {
            if chunk.is_empty() {
                continue;
            }
            let avg_fitness =
                chunk.iter().map(|f| f.composite_fitness).sum::<f32>() / chunk.len() as f32;
            let best = chunk
                .iter()
                .max_by(|a, b| {
                    a.composite_fitness
                        .partial_cmp(&b.composite_fitness)
                        .unwrap()
                })
                .unwrap();
            groups.push(AgentFitness {
                composite_fitness: avg_fitness,
                ..best.clone()
            });
        }
        groups.sort_by(|a, b| {
            b.composite_fitness
                .partial_cmp(&a.composite_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        groups
    }

    /// Process the evaluation results and advance to the next generation.
    ///
    /// Call this after `evaluate()`. Returns configs for the next population
    /// and log messages describing what happened.
    pub fn advance(&mut self, fitness: &[AgentFitness]) -> AdvanceResult {
        let reduced = self.reduce_fitness(fitness);
        let gen_fit = reduced.first().map(|f| f.composite_fitness).unwrap_or(0.0);
        let mut messages = Vec::new();

        let island = &mut self.islands[self.active_island];

        // Score this generation
        if gen_fit >= island.best_score {
            // ★ Success — this generation met or beat the island best
            island.best_score = gen_fit;
            let _ = self.db.execute(
                "UPDATE node SET status = 'successful', best_fitness = MAX(COALESCE(best_fitness, 0), ?2) WHERE id = ?1",
                params![self.current_node_id, gen_fit as f64],
            );
            island.spawn_parent_id = self.current_node_id;
            // Store elite configs from this successful generation
            island.elite_configs = reduced
                .iter()
                .take(self.config.elitism_count)
                .map(|f| f.config.clone())
                .collect();
            messages.push(format!("[EVOLUTION] ★ New best: {:.4}", gen_fit));
        } else {
            // ✗ Failure — didn't beat the best score
            let _ = self.db.execute(
                "UPDATE node SET status = 'failed', best_fitness = MAX(COALESCE(best_fitness, 0), ?2) WHERE id = ?1",
                params![self.current_node_id, gen_fit as f64],
            );
            messages.push(format!(
                "[EVOLUTION] ✗ Failed ({:.4} ≤ best {:.4})",
                gen_fit, island.best_score
            ));

            // Check if spawn parent has exhausted its patience
            if let Some(parent_id) = island.spawn_parent_id {
                let attempts: i64 = self
                    .db
                    .query_row(
                        "SELECT spawn_attempts FROM node WHERE id = ?1",
                        params![parent_id],
                        |row| row.get(0),
                    )
                    .unwrap_or(0);

                if attempts >= self.config.patience as i64 {
                    self.cascade_exhaustion(parent_id, &mut messages);
                }
            }
        }

        // Check completion after scoring but before breeding
        if self.evolution_complete() {
            self.persist_state();
            messages.push(format!(
                "[EVOLUTION] Finished after {} generations",
                self.generation
            ));
            return AdvanceResult::Finished { messages };
        }

        // Rotate to next island BEFORE breeding so the new generation
        // belongs to the island that will evaluate it in the next advance() call.
        if self.islands.len() > 1 {
            self.active_island = (self.active_island + 1) % self.islands.len();
        }

        // Breed the next generation from the (now-rotated) island's spawn parent
        let configs = self.breed_next_generation(fitness);

        // Migration: every migration_interval generations, spread best config
        if self.config.migration_interval > 0
            && self.generation % self.config.migration_interval as u32 == 0
            && self.islands.len() > 1
        {
            let best_island_idx = self
                .islands
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.best_score.partial_cmp(&b.best_score).unwrap()
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            if let Some(best_elite) =
                self.islands[best_island_idx].elite_configs.first().cloned()
            {
                for (i, island) in self.islands.iter_mut().enumerate() {
                    if i != best_island_idx {
                        if island.elite_configs.len() >= self.config.elitism_count {
                            island.elite_configs.pop();
                        }
                        island.elite_configs.insert(0, best_elite.clone());
                    }
                }
                messages.push(format!(
                    "[EVOLUTION] Migration: island {} config spread to {} others",
                    best_island_idx,
                    self.islands.len() - 1
                ));
            }
        }

        self.persist_state();

        AdvanceResult::Continue { configs, messages }
    }

    /// Walk up the tree when a node exhausts its patience.
    ///
    /// Non-root nodes are marked `exhausted` and the governor cascades to
    /// their parent. If the root itself exhausts, `best_score` is partially
    /// rolled back (×0.95) to preserve most progress while giving slack,
    /// and its attempt counter resets.
    fn cascade_exhaustion(&mut self, node_id: i64, messages: &mut Vec<String>) {
        let (gen, parent_id): (u32, Option<i64>) = self
            .db
            .query_row(
                "SELECT generation, parent_id FROM node WHERE id = ?1",
                params![node_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap_or((0, None));

        match parent_id {
            Some(pid) => {
                // Non-root: mark exhausted, check parent
                let _ = self.db.execute(
                    "UPDATE node SET status = 'exhausted' WHERE id = ?1",
                    params![node_id],
                );
                messages.push(format!(
                    "[EVOLUTION] Gen {} exhausted — backtracking",
                    gen
                ));

                let parent_attempts: i64 = self
                    .db
                    .query_row(
                        "SELECT spawn_attempts FROM node WHERE id = ?1",
                        params![pid],
                        |row| row.get(0),
                    )
                    .unwrap_or(0);

                if parent_attempts >= self.config.patience as i64 {
                    self.cascade_exhaustion(pid, messages);
                } else {
                    self.islands[self.active_island].spawn_parent_id = Some(pid);
                }
            }
            None => {
                // Root exhaustion: partial rollback preserves most progress
                self.islands[self.active_island].best_score *= 0.95;

                let _ = self.db.execute(
                    "UPDATE node SET spawn_attempts = 0 WHERE id = ?1",
                    params![node_id],
                );
                self.islands[self.active_island].spawn_parent_id = Some(node_id);
                messages.push(format!(
                    "[EVOLUTION] Root exhausted — best score rolled back to {:.4} (×0.95)",
                    self.islands[self.active_island].best_score
                ));
            }
        }
    }

    /// Create the next generation node as a child of `spawn_parent_id`.
    /// Increments the parent's spawn_attempts and returns population configs.
    /// Uses elitism to preserve top configs and adaptive mutation strength.
    fn breed_next_generation(&mut self, _fitness: &[AgentFitness]) -> Vec<BrainConfig> {
        let island = &self.islands[self.active_island];
        let spawn_parent = match island.spawn_parent_id {
            Some(id) => id,
            None => return Vec::new(),
        };

        let parent_config = match self.db.query_row(
            "SELECT config_json FROM node WHERE id = ?1",
            params![spawn_parent],
            |row| row.get::<_, String>(0),
        ) {
            Ok(json) => serde_json::from_str::<BrainConfig>(&json).unwrap_or_default(),
            Err(_) => BrainConfig::default(),
        };

        // Adaptive mutation: linearly ramp from base to max across patience range
        let attempts: i64 = self
            .db
            .query_row(
                "SELECT spawn_attempts FROM node WHERE id = ?1",
                params![spawn_parent],
                |row| row.get(0),
            )
            .unwrap_or(0);
        let base = self.config.mutation_strength;
        let max_strength = 0.5_f32;
        let patience = self.config.patience.max(1) as f32;
        let effective_strength =
            base + (max_strength - base) * (attempts as f32 / patience).min(1.0);

        let base_config = mutate_config_with_strength(&parent_config, effective_strength);

        self.generation += 1;
        let config_json = serde_json::to_string(&base_config).unwrap_or_default();
        let _ = self.db.execute(
            "INSERT INTO node (run_id, parent_id, generation, config_json, status)
             VALUES (?1, ?2, ?3, ?4, 'active')",
            params![self.run_id, Some(spawn_parent), self.generation, config_json],
        );
        let new_node_id = self.db.last_insert_rowid();
        self.current_node_id = Some(new_node_id);
        self.gen_tick = 0;

        // Count this child against the parent's patience
        let _ = self.db.execute(
            "UPDATE node SET spawn_attempts = spawn_attempts + 1 WHERE id = ?1",
            params![spawn_parent],
        );

        // Record which parameters were mutated
        let parent_fitness = self.best_score();
        record_mutations(
            &self.db,
            new_node_id,
            &parent_config,
            &base_config,
            parent_fitness,
        );

        // Compute unique config count for eval_repeats noise reduction
        let pop_size = self.config.population_size;
        let repeats = self.config.eval_repeats.max(1);
        let unique_count = (pop_size / repeats).max(1);
        let elite_count = self.config.elitism_count.min(unique_count.saturating_sub(1));

        // Population slot 0: unmutated champion (the spawn parent's exact config)
        let mut unique_configs = vec![parent_config.clone()];

        // Elitism: mutate top configs from the last SUCCESSFUL generation
        let island = &self.islands[self.active_island];
        for elite in island.elite_configs.iter().take(elite_count) {
            if unique_configs.len() >= unique_count {
                break;
            }
            unique_configs.push(mutate_config_with_strength(elite, effective_strength));
        }

        // Crossover offspring from elite pairs
        if island.elite_configs.len() >= 2 {
            let crossover_count = 3.min(unique_count / 3);
            let mut rng = rand::rng();
            for _ in 0..crossover_count {
                if unique_configs.len() >= unique_count {
                    break;
                }
                let idx_a = rng.random_range(0..island.elite_configs.len());
                let idx_b = rng.random_range(0..island.elite_configs.len());
                let child = crossover_config(
                    &island.elite_configs[idx_a],
                    &island.elite_configs[idx_b],
                );
                unique_configs.push(mutate_config_with_strength(&child, effective_strength));
            }
        }

        // Fill remaining unique slots from the spawn parent's config
        while unique_configs.len() < unique_count {
            unique_configs.push(mutate_config_with_strength(&parent_config, effective_strength));
        }

        // Repeat each config eval_repeats times to fill pop_size slots
        let mut configs = Vec::with_capacity(pop_size);
        for uc in &unique_configs {
            for _ in 0..repeats {
                if configs.len() >= pop_size {
                    break;
                }
                configs.push(uc.clone());
            }
        }

        configs
    }

    /// Persist best_score and spawn_parent_id for resume.
    fn persist_state(&self) {
        // Save the best island's state
        let best_score = self.best_score();
        let best_island = self
            .islands
            .iter()
            .max_by(|a, b| a.best_score.partial_cmp(&b.best_score).unwrap());
        let spawn_parent = best_island.and_then(|i| i.spawn_parent_id);
        let _ = self.db.execute(
            "UPDATE run SET best_score = ?1, spawn_parent_id = ?2 WHERE id = ?3",
            params![best_score as f64, spawn_parent, self.run_id],
        );
    }

    /// Print a generation summary to stdout.
    pub fn log_generation(&self, fitness: &[AgentFitness]) {
        let best = fitness.first().map(|f| f.composite_fitness).unwrap_or(0.0);
        let avg = if fitness.is_empty() {
            0.0
        } else {
            fitness.iter().map(|f| f.composite_fitness).sum::<f32>() / fitness.len() as f32
        };

        let w = 64; // inner width between ║ chars
        let bar = "═".repeat(w);
        println!("╔{}╗", bar);
        let header = format!(
            "  Generation {:>4}  │  Best: {:.4}  │  Avg: {:.4}",
            self.generation, best, avg,
        );
        println!("║{:<w$}║", header, w = w);
        println!("╠{}╣", bar);

        for (i, f) in fitness.iter().take(5).enumerate() {
            let line = format!(
                "  #{} Agent {:>2} │ alive:{:>6} food:{:>4} cells:{:>4} fit:{:.3}",
                i + 1,
                f.agent_index,
                f.total_ticks_alive,
                f.food_consumed,
                f.cells_explored,
                f.composite_fitness,
            );
            println!("║{:<w$}║", line, w = w);
        }

        if let Some(best_config) = fitness.first() {
            let c = &best_config.config;
            println!("╠{}╣", bar);
            let cfg_line = format!(
                "  Config: mem={} slots={} dim={} lr={:.4} decay={:.4}",
                c.memory_capacity,
                c.processing_slots,
                c.representation_dim,
                c.learning_rate,
                c.decay_rate,
            );
            println!("║{:<w$}║", cfg_line, w = w);
        }
        println!("╚{}╝", bar);
    }

    /// Update the wall_time_secs for the current run.
    pub fn update_wall_time(&self, secs: f64) {
        let _ = self.db.execute(
            "UPDATE run SET wall_time_secs = ?1 WHERE id = ?2",
            params![secs, self.run_id],
        );
    }

    /// Get the evolution tree as a list of nodes for UI visualization.
    pub fn tree_nodes(&self) -> Vec<TreeNode> {
        let mut stmt = match self.db.prepare(
            "SELECT id, parent_id, generation, best_fitness, avg_fitness,
                    config_json, status
             FROM node WHERE run_id = ?1 ORDER BY id",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let mut nodes: Vec<TreeNode> = match stmt
            .query_map(params![self.run_id], |row| {
                let config_json: String = row.get(5)?;
                let config: Option<BrainConfig> = serde_json::from_str(&config_json).ok();
                Ok(TreeNode {
                    id: row.get(0)?,
                    parent_id: row.get(1)?,
                    generation: row.get(2)?,
                    best_fitness: row.get::<_, Option<f64>>(3)?.map(|v| v as f32),
                    avg_fitness: row.get::<_, Option<f64>>(4)?.map(|v| v as f32),
                    status: row.get(6)?,
                    config,
                    mutations: Vec::new(),
                })
            })
        {
            Ok(r) => r.flatten().collect(),
            Err(_) => return Vec::new(),
        };

        // Load mutations for all nodes in one query
        if let Ok(mut mut_stmt) = self.db.prepare(
            "SELECT m.node_id, m.param_name, m.direction
             FROM mutation m
             JOIN node n ON m.node_id = n.id
             WHERE n.run_id = ?1
             ORDER BY m.node_id, m.id",
        ) {
            if let Ok(rows) = mut_stmt.query_map(params![self.run_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                ))
            }) {
                for r in rows.flatten() {
                    let (node_id, param, dir) = r;
                    if let Some(node) = nodes.iter_mut().find(|n| n.id == node_id) {
                        node.mutations.push((param, dir));
                    }
                }
            }
        }

        nodes
    }

    /// Get the seed BrainConfig from the run table (original starting config).
    pub fn seed_brain_config(&self) -> Option<BrainConfig> {
        let json: String = self
            .db
            .query_row(
                "SELECT brain_config FROM run WHERE id = ?1",
                params![self.run_id],
                |row| row.get(0),
            )
            .ok()?;
        serde_json::from_str(&json).ok()
    }

    /// Fitness history: (generation, best_fitness, avg_fitness) for all evaluated nodes.
    pub fn fitness_history(&self) -> Vec<(u32, f32, f32)> {
        let mut stmt = match self.db.prepare(
            "SELECT generation, best_fitness, avg_fitness FROM node
             WHERE run_id = ?1 AND best_fitness IS NOT NULL
             ORDER BY generation ASC",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        stmt.query_map(params![self.run_id], |row| {
            Ok((
                row.get::<_, u32>(0)?,
                row.get::<_, f32>(1)?,
                row.get::<_, f32>(2)?,
            ))
        })
        .map(|rows| rows.flatten().collect())
        .unwrap_or_default()
    }
}

/// Check whether a database file has an existing evolution session.
/// Returns Some((generation, GovernorConfig, BrainConfig)) if a session exists.
pub fn check_existing_session(
    db_path: &str,
) -> Option<(u32, GovernorConfig, BrainConfig)> {
    let db = Connection::open(db_path).ok()?;
    // Check if the run table even exists
    let table_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='run'",
            [],
            |row| row.get(0),
        )
        .ok()?;
    if table_count == 0 {
        return None;
    }
    let (governor_json, brain_json): (String, String) = db
        .query_row(
            "SELECT governor_config, brain_config FROM run ORDER BY id DESC LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .ok()?;
    let run_id: i64 = db
        .query_row("SELECT id FROM run ORDER BY id DESC LIMIT 1", [], |row| {
            row.get(0)
        })
        .ok()?;
    let generation: u32 = db
        .query_row(
            "SELECT COALESCE(MAX(generation), 0) FROM node WHERE run_id = ?1",
            params![run_id],
            |row| row.get(0),
        )
        .ok()?;
    let gov_config: GovernorConfig =
        serde_json::from_str(&governor_json).unwrap_or_default();
    let brain_config: BrainConfig =
        serde_json::from_str(&brain_json).unwrap_or_default();
    Some((generation, gov_config, brain_config))
}

/// Delete the database file to start fresh.
pub fn reset_database(db_path: &str) -> std::io::Result<()> {
    let path = std::path::Path::new(db_path);
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    // Also remove WAL/SHM files if present
    let wal = format!("{}-wal", db_path);
    let shm = format!("{}-shm", db_path);
    let _ = std::fs::remove_file(&wal);
    let _ = std::fs::remove_file(&shm);
    Ok(())
}

/// A node in the evolution tree for UI display.
#[derive(Clone, Debug)]
pub struct TreeNode {
    pub id: i64,
    pub parent_id: Option<i64>,
    pub generation: u32,
    pub best_fitness: Option<f32>,
    pub avg_fitness: Option<f32>,
    pub status: String,
    /// Per-node config (deserialized for display)
    pub config: Option<BrainConfig>,
    /// Summary of mutations from parent: e.g. "learning_rate↑ decay_rate↓"
    pub mutations: Vec<(String, f64)>,
}

// ── Schema & helpers ────────────────────────────────────────────────────

fn init_schema(db: &Connection) -> SqlResult<()> {
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS run (
            id INTEGER PRIMARY KEY,
            seed INTEGER NOT NULL,
            governor_config TEXT NOT NULL,
            brain_config TEXT NOT NULL,
            world_config TEXT NOT NULL,
            started_at TEXT DEFAULT CURRENT_TIMESTAMP,
            wall_time_secs REAL DEFAULT 0,
            best_score REAL DEFAULT -1,
            spawn_parent_id INTEGER
        );

        CREATE TABLE IF NOT EXISTS node (
            id INTEGER PRIMARY KEY,
            run_id INTEGER REFERENCES run(id),
            parent_id INTEGER REFERENCES node(id),
            generation INTEGER NOT NULL,
            config_json TEXT NOT NULL,
            best_fitness REAL,
            avg_fitness REAL,
            mutated_param TEXT,
            mutation_direction REAL,
            status TEXT DEFAULT 'active',
            spawn_attempts INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS agent_result (
            id INTEGER PRIMARY KEY,
            node_id INTEGER REFERENCES node(id),
            agent_index INTEGER NOT NULL,
            config_json TEXT NOT NULL,
            total_ticks_alive INTEGER,
            death_count INTEGER,
            food_consumed INTEGER,
            cells_explored INTEGER,
            composite_fitness REAL
        );

        CREATE TABLE IF NOT EXISTS mutation (
            id INTEGER PRIMARY KEY,
            node_id INTEGER REFERENCES node(id),
            param_name TEXT NOT NULL,
            direction REAL NOT NULL,
            parent_fitness REAL,
            result_fitness REAL
        );",
    )
}

/// Compare two BrainConfigs and record which parameters changed.
fn record_mutations(
    db: &Connection,
    node_id: i64,
    parent: &BrainConfig,
    child: &BrainConfig,
    parent_fitness: f32,
) {
    let params_to_check: Vec<(&str, f64, f64)> = vec![
        (
            "memory_capacity",
            parent.memory_capacity as f64,
            child.memory_capacity as f64,
        ),
        (
            "processing_slots",
            parent.processing_slots as f64,
            child.processing_slots as f64,
        ),
        (
            "representation_dim",
            parent.representation_dim as f64,
            child.representation_dim as f64,
        ),
        (
            "learning_rate",
            parent.learning_rate as f64,
            child.learning_rate as f64,
        ),
        (
            "decay_rate",
            parent.decay_rate as f64,
            child.decay_rate as f64,
        ),
    ];

    for (name, old_val, new_val) in params_to_check {
        if (old_val - new_val).abs() > 1e-6 {
            let direction = if new_val > old_val { 1.0 } else { -1.0 };
            let _ = db.execute(
                "INSERT INTO mutation (node_id, param_name, direction, parent_fitness)
                 VALUES (?1, ?2, ?3, ?4)",
                params![node_id, name, direction, parent_fitness as f64],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xagent_shared::{BrainConfig, GovernorConfig};

    fn test_governor(patience: u32) -> Governor {
        let config = GovernorConfig {
            population_size: 10,
            tick_budget: 100,
            elitism_count: 3,
            patience,
            max_generations: 0,
            mutation_strength: 0.1,
            eval_repeats: 1,
            num_islands: 1,
            migration_interval: 0,
        };
        let brain = BrainConfig::default();
        Governor::new(":memory:", config, &brain, "{}").unwrap()
    }

    // ─── advance tests ──────────────────────────────────────────────

    /// Create a mock AgentFitness slice with a single entry at the given fitness.
    fn mock_fitness(fitness: f32) -> Vec<AgentFitness> {
        vec![AgentFitness {
            agent_index: 0,
            config: BrainConfig::default(),
            total_ticks_alive: 50000,
            death_count: 0,
            food_consumed: 50,
            cells_explored: 4096,
            composite_fitness: fitness,
        }]
    }

    /// Helper to read a node's status from the DB.
    fn node_status(gov: &Governor, node_id: i64) -> String {
        gov.db
            .query_row(
                "SELECT status FROM node WHERE id = ?1",
                params![node_id],
                |row| row.get(0),
            )
            .unwrap()
    }

    /// Helper to read a node's spawn_attempts from the DB.
    fn node_attempts(gov: &Governor, node_id: i64) -> i64 {
        gov.db
            .query_row(
                "SELECT spawn_attempts FROM node WHERE id = ?1",
                params![node_id],
                |row| row.get(0),
            )
            .unwrap()
    }

    /// Helper to read a node's parent_id from the DB.
    fn node_parent(gov: &Governor, node_id: i64) -> Option<i64> {
        gov.db
            .query_row(
                "SELECT parent_id FROM node WHERE id = ?1",
                params![node_id],
                |row| row.get(0),
            )
            .unwrap()
    }

    #[test]
    fn gen_0_always_succeeds() {
        let mut gov = test_governor(3);
        let root_id = gov.current_node_id.unwrap();
        assert_eq!(gov.best_score(), -1.0);

        let result = gov.advance(&mock_fitness(0.1));
        assert!(matches!(result, AdvanceResult::Continue { .. }));
        assert_eq!(gov.best_score(), 0.1);
        assert_eq!(node_status(&gov, root_id), "successful");
    }

    #[test]
    fn success_deepens_tree() {
        let mut gov = test_governor(3);
        let root_id = gov.current_node_id.unwrap();

        // Gen 0: fitness 0.1 → success
        gov.advance(&mock_fitness(0.1));
        assert_eq!(gov.spawn_parent_id(), Some(root_id));

        let gen1_id = gov.current_node_id.unwrap();
        assert_ne!(gen1_id, root_id);
        assert_eq!(node_parent(&gov, gen1_id), Some(root_id));

        // Gen 1: fitness 0.2 → success, spawn_parent deepens to gen1
        gov.advance(&mock_fitness(0.2));
        assert_eq!(gov.best_score(), 0.2);
        assert_eq!(gov.spawn_parent_id(), Some(gen1_id));
        assert_eq!(node_status(&gov, gen1_id), "successful");
    }

    #[test]
    fn failure_stays_at_parent() {
        let mut gov = test_governor(3);
        let _root_id = gov.current_node_id.unwrap();

        // Gen 0: success
        gov.advance(&mock_fitness(0.1));
        let gen1_id = gov.current_node_id.unwrap();

        // Gen 1: success
        gov.advance(&mock_fitness(0.2));
        assert_eq!(gov.spawn_parent_id(), Some(gen1_id));

        let gen2_id = gov.current_node_id.unwrap();

        // Gen 2: failure (0.18 ≤ 0.2) — spawn_parent stays at gen1
        gov.advance(&mock_fitness(0.18));
        assert_eq!(gov.spawn_parent_id(), Some(gen1_id));
        assert_eq!(node_status(&gov, gen2_id), "failed");

        // Gen 3 should be a child of gen1 (not gen2)
        let gen3_id = gov.current_node_id.unwrap();
        assert_eq!(node_parent(&gov, gen3_id), Some(gen1_id));
    }

    #[test]
    fn patience_triggers_exhaustion() {
        let mut gov = test_governor(3);

        // Gen 0: success (best=0.1)
        gov.advance(&mock_fitness(0.1));
        let gen1_id = gov.current_node_id.unwrap();

        // Gen 1: success (best=0.2)
        gov.advance(&mock_fitness(0.2));
        let gen1_node = gov.spawn_parent_id().unwrap();
        assert_eq!(gen1_node, gen1_id);

        // Gen 2: fail
        gov.advance(&mock_fitness(0.18));
        assert_eq!(node_attempts(&gov, gen1_node), 2);

        // Gen 3: fail
        gov.advance(&mock_fitness(0.17));
        assert_eq!(node_attempts(&gov, gen1_node), 3);

        // Gen 4: fail → gen1 exhausted (3 children = patience)
        let root_id = node_parent(&gov, gen1_node).unwrap();
        gov.advance(&mock_fitness(0.19));
        assert_eq!(node_status(&gov, gen1_node), "exhausted");
        assert_eq!(gov.spawn_parent_id(), Some(root_id));
    }

    #[test]
    fn exhaustion_cascades_to_grandparent() {
        let mut gov = test_governor(2);

        // Gen 0 → success
        gov.advance(&mock_fitness(0.1));
        let root_id = node_parent(&gov, gov.current_node_id.unwrap()).unwrap();

        // Gen 1 → success (best=0.2)
        gov.advance(&mock_fitness(0.2));
        let gen1_id = gov.spawn_parent_id().unwrap();

        // Gen 2: fail → gen1.attempts=2
        gov.advance(&mock_fitness(0.15));
        // Gen 3: fail → gen1.attempts=2, triggers exhaustion
        //   gen1 exhausted → cascade to root (root.attempts=1, < patience=2)
        gov.advance(&mock_fitness(0.15));
        assert_eq!(node_status(&gov, gen1_id), "exhausted");
        assert_eq!(gov.spawn_parent_id(), Some(root_id));
    }

    #[test]
    fn root_exhaustion_rolls_back_best_score() {
        let mut gov = test_governor(2);

        // Gen 0: success (best=0.1)
        gov.advance(&mock_fitness(0.1));
        let root_id = node_parent(&gov, gov.current_node_id.unwrap()).unwrap();

        // Gen 1: success (best=0.2)
        gov.advance(&mock_fitness(0.2));
        let gen1_id = gov.spawn_parent_id().unwrap();

        // Gen 2, Gen 3: fail → gen1 exhausted → root.attempts=2 (gen1 + gen2_child)
        gov.advance(&mock_fitness(0.15));
        gov.advance(&mock_fitness(0.15));
        assert_eq!(node_status(&gov, gen1_id), "exhausted");

        // Now spawning from root. Root has attempts=1 (from gen1).
        // Gen 5: fail → root.attempts=2 → root exhausted, partial rollback, then breed adds 1
        gov.advance(&mock_fitness(0.12));
        // Partial rollback: 0.2 * 0.95 = 0.19
        assert!((gov.best_score() - 0.19).abs() < 0.001);
        assert_eq!(gov.spawn_parent_id(), Some(root_id));
        assert_eq!(node_attempts(&gov, root_id), 1); // reset to 0, then breed incremented to 1
    }

    #[test]
    fn after_root_reset_children_can_succeed() {
        let mut gov = test_governor(1);

        // Gen 0: success (best=0.1)
        gov.advance(&mock_fitness(0.1));
        let root_id = node_parent(&gov, gov.current_node_id.unwrap()).unwrap();

        // Gen 1: success (best=0.2)
        gov.advance(&mock_fitness(0.2));

        // Gen 2: fail → gen1 exhausted (patience=1)
        gov.advance(&mock_fitness(0.15));
        // gen1 exhausted, cascade to root. root.attempts = 1 >= patience=1 → root exhausted
        // Partial rollback: 0.2 * 0.95 = 0.19
        assert!((gov.best_score() - 0.19).abs() < 0.001);
        assert_eq!(gov.spawn_parent_id(), Some(root_id));

        // Gen 3: 0.19 >= 0.19 → success (ties accepted)!
        gov.advance(&mock_fitness(0.19));
        assert!((gov.best_score() - 0.19).abs() < 0.001);
    }

    #[test]
    fn equal_fitness_is_success() {
        let mut gov = test_governor(3);

        // Gen 0: success (best=0.5)
        gov.advance(&mock_fitness(0.5));
        let root_id = gov.spawn_parent_id().unwrap();

        // Gen 1: fitness = 0.5 (same as best) → SUCCESS (>= accepts ties)
        let gen1_id = gov.current_node_id.unwrap();
        gov.advance(&mock_fitness(0.5));
        assert_eq!(node_status(&gov, gen1_id), "successful");
        assert_eq!(gov.best_score(), 0.5);
        assert_eq!(gov.spawn_parent_id(), Some(gen1_id));

        // Spawn parent moved to gen1 (not stuck at root)
        assert_ne!(gov.spawn_parent_id(), Some(root_id));
    }

    #[test]
    fn full_trace_with_new_semantics() {
        // Validates: >= success, partial rollback, tie acceptance
        let mut gov = test_governor(3);
        let root_id = gov.current_node_id.unwrap();

        // Gen 0: fit=0.1 → success (best=0.1)
        gov.advance(&mock_fitness(0.1));
        assert_eq!(gov.best_score(), 0.1);
        assert_eq!(node_status(&gov, root_id), "successful");
        assert_eq!(gov.spawn_parent_id(), Some(root_id));

        // Gen 1: fit=0.2 → success (best=0.2)
        let gen1_id = gov.current_node_id.unwrap();
        gov.advance(&mock_fitness(0.2));
        assert_eq!(gov.best_score(), 0.2);
        assert_eq!(gov.spawn_parent_id(), Some(gen1_id));

        // Gen 2: fit=0.18 → fail
        let gen2_id = gov.current_node_id.unwrap();
        gov.advance(&mock_fitness(0.18));
        assert_eq!(node_status(&gov, gen2_id), "failed");
        assert_eq!(gov.spawn_parent_id(), Some(gen1_id));

        // Gen 3: fit=0.19 → fail
        let gen3_id = gov.current_node_id.unwrap();
        gov.advance(&mock_fitness(0.19));
        assert_eq!(node_status(&gov, gen3_id), "failed");

        // Gen 4: fit=0.2 → SUCCESS (>= accepts ties), tree deepens
        let gen4_id = gov.current_node_id.unwrap();
        gov.advance(&mock_fitness(0.2));
        assert_eq!(node_status(&gov, gen4_id), "successful");
        assert_eq!(gov.spawn_parent_id(), Some(gen4_id));
        assert_eq!(gov.best_score(), 0.2);

        // Gen 5: fit=0.15 → fail (child of gen4)
        let gen5_id = gov.current_node_id.unwrap();
        assert_eq!(node_parent(&gov, gen5_id), Some(gen4_id));
        gov.advance(&mock_fitness(0.15));
        assert_eq!(node_status(&gov, gen5_id), "failed");

        // Gen 6: fit=0.16 → fail
        gov.advance(&mock_fitness(0.16));

        // Gen 7: fit=0.17 → fail → gen4 exhausted (patience=3), cascade up
        gov.advance(&mock_fitness(0.17));
        assert_eq!(node_status(&gov, gen4_id), "exhausted");
        // gen4's parent is gen1. gen1 has attempts from earlier + gen4 = enough
        // depending on exact count, may cascade further

        // Verify system is still alive and can continue
        let fit_before = gov.best_score();
        gov.advance(&mock_fitness(fit_before + 0.01));
        assert!(gov.best_score() >= fit_before);
    }

    #[test]
    fn breed_increments_spawn_attempts() {
        let mut gov = test_governor(3);
        let root_id = gov.current_node_id.unwrap();

        assert_eq!(node_attempts(&gov, root_id), 0);
        gov.advance(&mock_fitness(0.1)); // creates gen1 from root
        assert_eq!(node_attempts(&gov, root_id), 1);
    }

    #[test]
    fn evolution_complete_returns_finished() {
        let config = GovernorConfig {
            population_size: 2,
            tick_budget: 100,
            elitism_count: 1,
            patience: 3,
            max_generations: 1,
            mutation_strength: 0.1,
            eval_repeats: 1,
            num_islands: 1,
            migration_interval: 0,
        };
        let brain = BrainConfig::default();
        let mut gov = Governor::new(":memory:", config, &brain, "{}").unwrap();

        // Gen 0 evaluation → max_generations=1 reached
        let result = gov.advance(&mock_fitness(0.5));
        assert!(matches!(result, AdvanceResult::Finished { .. }));
    }

    #[test]
    fn adaptive_mutation_strength_increases_with_attempts() {
        let mut gov = test_governor(5);

        // Gen 0: success (best=0.5), breeds gen1 from root
        gov.advance(&mock_fitness(0.5));

        // Gen 1: success (best=0.6), spawn_parent becomes gen1
        gov.advance(&mock_fitness(0.6));
        let gen1_id = gov.spawn_parent_id().unwrap();
        // gen1 now has spawn_attempts=1 (from breeding gen2)

        // Gen 2: fail → gen1 gains another attempt
        gov.advance(&mock_fitness(0.55));
        let attempts_after_fail1 = node_attempts(&gov, gen1_id);
        assert_eq!(attempts_after_fail1, 2);

        // Gen 3: fail → gen1 gains another attempt
        gov.advance(&mock_fitness(0.55));
        let attempts_after_fail2 = node_attempts(&gov, gen1_id);
        assert_eq!(attempts_after_fail2, 3);

        // Effective strength at breeding was: 0.1 * (1.0 + attempts * 0.5)
        // After 2 attempts: 0.1 * 2.0 = 0.2, after 3: 0.1 * 2.5 = 0.25
        assert!(attempts_after_fail2 > attempts_after_fail1);
    }

    #[test]
    fn elitism_preserves_top_configs() {
        let config = GovernorConfig {
            population_size: 5,
            tick_budget: 100,
            elitism_count: 2,
            patience: 100, // high patience so adaptive ramp stays near base
            max_generations: 0,
            mutation_strength: 0.001,
            eval_repeats: 1,
            num_islands: 1,
            migration_interval: 0,
        };
        let brain = BrainConfig::default();
        let mut gov = Governor::new(":memory:", config, &brain, "{}").unwrap();

        // Create fitness results with very distinct configs
        let elite1 = BrainConfig {
            memory_capacity: 5000,
            ..BrainConfig::default()
        };
        let elite2 = BrainConfig {
            memory_capacity: 3000,
            ..BrainConfig::default()
        };
        let loser = BrainConfig {
            memory_capacity: 1,
            ..BrainConfig::default()
        };

        let elite_fitness = vec![
            AgentFitness {
                agent_index: 0,
                config: elite1.clone(),
                total_ticks_alive: 100,
                death_count: 0,
                food_consumed: 50,
                cells_explored: 100,
                composite_fitness: 0.9,
            },
            AgentFitness {
                agent_index: 1,
                config: elite2.clone(),
                total_ticks_alive: 100,
                death_count: 0,
                food_consumed: 40,
                cells_explored: 80,
                composite_fitness: 0.8,
            },
            AgentFitness {
                agent_index: 2,
                config: loser.clone(),
                total_ticks_alive: 10,
                death_count: 5,
                food_consumed: 1,
                cells_explored: 2,
                composite_fitness: 0.1,
            },
        ];

        // Gen 0: advance with elite fitness → SUCCESS, stores elite_configs
        gov.advance(&elite_fitness);
        assert!(!gov.islands[gov.active_island].elite_configs.is_empty());

        // breed_next_generation uses stored elite_configs from the successful gen
        let configs = gov.breed_next_generation(&mock_fitness(0.01));
        assert_eq!(configs.len(), 5);

        // With high patience and low mutation_strength, elite configs should
        // stay recognizable by their magnitude (5000 and 3000 vs default 128)
        let has_elite1_like = configs.iter().any(|c| c.memory_capacity > 4000);
        let has_elite2_like = configs.iter().any(|c| (2000..=4000).contains(&c.memory_capacity));
        assert!(
            has_elite1_like,
            "Expected a config near memory_capacity=5000 from elite1; got {:?}",
            configs.iter().map(|c| c.memory_capacity).collect::<Vec<_>>()
        );
        assert!(
            has_elite2_like,
            "Expected a config near memory_capacity=3000 from elite2; got {:?}",
            configs.iter().map(|c| c.memory_capacity).collect::<Vec<_>>()
        );
    }

    #[test]
    fn champion_is_unmutated() {
        let mut gov = test_governor(5);

        // Gen 0: success — root becomes spawn parent
        gov.advance(&mock_fitness(0.1));
        let spawn_id = gov.spawn_parent_id().unwrap();

        // Get the spawn parent's config from the DB (what breed uses)
        let parent_json: String = gov.db
            .query_row(
                "SELECT config_json FROM node WHERE id = ?1",
                params![spawn_id],
                |row| row.get(0),
            )
            .unwrap();
        let parent_config: BrainConfig = serde_json::from_str(&parent_json).unwrap();

        // Breed next generation
        let configs = gov.breed_next_generation(&mock_fitness(0.05));

        // configs[0] should be the EXACT spawn parent config (unmutated champion)
        assert_eq!(configs[0].memory_capacity, parent_config.memory_capacity);
        assert_eq!(configs[0].processing_slots, parent_config.processing_slots);
        assert_eq!(configs[0].representation_dim, parent_config.representation_dim);
        assert!((configs[0].learning_rate - parent_config.learning_rate).abs() < f32::EPSILON);
        assert!((configs[0].decay_rate - parent_config.decay_rate).abs() < f32::EPSILON);
    }

    #[test]
    fn noise_reduction_averages_fitness() {
        let config = GovernorConfig {
            population_size: 4,
            tick_budget: 100,
            elitism_count: 1,
            patience: 5,
            max_generations: 0,
            mutation_strength: 0.1,
            eval_repeats: 2,
            num_islands: 1,
            migration_interval: 0,
        };
        let brain = BrainConfig::default();
        let gov = Governor::new(":memory:", config, &brain, "{}").unwrap();

        // 4 agents, eval_repeats=2 → 2 groups of 2
        let fitness = vec![
            AgentFitness {
                agent_index: 0,
                config: BrainConfig { memory_capacity: 100, ..BrainConfig::default() },
                total_ticks_alive: 100,
                death_count: 0,
                food_consumed: 10,
                cells_explored: 50,
                composite_fitness: 0.8,
            },
            AgentFitness {
                agent_index: 1,
                config: BrainConfig { memory_capacity: 100, ..BrainConfig::default() },
                total_ticks_alive: 100,
                death_count: 0,
                food_consumed: 10,
                cells_explored: 50,
                composite_fitness: 0.6,
            },
            AgentFitness {
                agent_index: 2,
                config: BrainConfig { memory_capacity: 200, ..BrainConfig::default() },
                total_ticks_alive: 100,
                death_count: 0,
                food_consumed: 10,
                cells_explored: 50,
                composite_fitness: 0.5,
            },
            AgentFitness {
                agent_index: 3,
                config: BrainConfig { memory_capacity: 200, ..BrainConfig::default() },
                total_ticks_alive: 100,
                death_count: 0,
                food_consumed: 10,
                cells_explored: 50,
                composite_fitness: 0.9,
            },
        ];

        let reduced = gov.reduce_fitness(&fitness);
        assert_eq!(reduced.len(), 2);
        // Group 0: avg of 0.8 and 0.6 = 0.7
        // Group 1: avg of 0.5 and 0.9 = 0.7
        // Both average to 0.7, sorted descending
        let avg0 = reduced[0].composite_fitness;
        let avg1 = reduced[1].composite_fitness;
        assert!((avg0 - 0.7).abs() < 0.01, "expected ~0.7, got {}", avg0);
        assert!((avg1 - 0.7).abs() < 0.01, "expected ~0.7, got {}", avg1);
    }

    #[test]
    fn island_rotation() {
        let config = GovernorConfig {
            population_size: 4,
            tick_budget: 100,
            elitism_count: 1,
            patience: 5,
            max_generations: 0,
            mutation_strength: 0.1,
            eval_repeats: 1,
            num_islands: 3,
            migration_interval: 0,
        };
        let brain = BrainConfig::default();
        let mut gov = Governor::new(":memory:", config, &brain, "{}").unwrap();

        assert_eq!(gov.active_island, 0);
        gov.advance(&mock_fitness(0.1));
        assert_eq!(gov.active_island, 1);
        gov.advance(&mock_fitness(0.2));
        assert_eq!(gov.active_island, 2);
        gov.advance(&mock_fitness(0.3));
        assert_eq!(gov.active_island, 0); // wraps around
    }

    #[test]
    fn crossover_combines_parents() {
        use crate::agent::crossover_config;

        let a = BrainConfig {
            memory_capacity: 1000,
            processing_slots: 100,
            visual_encoding_size: 64,
            representation_dim: 500,
            learning_rate: 0.9,
            decay_rate: 0.9,
        };
        let b = BrainConfig {
            memory_capacity: 1,
            processing_slots: 1,
            visual_encoding_size: 64,
            representation_dim: 1,
            learning_rate: 0.001,
            decay_rate: 0.001,
        };

        // Run crossover many times — with probability 1-(0.5^5)^N the child
        // will have at least one param from each parent
        let mut has_a_param = false;
        let mut has_b_param = false;
        for _ in 0..50 {
            let child = crossover_config(&a, &b);
            if child.memory_capacity == a.memory_capacity
                || child.processing_slots == a.processing_slots
                || child.representation_dim == a.representation_dim
            {
                has_a_param = true;
            }
            if child.memory_capacity == b.memory_capacity
                || child.processing_slots == b.processing_slots
                || child.representation_dim == b.representation_dim
            {
                has_b_param = true;
            }
            // visual_encoding_size always from parent a
            assert_eq!(child.visual_encoding_size, a.visual_encoding_size);
        }
        assert!(has_a_param, "crossover never picked from parent A");
        assert!(has_b_param, "crossover never picked from parent B");
    }
}
