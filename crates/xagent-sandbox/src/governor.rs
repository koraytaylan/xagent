//! Evolution governor — orchestrates natural selection across generations.
//!
//! The governor manages a population of agents, evaluates their fitness after
//! a fixed tick budget, selects the best performers, and breeds mutated offspring.
//! All state is persisted in a SQLite database (`xagent.db`), enabling resume
//! and post-hoc analysis of the evolution tree.

use std::sync::mpsc::{self, SyncSender};
use std::thread::JoinHandle;

use rand::Rng;
use rusqlite::{params, Connection, Result as SqlResult};
use serde::Serialize;
use xagent_shared::{BrainConfig, GovernorConfig};

use crate::momentum::MutationMomentum;

/// Payload sent to the background recording-writer thread.
struct RecordingPayload {
    node_id: i64,
    agent_count: i64,
    tick_count: i64,
    data: Vec<u8>,
}

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
        /// Effective mutation strength for this generation (used to perturb
        /// inherited weights in neuroevolution).
        mutation_strength: f32,
    },
    /// Max generations reached — stop the simulation.
    Finished { messages: Vec<String> },
}

/// An independent evolutionary lineage within the island model.
#[derive(Clone, Debug)]
struct Island {
    spawn_parent_id: Option<i64>,
    elite_configs: Vec<BrainConfig>,
    /// Per-island patience counter at the current spawn parent.
    /// Authoritative for exhaustion decisions (DB spawn_attempts is display-only).
    attempts: u32,
}

/// The evolution governor. Owns the SQLite connection and drives the
/// generational loop: evaluate → advance → breed → repeat.
///
/// # Algorithm
///
/// A generation "succeeds" if its fitness meets or exceeds the **spawn
/// parent's** fitness (not the island's all-time best). This enables
/// gradual development: each branch only needs to improve on its parent,
/// and backtracking naturally lowers the bar to wherever you backtrack to.
///
/// Each island tracks its own patience counter (`attempts`). When attempts
/// reach `patience`, the island backtracks one level up the tree and gets
/// a fresh patience budget. If the island is already at root, attempts
/// reset and exploration continues with the root as spawn parent.
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
    /// Cached best_score value (updated on advance).
    cached_best_score: f32,
    /// Cached tree nodes (invalidated on advance).
    cached_tree_nodes: Option<Vec<TreeNode>>,
    /// Per-island mutation momentum vectors.
    pub momentums: Vec<MutationMomentum>,
    /// Channel sender for offloading recording writes to a background thread.
    recording_sender: Option<SyncSender<RecordingPayload>>,
    /// Handle to the background writer thread (joined on drop).
    writer_thread: Option<JoinHandle<()>>,
}

/// Spawn a background thread that owns a dedicated SQLite connection and
/// drains `RecordingPayload` messages from a bounded channel, writing each
/// one to the `generation_recording` table.
///
/// Returns `None` if the thread cannot be spawned, allowing the caller to
/// continue without recording support.
fn spawn_recording_writer(db_path: &str) -> Option<(SyncSender<RecordingPayload>, JoinHandle<()>)> {
    let (tx, rx) = mpsc::sync_channel::<RecordingPayload>(4);
    let path = db_path.to_owned();
    let handle = std::thread::Builder::new()
        .name("recording-writer".into())
        .spawn(move || {
            let db = match Connection::open(&path) {
                Ok(c) => c,
                Err(e) => {
                    log::error!("[recording-writer] failed to open DB: {e}");
                    return;
                }
            };
            let _ = db.execute_batch(
                "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;",
            );
            for payload in rx {
                if let Err(e) = db.execute(
                    "INSERT OR REPLACE INTO generation_recording \
                     (node_id, agent_count, tick_count, data) \
                     VALUES (?1, ?2, ?3, ?4)",
                    params![
                        payload.node_id,
                        payload.agent_count,
                        payload.tick_count,
                        payload.data
                    ],
                ) {
                    log::error!("[recording-writer] DB write failed: {e}");
                }
            }
        });
    match handle {
        Ok(h) => Some((tx, h)),
        Err(e) => {
            log::warn!("[recording-writer] failed to spawn thread: {e} — recording disabled");
            None
        }
    }
}

impl Governor {
    /// Global best fitness across all evaluated nodes in this run.
    pub fn best_score(&self) -> f32 {
        self.cached_best_score
    }

    /// Refresh the cached best score from the database.
    fn refresh_best_score(&mut self) {
        self.cached_best_score = self
            .db
            .query_row(
                "SELECT COALESCE(MAX(best_fitness), -1.0) FROM node WHERE run_id = ?1",
                params![self.run_id],
                |row| row.get::<_, f64>(0),
            )
            .unwrap_or(-1.0) as f32;
    }

    /// Active island's spawn parent (for tree display).
    pub fn spawn_parent_id(&self) -> Option<i64> {
        self.islands
            .get(self.active_island)
            .and_then(|i| i.spawn_parent_id)
    }

    /// Fitness of the active island's current spawn parent.
    /// This is the bar a new generation must beat (beat-the-parent model).
    /// Returns -1.0 for root nodes that haven't been evaluated yet.
    fn spawn_parent_fitness(&self) -> f32 {
        let spawn_id = match self
            .islands
            .get(self.active_island)
            .and_then(|i| i.spawn_parent_id)
        {
            Some(id) => id,
            None => return -1.0,
        };
        self.db
            .query_row(
                "SELECT COALESCE(best_fitness, -1.0) FROM node WHERE id = ?1",
                params![spawn_id],
                |row| row.get::<_, f64>(0),
            )
            .unwrap_or(-1.0) as f32
    }

    /// Get the spawn parent's BrainConfig for the active island.
    fn spawn_parent_config(&self) -> Option<BrainConfig> {
        let spawn_id = self.islands.get(self.active_island)?.spawn_parent_id?;
        let json: String = self
            .db
            .query_row(
                "SELECT config_json FROM node WHERE id = ?1",
                params![spawn_id],
                |row| row.get(0),
            )
            .ok()?;
        serde_json::from_str(&json).ok()
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
        db.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;",
        )?;
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
                elite_configs: Vec::new(),
                attempts: 0,
            })
            .collect();
        let momentums: Vec<MutationMomentum> = (0..num_islands)
            .map(|_| MutationMomentum::new(config.momentum_decay))
            .collect();

        // Skip background writer for in-memory DBs — each Connection::open(":memory:")
        // creates an independent database, so the writer thread would not see the schema.
        let (recording_sender, writer_thread) = if db_path == ":memory:" {
            (None, None)
        } else {
            match spawn_recording_writer(db_path) {
                Some((tx, h)) => (Some(tx), Some(h)),
                None => (None, None),
            }
        };

        Ok(Self {
            db,
            config,
            run_id,
            current_node_id: Some(root_id),
            generation: 0,
            gen_tick: 0,
            islands,
            active_island: 0,
            cached_best_score: -1.0,
            cached_tree_nodes: None,
            momentums,
            recording_sender,
            writer_thread,
        })
    }

    /// Resume from an existing database. Finds the latest active node
    /// for the most recent run.
    pub fn resume(db_path: &str) -> SqlResult<Self> {
        let db = Connection::open(db_path)?;
        db.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;",
        )?;

        // Run migrations for any new columns (idempotent — silently ignores duplicates)
        let _ = db.execute_batch("ALTER TABLE node ADD COLUMN island_id INTEGER;");

        let (run_id, governor_json, spawn_parent_id, momentum_json): (
            i64,
            String,
            Option<i64>,
            String,
        ) = db.query_row(
            "SELECT id, governor_config, spawn_parent_id,
                        COALESCE(momentum_json, '[]')
                 FROM run ORDER BY id DESC LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )?;

        let config: GovernorConfig = serde_json::from_str(&governor_json).unwrap_or_default();

        // Prefer an active node; fall back to the most recent node of any status
        // (covers the case where the app was stopped after evaluation but before breeding).
        let (node_id, generation): (i64, u32) = db
            .query_row(
                "SELECT id, generation FROM node
                 WHERE run_id = ?1 AND status = 'active'
                 ORDER BY generation DESC LIMIT 1",
                params![run_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .or_else(|_| {
                db.query_row(
                    "SELECT id, generation FROM node
                     WHERE run_id = ?1
                     ORDER BY generation DESC LIMIT 1",
                    params![run_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
            })?;

        // Reconstruct all islands from the persisted best state
        let num_islands = config.num_islands.max(1);
        let islands: Vec<Island> = (0..num_islands)
            .map(|_| Island {
                spawn_parent_id,
                elite_configs: Vec::new(),
                attempts: 0,
            })
            .collect();
        let mut momentums: Vec<MutationMomentum> =
            serde_json::from_str(&momentum_json).unwrap_or_default();
        // Ensure we have one momentum per island (handles old DBs or island count changes)
        while momentums.len() < num_islands {
            momentums.push(MutationMomentum::new(config.momentum_decay));
        }
        momentums.truncate(num_islands);

        // Skip background writer for in-memory DBs (see Governor::new for rationale).
        let (recording_sender, writer_thread) = if db_path == ":memory:" {
            (None, None)
        } else {
            match spawn_recording_writer(db_path) {
                Some((tx, h)) => (Some(tx), Some(h)),
                None => (None, None),
            }
        };

        let mut gov = Self {
            db,
            config,
            run_id,
            current_node_id: Some(node_id),
            generation,
            gen_tick: 0,
            islands,
            active_island: 0,
            cached_best_score: -1.0,
            cached_tree_nodes: None,
            momentums,
            recording_sender,
            writer_thread,
        };
        gov.refresh_best_score();
        Ok(gov)
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
                    config: a.brain_config.clone(),
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
            let config_json = serde_json::to_string(&r.config).unwrap_or_default();
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

        // Update node avg fitness (best_fitness is set by advance() after noise reduction)
        let avg =
            results.iter().map(|r| r.composite_fitness).sum::<f32>() / results.len().max(1) as f32;

        let _ = self.db.execute(
            "UPDATE node SET avg_fitness = ?1 WHERE id = ?2",
            params![avg, node_id],
        );

        results.sort_by(|a, b| {
            b.composite_fitness
                .partial_cmp(&a.composite_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Reduce per-agent fitness to per-config-group fitness by averaging.
    /// Groups agents by `agent_index / eval_repeats` and averages composite_fitness.
    /// Must restore spawn order first since evaluate() sorts by fitness.
    fn reduce_fitness(&self, fitness: &[AgentFitness]) -> Vec<AgentFitness> {
        let repeats = self.config.eval_repeats.max(1);
        // Restore original spawn order so chunks match same-config groups
        let mut ordered = fitness.to_vec();
        ordered.sort_by_key(|f| f.agent_index);
        let mut groups: Vec<AgentFitness> = Vec::new();
        for chunk in ordered.chunks(repeats) {
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

        // Use the population average (mean of all reduced group fitnesses) for both
        // the success comparison and the stored bar.  The original approach used the
        // best group's average, which suffers from "winner's curse": the max of N
        // noisy estimates is systematically inflated, so the next generation almost
        // never reproduces it.  The mean is unbiased and has much lower variance
        // (σ/√N vs order-statistic inflation), giving a fair cross-generation bar.
        let gen_avg = if reduced.is_empty() {
            0.0
        } else {
            reduced.iter().map(|f| f.composite_fitness).sum::<f32>() / reduced.len() as f32
        };

        // Look up the spawn parent's fitness and config — this is the bar to beat.
        // Capture both before the success/failure branch so they refer to the same node
        // (backtracking changes spawn_parent_id, which would cause a mismatch).
        let parent_fitness = self.spawn_parent_fitness();
        let parent_config_for_momentum = self.spawn_parent_config();

        let island = &mut self.islands[self.active_island];

        // Score this generation against its parent (not the all-time best).
        // Both sides use the same metric (population average) so a generation
        // can only succeed when its overall quality genuinely meets or exceeds
        // the parent's.
        if gen_avg >= parent_fitness {
            // ★ Success — this generation's average meets or exceeds its parent's
            let _ = self.db.execute(
                "UPDATE node SET status = 'successful', best_fitness = ?2 WHERE id = ?1",
                params![self.current_node_id, gen_avg as f64],
            );
            // Update node config to the best performer's config so future
            // champions (bred from this node) use a proven, tested config.
            if let Some(best) = reduced.first() {
                let best_json = serde_json::to_string(&best.config).unwrap_or_default();
                let _ = self.db.execute(
                    "UPDATE node SET config_json = ?1 WHERE id = ?2",
                    params![best_json, self.current_node_id],
                );
            }
            island.spawn_parent_id = self.current_node_id;
            // Store elite configs from this successful generation
            island.elite_configs = reduced
                .iter()
                .take(self.config.elitism_count)
                .map(|f| f.config.clone())
                .collect();
            island.attempts = 0;
            self.refresh_best_score();
            let global_best = self.best_score();
            if gen_avg >= global_best {
                messages.push(format!(
                    "[EVOLUTION] ★ New global best: {:.4} (peak: {:.4})",
                    gen_avg, gen_fit
                ));
            } else {
                messages.push(format!(
                    "[EVOLUTION] ✓ Beat parent ({:.4} → {:.4}, peak: {:.4})",
                    parent_fitness, gen_avg, gen_fit
                ));
            }
        } else {
            // ✗ Failure — didn't beat the parent
            let _ = self.db.execute(
                "UPDATE node SET status = 'failed', best_fitness = ?2 WHERE id = ?1",
                params![self.current_node_id, gen_avg as f64],
            );
            messages.push(format!(
                "[EVOLUTION] ✗ Failed ({:.4} < parent {:.4})",
                gen_avg, parent_fitness
            ));

            // Check if this island has exhausted its patience at current spawn parent
            let should_backtrack = island.attempts >= self.config.patience;
            if should_backtrack {
                self.backtrack_island(&mut messages);
            }
        }

        // ── Momentum update: learn from individual winners ──────────────
        // An individual "winner" is an offspring whose fitness >= spawn parent.
        // Even in a failed generation, strong individuals contribute directional data.
        let had_winners;
        if let Some(ref pc) = parent_config_for_momentum {
            let winner_configs: Vec<BrainConfig> = reduced
                .iter()
                .filter(|f| f.composite_fitness >= parent_fitness)
                .map(|f| f.config.clone())
                .collect();
            had_winners = !winner_configs.is_empty();
            self.momentums[self.active_island].update(pc, &winner_configs);
            self.momentums[self.active_island].decay_step();
        } else {
            had_winners = false;
        }

        // Log momentum trends only when there were winners (avoids noise)
        if had_winners {
            let top = self.momentums[self.active_island].top_params(3);
            if !top.is_empty() {
                let trends: Vec<String> = top
                    .iter()
                    .filter(|(_, val)| val.abs() > 0.001) // Only show meaningful trends
                    .map(|(name, val)| {
                        let dir = if *val > 0.0 { "↑" } else { "↓" };
                        let short = match *name {
                            "memory_capacity" => "mem",
                            "processing_slots" => "slots",
                            "representation_dim" => "repr",
                            "learning_rate" => "lr",
                            "decay_rate" => "decay",
                            "distress_exponent" => "distress",
                            "habituation_sensitivity" => "hab",
                            "max_curiosity_bonus" => "curiosity",
                            "fatigue_recovery_sensitivity" => "fat_rec",
                            "fatigue_floor" => "fat_fl",
                            other => other,
                        };
                        format!("{}{}", short, dir)
                    })
                    .collect();
                if !trends.is_empty() {
                    messages.push(format!(
                        "[EVOLUTION] Momentum trending: {}",
                        trends.join(", ")
                    ));
                }
            }
        }

        // Check completion after scoring but before breeding
        if self.evolution_complete() {
            self.refresh_best_score();
            self.cached_tree_nodes = None;
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

        // Compute effective mutation strength for neuroevolution weight perturbation.
        // This mirrors the adaptive strength calculation inside breed_next_generation().
        let effective_strength = {
            let attempts = self.islands[self.active_island].attempts;
            let base = self.config.mutation_strength;
            let max_strength = 0.5_f32;
            let patience = self.config.patience.max(1) as f32;
            base + (max_strength - base) * (attempts as f32 / patience).min(1.0)
        };

        // Breed the next generation from the (now-rotated) island's spawn parent
        let configs = self.breed_next_generation(fitness);

        // Migration: every migration_interval generations, spread best config
        if self.config.migration_interval > 0
            && self.generation % self.config.migration_interval as u32 == 0
            && self.islands.len() > 1
        {
            // Find island with highest spawn parent fitness
            let best_island_idx = self
                .islands
                .iter()
                .enumerate()
                .map(|(i, island)| {
                    let fit = island.spawn_parent_id.map_or(-1.0, |id| {
                        self.db
                            .query_row(
                                "SELECT COALESCE(best_fitness, -1.0) FROM node WHERE id = ?1",
                                params![id],
                                |row| row.get::<_, f64>(0),
                            )
                            .unwrap_or(-1.0)
                    });
                    (i, fit)
                })
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            if let Some(best_elite) = self.islands[best_island_idx].elite_configs.first().cloned() {
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

        // Invalidate caches after DB changes
        self.refresh_best_score();
        self.cached_tree_nodes = None;

        self.persist_state();

        AdvanceResult::Continue {
            configs,
            messages,
            mutation_strength: effective_strength,
        }
    }

    /// Backtrack one level when the island exhausts patience at its current
    /// spawn parent. Marks the node as exhausted (for tree display) and moves
    /// the island to the parent node with a fresh patience budget.
    ///
    /// With beat-the-parent scoring, backtracking naturally lowers the bar
    /// to the parent's fitness, enabling exploration of alternative branches.
    /// At root, patience simply resets — no artificial score rollback needed.
    fn backtrack_island(&mut self, messages: &mut Vec<String>) {
        let island = &mut self.islands[self.active_island];
        let spawn_id = match island.spawn_parent_id {
            Some(id) => id,
            None => return,
        };

        // Mark the exhausted node in the DB (informational for tree display)
        let _ = self.db.execute(
            "UPDATE node SET status = 'exhausted' WHERE id = ?1",
            params![spawn_id],
        );

        let parent_id: Option<i64> = self
            .db
            .query_row(
                "SELECT parent_id FROM node WHERE id = ?1",
                params![spawn_id],
                |row| row.get(0),
            )
            .unwrap_or(None);

        match parent_id {
            Some(pid) => {
                // Non-root: move up one level, fresh patience budget.
                // The bar automatically lowers to pid's fitness.
                island.spawn_parent_id = Some(pid);
                island.attempts = 0;
                messages.push(format!(
                    "[EVOLUTION] Island {} exhausted — backtracking",
                    self.active_island
                ));
            }
            None => {
                // Root exhaustion: reset patience and keep exploring from root.
                // Since we compare against the parent (root) fitness, no rollback needed.
                island.attempts = 0;
                messages.push(format!(
                    "[EVOLUTION] Island {} root exhausted — resetting patience",
                    self.active_island
                ));
            }
        }
    }

    /// Create the next generation node as a child of `spawn_parent_id`.
    /// Increments the parent's spawn_attempts and returns population configs.
    /// Uses elitism to preserve top configs and adaptive mutation strength.
    fn breed_next_generation(&mut self, _fitness: &[AgentFitness]) -> Vec<BrainConfig> {
        let spawn_parent = match self.islands[self.active_island].spawn_parent_id {
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
        let attempts = self.islands[self.active_island].attempts;
        let base = self.config.mutation_strength;
        let max_strength = 0.5_f32;
        let patience = self.config.patience.max(1) as f32;
        let effective_strength =
            base + (max_strength - base) * (attempts as f32 / patience).min(1.0);

        let momentum = &self.momentums[self.active_island];
        let base_config = mutate_config_with_strength(&parent_config, effective_strength, momentum);

        self.generation += 1;
        // Store the champion config (parent_config) as the node's config.
        // base_config is only used for mutation records; parent_config is the
        // actual champion that will be evaluated, ensuring future spawn parents
        // reference a tested config rather than an untested random mutant.
        let config_json = serde_json::to_string(&parent_config).unwrap_or_default();
        let _ = self.db.execute(
            "INSERT INTO node (run_id, parent_id, generation, config_json, status, island_id)
             VALUES (?1, ?2, ?3, ?4, 'active', ?5)",
            params![
                self.run_id,
                Some(spawn_parent),
                self.generation,
                config_json,
                self.active_island as i64
            ],
        );
        let new_node_id = self.db.last_insert_rowid();
        self.current_node_id = Some(new_node_id);
        self.gen_tick = 0;

        // Increment per-island attempts (authoritative for patience tracking)
        self.islands[self.active_island].attempts += 1;

        // Also increment DB spawn_attempts for display/history
        let _ = self.db.execute(
            "UPDATE node SET spawn_attempts = spawn_attempts + 1 WHERE id = ?1",
            params![spawn_parent],
        );

        // Record which parameters were mutated
        let parent_fitness = self.spawn_parent_fitness();
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
        let elite_count = self
            .config
            .elitism_count
            .min(unique_count.saturating_sub(1));

        // Population slot 0: unmutated champion (the spawn parent's exact config)
        let mut unique_configs = vec![parent_config.clone()];

        // Elitism: mutate top configs from the last SUCCESSFUL generation
        let island = &self.islands[self.active_island];
        for elite in island.elite_configs.iter().take(elite_count) {
            if unique_configs.len() >= unique_count {
                break;
            }
            unique_configs.push(mutate_config_with_strength(
                elite,
                effective_strength,
                momentum,
            ));
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
                let child =
                    crossover_config(&island.elite_configs[idx_a], &island.elite_configs[idx_b]);
                unique_configs.push(mutate_config_with_strength(
                    &child,
                    effective_strength,
                    momentum,
                ));
            }
        }

        // Fill remaining unique slots from the spawn parent's config
        while unique_configs.len() < unique_count {
            unique_configs.push(mutate_config_with_strength(
                &parent_config,
                effective_strength,
                momentum,
            ));
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

    /// Persist best_score, spawn_parent_id, and momentum for resume.
    fn persist_state(&self) {
        let best_score = self.best_score();
        let spawn_parent = self
            .islands
            .get(self.active_island)
            .and_then(|i| i.spawn_parent_id);
        let momentum_json = serde_json::to_string(&self.momentums).unwrap_or_else(|_| "[]".into());
        let _ = self.db.execute(
            "UPDATE run SET best_score = ?1, spawn_parent_id = ?2, momentum_json = ?3 WHERE id = ?4",
            params![best_score as f64, spawn_parent, momentum_json, self.run_id],
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

    /// Serialize a generation's recording (synchronous) and send the
    /// resulting blob to the background writer thread for asynchronous
    /// SQLite persistence.  The channel send is non-blocking — if the
    /// channel is full a warning is logged and the recording is dropped.
    pub fn store_recording(&self, recording: &crate::replay::GenerationRecording) {
        let sender = match self.recording_sender.as_ref() {
            Some(s) => s,
            None => return,
        };
        let node_id = match self.current_node_id {
            Some(id) => id,
            None => return,
        };
        let agent_count = match i64::try_from(recording.agent_count) {
            Ok(v) => v,
            Err(_) => return,
        };
        let tick_count = match i64::try_from(recording.total_ticks) {
            Ok(v) if v > 0 => v,
            _ => return,
        };

        // Serialize TickRecords into a compact little-endian binary
        // format: 14 f32 fields per agent per tick (position[3], yaw,
        // energy, integrity, alive, motor_fwd, motor_turn,
        // prediction_error, exploration_rate, gradient, urgency,
        // fatigue_factor).
        let record_stride = 14usize;
        let tick_count_usize = match usize::try_from(tick_count) {
            Ok(v) => v,
            Err(_) => return,
        };
        let expected_floats = match tick_count_usize
            .checked_mul(recording.agent_count)
            .and_then(|v| v.checked_mul(record_stride))
        {
            Some(v) => v,
            None => return,
        };
        let mut bytes = Vec::with_capacity(expected_floats * std::mem::size_of::<f32>());
        for tick in 0..recording.total_ticks {
            let records = match recording.get_tick(tick) {
                Some(records) => records,
                None => return,
            };
            if records.len() != recording.agent_count {
                return;
            }
            for r in records {
                for &val in &[
                    r.position[0],
                    r.position[1],
                    r.position[2],
                    r.yaw,
                    r.energy,
                    r.integrity,
                    if r.alive { 1.0f32 } else { 0.0 },
                    r.motor_forward,
                    r.motor_turn,
                    r.prediction_error,
                    r.exploration_rate,
                    r.gradient,
                    r.urgency,
                    r.fatigue_factor,
                ] {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
            }
        }

        let float_size = std::mem::size_of::<f32>();
        if bytes.len() != expected_floats * float_size {
            return;
        }

        let payload = RecordingPayload {
            node_id,
            agent_count,
            tick_count,
            data: bytes,
        };

        if let Err(mpsc::TrySendError::Full(_)) = sender.try_send(payload) {
            log::warn!("[governor] recording channel full — dropping recording for node {node_id}");
        }
        // TrySendError::Disconnected is silently ignored (writer thread died).
    }

    /// Load a generation's recording from the database.
    /// Returns `(agent_count, tick_count, Vec<f32>)` or `None` if not
    /// found or the blob is malformed.
    pub fn load_recording(&self, node_id: i64) -> Option<(usize, u64, Vec<f32>)> {
        let row: (i64, i64, Vec<u8>) = self
            .db
            .query_row(
                "SELECT agent_count, tick_count, data \
                 FROM generation_recording WHERE node_id = ?1",
                params![node_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .ok()?;
        let (agent_count, tick_count, blob) = row;
        let agent_count = usize::try_from(agent_count).ok()?;
        let tick_count = usize::try_from(tick_count).ok()?;

        let float_size = std::mem::size_of::<f32>();
        if blob.len() % float_size != 0 {
            return None;
        }

        // Keep record_stride in sync with store_recording().
        let record_stride = 14usize;
        let actual_float_count = blob.len() / float_size;
        let expected_float_count = agent_count
            .checked_mul(tick_count)?
            .checked_mul(record_stride)?;
        if actual_float_count != expected_float_count {
            return None;
        }

        let mut floats = Vec::with_capacity(actual_float_count);
        for chunk in blob.chunks_exact(float_size) {
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            floats.push(f32::from_le_bytes(bytes));
        }

        Some((agent_count, tick_count as u64, floats))
    }

    /// Get the evolution tree as a list of nodes for UI visualization.
    /// Returns a cached copy if the tree hasn't changed since last call.
    pub fn tree_nodes(&mut self) -> Vec<TreeNode> {
        if let Some(cached) = &self.cached_tree_nodes {
            return cached.clone();
        }
        let nodes = self.tree_nodes_from_db();
        self.cached_tree_nodes = Some(nodes.clone());
        nodes
    }

    /// Fetch tree nodes directly from the database.
    fn tree_nodes_from_db(&self) -> Vec<TreeNode> {
        let mut stmt = match self.db.prepare(
            "SELECT id, parent_id, generation, best_fitness, avg_fitness,
                    config_json, status, island_id
             FROM node WHERE run_id = ?1 ORDER BY id",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let mut nodes: Vec<TreeNode> = match stmt.query_map(params![self.run_id], |row| {
            let config_json: String = row.get(5)?;
            let config: Option<BrainConfig> = serde_json::from_str(&config_json).ok();
            Ok(TreeNode {
                id: row.get(0)?,
                parent_id: row.get(1)?,
                generation: row.get(2)?,
                best_fitness: row.get::<_, Option<f64>>(3)?.map(|v| v as f32),
                avg_fitness: row.get::<_, Option<f64>>(4)?.map(|v| v as f32),
                status: row.get(6)?,
                island_id: row.get(7)?,
                config,
                mutations: Vec::new(),
            })
        }) {
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

    /// Per-island fitness history: island_id → Vec<(generation, best_fitness, avg_fitness)>.
    /// Nodes without an island_id (pre-migration data) are grouped under key -1.
    pub fn fitness_history_by_island(
        &self,
    ) -> std::collections::HashMap<i64, Vec<(u32, f32, f32)>> {
        let mut stmt = match self.db.prepare(
            "SELECT island_id, generation, best_fitness, avg_fitness FROM node
             WHERE run_id = ?1 AND best_fitness IS NOT NULL
             ORDER BY island_id, generation ASC",
        ) {
            Ok(s) => s,
            Err(_) => return std::collections::HashMap::new(),
        };
        let mut map: std::collections::HashMap<i64, Vec<(u32, f32, f32)>> =
            std::collections::HashMap::new();
        if let Ok(rows) = stmt.query_map(params![self.run_id], |row| {
            Ok((
                row.get::<_, Option<i64>>(0)?,
                row.get::<_, u32>(1)?,
                row.get::<_, f32>(2)?,
                row.get::<_, Option<f32>>(3)?.unwrap_or(0.0),
            ))
        }) {
            for r in rows.flatten() {
                let (island_id_opt, gen, best, avg) = r;
                let key = island_id_opt.unwrap_or(-1);
                map.entry(key).or_default().push((gen, best, avg));
            }
        }
        map
    }
}

impl Drop for Governor {
    fn drop(&mut self) {
        // Drop the sender first so the writer thread's recv loop exits.
        drop(self.recording_sender.take());
        if let Some(handle) = self.writer_thread.take() {
            let _ = handle.join();
        }
    }
}

/// Check whether a database file has an existing evolution session.
/// Returns Some((generation, GovernorConfig, BrainConfig)) if a session exists.
pub fn check_existing_session(db_path: &str) -> Option<(u32, GovernorConfig, BrainConfig)> {
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
    let gov_config: GovernorConfig = serde_json::from_str(&governor_json).unwrap_or_default();
    let brain_config: BrainConfig = serde_json::from_str(&brain_json).unwrap_or_default();
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
    pub island_id: Option<i64>,
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
            island_id INTEGER,
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
    )?;

    // Backwards-compatible migration: add momentum_json if missing
    let _ = db.execute_batch("ALTER TABLE run ADD COLUMN momentum_json TEXT DEFAULT '[]';");

    // Backwards-compatible migration: add island_id if missing
    let _ = db.execute_batch("ALTER TABLE node ADD COLUMN island_id INTEGER;");

    // Recording persistence: one little-endian f32 BLOB per generation
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS generation_recording (
            node_id     INTEGER PRIMARY KEY REFERENCES node(id),
            agent_count INTEGER NOT NULL,
            tick_count  INTEGER NOT NULL,
            data        BLOB NOT NULL
        );",
    )?;

    Ok(())
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
        (
            "distress_exponent",
            parent.distress_exponent as f64,
            child.distress_exponent as f64,
        ),
        (
            "habituation_sensitivity",
            parent.habituation_sensitivity as f64,
            child.habituation_sensitivity as f64,
        ),
        (
            "max_curiosity_bonus",
            parent.max_curiosity_bonus as f64,
            child.max_curiosity_bonus as f64,
        ),
        (
            "fatigue_recovery_sensitivity",
            parent.fatigue_recovery_sensitivity as f64,
            child.fatigue_recovery_sensitivity as f64,
        ),
        (
            "fatigue_floor",
            parent.fatigue_floor as f64,
            child.fatigue_floor as f64,
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
            momentum_decay: 0.9,
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
    fn root_exhaustion_resets_patience() {
        let mut gov = test_governor(2);

        // Gen 0: success — beats root (root has no fitness = -1.0)
        gov.advance(&mock_fitness(0.1));
        let root_id = node_parent(&gov, gov.current_node_id.unwrap()).unwrap();

        // Gen 1: success (beats gen0's 0.1)
        gov.advance(&mock_fitness(0.2));
        let gen1_id = gov.spawn_parent_id().unwrap();

        // Gen 2, Gen 3: fail → gen1 exhausted → backtrack to root (fresh patience)
        gov.advance(&mock_fitness(0.15));
        gov.advance(&mock_fitness(0.15));
        assert_eq!(node_status(&gov, gen1_id), "exhausted");
        assert_eq!(gov.spawn_parent_id(), Some(root_id));

        // Now at root with fresh patience budget.
        // Gen 4: fail at root (0.05 < root's 0.1)
        gov.advance(&mock_fitness(0.05));
        // Gen 5: fail at root → root exhausted, patience resets (no rollback)
        gov.advance(&mock_fitness(0.05));
        // Global best is still 0.2 (from gen1, stored in DB)
        assert!((gov.best_score() - 0.2).abs() < 0.001);
        // Still at root
        assert_eq!(gov.spawn_parent_id(), Some(root_id));
    }

    #[test]
    fn backtrack_lowers_bar_to_parent() {
        let mut gov = test_governor(1);

        // Gen 0: success — beats root (-1.0), root's fitness becomes 0.1
        gov.advance(&mock_fitness(0.1));
        let root_id = node_parent(&gov, gov.current_node_id.unwrap()).unwrap();

        // Gen 1: success (0.2 >= gen0's 0.1)
        gov.advance(&mock_fitness(0.2));

        // Gen 2: fail (0.15 < gen1's 0.2) → gen1 exhausted (patience=1), backtrack to root
        gov.advance(&mock_fitness(0.15));
        assert_eq!(gov.spawn_parent_id(), Some(root_id));

        // Gen 3: now only needs to beat root's fitness (0.1), NOT the global best (0.2)
        // 0.12 >= 0.1 → SUCCESS under beat-the-parent model
        gov.advance(&mock_fitness(0.12));
        assert_ne!(gov.spawn_parent_id(), Some(root_id)); // moved past root
                                                          // Global best is still 0.2
        assert!((gov.best_score() - 0.2).abs() < 0.001);
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
        // Validates: >= success, beat-the-parent, tie acceptance, backtracking
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
            momentum_decay: 0.9,
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
            momentum_decay: 0.9,
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

        // Reset momentum so it doesn't bias the elitism check below
        gov.momentums[gov.active_island] = crate::momentum::MutationMomentum::new(0.9);

        // breed_next_generation uses stored elite_configs from the successful gen
        let configs = gov.breed_next_generation(&mock_fitness(0.01));
        assert_eq!(configs.len(), 5);

        // With high patience and low mutation_strength, elite configs should
        // stay recognizable by their magnitude (5000 and 3000 vs default 128)
        let has_elite1_like = configs.iter().any(|c| c.memory_capacity > 4000);
        let has_elite2_like = configs
            .iter()
            .any(|c| (2000..=4000).contains(&c.memory_capacity));
        assert!(
            has_elite1_like,
            "Expected a config near memory_capacity=5000 from elite1; got {:?}",
            configs
                .iter()
                .map(|c| c.memory_capacity)
                .collect::<Vec<_>>()
        );
        assert!(
            has_elite2_like,
            "Expected a config near memory_capacity=3000 from elite2; got {:?}",
            configs
                .iter()
                .map(|c| c.memory_capacity)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn champion_is_unmutated() {
        let mut gov = test_governor(5);

        // Gen 0: success — root becomes spawn parent
        gov.advance(&mock_fitness(0.1));
        let spawn_id = gov.spawn_parent_id().unwrap();

        // Get the spawn parent's config from the DB (what breed uses)
        let parent_json: String = gov
            .db
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
        assert_eq!(
            configs[0].representation_dim,
            parent_config.representation_dim
        );
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
            momentum_decay: 0.9,
        };
        let brain = BrainConfig::default();
        let gov = Governor::new(":memory:", config, &brain, "{}").unwrap();

        // 4 agents, eval_repeats=2 → 2 groups of 2
        let fitness = vec![
            AgentFitness {
                agent_index: 0,
                config: BrainConfig {
                    memory_capacity: 100,
                    ..BrainConfig::default()
                },
                total_ticks_alive: 100,
                death_count: 0,
                food_consumed: 10,
                cells_explored: 50,
                composite_fitness: 0.8,
            },
            AgentFitness {
                agent_index: 1,
                config: BrainConfig {
                    memory_capacity: 100,
                    ..BrainConfig::default()
                },
                total_ticks_alive: 100,
                death_count: 0,
                food_consumed: 10,
                cells_explored: 50,
                composite_fitness: 0.6,
            },
            AgentFitness {
                agent_index: 2,
                config: BrainConfig {
                    memory_capacity: 200,
                    ..BrainConfig::default()
                },
                total_ticks_alive: 100,
                death_count: 0,
                food_consumed: 10,
                cells_explored: 50,
                composite_fitness: 0.5,
            },
            AgentFitness {
                agent_index: 3,
                config: BrainConfig {
                    memory_capacity: 200,
                    ..BrainConfig::default()
                },
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
    fn noise_reduction_handles_sorted_input() {
        // evaluate() returns results sorted by fitness descending.
        // reduce_fitness must restore spawn order before chunking.
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
            momentum_decay: 0.9,
        };
        let brain = BrainConfig::default();
        let gov = Governor::new(":memory:", config, &brain, "{}").unwrap();

        // Config A: agents 0,1 → fitness 0.3, 0.5 → correct avg = 0.4
        // Config B: agents 2,3 → fitness 0.9, 0.1 → correct avg = 0.5
        // Sorted by fitness descending (as evaluate returns):
        let fitness = vec![
            AgentFitness {
                agent_index: 2,
                composite_fitness: 0.9,
                config: BrainConfig {
                    memory_capacity: 200,
                    ..BrainConfig::default()
                },
                total_ticks_alive: 100,
                death_count: 0,
                food_consumed: 10,
                cells_explored: 50,
            },
            AgentFitness {
                agent_index: 1,
                composite_fitness: 0.5,
                config: BrainConfig {
                    memory_capacity: 100,
                    ..BrainConfig::default()
                },
                total_ticks_alive: 100,
                death_count: 0,
                food_consumed: 10,
                cells_explored: 50,
            },
            AgentFitness {
                agent_index: 0,
                composite_fitness: 0.3,
                config: BrainConfig {
                    memory_capacity: 100,
                    ..BrainConfig::default()
                },
                total_ticks_alive: 100,
                death_count: 0,
                food_consumed: 10,
                cells_explored: 50,
            },
            AgentFitness {
                agent_index: 3,
                composite_fitness: 0.1,
                config: BrainConfig {
                    memory_capacity: 200,
                    ..BrainConfig::default()
                },
                total_ticks_alive: 100,
                death_count: 0,
                food_consumed: 10,
                cells_explored: 50,
            },
        ];

        let reduced = gov.reduce_fitness(&fitness);
        assert_eq!(reduced.len(), 2);
        // Without the sort fix, chunks would be (agent2=0.9, agent1=0.5)=0.7
        // and (agent0=0.3, agent3=0.1)=0.2 — wrong pairing!
        // With the fix: (agent0=0.3, agent1=0.5)=0.4, (agent2=0.9, agent3=0.1)=0.5
        let best = reduced[0].composite_fitness;
        let second = reduced[1].composite_fitness;
        assert!((best - 0.5).abs() < 0.01, "expected ~0.5, got {}", best);
        assert!((second - 0.4).abs() < 0.01, "expected ~0.4, got {}", second);
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
            momentum_decay: 0.9,
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
    fn per_island_attempts_are_isolated() {
        // With 3 islands, each island's patience counter must be independent.
        // Previously, shared DB spawn_attempts caused cross-contamination:
        // all islands incrementing root's spawn_attempts meant root exhausted
        // after patience/num_islands rounds instead of patience rounds per island.
        let config = GovernorConfig {
            population_size: 4,
            tick_budget: 100,
            elitism_count: 1,
            patience: 3,
            max_generations: 0,
            mutation_strength: 0.1,
            eval_repeats: 1,
            num_islands: 3,
            migration_interval: 0,
            momentum_decay: 0.9,
        };
        let brain = BrainConfig::default();
        let mut gov = Governor::new(":memory:", config, &brain, "{}").unwrap();
        let root_id = gov.current_node_id.unwrap();

        // Gen 0-2: all islands succeed (move off root)
        gov.advance(&mock_fitness(0.1));
        gov.advance(&mock_fitness(0.1));
        gov.advance(&mock_fitness(0.1));

        // 3 rounds: island 0 always fails, islands 1 and 2 always succeed
        for _ in 0..3 {
            gov.advance(&mock_fitness(0.05)); // island 0: fail
            gov.advance(&mock_fitness(0.2)); // island 1: success
            gov.advance(&mock_fitness(0.2)); // island 2: success
        }

        // Island 0 exhausted at root and went through root exhaustion,
        // but islands 1 and 2 should NOT be affected — they should still
        // be at their most recent successful nodes, not stuck at root.
        assert_eq!(gov.islands[0].spawn_parent_id, Some(root_id));
        assert_ne!(gov.islands[1].spawn_parent_id, Some(root_id));
        assert_ne!(gov.islands[2].spawn_parent_id, Some(root_id));
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
            distress_exponent: 2.0,
            ..BrainConfig::default()
        };
        let b = BrainConfig {
            memory_capacity: 1,
            processing_slots: 1,
            visual_encoding_size: 64,
            representation_dim: 1,
            learning_rate: 0.001,
            decay_rate: 0.001,
            distress_exponent: 2.0,
            ..BrainConfig::default()
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

    // ─── gen_avg comparison tests ───────────────────────────────────

    /// Helper: create multi-agent fitness with varying scores.
    fn mock_multi_fitness(scores: &[(usize, f32)]) -> Vec<AgentFitness> {
        scores
            .iter()
            .map(|&(idx, fit)| AgentFitness {
                agent_index: idx,
                config: BrainConfig::default(),
                total_ticks_alive: 100,
                death_count: 0,
                food_consumed: 10,
                cells_explored: 50,
                composite_fitness: fit,
            })
            .collect()
    }

    #[test]
    fn gen_avg_determines_success_not_best_group() {
        // With eval_repeats=1, gen_avg = mean of all agents.
        // One high-scoring agent shouldn't carry a low-avg generation.
        let mut gov = test_governor(5);

        // Gen 0: avg = 0.3, succeeds (root = -1.0)
        let gen0_fitness = mock_multi_fitness(&[(0, 0.3), (1, 0.3), (2, 0.3), (3, 0.3)]);
        gov.advance(&gen0_fitness);
        // Root best_fitness is now the avg = 0.3
        let root_bar: f64 = gov
            .db
            .query_row(
                "SELECT best_fitness FROM node WHERE id = ?1",
                params![gov.spawn_parent_id().unwrap()],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            (root_bar - 0.3).abs() < 0.01,
            "Bar should be gen_avg (0.3), got {root_bar}"
        );

        // Gen 1: best agent = 0.5 but avg = 0.25 → should FAIL (0.25 < 0.3)
        let gen1_fitness = mock_multi_fitness(&[(0, 0.5), (1, 0.1), (2, 0.2), (3, 0.2)]);
        let gen1_id = gov.current_node_id.unwrap();
        gov.advance(&gen1_fitness);
        assert_eq!(
            node_status(&gov, gen1_id),
            "failed",
            "High best but low avg should fail"
        );
    }

    #[test]
    fn successful_node_config_updates_to_best_performer() {
        let mut gov = test_governor(5);

        // Create a distinctive "elite" config
        let elite_config = BrainConfig {
            memory_capacity: 999,
            ..BrainConfig::default()
        };
        let gen0_fitness = vec![AgentFitness {
            agent_index: 0,
            config: elite_config.clone(),
            total_ticks_alive: 100,
            death_count: 0,
            food_consumed: 50,
            cells_explored: 100,
            composite_fitness: 0.5,
        }];

        // Gen 0: succeeds — node config should update to the best performer
        gov.advance(&gen0_fitness);
        let root_id = gov.spawn_parent_id().unwrap();

        let stored_json: String = gov
            .db
            .query_row(
                "SELECT config_json FROM node WHERE id = ?1",
                params![root_id],
                |row| row.get(0),
            )
            .unwrap();
        let stored_config: BrainConfig = serde_json::from_str(&stored_json).unwrap();

        assert_eq!(
            stored_config.memory_capacity, 999,
            "Successful node should store best performer's config, got mem={}",
            stored_config.memory_capacity
        );
    }

    #[test]
    fn breed_respects_eval_repeats_grouping() {
        // With eval_repeats=2, breed should produce pairs of identical configs.
        let config = GovernorConfig {
            population_size: 6,
            tick_budget: 100,
            elitism_count: 1,
            patience: 3,
            max_generations: 0,
            mutation_strength: 0.1,
            eval_repeats: 2,
            num_islands: 1,
            migration_interval: 0,
            momentum_decay: 0.9,
        };
        let brain = BrainConfig::default();
        let mut gov = Governor::new(":memory:", config, &brain, "{}").unwrap();

        // Gen 0: succeed to enable breeding
        gov.advance(&mock_fitness(0.5));

        let configs = gov.breed_next_generation(&mock_fitness(0.1));
        // 6 agents / 2 repeats = 3 unique configs, each repeated twice
        assert_eq!(configs.len(), 6);

        // Agents 0,1 should be identical; 2,3 identical; 4,5 identical
        for pair_start in (0..6).step_by(2) {
            assert_eq!(
                configs[pair_start].memory_capacity,
                configs[pair_start + 1].memory_capacity,
                "Agents {} and {} should share config",
                pair_start,
                pair_start + 1
            );
            assert!(
                (configs[pair_start].learning_rate - configs[pair_start + 1].learning_rate).abs()
                    < f32::EPSILON,
                "Agents {} and {} should share config",
                pair_start,
                pair_start + 1
            );
        }
    }

    #[test]
    fn momentum_persists_across_resume() {
        use std::fs;

        let db_path = "/tmp/xagent_test_momentum_persist.db";
        let _ = fs::remove_file(db_path);

        let config = GovernorConfig {
            population_size: 10,
            tick_budget: 100,
            elitism_count: 3,
            patience: 5,
            max_generations: 0,
            mutation_strength: 0.1,
            eval_repeats: 1,
            num_islands: 2,
            migration_interval: 0,
            momentum_decay: 0.9,
        };
        let brain = BrainConfig::default();

        // Create governor, inject momentum, advance (which persists)
        {
            let mut gov = Governor::new(db_path, config.clone(), &brain, "{}").unwrap();
            gov.momentums[0]
                .momentum_mut()
                .insert("learning_rate".into(), 0.05);
            gov.momentums[1]
                .momentum_mut()
                .insert("decay_rate".into(), -0.03);
            gov.advance(&mock_fitness(0.1)); // triggers persist_state
        }

        // Resume and verify momentum survived
        {
            let gov = Governor::resume(db_path).unwrap();
            assert_eq!(gov.momentums.len(), 2);
            // Momentum was persisted after advance, which calls decay_step.
            // The injected values were also blended with winner data during advance.
            // Key assertion: momentum vectors exist and were deserialized (not empty defaults).
            let has_data = gov.momentums[0].get("learning_rate") != 0.0
                || gov.momentums[1].get("decay_rate") != 0.0;
            assert!(has_data, "momentum should have been persisted and restored");
        }

        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn governor_initializes_momentum_per_island() {
        let config = GovernorConfig {
            population_size: 10,
            tick_budget: 100,
            elitism_count: 3,
            patience: 5,
            max_generations: 0,
            mutation_strength: 0.1,
            eval_repeats: 1,
            num_islands: 3,
            migration_interval: 0,
            momentum_decay: 0.9,
        };
        let brain = BrainConfig::default();
        let gov = Governor::new(":memory:", config, &brain, "{}").unwrap();
        assert_eq!(gov.momentums.len(), 3);
    }

    #[test]
    fn breed_stores_island_id_on_node() {
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
            momentum_decay: 0.9,
        };
        let brain = BrainConfig::default();
        let mut gov = Governor::new(":memory:", config, &brain, "{}").unwrap();
        let root_id = gov.current_node_id.unwrap();

        // Root node should have island_id NULL (created before islands assigned)
        let root_island: Option<i64> = gov
            .db
            .query_row(
                "SELECT island_id FROM node WHERE id = ?1",
                params![root_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(root_island, None);

        // Advance 3 times — one generation per island (round-robin: 0, 1, 2)
        gov.advance(&mock_fitness(0.1)); // island 0 evaluates gen 0, breeds gen 1 for island 1
        gov.advance(&mock_fitness(0.1)); // island 1 evaluates gen 1, breeds gen 2 for island 2
        gov.advance(&mock_fitness(0.1)); // island 2 evaluates gen 2, breeds gen 3 for island 0

        // Check that bred nodes have island_id set
        let mut stmt = gov
            .db
            .prepare("SELECT id, island_id FROM node WHERE run_id = ?1 AND id != ?2 ORDER BY id")
            .unwrap();
        let rows: Vec<(i64, Option<i64>)> = stmt
            .query_map(params![gov.run_id, root_id], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap()
            .flatten()
            .collect();

        // Each bred node should have a non-null island_id
        assert!(!rows.is_empty());
        for (node_id, island_id) in &rows {
            assert!(
                island_id.is_some(),
                "node {} should have island_id set",
                node_id
            );
        }

        // Three advances over 3 islands: should see each of 0, 1, 2 represented
        let ids: Vec<i64> = rows
            .iter()
            .map(|(_, island_id)| island_id.unwrap())
            .collect();
        assert!(ids.contains(&0), "should have island 0");
        assert!(ids.contains(&1), "should have island 1");
        assert!(ids.contains(&2), "should have island 2");
    }

    #[test]
    fn fitness_history_by_island_returns_per_island_data() {
        let config = GovernorConfig {
            population_size: 4,
            tick_budget: 100,
            elitism_count: 1,
            patience: 5,
            max_generations: 0,
            mutation_strength: 0.1,
            eval_repeats: 1,
            num_islands: 2,
            migration_interval: 0,
            momentum_decay: 0.9,
        };
        let brain = BrainConfig::default();
        let mut gov = Governor::new(":memory:", config, &brain, "{}").unwrap();

        // Root (island_id=NULL, gen 0): fitness 0.1 — not counted per-island
        gov.advance(&mock_fitness(0.1));
        // Island 1, gen 1: fitness 0.2
        gov.advance(&mock_fitness(0.2));
        // Island 0, gen 2: fitness 0.3
        gov.advance(&mock_fitness(0.3));
        // Island 1, gen 3: fitness 0.4
        gov.advance(&mock_fitness(0.4));
        // Island 0, gen 4: fitness 0.5
        gov.advance(&mock_fitness(0.5));

        let history = gov.fitness_history_by_island();
        assert!(history.contains_key(&0), "should have island 0");
        assert!(history.contains_key(&1), "should have island 1");

        let island_0 = &history[&0];
        let island_1 = &history[&1];

        // Island 0: gens 2 and 4 evaluated (2 entries)
        // Island 1: gens 1 and 3 evaluated (2 entries)
        // Root (gen 0) excluded — has no island_id
        assert_eq!(island_0.len(), 2, "island 0 should have 2 data points");
        assert_eq!(island_1.len(), 2, "island 1 should have 2 data points");
    }

    #[test]
    fn fitness_history_by_island_tracks_failed_branches() {
        let config = GovernorConfig {
            population_size: 4,
            tick_budget: 100,
            elitism_count: 1,
            patience: 2,
            max_generations: 0,
            mutation_strength: 0.1,
            eval_repeats: 1,
            num_islands: 2,
            migration_interval: 0,
            momentum_decay: 0.9,
        };
        let brain = BrainConfig::default();
        let mut gov = Governor::new(":memory:", config, &brain, "{}").unwrap();

        // Gen 0 (island 0): success (root, no island_id — excluded from per-island map)
        gov.advance(&mock_fitness(0.5));
        // Gen 1 (island 1): success
        gov.advance(&mock_fitness(0.5));
        // Gen 2 (island 0): fail — fitness drops
        gov.advance(&mock_fitness(0.1));
        // Gen 3 (island 1): fail
        gov.advance(&mock_fitness(0.1));
        // Gen 4 (island 0): fail again — second data point for island 0
        gov.advance(&mock_fitness(0.1));

        let history = gov.fitness_history_by_island();
        // Both islands should have data points even for failed generations
        assert!(history.contains_key(&0));
        assert!(history.contains_key(&1));
        // Each island should have at least 2 data points
        assert!(history[&0].len() >= 2);
        assert!(history[&1].len() >= 2);
    }

    #[test]
    fn tree_nodes_include_island_id() {
        let config = GovernorConfig {
            population_size: 4,
            tick_budget: 100,
            elitism_count: 1,
            patience: 5,
            max_generations: 0,
            mutation_strength: 0.1,
            eval_repeats: 1,
            num_islands: 2,
            migration_interval: 0,
            momentum_decay: 0.9,
        };
        let brain = BrainConfig::default();
        let mut gov = Governor::new(":memory:", config, &brain, "{}").unwrap();

        let root_id = gov.current_node_id.unwrap();

        // Advance twice (one per island)
        gov.advance(&mock_fitness(0.1));
        gov.advance(&mock_fitness(0.2));

        let nodes = gov.tree_nodes();
        // Root should have island_id = None
        let root_node = nodes.iter().find(|n| n.id == root_id).unwrap();
        assert_eq!(root_node.island_id, None);

        // Non-root nodes should have island_id set
        let non_root: Vec<&TreeNode> = nodes.iter().filter(|n| n.id != root_id).collect();
        assert!(!non_root.is_empty());
        for node in non_root {
            assert!(
                node.island_id.is_some(),
                "node {} should have island_id",
                node.id
            );
        }
    }
}
