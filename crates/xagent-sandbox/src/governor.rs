//! Evolution governor — orchestrates natural selection across generations.
//!
//! The governor manages a population of agents, evaluates their fitness after
//! a fixed tick budget, selects the best performers, and breeds mutated offspring.
//! All state is persisted in a SQLite database (`xagent.db`), enabling resume
//! and post-hoc analysis of the evolution tree.

use rusqlite::{params, Connection, Result as SqlResult};
use serde::Serialize;
use xagent_shared::{BrainConfig, GovernorConfig};

use crate::agent::{mutate_config, Agent};

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

/// The evolution governor. Owns the SQLite connection and drives the
/// generational loop: evaluate → select → breed → repeat.
pub struct Governor {
    pub db: Connection,
    pub config: GovernorConfig,
    pub run_id: i64,
    pub current_node_id: Option<i64>,
    pub generation: u32,
    pub gen_tick: u64,
}

impl Governor {
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

        Ok(Self {
            db,
            config,
            run_id,
            current_node_id: Some(root_id),
            generation: 0,
            gen_tick: 0,
        })
    }

    /// Resume from an existing database. Finds the latest active node
    /// for the most recent run.
    pub fn resume(db_path: &str) -> SqlResult<Self> {
        let db = Connection::open(db_path)?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        let (run_id, governor_json): (i64, String) = db.query_row(
            "SELECT id, governor_config FROM run ORDER BY id DESC LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
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

        Ok(Self {
            db,
            config,
            run_id,
            current_node_id: Some(node_id),
            generation,
            gen_tick: 0,
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
        self.config.max_generations > 0 && self.generation as u64 >= self.config.max_generations
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

        // Normalize each axis to [0, 1] relative to generation max
        let max_alive = results
            .iter()
            .map(|r| r.total_ticks_alive)
            .max()
            .unwrap_or(1)
            .max(1) as f32;
        let max_food = results
            .iter()
            .map(|r| r.food_consumed)
            .max()
            .unwrap_or(1)
            .max(1) as f32;
        let max_cells = results
            .iter()
            .map(|r| r.cells_explored)
            .max()
            .unwrap_or(1)
            .max(1) as f32;

        for r in &mut results {
            let norm_alive = r.total_ticks_alive as f32 / max_alive;
            let norm_food = r.food_consumed as f32 / max_food;
            let norm_cells = r.cells_explored as f32 / max_cells;
            r.composite_fitness = norm_alive * 0.4 + norm_food * 0.3 + norm_cells * 0.3;
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

    /// Select the top performers and breed mutated offspring to fill the population.
    /// Returns a Vec of BrainConfigs for the next generation's agents.
    pub fn select_and_breed(&mut self, fitness: &[AgentFitness]) -> Vec<BrainConfig> {
        let elite_count = self.config.elitism_count.min(fitness.len());
        let pop_size = self.config.population_size;

        // Elite configs survive
        let elites: Vec<BrainConfig> = fitness[..elite_count]
            .iter()
            .map(|f| f.config.clone())
            .collect();

        let best_config = elites.first().cloned().unwrap_or_default();

        // Fill remaining slots by round-robin mutation of elites
        let mut next_gen = elites.clone();
        let mut elite_idx = 0;
        while next_gen.len() < pop_size {
            let parent = &elites[elite_idx % elite_count];
            let child = mutate_config(parent);
            next_gen.push(child);
            elite_idx += 1;
        }

        // Record mutation info for the new node
        // The "mutated_param" for the node is a summary — individual mutations
        // are recorded in the mutation table
        let parent_node_id = self.current_node_id;
        let parent_fitness = fitness.first().map(|f| f.composite_fitness).unwrap_or(0.0);

        // Create new node for next generation
        self.generation += 1;
        let config_json = serde_json::to_string(&best_config).unwrap_or_default();

        let _ = self.db.execute(
            "INSERT INTO node (run_id, parent_id, generation, config_json, status)
             VALUES (?1, ?2, ?3, ?4, 'active')",
            params![
                self.run_id,
                parent_node_id,
                self.generation,
                config_json,
            ],
        );
        let new_node_id = self.db.last_insert_rowid();
        self.current_node_id = Some(new_node_id);
        self.gen_tick = 0;

        // Record mutations (compare best config to parent node's config)
        if let Some(parent_id) = parent_node_id {
            if let Ok(parent_json) = self.db.query_row(
                "SELECT config_json FROM node WHERE id = ?1",
                params![parent_id],
                |row| row.get::<_, String>(0),
            ) {
                if let Ok(parent_config) = serde_json::from_str::<BrainConfig>(&parent_json) {
                    record_mutations(
                        &self.db,
                        new_node_id,
                        &parent_config,
                        &best_config,
                        parent_fitness,
                    );
                }
            }
        }

        next_gen
    }

    /// Check if fitness has regressed for `patience` consecutive generations.
    pub fn should_backtrack(&self) -> bool {
        let node_id = match self.current_node_id {
            Some(id) => id,
            None => return false,
        };

        // Walk up the tree collecting recent fitness values
        let mut fitnesses: Vec<f32> = Vec::new();
        let mut current = Some(node_id);
        let needed = self.config.patience as usize + 1;

        while let Some(nid) = current {
            if fitnesses.len() >= needed {
                break;
            }
            match self.db.query_row(
                "SELECT best_fitness, parent_id FROM node WHERE id = ?1",
                params![nid],
                |row| {
                    let f: Option<f64> = row.get(0)?;
                    let p: Option<i64> = row.get(1)?;
                    Ok((f.map(|v| v as f32), p))
                },
            ) {
                Ok((Some(f), parent)) => {
                    fitnesses.push(f);
                    current = parent;
                }
                Ok((None, parent)) => {
                    current = parent;
                }
                Err(_) => break,
            }
        }

        if fitnesses.len() < 2 {
            return false;
        }

        // Check if each generation was worse than its predecessor
        // fitnesses[0] is current (newest), fitnesses[1] is parent, etc.
        let regression_count = fitnesses
            .windows(2)
            .take_while(|w| w[0] <= w[1])
            .count();

        regression_count >= self.config.patience as usize
    }

    /// Backtrack to the last improving node and try a different direction.
    /// Returns the BrainConfig to use for the next generation, or None if
    /// we can't backtrack further.
    pub fn backtrack(&mut self) -> Option<BrainConfig> {
        let current_id = self.current_node_id?;

        // Mark current node as backtracked
        let _ = self.db.execute(
            "UPDATE node SET status = 'backtracked' WHERE id = ?1",
            params![current_id],
        );

        // Walk up to find a node that still has untried parameter directions
        let mut candidate = self.db.query_row(
            "SELECT parent_id FROM node WHERE id = ?1",
            params![current_id],
            |row| row.get::<_, Option<i64>>(0),
        ).ok()?;

        while let Some(node_id) = candidate {
            // Get all mutable param names
            let all_params = vec![
                "memory_capacity",
                "processing_slots",
                "representation_dim",
                "learning_rate",
                "decay_rate",
            ];

            // Get params+directions already tried from this node
            let mut tried: Vec<(String, f64)> = Vec::new();
            {
                let mut stmt = self
                    .db
                    .prepare(
                        "SELECT DISTINCT m.param_name, m.direction FROM mutation m
                         JOIN node n ON m.node_id = n.id
                         WHERE n.parent_id = ?1",
                    )
                    .ok()?;
                let rows = stmt
                    .query_map(params![node_id], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
                    })
                    .ok()?;
                for r in rows.flatten() {
                    tried.push(r);
                }
            }

            // Find an untried param+direction
            for param in &all_params {
                for &dir in &[1.0f64, -1.0] {
                    let already = tried
                        .iter()
                        .any(|(p, d)| p == *param && (*d - dir).abs() < 0.1);
                    if !already {
                        // Found an untried direction — use this node's config
                        if let Ok(config) = self.db.query_row(
                            "SELECT config_json FROM node WHERE id = ?1",
                            params![node_id],
                            |row| row.get::<_, String>(0),
                        ) {
                            if let Ok(base_config) = serde_json::from_str::<BrainConfig>(&config) {
                                log::info!(
                                    "[BACKTRACK] Reverting to node {} (gen {}), trying {} {}",
                                    node_id,
                                    self.generation,
                                    param,
                                    if dir > 0.0 { "↑" } else { "↓" }
                                );
                                self.current_node_id = Some(node_id);
                                return Some(base_config);
                            }
                        }
                    }
                }
            }

            // All directions exhausted at this node — mark and go up
            let _ = self.db.execute(
                "UPDATE node SET status = 'exhausted' WHERE id = ?1",
                params![node_id],
            );
            candidate = self
                .db
                .query_row(
                    "SELECT parent_id FROM node WHERE id = ?1",
                    params![node_id],
                    |row| row.get::<_, Option<i64>>(0),
                )
                .ok()?;
        }

        log::warn!("[BACKTRACK] No viable backtrack target found — tree exhausted");
        None
    }

    /// Print a generation summary to stdout.
    pub fn log_generation(&self, fitness: &[AgentFitness]) {
        let best = fitness.first().map(|f| f.composite_fitness).unwrap_or(0.0);
        let avg = if fitness.is_empty() {
            0.0
        } else {
            fitness.iter().map(|f| f.composite_fitness).sum::<f32>() / fitness.len() as f32
        };

        println!("╔══════════════════════════════════════════════════════╗");
        println!(
            "║  Generation {:>4}  │  Best: {:.4}  │  Avg: {:.4}       ║",
            self.generation, best, avg,
        );
        println!("╠══════════════════════════════════════════════════════╣");

        for (i, f) in fitness.iter().take(5).enumerate() {
            println!(
                "║  #{} Agent {:>2} │ alive:{:>6} food:{:>4} cells:{:>4} fit:{:.3} ║",
                i + 1,
                f.agent_index,
                f.total_ticks_alive,
                f.food_consumed,
                f.cells_explored,
                f.composite_fitness,
            );
        }

        if let Some(best_config) = fitness.first() {
            let c = &best_config.config;
            println!("╠══════════════════════════════════════════════════════╣");
            println!(
                "║  Config: mem={} slots={} dim={} lr={:.4} decay={:.4}  ║",
                c.memory_capacity,
                c.processing_slots,
                c.representation_dim,
                c.learning_rate,
                c.decay_rate,
            );
        }
        println!("╚══════════════════════════════════════════════════════╝");
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
                    mutated_param, mutation_direction, status
             FROM node WHERE run_id = ?1 ORDER BY id",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows = stmt
            .query_map(params![self.run_id], |row| {
                Ok(TreeNode {
                    id: row.get(0)?,
                    parent_id: row.get(1)?,
                    generation: row.get(2)?,
                    best_fitness: row.get::<_, Option<f64>>(3)?.map(|v| v as f32),
                    avg_fitness: row.get::<_, Option<f64>>(4)?.map(|v| v as f32),
                    mutated_param: row.get(5)?,
                    mutation_direction: row.get::<_, Option<f64>>(6)?.map(|v| v as f32),
                    status: row.get(7)?,
                })
            })
            .ok();

        match rows {
            Some(r) => r.flatten().collect(),
            None => Vec::new(),
        }
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
    pub mutated_param: Option<String>,
    pub mutation_direction: Option<f32>,
    pub status: String,
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
            wall_time_secs REAL DEFAULT 0
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
