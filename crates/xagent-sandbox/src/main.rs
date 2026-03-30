use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use egui_wgpu;
use glam::Vec3;
use log::info;
use rand::Rng;
use rayon::prelude::*;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};
use xagent_shared::{BrainConfig, FullConfig, GovernorConfig, SensoryFrame, WorldConfig};

use xagent_sandbox::agent::senses::OtherAgent;
use xagent_sandbox::agent::{
    mutate_config, senses, Agent, AgentBody, MAX_AGENTS,
};
use xagent_sandbox::recording::MetricsLogger;
use xagent_sandbox::renderer::camera::Camera;
use xagent_sandbox::renderer::font::TextItem;
use xagent_sandbox::renderer::hud::HudBar;
use xagent_sandbox::renderer::{GpuMesh, InstanceData, Renderer};
use xagent_sandbox::governor::{check_existing_session, reset_database, AdvanceResult, Governor};
use xagent_sandbox::gpu_compute::GpuBrainCompute;
use xagent_sandbox::headless;
use xagent_sandbox::overlay;
use xagent_sandbox::ui::{
    AgentSnapshot, EguiIntegration, EvolutionAction, EvolutionSnapshot, EvolutionState, Tab,
    TabContext,
};

use xagent_sandbox::world::WorldState;

// ── CLI ────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "xagent", about = "Emergent Cognitive Agent Sandbox")]
struct Cli {
    /// Brain preset: tiny, default, large
    #[arg(long, default_value = "default")]
    brain_preset: String,

    /// World preset: easy, normal, hard
    #[arg(long, default_value = "normal")]
    world_preset: String,

    /// Load full config from JSON file
    #[arg(long)]
    config: Option<String>,

    /// Random seed for world generation
    #[arg(long)]
    seed: Option<u64>,

    /// Simulation ticks per second
    #[arg(long)]
    tick_rate: Option<f32>,

    /// Run headless (no window, just simulation + logging)
    #[arg(long)]
    no_render: bool,

    /// Print current config as JSON to stdout and exit
    #[arg(long)]
    dump_config: bool,

    /// Enable CSV metrics logging to file
    #[arg(long)]
    log: bool,

    /// SQLite database path for evolution state (default: xagent.db)
    #[arg(long, default_value = "xagent.db")]
    db: String,

    /// Resume evolution from existing database
    #[arg(long)]
    resume: bool,

    /// Maximum generations to run (0 = unlimited)
    #[arg(long)]
    generations: Option<u64>,

    /// Print evolution tree from database and exit
    #[arg(long)]
    dump_tree: bool,

    /// Use GPU compute for brain encode + recall (experimental)
    #[arg(long)]
    gpu_brain: bool,
}

fn resolve_config(cli: &Cli) -> FullConfig {
    let mut config = if let Some(path) = &cli.config {
        let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Error: Failed to read config file '{}': {}", path, e);
            std::process::exit(1);
        });
        serde_json::from_str::<FullConfig>(&data).unwrap_or_else(|e| {
            eprintln!("Error: Failed to parse config file '{}': {}", path, e);
            std::process::exit(1);
        })
    } else {
        let brain = match cli.brain_preset.as_str() {
            "tiny" => BrainConfig::tiny(),
            "default" => BrainConfig::default(),
            "large" => BrainConfig::large(),
            other => {
                eprintln!("Error: Unknown brain preset '{}'. Choose: tiny, default, large", other);
                std::process::exit(1);
            }
        };
        let world = match cli.world_preset.as_str() {
            "easy" => WorldConfig::easy(),
            "normal" => WorldConfig::default(),
            "hard" => WorldConfig::hard(),
            other => {
                eprintln!("Error: Unknown world preset '{}'. Choose: easy, normal, hard", other);
                std::process::exit(1);
            }
        };
        FullConfig { brain, world, governor: GovernorConfig::default() }
    };

    // CLI overrides
    if let Some(seed) = cli.seed {
        config.world.seed = seed;
    }
    if let Some(tick_rate) = cli.tick_rate {
        config.world.tick_rate = tick_rate;
    }

    config
}

fn print_config(config: &FullConfig) {
    println!("── Active Configuration ──────────────────────────");
    println!(
        "  Brain: capacity={} slots={} vis={} dim={} lr={} decay={}",
        config.brain.memory_capacity,
        config.brain.processing_slots,
        config.brain.visual_encoding_size,
        config.brain.representation_dim,
        config.brain.learning_rate,
        config.brain.decay_rate,
    );
    println!(
        "  World: size={} depletion={} move_cost={} hazard={} regen={} food_val={} food_den={} tick_rate={} seed={}",
        config.world.world_size,
        config.world.energy_depletion_rate,
        config.world.movement_energy_cost,
        config.world.hazard_damage_rate,
        config.world.integrity_regen_rate,
        config.world.food_energy_value,
        config.world.food_density,
        config.world.tick_rate,
        config.world.seed,
    );
    println!("──────────────────────────────────────────────────");
}

const RESPAWN_COOLDOWN_FRAMES: u32 = 60;

/// Base simulation rate in Hz. The simulation runs at this rate (scaled by
/// speed multiplier) regardless of rendering frame rate.
const SIM_RATE: f32 = 60.0;
const SIM_DT: f32 = 1.0 / SIM_RATE;

/// Record per-tick telemetry into agent sparkline histories.
fn record_agent_histories(agent: &mut Agent) {
    let t = agent.brain.telemetry();
    let cap = 10_000;
    let h = &mut agent.prediction_error_history;
    if h.len() >= cap {
        h.pop_front();
    }
    h.push_back(t.prediction_error.clamp(0.0, 1.0));
    let h = &mut agent.exploration_rate_history;
    if h.len() >= cap {
        h.pop_front();
    }
    h.push_back(agent.brain.action_selector.exploration_rate());
    let ef = agent.body.body.internal.energy
        / agent.body.body.internal.max_energy.max(0.001);
    let h = &mut agent.energy_history;
    if h.len() >= cap {
        h.pop_front();
    }
    h.push_back(ef.clamp(0.0, 1.0));
    let inf = agent.body.body.internal.integrity
        / agent.body.body.internal.max_integrity.max(0.001);
    let h = &mut agent.integrity_history;
    if h.len() >= cap {
        h.pop_front();
    }
    h.push_back(inf.clamp(0.0, 1.0));
    let vals = agent.brain.action_selector.global_action_values();
    let mut aw = [0.0f32; 8];
    for (j, v) in vals.iter().enumerate().take(8) {
        aw[j] = *v;
    }
    let h = &mut agent.action_weight_history;
    if h.len() >= cap {
        h.pop_front();
    }
    h.push_back(aw);
}

struct App {
    brain_config: BrainConfig,
    world_config: WorldConfig,

    renderer: Option<Renderer>,
    window: Option<Arc<Window>>,
    camera: Camera,

    // Static terrain mesh (built once)
    terrain_gpu: Option<GpuMesh>,

    // Simulation
    world: Option<WorldState>,
    agents: Vec<Agent>,
    next_agent_id: u32,
    tick: u64,
    food_dirty: bool,
    food_gpu: Option<GpuMesh>,

    last_frame: Instant,

    /// Fixed-timestep accumulator — simulation ticks run at SIM_RATE Hz,
    /// decoupled from the render frame rate.
    sim_accumulator: f32,

    // Speed controls (multiplier for simulation rate)
    speed_multiplier: u32,
    paused: bool,
    render_3d: bool,

    // Session statistics
    total_prediction_error: f64,
    error_count: u64,

    // Telemetry selection
    selected_agent_idx: usize,
    viewport_hovered: bool,
    chart_window: usize,

    // CSV metrics logger
    logger: Option<MetricsLogger>,

    // GPU instancing for agents
    agent_instance_buffer: Option<wgpu::Buffer>,
    agent_instance_count: u32,

    // FPS tracking
    frame_times: VecDeque<Instant>,
    fps: f32,

    // Cached HUD data — only rebuilt when a simulation tick runs
    cached_hud_bars: Vec<HudBar>,
    hud_dirty: bool,

    // Cursor position for click-to-select
    cursor_pos: (f64, f64),

    // Heatmap overlay
    heatmap_enabled: bool,
    heatmap_gpu: Option<GpuMesh>,

    // Trail overlay for selected agent
    trail_gpu: Option<GpuMesh>,

    // Selection marker above focused agent
    marker_gpu: Option<GpuMesh>,

    // egui integration (IDE-style UI overlay)
    egui: Option<EguiIntegration>,

    // Console log buffer for the bottom panel
    console_log: VecDeque<String>,

    // Evolution governor (created on Start/Resume, not on App init)
    governor: Option<Governor>,
    evo_snapshot: EvolutionSnapshot,
    evo_wall_start: Instant,
    db_path: String,
    governor_config: GovernorConfig,

    // GPU brain compute (encode + recall offload)
    gpu_brain: bool,
    gpu_compute: Option<xagent_sandbox::gpu_compute::GpuBrainCompute>,

    // Per-frame GPU pipeline state: stores previous frame's sensory data
    // so collect() can apply results with the matching frames.
    gpu_prev_frames: Vec<Option<SensoryFrame>>,
    gpu_prev_agent_dims: Vec<(usize, usize)>,
    // Reusable batch buffers for GPU submit (avoid per-frame allocation)
    gpu_batch_features: Vec<f32>,
    gpu_batch_weights: Vec<f32>,
    gpu_batch_biases: Vec<f32>,
    gpu_batch_patterns: Vec<f32>,
    gpu_batch_active: Vec<u32>,

    // egui_dock tab state
    dock_state: egui_dock::DockState<Tab>,
}

impl App {
    fn new(
        brain_config: BrainConfig,
        world_config: WorldConfig,
        governor_config: GovernorConfig,
        enable_logging: bool,
        db_path: &str,
        gpu_brain: bool,
    ) -> Self {
        let logger = if enable_logging {
            match MetricsLogger::new() {
                Ok(l) => {
                    println!("[RECORDING] Logging metrics to: {}", l.file_name);
                    Some(l)
                }
                Err(e) => {
                    eprintln!("[RECORDING] Failed to create metrics log: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Check for existing session to determine initial UI state
        let mut evo_snapshot = EvolutionSnapshot::default();
        evo_snapshot.edit_brain = brain_config.clone();
        evo_snapshot.edit_governor = governor_config.clone();

        match check_existing_session(db_path) {
            Some((gen, prev_gov, prev_brain)) => {
                evo_snapshot.state = EvolutionState::HasSession { generation: gen };
                evo_snapshot.generation = gen;
                evo_snapshot.current_config = Some(prev_brain);
                evo_snapshot.population_size = prev_gov.population_size;
                evo_snapshot.tick_budget = prev_gov.tick_budget;
                evo_snapshot.elitism_count = prev_gov.elitism_count;
                evo_snapshot.patience = prev_gov.patience;
                evo_snapshot.max_generations = prev_gov.max_generations;
                evo_snapshot.eval_repeats = prev_gov.eval_repeats;
                evo_snapshot.num_islands = prev_gov.num_islands;
                evo_snapshot.migration_interval = prev_gov.migration_interval;
                // Also load tree/fitness from DB for the summary view
                if let Ok(gov) = Governor::resume(db_path) {
                    evo_snapshot.tree_nodes = gov.tree_nodes();
                    evo_snapshot.current_node_id = gov.current_node_id;
                    evo_snapshot.fitness_history = gov.fitness_history();
                }
                println!(
                    "[GOVERNOR] Found previous session at generation {} in {}",
                    gen, db_path
                );
            }
            None => {
                evo_snapshot.state = EvolutionState::Idle;
                println!("[GOVERNOR] No previous session — ready to configure");
            }
        }

        Self {
            brain_config,
            world_config,
            renderer: None,
            window: None,
            camera: Camera::new(16.0 / 9.0),
            terrain_gpu: None,
            world: None,
            agents: Vec::new(),
            next_agent_id: 0,
            tick: 0,
            food_dirty: true,
            food_gpu: None,
            last_frame: Instant::now(),
            sim_accumulator: 0.0,
            speed_multiplier: 1,
            paused: true,
            render_3d: true,
            total_prediction_error: 0.0,
            error_count: 0,
            selected_agent_idx: 0,
            viewport_hovered: false,
            chart_window: 120,
            logger,
            agent_instance_buffer: None,
            agent_instance_count: 0,
            frame_times: VecDeque::new(),
            fps: 0.0,
            cached_hud_bars: Vec::new(),
            hud_dirty: true,
            cursor_pos: (0.0, 0.0),
            heatmap_enabled: false,
            heatmap_gpu: None,
            trail_gpu: None,
            marker_gpu: None,
            egui: None,
            console_log: VecDeque::new(),
            governor: None,
            evo_snapshot,
            evo_wall_start: Instant::now(),
            db_path: db_path.to_string(),
            governor_config,
            gpu_brain,
            gpu_compute: None,
            gpu_prev_frames: Vec::new(),
            gpu_prev_agent_dims: Vec::new(),
            gpu_batch_features: Vec::new(),
            gpu_batch_weights: Vec::new(),
            gpu_batch_biases: Vec::new(),
            gpu_batch_patterns: Vec::new(),
            gpu_batch_active: Vec::new(),
            dock_state: egui_dock::DockState::new(vec![Tab::Evolution, Tab::Sandbox]),
        }
    }

    /// Spawn a new agent with the given BrainConfig at a random position.
    fn spawn_agent(&mut self, config: BrainConfig, generation: u32) {
        if self.agents.len() >= MAX_AGENTS {
            self.log_msg(format!("[SPAWN] Max agents ({}) reached", MAX_AGENTS));
            return;
        }
        let Some(world) = &self.world else { return };

        let mut rng = rand::rng();
        let half = world.config.world_size / 2.0 - 5.0;
        let x: f32 = rng.random_range(-half..half);
        let z: f32 = rng.random_range(-half..half);
        let y = world.terrain.height_at(x, z) + 1.0;

        let id = self.next_agent_id;
        self.next_agent_id += 1;

        let mut agent = Agent::new(id, Vec3::new(x, y, z), config, self.tick);
        agent.generation = generation;

        self.log_msg(format!(
            "[SPAWN] Agent {} (gen {}) at ({:.1}, {:.1})",
            id, generation, x, z
        ));

        self.agents.push(agent);
    }

    /// Ensure GPU compute pipeline matches current agent configuration.
    /// Creates or recreates the pipeline if dimensions changed.
    fn ensure_gpu_compute(&mut self) {
        if !self.gpu_brain || self.agents.is_empty() {
            return;
        }
        let n = self.agents.len() as u32;
        // Use max across all agents — mutations may give different structural params
        let dim = self.agents.iter().map(|a| a.brain.config.representation_dim).max().unwrap() as u32;
        let fc = self.agents.iter().map(|a| a.brain.encoder.feature_count()).max().unwrap() as u32;
        let cap = self.agents.iter().map(|a| a.brain.memory.max_capacity()).max().unwrap() as u32;

        if let Some(ref gpu) = self.gpu_compute {
            if gpu.matches_config(n, dim, fc, cap) {
                return;
            }
        }

        match GpuBrainCompute::try_new(n, dim, fc, cap) {
            Some(gpu) => {
                self.log_msg(format!(
                    "[GPU] Compute pipeline: {} agents, dim={}, features={}, memory={}",
                    n, dim, fc, cap
                ));
                self.gpu_compute = Some(gpu);
            }
            None => {
                self.log_msg("[GPU] No GPU available, falling back to CPU".into());
                self.gpu_brain = false;
            }
        }
    }

    fn handle_evolution_action(&mut self, action: EvolutionAction) {
        match action {
            EvolutionAction::None => {}
            EvolutionAction::Start => {
                let brain_config = self.evo_snapshot.edit_brain.clone();
                let gov_config = self.evo_snapshot.edit_governor.clone();
                let world_json =
                    serde_json::to_string(&self.world_config).unwrap_or_default();
                match Governor::new(&self.db_path, gov_config.clone(), &brain_config, &world_json)
                {
                    Ok(gov) => {
                        self.governor = Some(gov);
                        self.governor_config = gov_config.clone();
                        self.brain_config = brain_config;
                        self.evo_snapshot.state = EvolutionState::Running;
                        self.evo_snapshot.population_size = gov_config.population_size;
                        self.evo_snapshot.tick_budget = gov_config.tick_budget;
                        self.evo_snapshot.elitism_count = gov_config.elitism_count;
                        self.evo_snapshot.patience = gov_config.patience;
                        self.evo_snapshot.max_generations = gov_config.max_generations;
                        self.evo_snapshot.eval_repeats = gov_config.eval_repeats;
                        self.evo_snapshot.num_islands = gov_config.num_islands;
                        self.evo_snapshot.migration_interval = gov_config.migration_interval;
                        self.evo_wall_start = Instant::now();
                        self.paused = false;
                        self.spawn_evolution_population();
                        self.log_msg("[EVOLUTION] Started new run".into());
                    }
                    Err(e) => {
                        self.log_msg(format!("[EVOLUTION] Failed to start: {}", e));
                    }
                }
            }
            EvolutionAction::Resume => {
                match Governor::resume(&self.db_path) {
                    Ok(gov) => {
                        let cfg = gov.current_config();
                        self.evo_snapshot.state = EvolutionState::Running;
                        self.evo_snapshot.generation = gov.generation;
                        self.governor_config = gov.config.clone();
                        if let Some(c) = &cfg {
                            self.brain_config = c.clone();
                        }
                        self.governor = Some(gov);
                        self.evo_wall_start = Instant::now();
                        self.paused = false;
                        self.spawn_evolution_population();
                        self.log_msg("[EVOLUTION] Resumed from database".into());
                    }
                    Err(e) => {
                        self.log_msg(format!("[EVOLUTION] Failed to resume: {}", e));
                    }
                }
            }
            EvolutionAction::Pause => {
                self.evo_snapshot.state = EvolutionState::Paused;
                self.paused = true;
                self.log_msg("[EVOLUTION] Paused".into());
            }
            EvolutionAction::Unpause => {
                self.evo_snapshot.state = EvolutionState::Running;
                self.paused = false;
                self.log_msg("[EVOLUTION] Resumed".into());
            }
            EvolutionAction::Reset => {
                self.governor = None;
                self.agents.clear();
                self.next_agent_id = 0;
                self.tick = 0;
                self.paused = true;
                if let Err(e) = reset_database(&self.db_path) {
                    self.log_msg(format!("[EVOLUTION] Failed to reset DB: {}", e));
                }
                self.evo_snapshot = EvolutionSnapshot::default();
                self.evo_snapshot.edit_brain = self.brain_config.clone();
                self.evo_snapshot.edit_governor = self.governor_config.clone();
                self.log_msg("[EVOLUTION] Reset — ready to start fresh".into());
            }
        }
    }

    fn spawn_evolution_population(&mut self) {
        self.agents.clear();
        self.next_agent_id = 0;
        self.tick = 0;
        let seed = if let Some(gov) = &self.governor {
            gov.current_config().unwrap_or(self.brain_config.clone())
        } else {
            self.brain_config.clone()
        };
        let pop_size = self.governor_config.population_size;
        for i in 0..pop_size {
            let cfg = if i == 0 {
                seed.clone()
            } else {
                mutate_config(&seed)
            };
            self.spawn_agent(cfg, 0);
        }
    }

    fn spawn_population_from_configs(&mut self, configs: &[BrainConfig]) {
        self.agents.clear();
        self.next_agent_id = 0;
        self.tick = 0;
        for cfg in configs {
            self.spawn_agent(cfg.clone(), 0);
        }
    }

    /// Evaluate the current generation, advance to the next (or finish).
    fn advance_generation(&mut self) {
        let wall_secs = self.evo_wall_start.elapsed().as_secs_f64();

        let result = {
            let gov = match self.governor.as_mut() {
                Some(g) => g,
                None => return,
            };

            let fitness = gov.evaluate(&self.agents);
            gov.log_generation(&fitness);
            gov.update_wall_time(wall_secs);
            gov.advance(&fitness)
        };

        // Governor borrow released — safe to call self methods
        match result {
            AdvanceResult::Continue { configs, messages } => {
                for msg in messages {
                    self.log_msg(msg);
                }
                self.spawn_population_from_configs(&configs);
            }
            AdvanceResult::Finished { messages } => {
                for msg in messages {
                    self.log_msg(msg);
                }
                self.evo_snapshot.state = EvolutionState::Paused;
                self.paused = true;
            }
        }
    }

    fn log_msg(&mut self, msg: String) {
        println!("{}", msg);
        self.console_log.push_back(msg);
        if self.console_log.len() > 200 {
            self.console_log.pop_front();
        }
    }

    /// Spawn a child agent near a parent, with mutated config.
    #[allow(dead_code)]
    fn spawn_child(&mut self, parent_idx: usize) {
        if self.agents.len() >= MAX_AGENTS {
            return;
        }
        let parent = &self.agents[parent_idx];
        let parent_id = parent.id;
        let parent_gen = parent.generation;
        let parent_pos = parent.body.body.position;
        let parent_config = parent.brain.config.clone();

        let child_config = mutate_config(&parent_config);
        let mut rng = rand::rng();
        let offset = Vec3::new(
            rng.random_range(-5.0..5.0_f32),
            0.0,
            rng.random_range(-5.0..5.0_f32),
        );
        let child_pos = parent_pos + offset;

        let id = self.next_agent_id;
        self.next_agent_id += 1;

        let Some(world) = &self.world else { return };
        let half = world.config.world_size / 2.0 - 1.0;
        let cx = child_pos.x.clamp(-half, half);
        let cz = child_pos.z.clamp(-half, half);
        let cy = world.terrain.height_at(cx, cz) + 1.0;

        let mut child = Agent::new(id, Vec3::new(cx, cy, cz), child_config, self.tick);
        child.generation = parent_gen + 1;

        println!(
            "[REPRODUCE] Agent {} (gen {}) → child Agent {} (gen {}) at ({:.1}, {:.1})",
            parent_id, parent_gen, id, child.generation, cx, cz
        );
        println!(
            "  Child config: cap={} slots={} dim={} lr={:.4} decay={:.4}",
            child.brain.config.memory_capacity,
            child.brain.config.processing_slots,
            child.brain.config.representation_dim,
            child.brain.config.learning_rate,
            child.brain.config.decay_rate,
        );

        self.agents.push(child);
    }

    /// Pick the agent closest to the cursor via screen-space projection.
    fn pick_agent_at_cursor(&mut self) {
        let Some(renderer) = &self.renderer else { return };
        let w = renderer.config.width as f32;
        let h = renderer.config.height as f32;
        if w < 1.0 || h < 1.0 { return; }

        // Normalized device coordinates [-1, 1]
        let ndc_x = (self.cursor_pos.0 as f32 / w) * 2.0 - 1.0;
        let ndc_y = 1.0 - (self.cursor_pos.1 as f32 / h) * 2.0;

        let vp = self.camera.view_projection_matrix();
        let mut best_idx: Option<usize> = None;
        let mut best_dist_sq = f32::MAX;

        for (i, agent) in self.agents.iter().enumerate() {
            if !agent.body.body.alive { continue; }
            let pos = agent.body.body.position;
            let clip = vp * glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);
            if clip.w <= 0.0 { continue; } // behind camera
            let sx = clip.x / clip.w;
            let sy = clip.y / clip.w;
            let d = (sx - ndc_x).powi(2) + (sy - ndc_y).powi(2);
            if d < best_dist_sq {
                best_dist_sq = d;
                best_idx = Some(i);
            }
        }

        // Only select if click is reasonably close (within ~10% of screen)
        if let Some(idx) = best_idx {
            if best_dist_sq < 0.05 {
                self.selected_agent_idx = idx;
                self.agents[idx].trail_dirty = true;
                let a = &self.agents[idx];
                println!(
                    "[SELECT] Agent {} (gen {}) — phase: {}",
                    a.id, a.generation, a.brain.telemetry().behavior_phase()
                );
                self.hud_dirty = true;
            }
        }
    }

    fn print_session_summary(&mut self) {
        if let Some(logger) = &mut self.logger {
            let _ = logger.flush();
        }

        let avg_err = if self.error_count > 0 {
            self.total_prediction_error / self.error_count as f64
        } else {
            0.0
        };

        let total_deaths: u32 = self.agents.iter().map(|a| a.death_count).sum();

        let log_name = self
            .logger
            .as_ref()
            .map(|l| l.file_name.as_str())
            .unwrap_or("(none)");

        println!();
        println!("=== xagent Session Summary ===");
        println!("Total ticks: {}", self.tick);
        println!("Total agents spawned: {}", self.next_agent_id);
        println!("Living agents: {}", self.agents.len());
        println!("Total deaths: {}", total_deaths);
        println!("Avg prediction error: {:.2}", avg_err);
        println!("Log file: {}", log_name);
        println!();
    }

    /// Build HUD overlay bars for the selected agent.
    fn build_hud_bars(&self) -> Vec<HudBar> {
        let Some(agent) = self.agents.get(self.selected_agent_idx) else {
            return Vec::new();
        };

        let energy = agent.body.body.internal.energy_signal();
        let integrity = agent.body.body.internal.integrity_signal();
        let pred_err = agent.brain.telemetry().prediction_error.clamp(0.0, 1.0);
        let explore = agent.brain.telemetry().exploration_rate.clamp(0.0, 1.0);

        let bar_w = 0.35;
        let bar_h = 0.025;
        let left = -0.98;
        let top = 0.97;
        let gap = 0.035;
        let bg = [0.15, 0.15, 0.15];

        vec![
            // Energy — green
            HudBar {
                x: left,
                y: top,
                width: bar_w,
                height: bar_h,
                fill: energy,
                color: [0.2, 0.85, 0.2],
                bg_color: bg,
            },
            // Integrity — blue
            HudBar {
                x: left,
                y: top - gap,
                width: bar_w,
                height: bar_h,
                fill: integrity,
                color: [0.3, 0.5, 1.0],
                bg_color: bg,
            },
            // Prediction error — red
            HudBar {
                x: left,
                y: top - gap * 2.0,
                width: bar_w,
                height: bar_h,
                fill: pred_err,
                color: [0.95, 0.2, 0.15],
                bg_color: bg,
            },
            // Exploration rate — yellow
            HudBar {
                x: left,
                y: top - gap * 3.0,
                width: bar_w,
                height: bar_h,
                fill: explore,
                color: [0.95, 0.85, 0.1],
                bg_color: bg,
            },
        ]
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Spin the event loop continuously for uncapped frame rate
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("xagent — Emergent Cognitive Agent Sandbox")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window =
            Arc::new(event_loop.create_window(attrs).expect("Failed to create window"));
        let size = window.inner_size();
        self.camera.aspect = size.width as f32 / size.height.max(1) as f32;

        let renderer = Renderer::new(window.clone());

        // ── world ──────────────────────────────────────────────────
        let world = WorldState::new(self.world_config.clone());

        let terrain_mesh = world.terrain_mesh();
        self.terrain_gpu = Some(GpuMesh::from_mesh(&renderer.device, &terrain_mesh));

        // Dynamic food buffer — max ~500 food items × 24 vertices each
        let food_mesh = world.food_mesh();
        let mut food_gpu = GpuMesh::new_dynamic(&renderer.device, 12000, 18000);
        food_gpu.update_from_mesh(&renderer.queue, &food_mesh);
        self.food_gpu = Some(food_gpu);
        self.food_dirty = false;

        // Dynamic heatmap overlay buffer — max 4096 cells × 4 vertices each
        let heatmap_res = xagent_sandbox::agent::HEATMAP_RES;
        let max_verts = (heatmap_res * heatmap_res * 4) as u64;
        let max_idx = (heatmap_res * heatmap_res * 6) as u64;
        self.heatmap_gpu = Some(GpuMesh::new_dynamic(&renderer.device, max_verts, max_idx));

        // Trail overlay: linear ribbon, up to MAX_TRAIL_POINTS control points.
        // Each segment = 4 verts, 6 indices.
        let max_trail_segs = xagent_sandbox::agent::MAX_TRAIL_POINTS as u64;
        self.trail_gpu = Some(GpuMesh::new_dynamic(
            &renderer.device,
            max_trail_segs * 4,
            max_trail_segs * 6,
        ));

        // Selection marker: diamond (24 verts, 24 indices)
        self.marker_gpu = Some(GpuMesh::new_dynamic(&renderer.device, 24, 24));

        // Pre-allocate agent instance buffer
        let instance_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agent_instance_buffer"),
            size: (MAX_AGENTS * std::mem::size_of::<InstanceData>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.agent_instance_buffer = Some(instance_buffer);

        self.world = Some(world);
        self.renderer = Some(renderer);
        self.window = Some(window.clone());

        // ── egui integration ───────────────────────────────────────
        {
            let r = self.renderer.as_ref().unwrap();
            self.egui = Some(EguiIntegration::new(
                &r.device,
                r.config.format,
                &window,
                r.config.width,
                r.config.height,
            ));
        }

        self.tick = 0;
        self.last_frame = Instant::now();

        println!(
            "[CONTROLS] P/Space = pause | 1-6 = speed | G = toggle 3D | ESC = quit"
        );
        println!(
            "[CONTROLS] N = spawn agent | M = spawn mutated agent | Tab = cycle telemetry"
        );
        info!("Renderer + world + brain initialized — agent is alive");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        // Forward every event to egui first.
        // If egui consumed it (pointer over a panel, typing in a text field, etc.)
        // we skip our own camera/sim key handling for that event.
        let egui_consumed = if let (Some(egui), Some(window)) =
            (&mut self.egui, &self.window)
        {
            egui.on_window_event(window, &event)
        } else {
            false
        };

        // For mouse/scroll events over the 3D viewport, let camera controls through.
        // The viewport_hovered flag is set each frame when the pointer is over the
        // viewport image. When it's over other egui panels, block camera input.
        let pointer_on_viewport = self.viewport_hovered;

        match event {
            WindowEvent::CloseRequested => {
                info!("Window close requested, shutting down");
                self.print_session_summary();
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(new_size.width, new_size.height);
                    self.camera.aspect =
                        new_size.width as f32 / new_size.height.max(1) as f32;
                }
            }

            WindowEvent::KeyboardInput { event, .. } if !egui_consumed => {
                let pressed = event.state == ElementState::Pressed;
                match event.physical_key {
                    // ── camera controls ─────────────────────────────
                    PhysicalKey::Code(KeyCode::KeyW) => self.camera.move_forward = pressed,
                    PhysicalKey::Code(KeyCode::KeyS) => self.camera.move_backward = pressed,
                    PhysicalKey::Code(KeyCode::KeyA) => self.camera.move_left = pressed,
                    PhysicalKey::Code(KeyCode::KeyD) => self.camera.move_right = pressed,
                    PhysicalKey::Code(KeyCode::KeyE) => self.camera.move_up = pressed,
                    PhysicalKey::Code(KeyCode::ShiftLeft) => {
                        self.camera.move_down = pressed;
                    }
                    PhysicalKey::Code(KeyCode::Digit0) if pressed => {
                        self.camera.reset();
                    }

                    // ── simulation controls ─────────────────────────
                    PhysicalKey::Code(KeyCode::Escape) if pressed => {
                        self.print_session_summary();
                        event_loop.exit();
                    }
                    PhysicalKey::Code(KeyCode::KeyP | KeyCode::Space) if pressed => {
                        match self.evo_snapshot.state {
                            EvolutionState::Running => {
                                self.handle_evolution_action(EvolutionAction::Pause);
                            }
                            EvolutionState::Paused => {
                                self.handle_evolution_action(EvolutionAction::Unpause);
                            }
                            _ => {}
                        }
                    }
                    PhysicalKey::Code(KeyCode::Digit1) if pressed => {
                        self.speed_multiplier = 1;
                        println!("[SIM] Speed: 1x ({} ticks/sec)", SIM_RATE as u32);
                    }
                    PhysicalKey::Code(KeyCode::Digit2) if pressed => {
                        self.speed_multiplier = 2;
                        println!("[SIM] Speed: 2x ({} ticks/sec)", SIM_RATE as u32 * 2);
                    }
                    PhysicalKey::Code(KeyCode::Digit3) if pressed => {
                        self.speed_multiplier = 5;
                        println!("[SIM] Speed: 5x ({} ticks/sec)", SIM_RATE as u32 * 5);
                    }
                    PhysicalKey::Code(KeyCode::Digit4) if pressed => {
                        self.speed_multiplier = 10;
                        println!("[SIM] Speed: 10x ({} ticks/sec)", SIM_RATE as u32 * 10);
                    }
                    PhysicalKey::Code(KeyCode::Digit5) if pressed => {
                        self.speed_multiplier = 100;
                        println!("[SIM] Speed: 100x ({} ticks/sec)", SIM_RATE as u32 * 100);
                    }
                    PhysicalKey::Code(KeyCode::Digit6) if pressed => {
                        self.speed_multiplier = 1000;
                        println!("[SIM] Speed: 1000x ({} ticks/sec)", SIM_RATE as u32 * 1000);
                    }
                    PhysicalKey::Code(KeyCode::KeyH) if pressed => {
                        self.heatmap_enabled = !self.heatmap_enabled;
                        println!(
                            "[SIM] Heatmap: {}",
                            if self.heatmap_enabled { "ON" } else { "OFF" }
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyG) if pressed => {
                        self.render_3d = !self.render_3d;
                        self.log_msg(format!(
                            "[SIM] 3D render: {}",
                            if self.render_3d { "ON" } else { "OFF (fast mode)" }
                        ));
                    }

                    // ── agent spawning ──────────────────────────────
                    PhysicalKey::Code(KeyCode::KeyN) if pressed => {
                        self.spawn_agent(self.brain_config.clone(), 0);
                    }
                    PhysicalKey::Code(KeyCode::KeyM) if pressed => {
                        let mutated = mutate_config(&self.brain_config);
                        self.spawn_agent(mutated, 0);
                    }

                    // ── telemetry cycling ───────────────────────────
                    PhysicalKey::Code(KeyCode::Tab) if pressed => {
                        if !self.agents.is_empty() {
                            self.selected_agent_idx =
                                (self.selected_agent_idx + 1) % self.agents.len();
                            self.agents[self.selected_agent_idx].trail_dirty = true;
                            let a = &self.agents[self.selected_agent_idx];
                            println!(
                                "[TELEMETRY] Now showing Agent {} (gen {}, color: ({:.2},{:.2},{:.2}))",
                                a.id, a.generation, a.color[0], a.color[1], a.color[2]
                            );
                        }
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseInput { state, button, .. } if pointer_on_viewport => {
                if button == MouseButton::Left {
                    self.camera.is_mouse_dragging = state == ElementState::Pressed;
                    if state == ElementState::Released {
                        self.camera.last_mouse_pos = None;
                    }
                }
                // Right-click to select nearest agent
                if button == MouseButton::Right && state == ElementState::Released {
                    self.pick_agent_at_cursor();
                }
            }

            WindowEvent::CursorMoved { position, .. }
                if pointer_on_viewport || self.camera.is_mouse_dragging =>
            {
                self.cursor_pos = (position.x, position.y);
                self.camera.process_mouse_move(position.x, position.y);
            }

            WindowEvent::MouseWheel { delta, .. } if pointer_on_viewport => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.camera.process_scroll(scroll);
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.05);
                self.last_frame = now;

                // ── FPS tracking ──────────────────────────────────
                self.frame_times.push_back(now);
                while self.frame_times.len() > 300 {
                    self.frame_times.pop_front();
                }
                if self.frame_times.len() >= 2 {
                    let elapsed = (*self.frame_times.back().unwrap()
                        - *self.frame_times.front().unwrap())
                    .as_secs_f32();
                    self.fps = (self.frame_times.len() - 1) as f32 / elapsed.max(0.001);
                }

                // ── camera ─────────────────────────────────────────
                self.camera.update(dt);

                // ── simulation ticks (fixed timestep) ──────────
                if !self.paused {
                    self.sim_accumulator += dt * self.speed_multiplier as f32;
                    // Cap per-frame ticks proportional to speed, with a reasonable ceiling
                    let max_ticks = if self.render_3d {
                        (self.speed_multiplier * 2).min(2000)
                    } else {
                        (self.speed_multiplier * 10).min(10000)
                    };
                    let mut ticks_run = 0u32;

                    // Brain decimation: at high speed, run brain every Nth tick
                    // to keep CPU load manageable. Physics still runs every tick.
                    // sqrt scaling: 1x→1, 10x→3, 100x→10, 1000x→31
                    let brain_stride = ((self.speed_multiplier as f32).sqrt() as u64).max(1);

                    // Pre-allocate buffers outside the tick loop to avoid
                    // repeated heap allocations (12 agents × 10 ticks = 120/frame).
                    let mut all_positions: Vec<(Vec3, bool)> =
                        Vec::with_capacity(self.agents.len());

                    // Lazily initialize GPU compute if --gpu-brain was requested
                    self.ensure_gpu_compute();

                    let gpu_n = self.agents.len();
                    let (gpu_dim, _gpu_fc, gpu_cap) = if let Some(ref gpu) = self.gpu_compute {
                        (gpu.dim() as usize, gpu.feature_count() as usize, gpu.memory_capacity() as usize)
                    } else {
                        (0, 0, 0)
                    };

                    // ── Adaptive GPU/CPU decision ──
                    // GPU dispatches once per frame → 1 brain tick per frame.
                    // CPU rayon runs brain_stride-decimated ticks inside the loop.
                    // Use GPU only when expected brain ticks ≤ 2 per frame so
                    // agents get equivalent cognitive throughput either way.
                    let expected_ticks = (self.sim_accumulator / SIM_DT)
                        .min(max_ticks as f32) as u64;
                    let expected_brain_ticks = expected_ticks / brain_stride;
                    let use_gpu = self.gpu_compute.is_some() && expected_brain_ticks <= 2;

                    // ── GPU COLLECT: always drain in-flight results ──
                    // try_collect() unmaps staging buffers even if we don't
                    // apply the results, preventing stale buffer buildup.
                    // Only apply brain results when GPU is the active path
                    // this frame — avoids a spurious double brain tick on
                    // GPU→CPU transitions.
                    if let Some(gpu) = self.gpu_compute.as_mut() {
                        if let Some((encoded, similarities)) = gpu.try_collect() {
                            if use_gpu {
                                for (i, agent) in self.agents.iter_mut().enumerate() {
                                    if i < self.gpu_prev_frames.len() {
                                        if let Some(frame) = &self.gpu_prev_frames[i] {
                                            let (a_dim, a_cap) = self.gpu_prev_agent_dims[i];
                                            let enc = &encoded[i * gpu_dim..i * gpu_dim + a_dim];
                                            let sim = &similarities[i * gpu_cap..i * gpu_cap + a_cap];
                                            agent.cached_motor =
                                                agent.brain.tick_gpu(frame, enc, sim);
                                            record_agent_histories(agent);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // ── Frame time budget: never block UI longer than this ──
                    const TICK_BUDGET_MS: u128 = 12;
                    let tick_start = Instant::now();

                    while self.sim_accumulator >= SIM_DT && ticks_run < max_ticks {
                        self.sim_accumulator -= SIM_DT;
                        ticks_run += 1;

                        // Yield to UI every 16 ticks to maintain responsiveness
                        if ticks_run % 16 == 0
                            && tick_start.elapsed().as_millis() >= TICK_BUDGET_MS
                        {
                            break;
                        }

                        if let Some(world) = &mut self.world {
                            // ── Phase 1: Snapshot positions (sequential) ──
                            all_positions.clear();
                            all_positions.extend(
                                self.agents
                                    .iter()
                                    .map(|a| (a.body.body.position, a.body.body.alive)),
                            );

                            // ── Phase 2: Brain ticks (CPU rayon) ──
                            // When GPU is active this frame, agents use
                            // cached_motor from the per-frame collect above.
                            // Otherwise, CPU rayon runs brain with stride.
                            if !use_gpu {
                                let tick = self.tick;
                                let any_brain = self.agents.iter().enumerate().any(|(i, a)| {
                                    a.body.body.alive && (tick + i as u64) % brain_stride == 0
                                });

                                if any_brain {
                                    let world_ref: &WorldState = &*world;
                                    let all_pos = &all_positions;

                                    self.agents
                                        .par_iter_mut()
                                        .enumerate()
                                        .for_each(|(i, agent)| {
                                            if !agent.body.body.alive {
                                                return;
                                            }
                                            let run_brain =
                                                (tick + i as u64) % brain_stride == 0;
                                            if !run_brain {
                                                return;
                                            }

                                            let others: Vec<OtherAgent> = all_pos
                                                .iter()
                                                .enumerate()
                                                .filter(|(j, _)| *j != i)
                                                .map(|(_, (pos, alive))| OtherAgent {
                                                    position: *pos,
                                                    alive: *alive,
                                                })
                                                .collect();

                                            senses::extract_senses_with_others(
                                                &agent.body,
                                                world_ref,
                                                tick,
                                                &others,
                                                &mut agent.cached_frame,
                                            );

                                            agent.cached_motor =
                                                agent.brain.tick(&agent.cached_frame);
                                            record_agent_histories(agent);
                                        });
                                }
                            }

                            // ── Phase 3: Physics + stats (sequential, mutates world) ──
                            for i in 0..self.agents.len() {
                                let agent = &mut self.agents[i];
                                if !agent.body.body.alive {
                                    continue;
                                }

                                let motor = agent.cached_motor.clone();
                                let consumed = xagent_sandbox::physics::step(
                                    &mut agent.body, &motor, world, SIM_DT,
                                );
                                if consumed {
                                    self.food_dirty = true;
                                    agent.food_consumed += 1;
                                }
                                agent.total_ticks_alive += 1;
                                agent.record_heatmap(world.config.world_size);
                                agent.record_trail();

                                let run_brain =
                                    (self.tick + i as u64) % brain_stride == 0;
                                if run_brain && i == self.selected_agent_idx {
                                    let life_ticks = agent.age(self.tick);
                                    log_tick_to_csv(
                                        &mut self.logger,
                                        agent,
                                        world,
                                        &motor,
                                        life_ticks,
                                    );
                                    self.total_prediction_error +=
                                        agent.brain.telemetry().prediction_error
                                            as f64;
                                    self.error_count += 1;
                                }
                            }

                            // ── Phase 4: Collision resolution (sequential) ──
                            {
                                let min_dist: f32 = 2.0;
                                let min_dist_sq = min_dist * min_dist;
                                let n = self.agents.len();
                                for i in 0..n {
                                    if !self.agents[i].body.body.alive {
                                        continue;
                                    }
                                    for j in (i + 1)..n {
                                        if !self.agents[j].body.body.alive {
                                            continue;
                                        }
                                        let diff = self.agents[j].body.body.position
                                            - self.agents[i].body.body.position;
                                        let dist_sq = diff.length_squared();
                                        if dist_sq < min_dist_sq && dist_sq > 0.001 {
                                            let dist = dist_sq.sqrt();
                                            let overlap = min_dist - dist;
                                            let push = diff.normalize() * (overlap * 0.5);
                                            let (left, right) = self.agents.split_at_mut(j);
                                            left[i].body.body.position -= push;
                                            right[0].body.body.position += push;
                                        }
                                    }
                                }
                            }

                            let respawned = world.update(SIM_DT);
                            if respawned {
                                self.food_dirty = true;
                            }

                            // Handle death/respawn
                            let mut event_msgs: Vec<String> = Vec::new();
                            for agent in &mut self.agents {
                                if !agent.body.body.alive && agent.respawn_cooldown == 0 {
                                    let life_ticks = agent.age(self.tick);
                                    agent.longest_life = agent.longest_life.max(life_ticks);
                                    agent.death_count += 1;
                                    let cause = if agent.body.body.internal.energy <= 0.0 {
                                        "energy depletion"
                                    } else {
                                        "integrity failure"
                                    };
                                    // Death signal: retroactively punish actions that led here
                                    agent.brain.death_signal();
                                    event_msgs.push(format!(
                                        "[DEATH] Agent {} died ({}), lived {} ticks",
                                        agent.id, cause, life_ticks
                                    ));
                                    agent.respawn_cooldown = RESPAWN_COOLDOWN_FRAMES;
                                } else if !agent.body.body.alive && agent.respawn_cooldown > 0 {
                                    agent.respawn_cooldown -= 1;
                                    if agent.respawn_cooldown == 0 {
                                        let mut rng = rand::rng();
                                        let half = world.config.world_size / 2.0 - 5.0;
                                        let x: f32 = rng.random_range(-half..half);
                                        let z: f32 = rng.random_range(-half..half);
                                        let y = world.terrain.height_at(x, z) + 1.0;
                                        agent.body = AgentBody::new(Vec3::new(x, y, z));
                                        // Partial respawn: no "free heal" from dying
                                        agent.body.body.internal.energy =
                                            agent.body.body.internal.max_energy * 0.5;
                                        agent.body.body.internal.integrity =
                                            agent.body.body.internal.max_integrity * 0.7;
                                        agent.life_start_tick = self.tick;
                                        agent.has_reproduced = false;
                                        agent.generation += 1;
                                        agent.reset_trail();
                                        agent.brain.trauma(0.5);
                                        event_msgs.push(format!(
                                            "[RESPAWN] Agent {} at ({:.1}, {:.1})",
                                            agent.id, x, z
                                        ));
                                    }
                                }
                            }
                            for msg in event_msgs {
                                self.console_log.push_back(msg);
                                if self.console_log.len() > 200 {
                                    self.console_log.pop_front();
                                }
                            }
                        }

                        self.tick += 1;

                        // ── Governor tick tracking ──────────────
                        if let Some(gov) = &mut self.governor {
                            gov.tick();
                        }

                        // (telemetry is shown on-screen via HUD — no console spam)
                    }

                    // ── Generation completion check (after tick batch) ──
                    if let Some(gov) = &self.governor {
                        if gov.generation_complete() {
                            self.advance_generation();
                        }
                    }

                    // ── GPU SUBMIT: extract senses once per frame, submit async ──
                    if use_gpu && ticks_run > 0 {
                        if let Some(world) = &self.world {
                            // Snapshot current positions
                            all_positions.clear();
                            all_positions.extend(
                                self.agents
                                    .iter()
                                    .map(|a| (a.body.body.position, a.body.body.alive)),
                            );
                            let world_ref: &WorldState = world;
                            let all_pos = &all_positions;
                            let tick = self.tick;

                            // Parallel sensory extraction for ALL alive agents
                            struct AgentGpuSlice {
                                feats: Vec<f32>,
                                weights: Vec<f32>,
                                biases: Vec<f32>,
                                pats: Vec<f32>,
                                mask: Vec<u32>,
                                agent_dim: usize,
                                agent_fc: usize,
                                agent_cap: usize,
                                frame: SensoryFrame,
                            }
                            let per_agent: Vec<Option<AgentGpuSlice>> =
                                self.agents
                                    .par_iter_mut()
                                    .enumerate()
                                    .map(|(i, agent)| {
                                        if !agent.body.body.alive {
                                            return None;
                                        }
                                        let others: Vec<OtherAgent> = all_pos
                                            .iter()
                                            .enumerate()
                                            .filter(|(j, _)| *j != i)
                                            .map(|(_, (pos, a))| OtherAgent {
                                                position: *pos,
                                                alive: *a,
                                            })
                                            .collect();
                                        senses::extract_senses_with_others(
                                            &agent.body, world_ref, tick, &others,
                                            &mut agent.cached_frame,
                                        );
                                        let feats = agent.brain.encoder
                                            .extract_features(&agent.cached_frame).to_vec();
                                        let weights = agent.brain.encoder.weights().to_vec();
                                        let biases = agent.brain.encoder.biases().to_vec();
                                        let (pats, mask) = agent.brain.memory.gpu_pattern_data();
                                        Some(AgentGpuSlice {
                                            feats, weights, biases, pats, mask,
                                            agent_dim: agent.brain.config.representation_dim,
                                            agent_fc: agent.brain.encoder.feature_count(),
                                            agent_cap: agent.brain.memory.max_capacity(),
                                            frame: agent.cached_frame.clone(),
                                        })
                                    })
                                    .collect();

                            // Batch into GPU buffers + store frames for next collect
                            let gpu = self.gpu_compute.as_mut().unwrap();
                            self.gpu_batch_features.clear();
                            self.gpu_batch_weights.clear();
                            self.gpu_batch_biases.clear();
                            self.gpu_batch_patterns.clear();
                            self.gpu_batch_active.clear();

                            self.gpu_prev_frames.clear();
                            self.gpu_prev_frames.resize(gpu_n, None);
                            self.gpu_prev_agent_dims.clear();
                            self.gpu_prev_agent_dims.resize(gpu_n, (gpu_dim, gpu_cap));

                            for (i, data) in per_agent.into_iter().enumerate() {
                                if let Some(d) = data {
                                    gpu.pad_agent_into(
                                        &d.feats, &d.weights, &d.biases,
                                        &d.pats, &d.mask,
                                        d.agent_dim, d.agent_fc, d.agent_cap,
                                        &mut self.gpu_batch_features,
                                        &mut self.gpu_batch_weights,
                                        &mut self.gpu_batch_biases,
                                        &mut self.gpu_batch_patterns,
                                        &mut self.gpu_batch_active,
                                    );
                                    self.gpu_prev_agent_dims[i] = (d.agent_dim, d.agent_cap);
                                    self.gpu_prev_frames[i] = Some(d.frame);
                                } else {
                                    gpu.pad_agent_into(
                                        &[], &[], &[], &[], &[],
                                        0, 0, 0,
                                        &mut self.gpu_batch_features,
                                        &mut self.gpu_batch_weights,
                                        &mut self.gpu_batch_biases,
                                        &mut self.gpu_batch_patterns,
                                        &mut self.gpu_batch_active,
                                    );
                                }
                            }

                            // Submit once per frame (async, returns immediately)
                            gpu.submit(
                                &self.gpu_batch_features, &self.gpu_batch_weights,
                                &self.gpu_batch_biases, &self.gpu_batch_patterns,
                                &self.gpu_batch_active,
                            );
                        }
                    }

                    // Clamp accumulator: allow up to max_ticks worth of
                    // carry-over so the time budget can spread ticks across
                    // multiple frames, but never accumulate more than that.
                    self.sim_accumulator = self.sim_accumulator.min(SIM_DT * max_ticks as f32);

                    // Mark HUD dirty if any ticks ran this frame
                    if ticks_run > 0 {
                        self.hud_dirty = true;
                    }
                }

                // Fix selected index if agents were removed
                if !self.agents.is_empty() {
                    self.selected_agent_idx =
                        self.selected_agent_idx.min(self.agents.len() - 1);
                }

                // ── rebuild dynamic meshes ─────────────────────────
                if self.food_dirty {
                    if let (Some(renderer), Some(world), Some(food_gpu)) =
                        (&self.renderer, &self.world, &mut self.food_gpu)
                    {
                        let fm = world.food_mesh();
                        food_gpu.update_from_mesh(&renderer.queue, &fm);
                        self.food_dirty = false;
                    }
                }

                // ── rebuild heatmap overlay ─────────────────────────
                if self.heatmap_enabled {
                    if let (Some(renderer), Some(world), Some(heatmap_gpu)) =
                        (&self.renderer, &self.world, &mut self.heatmap_gpu)
                    {
                        if let Some(agent) = self.agents.get(self.selected_agent_idx) {
                            let mesh = overlay::build_heatmap_mesh(
                                &agent.heatmap,
                                world.config.world_size,
                                &world.terrain,
                            );
                            heatmap_gpu.update_from_mesh(&renderer.queue, &mesh);
                        }
                    }
                } else if let Some(heatmap_gpu) = &mut self.heatmap_gpu {
                    heatmap_gpu.num_indices = 0;
                }

                // ── rebuild trail overlay for selected agent (only when changed) ──
                if let (Some(renderer), Some(trail_gpu)) =
                    (&self.renderer, &mut self.trail_gpu)
                {
                    if let Some(agent) = self.agents.get_mut(self.selected_agent_idx) {
                        if agent.trail_dirty {
                            if agent.trail.len() > 1 && agent.body.body.alive {
                                let mesh = overlay::build_trail_mesh(
                                    &agent.trail,
                                    &agent.color,
                                );
                                trail_gpu.update_from_mesh(&renderer.queue, &mesh);
                            } else {
                                trail_gpu.num_indices = 0;
                            }
                            agent.trail_dirty = false;
                        }
                    } else {
                        trail_gpu.num_indices = 0;
                    }
                }

                // ── rebuild selection marker above focused agent ──────
                if let (Some(renderer), Some(marker_gpu)) =
                    (&self.renderer, &mut self.marker_gpu)
                {
                    if let Some(agent) = self.agents.get(self.selected_agent_idx) {
                        if agent.body.body.alive {
                            let mesh = overlay::build_marker_mesh(agent.body.body.position);
                            marker_gpu.update_from_mesh(&renderer.queue, &mesh);
                        } else {
                            marker_gpu.num_indices = 0;
                        }
                    } else {
                        marker_gpu.num_indices = 0;
                    }
                }

                // ── update agent instance buffer (only when sim ticked) ──
                if self.hud_dirty {
                    let instances: Vec<InstanceData> = self
                        .agents
                        .iter()
                        .map(|a| {
                            let color = if !a.body.body.alive {
                                use xagent_sandbox::agent::srgb_to_linear;
                                [srgb_to_linear(0.25), srgb_to_linear(0.25), srgb_to_linear(0.25)]
                            } else {
                                // Use the agent's assigned palette color, converted
                                // from sRGB to linear so the sRGB framebuffer
                                // produces the correct final color matching the sidebar.
                                use xagent_sandbox::agent::srgb_to_linear;
                                [
                                    srgb_to_linear(a.color[0]),
                                    srgb_to_linear(a.color[1]),
                                    srgb_to_linear(a.color[2]),
                                ]
                            };
                            InstanceData {
                                position: a.body.body.position.into(),
                                color,
                                scale: 2.0,
                                _pad: 0.0,
                            }
                        })
                        .collect();
                    self.agent_instance_count = instances.len() as u32;

                    if let (Some(renderer), Some(buf)) =
                        (&self.renderer, &self.agent_instance_buffer)
                    {
                        if !instances.is_empty() {
                            renderer
                                .queue
                                .write_buffer(buf, 0, bytemuck::cast_slice(&instances));
                        }
                    }
                }

                // ── update HUD + text geometry (only when dirty) ──
                if self.hud_dirty {
                    self.cached_hud_bars = self.build_hud_bars();

                    let mut text_items: Vec<TextItem> = Vec::new();

                    // FPS counter at top-right
                    text_items.push(TextItem {
                        text: format!("FPS: {:.0}", self.fps),
                        x: 0.62,
                        y: 0.97,
                        scale: 0.04,
                        color: [0.0, 0.0, 0.0, 1.0],
                    });

                    // Agent count below FPS
                    let alive = self.agents.iter().filter(|a| a.body.body.alive).count();
                    text_items.push(TextItem {
                        text: format!("Agents: {}/{}", alive, self.agents.len()),
                        x: 0.62,
                        y: 0.93,
                        scale: 0.03,
                        color: [0.0, 0.0, 0.0, 0.8],
                    });

                    // HUD bar labels
                    let label_x = -0.98 + 0.36;
                    let label_scale = 0.03;
                    let label_color = [0.0, 0.0, 0.0, 0.9];
                    let label_top = 0.97;
                    let label_gap = 0.035;

                    if let Some(agent) = self.agents.get(self.selected_agent_idx) {
                        let energy = agent.body.body.internal.energy_signal();
                        let integrity = agent.body.body.internal.integrity_signal();
                        let pred_err = agent.brain.telemetry().prediction_error;
                        let explore = agent.brain.telemetry().exploration_rate;

                        text_items.push(TextItem {
                            text: format!("Energy: {:.0}%", energy * 100.0),
                            x: label_x,
                            y: label_top,
                            scale: label_scale,
                            color: label_color,
                        });
                        text_items.push(TextItem {
                            text: format!("Integrity: {:.0}%", integrity * 100.0),
                            x: label_x,
                            y: label_top - label_gap,
                            scale: label_scale,
                            color: label_color,
                        });
                        text_items.push(TextItem {
                            text: format!("PredErr: {:.2}", pred_err),
                            x: label_x,
                            y: label_top - label_gap * 2.0,
                            scale: label_scale,
                            color: label_color,
                        });
                        text_items.push(TextItem {
                            text: format!("Explore: {:.0}%", explore * 100.0),
                            x: label_x,
                            y: label_top - label_gap * 3.0,
                            scale: label_scale,
                            color: label_color,
                        });

                        // Info block below bars
                        let info_y = label_top - label_gap * 4.5;
                        let info_scale = 0.028;
                        let info_color = [0.1, 0.1, 0.2, 0.9];
                        text_items.push(TextItem {
                            text: format!(
                                "Agent {} | Gen {} | Deaths: {}",
                                agent.id, agent.generation, agent.death_count
                            ),
                            x: -0.98,
                            y: info_y,
                            scale: info_scale,
                            color: info_color,
                        });
                        text_items.push(TextItem {
                            text: format!(
                                "Phase: {} | Quality: {:.0}%",
                                agent.brain.telemetry().behavior_phase(),
                                agent.brain.telemetry().decision_quality * 100.0
                            ),
                            x: -0.98,
                            y: info_y - 0.035,
                            scale: info_scale,
                            color: info_color,
                        });
                        text_items.push(TextItem {
                            text: format!("Tick: {}  Speed: {}x", self.tick, self.speed_multiplier),
                            x: -0.98,
                            y: info_y - 0.07,
                            scale: info_scale,
                            color: info_color,
                        });
                    }

                    if let Some(renderer) = &mut self.renderer {
                        renderer.update_hud(&self.cached_hud_bars, &[]);
                        renderer.update_text(&text_items);
                    }

                    self.hud_dirty = false;
                }

                // ── render (just draws — no geometry rebuilding) ──
                let inst_buf = self.agent_instance_buffer.as_ref();
                let inst_count = self.agent_instance_count;

                let mut pending_evo_action = EvolutionAction::None;

                if let Some(renderer) = &mut self.renderer {
                    let t = self.terrain_gpu.as_ref();
                    let f = self.food_gpu.as_ref();
                    let h = self.heatmap_gpu.as_ref().filter(|g| g.num_indices > 0);
                    let tr = self.trail_gpu.as_ref().filter(|g| g.num_indices > 0);
                    let mk = self.marker_gpu.as_ref().filter(|g| g.num_indices > 0);
                    let mut mesh_vec: Vec<&GpuMesh> = Vec::with_capacity(5);
                    if let Some(t) = t { mesh_vec.push(t); }
                    if let Some(f) = f { mesh_vec.push(f); }
                    if let Some(h) = h { mesh_vec.push(h); }
                    if let Some(tr) = tr { mesh_vec.push(tr); }
                    if let Some(mk) = mk { mesh_vec.push(mk); }

                    let vp = self.camera.view_projection_matrix();

                    // ── Offscreen 3D → egui surface pipeline ──────────
                    match renderer.begin_frame() {
                        Ok(mut frame_ctx) => {
                            // 1) Render 3D scene to offscreen viewport texture
                            if self.render_3d {
                                if let Some(egui) = &self.egui {
                                    renderer.render_3d_offscreen(
                                        &mesh_vec,
                                        &vp,
                                        inst_buf,
                                        inst_count,
                                        &mut frame_ctx.encoder,
                                        &egui.viewport_color_view,
                                    &egui.viewport_depth_view,
                                );
                            }
                            }

                            // 2) Render egui UI to the surface (viewport texture embedded)
                            if let (Some(egui), Some(window)) =
                                (&mut self.egui, &self.window)
                            {
                                let screen = egui_wgpu::ScreenDescriptor {
                                    size_in_pixels: [
                                        renderer.config.width,
                                        renderer.config.height,
                                    ],
                                    pixels_per_point: window.scale_factor() as f32,
                                };

                                let tick = self.tick;
                                let speed = self.speed_multiplier;
                                let fps = self.fps;
                                let render_3d = self.render_3d;
                                let evo_state = self.evo_snapshot.state.clone();
                                let viewport_tex_id = egui.viewport_texture_id;
                                let ppp = window.scale_factor() as f32;
                                let mut desired_vp = (0u32, 0u32);
                                let selected_idx = self.selected_agent_idx;

                                // Snapshot agent data for the UI closure
                                let agent_snaps: Vec<AgentSnapshot> = self.agents.iter().map(|a| {
                                    let telemetry = a.brain.telemetry();
                                    AgentSnapshot {
                                        id: a.id,
                                        gen: a.generation,
                                        energy: a.body.body.internal.energy,
                                        max_energy: a.body.body.internal.max_energy,
                                        integrity: a.body.body.internal.integrity,
                                        max_integrity: a.body.body.internal.max_integrity,
                                        alive: a.body.body.alive,
                                        deaths: a.death_count,
                                        color: a.color,
                                        longest_life: a.longest_life,
                                        exploration_rate: a.brain.action_selector.exploration_rate(),
                                        prediction_error: telemetry.prediction_error,
                                        action_weights: a.brain.action_selector.global_action_values().to_vec(),
                                        prediction_error_history: a.prediction_error_history.iter().copied().collect(),
                                        exploration_rate_history: a.exploration_rate_history.iter().copied().collect(),
                                        energy_history: a.energy_history.iter().copied().collect(),
                                        integrity_history: a.integrity_history.iter().copied().collect(),
                                        action_weight_history: a.action_weight_history.iter().copied().collect(),
                                    }
                                }).collect();

                                // Build evolution snapshot for the UI
                                if let Some(gov) = &self.governor {
                                    self.evo_snapshot.gen_tick = gov.gen_tick;
                                    self.evo_snapshot.generation = gov.generation;
                                    self.evo_snapshot.tree_nodes = gov.tree_nodes();
                                    self.evo_snapshot.current_node_id = gov.current_node_id;
                                    self.evo_snapshot.fitness_history = gov.fitness_history();
                                }
                                let wall = self.evo_wall_start.elapsed().as_secs_f64();
                                self.evo_snapshot.wall_time_secs = wall;
                                if wall > 0.0 {
                                    self.evo_snapshot.ticks_per_sec =
                                        self.evo_snapshot.gen_tick as f64 / wall;
                                }
                                // Move snapshot out so we can pass &mut to the closure
                                let mut evo_snap = std::mem::take(&mut self.evo_snapshot);
                                let mut evo_action = EvolutionAction::None;

                                let console_lines: Vec<&str> = self.console_log.iter()
                                    .map(|s| s.as_str()).collect();

                                let mut clicked_agent_idx: Option<usize> = None;
                                let mut open_agent_tab: Option<u32> = None;
                                let mut vp_hovered = false;
                                let mut chart_win = self.chart_window;
                                let dock_state = &mut self.dock_state;

                                egui.render(
                                    window,
                                    &renderer.device,
                                    &renderer.queue,
                                    &mut frame_ctx.encoder,
                                    &frame_ctx.view,
                                    screen,
                                    |ctx| {
                                        // ── Top bar ──────────────────────────
                                        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.label(
                                                    egui::RichText::new("xagent")
                                                        .strong()
                                                        .color(egui::Color32::from_rgb(120, 200, 255)),
                                                );
                                                ui.separator();
                                                ui.label(format!("Tick: {}", tick));
                                                ui.separator();
                                                ui.label(format!("Speed: {}x", speed));
                                                ui.separator();
                                                ui.label(format!("FPS: {:.0}", fps));
                                                ui.separator();
                                                ui.label(format!("Agents: {}", agent_snaps.len()));
                                                ui.separator();
                                                match &evo_state {
                                                    EvolutionState::Idle => {
                                                        ui.label(
                                                            egui::RichText::new("⏹ IDLE")
                                                                .color(egui::Color32::GRAY),
                                                        );
                                                    }
                                                    EvolutionState::HasSession { generation } => {
                                                        ui.label(
                                                            egui::RichText::new(format!("💾 SESSION @ Gen {}", generation))
                                                                .color(egui::Color32::from_rgb(100, 180, 255)),
                                                        );
                                                    }
                                                    EvolutionState::Running => {
                                                        ui.label(
                                                            egui::RichText::new("▶ RUNNING")
                                                                .color(egui::Color32::from_rgb(50, 200, 80)),
                                                        );
                                                    }
                                                    EvolutionState::Paused => {
                                                        ui.label(
                                                            egui::RichText::new("⏸ PAUSED")
                                                                .color(egui::Color32::YELLOW),
                                                        );
                                                    }
                                                }
                                                if !render_3d {
                                                    ui.separator();
                                                    ui.label(
                                                        egui::RichText::new("⚡ FAST (G)")
                                                            .color(egui::Color32::from_rgb(255, 160, 50)),
                                                    );
                                                }
                                            });
                                        });

                                        // ── Bottom console ───────────────────
                                        egui::TopBottomPanel::bottom("console")
                                            .resizable(true)
                                            .default_height(120.0)
                                            .show(ctx, |ui| {
                                                ui.label(
                                                    egui::RichText::new("Console")
                                                        .small()
                                                        .color(egui::Color32::GRAY),
                                                );
                                                ui.separator();
                                                egui::ScrollArea::vertical()
                                                    .stick_to_bottom(true)
                                                    .show(ui, |ui| {
                                                        for line in &console_lines {
                                                            let color = if line.contains("[DEATH]") {
                                                                egui::Color32::from_rgb(255, 100, 100)
                                                            } else if line.contains("[SPAWN]") || line.contains("[RESPAWN]") {
                                                                egui::Color32::from_rgb(100, 255, 100)
                                                            } else {
                                                                egui::Color32::LIGHT_GRAY
                                                            };
                                                            ui.label(
                                                                egui::RichText::new(*line)
                                                                    .monospace()
                                                                    .size(11.0)
                                                                    .color(color),
                                                            );
                                                        }
                                                    });
                                            });

                                        // ── Left sidebar: agent list ─────────
                                        egui::SidePanel::left("agent_list")
                                            .resizable(true)
                                            .default_width(200.0)
                                            .show(ctx, |ui| {
                                                ui.label(
                                                    egui::RichText::new("Agents")
                                                        .strong()
                                                        .size(14.0),
                                                );
                                                ui.separator();
                                                egui::ScrollArea::vertical().show(ui, |ui| {
                                                    for (idx, snap) in agent_snaps.iter().enumerate() {
                                                        let is_selected = idx == selected_idx;
                                                        let color = egui::Color32::from_rgb(
                                                            (snap.color[0] * 255.0) as u8,
                                                            (snap.color[1] * 255.0) as u8,
                                                            (snap.color[2] * 255.0) as u8,
                                                        );
                                                        let frame = if is_selected {
                                                            egui::Frame::NONE
                                                                .fill(egui::Color32::from_rgba_premultiplied(60, 60, 80, 255))
                                                                .inner_margin(4.0)
                                                                .corner_radius(3.0)
                                                        } else {
                                                            egui::Frame::NONE
                                                                .inner_margin(4.0)
                                                        };
                                                        let response = frame.show(ui, |ui| {
                                                            ui.horizontal(|ui| {
                                                                let (rect, _) = ui.allocate_exact_size(
                                                                    egui::vec2(10.0, 10.0),
                                                                    egui::Sense::hover(),
                                                                );
                                                                ui.painter().circle_filled(
                                                                    rect.center(),
                                                                    5.0,
                                                                    color,
                                                                );
                                                                let status = if !snap.alive { "💀" } else { "" };
                                                                ui.label(format!(
                                                                    "Agent {} (g{}) {}",
                                                                    snap.id, snap.gen, status
                                                                ));
                                                            });
                                                            // Compact vitals: E/I bars + combined history chart
                                                            let energy_frac = snap.energy / snap.max_energy.max(0.001);
                                                            let integrity_frac = snap.integrity / snap.max_integrity.max(0.001);
                                                            let bar_h = 4.0;

                                                            egui::Grid::new(format!("sidebar_vitals_{}", snap.id))
                                                                .num_columns(2)
                                                                .spacing([4.0, 1.0])
                                                                .show(ui, |ui| {
                                                                    ui.label(egui::RichText::new("E").small().color(egui::Color32::GRAY));
                                                                    let e_color = if energy_frac > 0.5 {
                                                                        egui::Color32::from_rgb(80, 200, 80)
                                                                    } else if energy_frac > 0.25 {
                                                                        egui::Color32::YELLOW
                                                                    } else {
                                                                        egui::Color32::from_rgb(220, 60, 60)
                                                                    };
                                                                    ui.add(egui::ProgressBar::new(energy_frac)
                                                                        .fill(e_color)
                                                                        .desired_height(bar_h));
                                                                    ui.end_row();

                                                                    ui.label(egui::RichText::new("I").small().color(egui::Color32::GRAY));
                                                                    ui.add(egui::ProgressBar::new(integrity_frac)
                                                                        .fill(egui::Color32::from_rgb(100, 150, 255))
                                                                        .desired_height(bar_h));
                                                                    ui.end_row();
                                                                });

                                                            // Combined 4-line mini chart (last 10k ticks, no legend)
                                                            let chart_h = 28.0;
                                                            let chart_w = ui.available_width().max(40.0);
                                                            let (chart_rect, _) = ui.allocate_exact_size(
                                                                egui::vec2(chart_w, chart_h),
                                                                egui::Sense::hover(),
                                                            );
                                                            let p = ui.painter();
                                                            p.rect_filled(chart_rect, 1.0, egui::Color32::from_gray(25));
                                                            let sidebar_window = 10_000;
                                                            let series: [(&[f32], egui::Color32); 4] = [
                                                                (&snap.energy_history, egui::Color32::from_rgb(80, 200, 80)),
                                                                (&snap.integrity_history, egui::Color32::from_rgb(100, 150, 255)),
                                                                (&snap.prediction_error_history, egui::Color32::from_rgb(200, 140, 60)),
                                                                (&snap.exploration_rate_history, egui::Color32::from_rgb(180, 100, 220)),
                                                            ];
                                                            for &(full_data, color) in &series {
                                                                let start = full_data.len().saturating_sub(sidebar_window);
                                                                let data = &full_data[start..];
                                                                if data.len() < 2 { continue; }
                                                                let n = data.len();
                                                                let pts: Vec<egui::Pos2> = data.iter().enumerate().map(|(i, &v)| {
                                                                    let x = chart_rect.left() + (i as f32 / (n - 1) as f32) * chart_rect.width();
                                                                    let y = chart_rect.bottom() - v.clamp(0.0, 1.0) * chart_rect.height();
                                                                    egui::pos2(x, y)
                                                                }).collect();
                                                                let stroke = egui::Stroke::new(1.0, color);
                                                                for pair in pts.windows(2) {
                                                                    p.line_segment([pair[0], pair[1]], stroke);
                                                                }
                                                            }

                                                            ui.label(
                                                                egui::RichText::new(format!(
                                                                    "Deaths: {} | Best: {}t",
                                                                    snap.deaths, snap.longest_life
                                                                ))
                                                                .small()
                                                                .color(egui::Color32::GRAY),
                                                            );
                                                        });
                                                        let resp = response.response.interact(egui::Sense::click());
                                                        if resp.clicked() {
                                                            clicked_agent_idx = Some(idx);
                                                        }
                                                        if resp.double_clicked() {
                                                            open_agent_tab = Some(snap.id);
                                                        }
                                                        ui.add_space(2.0);
                                                    }
                                                });
                                            });

                                        // ── Central dock area (tabs) ─────────
                                        egui::CentralPanel::default()
                                            .frame(egui::Frame::NONE)
                                            .show(ctx, |ui| {
                                                let mut tab_ctx = TabContext {
                                                    viewport_tex_id,
                                                    ppp,
                                                    desired_vp: &mut desired_vp,
                                                    viewport_hovered: &mut vp_hovered,
                                                    chart_window: &mut chart_win,
                                                    agents: &agent_snaps,
                                                    evolution: &mut evo_snap,
                                                    evolution_action: &mut evo_action,
                                                };
                                                egui_dock::DockArea::new(dock_state)
                                                    .style(egui_dock::Style::from_egui(ui.style().as_ref()))
                                                    .show_inside(ui, &mut tab_ctx);
                                            });
                                    },
                                );

                                self.viewport_hovered = vp_hovered;
                                self.chart_window = chart_win;

                                // Restore evolution snapshot (may have been mutated by UI)
                                self.evo_snapshot = evo_snap;

                                // Defer evolution action to after the renderer borrow
                                pending_evo_action = evo_action;

                                // Handle agent selection from sidebar click
                                if let Some(idx) = clicked_agent_idx {
                                    self.selected_agent_idx = idx;
                                }

                                // Handle double-click → open agent detail tab
                                if let Some(agent_id) = open_agent_tab {
                                    let tab = Tab::AgentDetail(agent_id);
                                    // Check if tab already exists
                                    let already_open = self.dock_state.iter_all_tabs()
                                        .any(|(_, t)| *t == tab);
                                    if !already_open {
                                        self.dock_state.push_to_focused_leaf(tab);
                                    }
                                }

                                // Resize offscreen textures if the panel changed size (takes effect next frame)
                                if desired_vp.0 > 0 && desired_vp.1 > 0 {
                                    egui.resize_viewport(
                                        &renderer.device,
                                        desired_vp.0,
                                        desired_vp.1,
                                    );
                                    // Update camera aspect to match viewport
                                    self.camera.aspect =
                                        desired_vp.0 as f32 / desired_vp.1.max(1) as f32;
                                }
                            }

                            renderer.finish_frame(frame_ctx);
                        }
                        Err(wgpu::SurfaceError::Lost) => {
                            let w = renderer.config.width;
                            let h = renderer.config.height;
                            renderer.resize(w, h);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("Out of memory!");
                            self.print_session_summary();
                            event_loop.exit();
                        }
                        Err(e) => {
                            log::warn!("Surface error: {:?}", e);
                        }
                    }
                }

                // Handle evolution actions (outside renderer borrow)
                self.handle_evolution_action(pending_evo_action);
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

// ── Free functions to avoid borrow conflicts ───────────────────────────

fn log_tick_to_csv(
    logger: &mut Option<MetricsLogger>,
    agent: &Agent,
    world: &WorldState,
    motor: &xagent_shared::MotorCommand,
    life_ticks: u64,
) {
    if let Some(logger) = logger {
        let biome = world
            .biome_map
            .biome_at(agent.body.body.position.x, agent.body.body.position.z);
        let _ = logger.log_tick(
            agent.id,
            agent.brain.telemetry(),
            agent.brain.config.memory_capacity,
            agent.body.body.internal.energy,
            agent.body.body.internal.max_energy,
            agent.body.body.internal.integrity,
            agent.body.body.internal.max_integrity,
            agent.body.body.position,
            agent.body.body.facing,
            biome,
            motor,
            agent.body.body.alive,
            agent.death_count,
            life_ticks,
            agent.generation,
        );
    }
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    info!("xagent sandbox starting...");
    println!("xagent v0.1.0 \u{2014} Emergent Cognitive Agent Sandbox");

    let mut config = resolve_config(&cli);

    // Override generations from CLI
    if let Some(gens) = cli.generations {
        config.governor.max_generations = gens;
    }

    if cli.dump_config {
        let json = serde_json::to_string_pretty(&config)
            .expect("Failed to serialize config");
        println!("{}", json);
        return;
    }

    if cli.dump_tree {
        headless::dump_tree(&cli.db);
        return;
    }

    print_config(&config);

    if cli.no_render {
        headless::run_headless(config, &cli.db, cli.resume, cli.gpu_brain);
    } else {
        let event_loop = EventLoop::new().expect("Failed to create event loop");
        let mut app = App::new(
            config.brain,
            config.world,
            config.governor,
            cli.log,
            &cli.db,
            cli.gpu_brain,
        );
        event_loop.run_app(&mut app).expect("Event loop error");
    }
}
