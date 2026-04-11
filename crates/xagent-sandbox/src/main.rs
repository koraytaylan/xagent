use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use egui_wgpu;
use glam::Vec3;
use log::info;
use rand::Rng;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};
use xagent_shared::{BrainConfig, FullConfig, GovernorConfig, WorldConfig};

use xagent_brain::buffers::{
    PHYS_STRIDE, P_ALIVE, P_DEATH_COUNT, P_ENERGY, P_EXPLORATION_RATE_OUT, P_FACING_X, P_FACING_Y,
    P_FACING_Z, P_FATIGUE_FACTOR_OUT, P_FOOD_COUNT, P_GRADIENT_OUT, P_INTEGRITY, P_MAX_ENERGY,
    P_MAX_INTEGRITY, P_MOTOR_FWD_OUT, P_MOTOR_TURN_OUT, P_POS_X, P_POS_Y, P_POS_Z,
    P_PREDICTION_ERROR, P_TICKS_ALIVE, P_URGENCY_OUT, P_VEL_X, P_VEL_Y, P_VEL_Z, P_YAW,
};
use xagent_brain::{AgentBrainState, GpuKernel};
use xagent_sandbox::agent::{mutate_brain_state, mutate_config, Agent, MAX_AGENTS};
use xagent_sandbox::governor::{check_existing_session, reset_database, AdvanceResult, Governor};
use xagent_sandbox::headless;
use xagent_sandbox::overlay;
use xagent_sandbox::recording::MetricsLogger;
use xagent_sandbox::renderer::camera::Camera;
use xagent_sandbox::renderer::font::TextItem;
use xagent_sandbox::renderer::hud::HudBar;
use xagent_sandbox::renderer::{GpuMesh, InstanceData, Renderer};
use xagent_sandbox::ui::{
    AgentSnapshot, EguiIntegration, EvolutionAction, EvolutionSnapshot, EvolutionState,
    ReplayState, SortMode, Tab, TabContext, WorldSnapshot,
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

    /// Run headless benchmark (no UI, no DB) and print ticks/sec
    #[arg(long)]
    bench: bool,

    /// Number of ticks for --bench mode (default: 10000)
    #[arg(long, default_value_t = 10_000)]
    bench_ticks: u64,

    /// Number of agents for --bench mode (default: 10)
    #[arg(long, default_value_t = 10)]
    bench_agents: usize,

    /// Override world size for --bench mode (default: from preset)
    #[arg(long)]
    world_size: Option<f32>,

    /// Run phase profiler: breaks down time by physics/vision/brain
    #[arg(long)]
    bench_profile: bool,
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
                eprintln!(
                    "Error: Unknown brain preset '{}'. Choose: tiny, default, large",
                    other
                );
                std::process::exit(1);
            }
        };
        let world = match cli.world_preset.as_str() {
            "easy" => WorldConfig::easy(),
            "normal" => WorldConfig::default(),
            "hard" => WorldConfig::hard(),
            other => {
                eprintln!(
                    "Error: Unknown world preset '{}'. Choose: easy, normal, hard",
                    other
                );
                std::process::exit(1);
            }
        };
        FullConfig {
            brain,
            world,
            governor: GovernorConfig::default(),
        }
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

/// Data collected on the main thread for upload once the background-created
/// fused kernel is ready.  Avoids passing world/agent refs across threads.
struct PendingUpload {
    heights: Vec<f32>,
    biomes: Vec<u32>,
    food_pos: Vec<(f32, f32, f32)>,
    food_consumed: Vec<bool>,
    food_timers: Vec<f32>,
    agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)>,
}

/// Base simulation rate in Hz. The simulation runs at this rate (scaled by
/// speed multiplier) regardless of rendering frame rate.
const SIM_RATE: f32 = 60.0;
const SIM_DT: f32 = 1.0 / SIM_RATE;

/// Minimum interval between expensive mesh rebuilds and snapshot copies.
/// At 100 ms (10 Hz) the visual difference is indistinguishable from per-frame.
const REBUILD_THROTTLE: Duration = Duration::from_millis(100);

/// Record per-tick telemetry into agent sparkline histories.
fn record_agent_histories(agent: &mut Agent) {
    let cap = 10_000;
    macro_rules! push_hist {
        ($h:expr, $v:expr) => {
            if $h.len() >= cap {
                $h.pop_front();
            }
            $h.push_back($v);
        };
    }
    push_hist!(
        agent.prediction_error_history,
        agent.cached_prediction_error
    );
    push_hist!(
        agent.exploration_rate_history,
        agent.cached_exploration_rate
    );
    let ef = agent.body.body.internal.energy / agent.body.body.internal.max_energy.max(0.001);
    push_hist!(agent.energy_history, ef.clamp(0.0, 1.0));
    let inf =
        agent.body.body.internal.integrity / agent.body.body.internal.max_integrity.max(0.001);
    push_hist!(agent.integrity_history, inf.clamp(0.0, 1.0));
    push_hist!(agent.fatigue_history, agent.cached_fatigue_factor);
}

/// State machine for non-blocking generation transitions.
/// Each variant represents one phase of the multi-frame handoff.
enum GenTransition {
    /// Waiting for async readback of the best agent's brain state.
    AwaitingReadback { result: AdvanceResult },
    /// Readback collected; waiting for staging buffers to drain before reset.
    AwaitingReset {
        result: AdvanceResult,
        inherited_state: Option<AgentBrainState>,
        /// Whether population has already been spawned (prevents re-spawn on retry).
        spawned: bool,
    },
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
    orbit_mode: bool,

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
    heatmap_dirty: bool,

    // Trail overlay for selected agent
    trail_gpu: Option<GpuMesh>,

    // Throttle timers — limit expensive rebuilds to ~10 Hz
    last_mesh_rebuild: Instant,
    last_snapshot_rebuild: Instant,
    snap_dirty: bool,
    cached_agent_snaps: Vec<AgentSnapshot>,
    last_snap_chart_window: usize,

    // Selection marker above focused agent
    marker_gpu: Option<GpuMesh>,

    // egui integration (IDE-style UI overlay)
    egui: Option<EguiIntegration>,

    // Console log buffer for the bottom panel
    console_log: VecDeque<String>,

    // Evolution governor (created on Start/Resume, not on App init)
    governor: Option<Governor>,
    evo_snapshot: EvolutionSnapshot,
    evo_wall_accumulated: f64,
    evo_wall_segment_start: Option<Instant>,
    tps_tick_count: u64,
    tps_last_reset: Instant,
    tps_display: f64,
    db_path: String,
    governor_config: GovernorConfig,

    // Non-blocking generation transition state machine
    gen_transition: Option<GenTransition>,

    // GPU fused kernel (all simulation computation in single dispatch)
    gpu_kernel: Option<GpuKernel>,
    pending_kernel: Option<std::thread::JoinHandle<GpuKernel>>,
    /// Data collected for upload once the background kernel is ready.
    pending_upload: Option<PendingUpload>,
    /// Inherited brain state + mutation strength deferred until a background
    /// kernel finishes creation (when population size changes at generation
    /// boundary and `can_reuse` is false).
    deferred_inherited: Option<(AgentBrainState, f32)>,

    // egui_dock tab state
    dock_state: egui_dock::DockState<Tab>,

    // Adaptive GPU tick budget — keeps dispatch under ~8ms wall time
    gpu_tick_budget: u32,

    // Diagnostic: log first readback Y vs terrain height
    readback_logged: bool,

    // Sidebar sort mode
    sort_mode: SortMode,

    // World snapshot for mini-map UI
    world_snapshot: WorldSnapshot,

    // Replay recording
    recording: Option<xagent_sandbox::replay::GenerationRecording>,
    last_recording: Option<xagent_sandbox::replay::GenerationRecording>,

    // Replay playback state
    replay_state: ReplayState,
}

impl App {
    fn new(
        brain_config: BrainConfig,
        world_config: WorldConfig,
        governor_config: GovernorConfig,
        enable_logging: bool,
        db_path: &str,
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
                if let Ok(mut gov) = Governor::resume(db_path) {
                    evo_snapshot.tree_nodes = gov.tree_nodes();
                    evo_snapshot.current_node_id = gov.current_node_id;
                    evo_snapshot
                        .fitness_history
                        .clone_from(gov.fitness_history_by_island());
                    evo_snapshot.best_fitness = gov.best_score();
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
            orbit_mode: false,
            readback_logged: false,
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
            heatmap_dirty: false,
            trail_gpu: None,
            last_mesh_rebuild: Instant::now(),
            last_snapshot_rebuild: Instant::now(),
            snap_dirty: true,
            cached_agent_snaps: Vec::new(),
            last_snap_chart_window: 120,
            marker_gpu: None,
            egui: None,
            console_log: VecDeque::new(),
            governor: None,
            evo_snapshot,
            evo_wall_accumulated: 0.0,
            evo_wall_segment_start: None,
            tps_tick_count: 0,
            tps_last_reset: Instant::now(),
            tps_display: 0.0,
            db_path: db_path.to_string(),
            governor_config,
            gen_transition: None,
            gpu_kernel: None,
            pending_kernel: None,
            pending_upload: None,
            deferred_inherited: None,
            dock_state: egui_dock::DockState::new(vec![Tab::Evolution, Tab::Sandbox]),
            gpu_tick_budget: 32,
            sort_mode: SortMode::Id,
            world_snapshot: WorldSnapshot::default(),
            recording: None,
            last_recording: None,
            replay_state: ReplayState::default(),
        }
    }

    /// Spawn a new agent with the given BrainConfig at a safe random position
    /// (not in a danger biome). The brain_idx is the agent's array index.
    fn spawn_agent(&mut self, config: BrainConfig, generation: u32) {
        if self.agents.len() >= MAX_AGENTS {
            return;
        }
        let Some(world) = &self.world else { return };

        let pos = world.safe_spawn_position();

        let id = self.next_agent_id;
        self.next_agent_id += 1;

        let brain_idx = self.agents.len() as u32;
        let mut agent = Agent::new(id, pos, brain_idx, config, self.tick);
        agent.generation = generation;

        self.agents.push(agent);
    }

    /// Ensure GpuKernel is initialized for the current population.
    fn ensure_gpu_kernel(&mut self) {
        if self.gpu_kernel.is_some() {
            return;
        }
        if self.agents.is_empty() {
            return;
        }
        let world = match &self.world {
            Some(w) => w,
            None => return,
        };

        // Check if background creation finished.
        if let Some(ref handle) = self.pending_kernel {
            if handle.is_finished() {
                let handle = self.pending_kernel.take().unwrap();
                let mk = handle
                    .join()
                    .expect("fused kernel background thread panicked");
                // Upload world + agent data (fast, main thread).
                if let Some(upload) = self.pending_upload.take() {
                    mk.upload_world(
                        &upload.heights,
                        &upload.biomes,
                        &upload.food_pos,
                        &upload.food_consumed,
                        &upload.food_timers,
                    );
                    mk.upload_agents(&upload.agent_data);
                }
                let ac = mk.agent_count();

                // Apply deferred inherited state from a generation transition
                // that occurred while the kernel was being recreated.
                if let Some((ref state, mutation_strength)) = self.deferred_inherited.take() {
                    let repeats = self.governor_config.eval_repeats.max(1);
                    let n = self.agents.len();
                    let champion = state.clone();
                    mk.batch_write_agent_states(n, |i| {
                        if i < repeats {
                            champion.clone()
                        } else {
                            mutate_brain_state(state, mutation_strength)
                        }
                    });
                }

                self.gpu_kernel = Some(mk);
                self.log_msg(format!("[GPU] GpuKernel ready ({} agents)", ac));
            }
            return; // still creating
        }

        // Collect data for upload (kept on main thread).
        let agent_count = self.agents.len() as u32;
        let food_count = world.food_items.len();
        let brain_config = self.brain_config.clone();
        let world_config = world.config.clone();

        self.pending_upload = Some(PendingUpload {
            heights: world.terrain.heights.clone(),
            biomes: world.biome_map.grid_as_u32(),
            food_pos: world
                .food_items
                .iter()
                .map(|f| (f.position.x, f.position.y, f.position.z))
                .collect(),
            food_consumed: world.food_items.iter().map(|f| f.consumed).collect(),
            food_timers: world.food_items.iter().map(|f| f.respawn_timer).collect(),
            agent_data: self
                .agents
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
                .collect(),
        });

        // Spawn background thread for device + shader compilation.
        self.pending_kernel = Some(std::thread::spawn(move || {
            GpuKernel::new(agent_count, food_count, &brain_config, &world_config)
        }));
        self.log_msg("[GPU] Creating GpuKernel (background)...".into());
    }

    fn handle_evolution_action(&mut self, action: EvolutionAction) {
        match action {
            EvolutionAction::None => {}
            EvolutionAction::Start => {
                let brain_config = self.evo_snapshot.edit_brain.clone();
                let gov_config = self.evo_snapshot.edit_governor.clone();
                let world_json = serde_json::to_string(&self.world_config).unwrap_or_default();
                match Governor::new(
                    &self.db_path,
                    gov_config.clone(),
                    &brain_config,
                    &world_json,
                ) {
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
                        self.evo_wall_accumulated = 0.0;
                        self.evo_wall_segment_start = Some(Instant::now());
                        self.tps_tick_count = 0;
                        self.tps_last_reset = Instant::now();
                        self.tps_display = 0.0;
                        self.paused = false;
                        self.gpu_kernel = None;
                        self.pending_kernel = None;
                        self.pending_upload = None;
                        self.spawn_evolution_population();
                        self.ensure_gpu_kernel();
                        self.log_msg("[EVOLUTION] Started new run".into());
                    }
                    Err(e) => {
                        self.log_msg(format!("[EVOLUTION] Failed to start: {}", e));
                    }
                }
            }
            EvolutionAction::Resume => match Governor::resume(&self.db_path) {
                Ok(gov) => {
                    let cfg = gov.current_config();
                    self.evo_snapshot.state = EvolutionState::Running;
                    self.evo_snapshot.generation = gov.generation;
                    self.governor_config = gov.config.clone();
                    if let Some(c) = &cfg {
                        self.brain_config = c.clone();
                    }
                    self.governor = Some(gov);
                    self.evo_wall_accumulated = 0.0;
                    self.evo_wall_segment_start = Some(Instant::now());
                    self.tps_tick_count = 0;
                    self.tps_last_reset = Instant::now();
                    self.tps_display = 0.0;
                    self.paused = false;
                    self.gpu_kernel = None;
                    self.pending_kernel = None;
                    self.pending_upload = None;
                    self.spawn_evolution_population();
                    self.ensure_gpu_kernel();
                    self.log_msg("[EVOLUTION] Resumed from database".into());
                }
                Err(e) => {
                    self.log_msg(format!("[EVOLUTION] Failed to resume: {}", e));
                }
            },
            EvolutionAction::Pause => {
                self.evo_snapshot.state = EvolutionState::Paused;
                self.paused = true;
                self.snap_dirty = true;
                if let Some(start) = self.evo_wall_segment_start.take() {
                    self.evo_wall_accumulated += start.elapsed().as_secs_f64();
                }
                self.tps_display = 0.0;
                self.log_msg("[EVOLUTION] Paused".into());
            }
            EvolutionAction::Unpause => {
                self.evo_snapshot.state = EvolutionState::Running;
                self.paused = false;
                self.evo_wall_segment_start = Some(Instant::now());
                self.tps_tick_count = 0;
                self.tps_last_reset = Instant::now();
                self.log_msg("[EVOLUTION] Resumed".into());
            }
            EvolutionAction::Reset => {
                self.governor = None;
                self.gpu_kernel = None;
                self.pending_kernel = None;
                self.pending_upload = None;
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
        let repeats = self.governor_config.eval_repeats.max(1);
        let unique_count = (pop_size / repeats).max(1);

        // Build unique configs matching breed_next_generation structure
        // so that reduce_fitness grouping (agent_index / eval_repeats)
        // correctly averages same-config runs.
        let mut unique_configs = vec![seed.clone()]; // slot 0: champion
        for _ in 1..unique_count {
            unique_configs.push(mutate_config(&seed));
        }

        // Repeat each config eval_repeats times
        for uc in &unique_configs {
            for _ in 0..repeats {
                if self.agents.len() >= pop_size {
                    break;
                }
                self.spawn_agent(uc.clone(), 0);
            }
        }

        // Start replay recording
        if let Some(world) = &self.world {
            let agent_info: Vec<(u32, [f32; 3])> =
                self.agents.iter().map(|a| (a.id, a.color)).collect();
            let initial_food: Vec<[f32; 3]> = world
                .food_items
                .iter()
                .map(|f| [f.position.x, f.position.y, f.position.z])
                .collect();
            let gen = self.governor.as_ref().map_or(0, |g| g.generation as u32);
            self.recording = Some(xagent_sandbox::replay::GenerationRecording::new(
                gen,
                &agent_info,
                &initial_food,
                self.governor_config.tick_budget as usize,
                self.brain_config.vision_width,
                self.brain_config.vision_height,
            ));
        }
    }

    fn spawn_population_from_configs(&mut self, configs: &[BrainConfig]) {
        self.agents.clear();
        self.next_agent_id = 0;
        self.tick = 0;
        for cfg in configs {
            self.spawn_agent(cfg.clone(), 0);
        }

        // Start replay recording
        if let Some(world) = &self.world {
            let agent_info: Vec<(u32, [f32; 3])> =
                self.agents.iter().map(|a| (a.id, a.color)).collect();
            let initial_food: Vec<[f32; 3]> = world
                .food_items
                .iter()
                .map(|f| [f.position.x, f.position.y, f.position.z])
                .collect();
            let gen = self.governor.as_ref().map_or(0, |g| g.generation as u32);
            self.recording = Some(xagent_sandbox::replay::GenerationRecording::new(
                gen,
                &agent_info,
                &initial_food,
                self.governor_config.tick_budget as usize,
                self.brain_config.vision_width,
                self.brain_config.vision_height,
            ));
        }
    }

    /// Evaluate the current generation and begin the async transition.
    /// The actual GPU work (readback, reset, upload) spans multiple frames
    /// via `poll_gen_transition`.
    fn advance_generation(&mut self) {
        // Persist the recording to SQLite before moving to the next generation
        if let (Some(ref recording), Some(ref mut gov)) = (&self.recording, &mut self.governor) {
            gov.store_recording(recording);
        }
        self.last_recording = self.recording.take();
        let wall_secs = self.evo_wall_accumulated
            + self
                .evo_wall_segment_start
                .map(|s| s.elapsed().as_secs_f64())
                .unwrap_or(0.0);

        // Evaluate fitness and decide whether evolution continues.
        // Only kick off the async brain-state readback when the result is
        // Continue — when Finished, no readback is needed.
        let (result, readback_requested) = {
            let gov = match self.governor.as_mut() {
                Some(g) => g,
                None => return,
            };

            let fitness = gov.evaluate(&self.agents);
            gov.log_generation(&fitness);
            gov.update_wall_time(wall_secs);

            let result = gov.advance(&fitness);

            // Only read back the best agent's brain state when evolution
            // continues — Finished needs no inherited state.
            let mut readback_requested = false;
            if matches!(result, AdvanceResult::Continue { .. }) {
                let best_idx = fitness.first().map(|f| f.agent_index).unwrap_or(0);
                if let Some(a) = self.agents.get(best_idx) {
                    if let Some(mk) = self.gpu_kernel.as_mut() {
                        if mk.request_agent_state(a.brain_idx) {
                            readback_requested = true;
                        }
                    }
                }
            }

            (result, readback_requested)
        };

        match result {
            AdvanceResult::Continue { .. } if readback_requested => {
                self.gen_transition = Some(GenTransition::AwaitingReadback { result });
            }
            AdvanceResult::Continue { .. } => {
                // No readback was issued (no best agent or no GPU kernel) —
                // skip directly to reset to avoid deadlocking in AwaitingReadback.
                self.gen_transition = Some(GenTransition::AwaitingReset {
                    result,
                    inherited_state: None,
                    spawned: false,
                });
            }
            AdvanceResult::Finished { .. } => {
                // Skip readback phase entirely — go straight to finish.
                self.gen_transition = Some(GenTransition::AwaitingReset {
                    result,
                    inherited_state: None,
                    spawned: false,
                });
            }
        }
    }

    /// Drive the multi-frame generation transition state machine.
    /// Called every frame; returns true while a transition is still in progress.
    fn poll_gen_transition(&mut self) -> bool {
        let transition = match self.gen_transition.take() {
            Some(t) => t,
            None => return false,
        };

        match transition {
            GenTransition::AwaitingReadback { result } => {
                if self.gpu_kernel.is_none() {
                    // No kernel — proceed without inherited state.
                    self.gen_transition = Some(GenTransition::AwaitingReset {
                        result,
                        inherited_state: None,
                        spawned: false,
                    });
                    return true;
                }

                // Poll for the async agent state readback.
                let collected = self.gpu_kernel.as_mut().unwrap().try_collect_agent_state();

                match collected {
                    Some(Some(state)) => {
                        // Readback complete — move to reset phase.
                        self.gen_transition = Some(GenTransition::AwaitingReset {
                            result,
                            inherited_state: Some(state),
                            spawned: false,
                        });
                    }
                    Some(None) => {
                        // map_async error — proceed without inherited state
                        // rather than hanging in AwaitingReadback forever.
                        self.gen_transition = Some(GenTransition::AwaitingReset {
                            result,
                            inherited_state: None,
                            spawned: false,
                        });
                    }
                    None => {
                        // Still waiting — re-enqueue for next frame.
                        self.gen_transition = Some(GenTransition::AwaitingReadback { result });
                    }
                }
                true
            }
            GenTransition::AwaitingReset {
                result,
                inherited_state,
                spawned,
            } => {
                let (done, spawned) =
                    self.try_finish_generation(&result, inherited_state.as_ref(), spawned);
                if !done {
                    // Reset not ready yet — re-enqueue for next frame.
                    self.gen_transition = Some(GenTransition::AwaitingReset {
                        result,
                        inherited_state,
                        spawned,
                    });
                    return true;
                }
                false
            }
        }
    }

    /// Try to complete a generation transition. Returns `(done, spawned)`
    /// where `done` is false if the GPU staging buffers aren't ready yet
    /// (caller should retry next frame), and `spawned` tracks whether
    /// population has been spawned (to avoid re-spawning on retry).
    fn try_finish_generation(
        &mut self,
        result: &AdvanceResult,
        inherited_state: Option<&AgentBrainState>,
        mut spawned: bool,
    ) -> (bool, bool) {
        match result {
            AdvanceResult::Continue {
                configs,
                messages,
                mutation_strength,
            } => {
                if !spawned {
                    for msg in messages {
                        self.log_msg(msg.clone());
                    }
                    self.spawn_population_from_configs(configs);
                    spawned = true;
                }
                let repeats = self.governor_config.eval_repeats.max(1);

                // Reuse existing kernel if population size matches,
                // otherwise drop and recreate in background.
                let can_reuse = self
                    .gpu_kernel
                    .as_ref()
                    .is_some_and(|mk| mk.agent_count() == self.agents.len() as u32);

                if can_reuse {
                    let mk = self.gpu_kernel.as_mut().unwrap();
                    // Non-blocking reset — return false to retry next frame
                    // instead of falling back to a blocking busy-wait.
                    if !mk.try_reset_agents(&self.brain_config) {
                        return (false, spawned);
                    }
                    let agent_data: Vec<_> = self
                        .agents
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
                    mk.upload_agents(&agent_data);
                } else {
                    self.gpu_kernel = None;
                    self.pending_kernel = None;
                    self.pending_upload = None;
                    self.ensure_gpu_kernel();
                    // Kernel is being recreated in the background —
                    // keep the transition active until it's ready.
                    if self.gpu_kernel.is_none() {
                        return (false, spawned);
                    }
                }

                // Inherit learned weights: champions get exact brain state,
                // mutants get perturbed weights for neuroevolution.
                if let Some(state) = inherited_state {
                    if let Some(ref mk) = self.gpu_kernel {
                        let ms = *mutation_strength;
                        let n = self.agents.len();
                        // Pre-clone once for all champion slots to avoid
                        // cloning inside the hot closure on every call.
                        let champion = state.clone();
                        mk.batch_write_agent_states(n, |i| {
                            if i < repeats {
                                champion.clone()
                            } else {
                                mutate_brain_state(state, ms)
                            }
                        });
                    } else {
                        // Kernel is being recreated in the background
                        // (population size changed). Stash inherited state
                        // so it can be applied once the kernel is ready.
                        self.deferred_inherited = Some((state.clone(), *mutation_strength));
                    }
                }
            }
            AdvanceResult::Finished { messages } => {
                for msg in messages {
                    self.log_msg(msg.clone());
                }
                self.evo_snapshot.state = EvolutionState::Paused;
                self.paused = true;
            }
        }
        (true, spawned)
    }

    fn log_msg(&mut self, message: String) {
        println!("{}", message);
        self.console_log.push_back(message);
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
        let parent_config = parent.brain_config.clone();

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

        let brain_idx = self.agents.len() as u32;
        let mut child = Agent::new(
            id,
            Vec3::new(cx, cy, cz),
            brain_idx,
            child_config,
            self.tick,
        );
        child.generation = parent_gen + 1;

        println!(
            "[REPRODUCE] Agent {} (gen {}) → child Agent {} (gen {}) at ({:.1}, {:.1})",
            parent_id, parent_gen, id, child.generation, cx, cz
        );
        println!(
            "  Child config: cap={} slots={} dim={} lr={:.4} decay={:.4}",
            child.brain_config.memory_capacity,
            child.brain_config.processing_slots,
            child.brain_config.representation_dim,
            child.brain_config.learning_rate,
            child.brain_config.decay_rate,
        );

        self.agents.push(child);
    }

    /// Pick the agent closest to the cursor via screen-space projection.
    fn pick_agent_at_cursor(&mut self) {
        let Some(renderer) = &self.renderer else {
            return;
        };
        let w = renderer.config.width as f32;
        let h = renderer.config.height as f32;
        if w < 1.0 || h < 1.0 {
            return;
        }

        // Normalized device coordinates [-1, 1]
        let ndc_x = (self.cursor_pos.0 as f32 / w) * 2.0 - 1.0;
        let ndc_y = 1.0 - (self.cursor_pos.1 as f32 / h) * 2.0;

        let vp = self.camera.view_projection_matrix();
        let mut best_idx: Option<usize> = None;
        let mut best_dist_sq = f32::MAX;

        for (i, agent) in self.agents.iter().enumerate() {
            if !agent.body.body.alive {
                continue;
            }
            let pos = agent.body.body.position;
            let clip = vp * glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);
            if clip.w <= 0.0 {
                continue;
            } // behind camera
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
                println!("[SELECT] Agent {} (gen {})", a.id, a.generation);
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
        let pred_err = agent.cached_prediction_error;
        let explore = agent.cached_exploration_rate;

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

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );
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

        // Trail overlay: combined ribbons for ALL agents.
        // Each segment = 4 verts, 6 indices.
        let max_trail_segs = xagent_sandbox::agent::MAX_TRAIL_POINTS as u64 * MAX_AGENTS as u64;
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

        println!("[CONTROLS] P/Space = pause | 1-6 = speed | G = toggle 3D");
        println!("[CONTROLS] N = spawn agent | M = spawn mutated agent | Tab = cycle telemetry");
        info!("Renderer + world + brain initialized — agent is alive");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Forward every event to egui first.
        // If egui consumed it (pointer over a panel, typing in a text field, etc.)
        // we skip our own camera/sim key handling for that event.
        let egui_consumed = if let (Some(egui), Some(window)) = (&mut self.egui, &self.window) {
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
                    self.camera.aspect = new_size.width as f32 / new_size.height.max(1) as f32;
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
                    PhysicalKey::Code(KeyCode::Digit7) if pressed => {
                        self.speed_multiplier = 10000;
                        println!(
                            "[SIM] Speed: 10000x ({} ticks/sec)",
                            SIM_RATE as u32 * 10000
                        );
                    }
                    PhysicalKey::Code(KeyCode::Digit8) if pressed => {
                        self.speed_multiplier = 100000;
                        println!(
                            "[SIM] Speed: 100000x ({} ticks/sec)",
                            SIM_RATE as u32 * 100000
                        );
                    }
                    PhysicalKey::Code(KeyCode::Digit9) if pressed => {
                        self.speed_multiplier = 1000000;
                        println!(
                            "[SIM] Speed: 1000000x ({} ticks/sec)",
                            SIM_RATE as u32 * 1000000
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyH) if pressed => {
                        self.heatmap_enabled = !self.heatmap_enabled;
                        self.heatmap_dirty = true;
                        println!(
                            "[SIM] Heatmap: {}",
                            if self.heatmap_enabled { "ON" } else { "OFF" }
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyG) if pressed => {
                        self.render_3d = !self.render_3d;
                        self.log_msg(format!(
                            "[SIM] 3D render: {}",
                            if self.render_3d {
                                "ON"
                            } else {
                                "OFF (fast mode)"
                            }
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
                if button == MouseButton::Right {
                    self.camera.is_mouse_dragging = state == ElementState::Pressed;
                    if state == ElementState::Released {
                        self.camera.last_mouse_pos = None;
                    }
                }
                // Left-click to select nearest agent
                if button == MouseButton::Left && state == ElementState::Released {
                    self.pick_agent_at_cursor();
                }
            }

            WindowEvent::CursorMoved { position, .. }
                if pointer_on_viewport || self.camera.is_mouse_dragging =>
            {
                self.cursor_pos = (position.x, position.y);
                if self.camera.orbit_mode {
                    self.camera.process_orbit_mouse_move(position.x, position.y);
                } else {
                    self.camera.process_mouse_move(position.x, position.y);
                }
            }

            WindowEvent::MouseWheel { delta, .. } if pointer_on_viewport => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                if self.camera.orbit_mode {
                    self.camera.process_orbit_scroll(scroll);
                } else {
                    self.camera.process_scroll(scroll);
                }
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
                // Detect orbit toggle: initialize orbit parameters from current camera
                if self.orbit_mode && !self.camera.orbit_mode {
                    if let Some(agent) = self.agents.get(self.selected_agent_idx) {
                        let diff = self.camera.position - agent.body.body.position;
                        let len = diff.length().max(0.001);
                        self.camera.orbit_distance = len.clamp(5.0, 200.0);
                        self.camera.orbit_yaw = diff.z.atan2(diff.x);
                        self.camera.orbit_pitch = (diff.y / len).asin().clamp(0.05, 1.4);
                    }
                }
                self.camera.orbit_mode = self.orbit_mode;
                if self.camera.orbit_mode {
                    if let Some(agent) = self.agents.get(self.selected_agent_idx) {
                        let target = agent.body.body.position;
                        self.camera.update_orbit(target);
                    }
                } else {
                    self.camera.update(dt);
                }

                // Ensure background kernel creation is started if needed and
                // collected when ready, even while the tick gate below
                // suppresses accumulation.
                self.ensure_gpu_kernel();

                // ── simulation ticks (fixed timestep, adaptive budget) ──
                // Skip ticks while a generation transition is in flight to
                // avoid GPU contention with the async readback/reset.
                // Also skip accumulation when the kernel is being recreated
                // in the background to prevent catch-up hitch when it lands.
                if !self.paused && self.gen_transition.is_none() && self.pending_kernel.is_none() {
                    self.sim_accumulator += dt * self.speed_multiplier as f32;
                    // Cap accumulator to 2× budget so debt stays bounded.
                    let max_acc = SIM_DT * self.gpu_tick_budget as f32 * 2.0;
                    self.sim_accumulator = self.sim_accumulator.min(max_acc);
                    let ticks_to_run =
                        ((self.sim_accumulator / SIM_DT) as u32).min(self.gpu_tick_budget);

                    if ticks_to_run > 0 {
                        if let Some(ref mut mk) = self.gpu_kernel {
                            let state_updated = mk.try_collect_state();
                            let t0 = Instant::now();
                            let dispatched = mk.dispatch_batch(self.tick, ticks_to_run);
                            let wall_ms = t0.elapsed().as_secs_f32() * 1000.0;

                            if dispatched {
                                // Grow budget aggressively — staging double-buffer
                                // is the real throttle (dispatch_batch returns false
                                // when both are in-flight). Only shrink if CPU
                                // encoding takes so long it stalls the render loop.
                                if wall_ms > 50.0 {
                                    self.gpu_tick_budget = (self.gpu_tick_budget * 3 / 4).max(32);
                                } else {
                                    self.gpu_tick_budget =
                                        (self.gpu_tick_budget + self.gpu_tick_budget / 4 + 1)
                                            .min(64_000);
                                }

                                self.tick += ticks_to_run as u64;
                                self.tps_tick_count += ticks_to_run as u64;
                                self.sim_accumulator -= SIM_DT * ticks_to_run as f32;
                                self.snap_dirty = true;

                                if let Some(gov) = &mut self.governor {
                                    for _ in 0..ticks_to_run {
                                        gov.tick();
                                    }
                                }
                            }
                            // else: GPU backpressure — skip this frame, retry next

                            if state_updated || mk.try_collect_state() {
                                let state = mk.cached_state();
                                for i in 0..self.agents.len() {
                                    let base = i * PHYS_STRIDE;
                                    if base + P_URGENCY_OUT >= state.len() {
                                        break;
                                    }
                                    let a = &mut self.agents[i];
                                    a.body.body.position = Vec3::new(
                                        state[base + P_POS_X],
                                        state[base + P_POS_Y],
                                        state[base + P_POS_Z],
                                    );
                                    a.body.body.alive = state[base + P_ALIVE] > 0.5;
                                    a.body.yaw = state[base + P_YAW];
                                    a.body.body.internal.energy = state[base + P_ENERGY];
                                    a.body.body.internal.integrity = state[base + P_INTEGRITY];
                                    a.body.body.internal.max_energy = state[base + P_MAX_ENERGY];
                                    a.body.body.internal.max_integrity =
                                        state[base + P_MAX_INTEGRITY];
                                    a.body.body.velocity = Vec3::new(
                                        state[base + P_VEL_X],
                                        state[base + P_VEL_Y],
                                        state[base + P_VEL_Z],
                                    );
                                    a.food_consumed = state[base + P_FOOD_COUNT] as u32;
                                    a.total_ticks_alive = state[base + P_TICKS_ALIVE] as u64;
                                    let new_deaths = state[base + P_DEATH_COUNT] as u32;
                                    if new_deaths > a.death_count {
                                        a.reset_trail();
                                    }
                                    a.death_count = new_deaths;
                                    a.body.body.facing = Vec3::new(
                                        state[base + P_FACING_X],
                                        state[base + P_FACING_Y],
                                        state[base + P_FACING_Z],
                                    );
                                    a.cached_prediction_error = state[base + P_PREDICTION_ERROR];
                                    a.cached_exploration_rate =
                                        state[base + P_EXPLORATION_RATE_OUT];
                                    a.cached_fatigue_factor = state[base + P_FATIGUE_FACTOR_OUT];
                                    a.cached_motor.forward = state[base + P_MOTOR_FWD_OUT];
                                    a.cached_motor.turn = state[base + P_MOTOR_TURN_OUT];
                                    a.cached_gradient = state[base + P_GRADIENT_OUT];
                                    a.cached_urgency = state[base + P_URGENCY_OUT];
                                }

                                // Diagnostic: log first readback Y vs terrain height
                                if !self.readback_logged {
                                    self.readback_logged = true;
                                    if let Some(world) = &self.world {
                                        let n = self.agents.len().min(5);
                                        for i in 0..n {
                                            let a = &self.agents[i];
                                            let p = a.body.body.position;
                                            let terrain_y = world.terrain.height_at(p.x, p.z);
                                            println!(
                                                "[TERRAIN-DIAG] Agent {} pos=({:.2}, {:.2}, {:.2}) terrain_y={:.2} diff={:.2}",
                                                i, p.x, p.y, p.z, terrain_y, p.y - terrain_y,
                                            );
                                        }
                                    }
                                }
                            }

                            // Async telemetry readback for the selected agent.
                            // `request_agent_telemetry` is gated internally: it no-ops
                            // if a readback for the same agent is already pending, and
                            // clears the old pending if the agent changed.
                            if self.selected_agent_idx < self.agents.len() {
                                let brain_idx = self.agents[self.selected_agent_idx].brain_idx;

                                mk.request_agent_telemetry(brain_idx);

                                // Collect any completed readback (non-blocking)
                                if let Some(tel) = mk.try_collect_telemetry() {
                                    let a = &mut self.agents[self.selected_agent_idx];
                                    // Only update fields NOT already populated by
                                    // the fresher async physics readback above.
                                    // Phys readback already sets: cached_motor,
                                    // cached_gradient, cached_urgency,
                                    // cached_fatigue_factor, cached_prediction_error,
                                    // cached_exploration_rate.
                                    a.cached_frame.vision.color = tel.vision_color;
                                    a.cached_mean_attenuation = tel.mean_attenuation;
                                    a.cached_curiosity_bonus = tel.curiosity_bonus;
                                    a.cached_motor_variance = tel.motor_variance;
                                }
                            }

                            self.food_dirty = true;

                            // Record tick for replay using the latest async physics
                            // readback plus per-agent cached telemetry. Position/yaw/
                            // alive/energy/integrity come from the physics readback
                            // stored on each agent body. Motor outputs, gradient/
                            // urgency, and the exploration/prediction/attenuation/
                            // curiosity/fatigue/motor-variance fields come from
                            // `cached_*` telemetry; in this path only the selected
                            // agent's cache is refreshed each frame.
                            // `credit_magnitude`, `patterns_recalled`, `phase`, and
                            // `vision_color` are not populated here and are left as
                            // defaults. `raw_gradient` mirrors `gradient` because
                            // there is no separate raw gradient source in this GPU
                            // readback path.
                            if let Some(ref mut rec) = self.recording {
                                // Async readback snapshots arrive independently of
                                // governor tick advancement, so record them at the
                                // next dense replay index.
                                let tick = rec.total_ticks;
                                let records: Vec<xagent_sandbox::replay::TickRecord> = self.agents.iter().map(|a| {
                                    xagent_sandbox::replay::TickRecord {
                                        position: [a.body.body.position.x, a.body.body.position.y, a.body.body.position.z],
                                        yaw: a.body.yaw,
                                        alive: a.body.body.alive,
                                        energy: a.body.body.internal.energy,
                                        integrity: a.body.body.internal.integrity,
                                        motor_forward: a.cached_motor.forward,
                                        motor_turn: a.cached_motor.turn,
                                        exploration_rate: a.cached_exploration_rate,
                                        prediction_error: a.cached_prediction_error,
                                        gradient: a.cached_gradient,
                                        raw_gradient: a.cached_gradient,
                                        urgency: a.cached_urgency,
                                        credit_magnitude: 0.0,
                                        patterns_recalled: 0,
                                        phase: xagent_sandbox::replay::GenerationRecording::phase_to_u8("RANDOM"),
                                        mean_attenuation: a.cached_mean_attenuation,
                                        curiosity_bonus: a.cached_curiosity_bonus,
                                        fatigue_factor: a.cached_fatigue_factor,
                                        motor_variance: a.cached_motor_variance,
                                        vision_color: None,
                                    }
                                }).collect();
                                rec.record_tick(tick, &records);
                            }
                        }

                        // Heatmap + trail recording
                        if let Some(world) = &self.world {
                            for agent in &mut self.agents {
                                if agent.body.body.alive {
                                    agent.record_heatmap(world.config.world_size);
                                    agent.record_trail();
                                }
                            }
                        }

                        // Sparkline histories + CSV logging
                        {
                            if let Some(world) = &self.world {
                                for agent in &mut self.agents {
                                    record_agent_histories(agent);
                                }
                                if self.selected_agent_idx < self.agents.len() {
                                    let agent = &self.agents[self.selected_agent_idx];
                                    let life_ticks = agent.age(self.tick);
                                    let motor = agent.cached_motor.clone();
                                    log_tick_to_csv(
                                        &mut self.logger,
                                        agent,
                                        world,
                                        &motor,
                                        life_ticks,
                                    );
                                    self.error_count += 1;
                                }
                            }
                        }

                        self.heatmap_dirty = true;
                        self.hud_dirty = true;
                    }

                    // ── Generation completion check (after tick batch) ──
                    if self.gen_transition.is_none() {
                        if let Some(gov) = &self.governor {
                            if gov.generation_complete() {
                                self.advance_generation();
                            }
                        }
                    }
                }

                // Drive the multi-frame generation transition state machine
                // (runs even when sim ticks are paused/skipped due to transition).
                self.poll_gen_transition();

                // Advance replay playback
                if self.replay_state.active && self.replay_state.playing {
                    let advance = (self.replay_state.speed as u64).max(1);
                    self.replay_state.current_tick = (self.replay_state.current_tick + advance)
                        .min(self.replay_state.total_ticks.saturating_sub(1));
                    if self.replay_state.current_tick
                        >= self.replay_state.total_ticks.saturating_sub(1)
                    {
                        self.replay_state.playing = false;
                    }
                }

                // Fix selected index if agents were removed
                if !self.agents.is_empty() {
                    self.selected_agent_idx = self.selected_agent_idx.min(self.agents.len() - 1);
                }

                // ── rebuild dynamic meshes (throttled to ~10 Hz) ──────
                let mesh_rebuild_due =
                    self.paused || self.last_mesh_rebuild.elapsed() >= REBUILD_THROTTLE;

                let mut did_rebuild = false;

                if self.food_dirty && mesh_rebuild_due {
                    if let (Some(renderer), Some(world), Some(food_gpu)) =
                        (&self.renderer, &self.world, &mut self.food_gpu)
                    {
                        let fm = world.food_mesh();
                        food_gpu.update_from_mesh(&renderer.queue, &fm);
                        self.food_dirty = false;
                        did_rebuild = true;
                    }
                }

                // ── rebuild heatmap overlay ─────────────────────────
                if self.heatmap_enabled && self.heatmap_dirty {
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
                    self.heatmap_dirty = false;
                } else if !self.heatmap_enabled {
                    if let Some(heatmap_gpu) = &mut self.heatmap_gpu {
                        heatmap_gpu.num_indices = 0;
                    }
                }

                // ── rebuild trail overlay for ALL agents (throttled) ──
                if mesh_rebuild_due {
                    if let (Some(renderer), Some(trail_gpu)) = (&self.renderer, &mut self.trail_gpu)
                    {
                        let any_dirty = self.agents.iter().any(|a| a.trail_dirty);
                        if any_dirty {
                            let agent_data: Vec<(&[[f32; 3]], &[f32; 3], bool)> = self
                                .agents
                                .iter()
                                .map(|a| {
                                    (a.trail.as_slice(), &a.color as &[f32; 3], a.body.body.alive)
                                })
                                .collect();
                            let mesh = overlay::build_all_trails_mesh(&agent_data);
                            if mesh.indices.is_empty() {
                                trail_gpu.num_indices = 0;
                            } else {
                                trail_gpu.update_from_mesh(&renderer.queue, &mesh);
                            }
                            for a in &mut self.agents {
                                a.trail_dirty = false;
                            }
                            did_rebuild = true;
                        }
                    }
                }

                if did_rebuild && !self.paused {
                    self.last_mesh_rebuild = Instant::now();
                }

                // ── rebuild selection marker above focused agent ──────
                if let (Some(renderer), Some(marker_gpu)) = (&self.renderer, &mut self.marker_gpu) {
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
                                [
                                    srgb_to_linear(0.25),
                                    srgb_to_linear(0.25),
                                    srgb_to_linear(0.25),
                                ]
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
                        let pred_err = agent.cached_prediction_error;
                        let explore = agent.cached_exploration_rate;

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
                            text: "Phase: GPU (TBD)".to_string(),
                            x: -0.98,
                            y: info_y - 0.035,
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
                    if let Some(t) = t {
                        mesh_vec.push(t);
                    }
                    if let Some(f) = f {
                        mesh_vec.push(f);
                    }
                    if let Some(h) = h {
                        mesh_vec.push(h);
                    }
                    if let Some(tr) = tr {
                        mesh_vec.push(tr);
                    }
                    if let Some(mk) = mk {
                        mesh_vec.push(mk);
                    }

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
                            if let (Some(egui), Some(window)) = (&mut self.egui, &self.window) {
                                let screen = egui_wgpu::ScreenDescriptor {
                                    size_in_pixels: [renderer.config.width, renderer.config.height],
                                    pixels_per_point: window.scale_factor() as f32,
                                };

                                let fps = self.fps;
                                let wall_time_secs = self.evo_snapshot.wall_time_secs;
                                let ticks_per_sec = self.evo_snapshot.ticks_per_sec;
                                let render_3d = self.render_3d;
                                let speed_multiplier = self.speed_multiplier;
                                let evo_state = self.evo_snapshot.state.clone();
                                let viewport_tex_id = egui.viewport_texture_id;
                                let ppp = window.scale_factor() as f32;
                                let mut desired_vp = (0u32, 0u32);
                                let selected_idx = self.selected_agent_idx;

                                // Snapshot agent data for the UI closure (throttled to ~10 Hz)
                                // Force rebuild when agents changed size or chart window changed.
                                if self.cached_agent_snaps.len() != self.agents.len()
                                    || self.chart_window != self.last_snap_chart_window
                                {
                                    self.snap_dirty = true;
                                }
                                let snap_rebuild_due = self.snap_dirty
                                    && (self.paused
                                        || self.last_snapshot_rebuild.elapsed()
                                            >= REBUILD_THROTTLE);
                                if snap_rebuild_due {
                                    let snap_window = self.chart_window * 2;
                                    self.cached_agent_snaps = self
                                        .agents
                                        .iter()
                                        .map(|a| {
                                            let tail =
                                                |d: &std::collections::VecDeque<f32>| -> Vec<f32> {
                                                    let skip = d.len().saturating_sub(snap_window);
                                                    d.iter().skip(skip).copied().collect()
                                                };
                                            AgentSnapshot {
                                                id: a.id,
                                                generation: a.generation,
                                                energy: a.body.body.internal.energy,
                                                max_energy: a.body.body.internal.max_energy,
                                                integrity: a.body.body.internal.integrity,
                                                max_integrity: a.body.body.internal.max_integrity,
                                                alive: a.body.body.alive,
                                                deaths: a.death_count,
                                                color: a.color,
                                                longest_life: a.longest_life,
                                                exploration_rate: a.cached_exploration_rate,
                                                prediction_error: a.cached_prediction_error,
                                                forward_weight_norm: 0.0, // GPU telemetry TBD
                                                turn_weight_norm: 0.0,    // GPU telemetry TBD
                                                prediction_error_history: tail(
                                                    &a.prediction_error_history,
                                                ),
                                                exploration_rate_history: tail(
                                                    &a.exploration_rate_history,
                                                ),
                                                energy_history: tail(&a.energy_history),
                                                integrity_history: tail(&a.integrity_history),
                                                gradient: a.cached_gradient,
                                                urgency: a.cached_urgency,
                                                food_consumed: a.food_consumed,
                                                total_ticks_alive: a.total_ticks_alive,
                                                motor_forward: a.cached_motor.forward,
                                                motor_turn: a.cached_motor.turn,
                                                phase: "GPU", // GPU telemetry TBD
                                                vision_color: a.cached_frame.vision.color.clone(),
                                                vision_width: a.cached_frame.vision.width,
                                                vision_height: a.cached_frame.vision.height,
                                                position: [
                                                    a.body.body.position.x,
                                                    a.body.body.position.y,
                                                    a.body.body.position.z,
                                                ],
                                                yaw: a.body.yaw,
                                                mean_attenuation: a.cached_mean_attenuation,
                                                curiosity_bonus: a.cached_curiosity_bonus,
                                                fatigue_factor: a.cached_fatigue_factor,
                                                motor_variance: a.cached_motor_variance,
                                                fatigue_history: tail(&a.fatigue_history),
                                            }
                                        })
                                        .collect();
                                    self.snap_dirty = false;
                                    self.last_snap_chart_window = self.chart_window;
                                    if !self.paused {
                                        self.last_snapshot_rebuild = Instant::now();
                                    }
                                }
                                let agent_snaps = self.cached_agent_snaps.as_slice();

                                // Build evolution snapshot for the UI
                                if let Some(gov) = &mut self.governor {
                                    self.evo_snapshot.gen_tick = gov.gen_tick;
                                    self.evo_snapshot.generation = gov.generation;
                                    self.evo_snapshot.tree_nodes = gov.tree_nodes();
                                    self.evo_snapshot.current_node_id = gov.current_node_id;
                                    self.evo_snapshot
                                        .fitness_history
                                        .clone_from(gov.fitness_history_by_island());
                                    self.evo_snapshot.best_fitness = gov.best_score();
                                }
                                let wall = self.evo_wall_accumulated
                                    + self
                                        .evo_wall_segment_start
                                        .map(|s| s.elapsed().as_secs_f64())
                                        .unwrap_or(0.0);
                                self.evo_snapshot.wall_time_secs = wall;
                                let tps_elapsed = self.tps_last_reset.elapsed().as_secs_f64();
                                if tps_elapsed >= 1.0 {
                                    self.tps_display = self.tps_tick_count as f64 / tps_elapsed;
                                    self.tps_tick_count = 0;
                                    self.tps_last_reset = Instant::now();
                                }
                                self.evo_snapshot.ticks_per_sec = self.tps_display;
                                // Move snapshot out so we can pass &mut to the closure
                                let mut evo_snap = std::mem::take(&mut self.evo_snapshot);
                                let gen_tick = evo_snap.gen_tick;
                                let tick_budget = evo_snap.tick_budget;
                                let evo_generation = evo_snap.generation;
                                let best_fitness = evo_snap.best_fitness;
                                let mut evo_action = EvolutionAction::None;

                                // Build world snapshot for mini-map
                                if let Some(world) = &self.world {
                                    self.world_snapshot.food_positions = world
                                        .food_items
                                        .iter()
                                        .filter(|f| !f.consumed)
                                        .map(|f| [f.position.x, f.position.z])
                                        .collect();
                                    self.world_snapshot.world_size = world.config.world_size;
                                }
                                let mut world_snap = std::mem::take(&mut self.world_snapshot);
                                // Pre-build biome image pixels if texture not yet created
                                let mut biome_image = if world_snap.biome_texture.is_none() {
                                    self.world.as_ref().map(|world| {
                                        use xagent_sandbox::world::biome::BiomeType;
                                        let res = 256usize;
                                        let ws = world.config.world_size;
                                        let half = ws / 2.0;
                                        let cell = ws / res as f32;
                                        let mut pixels = Vec::with_capacity(res * res);
                                        for row in 0..res {
                                            let z = -half + (row as f32 + 0.5) * cell;
                                            for col in 0..res {
                                                let x = -half + (col as f32 + 0.5) * cell;
                                                let biome = world.biome_map.biome_at(x, z);
                                                let c = match biome {
                                                    BiomeType::FoodRich => {
                                                        egui::Color32::from_rgb(25, 70, 20)
                                                    }
                                                    BiomeType::Barren => {
                                                        egui::Color32::from_rgb(60, 50, 30)
                                                    }
                                                    BiomeType::Danger => {
                                                        egui::Color32::from_rgb(80, 25, 15)
                                                    }
                                                };
                                                pixels.push(c);
                                            }
                                        }
                                        egui::ColorImage {
                                            size: [res, res],
                                            pixels,
                                        }
                                    })
                                } else {
                                    None
                                };

                                let console_lines: Vec<&str> =
                                    self.console_log.iter().map(|s| s.as_str()).collect();

                                let mut clicked_agent_idx: Option<usize> = None;
                                let mut open_agent_tab: Option<u32> = None;
                                let mut vp_hovered = false;
                                let mut chart_win = self.chart_window;
                                let mut sort_mode = self.sort_mode;
                                let dock_state = &mut self.dock_state;
                                let replay_state = &mut self.replay_state;
                                let last_recording = self.last_recording.as_ref();
                                let orbit_mode = &mut self.orbit_mode;

                                egui.render(
                                    window,
                                    &renderer.device,
                                    &renderer.queue,
                                    &mut frame_ctx.encoder,
                                    &frame_ctx.view,
                                    screen,
                                    |ctx| {
                                        // Build biome texture once (requires ctx)
                                        if let Some(image) = biome_image.take() {
                                            world_snap.biome_texture = Some(
                                                ctx.load_texture("biome_map", image, egui::TextureOptions::NEAREST)
                                            );
                                        }

                                        // ── Top bar ──────────────────────────
                                        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.label(
                                                    egui::RichText::new("xagent")
                                                        .strong()
                                                        .color(egui::Color32::from_rgb(120, 200, 255)),
                                                );
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
                                                if matches!(&evo_state, EvolutionState::Running | EvolutionState::Paused) {
                                                    ui.separator();
                                                    let hours = (wall_time_secs / 3600.0) as u64;
                                                    let mins = ((wall_time_secs % 3600.0) / 60.0) as u64;
                                                    let secs = (wall_time_secs % 60.0) as u64;
                                                    ui.label(format!("{}h {:02}m {:02}s", hours, mins, secs));
                                                    ui.separator();
                                                    let speed_label = speed_label(speed_multiplier);
                                                    ui.label(
                                                        egui::RichText::new(format!("⏩ {}", speed_label))
                                                            .color(if speed_multiplier > 1 {
                                                                egui::Color32::from_rgb(255, 200, 50)
                                                            } else {
                                                                egui::Color32::GRAY
                                                            }),
                                                    );
                                                    ui.separator();
                                                    ui.label(format!("{:.0} ticks/s", ticks_per_sec));
                                                }
                                                if matches!(&evo_state, EvolutionState::Running | EvolutionState::Paused) && best_fitness >= 0.0 {
                                                    ui.separator();
                                                    ui.label(
                                                        egui::RichText::new(format!("Best: {:.4}", best_fitness))
                                                            .color(egui::Color32::from_rgb(50, 200, 80)),
                                                    );
                                                }
                                                // Generation progress bar (compact, in toolbar)
                                                if matches!(&evo_state, EvolutionState::Running | EvolutionState::Paused) && tick_budget > 0 {
                                                    ui.separator();
                                                    let progress = gen_tick as f32 / tick_budget as f32;
                                                    ui.add(
                                                        egui::ProgressBar::new(progress)
                                                            .text(format!("Gen {} — {:.0}%", evo_generation, progress * 100.0))
                                                            .desired_width(160.0)
                                                            .animate(matches!(&evo_state, EvolutionState::Running)),
                                                    );
                                                }
                                                if !render_3d {
                                                    ui.separator();
                                                    ui.label(
                                                        egui::RichText::new("⚡ FAST (G)")
                                                            .color(egui::Color32::from_rgb(255, 160, 50)),
                                                    );
                                                }
                                                // ── Right-aligned controls ──
                                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                    if matches!(&evo_state, EvolutionState::Running | EvolutionState::Paused) {
                                                        if ui.add(egui::Button::new(
                                                            egui::RichText::new("🗑 Reset")
                                                                .color(egui::Color32::from_rgb(220, 80, 80)),
                                                        )).clicked() {
                                                            evo_action = EvolutionAction::Reset;
                                                        }
                                                        if matches!(&evo_state, EvolutionState::Running) {
                                                            if ui.button("⏸ Pause").clicked() {
                                                                evo_action = EvolutionAction::Pause;
                                                            }
                                                        } else {
                                                            if ui.button("▶ Resume").clicked() {
                                                                evo_action = EvolutionAction::Unpause;
                                                            }
                                                        }
                                                    }
                                                });
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
                                                            let color = if line.contains("ERROR") || line.contains("Failed to") {
                                                                egui::Color32::from_rgb(255, 100, 100)
                                                            } else if line.contains("best") || line.contains("Beat parent") {
                                                                egui::Color32::from_rgb(80, 220, 80)
                                                            } else if line.contains("Failed") || line.contains("exhausted") || line.contains("backtracking") {
                                                                egui::Color32::from_rgb(255, 140, 80)
                                                            } else if line.contains("Migration") {
                                                                egui::Color32::from_rgb(100, 180, 255)
                                                            } else if line.contains("Momentum") {
                                                                egui::Color32::from_rgb(180, 160, 255)
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
                                                ui.horizontal(|ui| {
                                                    ui.label(egui::RichText::new("Sort:").small().color(egui::Color32::GRAY));
                                                    egui::ComboBox::from_id_salt("agent_sort")
                                                        .selected_text(sort_mode.label())
                                                        .width(90.0)
                                                        .show_ui(ui, |ui| {
                                                            for mode in SortMode::ALL {
                                                                ui.selectable_value(&mut sort_mode, mode, mode.label());
                                                            }
                                                        });
                                                });
                                                ui.add_space(2.0);
                                                egui::ScrollArea::vertical().show(ui, |ui| {
                                                    let mut sorted_indices: Vec<usize> = (0..agent_snaps.len()).collect();
                                                    match sort_mode {
                                                        SortMode::Id => {} // already sorted by id
                                                        SortMode::Energy => {
                                                            sorted_indices.sort_by(|&a, &b| {
                                                                let ea = agent_snaps[a].energy / agent_snaps[a].max_energy.max(0.001);
                                                                let eb = agent_snaps[b].energy / agent_snaps[b].max_energy.max(0.001);
                                                                eb.partial_cmp(&ea).unwrap_or(std::cmp::Ordering::Equal)
                                                            });
                                                        }
                                                        SortMode::Integrity => {
                                                            sorted_indices.sort_by(|&a, &b| {
                                                                let ia = agent_snaps[a].integrity / agent_snaps[a].max_integrity.max(0.001);
                                                                let ib = agent_snaps[b].integrity / agent_snaps[b].max_integrity.max(0.001);
                                                                ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal)
                                                            });
                                                        }
                                                        SortMode::Deaths => {
                                                            sorted_indices.sort_by(|&a, &b| {
                                                                agent_snaps[a].deaths.cmp(&agent_snaps[b].deaths)
                                                            });
                                                        }
                                                        SortMode::LongestLife => {
                                                            sorted_indices.sort_by(|&a, &b| {
                                                                agent_snaps[b].longest_life.cmp(&agent_snaps[a].longest_life)
                                                            });
                                                        }
                                                        SortMode::PredictionError => {
                                                            sorted_indices.sort_by(|&a, &b| {
                                                                agent_snaps[a].prediction_error.partial_cmp(&agent_snaps[b].prediction_error)
                                                                    .unwrap_or(std::cmp::Ordering::Equal)
                                                            });
                                                        }
                                                        SortMode::Fitness => {
                                                            sorted_indices.sort_by(|&a, &b| {
                                                                let fa = agent_snaps[a].food_consumed as f64
                                                                    + agent_snaps[a].total_ticks_alive as f64 * 0.001;
                                                                let fb = agent_snaps[b].food_consumed as f64
                                                                    + agent_snaps[b].total_ticks_alive as f64 * 0.001;
                                                                fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
                                                            });
                                                        }
                                                    }
                                                    for &idx in &sorted_indices {
                                                        let snap = &agent_snaps[idx];
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
                                                                    snap.id, snap.generation, status
                                                                ));
                                                            });
                                                            ui.horizontal(|ui| {
                                                                ui.label(
                                                                    egui::RichText::new(snap.phase)
                                                                        .small()
                                                                        .color(match snap.phase {
                                                                            "ADAPTED" => egui::Color32::from_rgb(80, 200, 80),
                                                                            "LEARNING" => egui::Color32::from_rgb(200, 200, 80),
                                                                            "EXPLORING" => egui::Color32::from_rgb(200, 140, 60),
                                                                            _ => egui::Color32::from_rgb(150, 150, 150),
                                                                        }),
                                                                );
                                                                ui.label(
                                                                    egui::RichText::new(format!(
                                                                        "| D:{} F:{}",
                                                                        snap.deaths, snap.food_consumed
                                                                    ))
                                                                    .small()
                                                                    .color(egui::Color32::GRAY),
                                                                );
                                                            });
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
                                                    agents: agent_snaps,
                                                    evolution: &mut evo_snap,
                                                    evolution_action: &mut evo_action,
                                                    world: &world_snap,
                                                    replay: replay_state,
                                                    recording: last_recording,
                                                    orbit_mode,
                                                };
                                                egui_dock::DockArea::new(dock_state)
                                                    .style(egui_dock::Style::from_egui(ui.style().as_ref()))
                                                    .show_inside(ui, &mut tab_ctx);
                                            });
                                    },
                                );

                                self.viewport_hovered = vp_hovered;
                                self.chart_window = chart_win;
                                self.sort_mode = sort_mode;

                                // Restore evolution snapshot (may have been mutated by UI)
                                self.evo_snapshot = evo_snap;
                                // Restore world snapshot (biome texture may have been created)
                                self.world_snapshot = world_snap;

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
                                    let already_open =
                                        self.dock_state.iter_all_tabs().any(|(_, t)| *t == tab);
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
        // Create a stub BrainTelemetry since GPU brain doesn't expose per-tick telemetry yet
        let stub_telemetry = xagent_brain::BrainTelemetry::default();
        let _ = logger.log_tick(
            agent.id,
            &stub_telemetry,
            agent.brain_config.memory_capacity,
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
        let json = serde_json::to_string_pretty(&config).expect("Failed to serialize config");
        println!("{}", json);
        return;
    }

    if cli.bench_profile {
        let agent_count = cli.bench_agents;
        let total_ticks = cli.bench_ticks;
        if let Some(ws) = cli.world_size {
            config.world.world_size = ws;
        }
        xagent_sandbox::bench::run_profile(config.brain, config.world, agent_count, total_ticks);
        return;
    }

    if cli.bench {
        let agent_count = cli.bench_agents;
        let total_ticks = cli.bench_ticks;
        if let Some(ws) = cli.world_size {
            config.world.world_size = ws;
        }
        println!("Benchmark: {} agents, {} ticks", agent_count, total_ticks);
        let result =
            xagent_sandbox::bench::run_bench(config.brain, config.world, agent_count, total_ticks);
        println!(
            "Completed {} ticks in {:.2}s ({:.0} ticks/sec)",
            result.total_ticks, result.elapsed_secs, result.ticks_per_sec,
        );
        return;
    }

    if cli.dump_tree {
        headless::dump_tree(&cli.db);
        return;
    }

    print_config(&config);

    if cli.no_render {
        headless::run_headless(config, &cli.db, cli.resume, true);
    } else {
        let event_loop = EventLoop::new().expect("Failed to create event loop");
        let mut app = App::new(
            config.brain,
            config.world,
            config.governor,
            cli.log,
            &cli.db,
        );
        event_loop.run_app(&mut app).expect("Event loop error");
    }
}

// ── Pure helpers (extracted for testability) ─────────────────────────────

/// Map speed multiplier to a compact display label.
fn speed_label(multiplier: u32) -> &'static str {
    match multiplier {
        1 => "1×",
        2 => "2×",
        5 => "5×",
        10 => "10×",
        100 => "100×",
        1_000 => "1k×",
        10_000 => "10k×",
        100_000 => "100k×",
        1_000_000 => "1000k×",
        _ => "?×",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── speed_label ──────────────────────────────────────────────────

    #[test]
    fn speed_label_all_known_levels() {
        assert_eq!(speed_label(1), "1×");
        assert_eq!(speed_label(2), "2×");
        assert_eq!(speed_label(5), "5×");
        assert_eq!(speed_label(10), "10×");
        assert_eq!(speed_label(100), "100×");
        assert_eq!(speed_label(1_000), "1k×");
        assert_eq!(speed_label(10_000), "10k×");
        assert_eq!(speed_label(100_000), "100k×");
        assert_eq!(speed_label(1_000_000), "1000k×");
    }

    #[test]
    fn speed_label_unknown_returns_fallback() {
        assert_eq!(speed_label(42), "?×");
        assert_eq!(speed_label(0), "?×");
        assert_eq!(speed_label(999), "?×");
    }

    // ── key-to-multiplier mapping ────────────────────────────────────

    #[test]
    fn all_speed_levels_have_labels() {
        let levels: &[u32] = &[1, 2, 5, 10, 100, 1_000, 10_000, 100_000, 1_000_000];
        for &m in levels {
            assert_ne!(speed_label(m), "?×", "missing label for multiplier {}", m);
        }
    }
}
