//! Application state container for the windowed sandbox binary.
//!
//! Centralises the `App` struct, the non-blocking generation-transition state
//! machine, and the pending-upload staging type.  Behavioural methods live
//! in sibling binary modules (`gpu_orchestration`, `evolution`,
//! `render_pipeline`, `snapshot`) to keep each responsibility focused.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::window::Window;

use xagent_sandbox::agent::Agent;
use xagent_sandbox::governor::{check_existing_session, AdvanceResult, Governor};
use xagent_sandbox::renderer::camera::Camera;
use xagent_sandbox::renderer::hud::HudBar;
use xagent_sandbox::renderer::{GpuMesh, Renderer};
use xagent_sandbox::ui::{
    AgentSnapshot, EguiIntegration, EvolutionSnapshot, EvolutionState, ReplayState, SortMode, Tab,
    WorldSnapshot,
};
use xagent_sandbox::world::WorldState;
use xagent_shared::{BrainConfig, GovernorConfig, WorldConfig};

/// Base simulation rate in Hz. The simulation runs at this rate (scaled by
/// speed multiplier) regardless of rendering frame rate.
pub(crate) const SIM_RATE: f32 = 60.0;
pub(crate) const SIM_DT: f32 = 1.0 / SIM_RATE;

/// Minimum interval between expensive mesh rebuilds and snapshot copies.
/// At 100 ms (10 Hz) the visual difference is indistinguishable from per-frame.
pub(crate) const REBUILD_THROTTLE: Duration = Duration::from_millis(100);

/// Data collected on the main thread for upload once the background-created
/// fused kernel is ready.  Avoids passing world/agent refs across threads.
pub(crate) struct PendingUpload {
    pub(crate) heights: Vec<f32>,
    pub(crate) biomes: Vec<u32>,
    pub(crate) food_pos: Vec<(f32, f32, f32)>,
    pub(crate) food_consumed: Vec<bool>,
    pub(crate) food_timers: Vec<f32>,
    pub(crate) agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)>,
}

/// Render-side instrumentation for the runtime-decoupling work.
///
/// The simulation half of the baseline (ticks advanced, dispatches, snapshot /
/// telemetry requests and collections) lives in the worker thread (see
/// `sim_runtime::Worker`). The main thread only counts what it observes:
/// frames rendered and snapshots/generation-boundaries consumed. Together the
/// two halves show simulation progress is independent of render cadence. These
/// are diagnostic only — emitted through `log::debug!` roughly once per second
/// (see `App::log_runtime_counters`) and carry no UI surface.
#[derive(Default)]
pub(crate) struct RuntimeCounters {
    /// Frames rendered (redraw-handler invocations).
    pub(crate) frames_rendered: u64,
    /// State snapshots received from the worker and applied to CPU caches.
    pub(crate) snapshots_applied: u64,
    /// Generation-boundary events received from the worker.
    pub(crate) generation_boundaries: u64,
}

/// In-flight generation handoff: the worker reached the tick budget and the
/// main thread has evaluated fitness and requested the champion's brain state.
///
/// When the champion `AgentState` reply arrives, the main thread spawns the
/// next population and sends `ResetPopulation` to the worker. `Finished`
/// results complete immediately and never produce a pending handoff.
pub(crate) struct PendingGeneration {
    pub(crate) result: AdvanceResult,
    /// Identifies the matching `RequestAgentState` / `AgentState` pair.
    pub(crate) champion_request_id: u64,
}

pub(crate) struct App {
    pub(crate) brain_config: BrainConfig,
    pub(crate) world_config: WorldConfig,

    pub(crate) renderer: Option<Renderer>,
    pub(crate) window: Option<Arc<Window>>,
    pub(crate) camera: Camera,

    // Static terrain mesh (built once)
    pub(crate) terrain_gpu: Option<GpuMesh>,

    // Simulation
    pub(crate) world: Option<WorldState>,
    pub(crate) agents: Vec<Agent>,
    pub(crate) next_agent_id: u32,
    pub(crate) tick: u64,
    pub(crate) food_dirty: bool,
    pub(crate) food_gpu: Option<GpuMesh>,

    pub(crate) last_frame: Instant,

    // Speed controls (multiplier for simulation rate). The worker owns the
    // tick accumulator; the main thread only owns the user's chosen speed and
    // forwards it on change.
    pub(crate) speed_multiplier: u32,
    pub(crate) paused: bool,
    pub(crate) render_3d: bool,

    // Telemetry selection
    pub(crate) selected_agent_idx: usize,
    pub(crate) viewport_hovered: bool,
    pub(crate) chart_window: usize,
    pub(crate) orbit_mode: bool,

    // GPU instancing for agents
    pub(crate) agent_instance_buffer: Option<wgpu::Buffer>,
    pub(crate) agent_instance_count: u32,

    // FPS tracking
    pub(crate) frame_times: VecDeque<Instant>,
    pub(crate) fps: f32,

    // Cached HUD data — only rebuilt when a simulation tick runs
    pub(crate) cached_hud_bars: Vec<HudBar>,
    pub(crate) hud_dirty: bool,

    // Cursor position for click-to-select
    pub(crate) cursor_pos: (f64, f64),

    // Heatmap overlay
    pub(crate) heatmap_enabled: bool,
    pub(crate) heatmap_gpu: Option<GpuMesh>,
    pub(crate) heatmap_dirty: bool,

    // Trail overlay for selected agent
    pub(crate) trail_gpu: Option<GpuMesh>,

    // Throttle timers — limit expensive rebuilds to ~10 Hz
    pub(crate) last_mesh_rebuild: Instant,
    pub(crate) last_snapshot_rebuild: Instant,
    pub(crate) snap_dirty: bool,
    pub(crate) cached_agent_snaps: Vec<AgentSnapshot>,
    pub(crate) last_snap_chart_window: usize,

    // Selection marker above focused agent
    pub(crate) marker_gpu: Option<GpuMesh>,

    // egui integration (IDE-style UI overlay)
    pub(crate) egui: Option<EguiIntegration>,

    // Console log buffer for the bottom panel
    pub(crate) console_log: VecDeque<String>,

    // Evolution governor (created on Start/Resume, not on App init)
    pub(crate) governor: Option<Governor>,
    pub(crate) evo_snapshot: EvolutionSnapshot,
    pub(crate) evo_wall_accumulated: f64,
    pub(crate) evo_wall_segment_start: Option<Instant>,
    pub(crate) tps_tick_count: u64,
    pub(crate) tps_last_reset: Instant,
    pub(crate) tps_display: f64,
    pub(crate) db_path: String,
    pub(crate) governor_config: GovernorConfig,

    // In-flight generation handoff (worker reached the tick budget; awaiting
    // champion brain-state reply before sending the population reset).
    pub(crate) pending_generation: Option<PendingGeneration>,
    /// Monotonic id source pairing `RequestAgentState` with its `AgentState`.
    pub(crate) champion_request_counter: u64,

    // Simulation worker — owns the GPU kernel and all simulation-cadence state.
    // `None` until evolution starts; recreated on Start/Resume, dropped on Reset.
    pub(crate) sim_runtime: Option<crate::sim_runtime::SimRuntime>,
    /// Latest food state published by the worker, authoritative for the food
    /// mesh and mini-map once a snapshot has arrived.
    pub(crate) cached_food_state: Option<Vec<f32>>,
    /// Last control values forwarded to the worker, so commands are sent only
    /// on change (`None` until the first sync after the worker starts).
    pub(crate) sent_speed: Option<u32>,
    pub(crate) sent_paused: Option<bool>,
    pub(crate) sent_selected_agent: Option<u32>,

    // egui_dock tab state
    pub(crate) dock_state: egui_dock::DockState<Tab>,

    // Diagnostic: log first snapshot Y vs terrain height
    pub(crate) readback_logged: bool,

    // Runtime-decoupling instrumentation (diagnostic-only, see RuntimeCounters)
    pub(crate) runtime_counters: RuntimeCounters,
    pub(crate) last_counters_log: Instant,

    // Sidebar sort mode
    pub(crate) sort_mode: SortMode,

    // World snapshot for mini-map UI
    pub(crate) world_snapshot: WorldSnapshot,

    // Replay recording
    pub(crate) recording: Option<xagent_sandbox::replay::GenerationRecording>,
    pub(crate) last_recording: Option<xagent_sandbox::replay::GenerationRecording>,

    // Replay playback state
    pub(crate) replay_state: ReplayState,
}

impl App {
    pub(crate) fn new(
        brain_config: BrainConfig,
        world_config: WorldConfig,
        governor_config: GovernorConfig,
        db_path: &str,
    ) -> Self {
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
            speed_multiplier: 1,
            paused: true,
            render_3d: true,
            selected_agent_idx: 0,
            viewport_hovered: false,
            chart_window: 120,
            orbit_mode: false,
            readback_logged: false,
            runtime_counters: RuntimeCounters::default(),
            last_counters_log: Instant::now(),
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
            pending_generation: None,
            champion_request_counter: 0,
            sim_runtime: None,
            cached_food_state: None,
            sent_speed: None,
            sent_paused: None,
            sent_selected_agent: None,
            dock_state: egui_dock::DockState::new(vec![Tab::Evolution, Tab::Sandbox]),
            sort_mode: SortMode::Id,
            world_snapshot: WorldSnapshot::default(),
            recording: None,
            last_recording: None,
            replay_state: ReplayState::default(),
        }
    }

    pub(crate) fn log_msg(&mut self, message: String) {
        println!("{}", message);
        self.console_log.push_back(message);
        if self.console_log.len() > 200 {
            self.console_log.pop_front();
        }
    }

    pub(crate) fn print_session_summary(&self) {
        let total_deaths: u32 = self.agents.iter().map(|a| a.death_count).sum();

        println!();
        println!("=== xagent Session Summary ===");
        println!("Total ticks: {}", self.tick);
        println!("Total agents spawned: {}", self.next_agent_id);
        println!("Living agents: {}", self.agents.len());
        println!("Total deaths: {}", total_deaths);
        println!();
    }
}
