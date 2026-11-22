//! Application state container for the windowed sandbox binary.
//!
//! Centralises the `App` struct and the startup database classification.
//! Behavioural methods live in sibling binary modules (`gpu_orchestration`,
//! `evolution`, `render_pipeline`, `snapshot`) to keep each responsibility
//! focused.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::window::Window;

use xagent_sandbox::agent::Agent;
use xagent_sandbox::observatory::store_read::{detect_supervisor_run, SupervisorRunSummary};
use xagent_sandbox::renderer::camera::Camera;
use xagent_sandbox::renderer::hud::HudBar;
use xagent_sandbox::renderer::{GpuMesh, Renderer};
use xagent_sandbox::tick_window::{TickWindow, TickWindowConfig};
use xagent_sandbox::tick_window_store;
use xagent_sandbox::ui::{
    AgentSnapshot, EguiIntegration, EvolutionSnapshot, EvolutionState, ReplayState, SortMode, Tab,
    WorldSnapshot,
};
use xagent_sandbox::world::WorldState;
use xagent_shared::{BrainConfig, GovernorConfig, WorldConfig};

use crate::evolution::{
    apply_database_refusal, apply_legacy_choice, inspect_offer, ResetState, SessionOffer,
};
use crate::gpu_orchestration::BrowseTickWindow;
use crate::tick_window_view::{self, TickWindowView};

/// Base simulation rate in Hz. The simulation runs at this rate (scaled by
/// speed multiplier) regardless of rendering frame rate.
pub(crate) const SIM_RATE: f32 = 60.0;
pub(crate) const SIM_DT: f32 = 1.0 / SIM_RATE;

/// Minimum interval between expensive mesh rebuilds and snapshot copies.
/// At 100 ms (10 Hz) the visual difference is indistinguishable from per-frame.
pub(crate) const REBUILD_THROTTLE: Duration = Duration::from_millis(100);

/// Render-side instrumentation for the runtime-decoupling work.
///
/// The simulation half of the baseline (ticks advanced, dispatches, snapshot /
/// telemetry requests and collections) lives in the worker thread (see
/// `sim_runtime::Worker`). The main thread only counts what it observes:
/// frames rendered and snapshots consumed. Together the
/// two halves show simulation progress is independent of render cadence. These
/// are diagnostic only — emitted through `log::debug!` roughly once per second
/// (see `App::log_runtime_counters`) and carry no UI surface.
#[derive(Default)]
pub(crate) struct RuntimeCounters {
    /// Frames rendered (redraw-handler invocations).
    pub(crate) frames_rendered: u64,
    /// State snapshots received from the worker and applied to CPU caches.
    pub(crate) snapshots_applied: u64,
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

    // Facing-direction lines for every alive agent
    pub(crate) bearing_gpu: Option<GpuMesh>,

    // egui integration (IDE-style UI overlay)
    pub(crate) egui: Option<EguiIntegration>,

    // Console log buffer for the bottom panel
    pub(crate) console_log: VecDeque<String>,

    pub(crate) evo_snapshot: EvolutionSnapshot,
    pub(crate) evo_wall_accumulated: f64,
    pub(crate) evo_wall_segment_start: Option<Instant>,
    pub(crate) tps_tick_count: u64,
    /// Simulation ticks the current run has advanced; drives the top-bar
    /// simulated-time clock.
    pub(crate) session_tick_count: u64,
    pub(crate) tps_last_reset: Instant,
    pub(crate) tps_display: f64,
    pub(crate) db_path: String,
    /// Rolling window of the live supervisor run, fed from applied snapshots.
    pub(crate) tick_window: Option<TickWindow>,
    pub(crate) tick_window_view: TickWindowView,
    /// Read-only window of a browsed database's supervisor run.
    pub(crate) browse_tick_window: BrowseTickWindow,
    pub(crate) governor_config: GovernorConfig,

    /// The legacy-database choice has been shown for the current path.
    pub(crate) legacy_offered: Option<String>,
    /// Progress of the two-step database reset.
    pub(crate) reset_state: ResetState,

    // Simulation worker — owns the GPU kernel and all simulation-cadence state.
    // `None` until a session starts; recreated on Start/Resume, dropped on Reset.
    pub(crate) sim_runtime: Option<crate::sim_runtime::SimRuntime>,
    /// Resume input for the next supervisor worker start; taken by the start.
    pub(crate) supervisor_resume: Option<bool>,
    /// Run a live resume started from, held until the first supervisor sample
    /// so an early worker error can restore the session dialog.
    pub(crate) resume_pending: Option<SupervisorRunSummary>,
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

    // Replay playback state
    pub(crate) replay_state: ReplayState,
}

/// What the database at startup holds.
#[allow(clippy::large_enum_variant)] // Built once at startup.
pub(crate) enum StartupSession {
    /// A supervisor run; the user chooses to resume it live or browse it.
    Supervisor(SupervisorRunSummary),
    /// A generations database: only quit or start in a new file.
    Legacy {
        body: String,
        generation: Option<u64>,
    },
    /// A database that could not be inspected.
    Refused(String),
    /// No usable session.
    None,
}

/// Classifies the database at `db_path` without creating or writing it.
pub(crate) fn startup_session(db_path: &str) -> StartupSession {
    let (offer, body, generation) = inspect_offer(db_path);
    match (offer, body) {
        (SessionOffer::LegacyChoice, Some(body)) => StartupSession::Legacy { body, generation },
        (SessionOffer::Refuse, Some(body)) => StartupSession::Refused(body),
        // Detection opens the file, which would create a missing one.
        _ if !std::path::Path::new(db_path).exists() => StartupSession::None,
        _ => match detect_supervisor_run(db_path) {
            Ok(Some(summary)) => StartupSession::Supervisor(summary),
            Err(e) => {
                log::warn!("[SUPERVISOR] Could not inspect {db_path}: {e}");
                StartupSession::None
            }
            Ok(None) => StartupSession::None,
        },
    }
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
        // Set by the first Start on a legacy file, once the choice is shown.
        let legacy_offered: Option<String> = None;

        match startup_session(db_path) {
            StartupSession::Supervisor(summary) => {
                log::info!(
                    "[SUPERVISOR] Found supervisor run {} at tick {} in {}",
                    summary.run_id,
                    summary.last_tick,
                    db_path
                );
                evo_snapshot.state = EvolutionState::SupervisorSession;
                evo_snapshot.supervisor_session = Some(summary);
            }
            StartupSession::Legacy { body, generation } => {
                log::warn!("[SESSION] {body}");
                log::debug!("[SESSION] Legacy database last generation: {generation:?}");
                apply_legacy_choice(&mut evo_snapshot, body);
            }
            StartupSession::Refused(body) => {
                log::warn!("[SESSION] Could not inspect {db_path}: {body}");
                apply_database_refusal(&mut evo_snapshot, body);
            }
            StartupSession::None => {
                evo_snapshot.state = EvolutionState::Idle;
                log::info!("[SESSION] No previous session — ready to configure");
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
            bearing_gpu: None,
            egui: None,
            console_log: VecDeque::new(),
            evo_snapshot,
            evo_wall_accumulated: 0.0,
            evo_wall_segment_start: None,
            tps_tick_count: 0,
            session_tick_count: 0,
            tps_last_reset: Instant::now(),
            tps_display: 0.0,
            db_path: db_path.to_string(),
            governor_config,
            legacy_offered,
            reset_state: ResetState::Idle,
            sim_runtime: None,
            supervisor_resume: None,
            resume_pending: None,
            cached_food_state: None,
            sent_speed: None,
            sent_paused: None,
            sent_selected_agent: None,
            dock_state: egui_dock::DockState::new(vec![Tab::Evolution, Tab::Sandbox]),
            sort_mode: SortMode::Id,
            world_snapshot: WorldSnapshot::default(),
            replay_state: ReplayState::default(),
            tick_window: None,
            tick_window_view: TickWindowView::default(),
            browse_tick_window: BrowseTickWindow::default(),
        }
    }
}

/// Per-frame playback step, run after the UI pass: forwards a Replay request,
/// syncs the browse window, then renders the browse window when present, else
/// the live window.
pub(crate) fn drive_tick_window(
    ui: &mut egui::Ui,
    evo_snapshot: &mut EvolutionSnapshot,
    db_path: &str,
    tick_window_view: &mut TickWindowView,
    browse_tick_window: &mut BrowseTickWindow,
    live: Option<&TickWindow>,
) {
    crate::gpu_orchestration::forward_replay_request(
        &mut evo_snapshot.observatory_view,
        tick_window_view,
    );
    browse_tick_window.sync(
        db_path,
        crate::gpu_orchestration::browse_run(evo_snapshot),
        |p, r| tick_window_store::load_read_only(p, r, TickWindowConfig::default()),
    );
    if let Some(window) = browse_tick_window.window.as_ref().or(live) {
        tick_window_view::render(ui, tick_window_view, window);
    }
}

impl App {
    pub(crate) fn log_msg(&mut self, message: String) {
        println!("{}", message);
        self.console_log.push_back(message);
        if self.console_log.len() > 200 {
            self.console_log.pop_front();
        }
    }

    /// Advance replay playback by the configured speed while a recording plays,
    /// clamping at the final tick and stopping playback once it is reached.
    pub(crate) fn advance_replay_playback(&mut self) {
        if !(self.replay_state.active && self.replay_state.playing) {
            return;
        }
        let advance = (self.replay_state.speed as u64).max(1);
        self.replay_state.current_tick = (self.replay_state.current_tick + advance)
            .min(self.replay_state.total_ticks.saturating_sub(1));
        if self.replay_state.current_tick >= self.replay_state.total_ticks.saturating_sub(1) {
            self.replay_state.playing = false;
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

#[cfg(test)]
mod tests {
    use super::{startup_session, StartupSession};
    use xagent_sandbox::supervisor_store::{StoreError, SupervisorStore};
    use xagent_shared::FullConfig;

    /// Generation stored in the legacy fixture.
    const FIXTURE_GENERATION: i64 = 5;

    fn path_in(dir: &tempfile::TempDir, name: &str) -> String {
        dir.path().join(name).to_str().unwrap().to_string()
    }

    #[test]
    fn observatory_session_startup_supervisor_db_read_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir, "run.db");
        let store = SupervisorStore::create(&path, &FullConfig::default()).unwrap();
        drop(store);
        {
            let db = rusqlite::Connection::open(&path).unwrap();
            let mode: String = db
                .query_row("PRAGMA journal_mode=DELETE", [], |row| row.get(0))
                .unwrap();
            assert_eq!(mode, "delete");
        }
        let before = std::fs::read(&path).unwrap();
        assert!(matches!(
            startup_session(&path),
            StartupSession::Supervisor(_)
        ));
        let after = std::fs::read(&path).unwrap();
        assert!(before == after, "startup changed the database file");
    }

    #[test]
    fn observatory_session_startup_missing_db_not_created() {
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir, "missing.db");
        assert!(matches!(startup_session(&path), StartupSession::None));
        assert!(!std::path::Path::new(&path).exists());
    }

    #[test]
    fn startup_legacy_db_is_refused_with_store_message_and_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir, "gen.db");
        {
            let db = rusqlite::Connection::open(&path).unwrap();
            db.execute_batch(&format!(
                "CREATE TABLE node (id INTEGER PRIMARY KEY, generation INTEGER);
                 INSERT INTO node (generation) VALUES ({FIXTURE_GENERATION});"
            ))
            .unwrap();
        }
        let before = std::fs::read(&path).unwrap();
        match startup_session(&path) {
            StartupSession::Legacy { body, generation } => {
                let expected = StoreError::LegacyGenerations {
                    path: path.clone(),
                    generation,
                }
                .to_string();
                assert_eq!(body, expected);
                assert_eq!(generation, u64::try_from(FIXTURE_GENERATION).ok());
            }
            _ => panic!("expected a legacy session"),
        }
        assert_eq!(before, std::fs::read(&path).unwrap());
    }
}
