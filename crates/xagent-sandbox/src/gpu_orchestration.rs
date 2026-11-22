//! Main-thread orchestration of the simulation worker.
//!
//! The GPU kernel and all simulation-cadence scheduling live in the
//! [`crate::sim_runtime`] worker thread. This module is the main-thread side of
//! that boundary: it starts the worker for the current population, forwards
//! control commands (speed, pause, selection) on change, drains worker events
//! each frame, applies the newest state snapshot to the CPU-side agent caches,
//! and records the per-snapshot overlays/histories/replay. It performs no GPU
//! work and never drives the compute kernel directly.

use std::time::{Duration, Instant};

use glam::Vec3;

use xagent_brain::buffers::{
    PHYS_STRIDE, P_ALIVE, P_APPROACH_SENSE_RANGE_TICKS, P_APPROACH_TURNS_TOWARD,
    P_AVOIDANCE_SENSE_RANGE_TICKS, P_AVOIDANCE_TURNS_OPPOSING, P_AVOIDANCE_TURN_AWAY,
    P_DANGER_PATH_LENGTH, P_DEATH_COUNT, P_DISTANCE_TRAVELED, P_ENERGY, P_ENERGY_SPENT,
    P_EXPLORATION_RATE_OUT, P_FACING_X, P_FACING_Y, P_FACING_Z, P_FATIGUE_FACTOR_OUT, P_FOOD_COUNT,
    P_GRADIENT_OUT, P_HAZARD_ENTRIES, P_INTEGRITY, P_LAST_DEATH_TICK, P_MAX_ENERGY,
    P_MAX_INTEGRITY, P_MOTOR_FWD_OUT, P_MOTOR_TURN_OUT, P_POS_X, P_POS_Y, P_POS_Z,
    P_PREDICTION_ERROR, P_TICKS_ALIVE, P_URGENCY_OUT, P_VEL_X, P_VEL_Y, P_VEL_Z, P_YAW,
    TRAIL_RECORD_STRIDE,
};
use xagent_brain::AgentTelemetry;
use xagent_sandbox::observatory::{ModelSource, ObservatoryModel, ObservatoryRecord};
use xagent_sandbox::tick_window::{
    frame_from_snapshot, Pin, PinKind, TickWindow, TickWindowConfig, TickWindowError,
};
use xagent_sandbox::tick_window_store::TickWindowStoreError;
use xagent_sandbox::ui::{EvolutionSnapshot, EvolutionState, ObservatoryViewState};

use crate::app::App;
use crate::sim_runtime::{
    partition_events, SimCommand, SimEvent, SimInit, SimRuntime, StateSnapshot,
    SupervisorWorkerInit,
};
use crate::tick_window_view::TickWindowView;
use xagent_shared::{BrainConfig, FullConfig, GovernorConfig, WorldConfig};

/// Worker init for a supervisor session, or `None` outside one.
fn supervisor_worker_init(
    governor: &GovernorConfig,
    brain: &BrainConfig,
    world: &WorldConfig,
    db_path: &str,
    session_active: bool,
    resume: Option<bool>,
) -> Option<SupervisorWorkerInit> {
    if !session_active {
        return None;
    }
    Some(SupervisorWorkerInit {
        config: FullConfig {
            brain: brain.clone(),
            world: world.clone(),
            governor: governor.clone(),
        },
        db_path: Some(db_path.to_string()),
        resume: resume.unwrap_or(false),
    })
}

/// Forwards a pending Replay request to the playback view.
pub(crate) fn forward_replay_request(
    view_state: &mut ObservatoryViewState,
    view: &mut TickWindowView,
) {
    if let Some(tick) = view_state.replay_jump_request.take() {
        view.request_jump(tick);
    }
}

/// Read-only window of a browsed supervisor run, loaded once per session.
#[derive(Default)]
pub(crate) struct BrowseTickWindow {
    pub window: Option<TickWindow>,
    pub loaded_run: Option<i64>,
    pub loads: u32,
}

impl BrowseTickWindow {
    /// Loads the browsed run's window once; clears it when browsing ends.
    pub(crate) fn sync(
        &mut self,
        db_path: &str,
        browse_run: Option<i64>,
        mut load: impl FnMut(&str, Option<i64>) -> Result<TickWindow, TickWindowStoreError>,
    ) {
        match browse_run {
            Some(run) if self.loaded_run != Some(run) => {
                self.loads += 1;
                self.window = match load(db_path, Some(run)) {
                    Ok(window) => Some(window),
                    Err(e) => {
                        log::warn!("browse tick window load failed: {e}");
                        TickWindow::new(TickWindowConfig::default()).ok()
                    }
                };
                self.loaded_run = Some(run);
            }
            Some(_) => {}
            None => {
                self.window = None;
                self.loaded_run = None;
            }
        }
    }
}

/// The browsed run id, only while a supervisor database is being browsed.
pub(crate) fn browse_run(evo: &EvolutionSnapshot) -> Option<i64> {
    if evo.state != EvolutionState::BrowsingSupervisor {
        return None;
    }
    match evo.observatory.as_ref().map(|m| &m.source) {
        Some(ModelSource::Browse { run_id }) => Some(*run_id),
        _ => None,
    }
}

/// The live window a worker start creates: supervisor mode only.
fn live_tick_window(supervisor_mode: bool) -> Option<TickWindow> {
    if !supervisor_mode {
        return None;
    }
    match TickWindow::new(TickWindowConfig::default()) {
        Ok(window) => Some(window),
        Err(e) => {
            log::warn!("tick window unavailable: {e}");
            None
        }
    }
}

/// Captures one snapshot into the window; no effect without a window.
pub(crate) fn capture_tick_window_snapshot(
    window: Option<&mut TickWindow>,
    tick: u64,
    physics: &[f32],
    food: &[f32],
) -> Result<(), TickWindowError> {
    let Some(window) = window else {
        return Ok(());
    };
    let frame = frame_from_snapshot(tick, physics, food)?;
    window.capture(frame);
    Ok(())
}

/// The model records a supervisor event carries, empty for other events.
fn observatory_record(model: &ObservatoryModel, event: &SimEvent) -> Vec<ObservatoryRecord> {
    match event {
        SimEvent::SupervisorBoundary(boundary) => vec![model.boundary_record(boundary)],
        SimEvent::SupervisorTelemetry(frame) => {
            let mut records: Vec<ObservatoryRecord> = frame
                .slots
                .iter()
                .map(|standing| model.slot_record(standing))
                .collect();
            records.extend(frame.intake.iter().map(ObservatoryRecord::from));
            records.push(ObservatoryRecord::from(&frame.progress));
            records.extend(ObservatoryRecord::series_from_frame(frame));
            records
        }
        _ => Vec::new(),
    }
}

impl App {
    /// Start the simulation worker for the current world and population.
    ///
    /// Builds the kernel-creation parameters, spawns the worker, and seeds the
    /// forwarded-control cache so the first `sync_worker_controls` only sends
    /// commands that actually changed. No-op when there is no world or
    /// population yet.
    pub(crate) fn start_sim_worker(&mut self) {
        if self.agents.is_empty() {
            return;
        }
        let (food_count, world_config) = match &self.world {
            Some(world) => (world.food_items.len(), world.config.clone()),
            None => return,
        };

        let agent_count = u32::try_from(self.agents.len()).unwrap_or(u32::MAX);
        let selected_agent = self
            .agents
            .get(self.selected_agent_idx)
            .map_or(0, |a| a.brain_idx);

        let session_active = self.evo_snapshot.supervisor.is_some();
        let supervisor = supervisor_worker_init(
            &self.governor_config,
            &self.brain_config,
            &world_config,
            &self.db_path,
            session_active,
            self.supervisor_resume.take(),
        );
        let supervisor_mode = supervisor.is_some();
        let init = SimInit {
            agent_count,
            food_count,
            brain_config: self.brain_config.clone(),
            world_config,
            speed_multiplier: self.speed_multiplier,
            paused: self.paused,
            selected_agent,
            supervisor,
        };

        self.tick_window = live_tick_window(supervisor_mode);
        self.sim_runtime = Some(SimRuntime::start(init));
        self.sent_speed = Some(self.speed_multiplier);
        self.sent_paused = Some(self.paused);
        self.sent_selected_agent = Some(selected_agent);
        self.cached_food_state = None;
        self.log_msg(format!("[GPU] sim worker starting ({agent_count} agents)"));
    }

    /// Drop the simulation worker, shutting down and joining its thread.
    pub(crate) fn stop_sim_worker(&mut self) {
        // Dropping the handle requests Shutdown and joins the worker thread.
        self.sim_runtime = None;
        self.tick_window = None;
        self.sent_speed = None;
        self.sent_paused = None;
        self.sent_selected_agent = None;
        self.cached_food_state = None;
    }

    /// Forward speed, pause, and selection to the worker, but only on change.
    pub(crate) fn sync_worker_controls(&mut self) {
        if self.sim_runtime.is_none() {
            return;
        }

        let speed = self.speed_multiplier;
        if self.sent_speed != Some(speed) {
            if let Some(runtime) = &self.sim_runtime {
                runtime.send(SimCommand::SetSpeed(speed));
            }
            self.sent_speed = Some(speed);
        }

        let desired_paused = self.paused;
        if self.sent_paused != Some(desired_paused) {
            if let Some(runtime) = &self.sim_runtime {
                runtime.send(SimCommand::SetPaused(desired_paused));
            }
            self.sent_paused = Some(desired_paused);
        }

        let selected_brain = self
            .agents
            .get(self.selected_agent_idx)
            .map(|a| a.brain_idx);
        if let Some(brain_idx) = selected_brain {
            if self.sent_selected_agent != Some(brain_idx) {
                if let Some(runtime) = &self.sim_runtime {
                    runtime.send(SimCommand::SelectAgent(brain_idx));
                }
                self.sent_selected_agent = Some(brain_idx);
            }
        }
    }

    /// Drain all pending worker events, applying the newest snapshot and
    /// processing control events (kernel-ready, telemetry, supervisor samples,
    /// logs) in order.
    pub(crate) fn drain_sim_events(&mut self) {
        let Some(runtime) = &self.sim_runtime else {
            return;
        };
        let events = runtime.drain_events();
        if events.is_empty() {
            return;
        }
        let (latest_snapshot, control) = partition_events(events);

        // Latest-wins: apply only the newest plain snapshot.
        if let Some(snapshot) = latest_snapshot {
            self.apply_state_snapshot(&snapshot);
            self.on_fresh_snapshot();
        }

        for event in control {
            if let Some(model) = self.evo_snapshot.observatory.as_mut() {
                for r in observatory_record(model, &event) {
                    model.fold(r);
                }
            }
            match event {
                SimEvent::KernelReady { agent_count } => {
                    self.log_msg(format!(
                        "[GPU] sim worker kernel ready ({agent_count} agents)"
                    ));
                }
                // Plain snapshots are folded into latest-wins above.
                SimEvent::Snapshot(_) => {}
                // The observatory consumes these.
                SimEvent::SupervisorTelemetry(_) => {}
                SimEvent::SupervisorBoundary(_) => {}
                SimEvent::SupervisorSample(report) => {
                    self.resume_pending = None;
                    if let Some(sup) = self.evo_snapshot.supervisor.as_mut() {
                        sup.apply(&report);
                    }
                    if let Some(slot) = report.boundary.as_ref().and_then(|b| b.slot) {
                        self.log_msg(format!("[SUPERVISOR] replaced slot {slot}"));
                    }
                    if let Some(window) = self.tick_window.as_mut() {
                        if let Some(slot) = report.boundary.as_ref().and_then(|b| b.slot) {
                            window.pin(Pin {
                                tick: report.tick,
                                kind: PinKind::Replacement { slot },
                            });
                        }
                        if report.season.is_some() {
                            window.pin(Pin {
                                tick: report.tick,
                                kind: PinKind::SeasonFlip,
                            });
                        }
                    }
                    self.snap_dirty = true;
                }
                SimEvent::Telemetry {
                    agent_index,
                    telemetry,
                } => self.apply_telemetry(agent_index, telemetry),
                SimEvent::Log(message) => self.log_msg(message),
                SimEvent::Error(message) => {
                    if let Some(summary) = self.resume_pending.take() {
                        self.stop_sim_worker();
                        crate::evolution::apply_resume_refusal(
                            &mut self.evo_snapshot,
                            &message,
                            summary,
                        );
                        self.paused = true;
                        return;
                    }
                    self.log_msg(format!("[GPU] {message}"));
                }
            }
        }
    }

    /// Apply a worker physics/food snapshot to the CPU-side agent caches.
    ///
    /// Authoritative for position/yaw/alive/energy/integrity/velocity and the
    /// `cached_*` motor/gradient/urgency/prediction/exploration/fatigue fields,
    /// plus the food state that backs the food mesh and mini-map.
    fn apply_state_snapshot(&mut self, snapshot: &StateSnapshot) {
        // The worker owns tick advancement, so derive the ticks/sec display from
        // the delta between published snapshots.
        let advanced_ticks = snapshot.tick.saturating_sub(self.tick);
        self.tps_tick_count += advanced_ticks;
        self.session_tick_count += advanced_ticks;
        self.tick = snapshot.tick;
        if self.tick_window.is_some() {
            if let Err(e) = capture_tick_window_snapshot(
                self.tick_window.as_mut(),
                snapshot.tick,
                &snapshot.physics,
                &snapshot.food,
            ) {
                log::warn!("tick window capture failed: {e}");
                self.tick_window = None;
            }
        }
        let state = &snapshot.physics;
        let death_reference_tick = snapshot.tick.saturating_sub(1);
        for i in 0..self.agents.len() {
            let base = i * PHYS_STRIDE;
            // Guard every physics field we read below. Use the stride's highest
            // offset so adding new fields doesn't silently leave the guard stale.
            if base + PHYS_STRIDE > state.len() {
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
            a.body.body.internal.max_integrity = state[base + P_MAX_INTEGRITY];
            a.body.body.velocity = Vec3::new(
                state[base + P_VEL_X],
                state[base + P_VEL_Y],
                state[base + P_VEL_Z],
            );
            a.food_consumed = state[base + P_FOOD_COUNT] as u32;
            a.total_ticks_alive = state[base + P_TICKS_ALIVE] as u64;
            a.distance_traveled = state[base + P_DISTANCE_TRAVELED];
            a.energy_spent = state[base + P_ENERGY_SPENT];
            a.danger_path_length = state[base + P_DANGER_PATH_LENGTH];
            a.avoidance_sense_range_ticks = state[base + P_AVOIDANCE_SENSE_RANGE_TICKS];
            a.avoidance_turns_opposing = state[base + P_AVOIDANCE_TURNS_OPPOSING];
            a.avoidance_turn_away = state[base + P_AVOIDANCE_TURN_AWAY];
            a.hazard_entries = state[base + P_HAZARD_ENTRIES];
            a.approach_sense_range_ticks = state[base + P_APPROACH_SENSE_RANGE_TICKS];
            a.approach_turns_toward = state[base + P_APPROACH_TURNS_TOWARD];
            let new_deaths = state[base + P_DEATH_COUNT] as u32;
            let gpu_death_tick = state[base + P_LAST_DEATH_TICK] as u64;
            a.apply_death_count_readback(new_deaths, gpu_death_tick, death_reference_tick);
            a.body.body.facing = Vec3::new(
                state[base + P_FACING_X],
                state[base + P_FACING_Y],
                state[base + P_FACING_Z],
            );
            a.cached_prediction_error = state[base + P_PREDICTION_ERROR];
            a.cached_exploration_rate = state[base + P_EXPLORATION_RATE_OUT];
            a.cached_fatigue_factor = state[base + P_FATIGUE_FACTOR_OUT];
            a.cached_motor.forward = state[base + P_MOTOR_FWD_OUT];
            a.cached_motor.turn = state[base + P_MOTOR_TURN_OUT];
            a.cached_gradient = state[base + P_GRADIENT_OUT];
            a.cached_urgency = state[base + P_URGENCY_OUT];
        }

        // GPU trail samples, oldest first. Each agent skips the ones it already
        // applied, so the repeated ring contents of a snapshot cost nothing.
        for sample in &snapshot.trail_samples {
            for (agent, record) in self
                .agents
                .iter_mut()
                .zip(sample.records.as_chunks::<TRAIL_RECORD_STRIDE>().0)
            {
                agent.apply_trail_record(sample.sample_number, *record);
            }
        }

        // Keep the previous food cache rather than overwriting with the empty
        // vector the worker sends before its first food readback.
        if !snapshot.food.is_empty() {
            match &mut self.cached_food_state {
                Some(food) => food.clone_from(&snapshot.food),
                none => *none = Some(snapshot.food.clone()),
            }
        }

        self.runtime_counters.snapshots_applied += 1;
        self.snap_dirty = true;
        self.hud_dirty = true;

        // Diagnostic: log first snapshot Y vs terrain height.
        if !self.readback_logged {
            if let Some(world) = &self.world {
                let n = self.agents.len().min(5);
                if n > 0 {
                    for i in 0..n {
                        let a = &self.agents[i];
                        let p = a.body.body.position;
                        let terrain_y = world.terrain.height_at(p.x, p.z);
                        log::info!(
                            "[TERRAIN-DIAG] Agent {} pos=({:.2}, {:.2}, {:.2}) terrain_y={:.2} diff={:.2}",
                            i, p.x, p.y, p.z, terrain_y, p.y - terrain_y,
                        );
                    }
                    self.readback_logged = true;
                }
            }
        }
    }

    /// Sample per-snapshot CPU-side state that follows the published cadence:
    /// overlays/histories and the food/heatmap dirty flags.
    fn on_fresh_snapshot(&mut self) {
        self.record_overlays_and_histories();
        self.heatmap_dirty = true;
        self.food_dirty = true;
    }

    /// Apply selected-agent telemetry to the matching agent's derived caches.
    ///
    /// Telemetry owns only the selected agent's vision color and the derived
    /// brain fields; the physics snapshot stays authoritative for everything
    /// else. Ignored when the telemetry is for an agent that is no longer the
    /// selected one (e.g. an in-flight readback after a selection change).
    fn apply_telemetry(&mut self, agent_index: u32, telemetry: AgentTelemetry) {
        let Some(a) = self.agents.get_mut(self.selected_agent_idx) else {
            return;
        };
        if a.brain_idx != agent_index {
            return;
        }
        a.cached_frame.vision.color = telemetry.vision_color;
        a.cached_mean_attenuation = telemetry.mean_attenuation;
        a.cached_curiosity_bonus = telemetry.curiosity_bonus;
        a.cached_staleness = telemetry.staleness;
    }

    /// Emit the render-side runtime-decoupling counters at most once per second.
    ///
    /// The simulation-side counters are logged by the worker thread; together
    /// the two log lines show simulation progress decoupled from render cadence.
    /// Call once per rendered frame; the internal timer rate-limits the output.
    pub(crate) fn log_runtime_counters(&mut self) {
        /// Minimum wall-clock interval between runtime-counter log lines.
        const COUNTER_LOG_INTERVAL: Duration = Duration::from_secs(1);
        if self.last_counters_log.elapsed() < COUNTER_LOG_INTERVAL {
            return;
        }
        self.last_counters_log = Instant::now();
        let counters = &self.runtime_counters;
        log::debug!(
            "[RENDER] frames={} snapshots_applied={}",
            counters.frames_rendered,
            counters.snapshots_applied,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{
        browse_run, capture_tick_window_snapshot, forward_replay_request, live_tick_window,
        observatory_record, supervisor_worker_init, BrowseTickWindow,
    };
    use crate::sim_runtime::SimEvent;
    use crate::tick_window_view::TickWindowView;
    use xagent_brain::buffers::PHYS_STRIDE;
    use xagent_sandbox::observatory::{ModelSource, ObservatoryModel};
    use xagent_sandbox::tick_window::{
        record_from_physics_row, Pin, PinKind, TickWindow, TickWindowConfig, WindowFrame,
    };
    use xagent_sandbox::tick_window_store;
    use xagent_sandbox::ui::{EvolutionSnapshot, EvolutionState, ObservatoryViewState};
    use xagent_shared::{BrainConfig, GovernorConfig, WorldConfig};

    #[test]
    fn gui_supervisor_worker_init_only_in_supervisor_mode() {
        let brain = BrainConfig::default();
        let world = WorldConfig::default();
        let governor = GovernorConfig::default();

        assert!(supervisor_worker_init(&governor, &brain, &world, "a.db", false, None).is_none());

        let init = supervisor_worker_init(&governor, &brain, &world, "a.db", true, None)
            .expect("supervisor session yields init");
        assert!(!init.resume);
        assert_eq!(init.db_path.as_deref(), Some("a.db"));
    }

    #[test]
    fn observatory_session_orchestration_never_loads_store() {
        let source = include_str!("gpu_orchestration.rs");
        assert!(!source.contains(concat!("store_read", "::")));
    }

    #[test]
    fn observatory_session_non_supervisor_events_yield_no_records() {
        let model = ObservatoryModel::default();
        for event in [
            SimEvent::Log("line".to_string()),
            SimEvent::Error("failure".to_string()),
        ] {
            assert!(observatory_record(&model, &event).is_empty());
        }
    }

    #[test]
    fn gui_supervisor_main_thread_never_touches_kernel() {
        let sources = [
            ("gpu_orchestration.rs", include_str!("gpu_orchestration.rs")),
            ("evolution.rs", include_str!("evolution.rs")),
            ("snapshot.rs", include_str!("snapshot.rs")),
            ("app.rs", include_str!("app.rs")),
        ];
        let needles = [
            concat!("Gpu", "Kernel"),
            concat!("SupervisorRun", "::"),
            concat!("read_full_state", "_blocking"),
            concat!("on_", "sample("),
            concat!("TickWindow", "Store::open"),
        ];
        for (name, source) in sources {
            for needle in needles {
                assert!(!source.contains(needle), "{name} names {needle}");
            }
        }
    }

    fn empty_window() -> TickWindow {
        TickWindow::new(TickWindowConfig::default()).expect("default config")
    }

    fn test_frame(tick: u64) -> WindowFrame {
        WindowFrame {
            tick,
            agents: vec![record_from_physics_row(&[0.0; PHYS_STRIDE])],
            food: vec![],
        }
    }

    #[test]
    fn gui_jump_request_opens_view_at_requested_tick() {
        let mut view_state = ObservatoryViewState::default();
        view_state.replay_jump_request = Some(20);
        let mut window = empty_window();
        for tick in [10, 20, 30] {
            window.capture(test_frame(tick));
        }
        let mut view = TickWindowView::default();
        forward_replay_request(&mut view_state, &mut view);
        view.advance(&window, 1);
        assert_eq!(view.current_frame(&window).map(|f| f.tick), Some(20));
        assert_eq!(view_state.replay_jump_request, None);
    }

    #[test]
    fn gui_supervisor_snapshot_is_captured_and_pins_replacement_and_season() {
        let mut window = empty_window();
        let physics = vec![0.0f32; PHYS_STRIDE];
        capture_tick_window_snapshot(Some(&mut window), 5, &physics, &[]).expect("captured");
        window.pin(Pin {
            tick: 5,
            kind: PinKind::Replacement { slot: 0 },
        });
        window.pin(Pin {
            tick: 5,
            kind: PinKind::SeasonFlip,
        });
        assert_eq!(window.len(), 1);
        assert_eq!(window.newest_tick(), Some(5));
        assert_eq!(window.pins().len(), 2);
    }

    #[test]
    fn gui_generation_mode_captures_nothing() {
        let physics = vec![0.0f32; PHYS_STRIDE];
        let mut supervisor = live_tick_window(true).expect("supervisor mode creates a window");
        assert!(capture_tick_window_snapshot(Some(&mut supervisor), 5, &physics, &[]).is_ok());
        assert_eq!(supervisor.len(), 1);
        let mut generation = live_tick_window(false);
        assert!(generation.is_none());
        assert_eq!(
            capture_tick_window_snapshot(generation.as_mut(), 6, &physics, &[]),
            Ok(())
        );
        assert_eq!(supervisor.len(), 1);
        assert_eq!(supervisor.newest_tick(), Some(5));
    }

    #[test]
    fn gui_browse_sync_loads_once_and_clears_on_leave() {
        let mut browse = BrowseTickWindow::default();
        let mut calls = 0u32;
        let mut load = |_: &str, _: Option<i64>| {
            calls += 1;
            Ok(empty_window())
        };
        for _ in 0..3 {
            browse.sync("a.db", Some(7), &mut load);
        }
        assert_eq!(browse.loads, 1);
        assert!(browse.window.is_some());
        assert_eq!(browse.loaded_run, Some(7));
        browse.sync("a.db", None, &mut load);
        assert!(browse.window.is_none());
        assert_eq!(browse.loaded_run, None);
        browse.sync("a.db", Some(7), &mut load);
        assert_eq!(browse.loads, 2);
        assert_eq!(calls, 2);
    }

    #[test]
    fn gui_browse_run_only_in_browsing_supervisor() {
        let mut evo = EvolutionSnapshot::default();
        let mut model = ObservatoryModel::default();
        model.source = ModelSource::Browse { run_id: 7 };
        evo.observatory = Some(model);
        evo.state = EvolutionState::BrowsingSupervisor;
        assert_eq!(browse_run(&evo), Some(7));
        evo.state = EvolutionState::Idle;
        assert_eq!(browse_run(&evo), None);
        evo.state = EvolutionState::BrowsingSupervisor;
        if let Some(m) = evo.observatory.as_mut() {
            m.source = ModelSource::Live;
        }
        assert_eq!(browse_run(&evo), None);
    }

    #[test]
    fn gui_browse_load_leaves_db_bytes_unchanged() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("t.db");
        let path_str = path.to_str().expect("utf8 path").to_string();
        let run_id: i64;
        {
            let db = rusqlite::Connection::open(&path).expect("open");
            db.execute_batch("CREATE TABLE run (id INTEGER PRIMARY KEY, mode TEXT);")
                .expect("run table");
            tick_window_store::ensure_schema(&db).expect("schema");
            db.execute("INSERT INTO run (mode) VALUES ('supervisor')", [])
                .expect("insert");
            run_id = db.last_insert_rowid();
        }
        let before = std::fs::read(&path).expect("read");
        let mut browse = BrowseTickWindow::default();
        let mut load = |p: &str, r: Option<i64>| {
            tick_window_store::load_read_only(p, r, TickWindowConfig::default())
        };
        browse.sync(&path_str, Some(run_id), &mut load);
        let after = std::fs::read(&path).expect("read");
        assert_eq!(before, after);
        assert_eq!(browse.loads, 1);
        browse.sync(&path_str, Some(run_id), &mut load);
        assert_eq!(browse.loads, 1);
    }
}
