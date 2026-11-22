//! Evolution action handling, population spawning and the database gate.
//!
//! This module hosts `App::handle_evolution_action`: starting, resuming,
//! pausing and resetting the supervisor session. The database is classified
//! read-only before any open; a legacy generations file is refused untouched.

use std::path::{Path, PathBuf};
use std::time::Instant;

use glam::Vec3;
use rand::Rng;

use xagent_sandbox::agent::{mutate_config, Agent, MAX_AGENTS};
use xagent_sandbox::observatory::store_read::{load_run, SupervisorRunSummary};
use xagent_sandbox::observatory::{ModelSource, ObservatoryModel};
use xagent_sandbox::supervisor_store::{inspect_database, DatabaseKind, StoreError};
use xagent_sandbox::ui::{
    EvolutionAction, EvolutionSnapshot, EvolutionState, SessionChoice, SupervisorSnapshot,
};
use xagent_shared::BrainConfig;

use crate::app::App;

/// Longest resume error shown in the session dialog, in bytes.
const MAX_RESUME_ERROR_BYTES: usize = 256;

/// What the session may do with the database at its path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SessionOffer {
    /// The file may be opened as a supervisor store.
    Open,
    /// The file holds a generations run: only quit or start in a new file.
    LegacyChoice,
    /// The file could not be inspected.
    Refuse,
}

/// The offer for a database of `kind`.
pub(crate) fn session_offer(kind: DatabaseKind) -> SessionOffer {
    match kind {
        DatabaseKind::Missing | DatabaseKind::Empty | DatabaseKind::Supervisor => {
            SessionOffer::Open
        }
        DatabaseKind::LegacyGenerations { .. } => SessionOffer::LegacyChoice,
    }
}

/// Classify the database at `path` read-only: the offer and, when it is not
/// `Open`, the message to show. The legacy message is the store's own text.
pub(crate) fn inspect_offer(path: &str) -> (SessionOffer, Option<String>, Option<u64>) {
    match inspect_database(Path::new(path)) {
        Ok(kind) => {
            let (body, generation) = match &kind {
                DatabaseKind::LegacyGenerations { generation } => (
                    Some(
                        StoreError::LegacyGenerations {
                            path: path.to_owned(),
                            generation: *generation,
                        }
                        .to_string(),
                    ),
                    *generation,
                ),
                _ => (None, None),
            };
            (session_offer(kind), body, generation)
        }
        Err(error) => (SessionOffer::Refuse, Some(error.to_string()), None),
    }
}

/// The first of `<stem>.supervisor.db`, `<stem>.supervisor-2.db`, ... in the
/// directory of `path` for which `exists` is false.
pub(crate) fn next_fresh_db_path(path: &Path, exists: impl Fn(&Path) -> bool) -> PathBuf {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "xagent".to_string());
    let mut number = 1_u32;
    loop {
        let name = if number == 1 {
            format!("{stem}.supervisor.db")
        } else {
            format!("{stem}.supervisor-{number}.db")
        };
        let candidate = path.with_file_name(name);
        if !exists(&candidate) {
            return candidate;
        }
        number += 1;
    }
}

/// Two-step confirmation of a database reset.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum ResetState {
    /// No reset requested.
    #[default]
    Idle,
    /// A first request named this database; a second one deletes it.
    Armed(PathBuf),
}

/// Whether a reset request for `path` is the confirming second one.
pub(crate) fn reset_confirmed(state: &ResetState, path: &Path) -> bool {
    matches!(state, ResetState::Armed(armed) if armed == path)
}

/// Show the legacy choice: the store's message, offering only "Quit" or
/// "Start in a new file".
pub(crate) fn apply_legacy_choice(snapshot: &mut EvolutionSnapshot, body: String) {
    snapshot.supervisor = None;
    snapshot.observatory = None;
    snapshot.supervisor_session = None;
    snapshot.supervisor_resume_error = Some(body);
    snapshot.legacy_choice = true;
    snapshot.state = EvolutionState::Idle;
}

/// Show why the database could not be inspected; nothing is offered.
pub(crate) fn apply_database_refusal(snapshot: &mut EvolutionSnapshot, body: String) {
    snapshot.supervisor = None;
    snapshot.observatory = None;
    snapshot.supervisor_session = None;
    snapshot.supervisor_resume_error = Some(body);
    snapshot.legacy_choice = false;
    snapshot.state = EvolutionState::Idle;
}

/// `text` cut to at most [`MAX_RESUME_ERROR_BYTES`] on a char boundary.
fn truncate_error(text: &str) -> String {
    let mut end = text.len().min(MAX_RESUME_ERROR_BYTES);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

/// Enter the read-only browse view of a loaded run; no worker starts.
pub(crate) fn apply_browse_supervisor(snapshot: &mut EvolutionSnapshot, model: ObservatoryModel) {
    snapshot.observatory = Some(model);
    snapshot.state = EvolutionState::BrowsingSupervisor;
    snapshot.supervisor = None;
    snapshot.supervisor_session = None;
    snapshot.supervisor_resume_error = None;
}

/// Return to the session dialog after a refused live resume.
pub(crate) fn apply_resume_refusal(
    snapshot: &mut EvolutionSnapshot,
    message: &str,
    summary: SupervisorRunSummary,
) {
    log::warn!("[SUPERVISOR] Resume refused: {message}");
    snapshot.supervisor = None;
    snapshot.observatory = None;
    snapshot.supervisor_resume_error = Some(truncate_error(message));
    snapshot.supervisor_session = Some(summary);
    snapshot.state = EvolutionState::SupervisorSession;
}

impl App {
    /// Spawn a new agent with the given BrainConfig at a safe random position
    /// (not in a danger biome). The brain_idx is the agent's array index.
    pub(crate) fn spawn_agent(&mut self, config: BrainConfig, generation: u32) {
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

    /// Classify the database before any open and show a legacy or refused
    /// file. Returns the offer; only `Open` may proceed.
    fn gate_database(&mut self) -> SessionOffer {
        let (offer, body, _) = inspect_offer(&self.db_path);
        match (offer, body) {
            (SessionOffer::LegacyChoice, Some(body)) => {
                apply_legacy_choice(&mut self.evo_snapshot, body);
                self.legacy_offered = Some(self.db_path.clone());
                self.log_msg(
                    "[SESSION] Legacy generations database: quit or start in a new file".into(),
                );
            }
            (SessionOffer::Refuse, Some(body)) => {
                self.log_msg(format!("[SESSION] Could not inspect database: {body}"));
                apply_database_refusal(&mut self.evo_snapshot, body);
            }
            _ => {}
        }
        if offer != SessionOffer::LegacyChoice {
            self.legacy_offered = None;
            self.evo_snapshot.legacy_choice = false;
        }
        offer
    }

    /// Act on the answer to the supervisor-run dialog, if one was given.
    fn handle_session_choice(&mut self) {
        let choice = std::mem::take(&mut self.evo_snapshot.session_choice);
        if choice == SessionChoice::None {
            return;
        }
        if self.gate_database() != SessionOffer::Open {
            return;
        }
        let Some(summary) = self.evo_snapshot.supervisor_session.clone() else {
            return;
        };
        let loaded = load_run(&self.db_path, summary.run_id);
        let mut model = match loaded {
            Ok(model) => model,
            Err(e) => {
                self.evo_snapshot.supervisor_resume_error = Some(truncate_error(&e.to_string()));
                self.log_msg(format!("[SUPERVISOR] Could not load run: {e}"));
                return;
            }
        };
        if choice == SessionChoice::Browse {
            apply_browse_supervisor(&mut self.evo_snapshot, model);
            self.log_msg("[SUPERVISOR] Browsing stored run".into());
            return;
        }
        model.source = ModelSource::Live;
        self.governor_config = self.evo_snapshot.edit_governor.clone();
        self.brain_config = self.evo_snapshot.edit_brain.clone();
        self.evo_snapshot.observatory = Some(model);
        self.evo_snapshot.supervisor = Some(SupervisorSnapshot {
            total_ticks: summary.budget_ticks,
            ..Default::default()
        });
        self.evo_snapshot.supervisor_session = None;
        self.evo_snapshot.supervisor_resume_error = None;
        self.supervisor_resume = Some(true);
        self.resume_pending = Some(summary.clone());
        self.evo_snapshot.state = EvolutionState::Running;
        self.evo_wall_accumulated = 0.0;
        self.session_tick_count = 0;
        self.evo_wall_segment_start = Some(Instant::now());
        self.tps_tick_count = 0;
        self.tps_last_reset = Instant::now();
        self.tps_display = 0.0;
        self.paused = false;
        self.stop_sim_worker();
        self.spawn_evolution_population();
        self.start_sim_worker();
        if self.sim_runtime.is_none() {
            self.resume_pending = None;
            self.supervisor_resume = None;
            apply_resume_refusal(
                &mut self.evo_snapshot,
                "the simulation worker did not start",
                summary,
            );
            self.paused = true;
            return;
        }
        self.log_msg("[SUPERVISOR] Resumed supervisor run".into());
    }

    pub(crate) fn handle_evolution_action(&mut self, action: EvolutionAction) {
        self.handle_session_choice();
        if action != EvolutionAction::Reset && action != EvolutionAction::None {
            self.reset_state = ResetState::Idle;
        }
        match action {
            EvolutionAction::None => {}
            EvolutionAction::Start => {
                // A plain Start on a legacy file only shows the choice.
                if self.gate_database() != SessionOffer::Open {
                    return;
                }
                let brain_config = self.evo_snapshot.edit_brain.clone();
                let gov_config = self.evo_snapshot.edit_governor.clone();
                let population = gov_config.population_size as u64;
                let budget = gov_config.supervisor.agent_tick_budget;
                let total_ticks = if budget > 0 && population > 0 {
                    Some(budget / population)
                } else {
                    None
                };
                self.governor_config = gov_config;
                self.brain_config = brain_config;
                self.evo_snapshot.state = EvolutionState::Running;
                self.evo_snapshot.supervisor = Some(SupervisorSnapshot {
                    total_ticks,
                    ..Default::default()
                });
                self.evo_snapshot.observatory = Some(ObservatoryModel {
                    source: ModelSource::Live,
                    ..Default::default()
                });
                self.evo_wall_accumulated = 0.0;
                self.session_tick_count = 0;
                self.evo_wall_segment_start = Some(Instant::now());
                self.tps_tick_count = 0;
                self.tps_last_reset = Instant::now();
                self.tps_display = 0.0;
                self.paused = false;
                self.stop_sim_worker();
                self.spawn_evolution_population();
                self.start_sim_worker();
                self.log_msg("[EVOLUTION] Started supervisor run".into());
            }
            EvolutionAction::Resume => {
                // While the legacy choice is showing for this path, Resume is
                // the explicit "Start in a new file" answer.
                if self.legacy_offered.as_deref() == Some(self.db_path.as_str())
                    && self.gate_database() == SessionOffer::LegacyChoice
                {
                    let fresh = next_fresh_db_path(Path::new(&self.db_path), |p| p.exists());
                    self.db_path = fresh.to_string_lossy().into_owned();
                    self.legacy_offered = None;
                    self.evo_snapshot.supervisor_resume_error = None;
                    self.log_msg(format!("[SESSION] Starting in new file {}", self.db_path));
                    self.handle_evolution_action(EvolutionAction::Start);
                    return;
                }
                if self.gate_database() == SessionOffer::Open {
                    self.log_msg(
                        "[EVOLUTION] Resume applies to supervisor runs through the session dialog"
                            .into(),
                    );
                }
            }
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
                let path = PathBuf::from(&self.db_path);
                if !matches!(inspect_database(&path), Ok(DatabaseKind::Supervisor)) {
                    self.log_msg("[EVOLUTION] Reset applies only to a supervisor database".into());
                    return;
                }
                if !reset_confirmed(&self.reset_state, &path) {
                    self.reset_state = ResetState::Armed(path);
                    self.log_msg("[EVOLUTION] Reset again to delete the database file".into());
                    return;
                }
                self.reset_state = ResetState::Idle;
                self.stop_sim_worker();
                self.agents.clear();
                self.next_agent_id = 0;
                self.tick = 0;
                self.paused = true;
                if let Err(e) = std::fs::remove_file(&path) {
                    self.log_msg(format!("[EVOLUTION] Failed to reset DB: {}", e));
                }
                self.evo_snapshot = EvolutionSnapshot::default();
                self.evo_snapshot.observatory = None;
                self.resume_pending = None;
                self.supervisor_resume = None;
                self.evo_snapshot.edit_brain = self.brain_config.clone();
                self.evo_snapshot.edit_governor = self.governor_config.clone();
                self.log_msg("[EVOLUTION] Reset — ready to start fresh".into());
            }
        }
    }

    pub(crate) fn spawn_evolution_population(&mut self) {
        self.agents.clear();
        self.next_agent_id = 0;
        self.tick = 0;
        let seed = self.brain_config.clone();
        let pop_size = self.governor_config.population_size;
        // Every agent carries the spawn parent's config.
        for _ in 0..pop_size {
            self.spawn_agent(seed.clone(), 0);
        }
    }

    /// Spawn a child agent near a parent, with mutated config.
    #[allow(dead_code)]
    pub(crate) fn spawn_child(&mut self, parent_idx: usize) {
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
        // `mem_cost` / `proc_cost`: metabolic-cost proxies — kernel widths
        // are fixed at MEMORY_CAP=128, RECALL_K=16 (see issue #106).
        println!(
            "  Child config: mem_cost={} proc_cost={} dim={} lr={:.4} decay={:.4}",
            child.brain_config.memory_capacity,
            child.brain_config.processing_slots,
            child.brain_config.representation_dimension,
            child.brain_config.learning_rate,
            child.brain_config.decay_rate,
        );

        self.agents.push(child);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_browse_supervisor, apply_resume_refusal, inspect_offer, next_fresh_db_path,
        reset_confirmed, session_offer, ResetState, SessionOffer, MAX_RESUME_ERROR_BYTES,
    };
    use std::path::{Path, PathBuf};
    use xagent_sandbox::observatory::store_read::SupervisorRunSummary;
    use xagent_sandbox::observatory::{ModelSource, ObservatoryModel};
    use xagent_sandbox::supervisor_store::{DatabaseKind, StoreError};
    use xagent_sandbox::ui::{EvolutionSnapshot, EvolutionState, SupervisorSnapshot};

    /// Run id of the sample summary.
    const SAMPLE_RUN: i64 = 3;
    /// Message length that exceeds the error cap.
    const LONG_MESSAGE_BYTES: usize = 1000;

    fn summary() -> SupervisorRunSummary {
        SupervisorRunSummary {
            run_id: SAMPLE_RUN,
            last_tick: 0,
            budget_ticks: None,
        }
    }

    /// Generation stored in the legacy fixtures.
    const LEGACY_GENERATION: u64 = 4;

    #[test]
    fn session_offer_legacy_is_quit_or_fresh_only() {
        assert_eq!(session_offer(DatabaseKind::Missing), SessionOffer::Open);
        assert_eq!(session_offer(DatabaseKind::Empty), SessionOffer::Open);
        assert_eq!(session_offer(DatabaseKind::Supervisor), SessionOffer::Open);
        assert_eq!(
            session_offer(DatabaseKind::LegacyGenerations {
                generation: Some(LEGACY_GENERATION)
            }),
            SessionOffer::LegacyChoice
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("legacy.db");
        {
            let db = rusqlite::Connection::open(&path).expect("open");
            db.execute_batch(
                "CREATE TABLE node (id INTEGER PRIMARY KEY, generation INTEGER);
                 INSERT INTO node (generation) VALUES (4);",
            )
            .expect("legacy schema");
        }
        let before = std::fs::read(&path).expect("read");
        let path_text = path.to_string_lossy().into_owned();
        let (offer, body, generation) = inspect_offer(&path_text);
        assert_eq!(offer, SessionOffer::LegacyChoice);
        assert_eq!(generation, Some(LEGACY_GENERATION));
        let expected = StoreError::LegacyGenerations {
            path: path_text,
            generation: Some(LEGACY_GENERATION),
        }
        .to_string();
        assert_eq!(body.as_deref(), Some(expected.as_str()));
        assert_eq!(std::fs::read(&path).expect("read"), before);
    }

    #[test]
    fn session_offer_inspection_error_refuses() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("garbage.db");
        std::fs::write(&path, b"this is not a sqlite database at all, just text").expect("write");
        let (offer, body, _) = inspect_offer(&path.to_string_lossy());
        assert_eq!(offer, SessionOffer::Refuse);
        assert!(body.is_some());
    }

    #[test]
    fn next_fresh_db_path_skips_existing_siblings() {
        let base = Path::new("/data/run.db");
        let taken = |names: &'static [&'static str]| {
            move |p: &Path| {
                names
                    .iter()
                    .any(|n| p == Path::new("/data").join(n).as_path())
            }
        };
        assert_eq!(
            next_fresh_db_path(base, taken(&[])),
            PathBuf::from("/data/run.supervisor.db")
        );
        assert_eq!(
            next_fresh_db_path(base, taken(&["run.supervisor.db"])),
            PathBuf::from("/data/run.supervisor-2.db")
        );
        assert_eq!(
            next_fresh_db_path(base, taken(&["run.supervisor.db", "run.supervisor-2.db"])),
            PathBuf::from("/data/run.supervisor-3.db")
        );
    }

    #[test]
    fn reset_needs_two_confirmations_on_same_path() {
        let first = Path::new("/data/a.db");
        let other = Path::new("/data/b.db");
        let idle = ResetState::Idle;
        assert!(!reset_confirmed(&idle, first));
        let armed = ResetState::Armed(first.to_path_buf());
        assert!(!reset_confirmed(&armed, other));
        assert!(reset_confirmed(&armed, first));
    }

    #[test]
    fn observatory_session_browse_sets_state_without_worker() {
        let mut snapshot = EvolutionSnapshot::default();
        snapshot.supervisor = Some(SupervisorSnapshot::default());
        snapshot.supervisor_session = Some(summary());
        let model = ObservatoryModel {
            source: ModelSource::Browse { run_id: SAMPLE_RUN },
            ..Default::default()
        };
        apply_browse_supervisor(&mut snapshot, model);
        assert_eq!(snapshot.state, EvolutionState::BrowsingSupervisor);
        assert!(snapshot.supervisor.is_none());
        assert!(snapshot.supervisor_session.is_none());
        let model = snapshot.observatory.expect("model set");
        assert_eq!(model.source, ModelSource::Browse { run_id: SAMPLE_RUN });
    }

    #[test]
    fn observatory_session_resume_refusal_keeps_dialog() {
        let mut snapshot = EvolutionSnapshot::default();
        snapshot.supervisor = Some(SupervisorSnapshot::default());
        snapshot.observatory = Some(ObservatoryModel::default());
        let message = "e".repeat(LONG_MESSAGE_BYTES);
        apply_resume_refusal(&mut snapshot, &message, summary());
        assert_eq!(snapshot.state, EvolutionState::SupervisorSession);
        assert!(snapshot.supervisor.is_none());
        assert!(snapshot.observatory.is_none());
        let error = snapshot.supervisor_resume_error.expect("error set");
        assert!(error.len() <= MAX_RESUME_ERROR_BYTES);
        assert_eq!(snapshot.supervisor_session, Some(summary()));
    }
}
