//! SQLite persistence of a supervisor run.
//!
//! The store adds three tables to the governor schema: the occupant of each
//! slot, the bounded archive of retired lives and an event log. It also adds
//! six telemetry tables: slot snapshots, boundaries, boundary candidates,
//! series points, occupant intake and run progress. Every change a
//! boundary, season, idle sample or run end makes is written in one
//! transaction. Archive entries are loaded back through a read-only
//! connection that never writes to the database.

use std::fmt;
use std::path::Path;
use std::time::Duration;

use rusqlite::{params, Connection, OpenFlags, OptionalExtension, Result as SqlResult};
use xagent_brain::buffers::AgentBrainState;
use xagent_shared::{BrainConfig, FoodKind, FullConfig};

use crate::governor::{decode_brain_state, encode_brain_state, init_schema};
use crate::supervisor::{
    Archive, ArchiveChange, ArchiveEntry, BoundaryDetail, EligibilityGround, ParentRef,
    ReplaceReason, ReplacementOutcome, ReplacementPlan, SlotLedger,
};
use crate::supervisor_checkpoint::{FORMAT_VERSION, MAX_CHECKPOINT_BYTES};
use crate::supervisor_telemetry::{BoundaryRecord, TelemetryFrame};
use serde_json::json;

/// Busy timeout of the writer and of a read-only archive load.
pub const STORE_BUSY_TIMEOUT: Duration = Duration::from_millis(5_000);

/// Pragmas the writer connection opens with; they match the governor's.
pub const STORE_PRAGMAS: &str =
    "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;";

/// Event kind of a completed slot replacement.
pub const EVENT_REPLACE: &str = "replace";
/// Event kind of a boundary that replaced nothing.
pub const EVENT_IDLE: &str = "idle";
/// Event kind of a food-kind season change.
pub const EVENT_SEASON: &str = "season";
/// Event kind of an archive admission.
pub const EVENT_ARCHIVE: &str = "archive";
/// Event kind of an archive eviction.
pub const EVENT_EVICT: &str = "evict";

/// Value of `run.mode` for a supervisor run.
pub const MODE_SUPERVISOR: &str = "supervisor";

/// Parent kind recorded for an archive-entry parent.
const PARENT_KIND_ARCHIVE: &str = "archive_entry";
/// Parent kind recorded for a peer's birth brain.
const PARENT_KIND_PEER_BIRTH: &str = "peer_birth";
/// Parent kind recorded for a peer's learned state.
const PARENT_KIND_PEER_LEARNED: &str = "peer_learned";
/// Reason recorded for a death replacement.
const REASON_DEATH: &str = "death";
/// Reason recorded for an underperformer replacement.
const REASON_UNDERPERFORMER: &str = "underperformer";

/// Why a store write failed.
#[derive(Debug)]
pub enum StoreError {
    /// SQLite reported an error.
    Sqlite(rusqlite::Error),
    /// A brain state could not be encoded.
    Encode,
    /// The database already holds a run.
    NotFresh,
    /// Two inputs that must describe the same archive entry disagree.
    Mismatch,
    /// A value does not fit the database integer range.
    Range,
    /// A configuration could not be serialised.
    Json(serde_json::Error),
    /// A stored checkpoint payload is larger than the allowed maximum.
    CheckpointTooLarge {
        /// Stored payload length in bytes.
        len: i64,
        /// Largest payload length accepted.
        max: usize,
    },
    /// The stored checkpoint of a run is not at the requested tick.
    CheckpointTickMismatch {
        /// Run the resume targets.
        run_id: i64,
        /// Tick of the stored checkpoint, or `None` when there is none.
        stored: Option<i64>,
        /// Tick the resume asked for.
        requested: u64,
    },
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sqlite(err) => write!(f, "supervisor store query failed: {err}"),
            Self::Encode => write!(f, "brain state could not be encoded"),
            Self::NotFresh => write!(f, "database already holds a run"),
            Self::Mismatch => write!(f, "archive entries of one write disagree"),
            Self::Range => write!(f, "value exceeds the database integer range"),
            Self::Json(err) => write!(f, "configuration could not be serialised: {err}"),
            Self::CheckpointTooLarge { len, max } => {
                write!(
                    f,
                    "stored checkpoint of {len} bytes exceeds the {max} byte limit"
                )
            }
            Self::CheckpointTickMismatch {
                run_id,
                stored,
                requested,
            } => match stored {
                Some(stored) => write!(
                    f,
                    "run {run_id} has a checkpoint at tick {stored}, not tick {requested}"
                ),
                None => write!(
                    f,
                    "run {run_id} has no checkpoint to resume at tick {requested}"
                ),
            },
        }
    }
}

impl std::error::Error for StoreError {}

impl From<rusqlite::Error> for StoreError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Sqlite(err)
    }
}

impl From<serde_json::Error> for StoreError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

/// Why an archive entry could not be loaded.
#[derive(Debug)]
pub enum ArchiveLoadError {
    /// The database holds no supervisor run.
    NoSupervisorRun,
    /// The archive holds fewer than `rank + 1` entries.
    RankOutOfRange {
        /// Requested rank.
        rank: usize,
        /// Entries the archive holds.
        len: usize,
    },
    /// The stored brain blob of this entry cannot be decoded.
    MalformedBlob {
        /// Entry whose blob is malformed.
        entry_id: u64,
    },
    /// The stored configuration of this entry is not a valid brain config.
    MalformedConfig {
        /// Entry whose configuration is malformed.
        entry_id: u64,
    },
    /// SQLite reported an error.
    Sqlite(rusqlite::Error),
}

impl fmt::Display for ArchiveLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSupervisorRun => write!(f, "database holds no supervisor run"),
            Self::RankOutOfRange { rank, len } => {
                write!(f, "rank {rank} is beyond the {len} archived entries")
            }
            Self::MalformedBlob { entry_id } => {
                write!(f, "stored brain of archive entry {entry_id} is malformed")
            }
            Self::MalformedConfig { entry_id } => {
                write!(f, "stored config of archive entry {entry_id} is malformed")
            }
            Self::Sqlite(err) => write!(f, "archive query failed: {err}"),
        }
    }
}

impl std::error::Error for ArchiveLoadError {}

impl From<rusqlite::Error> for ArchiveLoadError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Sqlite(err)
    }
}

/// Creates the governor schema and, idempotently, the supervisor tables.
///
/// # Errors
///
/// Returns the SQLite error when a statement fails.
pub fn ensure_schema(db: &Connection) -> SqlResult<()> {
    init_schema(db)?;
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS supervisor_slot (
            run_id INTEGER NOT NULL REFERENCES run(id),
            slot INTEGER NOT NULL,
            occupant_id INTEGER NOT NULL,
            born_tick INTEGER NOT NULL,
            config_json TEXT NOT NULL,
            birth_brain BLOB NOT NULL,
            PRIMARY KEY (run_id, slot)
        );

        CREATE TABLE IF NOT EXISTS supervisor_archive (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES run(id),
            entry_id INTEGER NOT NULL,
            occupant_id INTEGER NOT NULL,
            born_tick INTEGER NOT NULL,
            retired_tick INTEGER NOT NULL,
            outcome REAL NOT NULL,
            config_json TEXT NOT NULL,
            birth_brain BLOB NOT NULL,
            UNIQUE (run_id, entry_id)
        );

        CREATE TABLE IF NOT EXISTS supervisor_event (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES run(id),
            tick INTEGER NOT NULL,
            kind TEXT NOT NULL,
            slot INTEGER,
            reason TEXT,
            occupant_id INTEGER,
            parent_kind TEXT,
            parent_ref INTEGER,
            outcome REAL,
            nourishing_kind TEXT
        );

        CREATE TABLE IF NOT EXISTS supervisor_slot_snapshot (
            run_id INTEGER NOT NULL REFERENCES run(id),
            tick INTEGER NOT NULL,
            slot INTEGER NOT NULL,
            occupant_id INTEGER NOT NULL,
            age_ticks INTEGER NOT NULL,
            alive INTEGER NOT NULL,
            meals INTEGER NOT NULL,
            deaths INTEGER NOT NULL,
            window_score REAL,
            peer_median REAL,
            ground TEXT NOT NULL,
            parent_kind TEXT,
            parent_ref INTEGER,
            PRIMARY KEY (run_id, tick, slot)
        );

        CREATE TABLE IF NOT EXISTS supervisor_boundary (
            run_id INTEGER NOT NULL REFERENCES run(id),
            tick INTEGER NOT NULL,
            outcome TEXT NOT NULL,
            slot INTEGER,
            reason TEXT,
            tiebreak_pool TEXT,
            tiebreak_index INTEGER,
            parent_kind TEXT,
            parent_ref INTEGER,
            parent_state TEXT,
            parent_pool_len INTEGER,
            parent_index INTEGER,
            retired_occupant INTEGER,
            retired_lifespan_ticks INTEGER,
            newborn_occupant INTEGER,
            config_diff TEXT,
            idle_too_young INTEGER,
            idle_no_full_window INTEGER,
            idle_no_peer_median INTEGER,
            idle_at_or_above_median INTEGER,
            PRIMARY KEY (run_id, tick)
        );

        CREATE TABLE IF NOT EXISTS supervisor_boundary_candidate (
            run_id INTEGER NOT NULL REFERENCES run(id),
            tick INTEGER NOT NULL,
            slot INTEGER NOT NULL,
            occupant_id INTEGER NOT NULL,
            ground TEXT NOT NULL,
            score REAL,
            median REAL,
            age_ticks INTEGER NOT NULL,
            PRIMARY KEY (run_id, tick, slot)
        );

        CREATE TABLE IF NOT EXISTS supervisor_series (
            run_id INTEGER NOT NULL REFERENCES run(id),
            tick INTEGER NOT NULL,
            metric TEXT NOT NULL,
            value REAL,
            PRIMARY KEY (run_id, tick, metric)
        );

        CREATE TABLE IF NOT EXISTS supervisor_occupant_intake (
            run_id INTEGER NOT NULL REFERENCES run(id),
            occupant_id INTEGER NOT NULL,
            age_ticks INTEGER NOT NULL,
            slot INTEGER NOT NULL,
            meals INTEGER NOT NULL,
            PRIMARY KEY (run_id, occupant_id, age_ticks)
        );

        CREATE TABLE IF NOT EXISTS supervisor_progress (
            run_id INTEGER NOT NULL REFERENCES run(id),
            tick INTEGER NOT NULL,
            total_ticks INTEGER,
            agent_tick_budget INTEGER NOT NULL,
            ticks_to_boundary INTEGER NOT NULL,
            nourishing_kind TEXT,
            next_flip_tick INTEGER,
            ticks_per_second REAL,
            PRIMARY KEY (run_id, tick)
        );

        CREATE TABLE IF NOT EXISTS supervisor_checkpoint (run_id INTEGER PRIMARY KEY REFERENCES run(id), tick INTEGER NOT NULL, format_version INTEGER NOT NULL, simulation_ticks INTEGER, payload BLOB NOT NULL);",
    )
}

/// A real number as a database value; non-finite numbers become NULL.
fn finite_real(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

/// The stored text of an eligibility ground.
fn ground_text(ground: &EligibilityGround) -> &'static str {
    match ground {
        EligibilityGround::Death => "death",
        EligibilityGround::BelowMedian { .. } => "below_median",
        EligibilityGround::TooYoung => "too_young",
        EligibilityGround::NoFullWindow => "no_full_window",
        EligibilityGround::NoPeerMedian => "no_peer_median",
        EligibilityGround::AtOrAboveMedian { .. } => "at_or_above_median",
    }
}

/// Kind, reference and brain state of a parent.
fn parent_columns(parent: ParentRef) -> Result<(&'static str, i64, &'static str), StoreError> {
    Ok(match parent {
        ParentRef::ArchiveEntry(id) => (PARENT_KIND_ARCHIVE, db_int(id)?, "birth"),
        ParentRef::PeerBirth(peer) => (PARENT_KIND_PEER_BIRTH, db_index(peer)?, "birth"),
        ParentRef::PeerLearned(peer) => (PARENT_KIND_PEER_LEARNED, db_index(peer)?, "learned"),
    })
}

/// Writes the boundary row and candidate rows of one frame.
fn insert_boundary(
    tx: &Connection,
    run_id: i64,
    tick: u64,
    record: &BoundaryRecord,
) -> Result<(), StoreError> {
    let tick = db_int(tick)?;
    match &record.detail {
        BoundaryDetail::Replaced {
            candidates,
            slot,
            reason,
            tiebreak,
            parent,
            parent_draw,
            ..
        } => {
            let reason = match reason {
                ReplaceReason::Death => REASON_DEATH,
                ReplaceReason::Underperformer => REASON_UNDERPERFORMER,
            };
            let pool = serde_json::to_string(&tiebreak.pool)?;
            let (kind, reference, state) = parent_columns(*parent)?;
            let (pool_len, draw_index) = match parent_draw {
                Some(draw) => (
                    Some(db_index(draw.pool_len)?),
                    Some(db_index(draw.drawn_index)?),
                ),
                None => (None, None),
            };
            let diff: Vec<serde_json::Value> = record
                .config_diff
                .iter()
                .map(|d| json!({"field": d.field, "parent": d.parent, "newborn": d.newborn}))
                .collect();
            let diff = serde_json::to_string(&diff)?;
            tx.execute(
                "INSERT INTO supervisor_boundary
                    (run_id, tick, outcome, slot, reason, tiebreak_pool, tiebreak_index,
                     parent_kind, parent_ref, parent_state, parent_pool_len, parent_index,
                     retired_occupant, retired_lifespan_ticks, newborn_occupant, config_diff)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
                params![
                    run_id,
                    tick,
                    EVENT_REPLACE,
                    db_index(*slot)?,
                    reason,
                    pool,
                    db_index(tiebreak.drawn_index)?,
                    kind,
                    reference,
                    state,
                    pool_len,
                    draw_index,
                    replaced_retired(&record.detail)?,
                    record.retired_lifespan_ticks.map(db_int).transpose()?,
                    record.newborn_occupant.map(db_int).transpose()?,
                    diff
                ],
            )?;
            for standing in candidates {
                let (score, median) = match standing.ground {
                    EligibilityGround::Death => (None, None),
                    EligibilityGround::BelowMedian { score, median } => (
                        finite_real(f64::from(score)),
                        finite_real(f64::from(median)),
                    ),
                    _ => continue,
                };
                tx.execute(
                    "INSERT INTO supervisor_boundary_candidate
                        (run_id, tick, slot, occupant_id, ground, score, median, age_ticks)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        run_id,
                        tick,
                        db_index(standing.slot)?,
                        db_int(standing.occupant_id)?,
                        ground_text(&standing.ground),
                        score,
                        median,
                        db_int(standing.age_ticks)?
                    ],
                )?;
            }
        }
        BoundaryDetail::Idle { reason, .. } => {
            tx.execute(
                "INSERT INTO supervisor_boundary
                    (run_id, tick, outcome, idle_too_young, idle_no_full_window,
                     idle_no_peer_median, idle_at_or_above_median)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    run_id,
                    tick,
                    EVENT_IDLE,
                    db_index(reason.too_young)?,
                    db_index(reason.no_full_window)?,
                    db_index(reason.no_peer_median)?,
                    db_index(reason.at_or_above_median)?
                ],
            )?;
        }
    }
    Ok(())
}

/// The retired occupant of a replaced boundary, NULL for an idle one.
fn replaced_retired(detail: &BoundaryDetail) -> Result<Option<i64>, StoreError> {
    match detail {
        BoundaryDetail::Replaced {
            retired_occupant, ..
        } => Ok(Some(db_int(*retired_occupant)?)),
        BoundaryDetail::Idle { .. } => Ok(None),
    }
}

/// A `u64` as a database integer.
fn db_int(value: u64) -> Result<i64, StoreError> {
    i64::try_from(value).map_err(|_| StoreError::Range)
}

/// A `usize` as a database integer.
fn db_index(value: usize) -> Result<i64, StoreError> {
    i64::try_from(value).map_err(|_| StoreError::Range)
}

/// The encoded blob of a brain state.
fn brain_blob(state: &AgentBrainState) -> Result<Vec<u8>, StoreError> {
    encode_brain_state(state).ok_or(StoreError::Encode)
}

/// Upserts the occupant of `slot`.
fn upsert_slot(
    tx: &Connection,
    run_id: i64,
    slot: usize,
    occupant_id: u64,
    born_tick: u64,
    config: &BrainConfig,
    brain: &AgentBrainState,
) -> Result<(), StoreError> {
    tx.execute(
        "INSERT OR REPLACE INTO supervisor_slot
            (run_id, slot, occupant_id, born_tick, config_json, birth_brain)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            run_id,
            db_index(slot)?,
            db_int(occupant_id)?,
            db_int(born_tick)?,
            serde_json::to_string(config)?,
            brain_blob(brain)?
        ],
    )?;
    Ok(())
}

/// Inserts the archive row of `entry`.
fn insert_archive_row(
    tx: &Connection,
    run_id: i64,
    entry: &ArchiveEntry,
) -> Result<(), StoreError> {
    tx.execute(
        "INSERT INTO supervisor_archive
            (run_id, entry_id, occupant_id, born_tick, retired_tick, outcome,
             config_json, birth_brain)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            run_id,
            db_int(entry.entry_id)?,
            db_int(entry.occupant_id)?,
            db_int(entry.born_tick)?,
            db_int(entry.retired_tick)?,
            f64::from(entry.outcome),
            serde_json::to_string(&entry.config)?,
            brain_blob(&entry.birth_brain)?
        ],
    )?;
    Ok(())
}

/// Deletes the archive row of an evicted entry and logs the eviction.
fn evict_entry(tx: &Connection, run_id: i64, tick: u64, entry_id: u64) -> Result<(), StoreError> {
    let id = db_int(entry_id)?;
    tx.execute(
        "DELETE FROM supervisor_archive WHERE run_id = ?1 AND entry_id = ?2",
        params![run_id, id],
    )?;
    tx.execute(
        "INSERT INTO supervisor_event (run_id, tick, kind, parent_kind, parent_ref)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![run_id, db_int(tick)?, EVENT_EVICT, PARENT_KIND_ARCHIVE, id],
    )?;
    Ok(())
}

/// Logs the admission of `entry` into the archive.
fn log_archive_event(
    tx: &Connection,
    run_id: i64,
    tick: u64,
    slot: Option<usize>,
    entry: &ArchiveEntry,
) -> Result<(), StoreError> {
    let slot = slot.map(db_index).transpose()?;
    tx.execute(
        "INSERT INTO supervisor_event
            (run_id, tick, kind, slot, occupant_id, parent_kind, parent_ref, outcome)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            run_id,
            db_int(tick)?,
            EVENT_ARCHIVE,
            slot,
            db_int(entry.occupant_id)?,
            PARENT_KIND_ARCHIVE,
            db_int(entry.entry_id)?,
            f64::from(entry.outcome)
        ],
    )?;
    Ok(())
}

/// Writer of one supervisor run.
pub struct SupervisorStore {
    db: Connection,
    run_id: i64,
}

impl SupervisorStore {
    /// Opens `db_path`, ensures the schema and records a supervisor run.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::NotFresh`] when the database already holds a run,
    /// and other [`StoreError`]s when a statement or serialisation fails.
    pub fn create(db_path: &str, config: &FullConfig) -> Result<Self, StoreError> {
        let mut db = Connection::open(db_path)?;
        db.execute_batch(STORE_PRAGMAS)?;
        ensure_schema(&db)?;
        let runs: i64 = db.query_row("SELECT COUNT(*) FROM run", [], |row| row.get(0))?;
        if runs > 0 {
            return Err(StoreError::NotFresh);
        }
        let governor_json = serde_json::to_string(&config.governor)?;
        let brain_json = serde_json::to_string(&config.brain)?;
        let world_json = serde_json::to_string(&config.world)?;
        let tx = db.transaction()?;
        tx.execute(
            "INSERT INTO run (seed, governor_config, brain_config, world_config, mode)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                config.world.seed.cast_signed(),
                governor_json,
                brain_json,
                world_json,
                MODE_SUPERVISOR
            ],
        )?;
        let run_id = tx.last_insert_rowid();
        tx.commit()?;
        Ok(Self { db, run_id })
    }

    /// Database id of the run.
    pub fn run_id(&self) -> i64 {
        self.run_id
    }

    /// Writes the occupant of `slot`.
    ///
    /// # Errors
    ///
    /// Returns a [`StoreError`] when encoding or a statement fails.
    pub fn record_slot(&mut self, slot: usize, ledger: &SlotLedger) -> Result<(), StoreError> {
        let tx = self.db.transaction()?;
        upsert_slot(
            &tx,
            self.run_id,
            slot,
            ledger.occupant_id,
            ledger.born_tick,
            &ledger.config,
            &ledger.birth_brain,
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Writes one completed replacement boundary in one transaction.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Mismatch`] when `outcome.archived` and `archived`
    /// name different entries, and other [`StoreError`]s on failure.
    pub fn record_boundary(
        &mut self,
        tick: u64,
        plan: &ReplacementPlan,
        outcome: &ReplacementOutcome,
        archived: Option<&ArchiveEntry>,
    ) -> Result<(), StoreError> {
        let outcome_id = outcome.archived.as_ref().map(|entry| entry.entry_id);
        if outcome_id != archived.map(|entry| entry.entry_id) {
            return Err(StoreError::Mismatch);
        }
        let reason = match plan.reason {
            ReplaceReason::Death => REASON_DEATH,
            ReplaceReason::Underperformer => REASON_UNDERPERFORMER,
        };
        let (parent_kind, parent_ref) = match plan.parent {
            ParentRef::ArchiveEntry(id) => (PARENT_KIND_ARCHIVE, db_int(id)?),
            ParentRef::PeerBirth(peer) => (PARENT_KIND_PEER_BIRTH, db_index(peer)?),
            ParentRef::PeerLearned(peer) => (PARENT_KIND_PEER_LEARNED, db_index(peer)?),
        };
        let run_id = self.run_id;
        let tx = self.db.transaction()?;
        tx.execute(
            "INSERT INTO supervisor_event
                (run_id, tick, kind, slot, reason, occupant_id, parent_kind, parent_ref)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                run_id,
                db_int(tick)?,
                EVENT_REPLACE,
                db_index(plan.slot)?,
                reason,
                db_int(outcome.occupant_id)?,
                parent_kind,
                parent_ref
            ],
        )?;
        if let Some(entry) = archived {
            insert_archive_row(&tx, run_id, entry)?;
        }
        if let Some(evicted) = outcome.evicted {
            evict_entry(&tx, run_id, tick, evicted)?;
        }
        upsert_slot(
            &tx,
            run_id,
            plan.slot,
            outcome.occupant_id,
            plan.tick,
            &outcome.newborn_config,
            &outcome.newborn_brain,
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Writes a boundary that replaced nothing.
    ///
    /// # Errors
    ///
    /// Returns a [`StoreError`] when a statement fails.
    pub fn record_idle_boundary(&mut self, tick: u64) -> Result<(), StoreError> {
        let tx = self.db.transaction()?;
        tx.execute(
            "INSERT INTO supervisor_event (run_id, tick, kind) VALUES (?1, ?2, ?3)",
            params![self.run_id, db_int(tick)?, EVENT_IDLE],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Writes a season change to `kind`.
    ///
    /// # Errors
    ///
    /// Returns a [`StoreError`] when serialisation or a statement fails.
    pub fn record_season(&mut self, tick: u64, kind: FoodKind) -> Result<(), StoreError> {
        let kind_json = serde_json::to_string(&kind)?;
        let tx = self.db.transaction()?;
        tx.execute(
            "INSERT INTO supervisor_event (run_id, tick, kind, nourishing_kind)
             VALUES (?1, ?2, ?3, ?4)",
            params![self.run_id, db_int(tick)?, EVENT_SEASON, kind_json],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Writes the archive admissions made at the run end in one transaction.
    ///
    /// An entry that a later change evicted is no longer held by `archive` and
    /// gets no row.
    ///
    /// # Errors
    ///
    /// Returns a [`StoreError`] when encoding or a statement fails.
    pub fn record_archive_changes(
        &mut self,
        tick: u64,
        changes: &[ArchiveChange],
        archive: &Archive,
    ) -> Result<(), StoreError> {
        let run_id = self.run_id;
        let tx = self.db.transaction()?;
        for change in changes {
            if let Some(entry) = archive.get(change.entry.entry_id) {
                insert_archive_row(&tx, run_id, entry)?;
            }
            log_archive_event(&tx, run_id, tick, Some(change.slot), &change.entry)?;
            if let Some(evicted) = change.evicted {
                evict_entry(&tx, run_id, tick, evicted)?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Writes one telemetry frame in one transaction.
    ///
    /// A frame that is not marked for persistence writes nothing. A key that
    /// is already stored is an error.
    ///
    /// # Errors
    ///
    /// Returns a [`StoreError`] when a value is out of range, serialisation
    /// fails or a statement fails.
    pub fn record_telemetry(&mut self, frame: &TelemetryFrame) -> Result<(), StoreError> {
        if !frame.persist {
            return Ok(());
        }
        let run_id = self.run_id;
        let tick = db_int(frame.tick)?;
        let tx = self.db.transaction()?;
        for standing in &frame.slots {
            let parent = standing.parent.map(parent_columns).transpose()?;
            tx.execute(
                "INSERT INTO supervisor_slot_snapshot
                    (run_id, tick, slot, occupant_id, age_ticks, alive, meals, deaths,
                     window_score, peer_median, ground, parent_kind, parent_ref)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                params![
                    run_id,
                    tick,
                    db_index(standing.slot)?,
                    db_int(standing.occupant_id)?,
                    db_int(standing.age_ticks)?,
                    i64::from(standing.alive),
                    i64::from(standing.meals),
                    i64::from(standing.deaths),
                    standing
                        .window_score
                        .and_then(|v| finite_real(f64::from(v))),
                    standing.peer_median.and_then(|v| finite_real(f64::from(v))),
                    ground_text(&standing.ground),
                    parent.map(|p| p.0),
                    parent.map(|p| p.1)
                ],
            )?;
        }
        if let Some(record) = &frame.boundary {
            insert_boundary(&tx, run_id, frame.tick, record)?;
        }
        for point in &frame.series {
            tx.execute(
                "INSERT INTO supervisor_series (run_id, tick, metric, value)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    run_id,
                    db_int(point.tick)?,
                    point.metric.as_str(),
                    finite_real(point.value)
                ],
            )?;
        }
        for row in &frame.intake {
            tx.execute(
                "INSERT INTO supervisor_occupant_intake
                    (run_id, occupant_id, age_ticks, slot, meals)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    run_id,
                    db_int(row.occupant_id)?,
                    db_int(row.age_ticks)?,
                    db_index(row.slot)?,
                    i64::from(row.meals)
                ],
            )?;
        }
        let progress = &frame.progress;
        let nourishing = progress
            .nourishing
            .map(|kind| serde_json::to_string(&kind))
            .transpose()?;
        tx.execute(
            "INSERT INTO supervisor_progress
                (run_id, tick, total_ticks, agent_tick_budget, ticks_to_boundary,
                 nourishing_kind, next_flip_tick, ticks_per_second)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                run_id,
                db_int(progress.tick)?,
                progress.total_ticks.map(db_int).transpose()?,
                db_int(progress.agent_tick_budget)?,
                db_int(progress.ticks_to_boundary)?,
                nourishing,
                progress.next_flip_tick.map(db_int).transpose()?,
                progress.ticks_per_second.and_then(finite_real)
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Records the cumulative simulation ticks of the run.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Range`] when `ticks` exceeds the database range.
    pub fn record_ticks(&mut self, ticks: u64) -> Result<(), StoreError> {
        let ticks = i64::try_from(ticks).map_err(|_| StoreError::Range)?;
        let tx = self.db.transaction()?;
        tx.execute(
            "UPDATE run SET simulation_ticks = ?1 WHERE id = ?2",
            params![ticks, self.run_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Replaces the run's single checkpoint row with `payload` taken at `tick`.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Range`] when `tick` exceeds the database range,
    /// and [`StoreError::Sqlite`] when the run row is missing or a statement fails.
    pub fn write_checkpoint(&mut self, tick: u64, payload: &[u8]) -> Result<(), StoreError> {
        let tick = db_int(tick)?;
        let tx = self.db.transaction()?;
        let changed = tx.execute(
            "INSERT OR REPLACE INTO supervisor_checkpoint
                (run_id, tick, format_version, simulation_ticks, payload)
             SELECT id, ?1, ?2, simulation_ticks, ?3 FROM run WHERE id = ?4",
            params![tick, i64::from(FORMAT_VERSION), payload, self.run_id],
        )?;
        if changed == 0 {
            return Err(StoreError::Sqlite(rusqlite::Error::QueryReturnedNoRows));
        }
        tx.commit()?;
        Ok(())
    }

    /// The governor, brain and world configuration JSON of the run row.
    ///
    /// # Errors
    ///
    /// Returns a [`StoreError`] when the run row cannot be read.
    pub fn run_configs(&self) -> Result<(String, String, String), StoreError> {
        Ok(self.db.query_row(
            "SELECT governor_config, brain_config, world_config FROM run WHERE id = ?1",
            params![self.run_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?)
    }

    /// Reopens the database for a resume at `tick` of run `run_id`, deleting
    /// the run's rows at or after `tick` in one transaction.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::CheckpointTickMismatch`] when the run is not a
    /// supervisor run or its checkpoint is not at `tick`, and other
    /// [`StoreError`]s when a statement fails.
    pub fn reopen_for_resume(db_path: &str, run_id: i64, tick: u64) -> Result<Self, StoreError> {
        let requested = db_int(tick)?;
        let mut db = Connection::open(db_path)?;
        db.execute_batch(STORE_PRAGMAS)?;
        let stored: Option<i64> = db
            .query_row(
                "SELECT c.tick FROM supervisor_checkpoint c JOIN run r ON r.id = c.run_id
                 WHERE c.run_id = ?1 AND r.mode = ?2",
                params![run_id, MODE_SUPERVISOR],
                |row| row.get(0),
            )
            .optional()?;
        if stored != Some(requested) {
            return Err(StoreError::CheckpointTickMismatch {
                run_id,
                stored,
                requested: tick,
            });
        }
        let tx = db.transaction()?;
        let names: Vec<String> = {
            let mut stmt = tx.prepare(
                "SELECT m.name FROM sqlite_master m WHERE m.type = 'table'
                 AND m.name NOT LIKE 'sqlite_%'
                 AND m.name NOT IN ('supervisor_checkpoint', 'supervisor_archive')
                 AND EXISTS (SELECT 1 FROM pragma_table_info(m.name) c WHERE c.name = 'run_id')
                 AND EXISTS (SELECT 1 FROM pragma_table_info(m.name) c WHERE c.name = 'tick')",
            )?;
            let rows = stmt.query_map([], |row| row.get(0))?;
            rows.collect::<SqlResult<Vec<String>>>()?
        };
        for name in names {
            let quoted = name.replace('"', "\"\"");
            tx.execute(
                &format!("DELETE FROM \"{quoted}\" WHERE run_id = ?1 AND tick >= ?2"),
                params![run_id, requested],
            )?;
        }
        tx.execute(
            "DELETE FROM supervisor_archive WHERE run_id = ?1 AND retired_tick >= ?2",
            params![run_id, requested],
        )?;
        tx.execute(
            "UPDATE run SET simulation_ticks =
                (SELECT simulation_ticks FROM supervisor_checkpoint WHERE run_id = ?1)
             WHERE id = ?1
               AND (SELECT simulation_ticks FROM supervisor_checkpoint WHERE run_id = ?1)
                   IS NOT NULL",
            params![run_id],
        )?;
        tx.commit()?;
        Ok(Self { db, run_id })
    }
}

/// A checkpoint row read back from a supervisor database.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredCheckpoint {
    /// Run the checkpoint belongs to.
    pub run_id: i64,
    /// Tick the checkpoint was taken at.
    pub tick: u64,
    /// Envelope format version the payload was written with.
    pub format_version: u32,
    /// Opaque checkpoint bytes.
    pub payload: Vec<u8>,
}

/// An integer column that does not fit its unsigned type.
fn out_of_range(index: usize, value: i64) -> StoreError {
    StoreError::Sqlite(rusqlite::Error::IntegralValueOutOfRange(index, value))
}

/// Loads the checkpoint of the newest supervisor run without writing to or
/// creating the database; a missing file, run or checkpoint gives `None`.
///
/// # Errors
///
/// Returns [`StoreError::CheckpointTooLarge`] when the payload exceeds
/// [`MAX_CHECKPOINT_BYTES`], and other [`StoreError`]s when a read fails.
pub fn read_checkpoint(db_path: &str) -> Result<Option<StoredCheckpoint>, StoreError> {
    if !Path::new(db_path).exists() {
        return Ok(None);
    }
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let db = Connection::open_with_flags(db_path, flags)?;
    db.busy_timeout(STORE_BUSY_TIMEOUT)?;
    let tables: i64 = db.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'
         AND name IN ('run', 'supervisor_checkpoint')",
        [],
        |row| row.get(0),
    )?;
    if tables < 2 {
        return Ok(None);
    }
    let run_id: Option<i64> = db
        .query_row(
            "SELECT id FROM run WHERE mode = ?1 ORDER BY id DESC LIMIT 1",
            params![MODE_SUPERVISOR],
            |row| row.get(0),
        )
        .optional()?;
    let Some(run_id) = run_id else {
        return Ok(None);
    };
    let meta: Option<(i64, i64, i64)> = db
        .query_row(
            "SELECT tick, format_version, length(payload) FROM supervisor_checkpoint
             WHERE run_id = ?1",
            params![run_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    let Some((tick, format_version, len)) = meta else {
        return Ok(None);
    };
    if usize::try_from(len).map_or(true, |len| len > MAX_CHECKPOINT_BYTES) {
        return Err(StoreError::CheckpointTooLarge {
            len,
            max: MAX_CHECKPOINT_BYTES,
        });
    }
    let tick = u64::try_from(tick).map_err(|_| out_of_range(0, tick))?;
    let format_version =
        u32::try_from(format_version).map_err(|_| out_of_range(1, format_version))?;
    let payload: Vec<u8> = db.query_row(
        "SELECT payload FROM supervisor_checkpoint WHERE run_id = ?1",
        params![run_id],
        |row| row.get(0),
    )?;
    Ok(Some(StoredCheckpoint {
        run_id,
        tick,
        format_version,
        payload,
    }))
}

/// An archive entry read back from a supervisor database.
#[derive(Clone, Debug)]
pub struct StoredArchiveEntry {
    /// Identifier of the entry.
    pub entry_id: u64,
    /// Occupant the entry records.
    pub occupant_id: u64,
    /// Tick the occupant was born.
    pub born_tick: u64,
    /// Tick the occupant was retired.
    pub retired_tick: u64,
    /// Life outcome used for ranking.
    pub outcome: f64,
    /// Heritable configuration the occupant was born with.
    pub config: BrainConfig,
    /// Brain the occupant was born with.
    pub brain: AgentBrainState,
}

/// Whether the database has the tables and column a supervisor run needs.
fn has_supervisor_layout(db: &Connection) -> SqlResult<bool> {
    let mode_columns: i64 = db.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('run') WHERE name = 'mode'",
        [],
        |row| row.get(0),
    )?;
    let archive_tables: i64 = db.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'supervisor_archive'",
        [],
        |row| row.get(0),
    )?;
    Ok(mode_columns > 0 && archive_tables > 0)
}

/// One selected archive row.
type ArchiveRow = (i64, i64, i64, i64, f64, String, Option<Vec<u8>>);

/// Loads the archive entry at `rank` of the latest supervisor run without
/// writing to the database.
///
/// # Errors
///
/// Returns [`ArchiveLoadError`] when the database cannot be read, holds no
/// supervisor run, has no entry at `rank`, or the stored entry is malformed.
pub fn load_archive_entry_read_only(
    db_path: &str,
    rank: usize,
) -> Result<StoredArchiveEntry, ArchiveLoadError> {
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let db = Connection::open_with_flags(db_path, flags)?;
    db.busy_timeout(STORE_BUSY_TIMEOUT)?;
    if !has_supervisor_layout(&db)? {
        return Err(ArchiveLoadError::NoSupervisorRun);
    }
    let run_id: i64 = db
        .query_row(
            "SELECT id FROM run WHERE mode = ?1 ORDER BY id DESC LIMIT 1",
            params![MODE_SUPERVISOR],
            |row| row.get(0),
        )
        .optional()?
        .ok_or(ArchiveLoadError::NoSupervisorRun)?;
    let offset = i64::try_from(rank).unwrap_or(i64::MAX);
    let row: Option<ArchiveRow> = db
        .query_row(
            "SELECT entry_id, occupant_id, born_tick, retired_tick, outcome,
                    config_json, birth_brain
             FROM supervisor_archive WHERE run_id = ?1
             ORDER BY outcome DESC, entry_id ASC LIMIT 1 OFFSET ?2",
            params![run_id, offset],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            },
        )
        .optional()?;
    let Some((entry_id, occupant_id, born_tick, retired_tick, outcome, config_json, blob)) = row
    else {
        let held: i64 = db.query_row(
            "SELECT COUNT(*) FROM supervisor_archive WHERE run_id = ?1",
            params![run_id],
            |row| row.get(0),
        )?;
        return Err(ArchiveLoadError::RankOutOfRange {
            rank,
            len: usize::try_from(held).unwrap_or(0),
        });
    };
    let entry_id = entry_id.cast_unsigned();
    let brain = blob
        .as_deref()
        .and_then(decode_brain_state)
        .ok_or(ArchiveLoadError::MalformedBlob { entry_id })?;
    let config = serde_json::from_str::<BrainConfig>(&config_json)
        .map_err(|_| ArchiveLoadError::MalformedConfig { entry_id })?;
    Ok(StoredArchiveEntry {
        entry_id,
        occupant_id: occupant_id.cast_unsigned(),
        born_tick: born_tick.cast_unsigned(),
        retired_tick: retired_tick.cast_unsigned(),
        outcome,
        config,
        brain,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use xagent_shared::GovernorConfig;

    use super::*;
    use crate::agent::fresh_brain_state;
    use crate::governor::Governor;
    use crate::supervisor::Admission;

    /// Seed of the random streams the tests draw brains from.
    const TEST_BRAIN_SEED: u64 = 7;
    /// World seed above the signed range, to prove the signed cast.
    const TEST_WORLD_SEED: u64 = u64::MAX - 5;
    /// Archive capacity of the mirror test.
    const MIRROR_CAPACITY: usize = 2;
    /// Outcomes offered to the mirror test's archive, in order.
    const MIRROR_OUTCOMES: [f32; 5] = [0.3, 0.1, 0.5, 0.2, 0.9];

    /// Offset of a fixture entry's occupant id from its entry id.
    const TEST_OCCUPANT_ID_OFFSET: u64 = 10;
    /// Offset of a fixture entry's retired tick from its entry id.
    const TEST_RETIRED_TICK_OFFSET: u64 = 100;
    /// Number of supervisor tables the schema creates.
    const SUPERVISOR_TABLE_COUNT: i64 = 3;
    /// Archive capacity of the rank-order test.
    const RANK_CAPACITY: usize = 3;
    /// Outcomes offered to the rank-order test's archive.
    const RANK_OUTCOMES: [f32; 3] = [0.2, 0.8, 0.5];
    /// Archive capacity of the two-entry tests.
    const PAIR_CAPACITY: usize = 2;
    /// Outcomes offered to the two-entry tests' archive.
    const PAIR_OUTCOMES: [f32; 2] = [0.4, 0.6];
    /// Rank one past the last entry of the two-entry archive.
    const RANK_BEYOND_PAIR: usize = 2;
    /// Entry id of the hand-inserted malformed row.
    const MALFORMED_ENTRY_ID: u64 = 1;
    /// Outcome of the hand-inserted malformed row.
    const MALFORMED_OUTCOME: f64 = 0.5;
    /// A birth-brain blob too short to decode.
    const MALFORMED_BLOB: [u8; 3] = [1, 2, 3];

    fn test_entry(entry_id: u64, outcome: f32) -> ArchiveEntry {
        let mut rng = SmallRng::seed_from_u64(TEST_BRAIN_SEED + entry_id);
        ArchiveEntry {
            entry_id,
            occupant_id: entry_id + TEST_OCCUPANT_ID_OFFSET,
            born_tick: entry_id,
            retired_tick: entry_id + TEST_RETIRED_TICK_OFFSET,
            outcome,
            config: BrainConfig::default(),
            birth_brain: fresh_brain_state(&BrainConfig::default(), &mut rng),
        }
    }

    fn db_path(dir: &tempfile::TempDir, name: &str) -> String {
        dir.path().join(name).to_str().unwrap().to_string()
    }

    fn supervisor_config() -> FullConfig {
        let mut config = FullConfig::default();
        config.world.seed = TEST_WORLD_SEED;
        config
    }

    /// Checkpoints written by the overwrite test.
    const OVERWRITE_COUNT: u64 = 50;
    /// Tick the rewind tests checkpoint at.
    const REWIND_TICK: u64 = 10;
    /// Offset above `REWIND_TICK` of the first row written after the checkpoint.
    const REWIND_NEXT_TICK_OFFSET: u64 = 1;
    /// Offset above `REWIND_TICK` of the second row written after the checkpoint.
    const REWIND_LATE_TICK_OFFSET: u64 = 2;
    /// Offset above `REWIND_TICK` at which `next_ids` inserts its probe rows.
    const PROBE_TICK_OFFSET: u64 = 5;
    /// Checkpoint payload used by the rewind history.
    const CHECKPOINT_PAYLOAD: [u8; 1] = [9];
    /// Checkpoint payload used by the read-back round trip.
    const ROUND_TRIP_PAYLOAD: [u8; 3] = [1, 2, 3];
    /// Outcome of the archive rows the rewind fixture inserts.
    const FIXTURE_OUTCOME: f64 = 0.5;
    /// Ticks seeded for each run, below, at and after `REWIND_TICK`.
    const REWIND_TICKS: [u64; 4] = [
        REWIND_TICK - 1,
        REWIND_TICK,
        REWIND_TICK + REWIND_NEXT_TICK_OFFSET,
        REWIND_TICK + REWIND_LATE_TICK_OFFSET,
    ];

    #[test]
    fn store_checkpoint_single_row_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "ckpt.db");
        let mut store = SupervisorStore::create(&path, &supervisor_config()).unwrap();
        for tick in 1..=OVERWRITE_COUNT {
            let payload = tick.to_le_bytes();
            store.write_checkpoint(tick, &payload).unwrap();
        }
        let (count, tick, payload): (i64, i64, Vec<u8>) = store
            .db
            .query_row(
                "SELECT COUNT(*), tick, payload FROM supervisor_checkpoint",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(tick.cast_unsigned(), OVERWRITE_COUNT);
        assert_eq!(payload, OVERWRITE_COUNT.to_le_bytes().to_vec());
    }

    #[test]
    fn store_checkpoint_read_only_does_not_create_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "absent.db");
        assert_eq!(read_checkpoint(&path).unwrap(), None);
        assert!(!Path::new(&path).exists());
    }

    #[test]
    fn store_checkpoint_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "none.db");
        let mut store = SupervisorStore::create(&path, &supervisor_config()).unwrap();
        assert_eq!(read_checkpoint(&path).unwrap(), None);
        store
            .write_checkpoint(REWIND_TICK, &ROUND_TRIP_PAYLOAD)
            .unwrap();
        let stored = read_checkpoint(&path).unwrap().unwrap();
        assert_eq!(stored.run_id, store.run_id());
        assert_eq!(stored.tick, REWIND_TICK);
        assert_eq!(stored.format_version, FORMAT_VERSION);
        assert_eq!(stored.payload, ROUND_TRIP_PAYLOAD.to_vec());
    }

    /// Inserts one row per table for `run_id` at `tick`.
    fn insert_rewind_rows(db: &Connection, run_id: i64, tick: u64) {
        let tick = tick.cast_signed();
        db.execute(
            "INSERT INTO supervisor_event (run_id, tick, kind) VALUES (?1, ?2, 'idle')",
            params![run_id, tick],
        )
        .unwrap();
        db.execute(
            "INSERT INTO test_extra (run_id, tick) VALUES (?1, ?2)",
            params![run_id, tick],
        )
        .unwrap();
        db.execute(
            "INSERT INTO supervisor_archive
                (run_id, entry_id, occupant_id, born_tick, retired_tick, outcome,
                 config_json, birth_brain)
             VALUES (?1, ?2, 1, 0, ?2, ?3, 'c', X'00')",
            params![run_id, tick, FIXTURE_OUTCOME],
        )
        .unwrap();
    }

    /// Builds the two-run history; returns the first run id and its store.
    fn seed_rewind_history(path: &str) -> (i64, SupervisorStore) {
        let mut store = SupervisorStore::create(path, &supervisor_config()).unwrap();
        let first = store.run_id();
        store
            .db
            .execute_batch(
                "CREATE TABLE test_extra (id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL,
                 tick INTEGER NOT NULL);",
            )
            .unwrap();
        store
            .write_checkpoint(REWIND_TICK, &CHECKPOINT_PAYLOAD)
            .unwrap();
        for tick in REWIND_TICKS {
            insert_rewind_rows(&store.db, first, tick);
        }
        store
            .db
            .execute(
                "INSERT INTO run (seed, governor_config, brain_config, world_config, mode)
                 VALUES (1, '{}', '{}', '{}', 'supervisor')",
                [],
            )
            .unwrap();
        let second = store.db.last_insert_rowid();
        for tick in REWIND_TICKS {
            insert_rewind_rows(&store.db, second, tick);
        }
        (first, store)
    }

    /// Rows of `table` for `run_id`, as ticks in ascending order.
    fn run_ticks(db: &Connection, table: &str, column: &str, run_id: i64) -> Vec<i64> {
        let mut stmt = db
            .prepare(&format!(
                "SELECT {column} FROM {table} WHERE run_id = ?1 ORDER BY {column}"
            ))
            .unwrap();
        let rows = stmt.query_map(params![run_id], |row| row.get(0)).unwrap();
        rows.map(Result::unwrap).collect()
    }

    /// Inserts one probe row per table at a fixed tick and returns the ids
    /// assigned to those rows.
    fn next_ids(db: &Connection, run_id: i64) -> Vec<i64> {
        insert_rewind_rows(db, run_id, REWIND_TICK + PROBE_TICK_OFFSET);
        ["supervisor_event", "test_extra", "supervisor_archive"]
            .iter()
            .map(|table| {
                db.query_row(&format!("SELECT MAX(id) FROM {table}"), [], |row| {
                    row.get(0)
                })
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn store_reopen_rewinds_rows_at_or_after_tick() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "rewind.db");
        let (first, store) = seed_rewind_history(&path);
        let second = first + 1;
        drop(store);
        let reopened = SupervisorStore::reopen_for_resume(&path, first, REWIND_TICK).unwrap();
        assert_eq!(reopened.run_id(), first);
        let below = vec![(REWIND_TICK - 1).cast_signed()];
        let full: Vec<i64> = REWIND_TICKS.iter().map(|tick| tick.cast_signed()).collect();
        for (table, column) in [
            ("supervisor_event", "tick"),
            ("test_extra", "tick"),
            ("supervisor_archive", "retired_tick"),
        ] {
            assert_eq!(run_ticks(&reopened.db, table, column, first), below);
            assert_eq!(run_ticks(&reopened.db, table, column, second), full);
        }
        assert_eq!(row_count(&reopened.db, "supervisor_checkpoint"), 1);
        let rewound_ids = next_ids(&reopened.db, second);

        let path_b = db_path(&dir, "plain.db");
        let (_, plain) = seed_rewind_history(&path_b);
        assert_eq!(next_ids(&plain.db, second), rewound_ids);

        drop(reopened);
        let err =
            SupervisorStore::reopen_for_resume(&path, first, REWIND_TICK + REWIND_NEXT_TICK_OFFSET)
                .err()
                .unwrap();
        assert!(matches!(
            err,
            StoreError::CheckpointTickMismatch {
                stored: Some(_),
                ..
            }
        ));
    }

    /// Creates a store, offers `outcomes` to a bounded archive and mirrors
    /// every admission; returns the archive.
    fn mirrored_archive(path: &str, outcomes: &[f32], capacity: usize) -> Archive {
        let mut store = SupervisorStore::create(path, &supervisor_config()).unwrap();
        let mut archive = Archive::new(capacity);
        for (tick, outcome) in (1u64..).zip(outcomes) {
            let entry = test_entry(archive.next_entry_id(), *outcome);
            if let Admission::Admitted { evicted } = archive.insert(entry.clone()) {
                let change = ArchiveChange {
                    slot: 0,
                    entry,
                    evicted,
                };
                store
                    .record_archive_changes(tick, &[change], &archive)
                    .unwrap();
            }
        }
        archive
    }

    fn table_count(db: &Connection) -> i64 {
        db.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN
             ('supervisor_slot', 'supervisor_archive', 'supervisor_event')",
            [],
            |row| row.get(0),
        )
        .unwrap()
    }

    fn file_hash(path: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        std::fs::read(path).unwrap().hash(&mut hasher);
        hasher.finish()
    }

    fn generation_database(path: &str) {
        let _gov = Governor::new_seeded(
            path,
            GovernorConfig::default(),
            &BrainConfig::default(),
            "{}",
            1,
        )
        .unwrap();
    }

    /// Tables the telemetry schema adds.
    const TELEMETRY_TABLES: [&str; 6] = [
        "supervisor_slot_snapshot",
        "supervisor_boundary",
        "supervisor_boundary_candidate",
        "supervisor_series",
        "supervisor_occupant_intake",
        "supervisor_progress",
    ];

    fn row_count(db: &Connection, table: &str) -> i64 {
        db.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
            row.get(0)
        })
        .unwrap()
    }

    /// Age in ticks of every fixture standing.
    const TEST_STANDING_AGE_TICKS: u64 = 500;
    /// Meal count of every fixture standing and intake row.
    const TEST_STANDING_MEALS: u32 = 3;
    /// Peer median of every fixture standing and below/above-median ground.
    const TEST_PEER_MEDIAN: f32 = 0.5;
    /// Score of the fixture candidate below the median.
    const TEST_LOW_SCORE: f32 = 0.1;
    /// Score of the fixture candidate above the median.
    const TEST_HIGH_SCORE: f32 = 0.9;
    /// Window score of the fixture slot snapshot.
    const TEST_SNAPSHOT_SCORE: f32 = 0.2;
    /// Tick of the fixture frame holding a replacement boundary.
    const TEST_REPLACE_TICK: u64 = 100;
    /// Tick of the fixture frame holding an idle boundary.
    const TEST_IDLE_TICK: u64 = 200;
    /// Tick of the fixture frame that is not persisted.
    const TEST_SKIPPED_TICK: u64 = 300;
    /// Occupant id of the fixture slot zero occupant.
    const TEST_OCCUPANT_A: u64 = 10;
    /// Occupant id of the fixture slot one occupant.
    const TEST_OCCUPANT_B: u64 = 11;
    /// Occupant id of the fixture slot two occupant.
    const TEST_OCCUPANT_C: u64 = 12;
    /// Occupant id of the fixture newborn.
    const TEST_NEWBORN: u64 = 20;
    /// Agent tick budget of the fixture progress record.
    const TEST_TICK_BUDGET: u64 = 1000;
    /// Ticks to the next boundary in the fixture progress record.
    const TEST_TICKS_TO_BOUNDARY: u64 = 10;
    /// Series value of the fixture series point.
    const TEST_SERIES_VALUE: f64 = 0.25;
    /// Entry id of the fixture archive parent.
    const TEST_PARENT_ENTRY: u64 = 4;
    /// Peer slot of the fixture learned parent.
    const TEST_PEER_SLOT: usize = 2;
    /// Parent pool length of the fixture parent draw.
    const TEST_POOL_LEN: usize = 3;
    /// Retired lifespan in ticks of the fixture replacement.
    const TEST_LIFESPAN: u64 = 100;
    /// Idle counts of the fixture idle boundary, in field order.
    const TEST_IDLE_COUNTS: [usize; 4] = [1, 2, 3, 4];
    /// Value carried by the newborn side of the fixture config difference.
    const TEST_NEWBORN_FIELD_VALUE: u64 = 2;
    /// Occupant id hand-inserted into the pre-existing slot row of the migration test.
    const TEST_MIGRATION_OCCUPANT: i64 = 5;
    /// Tick hand-inserted into the pre-existing event row of the migration test.
    const TEST_MIGRATION_EVENT_TICK: i64 = 9;
    /// Number of candidate rows a replaced boundary with one death and one below-median standing writes.
    const TEST_CANDIDATE_ROWS: i64 = 2;

    fn standing(
        slot: usize,
        occupant_id: u64,
        ground: EligibilityGround,
        window_score: Option<f32>,
    ) -> crate::supervisor::SlotStanding {
        crate::supervisor::SlotStanding {
            slot,
            occupant_id,
            age_ticks: TEST_STANDING_AGE_TICKS,
            alive: true,
            meals: TEST_STANDING_MEALS,
            deaths: 1,
            window_score,
            peer_median: Some(TEST_PEER_MEDIAN),
            ground,
            parent: Some(ParentRef::PeerLearned(TEST_PEER_SLOT)),
        }
    }

    fn progress(tick: u64) -> crate::supervisor_telemetry::ProgressRecord {
        crate::supervisor_telemetry::ProgressRecord {
            tick,
            total_ticks: None,
            agent_tick_budget: TEST_TICK_BUDGET,
            ticks_to_boundary: TEST_TICKS_TO_BOUNDARY,
            nourishing: Some(FoodKind::First),
            next_flip_tick: None,
            ticks_per_second: Some(f64::NAN),
        }
    }

    fn frame(tick: u64, persist: bool) -> TelemetryFrame {
        use crate::supervisor::{IdleReason, TieBreak};
        use crate::supervisor_telemetry::{ConfigFieldDiff, IntakeRow, Metric, SeriesPoint};
        let boundary = if tick == TEST_REPLACE_TICK {
            BoundaryRecord {
                detail: BoundaryDetail::Replaced {
                    tick,
                    candidates: vec![
                        standing(0, TEST_OCCUPANT_A, EligibilityGround::Death, None),
                        standing(
                            1,
                            TEST_OCCUPANT_B,
                            EligibilityGround::BelowMedian {
                                score: TEST_LOW_SCORE,
                                median: TEST_PEER_MEDIAN,
                            },
                            Some(TEST_LOW_SCORE),
                        ),
                        standing(
                            TEST_PEER_SLOT,
                            TEST_OCCUPANT_C,
                            EligibilityGround::AtOrAboveMedian {
                                score: TEST_HIGH_SCORE,
                                median: TEST_PEER_MEDIAN,
                            },
                            Some(TEST_HIGH_SCORE),
                        ),
                    ],
                    slot: 0,
                    reason: ReplaceReason::Death,
                    tiebreak: TieBreak {
                        pool: vec![0, 1],
                        drawn_index: 0,
                    },
                    parent: ParentRef::ArchiveEntry(TEST_PARENT_ENTRY),
                    parent_draw: Some(crate::supervisor::ParentDraw {
                        pool_len: TEST_POOL_LEN,
                        drawn_index: 1,
                    }),
                    retired_occupant: TEST_OCCUPANT_A,
                    retired_born_tick: 0,
                },
                newborn_occupant: Some(TEST_NEWBORN),
                config_diff: vec![ConfigFieldDiff {
                    field: "a".to_string(),
                    parent: json!(1),
                    newborn: json!(TEST_NEWBORN_FIELD_VALUE),
                }],
                retired_lifespan_ticks: Some(TEST_LIFESPAN),
            }
        } else {
            BoundaryRecord {
                detail: BoundaryDetail::Idle {
                    tick,
                    standings: Vec::new(),
                    reason: IdleReason {
                        too_young: TEST_IDLE_COUNTS[0],
                        no_full_window: TEST_IDLE_COUNTS[1],
                        no_peer_median: TEST_IDLE_COUNTS[2],
                        at_or_above_median: TEST_IDLE_COUNTS[3],
                    },
                },
                newborn_occupant: None,
                config_diff: Vec::new(),
                retired_lifespan_ticks: None,
            }
        };
        TelemetryFrame {
            tick,
            slots: vec![standing(
                0,
                TEST_OCCUPANT_A,
                EligibilityGround::TooYoung,
                Some(TEST_SNAPSHOT_SCORE),
            )],
            boundary: Some(boundary),
            series: vec![SeriesPoint {
                tick,
                metric: Metric::FoodPerAgentTick,
                value: TEST_SERIES_VALUE,
            }],
            intake: vec![IntakeRow {
                occupant_id: TEST_OCCUPANT_A,
                slot: 0,
                age_ticks: tick,
                meals: TEST_STANDING_MEALS,
            }],
            progress: progress(tick),
            persist,
        }
    }

    #[test]
    fn telemetry_migration_keeps_old_rows_and_adds_tables_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "old.db");
        let db = Connection::open(&path).unwrap();
        db.execute_batch(STORE_PRAGMAS).unwrap();
        init_schema(&db).unwrap();
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS supervisor_slot (
            run_id INTEGER NOT NULL REFERENCES run(id),
            slot INTEGER NOT NULL,
            occupant_id INTEGER NOT NULL,
            born_tick INTEGER NOT NULL,
            config_json TEXT NOT NULL,
            birth_brain BLOB NOT NULL,
            PRIMARY KEY (run_id, slot)
        );

        CREATE TABLE IF NOT EXISTS supervisor_archive (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES run(id),
            entry_id INTEGER NOT NULL,
            occupant_id INTEGER NOT NULL,
            born_tick INTEGER NOT NULL,
            retired_tick INTEGER NOT NULL,
            outcome REAL NOT NULL,
            config_json TEXT NOT NULL,
            birth_brain BLOB NOT NULL,
            UNIQUE (run_id, entry_id)
        );

        CREATE TABLE IF NOT EXISTS supervisor_event (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES run(id),
            tick INTEGER NOT NULL,
            kind TEXT NOT NULL,
            slot INTEGER,
            reason TEXT,
            occupant_id INTEGER,
            parent_kind TEXT,
            parent_ref INTEGER,
            outcome REAL,
            nourishing_kind TEXT
        );",
        )
        .unwrap();
        db.execute(
            "INSERT INTO run (seed, governor_config, brain_config, world_config, mode)
             VALUES (1, '{}', '{}', '{}', 'supervisor')",
            [],
        )
        .unwrap();
        let run_id = db.last_insert_rowid();
        db.execute(
            "INSERT INTO supervisor_slot VALUES (?1, 0, ?2, 6, 'c', X'00')",
            params![run_id, TEST_MIGRATION_OCCUPANT],
        )
        .unwrap();
        db.execute(
            "INSERT INTO supervisor_archive
                (run_id, entry_id, occupant_id, born_tick, retired_tick, outcome,
                 config_json, birth_brain)
             VALUES (?1, 1, 2, 3, 4, 0.5, 'c', X'00')",
            params![run_id],
        )
        .unwrap();
        db.execute(
            "INSERT INTO supervisor_event (run_id, tick, kind) VALUES (?1, ?2, 'idle')",
            params![run_id, TEST_MIGRATION_EVENT_TICK],
        )
        .unwrap();
        ensure_schema(&db).unwrap();
        ensure_schema(&db).unwrap();
        for table in ["supervisor_slot", "supervisor_archive", "supervisor_event"] {
            assert_eq!(row_count(&db, table), 1, "{table}");
        }
        let occupant: i64 = db
            .query_row("SELECT occupant_id FROM supervisor_slot", [], |r| r.get(0))
            .unwrap();
        assert_eq!(occupant, TEST_MIGRATION_OCCUPANT);
        let tick: i64 = db
            .query_row("SELECT tick FROM supervisor_event", [], |r| r.get(0))
            .unwrap();
        assert_eq!(tick, TEST_MIGRATION_EVENT_TICK);
        for table in TELEMETRY_TABLES {
            let tables: i64 = db
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
                    params![table],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(tables, 1, "{table}");
            let notnull: i64 = db
                .query_row(
                    &format!("SELECT \"notnull\" FROM pragma_table_info('{table}') WHERE name = 'run_id'"),
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(notnull, 1, "{table}");
            let fk: i64 = db
                .query_row(
                    &format!(
                        "SELECT COUNT(*) FROM pragma_foreign_key_list('{table}')
                         WHERE \"table\" = 'run' AND \"from\" = 'run_id' AND \"to\" = 'id'"
                    ),
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(fk, 1, "{table}");
        }
    }

    #[test]
    fn telemetry_writer_round_trips_frames() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "telemetry.db");
        let mut store = SupervisorStore::create(&path, &supervisor_config()).unwrap();
        store
            .record_telemetry(&frame(TEST_REPLACE_TICK, true))
            .unwrap();
        assert_eq!(
            row_count(&store.db, "supervisor_boundary_candidate"),
            TEST_CANDIDATE_ROWS
        );
        assert_eq!(row_count(&store.db, "supervisor_slot_snapshot"), 1);
        let outcome: String = store
            .db
            .query_row("SELECT outcome FROM supervisor_boundary", [], |r| r.get(0))
            .unwrap();
        assert_eq!(outcome, "replace");
        let ground: String = store
            .db
            .query_row(
                "SELECT ground FROM supervisor_boundary_candidate WHERE slot = 0",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(ground, "death");
        store
            .record_telemetry(&frame(TEST_IDLE_TICK, true))
            .unwrap();
        let (idle, diff): (i64, Option<String>) = store
            .db
            .query_row(
                "SELECT idle_too_young, config_diff FROM supervisor_boundary WHERE tick = ?1",
                params![i64::try_from(TEST_IDLE_TICK).unwrap()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(idle, 1);
        assert!(diff.is_none());
        assert_eq!(
            row_count(&store.db, "supervisor_boundary_candidate"),
            TEST_CANDIDATE_ROWS
        );
        let before: Vec<i64> = TELEMETRY_TABLES
            .iter()
            .map(|t| row_count(&store.db, t))
            .collect();
        store
            .record_telemetry(&frame(TEST_SKIPPED_TICK, false))
            .unwrap();
        let after: Vec<i64> = TELEMETRY_TABLES
            .iter()
            .map(|t| row_count(&store.db, t))
            .collect();
        assert_eq!(before, after);
        assert!(store
            .record_telemetry(&frame(TEST_REPLACE_TICK, true))
            .is_err());
        let after_error: Vec<i64> = TELEMETRY_TABLES
            .iter()
            .map(|t| row_count(&store.db, t))
            .collect();
        assert_eq!(before, after_error);
    }

    #[test]
    fn telemetry_nan_score_is_stored_as_null() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "nan.db");
        let mut store = SupervisorStore::create(&path, &supervisor_config()).unwrap();
        let mut nan_frame = frame(TEST_IDLE_TICK, true);
        nan_frame.slots = vec![standing(
            0,
            TEST_OCCUPANT_A,
            EligibilityGround::TooYoung,
            Some(f32::NAN),
        )];
        store.record_telemetry(&nan_frame).unwrap();
        let score: Option<f64> = store
            .db
            .query_row(
                "SELECT window_score FROM supervisor_slot_snapshot",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(score.is_none());
    }

    #[test]
    fn store_schema_is_idempotent() {
        let db = Connection::open_in_memory().unwrap();
        ensure_schema(&db).unwrap();
        ensure_schema(&db).unwrap();
        assert_eq!(table_count(&db), SUPERVISOR_TABLE_COUNT);
    }

    #[test]
    fn store_create_records_supervisor_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "create.db");
        let store = SupervisorStore::create(&path, &supervisor_config()).unwrap();
        let run_id = store.run_id();
        drop(store);
        let db = Connection::open(&path).unwrap();
        let (count, mode, seed, id): (i64, String, i64, i64) = db
            .query_row("SELECT COUNT(*), mode, seed, id FROM run", [], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(mode, MODE_SUPERVISOR);
        assert_eq!(seed, TEST_WORLD_SEED.cast_signed());
        assert_eq!(id, run_id);
    }

    #[test]
    fn store_refuses_non_fresh_database() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "used.db");
        generation_database(&path);
        let result = SupervisorStore::create(&path, &supervisor_config());
        assert!(matches!(result, Err(StoreError::NotFresh)));
        let db = Connection::open(&path).unwrap();
        let runs: i64 = db
            .query_row("SELECT COUNT(*) FROM run", [], |row| row.get(0))
            .unwrap();
        assert_eq!(runs, 1);
    }

    #[test]
    fn store_schema_on_generation_database_keeps_it_resumable() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "resumable.db");
        generation_database(&path);
        let db = Connection::open(&path).unwrap();
        ensure_schema(&db).unwrap();
        drop(db);
        assert!(Governor::resume(&path).is_ok());
        let db = Connection::open(&path).unwrap();
        let mode: String = db
            .query_row("SELECT mode FROM run", [], |row| row.get(0))
            .unwrap();
        assert_eq!(mode, "generations");
    }

    #[test]
    fn store_archive_table_mirrors_bounded_archive() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "mirror.db");
        let archive = mirrored_archive(&path, &MIRROR_OUTCOMES, MIRROR_CAPACITY);
        assert_eq!(archive.len(), MIRROR_CAPACITY);
        let db = Connection::open(&path).unwrap();
        let mut statement = db
            .prepare(
                "SELECT entry_id, outcome FROM supervisor_archive
                 ORDER BY outcome DESC, entry_id ASC",
            )
            .unwrap();
        let rows: Vec<(u64, f64)> = statement
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?.cast_unsigned(), row.get(1)?))
            })
            .unwrap()
            .map(Result::unwrap)
            .collect();
        let expected: Vec<(u64, f64)> = archive
            .entries()
            .iter()
            .map(|entry| (entry.entry_id, f64::from(entry.outcome)))
            .collect();
        assert_eq!(rows, expected);
    }

    #[test]
    fn archive_loader_returns_rank_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "rank.db");
        let archive = mirrored_archive(&path, &RANK_OUTCOMES, RANK_CAPACITY);
        let loaded: Vec<StoredArchiveEntry> = (0..RANK_CAPACITY)
            .map(|rank| load_archive_entry_read_only(&path, rank).unwrap())
            .collect();
        assert!(loaded
            .windows(2)
            .all(|pair| pair[0].outcome >= pair[1].outcome));
        for (stored, entry) in loaded.iter().zip(archive.entries()) {
            assert_eq!(stored.entry_id, entry.entry_id);
            assert_eq!(stored.outcome, f64::from(entry.outcome));
            assert_eq!(stored.brain.brain_state, entry.birth_brain.brain_state);
        }
    }

    #[test]
    fn archive_loader_leaves_database_bytes_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "bytes.db");
        mirrored_archive(&path, &PAIR_OUTCOMES, PAIR_CAPACITY);
        let before = file_hash(&path);
        load_archive_entry_read_only(&path, 0).unwrap();
        load_archive_entry_read_only(&path, 1).unwrap();
        assert_eq!(file_hash(&path), before);
    }

    #[test]
    fn archive_loader_rejects_rank_beyond_archive() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "beyond.db");
        mirrored_archive(&path, &PAIR_OUTCOMES, PAIR_CAPACITY);
        let result = load_archive_entry_read_only(&path, RANK_BEYOND_PAIR);
        assert!(matches!(
            result,
            Err(ArchiveLoadError::RankOutOfRange { rank, len })
                if rank == RANK_BEYOND_PAIR && len == PAIR_CAPACITY
        ));
    }

    #[test]
    fn archive_loader_rejects_generation_database() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "generations.db");
        generation_database(&path);
        let result = load_archive_entry_read_only(&path, 0);
        assert!(matches!(result, Err(ArchiveLoadError::NoSupervisorRun)));
    }

    #[test]
    fn archive_loader_rejects_malformed_blob() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir, "malformed.db");
        let store = SupervisorStore::create(&path, &supervisor_config()).unwrap();
        let run_id = store.run_id();
        drop(store);
        let db = Connection::open(&path).unwrap();
        db.execute(
            "INSERT INTO supervisor_archive
                (run_id, entry_id, occupant_id, born_tick, retired_tick, outcome,
                 config_json, birth_brain)
             VALUES (?1, ?4, 1, 0, 1, ?5, ?2, ?3)",
            params![
                run_id,
                serde_json::to_string(&BrainConfig::default()).unwrap(),
                MALFORMED_BLOB.to_vec(),
                MALFORMED_ENTRY_ID.cast_signed(),
                MALFORMED_OUTCOME
            ],
        )
        .unwrap();
        drop(db);
        let result = load_archive_entry_read_only(&path, 0);
        assert!(matches!(
            result,
            Err(ArchiveLoadError::MalformedBlob {
                entry_id: MALFORMED_ENTRY_ID
            })
        ));
    }
}
