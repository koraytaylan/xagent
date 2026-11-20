//! SQLite persistence of a supervisor run.
//!
//! The store adds three tables to the governor schema: the occupant of each
//! slot, the bounded archive of retired lives and an event log. Every change a
//! boundary, season, idle sample or run end makes is written in one
//! transaction. Archive entries are loaded back through a read-only
//! connection that never writes to the database.

use std::fmt;
use std::time::Duration;

use rusqlite::{params, Connection, OpenFlags, OptionalExtension, Result as SqlResult};
use xagent_brain::buffers::AgentBrainState;
use xagent_shared::{BrainConfig, FoodKind, FullConfig};

use crate::governor::{decode_brain_state, encode_brain_state, init_schema};
use crate::supervisor::{
    Archive, ArchiveChange, ArchiveEntry, ParentRef, ReplaceReason, ReplacementOutcome,
    ReplacementPlan, SlotLedger,
};

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
        );",
    )
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
