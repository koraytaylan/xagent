//! Read-only loading of a stored supervisor run into an observatory model.
//!
//! Nothing here writes to, creates or migrates a database: the connection is
//! opened read-only, a missing file is reported instead of created, and no
//! schema or writer routine is called.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use xagent_shared::{BrainConfig, FoodKind};

use super::{
    ArchiveRow, BoundaryRecord, ConfigDelta, EligibilityGround, EligibleSlot, ModelSource,
    ObservatoryModel, ObservatoryRecord, OccupantRecord, SeasonState, SeriesPoint, SlotRow,
    TimelineEvent, TimelineKind, MAX_ARCHIVE_ROWS, MAX_BOUNDARY_RECORDS, MAX_DISPLAY_TEXT_BYTES,
    MAX_INTAKE_POINTS, MAX_SERIES_POINTS, MAX_SLOTS, MAX_TIMELINE_EVENTS, MAX_TRACKED_OCCUPANTS,
};
use crate::governor::decode_brain_state;
use crate::supervisor::ParentRef as SourceParent;
use crate::supervisor_store::{
    EVENT_ARCHIVE, EVENT_EVICT, EVENT_IDLE, EVENT_REPLACE, EVENT_SEASON, MODE_SUPERVISOR,
    STORE_BUSY_TIMEOUT,
};
use crate::supervisor_telemetry::{IntakeRow, Metric, ProgressRecord};

/// Tables a stored supervisor run is read from.
const REQUIRED_TABLES: [&str; 10] = [
    "run",
    "supervisor_slot",
    "supervisor_archive",
    "supervisor_event",
    "supervisor_slot_snapshot",
    "supervisor_boundary",
    "supervisor_boundary_candidate",
    "supervisor_series",
    "supervisor_occupant_intake",
    "supervisor_progress",
];
/// Parent kind stored for an archive-entry parent.
const PARENT_KIND_ARCHIVE: &str = "archive_entry";
/// Parent kind stored for a peer's birth brain.
const PARENT_KIND_PEER_BIRTH: &str = "peer_birth";
/// Parent kind stored for a peer's learned state.
const PARENT_KIND_PEER_LEARNED: &str = "peer_learned";
/// Eligibility ground stored for a death.
const GROUND_DEATH: &str = "death";
/// Eligibility ground stored for a score below the median.
const GROUND_BELOW_MEDIAN: &str = "below_median";
/// Fold rank of archive rows within a tick.
const ORDER_ARCHIVE: u8 = 0;
/// Fold rank of occupants within a tick.
const ORDER_OCCUPANT: u8 = 1;
/// Fold rank of intake rows within a tick.
const ORDER_INTAKE: u8 = 2;
/// Fold rank of slot rows within a tick.
const ORDER_SLOT: u8 = 3;
/// Fold rank of boundaries within a tick.
const ORDER_BOUNDARY: u8 = 4;
/// Fold rank of series points within a tick.
const ORDER_SERIES: u8 = 5;
/// Fold rank of season state within a tick.
const ORDER_SEASON: u8 = 6;
/// Fold rank of progress within a tick.
const ORDER_PROGRESS: u8 = 7;
/// Fold rank of timeline events within a tick.
const ORDER_EVENT: u8 = 8;
/// Stored series metrics.
const KNOWN_METRICS: [Metric; 9] = [
    Metric::FoodPerAgentTick,
    Metric::DeathsPerAgentTick,
    Metric::WindowScoreSpread,
    Metric::RetiredLifespanTicks,
    Metric::ArchiveBestOutcome,
    Metric::ArchiveMedianOutcome,
    Metric::LateTowardNourishing,
    Metric::LateAwayHarmful,
    Metric::LateNourishingShare,
];
/// Most candidate rows read.
const MAX_CANDIDATE_ROWS: usize = MAX_BOUNDARY_RECORDS * MAX_SLOTS;
/// Most intake rows read; each tracked occupant keeps at most this many points.
const MAX_INTAKE_ROWS: usize = MAX_TRACKED_OCCUPANTS * MAX_INTAKE_POINTS;

/// Why a stored run could not be browsed.
#[derive(Debug)]
pub enum BrowseError {
    /// The database path is not an existing regular file.
    Missing,
    /// SQLite reported an error.
    Sqlite(rusqlite::Error),
    /// The run id is not a supervisor run.
    NoRun {
        /// Run id that was asked for.
        run_id: i64,
    },
    /// A stored value cannot be read as a model record.
    Malformed {
        /// What is wrong with the stored value.
        detail: String,
    },
}

impl fmt::Display for BrowseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing => write!(f, "database file does not exist"),
            Self::Sqlite(err) => write!(f, "database query failed: {err}"),
            Self::NoRun { run_id } => write!(f, "run {run_id} is not a supervisor run"),
            Self::Malformed { detail } => write!(f, "stored record is malformed: {detail}"),
        }
    }
}

impl std::error::Error for BrowseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sqlite(err) => Some(err),
            _ => None,
        }
    }
}

impl From<rusqlite::Error> for BrowseError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Sqlite(err)
    }
}

/// The latest supervisor run of a database.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SupervisorRunSummary {
    /// Run id in the database.
    pub run_id: i64,
    /// Latest simulation tick stored for the run.
    pub last_tick: u64,
    /// Tick budget of the run, when it has one.
    pub budget_ticks: Option<u64>,
}

fn malformed(detail: &str) -> BrowseError {
    BrowseError::Malformed {
        detail: detail.to_owned(),
    }
}

fn to_u64(value: i64, what: &str) -> Result<u64, BrowseError> {
    u64::try_from(value).map_err(|_| malformed(&format!("{what} is out of range")))
}

fn to_usize(value: i64, what: &str) -> Result<usize, BrowseError> {
    usize::try_from(value).map_err(|_| malformed(&format!("{what} is out of range")))
}

fn to_u32(value: i64, what: &str) -> Result<u32, BrowseError> {
    u32::try_from(value).map_err(|_| malformed(&format!("{what} is out of range")))
}

fn req<T>(value: Option<T>, what: &str) -> Result<T, BrowseError> {
    value.ok_or_else(|| malformed(&format!("{what} is missing")))
}

fn finite(value: f64, what: &str) -> Result<f64, BrowseError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(malformed(&format!("{what} is not finite")))
    }
}

fn opt_finite(value: Option<f64>, what: &str) -> Result<Option<f64>, BrowseError> {
    value.map(|v| finite(v, what)).transpose()
}

/// Narrows a stored value to the model's single precision; the stored value
/// and the narrowed value must both be finite.
#[allow(clippy::cast_possible_truncation)]
fn narrow(value: f64, what: &str) -> Result<f32, BrowseError> {
    let narrowed = finite(value, what)? as f32;
    if narrowed.is_finite() {
        Ok(narrowed)
    } else {
        Err(malformed(&format!("{what} is out of range")))
    }
}

fn opt_narrow(value: Option<f64>, what: &str) -> Result<Option<f32>, BrowseError> {
    value.map(|v| narrow(v, what)).transpose()
}

/// A record with the tick and category rank that fix its fold position.
struct Item {
    tick: u64,
    order: u8,
    record: ObservatoryRecord,
    /// Stored parent, resolved against the model when the record is folded.
    parent: Option<SourceParent>,
}

impl Item {
    fn new(tick: u64, order: u8, record: ObservatoryRecord) -> Self {
        Self {
            tick,
            order,
            record,
            parent: None,
        }
    }
}

fn limit(cap: usize) -> Result<i64, BrowseError> {
    i64::try_from(cap).map_err(|_| malformed("cap is out of range"))
}

/// Truncates `text` to the display limit at a character boundary.
fn truncated(mut text: String) -> String {
    if text.len() > MAX_DISPLAY_TEXT_BYTES {
        let mut end = MAX_DISPLAY_TEXT_BYTES;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    text
}

/// Opens `db_path` read-only; a missing file is never created.
///
/// # Errors
///
/// Returns [`BrowseError::Missing`] when the path is not a regular file and
/// [`BrowseError::Sqlite`] when the open or a pragma fails.
pub fn open_read_only(db_path: &str) -> Result<Connection, BrowseError> {
    if !Path::new(db_path).is_file() {
        return Err(BrowseError::Missing);
    }
    let db = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    db.busy_timeout(STORE_BUSY_TIMEOUT)?;
    db.execute_batch("PRAGMA query_only = ON;")?;
    Ok(db)
}

fn has_tables(db: &Connection) -> Result<bool, BrowseError> {
    let mut stmt =
        db.prepare("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1")?;
    for name in REQUIRED_TABLES {
        let found: i64 = stmt.query_row([name], |row| row.get(0))?;
        if found == 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Finds the latest supervisor run of the database at `db_path`.
///
/// Returns `None` when the path is missing, the tables are absent or no
/// supervisor run is stored.
///
/// # Errors
///
/// Returns a [`BrowseError`] when a query fails or a stored value is malformed.
pub fn detect_supervisor_run(db_path: &str) -> Result<Option<SupervisorRunSummary>, BrowseError> {
    let db = match open_read_only(db_path) {
        Ok(db) => db,
        Err(BrowseError::Missing) => return Ok(None),
        Err(err) => return Err(err),
    };
    if !has_tables(&db)? {
        return Ok(None);
    }
    let latest: Option<(i64, Option<i64>)> = db
        .query_row(
            "SELECT id, simulation_ticks FROM run WHERE mode = ?1 ORDER BY id DESC LIMIT 1",
            params![MODE_SUPERVISOR],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let Some((run_id, ticks)) = latest else {
        return Ok(None);
    };
    let last_tick = to_u64(ticks.unwrap_or(0), "simulation ticks")?;
    let budget: Option<Option<i64>> = db
        .query_row(
            "SELECT total_ticks FROM supervisor_progress WHERE run_id = ?1
             ORDER BY tick DESC LIMIT 1",
            params![run_id],
            |row| row.get(0),
        )
        .optional()?;
    let budget_ticks = budget
        .flatten()
        .map(|value| to_u64(value, "tick budget"))
        .transpose()?;
    Ok(Some(SupervisorRunSummary {
        run_id,
        last_tick,
        budget_ticks,
    }))
}

/// Loads the stored records of supervisor run `run_id` into a model.
///
/// # Errors
///
/// Returns [`BrowseError::NoRun`] when the run is not a supervisor run, and
/// other [`BrowseError`]s when a query fails or a stored value is malformed.
pub fn load_run(db_path: &str, run_id: i64) -> Result<ObservatoryModel, BrowseError> {
    let db = open_read_only(db_path)?;
    if !has_tables(&db)? {
        return Err(BrowseError::NoRun { run_id });
    }
    let snapshot = db.unchecked_transaction()?;
    let db: &Connection = &snapshot;
    let found: Option<i64> = db
        .query_row(
            "SELECT id FROM run WHERE id = ?1 AND mode = ?2",
            params![run_id, MODE_SUPERVISOR],
            |row| row.get(0),
        )
        .optional()?;
    if found.is_none() {
        return Err(BrowseError::NoRun { run_id });
    }
    let mut items = load_archive(db, run_id)?;
    items.extend(load_slot_occupants(db, run_id)?);
    let born: BTreeMap<u64, u64> = items
        .iter()
        .filter_map(|item| match &item.record {
            ObservatoryRecord::Occupant(o) => Some((o.occupant_id, o.born_tick)),
            _ => None,
        })
        .collect();
    items.extend(load_intake(db, run_id, &born)?);
    items.extend(load_slots(db, run_id)?);
    items.extend(load_boundaries(db, run_id)?);
    items.extend(load_series(db, run_id)?);
    items.extend(load_progress(db, run_id)?);
    items.extend(load_events(db, run_id)?);
    items.sort_by_key(|item| (item.tick, item.order));
    let mut model = ObservatoryModel::default();
    for item in items {
        let mut record = item.record;
        let resolved = item.parent.as_ref().and_then(|p| model.resolve_parent(p));
        match &mut record {
            ObservatoryRecord::Slot(row) => row.parent = resolved,
            ObservatoryRecord::Boundary(boundary) => boundary.parent = resolved,
            _ => {}
        }
        model.fold(record);
    }
    model.source = ModelSource::Browse { run_id };
    Ok(model)
}

fn parse_config(json: &str) -> Result<BrainConfig, BrowseError> {
    serde_json::from_str(json).map_err(|_| malformed("stored config is not valid"))
}

fn check_brain(blob: &[u8]) -> Result<(), BrowseError> {
    decode_brain_state(blob)
        .map(|_| ())
        .ok_or_else(|| malformed("stored brain cannot be decoded"))
}

fn load_archive(db: &Connection, run_id: i64) -> Result<Vec<Item>, BrowseError> {
    let mut stmt = db.prepare(
        "SELECT entry_id, occupant_id, born_tick, retired_tick, outcome, config_json, birth_brain
         FROM supervisor_archive WHERE run_id = ?1
         ORDER BY retired_tick DESC, entry_id DESC LIMIT ?2",
    )?;
    let mut rows = stmt.query(params![run_id, limit(MAX_ARCHIVE_ROWS)?])?;
    let mut items = Vec::new();
    while let Some(row) = rows.next()? {
        let occupant_id = to_u64(row.get("occupant_id")?, "archive occupant")?;
        let born_tick = to_u64(row.get("born_tick")?, "archive born tick")?;
        let retired_tick = to_u64(row.get("retired_tick")?, "archive retired tick")?;
        let outcome: f64 = row.get("outcome")?;
        let config = parse_config(&row.get::<_, String>("config_json")?)?;
        check_brain(&row.get::<_, Vec<u8>>("birth_brain")?)?;
        items.push(Item::new(
            retired_tick,
            ORDER_ARCHIVE,
            ObservatoryRecord::Archive(ArchiveRow {
                entry_id: to_u64(row.get("entry_id")?, "archive entry")?,
                occupant_id,
                outcome: narrow(outcome, "archive outcome")?,
                age_ticks: retired_tick.saturating_sub(born_tick),
                config,
            }),
        ));
        items.push(Item::new(
            born_tick,
            ORDER_OCCUPANT,
            ObservatoryRecord::Occupant(OccupantRecord {
                occupant_id,
                parent: None,
                born_tick,
                retired_tick: Some(retired_tick),
                intake: Vec::new(),
                late: None,
            }),
        ));
    }
    items.reverse();
    Ok(items)
}

fn load_slot_occupants(db: &Connection, run_id: i64) -> Result<Vec<Item>, BrowseError> {
    let mut stmt = db.prepare(
        "SELECT occupant_id, born_tick, config_json, birth_brain FROM supervisor_slot
         WHERE run_id = ?1 ORDER BY born_tick DESC, slot DESC LIMIT ?2",
    )?;
    let mut rows = stmt.query(params![run_id, limit(MAX_TRACKED_OCCUPANTS)?])?;
    let mut records = Vec::new();
    while let Some(row) = rows.next()? {
        parse_config(&row.get::<_, String>("config_json")?)?;
        check_brain(&row.get::<_, Vec<u8>>("birth_brain")?)?;
        let born_tick = to_u64(row.get("born_tick")?, "slot born tick")?;
        records.push(Item::new(
            born_tick,
            ORDER_OCCUPANT,
            ObservatoryRecord::Occupant(OccupantRecord {
                occupant_id: to_u64(row.get("occupant_id")?, "slot occupant")?,
                parent: None,
                born_tick,
                retired_tick: None,
                intake: Vec::new(),
                late: None,
            }),
        ));
    }
    records.reverse();
    Ok(records)
}

fn load_intake(
    db: &Connection,
    run_id: i64,
    born: &BTreeMap<u64, u64>,
) -> Result<Vec<Item>, BrowseError> {
    let mut stmt = db.prepare(
        "SELECT occupant_id, slot, age_ticks, meals FROM supervisor_occupant_intake
         WHERE run_id = ?1 ORDER BY age_ticks DESC, occupant_id DESC LIMIT ?2",
    )?;
    let mut rows = stmt.query(params![run_id, limit(MAX_INTAKE_ROWS)?])?;
    let mut records = Vec::new();
    while let Some(row) = rows.next()? {
        let intake = IntakeRow {
            occupant_id: to_u64(row.get("occupant_id")?, "intake occupant")?,
            slot: to_usize(row.get("slot")?, "intake slot")?,
            age_ticks: to_u64(row.get("age_ticks")?, "intake age")?,
            meals: to_u32(row.get("meals")?, "intake meals")?,
        };
        let tick = born
            .get(&intake.occupant_id)
            .map_or(intake.age_ticks, |b| b.saturating_add(intake.age_ticks));
        records.push(Item::new(
            tick,
            ORDER_INTAKE,
            ObservatoryRecord::from(&intake),
        ));
    }
    records.reverse();
    Ok(records)
}

fn parse_parent(
    kind: Option<String>,
    reference: Option<i64>,
) -> Result<Option<SourceParent>, BrowseError> {
    let Some(kind) = kind else {
        return Ok(None);
    };
    let reference = req(reference, "parent reference")?;
    Ok(Some(match kind.as_str() {
        PARENT_KIND_ARCHIVE => SourceParent::ArchiveEntry(to_u64(reference, "parent entry")?),
        PARENT_KIND_PEER_BIRTH => SourceParent::PeerBirth(to_usize(reference, "parent slot")?),
        PARENT_KIND_PEER_LEARNED => SourceParent::PeerLearned(to_usize(reference, "parent slot")?),
        _ => return Err(malformed("parent kind is unknown")),
    }))
}

fn load_slots(db: &Connection, run_id: i64) -> Result<Vec<Item>, BrowseError> {
    let mut stmt = db.prepare(
        "SELECT tick, slot, occupant_id, age_ticks, alive, meals, deaths, window_score, peer_median,
                parent_kind, parent_ref
         FROM supervisor_slot_snapshot
         WHERE run_id = ?1
           AND tick = (SELECT MAX(tick) FROM supervisor_slot_snapshot WHERE run_id = ?1)
         ORDER BY slot LIMIT ?2",
    )?;
    let mut rows = stmt.query(params![run_id, limit(MAX_SLOTS)?])?;
    let mut records = Vec::new();
    while let Some(row) = rows.next()? {
        let parent = parse_parent(row.get("parent_kind")?, row.get("parent_ref")?)?;
        let mut item = Item::new(
            to_u64(row.get("tick")?, "slot tick")?,
            ORDER_SLOT,
            ObservatoryRecord::Slot(SlotRow {
                slot: to_usize(row.get("slot")?, "slot")?,
                occupant_id: to_u64(row.get("occupant_id")?, "slot occupant")?,
                age_ticks: to_u64(row.get("age_ticks")?, "slot age")?,
                alive: row.get::<_, i64>("alive")? != 0,
                meals: to_u32(row.get("meals")?, "slot meals")?,
                deaths: to_u32(row.get("deaths")?, "slot deaths")?,
                window_score: opt_narrow(row.get("window_score")?, "window score")?,
                peer_median: opt_narrow(row.get("peer_median")?, "peer median")?,
                parent: None,
            }),
        );
        item.parent = parent;
        records.push(item);
    }
    Ok(records)
}

/// Eligible slots of each boundary tick at or after `from_tick`.
fn load_candidates(
    db: &Connection,
    run_id: i64,
    from_tick: i64,
) -> Result<BTreeMap<u64, Vec<EligibleSlot>>, BrowseError> {
    let mut stmt = db.prepare(
        "SELECT tick, slot, ground, score, median FROM supervisor_boundary_candidate
         WHERE run_id = ?1 AND tick >= ?2 AND tick IN
            (SELECT tick FROM supervisor_boundary WHERE run_id = ?1
             ORDER BY tick DESC LIMIT ?3)
         ORDER BY tick, slot LIMIT ?4",
    )?;
    let mut rows = stmt.query(params![
        run_id,
        from_tick,
        limit(MAX_BOUNDARY_RECORDS)?,
        limit(MAX_CANDIDATE_ROWS)?
    ])?;
    let mut by_tick: BTreeMap<u64, Vec<EligibleSlot>> = BTreeMap::new();
    while let Some(row) = rows.next()? {
        let ground: String = row.get("ground")?;
        let ground = match ground.as_str() {
            GROUND_DEATH => EligibilityGround::Died,
            GROUND_BELOW_MEDIAN => EligibilityGround::BelowMedian {
                score: narrow(
                    req(row.get("score")?, "candidate score")?,
                    "candidate score",
                )?,
                median: narrow(
                    req(row.get("median")?, "candidate median")?,
                    "candidate median",
                )?,
            },
            _ => return Err(malformed("candidate ground is unknown")),
        };
        by_tick
            .entry(to_u64(row.get("tick")?, "candidate tick")?)
            .or_default()
            .push(EligibleSlot {
                slot: to_usize(row.get("slot")?, "candidate slot")?,
                ground,
            });
    }
    Ok(by_tick)
}

fn parse_config_diff(json: Option<String>) -> Result<Vec<ConfigDelta>, BrowseError> {
    let Some(json) = json else {
        return Ok(Vec::new());
    };
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(&json).map_err(|_| malformed("config diff is not valid"))?;
    entries
        .iter()
        .map(|entry| {
            let field = entry
                .get("field")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| malformed("config diff field is missing"))?;
            let side = |key: &str| {
                entry
                    .get(key)
                    .map(|value| truncated(value.to_string()))
                    .ok_or_else(|| malformed("config diff value is missing"))
            };
            Ok(ConfigDelta {
                field: truncated(field.to_owned()),
                parent: side("parent")?,
                child: side("newborn")?,
            })
        })
        .collect()
}

fn load_boundaries(db: &Connection, run_id: i64) -> Result<Vec<Item>, BrowseError> {
    let mut stmt = db.prepare(
        "SELECT tick, outcome, slot, tiebreak_pool, tiebreak_index, parent_kind, parent_ref,
                config_diff, idle_too_young, idle_no_full_window, idle_no_peer_median,
                idle_at_or_above_median
         FROM supervisor_boundary WHERE run_id = ?1 ORDER BY tick DESC LIMIT ?2",
    )?;
    let mut rows = stmt.query(params![run_id, limit(MAX_BOUNDARY_RECORDS)?])?;
    let mut stored = Vec::new();
    let mut oldest: Option<i64> = None;
    while let Some(row) = rows.next()? {
        let tick: i64 = row.get("tick")?;
        oldest = Some(tick);
        let outcome: String = row.get("outcome")?;
        let config_diff = parse_config_diff(row.get("config_diff")?)?;
        let mut source_parent = None;
        let record = if outcome == EVENT_REPLACE {
            let pool: Vec<usize> = serde_json::from_str(&req(
                row.get::<_, Option<String>>("tiebreak_pool")?,
                "tie-break pool",
            )?)
            .map_err(|_| malformed("tie-break pool is not valid"))?;
            let drawn = to_usize(
                req(row.get("tiebreak_index")?, "tie-break index")?,
                "tie-break index",
            )?;
            source_parent = parse_parent(row.get("parent_kind")?, row.get("parent_ref")?)?;
            BoundaryRecord {
                tick: to_u64(tick, "boundary tick")?,
                eligible: Vec::new(),
                picked: Some(to_usize(
                    req(row.get("slot")?, "boundary slot")?,
                    "boundary slot",
                )?),
                tie_break: Some(truncated(format!("pool {pool:?}, drew index {drawn}"))),
                parent: None,
                config_diff,
                idle_reason: None,
            }
        } else if outcome == EVENT_IDLE {
            let count = |name: &str| -> Result<usize, BrowseError> {
                to_usize(req(row.get(name)?, name)?, name)
            };
            BoundaryRecord {
                tick: to_u64(tick, "boundary tick")?,
                eligible: Vec::new(),
                picked: None,
                tie_break: None,
                parent: None,
                config_diff,
                idle_reason: Some(truncated(format!(
                    "too young {}, no full window {}, no peer median {}, at or above median {}",
                    count("idle_too_young")?,
                    count("idle_no_full_window")?,
                    count("idle_no_peer_median")?,
                    count("idle_at_or_above_median")?
                ))),
            }
        } else {
            return Err(malformed("boundary outcome is unknown"));
        };
        stored.push((record, source_parent));
    }
    let Some(oldest) = oldest else {
        return Ok(Vec::new());
    };
    let mut candidates = load_candidates(db, run_id, oldest)?;
    let mut items = Vec::new();
    for (mut record, parent) in stored.into_iter().rev() {
        if record.picked.is_some() {
            record.eligible = candidates.remove(&record.tick).unwrap_or_default();
        }
        let mut item = Item::new(
            record.tick,
            ORDER_BOUNDARY,
            ObservatoryRecord::Boundary(record),
        );
        item.parent = parent;
        items.push(item);
    }
    Ok(items)
}

/// Retired lifespans of the boundaries at or after `from_tick`, by tick.
fn load_lifespans(
    db: &Connection,
    run_id: i64,
    from_tick: i64,
) -> Result<BTreeMap<u64, u64>, BrowseError> {
    let mut stmt = db.prepare(
        "SELECT tick, retired_lifespan_ticks FROM supervisor_boundary
         WHERE run_id = ?1 AND tick >= ?2 AND retired_lifespan_ticks IS NOT NULL
         ORDER BY tick DESC LIMIT ?3",
    )?;
    let mut rows = stmt.query(params![run_id, from_tick, limit(MAX_BOUNDARY_RECORDS)?])?;
    let mut lifespans = BTreeMap::new();
    while let Some(row) = rows.next()? {
        lifespans.insert(
            to_u64(row.get("tick")?, "boundary tick")?,
            to_u64(row.get("retired_lifespan_ticks")?, "retired lifespan")?,
        );
    }
    Ok(lifespans)
}

fn load_series(db: &Connection, run_id: i64) -> Result<Vec<Item>, BrowseError> {
    let mut stmt = db.prepare(
        "SELECT tick, metric, value FROM supervisor_series
         WHERE run_id = ?1 AND tick IN
            (SELECT DISTINCT tick FROM supervisor_series WHERE run_id = ?1
             ORDER BY tick DESC LIMIT ?2)
         ORDER BY tick",
    )?;
    let mut rows = stmt.query(params![run_id, limit(MAX_SERIES_POINTS)?])?;
    let mut by_tick: BTreeMap<u64, Vec<(String, Option<f32>)>> = BTreeMap::new();
    let mut oldest: Option<i64> = None;
    while let Some(row) = rows.next()? {
        let tick: i64 = row.get("tick")?;
        if oldest.is_none() {
            oldest = Some(tick);
        }
        let name: String = row.get("metric")?;
        if !KNOWN_METRICS.iter().any(|m| m.as_str() == name) {
            return Err(malformed("series metric is unknown"));
        }
        let value = opt_narrow(row.get("value")?, "series value")?;
        by_tick
            .entry(to_u64(tick, "series tick")?)
            .or_default()
            .push((name, value));
    }
    let Some(oldest) = oldest else {
        return Ok(Vec::new());
    };
    let lifespans = load_lifespans(db, run_id, oldest)?;
    let mut items = Vec::new();
    for (tick, values) in by_tick {
        let value = |metric: Metric| {
            values
                .iter()
                .find(|(name, _)| name == metric.as_str())
                .and_then(|(_, value)| *value)
        };
        items.push(Item::new(
            tick,
            ORDER_SERIES,
            ObservatoryRecord::Series(SeriesPoint {
                tick,
                food_per_agent_tick: value(Metric::FoodPerAgentTick).unwrap_or_default(),
                deaths_per_agent_tick: value(Metric::DeathsPerAgentTick).unwrap_or_default(),
                score_spread: value(Metric::WindowScoreSpread),
                retired_lifespan: lifespans.get(&tick).copied(),
                archive_best: value(Metric::ArchiveBestOutcome),
                archive_median: value(Metric::ArchiveMedianOutcome),
            }),
        ));
    }
    Ok(items)
}

fn load_progress(db: &Connection, run_id: i64) -> Result<Vec<Item>, BrowseError> {
    let mut stmt = db.prepare(
        "SELECT tick, total_ticks, agent_tick_budget, ticks_to_boundary, nourishing_kind,
                next_flip_tick, ticks_per_second
         FROM supervisor_progress WHERE run_id = ?1 ORDER BY tick DESC LIMIT 1",
    )?;
    let mut rows = stmt.query(params![run_id])?;
    let Some(row) = rows.next()? else {
        return Ok(Vec::new());
    };
    let nourishing: Option<String> = row.get("nourishing_kind")?;
    let nourishing: Option<FoodKind> = nourishing
        .map(|json| serde_json::from_str(&json))
        .transpose()
        .map_err(|_| malformed("nourishing kind is not valid"))?;
    let total: Option<i64> = row.get("total_ticks")?;
    let flip: Option<i64> = row.get("next_flip_tick")?;
    let record = ProgressRecord {
        tick: to_u64(row.get("tick")?, "progress tick")?,
        total_ticks: total.map(|v| to_u64(v, "total ticks")).transpose()?,
        agent_tick_budget: to_u64(row.get("agent_tick_budget")?, "agent tick budget")?,
        ticks_to_boundary: to_u64(row.get("ticks_to_boundary")?, "ticks to boundary")?,
        nourishing,
        next_flip_tick: flip.map(|v| to_u64(v, "next flip tick")).transpose()?,
        ticks_per_second: opt_finite(row.get("ticks_per_second")?, "ticks per second")?,
    };
    let mut items = Vec::new();
    if let Some(season) = SeasonState::from_progress(&record) {
        items.push(Item::new(
            record.tick,
            ORDER_SEASON,
            ObservatoryRecord::Season(season),
        ));
    }
    items.push(Item::new(
        record.tick,
        ORDER_PROGRESS,
        ObservatoryRecord::from(&record),
    ));
    Ok(items)
}

fn event_kind(kind: &str) -> TimelineKind {
    match kind {
        EVENT_REPLACE => TimelineKind::Replacement,
        EVENT_IDLE => TimelineKind::Idle,
        EVENT_SEASON => TimelineKind::Season,
        EVENT_ARCHIVE | EVENT_EVICT => TimelineKind::Archive,
        _ => TimelineKind::Run,
    }
}

fn load_events(db: &Connection, run_id: i64) -> Result<Vec<Item>, BrowseError> {
    let mut stmt = db.prepare(
        "SELECT tick, kind, slot, reason, occupant_id, outcome, nourishing_kind
         FROM supervisor_event WHERE run_id = ?1 ORDER BY tick DESC, id DESC LIMIT ?2",
    )?;
    let mut rows = stmt.query(params![run_id, limit(MAX_TIMELINE_EVENTS)?])?;
    let mut events = Vec::new();
    while let Some(row) = rows.next()? {
        let kind: String = row.get("kind")?;
        let mut parts = vec![kind.clone()];
        if let Some(slot) = row.get::<_, Option<i64>>("slot")? {
            parts.push(format!("slot {}", to_usize(slot, "event slot")?));
        }
        if let Some(reason) = row.get::<_, Option<String>>("reason")? {
            parts.push(format!("reason {reason}"));
        }
        if let Some(occupant) = row.get::<_, Option<i64>>("occupant_id")? {
            parts.push(format!("occupant {}", to_u64(occupant, "event occupant")?));
        }
        if let Some(outcome) = opt_finite(row.get("outcome")?, "event outcome")? {
            parts.push(format!("outcome {outcome}"));
        }
        if let Some(nourishing) = row.get::<_, Option<String>>("nourishing_kind")? {
            parts.push(format!("nourishing {nourishing}"));
        }
        let tick = to_u64(row.get("tick")?, "event tick")?;
        events.push(Item::new(
            tick,
            ORDER_EVENT,
            ObservatoryRecord::Event(TimelineEvent {
                tick,
                kind: event_kind(&kind),
                text: truncated(parts.join(" ")),
            }),
        ));
    }
    events.reverse();
    Ok(events)
}
