//! Read-only browsing of a stored supervisor run.

use rusqlite::Connection;
use xagent_sandbox::observatory::store_read::BrowseError;
use xagent_sandbox::observatory::store_read::{detect_supervisor_run, load_run, open_read_only};
use xagent_sandbox::observatory::{
    ModelSource, ObservatoryModel, ObservatoryRecord, SeasonState, SlotRow,
};
use xagent_sandbox::supervisor::{BoundaryDetail, EligibilityGround, IdleReason, SlotStanding};
use xagent_sandbox::supervisor_store::SupervisorStore;
use xagent_sandbox::supervisor_telemetry::{
    BoundaryRecord as SourceBoundary, IntakeRow, Metric, ProgressRecord,
    SeriesPoint as SourcePoint, TelemetryFrame,
};
use xagent_shared::{FoodKind, FullConfig};

/// Tick of the first stored frame.
const FIRST_TICK: u64 = 10;
/// Tick of the second stored frame, which holds the idle boundary.
const SECOND_TICK: u64 = 20;
/// Cumulative simulation ticks recorded for the run.
const RUN_TICKS: u64 = 25;
/// Tick budget of the run.
const TOTAL_TICKS: u64 = 1_000;
/// Per-agent tick budget of the run.
const AGENT_BUDGET: u64 = 100;
/// Ticks from the second frame to the next boundary.
const TICKS_TO_BOUNDARY: u64 = 5;
/// Tick of the next season flip.
const FLIP_TICK: u64 = 50;
/// Observed speed stored with the progress row.
const SPEED: f64 = 5.0;
/// Window score of the first slot.
const SCORE: f32 = 0.5;
/// Peer median of the first slot.
const MEDIAN: f32 = 0.25;
/// Food series value.
const FOOD: f64 = 0.5;
/// Death series value.
const DEATHS: f64 = 0.25;
/// Number of slots in each frame.
const SLOT_COUNT: usize = 2;
/// Slots counted as too young in the idle boundary.
const TOO_YOUNG_COUNT: usize = 2;
/// Run id that no database holds.
const RUN_ID_ABSENT: i64 = 1;
/// Run id of the fixture's only run.
const FIXTURE_RUN_ID: i64 = 1;
/// Fold rank of slot rows within a tick.
const RANK_SLOT: u8 = 3;
/// Fold rank of boundaries within a tick.
const RANK_BOUNDARY: u8 = 4;
/// Fold rank of series points within a tick.
const RANK_SERIES: u8 = 5;
/// Fold rank of season state within a tick.
const RANK_SEASON: u8 = 6;
/// Fold rank of progress within a tick.
const RANK_PROGRESS: u8 = 7;
/// Fold rank of timeline events within a tick.
const RANK_EVENT: u8 = 8;

fn standing(slot: usize) -> SlotStanding {
    SlotStanding {
        slot,
        occupant_id: u64::try_from(slot).expect("slot fits"),
        age_ticks: SECOND_TICK,
        alive: true,
        meals: 3,
        deaths: 1,
        window_score: Some(SCORE),
        peer_median: Some(MEDIAN),
        ground: EligibilityGround::TooYoung,
        parent: None,
    }
}

fn progress(tick: u64) -> ProgressRecord {
    ProgressRecord {
        tick,
        total_ticks: Some(TOTAL_TICKS),
        agent_tick_budget: AGENT_BUDGET,
        ticks_to_boundary: TICKS_TO_BOUNDARY,
        nourishing: Some(FoodKind::First),
        next_flip_tick: Some(FLIP_TICK),
        ticks_per_second: Some(SPEED),
    }
}

fn idle_boundary() -> SourceBoundary {
    SourceBoundary {
        detail: BoundaryDetail::Idle {
            tick: SECOND_TICK,
            standings: Vec::new(),
            reason: IdleReason {
                too_young: TOO_YOUNG_COUNT,
                no_full_window: 0,
                no_peer_median: 0,
                at_or_above_median: 0,
            },
        },
        newborn_occupant: None,
        config_diff: Vec::new(),
        retired_lifespan_ticks: None,
    }
}

fn frame(tick: u64, boundary: Option<SourceBoundary>) -> TelemetryFrame {
    let point = |metric, value| SourcePoint {
        tick,
        metric,
        value,
    };
    TelemetryFrame {
        tick,
        slots: (0..SLOT_COUNT).map(standing).collect(),
        boundary,
        series: vec![
            point(Metric::FoodPerAgentTick, FOOD),
            point(Metric::DeathsPerAgentTick, DEATHS),
        ],
        intake: vec![IntakeRow {
            occupant_id: 0,
            slot: 0,
            age_ticks: tick,
            meals: 1,
        }],
        progress: progress(tick),
        persist: true,
    }
}

fn frames() -> Vec<TelemetryFrame> {
    vec![
        frame(FIRST_TICK, None),
        frame(SECOND_TICK, Some(idle_boundary())),
    ]
}

fn season_text() -> String {
    format!(
        "season nourishing {}",
        serde_json::to_string(&FoodKind::First).expect("kind serialises")
    )
}

/// Builds a database with one supervisor run and returns its path.
fn fixture(dir: &tempfile::TempDir) -> String {
    let path = dir.path().join("run.db").to_string_lossy().into_owned();
    let mut store = SupervisorStore::create(&path, &FullConfig::default()).expect("create");
    for frame in frames() {
        store.record_telemetry(&frame).expect("telemetry");
    }
    store
        .record_season(FIRST_TICK, FoodKind::First)
        .expect("season");
    store.record_idle_boundary(SECOND_TICK).expect("idle");
    store.record_ticks(RUN_TICKS).expect("ticks");
    drop(store);
    let db = Connection::open(&path).expect("reopen");
    db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
        .expect("checkpoint");
    drop(db);
    path
}

#[test]
fn observatory_browse_file_byte_identical() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir);
    let before = std::fs::read(&path).expect("read before");
    let summary = detect_supervisor_run(&path)
        .expect("detect")
        .expect("run found");
    load_run(&path, summary.run_id).expect("load");
    let after = std::fs::read(&path).expect("read after");
    assert_eq!(before, after);
}

#[test]
fn observatory_browse_missing_path_not_created() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("absent.db").to_string_lossy().into_owned();
    assert_eq!(detect_supervisor_run(&path).expect("detect"), None);
    assert!(load_run(&path, RUN_ID_ABSENT).is_err());
    assert!(!std::path::Path::new(&path).exists());
}

#[test]
fn observatory_browse_refuses_insert() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir);
    let summary = detect_supervisor_run(&path)
        .expect("detect")
        .expect("run found");
    let db = open_read_only(&path).expect("open");
    let result = db.execute(
        "INSERT INTO supervisor_event (run_id, tick, kind) VALUES (?1, ?2, ?3)",
        rusqlite::params![summary.run_id, FIRST_TICK, "x"],
    );
    assert!(result.is_err());
}

/// Rewrites one stored series row through a writable connection.
fn corrupt_series(path: &str, assignment: &str) {
    let db = Connection::open(path).expect("open writable");
    db.execute_batch(&format!(
        "UPDATE supervisor_series SET {assignment} WHERE rowid = (SELECT MIN(rowid) FROM supervisor_series);"
    ))
    .expect("corrupt");
}

#[test]
fn observatory_browse_unknown_metric_is_malformed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir);
    corrupt_series(&path, "metric = 'unknown_metric'");
    let err = load_run(&path, FIXTURE_RUN_ID).expect_err("malformed");
    assert!(matches!(err, BrowseError::Malformed { .. }));
}

#[test]
fn observatory_browse_overflowing_float_is_malformed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir);
    corrupt_series(&path, "value = 1e300");
    let err = load_run(&path, FIXTURE_RUN_ID).expect_err("malformed");
    assert!(matches!(err, BrowseError::Malformed { .. }));
}

#[test]
fn observatory_browse_matches_live_fold() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir);
    let summary = detect_supervisor_run(&path)
        .expect("detect")
        .expect("run found");
    assert_eq!(summary.last_tick, RUN_TICKS);
    assert_eq!(summary.budget_ticks, Some(TOTAL_TICKS));
    let mut browsed = load_run(&path, summary.run_id).expect("load");
    assert_eq!(
        browsed.source,
        ModelSource::Browse {
            run_id: summary.run_id
        }
    );

    let all = frames();
    let last = all.last().expect("frames");
    let mut records: Vec<(u64, u8, ObservatoryRecord)> = Vec::new();
    for standing in &last.slots {
        records.push((
            SECOND_TICK,
            RANK_SLOT,
            ObservatoryRecord::Slot(SlotRow {
                slot: standing.slot,
                occupant_id: standing.occupant_id,
                age_ticks: standing.age_ticks,
                alive: standing.alive,
                meals: standing.meals,
                deaths: standing.deaths,
                window_score: standing.window_score,
                peer_median: standing.peer_median,
                parent: None,
            }),
        ));
    }
    for frame in &all {
        if let Some(boundary) = &frame.boundary {
            records.push((
                SECOND_TICK,
                RANK_BOUNDARY,
                ObservatoryRecord::Boundary(xagent_sandbox::observatory::BoundaryRecord::from(
                    boundary,
                )),
            ));
        }
        records.push((
            frame.tick,
            RANK_SERIES,
            ObservatoryRecord::series_from_frame(frame).expect("series"),
        ));
    }
    let record = progress(SECOND_TICK);
    records.push((
        SECOND_TICK,
        RANK_SEASON,
        ObservatoryRecord::Season(SeasonState::from_progress(&record).expect("season")),
    ));
    records.push((SECOND_TICK, RANK_PROGRESS, ObservatoryRecord::from(&record)));
    for (tick, kind, text) in [
        (
            FIRST_TICK,
            xagent_sandbox::observatory::TimelineKind::Season,
            season_text(),
        ),
        (
            SECOND_TICK,
            xagent_sandbox::observatory::TimelineKind::Idle,
            "idle".to_owned(),
        ),
    ] {
        records.push((
            tick,
            RANK_EVENT,
            ObservatoryRecord::Event(xagent_sandbox::observatory::TimelineEvent {
                tick,
                kind,
                text,
            }),
        ));
    }
    records.sort_by_key(|(tick, rank, _)| (*tick, *rank));
    let mut live = ObservatoryModel::default();
    for (_, _, record) in records {
        live.fold(record);
    }

    live.source = ModelSource::Live;
    live.progress.ticks_per_sec = None;
    browsed.source = ModelSource::Live;
    browsed.progress.ticks_per_sec = None;
    assert_eq!(browsed, live);
}

#[test]
fn observatory_browse_no_tables_is_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let empty = dir.path().join("empty.db");
    std::fs::write(&empty, b"").expect("write empty");
    assert_eq!(
        detect_supervisor_run(&empty.to_string_lossy()).expect("detect empty"),
        None
    );

    let bare = dir.path().join("bare.db");
    let db = Connection::open(&bare).expect("open bare");
    db.execute_batch("CREATE TABLE run (id INTEGER PRIMARY KEY, mode TEXT);")
        .expect("create run");
    drop(db);
    assert_eq!(
        detect_supervisor_run(&bare.to_string_lossy()).expect("detect bare"),
        None
    );
}
