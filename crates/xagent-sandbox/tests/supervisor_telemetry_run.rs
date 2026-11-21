//! Telemetry recording in the supervisor driver: decisions are identical with
//! recording on and off, rows are consistent, frames repeat, and a failing
//! telemetry write never changes the run.

use rusqlite::types::Value;
use rusqlite::Connection;
use xagent_brain::buffers::AgentBrainState;
use xagent_brain::GpuKernel;
use xagent_sandbox::headless::{run_supervisor_with_recording, HeadlessOptions};
use xagent_sandbox::supervisor::{stream_seed, TickSchedule, SEASON_STREAM};
use xagent_sandbox::supervisor_run::{SampleReport, SupervisorRun};
use xagent_sandbox::supervisor_store::SupervisorStore;
use xagent_sandbox::supervisor_telemetry::{TelemetryFrame, TELEMETRY_CADENCE_SAMPLES};
use xagent_sandbox::world::WorldState;
use xagent_shared::{FullConfig, Inheritance, SupervisorConfig, SUPERVISOR_SAMPLE_TICKS};

/// Agents in every run.
const AGENTS: usize = 6;
/// Boundary interval in ticks.
const BOUNDARY_TICKS: u64 = 1_000;
/// Minimum age before an underperformer can be replaced.
const MIN_AGE_TICKS: u64 = 2_000;
/// Score window in ticks.
const WINDOW_TICKS: u64 = 1_000;
/// Season period in ticks.
const SEASON_TICKS: u64 = 3_000;
/// Ticks each agent lives in a run.
const RUN_TICKS: u64 = 20_000;
/// Agent-tick budget of a run.
const BUDGET: u64 = AGENTS as u64 * RUN_TICKS;
/// Archive capacity of a run.
const ARCHIVE: usize = 8;
/// World seed of a run.
const SEED: u64 = 5;
/// Rate fed to the run in the speed test.
const RATE: f64 = 1000.0;

/// The telemetry tables.
const TELEMETRY_TABLES: [&str; 6] = [
    "supervisor_slot_snapshot",
    "supervisor_boundary",
    "supervisor_boundary_candidate",
    "supervisor_series",
    "supervisor_occupant_intake",
    "supervisor_progress",
];

/// What a guard does for an adapter state and the `XAGENT_REQUIRE_GPU` value.
#[derive(Debug, PartialEq, Eq)]
enum GuardDecision {
    Run,
    Skip,
    Fail,
}

/// Pure guard rule: fail closed when the GPU is required but absent.
fn gpu_guard_decision(available: bool, required: Option<&str>) -> GuardDecision {
    if available {
        GuardDecision::Run
    } else if required == Some("1") {
        GuardDecision::Fail
    } else {
        GuardDecision::Skip
    }
}

/// Panics without an adapter when `XAGENT_REQUIRE_GPU=1`; otherwise reports
/// whether to run.
fn gpu_or_skip() -> bool {
    let required = std::env::var("XAGENT_REQUIRE_GPU").ok();
    match gpu_guard_decision(GpuKernel::is_available(), required.as_deref()) {
        GuardDecision::Run => true,
        GuardDecision::Skip => {
            println!("skipping: no GPU adapter");
            false
        }
        GuardDecision::Fail => panic!("XAGENT_REQUIRE_GPU=1 but no GPU adapter is available"),
    }
}

#[test]
fn gpu_guard_decision_covers_every_combination() {
    assert_eq!(gpu_guard_decision(true, None), GuardDecision::Run);
    assert_eq!(gpu_guard_decision(true, Some("1")), GuardDecision::Run);
    assert_eq!(gpu_guard_decision(false, None), GuardDecision::Skip);
    assert_eq!(gpu_guard_decision(false, Some("1")), GuardDecision::Fail);
}

fn config(kinds: bool) -> FullConfig {
    let mut config = FullConfig::default();
    config.governor.population_size = AGENTS;
    config.governor.supervisor = SupervisorConfig {
        boundary_interval_ticks: BOUNDARY_TICKS,
        min_age_ticks: MIN_AGE_TICKS,
        score_window_ticks: WINDOW_TICKS,
        archive_capacity: ARCHIVE,
        season_period_ticks: SEASON_TICKS,
        inheritance: Inheritance::Darwinian,
        agent_tick_budget: BUDGET,
    };
    config.world.food_kinds_enabled = kinds;
    config.world.seed = SEED;
    config
}

fn kernel_for(config: &FullConfig) -> GpuKernel {
    let world = WorldState::new(config.world.clone());
    GpuKernel::new(
        u32::try_from(AGENTS).expect("agent count fits"),
        world.food_items.len(),
        &config.brain,
        &config.world,
    )
}

/// A temporary database path.
fn temp_db(dir: &tempfile::TempDir, name: &str) -> String {
    dir.path()
        .join(name)
        .to_str()
        .expect("utf-8 path")
        .to_owned()
}

/// All rows of `table` without the autoincrement `id`, in insertion order.
fn rows(db: &Connection, table: &str) -> Vec<Vec<Value>> {
    let mut columns = db
        .prepare(&format!("SELECT * FROM {table}"))
        .expect("prepare")
        .column_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let has_id = columns.iter().any(|c| c == "id");
    columns.retain(|c| c != "id");
    let order = if has_id { "id" } else { "rowid" };
    let sql = format!(
        "SELECT {} FROM {table} ORDER BY {order}",
        columns.join(", ")
    );
    let mut stmt = db.prepare(&sql).expect("prepare rows");
    let count = columns.len();
    stmt.query_map([], |row| {
        (0..count).map(|i| row.get::<_, Value>(i)).collect()
    })
    .expect("query")
    .map(|r| r.expect("row"))
    .collect()
}

fn count(db: &Connection, table: &str) -> i64 {
    db.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
        row.get(0)
    })
    .expect("count")
}

/// Runs the headless supervisor into a fresh database and opens it.
fn headless(config: &FullConfig, dir: &tempfile::TempDir, name: &str, record: bool) -> Connection {
    let path = temp_db(dir, name);
    run_supervisor_with_recording(config.clone(), &path, &HeadlessOptions::default(), record)
        .expect("supervisor run");
    Connection::open(&path).expect("open")
}

/// Ticks in `sql`'s first column.
fn ticks(db: &Connection, sql: &str) -> Vec<u64> {
    let mut stmt = db.prepare(sql).expect("prepare");
    let found: Vec<i64> = stmt
        .query_map([], |row| row.get(0))
        .expect("query")
        .map(|r| r.expect("row"))
        .collect();
    found
        .into_iter()
        .map(|t| u64::try_from(t).expect("tick"))
        .collect()
}

#[test]
fn recording_on_off_same_decisions() {
    if !gpu_or_skip() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let config = config(false);
    let on = headless(&config, &dir, "on.db", true);
    let off = headless(&config, &dir, "off.db", false);
    for table in ["supervisor_slot", "supervisor_archive", "supervisor_event"] {
        let recorded = rows(&on, table);
        assert!(!recorded.is_empty(), "{table} has rows");
        assert_eq!(recorded, rows(&off, table), "{table}");
    }
    assert!(
        count(&on, "supervisor_event") > 0,
        "the run recorded events"
    );
    for table in TELEMETRY_TABLES {
        assert_eq!(count(&off, table), 0, "{table} is empty when off");
    }
    assert!(count(&on, "supervisor_series") > 0);
}

/// Checks the row relations of one recorded database.
fn check_rows(db: &Connection, agents: i64) {
    let mut boundary_keys: Vec<(i64, i64, i64)> = db
        .prepare("SELECT tick, slot, newborn_occupant FROM supervisor_boundary WHERE outcome = 'replace'")
        .expect("prepare")
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .expect("query")
        .map(|r| r.expect("row"))
        .collect();
    let mut event_keys: Vec<(i64, i64, i64)> = db
        .prepare("SELECT tick, slot, occupant_id FROM supervisor_event WHERE kind = 'replace'")
        .expect("prepare")
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .expect("query")
        .map(|r| r.expect("row"))
        .collect();
    assert!(!boundary_keys.is_empty(), "the run replaced slots");
    boundary_keys.sort_unstable();
    event_keys.sort_unstable();
    assert_eq!(boundary_keys, event_keys);

    let bad: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM supervisor_boundary_candidate \
             WHERE ground = 'below_median' AND NOT (score < median)",
            [],
            |r| r.get(0),
        )
        .expect("count");
    assert_eq!(bad, 0, "below-median candidates sit under the median");

    let bad_idle: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM supervisor_boundary WHERE outcome = 'idle' AND \
             idle_too_young + idle_no_full_window + idle_no_peer_median \
             + idle_at_or_above_median != ?1",
            [agents],
            |r| r.get(0),
        )
        .expect("count");
    assert_eq!(bad_idle, 0, "idle reason counts sum to the agent count");

    let cadence = SUPERVISOR_SAMPLE_TICKS * TELEMETRY_CADENCE_SAMPLES;
    let series = ticks(
        db,
        "SELECT tick FROM supervisor_series WHERE metric = 'food_per_agent_tick'",
    );
    let last = ticks(db, "SELECT MAX(tick) FROM supervisor_progress")
        .pop()
        .expect("progress rows");
    let mut tick = cadence;
    while tick <= last {
        assert!(series.contains(&tick), "food series at tick {tick}");
        tick += cadence;
    }

    let bad_progress: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM supervisor_progress WHERE ticks_to_boundary <= 0",
            [],
            |r| r.get(0),
        )
        .expect("count");
    assert_eq!(bad_progress, 0, "every boundary lies ahead");
}

#[test]
fn recording_writes_consistent_rows() {
    if !gpu_or_skip() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let agents = i64::try_from(AGENTS).expect("fits");
    check_rows(&headless(&config(false), &dir, "plain.db", true), agents);

    let seasonal = config(true);
    let db = headless(&seasonal, &dir, "seasons.db", true);
    check_rows(&db, agents);
    let mut schedule = TickSchedule::new(stream_seed(SEED, SEASON_STREAM), SEASON_TICKS);
    let mut season_ticks = Vec::new();
    while schedule.next_tick() <= RUN_TICKS {
        season_ticks.push(schedule.next_tick());
        schedule.advance();
    }
    let mut late = ticks(
        &db,
        "SELECT tick FROM supervisor_series WHERE metric LIKE 'late_%'",
    );
    late.sort_unstable();
    late.dedup();
    assert!(!late.is_empty(), "late-window rows exist");
    assert_eq!(
        late, season_ticks,
        "late rows appear exactly at season ticks"
    );
}

struct Driven {
    reports: Vec<SampleReport>,
    births: Vec<Vec<u32>>,
    frames: Vec<Option<TelemetryFrame>>,
}

fn bits(state: &AgentBrainState) -> Vec<u32> {
    state
        .brain_state
        .iter()
        .chain(&state.patterns)
        .map(|v| v.to_bits())
        .collect()
}

/// Drives a run to its end, taking the frame after every sample.
fn drive(
    config: &FullConfig,
    store: Option<SupervisorStore>,
    rate: Option<f64>,
    after_start: impl FnOnce(),
) -> Driven {
    let mut kernel = kernel_for(config);
    let mut run = SupervisorRun::start(&mut kernel, config, store, false).expect("run starts");
    after_start();
    if let Some(rate) = rate {
        run.note_ticks_per_second(rate);
    }
    let mut reports = Vec::new();
    let mut frames = Vec::new();
    let mut tick = 0;
    while !run.is_finished() {
        kernel.dispatch_batch(
            tick,
            u32::try_from(SUPERVISOR_SAMPLE_TICKS).expect("sample fits"),
        );
        while !kernel.try_collect_state_snapshot() {
            std::hint::spin_loop();
        }
        let physics = kernel.cached_state().to_vec();
        let food = kernel.cached_food_state().map(<[f32]>::to_vec);
        tick += SUPERVISOR_SAMPLE_TICKS;
        reports.push(
            run.on_sample(&mut kernel, tick, &physics, food.as_deref())
                .expect("sample is handled"),
        );
        frames.push(run.take_frame());
    }
    let births = (0..AGENTS)
        .map(|slot| bits(run.birth_state(slot).expect("slot exists")))
        .collect();
    Driven {
        reports,
        births,
        frames,
    }
}

#[test]
fn take_frame_is_some_after_every_sample_and_repeats() {
    if !gpu_or_skip() {
        return;
    }
    let config = config(true);
    let first = drive(&config, None, None, || {});
    let second = drive(&config, None, None, || {});
    let rated = drive(&config, None, Some(RATE), || {});
    assert!(first.frames.iter().all(Option::is_some));
    assert!(first
        .frames
        .iter()
        .flatten()
        .all(|f| f.progress.ticks_per_second.is_none()));
    assert_eq!(first.frames, second.frames);
    assert_eq!(first.frames.len(), rated.frames.len());
    for (plain, with_rate) in first.frames.iter().zip(&rated.frames) {
        let mut expected = plain.clone().expect("frame");
        let actual = with_rate.clone().expect("frame");
        assert_eq!(actual.progress.ticks_per_second, Some(RATE));
        expected.progress.ticks_per_second = Some(RATE);
        assert_eq!(actual, expected);
    }
}

#[test]
fn telemetry_write_failure_keeps_decisions() {
    if !gpu_or_skip() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let config = config(false);
    let path = temp_db(&dir, "failing.db");
    let store = SupervisorStore::create(&path, &config).expect("store opens");
    let failed = drive(&config, Some(store), None, || {
        let other = Connection::open(&path).expect("second connection");
        other
            .execute_batch("DROP TABLE supervisor_series")
            .expect("table dropped");
    });
    let plain = drive(&config, None, None, || {});
    assert_eq!(failed.reports, plain.reports);
    assert_eq!(failed.births, plain.births);
    assert!(failed.frames.iter().all(Option::is_some));
    assert_eq!(failed.frames, plain.frames);
}
