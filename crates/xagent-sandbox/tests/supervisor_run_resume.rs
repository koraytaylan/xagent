//! Checkpoint and resume of a supervisor run: the resumed run equals the
//! uninterrupted one in reports and store rows.

use rusqlite::types::Value;
use rusqlite::Connection;
use xagent_brain::buffers::{
    FOOD_POSITION_X, FOOD_POSITION_Z, FOOD_STATE_STRIDE, PHYS_STRIDE, P_ALIVE, P_FACING_Z,
    P_FOOD_COUNT, P_MOTOR_TURN_OUT,
};
use xagent_brain::GpuKernel;
use xagent_sandbox::run_summary::RunSummaryAccumulator;
use xagent_sandbox::supervisor_checkpoint::CheckpointError;
use xagent_sandbox::supervisor_run::{
    load_resume_point, SampleReport, SupervisorRun, SupervisorRunError,
};
use xagent_sandbox::supervisor_store::SupervisorStore;
use xagent_sandbox::world::WorldState;
use xagent_shared::{FoodKind, FullConfig, Inheritance, SupervisorConfig, SUPERVISOR_SAMPLE_TICKS};

/// Agents in every run.
const AGENTS: usize = 10;
/// Boundary interval in ticks.
const BOUNDARY_TICKS: u64 = 1_000;
/// Minimum age before an underperformer can be replaced.
const MIN_AGE_TICKS: u64 = 2_000;
/// Score window in ticks.
const WINDOW_TICKS: u64 = 1_000;
/// Season period in ticks.
const SEASON_TICKS: u64 = 3_000;
/// Agent-tick budget of a run.
const BUDGET: u64 = 10 * 20_000;
/// Archive capacity of a run.
const ARCHIVE: usize = 8;
/// World seed of a run.
const SEED: u64 = 5;
/// Checkpoint write after which the interrupted run stops.
const STOP_AFTER_CHECKPOINTS: usize = 2;
/// Agents of the accumulator round-trip test.
const ACCUMULATOR_AGENTS: usize = 2;
/// Samples before the accumulator is captured.
const SAMPLES_BEFORE_CAPTURE: u8 = 5;
/// Samples after the accumulator is captured.
const SAMPLES_AFTER_CAPTURE: u8 = 10;
/// Turn output that is not exactly representable in binary.
const INEXACT_TURN: f32 = 0.1;

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

fn supervisor_config() -> FullConfig {
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
    config.world.food_kinds_enabled = true;
    config.world.seed = SEED;
    config
}

fn new_kernel(config: &FullConfig) -> GpuKernel {
    let food_count = WorldState::new(config.world.clone()).food_items.len();
    let agents = u32::try_from(config.governor.population_size).expect("population fits u32");
    GpuKernel::new(agents, food_count, &config.brain, &config.world)
}

fn batch_ticks() -> u32 {
    u32::try_from(SUPERVISOR_SAMPLE_TICKS).expect("sample period fits u32")
}

/// Drives the run like the headless loop. Returns the reports and whether
/// the run stopped after `stop_after` checkpoint writes, before `on_sample`.
fn drive(
    run: &mut SupervisorRun,
    kernel: &mut GpuKernel,
    stop_after: Option<usize>,
) -> (Vec<SampleReport>, bool) {
    let mut reports = Vec::new();
    let mut written = 0;
    while !run.is_finished() {
        let tick = run.next_sample_tick();
        let (physics, food) = if run.checkpoint_due(tick) {
            let _ = kernel.dispatch_ticks(run.tick(), batch_ticks());
            let checkpoint = kernel.read_checkpoint_blocking();
            run.write_checkpoint(tick, &checkpoint)
                .expect("write checkpoint");
            written += 1;
            if stop_after == Some(written) {
                return (reports, true);
            }
            (
                checkpoint.physics().to_vec(),
                checkpoint.food().map(<[f32]>::to_vec),
            )
        } else {
            let _ = kernel.dispatch_batch(run.tick(), batch_ticks());
            while !kernel.try_collect_state_snapshot() {
                std::thread::yield_now();
            }
            (
                kernel.cached_state().to_vec(),
                kernel.cached_food_state().map(<[f32]>::to_vec),
            )
        };
        reports.push(
            run.on_sample(kernel, tick, &physics, food.as_deref())
                .expect("sample"),
        );
    }
    (reports, false)
}

/// Every table as `(name, rows by rowid)`, each row led by its rowid.
fn all_rows(db: &Connection) -> Vec<(String, Vec<Vec<Value>>)> {
    let names: Vec<String> = db
        .prepare(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
             ORDER BY name",
        )
        .expect("prepare names")
        .query_map([], |row| row.get(0))
        .expect("names")
        .map(|r| r.expect("name"))
        .collect();
    names
        .into_iter()
        .map(|name| {
            // Resume rewrites every slot row with `INSERT OR REPLACE`, which
            // hands each a fresh rowid, so that table is compared by its (run_id, slot) key.
            let by_key = name == "supervisor_slot";
            let order = if by_key { "run_id, slot" } else { "rowid" };
            let mut stmt = db
                .prepare(&format!("SELECT rowid, * FROM {name} ORDER BY {order}"))
                .expect("prepare rows");
            let count = stmt.column_count();
            // The run row's wall-clock start time is the one value that
            // legitimately differs between two runs.
            let wall_clock: Vec<bool> = stmt
                .column_names()
                .iter()
                .enumerate()
                .map(|(i, c)| (name == "run" && *c == "started_at") || (by_key && i == 0))
                .collect();
            let rows = stmt
                .query_map([], |row| {
                    (0..count)
                        .map(|i| {
                            if wall_clock[i] {
                                Ok(Value::Null)
                            } else {
                                row.get::<_, Value>(i)
                            }
                        })
                        .collect()
                })
                .expect("query")
                .map(|r| r.expect("row"))
                .collect();
            (name, rows)
        })
        .collect()
}

/// A short account of how two differing values differ.
fn describe_difference(left: &Value, right: &Value) -> String {
    match (left, right) {
        (Value::Blob(a), Value::Blob(b)) => {
            let first = a.iter().zip(b).position(|(x, y)| x != y);
            format!(
                "blobs of {} and {} bytes, first differing byte {first:?}",
                a.len(),
                b.len()
            )
        }
        _ => format!("{left:?} versus {right:?}"),
    }
}

/// Runs a fresh store at `path` until the interrupted run's stop.
fn run_interrupted(path: &str, config: &FullConfig) {
    let store = SupervisorStore::create(path, config).expect("create store");
    let mut kernel = new_kernel(config);
    let mut run = SupervisorRun::start(&mut kernel, config, Some(store), true).expect("start");
    let (_, stopped) = drive(&mut run, &mut kernel, Some(STOP_AFTER_CHECKPOINTS));
    assert!(stopped, "the run reached its second checkpoint");
}

#[test]
fn supervisor_run_resume_matches_uninterrupted() {
    if !gpu_or_skip() {
        return;
    }
    let config = supervisor_config();
    let dir = tempfile::tempdir().expect("tempdir");
    let full_path = dir.path().join("full.db");
    let cut_path = dir.path().join("cut.db");
    let full_path = full_path.to_str().expect("utf8 path");
    let cut_path = cut_path.to_str().expect("utf8 path");

    let store = SupervisorStore::create(full_path, &config).expect("create store");
    let mut kernel = new_kernel(&config);
    let mut run = SupervisorRun::start(&mut kernel, &config, Some(store), true).expect("start");
    let (full_reports, _) = drive(&mut run, &mut kernel, None);
    let full_summary = run.finish().expect("finish").summary;

    run_interrupted(cut_path, &config);
    let point = load_resume_point(cut_path).expect("resume point");
    let tick = point.tick;
    let store = SupervisorStore::reopen_for_resume(cut_path, point.run_id, tick)
        .expect("reopen for resume");
    let mut kernel = new_kernel(&point.config);
    let mut run = SupervisorRun::resume(
        &mut kernel,
        &point.config,
        Some(store),
        point.state,
        &point.kernel,
    )
    .expect("resume");
    let mut resumed_reports = vec![run
        .on_sample(&mut kernel, tick, &point.physics, point.food.as_deref())
        .expect("first resumed sample")];
    let (rest, _) = drive(&mut run, &mut kernel, None);
    resumed_reports.extend(rest);
    let resumed_summary = run.finish().expect("finish").summary;

    let expected: Vec<SampleReport> = full_reports
        .into_iter()
        .filter(|r| r.tick >= tick)
        .collect();
    assert!(expected.len() > 1, "the run continued past the checkpoint");
    assert_eq!(resumed_reports, expected);
    assert_eq!(resumed_summary, full_summary);

    let full_db = Connection::open(full_path).expect("open full");
    full_db
        .busy_timeout(std::time::Duration::from_secs(5))
        .expect("busy timeout");
    let cut_db = Connection::open(cut_path).expect("open cut");
    cut_db
        .busy_timeout(std::time::Duration::from_secs(5))
        .expect("busy timeout");
    let full_rows = all_rows(&full_db);
    let cut_rows = all_rows(&cut_db);
    for ((name, a), (other, b)) in full_rows.iter().zip(&cut_rows) {
        assert_eq!(name, other);
        assert_eq!(a.len(), b.len(), "table {name} row count differs");
        for (r, (left, right)) in a.iter().zip(b).enumerate() {
            for (c, (x, y)) in left.iter().zip(right).enumerate() {
                assert!(
                    x == y,
                    "table {name} row {r} column {c} differs: {}",
                    describe_difference(x, y)
                );
            }
        }
    }
    assert_eq!(full_rows.len(), cut_rows.len());
}

#[test]
fn supervisor_run_resume_refuses_config_mismatch() {
    if !gpu_or_skip() {
        return;
    }
    let config = supervisor_config();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("mismatch.db");
    let path = path.to_str().expect("utf8 path");
    run_interrupted(path, &config);
    load_resume_point(path).expect("intact checkpoint loads");
    let db = Connection::open(path).expect("open");
    db.busy_timeout(std::time::Duration::from_secs(5))
        .expect("busy timeout");
    db.execute("UPDATE run SET brain_config = brain_config || ' '", [])
        .expect("edit brain config");
    drop(db);
    let error = load_resume_point(path).expect_err("edited config is refused");
    assert_eq!(error, CheckpointError::ConfigMismatch);
    // The run error type carries the same refusal.
    assert!(matches!(
        SupervisorRunError::from(error),
        SupervisorRunError::Checkpoint(CheckpointError::ConfigMismatch)
    ));
}

#[test]
fn run_summary_accumulator_serde_round_trip() {
    let mut original =
        RunSummaryAccumulator::new(Some(Inheritance::Darwinian), SEED, ACCUMULATOR_AGENTS, true);
    original
        .begin_season(0, Some(FoodKind::First))
        .expect("first season");
    let mut food = vec![0.0; FOOD_STATE_STRIDE];
    food[FOOD_POSITION_X] = 1.0;
    food[FOOD_POSITION_Z] = 0.0;
    let physics = |meals: u8| {
        let mut rows = vec![0.0; ACCUMULATOR_AGENTS * PHYS_STRIDE];
        for agent in 0..ACCUMULATOR_AGENTS {
            let base = agent * PHYS_STRIDE;
            rows[base + P_ALIVE] = 1.0;
            rows[base + P_FACING_Z] = 1.0;
            rows[base + P_MOTOR_TURN_OUT] = INEXACT_TURN;
            rows[base + P_FOOD_COUNT] = f32::from(meals);
        }
        rows
    };
    for i in 1..=SAMPLES_BEFORE_CAPTURE {
        original
            .record_physics_sample(u64::from(i), &physics(i), Some(&food))
            .expect("sample");
    }
    let text = serde_json::to_string(&original).expect("serialise");
    let mut copy: RunSummaryAccumulator = serde_json::from_str(&text).expect("deserialise");
    assert_eq!(text, serde_json::to_string(&copy).expect("reserialise"));

    for acc in [&mut original, &mut copy] {
        acc.begin_season(u64::from(SAMPLES_BEFORE_CAPTURE), Some(FoodKind::Second))
            .expect("second season");
        for i in (SAMPLES_BEFORE_CAPTURE + 1)..=SAMPLES_AFTER_CAPTURE {
            acc.record_physics_sample(u64::from(i), &physics(i), Some(&food))
                .expect("sample");
        }
    }
    let a = original
        .finish(u64::from(SAMPLES_AFTER_CAPTURE))
        .expect("finish original");
    let b = copy
        .finish(u64::from(SAMPLES_AFTER_CAPTURE))
        .expect("finish copy");
    assert_eq!(a, b);
    let kinds = a.kinds.expect("kind totals");
    assert!(kinds.late_toward_nourishing_samples > 0);
    assert_ne!(kinds.late_toward_nourishing_sum, 0.0);
}
