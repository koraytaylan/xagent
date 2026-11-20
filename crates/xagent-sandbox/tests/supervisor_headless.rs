//! Headless driver selection: the supervisor loop, its refusals and the default
//! generation driver.

use rusqlite::types::Value;
use rusqlite::Connection;
use xagent_brain::GpuKernel;
use xagent_sandbox::headless::{run_headless_with_options, HeadlessError, HeadlessOptions};
use xagent_sandbox::run_summary::RunSummary;
use xagent_shared::{EvolutionMode, FullConfig, Inheritance, SupervisorConfig};

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
/// Population of the generation-mode run.
const GENERATION_POPULATION: usize = 5;
/// Tick budget of the generation-mode run.
const GENERATION_TICKS: u64 = 200;
/// Generations of the generation-mode run.
const GENERATION_COUNT: u64 = 2;

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
    config.governor.evolution_mode = EvolutionMode::Supervisor;
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

/// All rows of `table` without the autoincrement `id`, ordered by `id`.
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

#[test]
fn gpu_guard_fails_closed_when_required() {
    assert_eq!(gpu_guard_decision(true, Some("1")), GuardDecision::Run);
    assert_eq!(gpu_guard_decision(false, None), GuardDecision::Skip);
    assert_eq!(gpu_guard_decision(false, Some("0")), GuardDecision::Skip);
    assert_eq!(gpu_guard_decision(false, Some("1")), GuardDecision::Fail);
    let result = std::panic::catch_unwind(|| {
        assert!(
            gpu_guard_decision(false, Some("1")) != GuardDecision::Fail,
            "fail closed"
        );
    });
    assert!(result.is_err());
}

#[test]
fn supervisor_headless_store_is_reproducible() {
    if !gpu_or_skip() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let mut stores = Vec::new();
    for name in ["a.db", "b.db"] {
        let path = dir.path().join(name);
        run_headless_with_options(
            supervisor_config(),
            path.to_str().expect("utf8 path"),
            false,
            &HeadlessOptions::default(),
        )
        .expect("supervisor run");
        let db = Connection::open(&path).expect("open");
        stores.push((
            rows(&db, "supervisor_event"),
            rows(&db, "supervisor_archive"),
            rows(&db, "supervisor_slot"),
        ));
    }
    assert!(!stores[0].0.is_empty(), "the run recorded events");
    assert_eq!(stores[0], stores[1]);
}

#[test]
fn supervisor_headless_writes_run_summary() {
    if !gpu_or_skip() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("run.db");
    let summary_path = dir.path().join("summary.json");
    let options = HeadlessOptions {
        run_summary: Some(summary_path.clone()),
    };
    run_headless_with_options(
        supervisor_config(),
        db_path.to_str().expect("utf8 path"),
        false,
        &options,
    )
    .expect("supervisor run");
    let text = std::fs::read_to_string(&summary_path).expect("summary written");
    let summary: RunSummary = serde_json::from_str(&text).expect("summary parses");
    assert_eq!(summary.mode, EvolutionMode::Supervisor);
}

#[test]
fn supervisor_headless_refuses_resume() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("resume.db");
    let result = run_headless_with_options(
        supervisor_config(),
        db_path.to_str().expect("utf8 path"),
        true,
        &HeadlessOptions::default(),
    );
    assert!(matches!(result, Err(HeadlessError::ResumeInSupervisorMode)));
    assert!(!db_path.exists(), "no database was touched");
}

#[test]
fn generation_mode_is_the_default_driver() {
    if !gpu_or_skip() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("generations.db");
    let mut config = FullConfig::default();
    config.governor.population_size = GENERATION_POPULATION;
    config.governor.tick_budget = GENERATION_TICKS;
    config.governor.max_generations = GENERATION_COUNT;
    run_headless_with_options(
        config,
        db_path.to_str().expect("utf8 path"),
        false,
        &HeadlessOptions::default(),
    )
    .expect("generation run");
    let db = Connection::open(&db_path).expect("open");
    assert!(count(&db, "node") > 0, "generation nodes were written");
    let tables: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'supervisor_%'",
            [],
            |row| row.get(0),
        )
        .expect("tables");
    assert_eq!(tables, 0, "no supervisor tables in generation mode");
}
