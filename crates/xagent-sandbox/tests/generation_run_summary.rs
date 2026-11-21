//! Generation-mode run summary and the agent-tick budget conversion.

use rusqlite::Connection;
use xagent_sandbox::headless::{
    max_generations_for_agent_ticks, run_headless_with_options, HeadlessError, HeadlessOptions,
};
use xagent_sandbox::run_summary::RunSummary;
use xagent_shared::FullConfig;

/// Agents in every run.
const AGENTS: usize = 4;
/// Ticks per generation.
const TICK_BUDGET: u64 = 2_000;
/// Generations the budget conversion test runs.
const BUDGET_GENERATIONS: u64 = 3;
/// Converter result for a one-generation budget: `q + 1` with `q = 1`.
const ONE_GENERATION_MAX: u64 = 2;
/// Generations of the summary tests.
const SUMMARY_GENERATIONS: u64 = 2;

/// Fail closed when the GPU is required but absent.
fn gpu_adapter_present_when_required() {
    if std::env::var("XAGENT_REQUIRE_GPU").is_ok() {
        assert!(
            xagent_brain::GpuKernel::is_available(),
            "XAGENT_REQUIRE_GPU set but no GPU/lavapipe adapter — CI would silently skip the GPU suite"
        );
    }
}

/// Whether to run a GPU test.
fn gpu_or_skip() -> bool {
    gpu_adapter_present_when_required();
    if xagent_brain::GpuKernel::is_available() {
        true
    } else {
        println!("skipping: no GPU adapter");
        false
    }
}

fn config(max_generations: u64) -> FullConfig {
    let mut config = FullConfig::default();
    config.governor.population_size = AGENTS;
    config.governor.tick_budget = TICK_BUDGET;
    config.governor.max_generations = max_generations;
    config
}

/// Run a generation run and return its summary and database.
fn run(config: FullConfig) -> (RunSummary, Connection, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("generations.db");
    let summary_path = dir.path().join("summary.json");
    let options = HeadlessOptions {
        run_summary: Some(summary_path.clone()),
        ..HeadlessOptions::default()
    };
    run_headless_with_options(
        config,
        db_path.to_str().expect("utf8 path"),
        false,
        &options,
    )
    .expect("generation run");
    let json = std::fs::read_to_string(&summary_path).expect("summary file");
    let summary: RunSummary = serde_json::from_str(&json).expect("summary json");
    let db = Connection::open(&db_path).expect("open");
    (summary, db, dir)
}

fn scalar(db: &Connection, sql: &str) -> u64 {
    db.query_row(sql, [], |row| row.get::<_, i64>(0))
        .map(|v| u64::try_from(v).expect("non-negative"))
        .expect("scalar")
}

#[test]
fn gpu_guard_fails_closed_when_required() {
    gpu_adapter_present_when_required();
}

#[test]
fn max_generations_for_agent_ticks_rejects_bad_budgets() {
    let product = AGENTS as u64 * TICK_BUDGET;
    assert!(matches!(
        max_generations_for_agent_ticks(0, AGENTS, TICK_BUDGET),
        Err(HeadlessError::InvalidAgentTickBudget)
    ));
    assert!(matches!(
        max_generations_for_agent_ticks(product, 0, TICK_BUDGET),
        Err(HeadlessError::InvalidAgentTickBudget)
    ));
    assert!(matches!(
        max_generations_for_agent_ticks(product, AGENTS, 0),
        Err(HeadlessError::InvalidAgentTickBudget)
    ));
    assert!(matches!(
        max_generations_for_agent_ticks(product + 1, AGENTS, TICK_BUDGET),
        Err(HeadlessError::InvalidAgentTickBudget)
    ));
    assert!(matches!(
        max_generations_for_agent_ticks(u64::MAX, usize::MAX, u64::MAX),
        Err(HeadlessError::InvalidAgentTickBudget)
    ));
    assert!(matches!(
        max_generations_for_agent_ticks(u64::MAX, 1, 1),
        Err(HeadlessError::InvalidAgentTickBudget)
    ));
    assert_eq!(
        max_generations_for_agent_ticks(product, AGENTS, TICK_BUDGET).expect("one generation"),
        ONE_GENERATION_MAX
    );
}

#[test]
fn generation_summary_matches_agent_results() {
    if !gpu_or_skip() {
        return;
    }
    let (summary, db, _dir) = run(config(SUMMARY_GENERATIONS + 1));
    let generations = scalar(&db, "SELECT COUNT(DISTINCT node_id) FROM agent_result");
    assert_eq!(generations, SUMMARY_GENERATIONS);
    assert_eq!(
        summary.nourishing_meals,
        scalar(&db, "SELECT SUM(food_consumed) FROM agent_result")
    );
    assert_eq!(
        summary.deaths,
        scalar(&db, "SELECT SUM(death_count) FROM agent_result")
    );
    assert_eq!(
        summary.agent_ticks,
        generations * TICK_BUDGET * AGENTS as u64
    );
    assert_eq!(summary.seasons, generations);
}

#[test]
fn generation_budget_runs_exact_agent_ticks() {
    if !gpu_or_skip() {
        return;
    }
    let budget = BUDGET_GENERATIONS * TICK_BUDGET * AGENTS as u64;
    let max = max_generations_for_agent_ticks(budget, AGENTS, TICK_BUDGET).expect("converter");
    let (summary, db, _dir) = run(config(max));
    assert_eq!(summary.agent_ticks, budget);
    assert_eq!(
        scalar(&db, "SELECT COUNT(DISTINCT node_id) FROM agent_result"),
        BUDGET_GENERATIONS
    );
    assert_eq!(
        scalar(&db, "SELECT COUNT(*) FROM agent_result"),
        BUDGET_GENERATIONS * AGENTS as u64
    );
}

#[test]
fn generation_summary_kinds_follow_each_generation() {
    if !gpu_or_skip() {
        return;
    }
    let mut cfg = config(SUMMARY_GENERATIONS + 1);
    cfg.world.food_kinds_enabled = true;
    cfg.brain.second_odour_sense_enabled = true;
    let (summary, db, _dir) = run(cfg);
    let generations = scalar(&db, "SELECT COUNT(DISTINCT node_id) FROM agent_result");
    let kinds = summary.kinds.expect("kinds are on");
    assert_eq!(summary.seasons, generations);
    assert!(kinds.early_nourishing_meals <= summary.nourishing_meals);
    assert!(kinds.late_nourishing_meals <= summary.nourishing_meals);
    assert!(kinds.early_harmful_meals <= summary.harmful_meals);
    assert!(kinds.late_harmful_meals <= summary.harmful_meals);
}
