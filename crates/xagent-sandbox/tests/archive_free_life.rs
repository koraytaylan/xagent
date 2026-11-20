//! Free lives born from supervisor archive entries, loaded read only.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

use xagent_brain::buffers::TrunkLesions;
use xagent_brain::GpuKernel;
use xagent_sandbox::headless::{
    run_free_life, run_headless_with_options, BrainSource, FreeLifeError, FreeLifeOptions,
    FreeLifeReport, HeadlessOptions,
};
use xagent_sandbox::supervisor_store::{load_archive_entry_read_only, ArchiveLoadError};
use xagent_shared::{EvolutionMode, FullConfig, Inheritance, SupervisorConfig};

/// Agents in the supervisor run.
const SUPERVISOR_AGENTS: usize = 10;
/// Boundary interval in ticks.
const BOUNDARY_TICKS: u64 = 1_000;
/// Minimum age before an underperformer can be replaced.
const MIN_AGE_TICKS: u64 = 2_000;
/// Score window in ticks.
const WINDOW_TICKS: u64 = 1_000;
/// Season period in ticks.
const SEASON_TICKS: u64 = 3_000;
/// Agent-tick budget of the supervisor run.
const BUDGET: u64 = 10 * 20_000;
/// Archive capacity of the supervisor run.
const ARCHIVE: usize = 8;
/// World seed of the supervisor run.
const SUPERVISOR_SEED: u64 = 5;
/// Fewest archive entries the setup must produce.
const MIN_ENTRIES: usize = 2;
/// Agents in a free life.
const FREE_LIFE_AGENTS: usize = 3;
/// Requested free-life ticks before rounding up to a batch multiple.
const TARGET_TICKS: u64 = 1_000;
/// Seed of a free life.
const FREE_LIFE_SEED: u64 = 7;
/// Rank of the best archive entry.
const BEST_RANK: usize = 0;
/// Agents in a bare kernel used only to read its batch size.
const FRESH_KERNEL_AGENTS: u32 = 1;
/// Food items in a bare kernel.
const FRESH_KERNEL_FOOD: usize = 0;
/// Name of the supervisor database file.
const DB_FILE: &str = "supervisor.db";

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

/// Panics without an adapter when `XAGENT_REQUIRE_GPU=1` (as
/// `gpu_adapter_present_when_required` does); otherwise reports whether to run.
fn gpu_or_skip() -> bool {
    let required = std::env::var("XAGENT_REQUIRE_GPU").ok();
    match gpu_guard_decision(GpuKernel::is_available(), required.as_deref()) {
        GuardDecision::Run => true,
        GuardDecision::Skip => {
            println!("skipping: no GPU adapter");
            false
        }
        GuardDecision::Fail => panic!("XAGENT_REQUIRE_GPU set but no GPU adapter is available"),
    }
}

fn supervisor_config() -> FullConfig {
    let mut config = FullConfig::default();
    config.governor.evolution_mode = EvolutionMode::Supervisor;
    config.governor.population_size = SUPERVISOR_AGENTS;
    config.governor.supervisor = SupervisorConfig {
        boundary_interval_ticks: BOUNDARY_TICKS,
        min_age_ticks: MIN_AGE_TICKS,
        score_window_ticks: WINDOW_TICKS,
        archive_capacity: ARCHIVE,
        season_period_ticks: SEASON_TICKS,
        inheritance: Inheritance::Darwinian,
        agent_tick_budget: BUDGET,
    };
    config.world.seed = SUPERVISOR_SEED;
    config
}

/// Build a supervisor database whose archive holds at least two entries.
fn archive_db(dir: &Path) -> String {
    let path = dir.join(DB_FILE).to_string_lossy().into_owned();
    run_headless_with_options(
        supervisor_config(),
        &path,
        false,
        &HeadlessOptions::default(),
    )
    .expect("supervisor run");
    let mut len = 0;
    loop {
        match load_archive_entry_read_only(&path, len) {
            Ok(_) => len += 1,
            Err(ArchiveLoadError::RankOutOfRange { .. }) => break,
            Err(other) => panic!("archive probe at rank {len} failed: {other:?}"),
        }
    }
    assert!(
        len >= MIN_ENTRIES,
        "archive holds {len} entries, setup needs at least {MIN_ENTRIES}"
    );
    path
}

fn options(config: &FullConfig, db_path: &str, rank: usize) -> FreeLifeOptions {
    let kernel = GpuKernel::new(
        FRESH_KERNEL_AGENTS,
        FRESH_KERNEL_FOOD,
        &config.brain,
        &config.world,
    );
    let batch = u64::from(kernel.kernel_batch_size());
    FreeLifeOptions {
        agent_count: FREE_LIFE_AGENTS,
        ticks: TARGET_TICKS.div_ceil(batch) * batch,
        seed: FREE_LIFE_SEED,
        brain_source: BrainSource::Archive {
            db_path: db_path.to_string(),
            rank,
        },
        lesions: TrunkLesions::NONE,
        log_path: None,
    }
}

fn file_hash(path: &str) -> (u64, Vec<u8>) {
    let bytes = std::fs::read(path).expect("read db");
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    (hasher.finish(), bytes)
}

#[test]
fn gpu_guard_fails_closed_when_required() {
    assert_eq!(gpu_guard_decision(true, Some("1")), GuardDecision::Run);
    assert_eq!(gpu_guard_decision(false, None), GuardDecision::Skip);
    assert_eq!(gpu_guard_decision(false, Some("0")), GuardDecision::Skip);
    assert_eq!(gpu_guard_decision(false, Some("1")), GuardDecision::Fail);
}

#[test]
fn archive_free_life_runs_rank_zero_entry() {
    if !gpu_or_skip() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let db = archive_db(dir.path());
    let entry = load_archive_entry_read_only(&db, BEST_RANK).expect("rank 0");
    let config = FullConfig::default();
    let opts = options(&config, &db, BEST_RANK);
    let report = run_free_life(&config, &opts).expect("archive free life");
    assert_eq!(report.brain_source, "archive");
    assert_eq!(
        report.archive_entry_id,
        Some(i64::try_from(entry.entry_id).expect("id fits"))
    );
    assert_eq!(report.agent_count, FREE_LIFE_AGENTS);
    assert_eq!(report.agents.len(), FREE_LIFE_AGENTS);
    assert_eq!(report.ticks, opts.ticks);
    assert_eq!(report.champion_node_id, None);
    assert_eq!(report.champion_generation, None);
}

#[test]
fn archive_free_life_leaves_database_bytes_unchanged() {
    if !gpu_or_skip() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let db = archive_db(dir.path());
    let (before_hash, before) = file_hash(&db);
    let config = FullConfig::default();
    run_free_life(&config, &options(&config, &db, BEST_RANK)).expect("archive free life");
    let (after_hash, after) = file_hash(&db);
    assert_eq!(before_hash, after_hash);
    assert!(before == after, "database bytes changed");
}

#[test]
fn archive_free_life_rank_beyond_archive_is_error() {
    if !gpu_or_skip() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let db = archive_db(dir.path());
    let config = FullConfig::default();
    let rank = ARCHIVE + 1;
    match run_free_life(&config, &options(&config, &db, rank)) {
        Err(FreeLifeError::ArchiveLoad(ArchiveLoadError::RankOutOfRange { rank: r, .. })) => {
            assert_eq!(r, rank);
        }
        other => panic!("expected RankOutOfRange, got {other:?}"),
    }
}

#[test]
fn archive_free_life_is_reproducible() {
    if !gpu_or_skip() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let db = archive_db(dir.path());
    let config = FullConfig::default();
    let opts = options(&config, &db, BEST_RANK);
    let first = run_free_life(&config, &opts).expect("first");
    let second = run_free_life(&config, &opts).expect("second");
    assert_eq!(first, second);
}

#[test]
fn free_life_report_without_archive_field_deserializes() {
    let json = r#"{"agent_count":1,"ticks":10,"seed":3,"brain_source":"fresh",
        "champion_node_id":null,"champion_generation":null,"lesions":[],
        "kernel_batch_size":1,"brain_ticks":0,"agents":[]}"#;
    let report: FreeLifeReport = serde_json::from_str(json).expect("deserialize");
    assert_eq!(report.archive_entry_id, None);
    assert_eq!(report.brain_source, "fresh");
}
