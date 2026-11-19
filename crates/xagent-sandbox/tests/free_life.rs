//! Free-life run: lesion parsing (CPU) and seeded runs (GPU).

use std::path::Path;

use rand::SeedableRng;
use tempfile::TempDir;
use xagent_brain::buffers::TrunkLesions;
use xagent_brain::GpuKernel;
use xagent_sandbox::agent::{fresh_brain_state, MAX_AGENTS};
use xagent_sandbox::free_life::tick_log::read_tick_log;
use xagent_sandbox::governor::Governor;
use xagent_sandbox::headless::{
    lesion_names, parse_lesions, run_free_life, BrainSource, FreeLifeError, FreeLifeOptions,
    FreeLifeReport,
};
use xagent_shared::{FullConfig, GovernorConfig};

/// Agents in a standard test run.
const TEST_AGENTS: usize = 3;
/// Requested ticks before rounding up to a batch multiple.
const TARGET_TICKS: u64 = 3_000;
/// Seed of the standard test run.
const TEST_SEED: u64 = 5;
/// Seed that differs from the standard one.
const OTHER_SEED: u64 = 6;
/// Seed of the fixture governor database.
const FIXTURE_SEED: u64 = 9;
/// Seed of the fixture birth brain.
const FIXTURE_BRAIN_SEED: u64 = 11;
/// Node id no fixture database holds.
const MISSING_NODE: i64 = 987_654;
/// Vision stride the runner rejects.
const BAD_VISION_STRIDE: u32 = 2;
/// Agents in a bare kernel used only to read its batch size or defaults.
const FRESH_KERNEL_AGENTS: u32 = 1;
/// Food items in a bare kernel.
const FRESH_KERNEL_FOOD: usize = 0;
/// A tick count of zero.
const ZERO_TICKS: u64 = 0;
/// One tick, used to step off a batch multiple.
const ONE_TICK: u64 = 1;
/// Batch size at which no tick count is off a batch multiple.
const UNIT_BATCH: u64 = 1;
/// Offset from the row count to the index of the last log row.
const LAST_ROW_OFFSET: usize = 1;
/// Column index of the generation in the stored node query.
const GENERATION_COLUMN: usize = 0;
/// An agent count below the accepted range.
const ZERO_AGENTS: usize = 0;
/// Distance above the maximum agent count that must be rejected.
const OVER_MAX_OFFSET: usize = 1;
/// Name of the fixture database file.
const FIXTURE_DB: &str = "governor.db";
/// Name of the tick log file.
const LOG_FILE: &str = "free-life.log";
/// Column that holds the brain tick in the tick log.
const BRAIN_TICK_COLUMN: &str = "brain_tick";

/// True when a GPU is usable; otherwise skips, or panics under
/// `XAGENT_REQUIRE_GPU=1`.
fn gpu_ready() -> bool {
    if GpuKernel::is_available() {
        return true;
    }
    assert!(
        std::env::var("XAGENT_REQUIRE_GPU").as_deref() != Ok("1"),
        "XAGENT_REQUIRE_GPU=1 but no GPU adapter is available"
    );
    println!("skipping: no GPU adapter");
    false
}

/// Kernel batch size of the default config.
fn batch_size(config: &FullConfig) -> u64 {
    let kernel = GpuKernel::new(
        FRESH_KERNEL_AGENTS,
        FRESH_KERNEL_FOOD,
        &config.brain,
        &config.world,
    );
    u64::from(kernel.kernel_batch_size())
}

/// Options of a standard run on `config`.
fn options(config: &FullConfig) -> FreeLifeOptions {
    let batch = batch_size(config);
    FreeLifeOptions {
        agent_count: TEST_AGENTS,
        ticks: TARGET_TICKS.div_ceil(batch) * batch,
        seed: TEST_SEED,
        brain_source: BrainSource::Fresh,
        lesions: TrunkLesions::NONE,
        log_path: None,
    }
}

fn run(config: &FullConfig, options: &FreeLifeOptions) -> FreeLifeReport {
    run_free_life(config, options).expect("free life runs")
}

/// Create a governor database whose current node holds `brain_state`.
fn fixture_db(
    dir: &Path,
    config: &FullConfig,
    mutate: impl FnOnce(&mut Vec<f32>),
) -> (String, i64) {
    let path = dir.join(FIXTURE_DB).to_string_lossy().into_owned();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(FIXTURE_BRAIN_SEED);
    let mut brain = fresh_brain_state(&config.brain, &mut rng);
    mutate(&mut brain.brain_state);
    let governor = Governor::new_seeded(
        &path,
        GovernorConfig::default(),
        &config.brain,
        "{}",
        FIXTURE_SEED,
    )
    .expect("governor database");
    let node = governor.current_node_id.expect("current node");
    assert!(governor.store_champion_brain(node, &brain));
    drop(governor);
    (path, node)
}

fn stored_generation(db_path: &str, node: i64) -> i64 {
    let db =
        rusqlite::Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .expect("open fixture");
    db.query_row("SELECT generation FROM node WHERE id = ?1", [node], |r| {
        r.get(GENERATION_COLUMN)
    })
    .expect("generation")
}

#[test]
fn free_life_parses_lesion_lists() {
    assert_eq!(parse_lesions("").unwrap(), TrunkLesions::NONE);
    assert_eq!(parse_lesions("none").unwrap(), TrunkLesions::NONE);
    assert_eq!(parse_lesions("all").unwrap(), TrunkLesions::ALL);
    let some = parse_lesions(" turn-weights , critic ").unwrap();
    assert!(some.turn_weights && some.critic);
    assert!(!some.forward_weights && !some.memory_blend);
    let mem = parse_lesions("memory-blend,forward-weights").unwrap();
    assert!(mem.memory_blend && mem.forward_weights);
}

#[test]
fn free_life_rejects_unknown_lesion_names() {
    for list in ["Critic", "turn-weights,", "bogus"] {
        assert!(
            matches!(parse_lesions(list), Err(FreeLifeError::UnknownLesion(_))),
            "{list}"
        );
    }
    match parse_lesions("critic,bogus") {
        Err(FreeLifeError::UnknownLesion(name)) => assert_eq!(name, "bogus"),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn free_life_lesion_names_are_stable() {
    assert!(lesion_names(TrunkLesions::NONE).is_empty());
    assert_eq!(
        lesion_names(TrunkLesions::ALL),
        ["turn-weights", "forward-weights", "memory-blend", "critic"]
    );
    let critic = TrunkLesions {
        critic: true,
        ..TrunkLesions::NONE
    };
    assert_eq!(lesion_names(critic), ["critic"]);
}

#[test]
fn free_life_is_reproducible_for_identical_options() {
    if !gpu_ready() {
        return;
    }
    let config = FullConfig::default();
    let opts = options(&config);
    assert_eq!(run(&config, &opts), run(&config, &opts));
}

#[test]
fn free_life_different_seed_gives_different_report() {
    if !gpu_ready() {
        return;
    }
    let config = FullConfig::default();
    let opts = options(&config);
    let other = FreeLifeOptions {
        seed: OTHER_SEED,
        ..opts.clone()
    };
    assert_ne!(run(&config, &opts), run(&config, &other));
}

#[test]
fn free_life_brain_ticks_and_samples_match_ticks() {
    if !gpu_ready() {
        return;
    }
    let config = FullConfig::default();
    let opts = options(&config);
    let report = run(&config, &opts);
    let batch = u64::from(report.kernel_batch_size);
    assert_eq!(report.brain_ticks, opts.ticks / batch);
    assert_eq!(report.agents.len(), TEST_AGENTS);
    assert_eq!(report.agent_count, TEST_AGENTS);
    assert_eq!(report.brain_source, "fresh");
    assert_eq!(report.champion_node_id, None);
    for agent in &report.agents {
        assert_eq!(agent.whole.samples, report.brain_ticks);
        assert!(agent.late.samples <= agent.whole.samples);
    }
}

#[test]
fn free_life_lesions_change_report_and_are_listed() {
    if !gpu_ready() {
        return;
    }
    let config = FullConfig::default();
    let opts = options(&config);
    let control = run(&config, &opts);
    assert!(control.lesions.is_empty());
    let lesioned = run(
        &config,
        &FreeLifeOptions {
            lesions: TrunkLesions::ALL,
            ..opts
        },
    );
    assert_eq!(
        lesioned.lesions,
        ["turn-weights", "forward-weights", "memory-blend", "critic"]
    );
    assert_ne!(control.agents, lesioned.agents);
}

#[test]
fn free_life_leaves_fresh_kernel_with_lesions_off() {
    if !gpu_ready() {
        return;
    }
    let config = FullConfig::default();
    let kernel = GpuKernel::new(
        FRESH_KERNEL_AGENTS,
        FRESH_KERNEL_FOOD,
        &config.brain,
        &config.world,
    );
    assert!(kernel.trunk_lesions().is_none());
    assert!(!kernel.brain_tick_probe_enabled());
}

#[test]
fn free_life_champion_source_reports_node_and_generation() {
    if !gpu_ready() {
        return;
    }
    let config = FullConfig::default();
    let dir = TempDir::new().expect("tempdir");
    let (db_path, node) = fixture_db(dir.path(), &config, |_| {});
    let generation = stored_generation(&db_path, node);
    let opts = FreeLifeOptions {
        brain_source: BrainSource::Champion {
            db_path,
            node_id: None,
        },
        ..options(&config)
    };
    let report = run(&config, &opts);
    assert_eq!(report.brain_source, "champion");
    assert_eq!(report.champion_node_id, Some(node));
    assert_eq!(report.champion_generation, Some(generation));
}

#[test]
fn free_life_champion_wrong_layout_is_rejected() {
    if !gpu_ready() {
        return;
    }
    let config = FullConfig::default();
    let dir = TempDir::new().expect("tempdir");
    let (db_path, _) = fixture_db(dir.path(), &config, |state| {
        state.pop();
    });
    let opts = FreeLifeOptions {
        brain_source: BrainSource::Champion {
            db_path,
            node_id: None,
        },
        ..options(&config)
    };
    assert!(matches!(
        run_free_life(&config, &opts),
        Err(FreeLifeError::ChampionLayout { .. })
    ));
}

#[test]
fn free_life_missing_champion_node_is_reported() {
    if !gpu_ready() {
        return;
    }
    let config = FullConfig::default();
    let dir = TempDir::new().expect("tempdir");
    let (db_path, _) = fixture_db(dir.path(), &config, |_| {});
    let opts = FreeLifeOptions {
        brain_source: BrainSource::Champion {
            db_path,
            node_id: Some(MISSING_NODE),
        },
        ..options(&config)
    };
    assert!(matches!(
        run_free_life(&config, &opts),
        Err(FreeLifeError::ChampionLoad(_))
    ));
}

#[test]
fn free_life_log_matches_report() {
    if !gpu_ready() {
        return;
    }
    let config = FullConfig::default();
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join(LOG_FILE);
    let opts = FreeLifeOptions {
        log_path: Some(path.clone()),
        ..options(&config)
    };
    let report = run(&config, &opts);
    let log = read_tick_log(&path).expect("tick log");
    let agents = u64::try_from(TEST_AGENTS).expect("agents fit");
    assert_eq!(log.rows, agents * report.brain_ticks);
    let tick_col = log
        .column_index(BRAIN_TICK_COLUMN)
        .expect("brain_tick column");
    let width = log.columns.len();
    let last_row = usize::try_from(log.rows)
        .expect("rows fit")
        .checked_sub(LAST_ROW_OFFSET)
        .expect("log has rows");
    let last_tick = log.values[last_row * width + tick_col];
    let expected_tick =
        u16::try_from(report.brain_ticks - ONE_TICK).expect("last tick fits in u16");
    assert_eq!(last_tick, f32::from(expected_tick));
}

#[test]
fn free_life_rejects_vision_stride_other_than_one() {
    if !gpu_ready() {
        return;
    }
    let mut config = FullConfig::default();
    let opts = options(&config);
    config.brain.vision_stride = BAD_VISION_STRIDE;
    assert!(matches!(
        run_free_life(&config, &opts),
        Err(FreeLifeError::VisionStride(BAD_VISION_STRIDE))
    ));
}

#[test]
fn free_life_rejects_agent_counts_outside_range() {
    if !gpu_ready() {
        return;
    }
    let config = FullConfig::default();
    for count in [ZERO_AGENTS, MAX_AGENTS + OVER_MAX_OFFSET] {
        let opts = FreeLifeOptions {
            agent_count: count,
            ..options(&config)
        };
        assert!(matches!(
            run_free_life(&config, &opts),
            Err(FreeLifeError::AgentCount(n)) if n == count
        ));
    }
}

#[test]
fn free_life_rejects_ticks_not_a_multiple_of_batch() {
    if !gpu_ready() {
        return;
    }
    let config = FullConfig::default();
    let base = options(&config);
    let mut bad = vec![ZERO_TICKS, base.ticks + ONE_TICK];
    if batch_size(&config) == UNIT_BATCH {
        bad.pop();
    }
    for ticks in bad {
        let opts = FreeLifeOptions {
            ticks,
            ..base.clone()
        };
        assert!(matches!(
            run_free_life(&config, &opts),
            Err(FreeLifeError::Ticks { .. })
        ));
    }
}
