//! Food-kind free life: frozen control and summary file.

use std::path::PathBuf;

use xagent_brain::buffers::TrunkLesions;
use xagent_brain::GpuKernel;
use xagent_sandbox::food_kind_metrics::{Arm, RunFoodKindSummary};
use xagent_sandbox::headless::{
    run_free_life_food_kind, BrainSource, FoodKindOutcome, FoodKindRun, FreeLifeOptions,
};
use xagent_sandbox::world::WorldState;
use xagent_shared::FullConfig;

/// Agents in a test run.
const TEST_AGENTS: usize = 2;
/// Requested ticks before rounding up to a batch multiple.
const TARGET_TICKS: u64 = 3_000;
/// Seed of the test runs.
const TEST_SEED: u64 = 5;
/// Seed of the bare kernel that reads the nourishing draw.
const DRAW_RESET_SEED: u64 = 0;
/// Name of the sense whose turn weights start and stay at zero when frozen.
const SECOND_ODOUR: &str = "second_odour";
/// Name of the summary file.
const SUMMARY_FILE: &str = "xagent-food-kind-free-life-summary";

/// Panics without an adapter when `XAGENT_REQUIRE_GPU=1`; otherwise reports
/// whether to skip.
fn gpu_or_skip() -> bool {
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
    let kernel = GpuKernel::new(1, 0, &config.brain, &config.world);
    u64::from(kernel.kernel_batch_size())
}

fn options(config: &FullConfig, source: BrainSource) -> FreeLifeOptions {
    let batch = batch_size(config);
    FreeLifeOptions {
        agent_count: TEST_AGENTS,
        ticks: TARGET_TICKS.div_ceil(batch) * batch,
        seed: TEST_SEED,
        brain_source: source,
        lesions: TrunkLesions::NONE,
        log_path: None,
    }
}

fn run(config: &FullConfig, freeze: bool, path: Option<PathBuf>) -> FoodKindOutcome {
    let run = FoodKindRun {
        freeze_policy: freeze,
        summary_path: path,
    };
    run_free_life_food_kind(config, &options(config, BrainSource::Fresh), &run)
        .expect("food-kind free life runs")
}

#[test]
fn food_kind_free_life_frozen_arm_keeps_turn_weights() {
    if !gpu_or_skip() {
        return;
    }
    let config = FullConfig::default();
    let frozen = run(&config, true, None);
    assert_eq!(frozen.summary.arm, Arm::Frozen);
    let mut checked_second = false;
    for (birth, end) in frozen
        .birth_turn_weights
        .iter()
        .zip(&frozen.final_turn_weights)
    {
        for (sense, name) in frozen.sense_names.iter().enumerate() {
            for (b, e) in birth[sense].iter().zip(&end[sense]) {
                if *name == SECOND_ODOUR {
                    checked_second = true;
                    assert_eq!(*e, 0.0, "second_odour turn weight moved");
                } else {
                    assert_eq!(b.to_bits(), e.to_bits(), "{name} turn weight moved");
                }
            }
        }
    }
    assert!(checked_second, "second_odour sense not found");
    let learning = run(&config, false, None);
    assert_eq!(learning.summary.arm, Arm::Learning);
    assert_eq!(
        learning.summary.nourishing_kind,
        frozen.summary.nourishing_kind
    );
}

#[test]
fn food_kind_free_life_summary_is_complete() {
    if !gpu_or_skip() {
        return;
    }
    let config = FullConfig::default();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(SUMMARY_FILE);
    let outcome = run(&config, false, Some(path.clone()));
    let text = std::fs::read_to_string(&path).expect("summary file is written");
    let summary: RunFoodKindSummary = serde_json::from_str(&text).expect("summary parses");
    assert_eq!(summary, outcome.summary);
    assert_eq!(summary.agent_count, TEST_AGENTS);
    assert_eq!(summary.agents.len(), TEST_AGENTS);
    assert!(summary.fresh_brains);
    assert_eq!(summary.seed, TEST_SEED);
    assert_eq!(summary.arm, Arm::Learning);

    let mut brain = config.brain.clone();
    brain.second_odour_sense_enabled = true;
    let mut world_config = config.world.clone();
    world_config.food_kinds_enabled = true;
    world_config.seed = TEST_SEED;
    let world = WorldState::new(world_config.clone());
    let mut kernel = GpuKernel::new(1, world.food_items.len(), &brain, &world_config);
    kernel.reset_agents_seeded(&brain, DRAW_RESET_SEED);
    assert_eq!(kernel.nourishing_food_kind(), Some(summary.nourishing_kind));

    for a in &summary.agents {
        let optional = [
            a.turn_toward_nourishing_whole,
            a.turn_toward_nourishing_late,
            a.turn_away_harmful_whole,
            a.turn_away_harmful_late,
            a.readout_balanced_accuracy,
            a.readout_shuffled_accuracy,
        ];
        assert!(optional.iter().flatten().all(|v| v.is_finite()));
        let required = [
            a.signed_credit_pair_first,
            a.signed_credit_pair_second,
            a.signed_weight_difference_pair_first,
            a.signed_weight_difference_pair_second,
        ];
        assert!(required.iter().all(|v| v.is_finite()));
    }
}
