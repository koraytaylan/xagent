//! Supervisor food-kind behaviour: kinds are off by default, and with kinds on
//! the seasons flip on the season schedule, alternate kinds, and keep the
//! kernel's nourishing kind equal to the reported kind.

use xagent_brain::buffers::food_kinds_slot_value;
use xagent_brain::GpuKernel;
use xagent_sandbox::supervisor::{stream_seed, TickSchedule, SEASON_STREAM};
use xagent_sandbox::supervisor_run::{SampleReport, SupervisorRun};
use xagent_sandbox::world::WorldState;
use xagent_shared::{FoodKind, FullConfig, Inheritance, SupervisorConfig};

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
/// Agent-tick budget of the small run.
const SMALL_BUDGET: u64 = 10 * 2_000;
/// Archive capacity of a run.
const ARCHIVE: usize = 8;
/// World seed of a run.
const SEED: u64 = 5;
/// Flip count the run must reach so that alternation is exercised at least twice.
const MIN_SEASON_FLIPS: usize = 3;
/// Ticks per dispatch, one sample period.
const SAMPLE: u64 = 100;
/// Both inheritance modes.
const MODES: [Inheritance; 2] = [Inheritance::WeightCopy, Inheritance::Darwinian];

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

fn config(kinds: bool, inheritance: Inheritance) -> FullConfig {
    let mut config = FullConfig::default();
    config.governor.population_size = AGENTS;
    config.governor.supervisor = SupervisorConfig {
        boundary_interval_ticks: BOUNDARY_TICKS,
        min_age_ticks: MIN_AGE_TICKS,
        score_window_ticks: WINDOW_TICKS,
        archive_capacity: ARCHIVE,
        season_period_ticks: SEASON_TICKS,
        inheritance,
        agent_tick_budget: BUDGET,
    };
    config.world.food_kinds_enabled = kinds;
    config.world.seed = SEED;
    config
}

fn kernel_for(config: &FullConfig, agents: usize) -> GpuKernel {
    let world = WorldState::new(config.world.clone());
    GpuKernel::new(
        u32::try_from(agents).expect("agent count fits"),
        world.food_items.len(),
        &config.brain,
        &config.world,
    )
}

/// Dispatches one sample period and copies the readback.
fn advance(kernel: &mut GpuKernel, tick: u64) -> (Vec<f32>, Option<Vec<f32>>) {
    kernel.dispatch_batch(tick, u32::try_from(SAMPLE).expect("sample fits"));
    while !kernel.try_collect_state_snapshot() {
        std::hint::spin_loop();
    }
    (
        kernel.cached_state().to_vec(),
        kernel.cached_food_state().map(<[f32]>::to_vec),
    )
}

/// The other food kind.
fn other(kind: FoodKind) -> FoodKind {
    match kind {
        FoodKind::First => FoodKind::Second,
        FoodKind::Second => FoodKind::First,
    }
}

#[test]
fn gpu_guard_fails_closed_when_required() {
    assert_eq!(gpu_guard_decision(true, None), GuardDecision::Run);
    assert_eq!(gpu_guard_decision(true, Some("1")), GuardDecision::Run);
    assert_eq!(gpu_guard_decision(false, None), GuardDecision::Skip);
    assert_eq!(gpu_guard_decision(false, Some("0")), GuardDecision::Skip);
    assert_eq!(gpu_guard_decision(false, Some("1")), GuardDecision::Fail);
}

#[test]
fn supervisor_food_kinds_off_by_default() {
    if !gpu_or_skip() {
        return;
    }
    let mut config = FullConfig::default();
    assert!(!config.world.food_kinds_enabled);
    config.governor.population_size = AGENTS;
    config.governor.supervisor = SupervisorConfig {
        boundary_interval_ticks: BOUNDARY_TICKS,
        min_age_ticks: MIN_AGE_TICKS,
        score_window_ticks: WINDOW_TICKS,
        archive_capacity: ARCHIVE,
        season_period_ticks: SEASON_TICKS,
        inheritance: Inheritance::WeightCopy,
        agent_tick_budget: SMALL_BUDGET,
    };
    config.world.seed = SEED;
    let mut kernel = kernel_for(&config, AGENTS);
    let mut run = SupervisorRun::start(&mut kernel, &config, None, false).expect("run starts");
    assert_eq!(kernel.nourishing_food_kind(), None);
    assert_eq!(food_kinds_slot_value(None), 0.0);
    let mut reports: Vec<SampleReport> = Vec::new();
    let mut tick = 0;
    while !run.is_finished() {
        let (physics, food) = advance(&mut kernel, tick);
        tick += SAMPLE;
        reports.push(
            run.on_sample(&mut kernel, tick, &physics, food.as_deref())
                .expect("sample is handled"),
        );
    }
    run.finish().expect("run finishes");
    assert_eq!(kernel.nourishing_food_kind(), None);
    assert!(reports.iter().all(|r| r.season.is_none()));
}

#[test]
fn supervisor_seasons_flip_on_schedule() {
    if !gpu_or_skip() {
        return;
    }
    for inheritance in MODES {
        let config = config(true, inheritance);
        let mut kernel = kernel_for(&config, AGENTS);
        let mut run = SupervisorRun::start(&mut kernel, &config, None, false).expect("run starts");
        let initial = kernel.nourishing_food_kind().expect("kinds are on");
        let mut reports: Vec<(u64, SampleReport, Option<FoodKind>)> = Vec::new();
        let mut tick = 0;
        while !run.is_finished() {
            let (physics, food) = advance(&mut kernel, tick);
            tick += SAMPLE;
            let report = run
                .on_sample(&mut kernel, tick, &physics, food.as_deref())
                .expect("sample is handled");
            let now = kernel.nourishing_food_kind();
            reports.push((tick, report, now));
        }
        let final_tick = tick;
        let mut schedule =
            TickSchedule::new(stream_seed(config.world.seed, SEASON_STREAM), SEASON_TICKS);
        let mut expected = Vec::new();
        while schedule.next_tick() <= final_tick {
            expected.push(schedule.next_tick());
            schedule.advance();
        }
        let flipped: Vec<u64> = reports
            .iter()
            .filter(|(_, r, _)| r.season.is_some())
            .map(|(t, _, _)| *t)
            .collect();
        assert_eq!(flipped, expected, "{inheritance:?}");
        assert!(
            flipped.len() >= MIN_SEASON_FLIPS,
            "{inheritance:?}: too few flips"
        );
        let mut current = initial;
        for (tick, report, now) in &reports {
            if let Some(season) = report.season {
                assert_eq!(season, other(current), "{inheritance:?} tick {tick}");
                current = season;
            }
            assert_eq!(*now, Some(current), "{inheritance:?} tick {tick}");
        }
        run.finish().expect("run finishes");
    }
}

#[test]
fn supervisor_start_draws_world_slot_one() {
    if !gpu_or_skip() {
        return;
    }
    for inheritance in MODES {
        let config = config(true, inheritance);
        let mut kernel = kernel_for(&config, AGENTS);
        SupervisorRun::start(&mut kernel, &config, None, false).expect("run starts");
        // Two draws happen: construction takes slot 0 and the single reset in
        // start takes slot 1, which is the kind left in the kernel.
        assert_eq!(
            kernel.nourishing_food_kind(),
            Some(config.world.nourishing_food_kind(1)),
            "{inheritance:?}"
        );
    }
}
