//! Shared supervisor driver: reproducibility, batches, boundaries, seasons and
//! setup validation.

use rusqlite::Connection;
use xagent_brain::buffers::{AgentBrainState, PHYS_STRIDE, P_ALIVE, P_DEATH_COUNT, P_FOOD_COUNT};
use xagent_brain::GpuKernel;
use xagent_sandbox::run_summary::RunSummary;
use xagent_sandbox::supervisor::{stream_seed, ParentRef, TickSchedule, SEASON_STREAM};
use xagent_sandbox::supervisor_run::{SampleReport, SupervisorRun, SupervisorRunError};
use xagent_sandbox::supervisor_store::SupervisorStore;
use xagent_sandbox::world::WorldState;
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
/// Ticks per dispatch, one sample period.
const SAMPLE: u64 = 100;

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

fn bits(state: &AgentBrainState) -> Vec<u32> {
    state
        .brain_state
        .iter()
        .chain(&state.patterns)
        .map(|v| v.to_bits())
        .collect()
}

struct Driven {
    reports: Vec<SampleReport>,
    births: Vec<Vec<u32>>,
    summary: Option<RunSummary>,
    batches: u64,
    batch_size: u64,
    ticks: u64,
}

fn drive(config: &FullConfig, measure: bool) -> Driven {
    let mut kernel = kernel_for(config, AGENTS);
    let mut run = SupervisorRun::start(&mut kernel, config, None, measure).expect("run starts");
    let mut reports = Vec::new();
    let mut tick = 0;
    while !run.is_finished() {
        let (physics, food) = advance(&mut kernel, tick);
        tick += SAMPLE;
        reports.push(
            run.on_sample(&mut kernel, tick, &physics, food.as_deref())
                .expect("sample is handled"),
        );
    }
    let births = (0..AGENTS)
        .map(|slot| bits(run.birth_state(slot).expect("slot exists")))
        .collect();
    let summary = run.finish().expect("run finishes").summary;
    Driven {
        reports,
        births,
        summary,
        batches: kernel.probe_kernel_batches(),
        batch_size: u64::from(kernel.kernel_batch_size()),
        ticks: tick,
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
fn supervisor_run_is_reproducible() {
    if !gpu_or_skip() {
        return;
    }
    let config = config(false, Inheritance::Darwinian);
    let first = drive(&config, false);
    let second = drive(&config, false);
    assert_eq!(first.reports, second.reports);
    assert_eq!(first.births, second.births);
    assert!(first.reports.iter().any(|r| r.boundary.is_some()));
}

#[test]
fn supervisor_run_uses_one_kernel_and_exact_batches() {
    if !gpu_or_skip() {
        return;
    }
    let config = config(false, Inheritance::Darwinian);
    let driven = drive(&config, false);
    assert_eq!(driven.ticks, BUDGET / u64::try_from(AGENTS).expect("fits"));
    assert_eq!(driven.batches, driven.ticks / driven.batch_size);
    assert!(driven.reports.last().is_some_and(|r| r.finished));
}

#[test]
fn supervisor_run_boundary_writes_only_replaced_slot() {
    if !gpu_or_skip() {
        return;
    }
    let config = config(false, Inheritance::Darwinian);
    let mut kernel = kernel_for(&config, AGENTS);
    let mut run = SupervisorRun::start(&mut kernel, &config, None, false).expect("run starts");
    let mut tick = 0;
    while !run.is_finished() {
        let (physics, food) = advance(&mut kernel, tick);
        tick += SAMPLE;
        let rows_before = kernel.read_full_state_blocking().to_vec();
        let brains_before: Vec<_> = (0..AGENTS)
            .map(|i| bits(&kernel.read_agent_state(u32::try_from(i).expect("fits"))))
            .collect();
        let report = run
            .on_sample(&mut kernel, tick, &physics, food.as_deref())
            .expect("sample is handled");
        let Some(slot) = report.boundary.and_then(|b| b.slot) else {
            continue;
        };
        let rows_after = kernel.read_full_state_blocking().to_vec();
        for i in 0..AGENTS {
            let range = i * PHYS_STRIDE..(i + 1) * PHYS_STRIDE;
            let brain = bits(&kernel.read_agent_state(u32::try_from(i).expect("fits")));
            if i == slot {
                assert_eq!(rows_after[range.start + P_FOOD_COUNT], 0.0);
                assert_eq!(rows_after[range.start + P_DEATH_COUNT], 0.0);
                assert_eq!(rows_after[range.start + P_ALIVE], 1.0);
                let written = kernel.read_agent_state(u32::try_from(slot).expect("fits"));
                let birth = run.birth_state(slot).expect("slot");
                assert_eq!(written.patterns, birth.patterns);
                assert_ne!(brain, brains_before[i]);
            } else {
                assert_eq!(rows_before[range.clone()], rows_after[range]);
                assert_eq!(brain, brains_before[i]);
            }
        }
        return;
    }
    panic!("no replacement happened");
}

#[test]
fn supervisor_run_flips_kind_only_at_season_ticks() {
    if !gpu_or_skip() {
        return;
    }
    let config = config(true, Inheritance::Darwinian);
    let mut kernel = kernel_for(&config, AGENTS);
    let mut run = SupervisorRun::start(&mut kernel, &config, None, false).expect("run starts");
    let mut schedule = TickSchedule::new(stream_seed(SEED, SEASON_STREAM), SEASON_TICKS);
    let mut kind = kernel.nourishing_food_kind().expect("kinds are on");
    let (mut tick, mut flips) = (0, 0);
    while !run.is_finished() {
        let (physics, food) = advance(&mut kernel, tick);
        tick += SAMPLE;
        let report = run
            .on_sample(&mut kernel, tick, &physics, food.as_deref())
            .expect("sample is handled");
        let now = kernel.nourishing_food_kind().expect("kinds are on");
        let season_tick = tick == schedule.next_tick();
        if season_tick {
            schedule.advance();
            flips += 1;
        }
        assert_eq!(now != kind, season_tick, "tick {tick}");
        assert_eq!(report.season.is_some(), season_tick, "tick {tick}");
        kind = now;
    }
    assert!(flips > 0, "no season flipped");
}

#[test]
fn supervisor_run_rejects_invalid_setup() {
    if !gpu_or_skip() {
        return;
    }
    let good = config(false, Inheritance::Darwinian);
    let mut kernel = kernel_for(&good, AGENTS - 1);
    let err = SupervisorRun::start(&mut kernel, &good, None, false).err();
    assert!(matches!(err, Some(SupervisorRunError::AgentCount { .. })));

    let mut bad = good.clone();
    bad.governor.supervisor.boundary_interval_ticks = BOUNDARY_TICKS + 50;
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("run.db");
    let path = path.to_str().expect("utf-8 path");
    let store = SupervisorStore::create(path, &bad).expect("store opens");
    let mut kernel = kernel_for(&bad, AGENTS);
    let err = SupervisorRun::start(&mut kernel, &bad, Some(store), false).err();
    assert!(matches!(err, Some(SupervisorRunError::Config(_))));
    let db = Connection::open(path).expect("db opens");
    let slots: i64 = db
        .query_row("SELECT COUNT(*) FROM supervisor_slot", [], |row| row.get(0))
        .expect("count");
    assert_eq!(slots, 0);
}

#[test]
fn supervisor_run_weight_copy_reads_learned_parent() {
    if !gpu_or_skip() {
        return;
    }
    let config = config(false, Inheritance::WeightCopy);
    let mut kernel = kernel_for(&config, AGENTS);
    let mut run = SupervisorRun::start(&mut kernel, &config, None, false).expect("run starts");
    let mut tick = 0;
    while !run.is_finished() {
        let (physics, food) = advance(&mut kernel, tick);
        tick += SAMPLE;
        let learned: Vec<_> = (0..AGENTS)
            .map(|i| kernel.read_agent_state(u32::try_from(i).expect("fits")))
            .collect();
        let report = run
            .on_sample(&mut kernel, tick, &physics, food.as_deref())
            .expect("sample is handled");
        let Some(boundary) = report.boundary else {
            continue;
        };
        let (Some(slot), Some(ParentRef::PeerLearned(peer))) = (boundary.slot, boundary.parent)
        else {
            continue;
        };
        let newborn = run.birth_state(slot).expect("slot");
        let distance = |a: &AgentBrainState, b: &AgentBrainState| -> f32 {
            a.brain_state
                .iter()
                .zip(&b.brain_state)
                .map(|(x, y)| (x - y).abs())
                .sum()
        };
        let to_learned = distance(newborn, &learned[peer]);
        let to_birth = distance(newborn, run.birth_state(peer).expect("peer"));
        assert!(to_learned < to_birth, "{to_learned} !< {to_birth}");
        return;
    }
    panic!("no weight-copy replacement happened");
}

#[test]
fn supervisor_run_summary_counts_agent_ticks() {
    if !gpu_or_skip() {
        return;
    }
    let config = config(true, Inheritance::Darwinian);
    let driven = drive(&config, true);
    let summary = driven.summary.expect("summary is measured");
    assert_eq!(summary.mode, EvolutionMode::Supervisor);
    assert_eq!(summary.inheritance, Some(Inheritance::Darwinian));
    assert_eq!(summary.agent_ticks, BUDGET);
    assert!(summary.seasons >= 1);
}
