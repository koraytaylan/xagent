//! Supervisor GPU parity: a run is independent of how the kernel batches its
//! ticks, and newborns keep the modulator flags the base config turns off.

use rusqlite::Connection;
use xagent_brain::buffers::AgentBrainState;
use xagent_brain::GpuKernel;
use xagent_sandbox::supervisor_run::{SampleReport, SupervisorRun};
use xagent_sandbox::supervisor_store::SupervisorStore;
use xagent_sandbox::world::WorldState;
use xagent_shared::{BrainConfig, FullConfig, Inheritance, SupervisorConfig};

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

fn bits(state: &AgentBrainState) -> Vec<u32> {
    state
        .brain_state
        .iter()
        .chain(&state.patterns)
        .map(|v| v.to_bits())
        .collect()
}

/// Collects the dispatched state and copies the readback.
fn collect(kernel: &mut GpuKernel) -> (Vec<f32>, Option<Vec<f32>>) {
    while !kernel.try_collect_state_snapshot() {
        std::hint::spin_loop();
    }
    (
        kernel.cached_state().to_vec(),
        kernel.cached_food_state().map(<[f32]>::to_vec),
    )
}

/// Fused path: one dispatch of a whole sample period.
fn advance_fused(kernel: &mut GpuKernel, tick: u64) -> (Vec<f32>, Option<Vec<f32>>) {
    kernel.dispatch_batch(tick, u32::try_from(SAMPLE).expect("sample fits"));
    collect(kernel)
}

/// Split path: the same ticks as chunks of the kernel batch size.
fn advance_split(kernel: &mut GpuKernel, tick: u64) -> (Vec<f32>, Option<Vec<f32>>) {
    let batch = kernel.kernel_batch_size();
    let total = u32::try_from(SAMPLE).expect("sample fits");
    let mut done = 0;
    while done < total {
        let chunk = batch.min(total - done);
        kernel.dispatch_ticks(tick + u64::from(done), chunk);
        done += chunk;
    }
    assert!(kernel.request_state_snapshot(), "snapshot request accepted");
    collect(kernel)
}

type SampleBits = (Vec<u32>, Option<Vec<u32>>);

fn drive(config: &FullConfig, split: bool) -> (Vec<SampleBits>, Vec<SampleReport>, Vec<Vec<u32>>) {
    let mut kernel = kernel_for(config, AGENTS);
    let mut run = SupervisorRun::start(&mut kernel, config, None, false).expect("run starts");
    let (mut samples, mut reports) = (Vec::new(), Vec::new());
    let mut tick = 0;
    while !run.is_finished() {
        let (physics, food) = if split {
            advance_split(&mut kernel, tick)
        } else {
            advance_fused(&mut kernel, tick)
        };
        tick += SAMPLE;
        samples.push((
            physics.iter().map(|v| v.to_bits()).collect(),
            food.as_ref()
                .map(|f| f.iter().map(|v| v.to_bits()).collect()),
        ));
        reports.push(
            run.on_sample(&mut kernel, tick, &physics, food.as_deref())
                .expect("sample is handled"),
        );
    }
    let births = (0..AGENTS)
        .map(|slot| bits(run.birth_state(slot).expect("slot exists")))
        .collect();
    (samples, reports, births)
}

#[test]
fn gpu_guard_decision_fails_closed() {
    assert_eq!(gpu_guard_decision(false, Some("1")), GuardDecision::Fail);
    assert_eq!(gpu_guard_decision(false, None), GuardDecision::Skip);
    assert_eq!(gpu_guard_decision(false, Some("0")), GuardDecision::Skip);
    assert_eq!(gpu_guard_decision(true, Some("1")), GuardDecision::Run);
    assert_eq!(gpu_guard_decision(true, None), GuardDecision::Run);
}

#[test]
fn supervisor_fused_matches_split_dispatch() {
    if !gpu_or_skip() {
        return;
    }
    for inheritance in [Inheritance::WeightCopy, Inheritance::Darwinian] {
        let config = config(false, inheritance);
        let (samples_a, reports_a, births_a) = drive(&config, false);
        let (samples_b, reports_b, births_b) = drive(&config, true);
        assert!(!reports_a.is_empty());
        assert_eq!(samples_a, samples_b, "{inheritance:?} physics or food bits");
        assert_eq!(reports_a, reports_b, "{inheritance:?} reports");
        assert_eq!(births_a, births_b, "{inheritance:?} birth brains");
        assert!(
            reports_a
                .iter()
                .any(|r| r.boundary.as_ref().is_some_and(|b| b.slot.is_some())),
            "{inheritance:?}: no replacement happened"
        );
    }
}

#[test]
fn supervisor_newborns_keep_modulator_flags_off() {
    if !gpu_or_skip() {
        return;
    }
    let mut config = config(false, Inheritance::WeightCopy);
    config.brain.motor_fatigue_enabled = false;
    config.brain.klinotaxis_turn_gain_enabled = false;
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("run.db");
    let path = path.to_str().expect("utf-8 path");
    let store = SupervisorStore::create(path, &config).expect("store opens");
    let mut kernel = kernel_for(&config, AGENTS);
    let mut run =
        SupervisorRun::start(&mut kernel, &config, Some(store), true).expect("run starts");
    let mut tick = 0;
    while !run.is_finished() {
        let (physics, food) = advance_fused(&mut kernel, tick);
        tick += SAMPLE;
        run.on_sample(&mut kernel, tick, &physics, food.as_deref())
            .expect("sample is handled");
    }
    run.finish().expect("run finishes");

    let db = Connection::open(path).expect("db opens");
    let mut stmt = db
        .prepare("SELECT config_json FROM supervisor_slot WHERE occupant_id >= 10")
        .expect("query prepares");
    let rows: Vec<String> = stmt
        .query_map([], |row| row.get(0))
        .expect("query runs")
        .collect::<Result<_, _>>()
        .expect("rows read");
    assert!(!rows.is_empty(), "no newborn rows");
    for json in rows {
        let brain: BrainConfig = serde_json::from_str(&json).expect("config parses");
        assert!(!brain.motor_fatigue_enabled);
        assert!(!brain.klinotaxis_turn_gain_enabled);
    }
}
