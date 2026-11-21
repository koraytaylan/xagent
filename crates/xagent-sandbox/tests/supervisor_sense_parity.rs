//! Byte-level regression baseline for seeded supervisor runs.
//!
//! Each scenario drives a seeded ten-agent supervisor run on one GPU kernel,
//! hashes its physics at every sample, the brain of every newborn and the slot,
//! occupant id and decision row (tick, reason, parent, archive length, best
//! outcome) of every boundary report with FNV-1a 64-bit, and compares
//! the hash with the line of the same name in
//! `tests/fixtures/supervisor_sense_parity.golden`.
//!
//! Scenarios: `supervisor_raw_vision` (visual cortex off) and
//! `supervisor_cortex` (visual cortex on).
//!
//! Modes: compare is the default. With
//! `XAGENT_RECORD_SUPERVISOR_SENSE_PARITY=1` a scenario rewrites only its own
//! golden line (creating the file when absent) and does not compare.
//!
//! The GPU hashes are specific to the host and adapter that recorded them, so
//! the GPU scenarios are ignored by default and run with `--include-ignored`.

use std::fs;
use std::path::{Path, PathBuf};

use xagent_brain::buffers::{PHYS_STRIDE, P_HARMFUL_MEAL_COUNT};
use xagent_brain::GpuKernel;
use xagent_sandbox::supervisor::{ParentRef, ReplaceReason};
use xagent_sandbox::supervisor_run::{BoundaryReport, SupervisorRun};
use xagent_sandbox::world::WorldState;
use xagent_shared::{BrainConfig, EvolutionMode, FullConfig, SupervisorConfig};

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Environment variable that switches the test into record mode.
const RECORD_VARIABLE: &str = "XAGENT_RECORD_SUPERVISOR_SENSE_PARITY";
/// Golden file, relative to the crate root.
const GOLDEN_PATH: &str = "tests/fixtures/supervisor_sense_parity.golden";

/// Scenario name of the raw-vision run.
const RAW_VISION: &str = "supervisor_raw_vision";
/// Scenario name of the visual-cortex run.
const CORTEX: &str = "supervisor_cortex";

/// Agents in every run.
const AGENTS: usize = 10;
/// World seed of every run.
const SEED: u64 = 5;
/// Boundary interval in ticks.
const BOUNDARY_TICKS: u64 = 1_000;
/// Minimum age before an underperformer can be replaced.
const MIN_AGE_TICKS: u64 = 2_000;
/// Score window in ticks.
const WINDOW_TICKS: u64 = 1_000;
/// Season period in ticks.
const SEASON_TICKS: u64 = 3_000;
/// Archive capacity of a run.
const ARCHIVE: usize = 8;
/// Agent-tick budget of a run.
const BUDGET: u64 = 30_000;
/// Ticks per dispatch, one sample period.
const SAMPLE: u32 = 100;

/// Physics slots per agent that are hashed.
const PRE_SCENARIO_PHYSICS_SLOTS: usize = P_HARMFUL_MEAL_COUNT;

/// Folds `bytes` into an FNV-1a 64-bit hash state.
fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Hashes `bytes` from the FNV-1a offset basis.
fn hash_bytes(bytes: &[u8]) -> u64 {
    fnv1a(FNV_OFFSET_BASIS, bytes)
}

/// Little-endian bytes of `values`.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn crate_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn record_mode() -> bool {
    std::env::var(RECORD_VARIABLE).is_ok_and(|value| value == "1")
}

/// Fail-closed adapter guard: returns false only when no adapter exists and
/// `XAGENT_REQUIRE_GPU` is unset; panics when it is set without an adapter.
fn require_gpu() -> bool {
    if GpuKernel::is_available() {
        return true;
    }
    assert!(
        std::env::var_os("XAGENT_REQUIRE_GPU").is_none(),
        "XAGENT_REQUIRE_GPU is set but no GPU adapter is available"
    );
    println!("skipping: no GPU adapter");
    false
}

/// In record mode replaces or appends the scenario's golden line, creating the
/// file when absent; otherwise asserts the hash equals the golden line.
fn check_golden(scenario: &str, hash: u64) {
    let path = crate_path(GOLDEN_PATH);
    let line = format!("{scenario} {hash:016x}");
    if record_mode() {
        let existing = fs::read_to_string(&path).unwrap_or_default();
        let mut lines: Vec<String> = existing.lines().map(str::to_owned).collect();
        match lines
            .iter_mut()
            .find(|l| l.split_whitespace().next() == Some(scenario))
        {
            Some(slot) => *slot = line,
            None => lines.push(line),
        }
        let parent = path.parent().expect("golden path has a parent directory");
        fs::create_dir_all(parent).expect("create the fixtures directory");
        fs::write(&path, lines.join("\n") + "\n").expect("write the golden file");
        println!("recorded {scenario} {hash:016x}");
        return;
    }
    let golden = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "scenario {scenario} ({hash:016x}): golden file {} is unreadable: {e}",
            path.display()
        )
    });
    let recorded = golden
        .lines()
        .find_map(|l| {
            let mut parts = l.split_whitespace();
            (parts.next() == Some(scenario)).then(|| parts.next().unwrap_or("").to_owned())
        })
        .unwrap_or_else(|| {
            panic!("golden file has no line for scenario {scenario} (got {hash:016x})")
        });
    assert_eq!(
        recorded,
        format!("{hash:016x}"),
        "scenario {scenario}: golden {recorded} != actual {hash:016x}"
    );
}

fn config(visual_cortex_enabled: bool) -> FullConfig {
    let mut config = FullConfig::default();
    config.world.seed = SEED;
    config.governor.evolution_mode = EvolutionMode::Supervisor;
    config.governor.population_size = AGENTS;
    config.brain = BrainConfig {
        visual_cortex_enabled,
        ..Default::default()
    };
    let inheritance = config.governor.supervisor.inheritance;
    config.governor.supervisor = SupervisorConfig {
        boundary_interval_ticks: BOUNDARY_TICKS,
        min_age_ticks: MIN_AGE_TICKS,
        score_window_ticks: WINDOW_TICKS,
        archive_capacity: ARCHIVE,
        season_period_ticks: SEASON_TICKS,
        inheritance,
        agent_tick_budget: BUDGET,
    };
    config
}

/// Absent-value marker for optional decision fields.
const NONE_CODE: u64 = u64::MAX;

/// Decision row of a boundary beyond its slot and occupant id, as
/// little-endian `u64` values: tick, reason code, parent kind and value,
/// archive length and best-outcome bits.
fn decision_row_bytes(boundary: &BoundaryReport) -> Vec<u8> {
    let reason = match boundary.reason {
        None => NONE_CODE,
        Some(ReplaceReason::Death) => 0,
        Some(ReplaceReason::Underperformer) => 1,
    };
    let to_u64 = |v: usize| u64::try_from(v).expect("index fits u64");
    let (parent_kind, parent_value) = match boundary.parent {
        None => (NONE_CODE, NONE_CODE),
        Some(ParentRef::ArchiveEntry(id)) => (0, id),
        Some(ParentRef::PeerBirth(slot)) => (1, to_u64(slot)),
        Some(ParentRef::PeerLearned(slot)) => (2, to_u64(slot)),
    };
    let best = boundary
        .best_outcome
        .map_or(NONE_CODE, |v| u64::from(v.to_bits()));
    [
        boundary.tick,
        reason,
        parent_kind,
        parent_value,
        to_u64(boundary.archive_len),
        best,
    ]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect()
}

/// Hash of one seeded supervisor run and its replacement count.
fn scenario_hash(visual_cortex_enabled: bool) -> (u64, usize) {
    let config = config(visual_cortex_enabled);
    assert!(
        !config.world.food_kinds_enabled,
        "food kinds must be off by default"
    );
    let world = WorldState::new(config.world.clone());
    let mut kernel = GpuKernel::new(
        u32::try_from(AGENTS).expect("agent count fits"),
        world.food_items.len(),
        &config.brain,
        &config.world,
    );
    let mut run = SupervisorRun::start(&mut kernel, &config, None, false).expect("run starts");
    let mut physics_bytes = Vec::new();
    let mut birth_bytes = Vec::new();
    let mut boundary_bytes = Vec::new();
    let mut replacements = 0;
    let mut tick = 0;
    while !run.is_finished() {
        kernel.dispatch_batch(tick, SAMPLE);
        while !kernel.try_collect_state_snapshot() {
            std::hint::spin_loop();
        }
        let physics = kernel.cached_state().to_vec();
        let food = kernel.cached_food_state().map(<[f32]>::to_vec);
        tick += u64::from(SAMPLE);

        let full = kernel.read_full_state_blocking();
        assert_eq!(
            full.len(),
            AGENTS * PHYS_STRIDE,
            "physics buffer must hold every agent at PHYS_STRIDE"
        );
        for agent in full.as_chunks::<PHYS_STRIDE>().0 {
            physics_bytes.extend(f32_bytes(&agent[..PRE_SCENARIO_PHYSICS_SLOTS]));
        }

        let report = run
            .on_sample(&mut kernel, tick, &physics, food.as_deref())
            .expect("sample is handled");
        if let Some(boundary) = report.boundary {
            let slot = boundary
                .slot
                .map_or(NONE_CODE, |s| u64::try_from(s).expect("slot fits u64"));
            let occupant = boundary.occupant_id.unwrap_or(NONE_CODE);
            boundary_bytes.extend(slot.to_le_bytes());
            boundary_bytes.extend(occupant.to_le_bytes());
            boundary_bytes.extend(decision_row_bytes(&boundary));
            if let Some(slot) = boundary.slot {
                replacements += 1;
                let birth = run.birth_state(slot).expect("slot exists");
                birth_bytes.extend(f32_bytes(&birth.brain_state));
            }
        }
    }
    run.finish().expect("run finishes");
    let mut bytes = physics_bytes;
    bytes.extend(birth_bytes);
    bytes.extend(boundary_bytes);
    (hash_bytes(&bytes), replacements)
}

fn run_golden(scenario: &str, visual_cortex_enabled: bool) {
    if !require_gpu() {
        return;
    }
    let (hash, replacements) = scenario_hash(visual_cortex_enabled);
    assert!(replacements >= 1, "scenario {scenario} had no replacement");
    check_golden(scenario, hash);
}

/// The seeded raw-vision supervisor run hashes to its golden line.
#[test]
#[ignore = "requires a GPU; run explicitly with --include-ignored"]
fn supervisor_raw_vision_matches_golden() {
    run_golden(RAW_VISION, false);
}

/// The seeded visual-cortex supervisor run hashes to its golden line.
#[test]
#[ignore = "requires a GPU; run explicitly with --include-ignored"]
fn supervisor_cortex_matches_golden() {
    run_golden(CORTEX, true);
}

/// The raw-vision scenario hashes identically on two runs.
#[test]
#[ignore = "requires a GPU; run explicitly with --include-ignored"]
fn supervisor_sense_parity_is_reproducible() {
    if !require_gpu() {
        return;
    }
    let (first, _) = scenario_hash(false);
    let (second, _) = scenario_hash(false);
    assert_eq!(first, second, "{first:016x} != {second:016x}");
}

/// Every golden line is `name hex16` and both scenarios are present.
#[test]
fn golden_file_parses() {
    let path = crate_path(GOLDEN_PATH);
    let golden = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("golden file {} is unreadable: {e}", path.display()));
    let mut names = Vec::new();
    for line in golden.lines().filter(|l| !l.trim().is_empty()) {
        let parts: Vec<&str> = line.split(' ').collect();
        assert_eq!(parts.len(), 2, "line {line:?} is not `name hex16`");
        assert!(
            parts[1].len() == 16
                && parts[1]
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "line {line:?} has no 16-digit lowercase hex value"
        );
        names.push(parts[0]);
    }
    for scenario in [RAW_VISION, CORTEX] {
        assert!(names.contains(&scenario), "golden file lacks {scenario}");
    }
}
