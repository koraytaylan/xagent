//! Kernel checkpoint capture in one submit, and bit-identical restore.

use xagent_brain::gpu_kernel::{
    KernelCheckpoint, KernelCpuState, KernelRestoreError, KernelSection,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, FoodKind, WorldConfig};

const AGENTS: usize = 4;
const FOOD_ITEMS: usize = 4;
const TERRAIN_SIDE: usize = 129;
const BIOME_SIDE: usize = 256;
const SPACING: f32 = 5.0;
const AGENT_HEIGHT: f32 = 1.0;
const FOOD_HEIGHT: f32 = 0.35;
const FULL_METER: f32 = 100.0;
const SEED: u64 = 21;
const WARMUP_BATCHES: u64 = 3;
const MORE_BATCHES: u64 = 3;
const NAN_PAYLOAD_BITS: u32 = 0x7fc0_1234;
const DRAWS: u64 = 0x0102_0304_0506_0708;
const CONFIG_INDEX: u32 = 1;
/// Deliberately shorter than `CPU_STATE_LEN`.
const SHORT_CPU_STATE_LEN: usize = 3;
/// Extra batches a restore target runs before it is overwritten.
const DIVERGE_BATCHES: u64 = 2;

fn gpu_or_skip() -> bool {
    let required = std::env::var_os("XAGENT_REQUIRE_GPU").is_some();
    let available = GpuKernel::is_available();
    assert!(
        available || !required,
        "no GPU adapter available but XAGENT_REQUIRE_GPU is set"
    );
    if !available {
        eprintln!("Skipping: no GPU/fallback adapter available");
    }
    available
}

fn count(n: usize) -> u32 {
    u32::try_from(n).expect("count fits u32")
}

fn build_kernel(brain: &BrainConfig) -> GpuKernel {
    let world = WorldConfig::default();
    let mut kernel = GpuKernel::new(count(AGENTS), FOOD_ITEMS, brain, &world);
    kernel.reset_agents_seeded(brain, SEED);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    let items: Vec<(f32, f32, f32)> = (0..FOOD_ITEMS)
        .map(|i| (SPACING * i as f32 + 2.0, FOOD_HEIGHT, 3.0))
        .collect();
    kernel.upload_world(
        &heights,
        &biomes,
        &items,
        &[false; FOOD_ITEMS],
        &[0.0; FOOD_ITEMS],
    );
    let agents: Vec<_> = (0..AGENTS)
        .map(|i| {
            (
                glam::Vec3::new(SPACING * i as f32, AGENT_HEIGHT, 0.0),
                FULL_METER,
                FULL_METER,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agents);
    kernel
}

fn run_batches(kernel: &mut GpuKernel, from: u64, batches: u64) {
    let batch = kernel.kernel_batch_size();
    for b in from..from + batches {
        kernel.dispatch_batch(b * u64::from(batch), batch);
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn kernel_cpu_state_round_trips_bits() {
    let state = KernelCpuState {
        food_kind_draws: DRAWS,
        nourishing_food_kind: Some(FoodKind::Second),
        active_config_index: CONFIG_INDEX,
        freeze_steering_weights: true,
        trunk_lesions: xagent_brain::buffers::TrunkLesions {
            turn_weights: true,
            forward_weights: false,
            memory_blend: true,
            critic: false,
        },
        danger_percept_enabled: true,
        danger_percept_blinded: false,
        brain_tick_probe: true,
        speed_cost_exponent: f32::from_bits(NAN_PAYLOAD_BITS),
    };
    let bytes = state.to_bytes();
    let back = KernelCpuState::from_bytes(&bytes).expect("round trip");
    assert_eq!(back.speed_cost_exponent.to_bits(), NAN_PAYLOAD_BITS);
    assert_eq!(back.food_kind_draws, state.food_kind_draws);
    assert_eq!(back.nourishing_food_kind, state.nourishing_food_kind);
    assert_eq!(back.active_config_index, state.active_config_index);
    assert_eq!(back.trunk_lesions, state.trunk_lesions);
    assert_eq!(back.to_bytes(), bytes);

    let negative_zero = KernelCpuState {
        speed_cost_exponent: -0.0,
        nourishing_food_kind: None,
        ..state
    };
    let back = KernelCpuState::from_bytes(&negative_zero.to_bytes()).expect("round trip");
    assert_eq!(back.speed_cost_exponent.to_bits(), (-0.0_f32).to_bits());
    assert_eq!(back.nourishing_food_kind, None);

    let err = KernelCpuState::from_bytes(&bytes[1..]).expect_err("short");
    assert_eq!(
        err,
        KernelRestoreError::CpuStateLength {
            expected: bytes.len(),
            actual: bytes.len() - 1,
        }
    );
}

#[test]
fn kernel_checkpoint_restore_continues_bit_identically() {
    if !gpu_or_skip() {
        return;
    }
    let brain = BrainConfig::default();
    let mut original = build_kernel(&brain);
    run_batches(&mut original, 0, WARMUP_BATCHES);
    let checkpoint = original.read_checkpoint_blocking();
    assert!(checkpoint.food().is_some());

    let mut restored = build_kernel(&brain);
    restored
        .restore_checkpoint(&checkpoint)
        .expect("restore into fresh kernel");

    run_batches(&mut original, WARMUP_BATCHES, MORE_BATCHES);
    run_batches(&mut restored, WARMUP_BATCHES, MORE_BATCHES);
    let a = original.read_checkpoint_blocking();
    let b = restored.read_checkpoint_blocking();
    assert_eq!(bits(a.physics()), bits(b.physics()));
    assert_eq!(bits(a.food().expect("food")), bits(b.food().expect("food")));
    for i in 0..count(AGENTS) {
        let x = original.read_agent_state(i);
        let y = restored.read_agent_state(i);
        assert_eq!(bits(&x.brain_state), bits(&y.brain_state), "brain {i}");
        assert_eq!(bits(&x.patterns), bits(&y.patterns), "patterns {i}");
    }
}

#[test]
fn kernel_checkpoint_restore_over_dispatched_kernel() {
    if !gpu_or_skip() {
        return;
    }
    let brain = BrainConfig::default();
    let mut original = build_kernel(&brain);
    run_batches(&mut original, 0, WARMUP_BATCHES);
    let checkpoint = original.read_checkpoint_blocking();

    let mut restored = build_kernel(&brain);
    run_batches(&mut restored, 0, WARMUP_BATCHES + DIVERGE_BATCHES);
    restored
        .restore_checkpoint(&checkpoint)
        .expect("restore over live kernel");

    run_batches(&mut original, WARMUP_BATCHES, MORE_BATCHES);
    run_batches(&mut restored, WARMUP_BATCHES, MORE_BATCHES);
    let a = original.read_checkpoint_blocking();
    let b = restored.read_checkpoint_blocking();
    assert_eq!(bits(a.physics()), bits(b.physics()));
    assert_eq!(bits(a.food().expect("food")), bits(b.food().expect("food")));
    for i in 0..count(AGENTS) {
        let x = original.read_agent_state(i);
        let y = restored.read_agent_state(i);
        assert_eq!(bits(&x.brain_state), bits(&y.brain_state), "brain {i}");
        assert_eq!(bits(&x.patterns), bits(&y.patterns), "patterns {i}");
    }
}

/// Environment variable that turns on the packed encoder at kernel build.
const PACKED_ENCODER_ENV: &str = "XAGENT_BRAIN_PACKED_ENCODER";
/// Name of the child test that runs with the packed encoder enabled.
const PACKED_CHILD_TEST: &str = "kernel_checkpoint_packed_restore_child";

/// Restore over a dispatched kernel with the packed encoder (and its cache
/// headers) enabled. The encoder is chosen from the environment at build
/// time, so the check runs in a child process instead of mutating this
/// process's environment while other tests run.
#[test]
fn kernel_checkpoint_restore_over_dispatched_packed_kernel() {
    if !gpu_or_skip() {
        return;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let output = std::process::Command::new(exe)
        .args([
            PACKED_CHILD_TEST,
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(PACKED_ENCODER_ENV, "1")
        .output()
        .expect("spawn packed child test");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "packed child failed: {stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("1 passed"),
        "packed child did not run: {stdout}"
    );
}

/// Child of `kernel_checkpoint_restore_over_dispatched_packed_kernel`; a
/// no-op unless the packed encoder environment variable is set.
#[test]
fn kernel_checkpoint_packed_restore_child() {
    if std::env::var(PACKED_ENCODER_ENV).as_deref() != Ok("1") || !gpu_or_skip() {
        return;
    }
    kernel_checkpoint_restore_over_dispatched_kernel();
}

#[test]
fn kernel_checkpoint_one_submit() {
    if !gpu_or_skip() {
        return;
    }
    let brain = BrainConfig::default();
    let mut kernel = build_kernel(&brain);
    run_batches(&mut kernel, 0, WARMUP_BATCHES);
    let batches = kernel.probe_kernel_batches();

    let before = kernel.queue_submit_count();
    let _ = kernel.read_full_state_blocking();
    assert_eq!(kernel.queue_submit_count(), before + 1);

    let before = kernel.queue_submit_count();
    let _ = kernel.read_checkpoint_blocking();
    assert_eq!(kernel.queue_submit_count(), before + 1);
    assert_eq!(kernel.probe_kernel_batches(), batches);
}

#[test]
fn kernel_checkpoint_refuses_wrong_length() {
    if !gpu_or_skip() {
        return;
    }
    let brain = BrainConfig::default();
    let mut kernel = build_kernel(&brain);
    run_batches(&mut kernel, 0, 1);
    let checkpoint = kernel.read_checkpoint_blocking();
    let before = kernel.read_checkpoint_blocking();

    let mut sections: Vec<KernelSection> = checkpoint.sections().to_vec();
    let target = sections
        .iter_mut()
        .find(|s| s.name == "brain_state")
        .expect("section");
    target.bytes.pop();
    let bad = KernelCheckpoint::new(sections, *checkpoint.cpu());
    let err = kernel.restore_checkpoint(&bad).expect_err("length");
    assert!(matches!(
        err,
        KernelRestoreError::LengthMismatch {
            name: "brain_state",
            ..
        }
    ));

    let err = KernelCpuState::from_bytes(&[0; SHORT_CPU_STATE_LEN]).expect_err("cpu");
    assert!(matches!(err, KernelRestoreError::CpuStateLength { .. }));

    let mut missing: Vec<KernelSection> = checkpoint.sections().to_vec();
    missing.retain(|s| s.name != "pattern");
    let err = kernel
        .restore_checkpoint(&KernelCheckpoint::new(missing, *checkpoint.cpu()))
        .expect_err("missing");
    assert_eq!(err, KernelRestoreError::MissingSection("pattern"));

    let mut duplicated: Vec<KernelSection> = checkpoint.sections().to_vec();
    duplicated.push(duplicated[0].clone());
    let err = kernel
        .restore_checkpoint(&KernelCheckpoint::new(duplicated, *checkpoint.cpu()))
        .expect_err("duplicate");
    assert_eq!(err, KernelRestoreError::DuplicateSection("agent_phys"));

    let after = kernel.read_checkpoint_blocking();
    assert_eq!(before.sections(), after.sections());
}
