//! Hardware-only FP32 prefetch of independent encoder-credit weight updates.
//! Every arm uses cooperative whitening, fused predictor updates and the
//! production eight-item encoder/predictor prefetch. Candidates additionally
//! load four or eight credit inputs before applying the unchanged per-weight
//! update expression and clamp. Cross-pipeline rounding differences are
//! reported; repeated execution of each arm must remain byte-for-byte stable.
//! No dispatch, barrier or workgroup allocation is added.

use std::{error::Error, fmt::Write, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::dense_prefetch::prefetch_passes;
use super::dense_prefetch_validation::make_pipeline;
use super::predictor_fusion::fuse_inline_predictor;
use super::rounding_validation::{assert_inactive_agent_unchanged, compare_rounding_state};
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, REFRESH_CYCLES,
};
use super::*;

/// Match the current production prefetch for encoder and predictor inputs.
const BASE_DENSE_PREFETCH: u32 = 8;
/// Independent credit input blocks trade outstanding loads against registers.
const CREDIT_PREFETCH_FACTORS: [u32; 2] = [4, 8];
/// Baseline plus four-item and eight-item credit blocks share one checkpoint.
const ARM_COUNT: usize = 3;
/// The first arm retains the production scalar credit-update loop.
const BASELINE_ARM: usize = 0;
/// Stable names associate pipeline allocation, parity and full-cycle timings.
const ARM_NAMES: [&str; ARM_COUNT] = [
    "credit_prefetch_baseline_dense8",
    "credit_prefetch_4_dense8",
    "credit_prefetch_8_dense8",
];
/// Verify both sides of death and refresh boundaries plus one hundred cycles.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Populate the recent-experience window before timing learned credit updates.
const WARMUP_CYCLES: u32 = 256;
/// One hundred full cycles amortize submission and GPU-completion overhead.
const TIMED_CYCLES: u32 = 100;
/// Rotate all arms over five trials and report a median per arm.
const TIMING_ROUNDS: usize = 5;
/// Every mutable simulation buffer, including all learned weights, is checked.
const MUTABLE_BUFFERS: usize = 13;
/// The boundary fixture keeps this agent dead without requesting a respawn.
const INACTIVE_AGENT: u32 = 1;
/// Two explicit deaths exercise both initial and scheduled-refresh resets.
const EXPECTED_FORCED_DEATHS: f32 = 2.0;
/// Stop before learning to detect compiler differences preceding credit work.
const BEFORE_CREDIT_PASS_LIMIT: u32 = 6;
/// The normal schedule runs all seven cooperative brain passes.
const COMPLETE_BRAIN: u32 = 7;
/// Each snapshot word is either an IEEE binary32 value or a 32-bit integer.
const WORD_BYTES: usize = std::mem::size_of::<u32>();
/// Snapshot order is defined by cycle_profile::state_buffers.
const BUFFER_NAMES: [&str; MUTABLE_BUFFERS] = [
    "physics",
    "decisions",
    "food",
    "food_flags",
    "food_grid",
    "agent_grid",
    "collision_scratch",
    "sensory",
    "brain",
    "brain_scratch",
    "patterns",
    "trail_ring",
    "sensory_next",
];

/// The complete original loop leaves threshold and scale computation intact.
const CREDIT_LOOP: &str = r"                for (var j = lane_enc; j < FEATURE_COUNT; j += DENSE_INNER_LANES) {
                    var w = brain_state[brain_base + O_ENC_WEIGHTS + j * ENCODED_DIMENSION + dim] + scale * s_features[j];
                    w = clamp(w, -2.0, 2.0);
                    brain_state[brain_base + O_ENC_WEIGHTS + j * ENCODED_DIMENSION + dim] = w;
                }
";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn credit_prefetch_loop(factor: u32) -> String {
    let mut source = format!(
        "                for (var j = lane_enc; j < FEATURE_COUNT; j += DENSE_INNER_LANES * {factor}u) {{\n"
    );
    for item in 0..factor {
        writeln!(
            source,
            "                    let credit_index_{item} = j + {item}u * DENSE_INNER_LANES;"
        )
        .unwrap();
        writeln!(source, "                    var credit_feature_{item}: f32 = 0.0;\n                    var credit_weight_{item}: f32 = 0.0;").unwrap();
        writeln!(source, "                    if (credit_index_{item} < FEATURE_COUNT) {{\n                        credit_feature_{item} = s_features[credit_index_{item}];\n                        credit_weight_{item} = brain_state[brain_base + O_ENC_WEIGHTS + credit_index_{item} * ENCODED_DIMENSION + dim];\n                    }}").unwrap();
    }
    // Each invocation owns every weight it loads. Independent weight updates
    // retain their source expressions; compiler rounding is measured below.
    for item in 0..factor {
        writeln!(source, "                    if (credit_index_{item} < FEATURE_COUNT) {{\n                        var weight_{item} = credit_weight_{item} + scale * credit_feature_{item};\n                        weight_{item} = clamp(weight_{item}, -2.0, 2.0);\n                        brain_state[brain_base + O_ENC_WEIGHTS + credit_index_{item} * ENCODED_DIMENSION + dim] = weight_{item};\n                    }}").unwrap();
    }
    source.push_str("                }\n");
    source
}

fn credit_prefetch_passes(baseline: &str, factor: u32) -> String {
    assert!(CREDIT_PREFETCH_FACTORS.contains(&factor));
    assert_eq!(baseline.matches(CREDIT_LOOP).count(), 1);
    assert!(baseline.contains("const DENSE_INNER_LANES: u32 = 4u;"));
    let source = baseline.replacen(CREDIT_LOOP, &credit_prefetch_loop(factor), 1);
    for unchanged in ["workgroupBarrier();", "storageBarrier();", "var<workgroup>"] {
        assert_eq!(
            source.matches(unchanged).count(),
            baseline.matches(unchanged).count()
        );
    }
    source
}

struct PipelineArms {
    parked: [Option<wgpu::ComputePipeline>; ARM_COUNT],
    active: usize,
}

impl PipelineArms {
    fn prepare() -> (GpuKernel, Self) {
        let mut kernel = prepare_kernel();
        let fused = fuse_inline_predictor(&compose_brain_passes(true));
        let baseline = prefetch_passes(&fused, BASE_DENSE_PREFETCH);
        kernel.kernel_pipeline = make_pipeline(&kernel, &baseline, ARM_NAMES[BASELINE_ARM]);
        let candidates =
            CREDIT_PREFETCH_FACTORS.map(|factor| credit_prefetch_passes(&baseline, factor));
        let parked = [
            None,
            Some(make_pipeline(&kernel, &candidates[0], ARM_NAMES[1])),
            Some(make_pipeline(&kernel, &candidates[1], ARM_NAMES[2])),
        ];
        (
            kernel,
            Self {
                parked,
                active: BASELINE_ARM,
            },
        )
    }

    fn activate(&mut self, kernel: &mut GpuKernel, arm: usize) {
        if arm != self.active {
            let next = self.parked[arm].take().unwrap();
            let previous = std::mem::replace(&mut kernel.kernel_pipeline, next);
            assert!(self.parked[self.active].replace(previous).is_none());
            self.active = arm;
        }
    }
}

fn advance(kernel: &mut GpuKernel, start_tick: u32, cycles: u32) {
    kernel.dispatch_ticks(u64::from(start_tick), cycles * kernel.brain_tick_stride);
    kernel.poll_wait();
}

fn field_label(kernel: &GpuKernel, buffer: &str, word: usize) -> String {
    if buffer == "physics" {
        let agent = word / PHYS_STRIDE;
        let offset = word % PHYS_STRIDE;
        let field = match offset {
            P_POS_X => "position_x",
            P_POS_Y => "position_y",
            P_POS_Z => "position_z",
            P_PREDICTION_ERROR => "prediction_error",
            P_MOTOR_FWD_OUT => "motor_forward",
            P_MOTOR_TURN_OUT => "motor_turn",
            P_ENERGY => "energy",
            P_INTEGRITY => "integrity",
            _ => "physics_offset",
        };
        return format!("agent={agent} field={field} offset={offset}");
    }
    if buffer == "brain" {
        let agent = word / kernel.layout.brain_stride;
        let offset = word % kernel.layout.brain_stride;
        let encoder_end = kernel.layout.feature_count * ENCODED_DIMENSION;
        if offset < encoder_end {
            return format!(
                "agent={agent} field=encoder_weight feature={} dimension={}",
                offset / ENCODED_DIMENSION,
                offset % ENCODED_DIMENSION,
            );
        }
        let predictor_start = encoder_end + ENCODED_DIMENSION;
        if (predictor_start..fixed_tail_base(kernel.layout.brain_stride)).contains(&offset) {
            let weight = offset - predictor_start;
            return format!(
                "agent={agent} field=predictor_weight output={} input={}",
                weight / ENCODED_DIMENSION,
                weight % ENCODED_DIMENSION,
            );
        }
        return format!("agent={agent} field=brain_offset offset={offset}");
    }
    if buffer == "decisions" {
        return format!(
            "agent={} field=decision_offset offset={}",
            word / DECISION_STRIDE,
            word % DECISION_STRIDE
        );
    }
    format!("word={word}")
}

fn report_first_differences(
    kernel: &GpuKernel,
    reference: &[Vec<u8>],
    actual: &[Vec<u8>],
    label: &str,
) {
    for ((name, reference), actual) in BUFFER_NAMES.iter().zip(reference).zip(actual) {
        if let Some((word, (expected, observed))) = reference
            .as_chunks::<WORD_BYTES>()
            .0
            .iter()
            .zip(actual.as_chunks::<WORD_BYTES>().0)
            .enumerate()
            .find(|(_, (expected, observed))| expected != observed)
        {
            let expected = u32::from_le_bytes(*expected);
            let observed = u32::from_le_bytes(*observed);
            let field = field_label(kernel, name, word);
            println!("CREDIT_PREFETCH_FIRST_DIFFERENCE label={label} buffer={name} {field} reference_bits={expected:08x} candidate_bits={observed:08x} reference_f32={:.9e} candidate_f32={:.9e}", f32::from_bits(expected), f32::from_bits(observed));
        }
    }
}

fn capture_sequence(kernel: &mut GpuKernel) -> TestResult<Vec<Vec<Vec<u8>>>> {
    let mut states = Vec::with_capacity(PARITY_CHUNKS.len());
    let mut cycle = 0;
    for cycles in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        let tick = cycle * kernel.brain_tick_stride;
        advance(kernel, tick, cycles);
        states.push(capture_state(kernel)?);
        cycle += cycles;
    }
    Ok(states)
}

fn mature_single_cycle_diagnostics(
    kernel: &mut GpuKernel,
    pipelines: &mut PipelineArms,
    initial: &super::cycle_profile::Checkpoint,
) -> TestResult {
    restore(kernel, initial);
    pipelines.activate(kernel, BASELINE_ARM);
    advance(kernel, 0, WARMUP_CYCLES);
    let mature = checkpoint(kernel);
    let tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    for pass_limit in [BEFORE_CREDIT_PASS_LIMIT, COMPLETE_BRAIN] {
        // Incomplete-prefix output is diagnostic only and is never continued.
        // Both repeats and every arm start from the same complete checkpoint.
        kernel.probe.kernel_pass_limit = pass_limit;
        let mut reference = None;
        for (arm, name) in ARM_NAMES.iter().enumerate() {
            restore(kernel, &mature);
            pipelines.activate(kernel, arm);
            advance(kernel, tick, 1);
            let actual = capture_state(kernel)?;
            restore(kernel, &mature);
            advance(kernel, tick, 1);
            assert_state_equal(kernel, &actual, &capture_state(kernel)?);
            if arm == BASELINE_ARM {
                reference = Some(actual);
            } else {
                let label = format!("mature_single_cycle_passes_{pass_limit}_{name}");
                let expected = reference.as_ref().unwrap();
                report_first_differences(kernel, expected, &actual, &label);
                compare_rounding_state(kernel, expected, &actual, &label);
            }
        }
    }
    kernel.probe.kernel_pass_limit = COMPLETE_BRAIN;
    Ok(())
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn prefetched_encoder_credit_reports_fp32_drift_and_repeatability() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut pipelines) = PipelineArms::prepare();
    prepare_boundary_scene(&kernel);
    let initial = checkpoint(&kernel);
    let initial_state = capture_state(&kernel)?;
    let deaths_before = kernel.read_full_state_blocking()[P_DEATH_COUNT];
    let expected = capture_sequence(&mut kernel)?;
    assert!(
        kernel.read_full_state_blocking()[P_DEATH_COUNT] >= deaths_before + EXPECTED_FORCED_DEATHS
    );
    for (arm, name) in ARM_NAMES.iter().enumerate() {
        restore(&mut kernel, &initial);
        pipelines.activate(&mut kernel, arm);
        let actual = capture_sequence(&mut kernel)?;
        let mut cycle = 0;
        for ((cycles, expected), actual) in PARITY_CHUNKS.into_iter().zip(&expected).zip(&actual) {
            cycle += cycles;
            let label = format!("{name}_cycles_{cycle}");
            if cycle == 1 {
                report_first_differences(&kernel, expected, actual, &label);
            }
            if arm == BASELINE_ARM {
                assert_state_equal(&kernel, expected, actual);
            }
            compare_rounding_state(&kernel, expected, actual, &label);
            assert_inactive_agent_unchanged(
                &kernel,
                &initial_state,
                actual,
                INACTIVE_AGENT,
                &label,
            );
        }
        restore(&mut kernel, &initial);
        let repeated = capture_sequence(&mut kernel)?;
        for (first, second) in actual.iter().zip(&repeated) {
            assert_state_equal(&kernel, first, second);
        }
        println!("CREDIT_PREFETCH_REPEATABILITY variant={name} cycles={cycle} same_arm_exact_buffers={MUTABLE_BUFFERS} cross_arm_precision=fp32 dead_agent_unchanged=true death_boundary=true");
    }
    mature_single_cycle_diagnostics(&mut kernel, &mut pipelines, &initial)?;
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run in release mode with --ignored --nocapture"]
fn benchmark_prefetched_encoder_credit_against_dense8() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut pipelines) = PipelineArms::prepare();
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    let tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    let mut timings: [Vec<f64>; ARM_COUNT] = std::array::from_fn(|_| Vec::new());
    let mut repeated_states: [Option<Vec<Vec<u8>>>; ARM_COUNT] = std::array::from_fn(|_| None);
    for round in 0..TIMING_ROUNDS {
        let mut states: [Option<Vec<Vec<u8>>>; ARM_COUNT] = std::array::from_fn(|_| None);
        for offset in 0..ARM_COUNT {
            let arm = (round + offset) % ARM_COUNT;
            restore(&mut kernel, &warm);
            pipelines.activate(&mut kernel, arm);
            let start = Instant::now();
            advance(&mut kernel, tick, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            let actual = capture_state(&kernel)?;
            if let Some(first) = &repeated_states[arm] {
                assert_state_equal(&kernel, first, &actual);
            } else {
                repeated_states[arm] = Some(actual.clone());
            }
            states[arm] = Some(actual);
        }
        for (arm, state) in states.iter().enumerate().skip(1) {
            compare_rounding_state(
                &kernel,
                states[BASELINE_ARM].as_ref().unwrap(),
                state.as_ref().unwrap(),
                &format!("timed_round_{round}_{}", ARM_NAMES[arm]),
            );
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let baseline = timings[BASELINE_ARM][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    for (name, samples) in ARM_NAMES.iter().zip(timings) {
        let seconds = samples[TIMING_ROUNDS / 2];
        println!(
            "CREDIT_PREFETCH variant={name} agents={} ticks={ticks} rounds={TIMING_ROUNDS} seconds={seconds:.9} tps={:.3} speedup={:.3} same_arm_exact_buffers={MUTABLE_BUFFERS} cross_arm_precision=fp32",
            kernel.agent_count, f64::from(ticks) / seconds, baseline / seconds,
        );
    }
    Ok(())
}
