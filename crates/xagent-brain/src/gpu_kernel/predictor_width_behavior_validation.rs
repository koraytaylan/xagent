//! Hardware-only trajectory diagnostics for wider FP32 predictor reductions.
//! Both arms use cooperative whitening, fused prediction and prefetch eight;
//! only the predictor's lane width differs. Numerical and behavioral drift
//! are reported, while repeated runs of each arm must match all mutable state.
//! These diagnostics do not assert statistical or behavioral equivalence.

use std::error::Error;

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::dense_prefetch::prefetch_passes;
use super::dense_prefetch_validation::make_pipeline;
use super::predictor_fusion::fuse_inline_predictor;
use super::predictor_width_validation::wider_predictor;
use super::rounding_validation::{
    assert_inactive_agent_unchanged, compare_rounding_state, extract_behavior,
    report_seeded_behavior, BehaviorSummary,
};
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, REFRESH_CYCLES,
};
use super::*;

/// The production exact predictor accumulates four ordered partial sums.
const BASELINE_LANES: u32 = 4;
/// Sixteen lanes shorten each predictor accumulation before a balanced sum.
const CANDIDATE_LANES: u32 = 16;
/// Both arms use the same production dense software-prefetch factor.
const PREFETCH_FACTOR: u32 = 8;
/// Independent brain seeds retain one common seeded flat-world geometry.
const BRAIN_SEEDS: [u64; 3] = [42, 314, 2026];
/// Includes both sides of whitening refreshes, forced death and continuation.
const CONTINUATION_CHUNKS: [u32; 8] = [1, 18, 1, 1, 19, 1, 59, 900];
/// The earlier behavior report exposes drift before the long continuation.
const SHORT_CYCLES: u32 = 100;
/// The final report follows one thousand evolving brain cycles.
const TOTAL_CYCLES: u32 = 1_000;
/// This fixture agent remains inactive without requesting a respawn.
const INACTIVE_AGENT: u32 = 1;
/// Death is forced at the initial state and again at the refresh boundary.
const EXPECTED_FORCED_DEATHS: f32 = 2.0;
/// Every mutable production buffer participates in the repeatability check.
const MUTABLE_BUFFERS: usize = 13;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type Snapshot = Vec<Vec<u8>>;
type BehaviorPair = (u64, BehaviorSummary, BehaviorSummary);

fn prepare() -> (GpuKernel, wgpu::ComputePipeline) {
    let mut kernel = prepare_kernel();
    let fused = fuse_inline_predictor(&compose_brain_passes(true));
    let prefetched = prefetch_passes(&fused, PREFETCH_FACTOR);
    let baseline = wider_predictor(&prefetched, BASELINE_LANES);
    let candidate = wider_predictor(&prefetched, CANDIDATE_LANES);
    kernel.kernel_pipeline = make_pipeline(&kernel, &baseline, "predictor_behavior_width_4");
    let alternate = make_pipeline(&kernel, &candidate, "predictor_behavior_width_16");
    (kernel, alternate)
}

fn run_trajectory(
    kernel: &mut GpuKernel,
    initial_state: &[Vec<u8>],
    repeated: Option<&[Snapshot]>,
    label: &str,
) -> TestResult<Vec<Snapshot>> {
    if let Some(expected) = repeated {
        assert_eq!(expected.len(), CONTINUATION_CHUNKS.len());
    }
    let mut snapshots = Vec::with_capacity(CONTINUATION_CHUNKS.len());
    let mut cycle = 0;
    for (index, cycles) in CONTINUATION_CHUNKS.into_iter().enumerate() {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        kernel.dispatch_ticks(
            u64::from(cycle * kernel.brain_tick_stride),
            cycles * kernel.brain_tick_stride,
        );
        kernel.poll_wait();
        cycle += cycles;
        let state = capture_state(kernel)?;
        assert_eq!(state.len(), MUTABLE_BUFFERS);
        assert_inactive_agent_unchanged(
            kernel,
            initial_state,
            &state,
            INACTIVE_AGENT,
            &format!("{label} cycles={cycle}"),
        );
        if let Some(expected) = repeated {
            assert_state_equal(kernel, &expected[index], &state);
        } else {
            snapshots.push(state);
        }
    }
    assert_eq!(cycle, TOTAL_CYCLES);
    assert!(
        kernel.read_full_state_blocking()[P_DEATH_COUNT] >= EXPECTED_FORCED_DEATHS,
        "{label}: forced death boundaries were not reached"
    );
    Ok(snapshots)
}

fn report_trajectory(
    kernel: &GpuKernel,
    seed: u64,
    reference: &[Snapshot],
    candidate: &[Snapshot],
    short_behavior: &mut Vec<BehaviorPair>,
    final_behavior: &mut Vec<BehaviorPair>,
) {
    assert_eq!(reference.len(), CONTINUATION_CHUNKS.len());
    assert_eq!(candidate.len(), reference.len());
    let mut cycle = 0;
    for ((cycles, reference), candidate) in CONTINUATION_CHUNKS
        .into_iter()
        .zip(reference)
        .zip(candidate)
    {
        cycle += cycles;
        compare_rounding_state(
            kernel,
            reference,
            candidate,
            &format!(
                "predictor_width baseline_lanes={BASELINE_LANES} candidate_lanes={CANDIDATE_LANES} prefetch={PREFETCH_FACTOR} brain_seed={seed} cycles={cycle}"
            ),
        );
        if [SHORT_CYCLES, TOTAL_CYCLES].contains(&cycle) {
            let pair = (
                seed,
                extract_behavior(kernel, reference),
                extract_behavior(kernel, candidate),
            );
            if cycle == SHORT_CYCLES {
                short_behavior.push(pair);
            } else {
                final_behavior.push(pair);
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU; run explicitly in release mode with --ignored --nocapture"]
fn predictor_width_16_prefetch_8_reports_seeded_behavior_and_repeats() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut alternate) = prepare();
    let common_world = checkpoint(&kernel);
    let brain = BrainConfig {
        vision_stride: 1,
        ..BrainConfig::default()
    };
    let mut short_behavior = Vec::new();
    let mut final_behavior = Vec::new();
    for seed in BRAIN_SEEDS {
        restore(&mut kernel, &common_world);
        kernel.reset_agents_seeded(&brain, seed);
        prepare_boundary_scene(&kernel);
        let initial_state = capture_state(&kernel)?;
        let initial = checkpoint(&kernel);
        let baseline_label = format!("predictor_width lanes={BASELINE_LANES} brain_seed={seed}");
        let reference = run_trajectory(&mut kernel, &initial_state, None, &baseline_label)?;
        restore(&mut kernel, &initial);
        run_trajectory(
            &mut kernel,
            &initial_state,
            Some(&reference),
            &baseline_label,
        )?;

        std::mem::swap(&mut kernel.kernel_pipeline, &mut alternate);
        restore(&mut kernel, &initial);
        let candidate_label = format!("predictor_width lanes={CANDIDATE_LANES} brain_seed={seed}");
        let candidate = run_trajectory(&mut kernel, &initial_state, None, &candidate_label)?;
        restore(&mut kernel, &initial);
        run_trajectory(
            &mut kernel,
            &initial_state,
            Some(&candidate),
            &candidate_label,
        )?;
        std::mem::swap(&mut kernel.kernel_pipeline, &mut alternate);

        report_trajectory(
            &kernel,
            seed,
            &reference,
            &candidate,
            &mut short_behavior,
            &mut final_behavior,
        );
        println!(
            "PREDICTOR_WIDTH_BEHAVIOR brain_seed={seed} baseline_lanes={BASELINE_LANES} candidate_lanes={CANDIDATE_LANES} prefetch={PREFETCH_FACTOR} cycles={TOTAL_CYCLES} repeat_exact_buffers={MUTABLE_BUFFERS} inactive_brain_patterns_exact=true forced_death_boundary=true equivalence=not_asserted"
        );
    }
    assert_eq!(short_behavior.len(), BRAIN_SEEDS.len());
    assert_eq!(final_behavior.len(), BRAIN_SEEDS.len());
    report_seeded_behavior(
        &short_behavior,
        "predictor_width_16_prefetch_8_flat_world_brain_seeds_cycles_100",
    );
    report_seeded_behavior(
        &final_behavior,
        "predictor_width_16_prefetch_8_flat_world_brain_seeds_cycles_1000",
    );
    Ok(())
}
