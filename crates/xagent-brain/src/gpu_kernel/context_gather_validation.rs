//! Hardware-only parallel gathering of recalled context into existing scratch.
//! All arms explicitly use production global encoder credit, cached vision,
//! cooperative whitening, dense prefetch8 and the sixteen-lane predictor.
//! Raw pattern values are staged before the original scalar ordered blend.

use std::{error::Error, time::Instant};

use super::cached_combined_validation::prepare;
use super::context_gather::gather_context;
use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::global_credit::Pipelines;
use super::predictor_fusion::fuse_inline_predictor;
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// Zero retains the ordered scalar loop; the others select gather team sizes.
const GATHER_LANES: [u32; 3] = [0, 16, 8];
const ARM_COUNT: usize = GATHER_LANES.len();
const ARM_NAMES: [&str; ARM_COUNT] = ["scalar_context", "gather16_context", "gather8_context"];
/// Repeat the candidate from identical checkpoints to expose scratch races.
const CANDIDATE_REPLAYS: usize = 2;
/// Match the current optimized production dense-loop configuration.
const PREFETCH: u32 = 8;
const PREDICTOR_LANES: u32 = 16;
/// Checkpoints surround forced deaths and both whitening refresh boundaries.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Populate episodic memory before measuring either arm.
const WARMUP_CYCLES: u32 = 256;
/// A hundred evolving cycles amortize submission and completion overhead.
const TIMED_CYCLES: u32 = 100;
/// Rotate the first arm and report an odd-sample median.
const TIMING_ROUNDS: usize = 5;
/// The odd retina also exercises non-default brain and sensory layouts.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// This agent remains inactive without a pending respawn.
const INACTIVE_AGENT: u32 = 1;
/// All mutable simulation storage buffers participate in every comparison.
const MUTABLE_BUFFERS: usize = 13;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn pipelines(kernel: &GpuKernel) -> [Option<Pipelines>; ARM_COUNT] {
    let original = predictor_width::wider_predictor(
        &dense_prefetch::prefetch_passes(
            &fuse_inline_predictor(&compose_brain_passes(true)),
            PREFETCH,
        ),
        PREDICTOR_LANES,
    );
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    GATHER_LANES.map(|lanes| {
        let source = if lanes == 0 {
            original.clone()
        } else {
            gather_context(&original, lanes)
        };
        if lanes != 0 {
            assert_ne!(
                source, original,
                "the candidate must replace the context loop"
            );
        }
        Some(Pipelines::new(kernel, &source, &constants).unwrap())
    })
}

fn advance(kernel: &mut GpuKernel, cycle: u32, cycles: u32) {
    assert!(kernel.global_credit_active());
    kernel.dispatch_ticks(
        u64::from(cycle * kernel.brain_tick_stride),
        cycles * kernel.brain_tick_stride,
    );
    kernel.poll_wait();
}

fn trajectory(kernel: &mut GpuKernel) -> TestResult<Vec<State>> {
    let mut cycle = 0;
    let mut states = Vec::with_capacity(PARITY_CHUNKS.len());
    for cycles in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, cycle, cycles);
        cycle += cycles;
        let state = capture_state(kernel)?;
        assert_eq!(state.len(), MUTABLE_BUFFERS);
        states.push(state);
    }
    Ok(states)
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn context_gather_preserves_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare(width, height, true);
        let mut variants = pipelines(&kernel);
        let saved = checkpoint(&kernel);
        let initial = capture_state(&kernel)?;
        std::mem::swap(&mut kernel.global_credit, &mut variants[0]);
        let expected = trajectory(&mut kernel)?;
        std::mem::swap(&mut kernel.global_credit, &mut variants[0]);
        for (arm, pipeline) in variants.iter_mut().enumerate().skip(1) {
            // A second replay also detects nondeterministic scratch access.
            for _ in 0..CANDIDATE_REPLAYS {
                restore(&mut kernel, &saved);
                std::mem::swap(&mut kernel.global_credit, pipeline);
                let actual = trajectory(&mut kernel)?;
                for (expected, actual) in expected.iter().zip(&actual) {
                    assert_state_equal(&kernel, expected, actual);
                    assert_inactive_agent_unchanged(
                        &kernel,
                        &initial,
                        actual,
                        INACTIVE_AGENT,
                        "context gather",
                    );
                }
                std::mem::swap(&mut kernel.global_credit, pipeline);
            }
            println!("CONTEXT_GATHER_PARITY width={width} height={height} lanes={} cycles=100 exact_buffers={MUTABLE_BUFFERS} death_refresh=true candidate_replays={CANDIDATE_REPLAYS}", GATHER_LANES[arm]);
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run in release mode with --ignored --nocapture"]
fn benchmark_context_gather_with_global_credit() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare(FIELDS[0].0, FIELDS[0].1, false);
    let mut variants = pipelines(&kernel);
    std::mem::swap(&mut kernel.global_credit, &mut variants[0]);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let expected = capture_state(&kernel)?;
    assert_eq!(expected.len(), MUTABLE_BUFFERS);
    std::mem::swap(&mut kernel.global_credit, &mut variants[0]);
    for pipeline in variants.iter_mut().skip(1) {
        restore(&mut kernel, &warm);
        std::mem::swap(&mut kernel.global_credit, pipeline);
        advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
        assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
        std::mem::swap(&mut kernel.global_credit, pipeline);
    }
    let mut timings: [Vec<f64>; ARM_COUNT] = std::array::from_fn(|_| Vec::new());
    for round in 0..TIMING_ROUNDS {
        for offset in 0..ARM_COUNT {
            let arm = (round + offset) % ARM_COUNT;
            restore(&mut kernel, &warm);
            std::mem::swap(&mut kernel.global_credit, &mut variants[arm]);
            let start = Instant::now();
            advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
            std::mem::swap(&mut kernel.global_credit, &mut variants[arm]);
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let reference = timings[0][TIMING_ROUNDS / 2];
    for (name, samples) in ARM_NAMES.into_iter().zip(timings) {
        let seconds = samples[TIMING_ROUNDS / 2];
        println!("CONTEXT_GATHER_TIMING arm={name} warmup_cycles={WARMUP_CYCLES} cycles={TIMED_CYCLES} rounds={TIMING_ROUNDS} seconds={seconds:.9} speedup={:.3} exact_buffers={MUTABLE_BUFFERS} global_credit=true extra_workgroup_bytes=0", reference / seconds);
    }
    Ok(())
}
