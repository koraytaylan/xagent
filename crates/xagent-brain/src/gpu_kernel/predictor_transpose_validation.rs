//! Hardware-only transposed predictor storage with contiguous output lanes.
//! Upload and readback conversions occur outside timing; public host APIs and
//! alternative dispatch routes do not support this experimental layout.

use std::{error::Error, time::Instant};

use super::cached_combined_validation::prepare;
use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::global_credit::Pipelines;
use super::predictor_fusion::fuse_inline_predictor;
use super::*;

/// Compare matching FP32 reductions while varying physical matrix layout.
const LANE_WIDTHS: [u32; 4] = [4, 8, 16, 32];
const WARMUP_LANES: u32 = 16;
const PREFETCH: u32 = 8;
const WARMUP: u32 = 256;
const TIMED_CYCLES: u32 = 100;
/// Alternate five pairs for median timing and exact repetition.
const ROUNDS: usize = 5;
/// The captured state places brain storage at this index.
const BRAIN_BUFFER: usize = 8;
const WORD_BYTES: usize = size_of::<f32>();
const WIDTH: u32 = 8;
const HEIGHT: u32 = 6;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn transpose_passes(source: &str, lanes: u32) -> String {
    let output_tile = BRAIN_WORKGROUP_THREADS / lanes;
    let first = source.find("fn coop_predict_and_act(").unwrap();
    let last = first
        + source[first..]
            .find("    // ── Recalled cosine similarities:")
            .unwrap();
    let original = &source[first..last];
    let specialized = original
        .replace("DENSE_INNER_LANES", &format!("{lanes}u"))
        .replace("DENSE_OUTPUT_TILE", &format!("{output_tile}u"));
    let mut changed = specialized
        .replace(
            &format!("let output_in_tile = tid / {lanes}u;"),
            &format!("let output_in_tile = tid % {output_tile}u;"),
        )
        .replace(
            &format!("let lane = tid % {lanes}u;"),
            &format!("let lane = tid / {output_tile}u;"),
        )
        .replace(
            "s_dense_partials[tid + stride]",
            &format!("s_dense_partials[tid + stride * {output_tile}u]"),
        );
    if lanes == LANE_WIDTHS[0] {
        for item in 1..lanes {
            changed = changed.replace(
                &format!("s_dense_partials[base + {item}u]"),
                &format!("s_dense_partials[base + {}u]", item * output_tile),
            );
        }
    }
    for item in 0..PREFETCH {
        let old = format!("O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + input_index_{item}");
        assert_eq!(changed.matches(&old).count(), 2);
        changed = changed.replace(
            &old,
            &format!("O_PREDICTOR_WEIGHTS + input_index_{item} * PREDICTOR_DIMENSION + dim"),
        );
    }
    assert_ne!(changed, original);
    assert!(!changed.contains("tid + stride]"));
    format!("{}{}{}", &source[..first], changed, &source[last..])
}

/// The predictor is square, so the same conversion exports either layout.
fn transpose_state(kernel: &GpuKernel, brain: &mut [u8]) {
    assert_eq!(PREDICTOR_DIMENSION, ENCODED_DIMENSION);
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        let base = agent * kernel.layout.brain_stride
            + kernel.layout.feature_count * ENCODED_DIMENSION
            + ENCODED_DIMENSION;
        for row in 0..PREDICTOR_DIMENSION {
            for column in 0..row {
                let first = (base + row * ENCODED_DIMENSION + column) * WORD_BYTES;
                let second = (base + column * ENCODED_DIMENSION + row) * WORD_BYTES;
                for byte in 0..WORD_BYTES {
                    brain.swap(first + byte, second + byte);
                }
            }
        }
    }
}

fn advance(kernel: &mut GpuKernel, cycle: u32, cycles: u32) {
    assert!(kernel.global_credit_active());
    kernel.dispatch_ticks(
        u64::from(cycle * kernel.brain_tick_stride),
        cycles * kernel.brain_tick_stride,
    );
    kernel.poll_wait();
}

#[test]
#[ignore = "GPU benchmark; run in release mode with --ignored --nocapture"]
fn benchmark_transposed_predictor_storage() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare(WIDTH, HEIGHT, false);
    let base_source = dense_prefetch::prefetch_passes(
        &fuse_inline_predictor(&compose_brain_passes(true)),
        PREFETCH,
    );
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    kernel.global_credit = Some(
        Pipelines::new(
            &kernel,
            &predictor_width::wider_predictor(&base_source, WARMUP_LANES),
            &constants,
        )
        .unwrap(),
    );
    advance(&mut kernel, 0, WARMUP);
    let warm = checkpoint(&kernel);
    let mut transposed = capture_state(&kernel)?[BRAIN_BUFFER].clone();
    transpose_state(&kernel, &mut transposed);
    for lanes in LANE_WIDTHS {
        let source = predictor_width::wider_predictor(&base_source, lanes);
        let candidate = transpose_passes(&source, lanes);
        // Check outside the transform so an identity stub cannot pass parity.
        assert_ne!(candidate, source);
        let first = candidate.find("fn coop_predict_and_act(").unwrap();
        let last = first
            + candidate[first..]
                .find("    // ── Recalled cosine similarities:")
                .unwrap();
        let predictor = &candidate[first..last];
        let output_tile = BRAIN_WORKGROUP_THREADS / lanes;
        assert!(predictor.contains(&format!("let output_in_tile = tid % {output_tile}u;")));
        assert!(predictor.contains(&format!("let lane = tid / {output_tile}u;")));
        let mut pipelines = [source.clone(), candidate]
            .map(|source| Some(Pipelines::new(&kernel, &source, &constants).unwrap()));
        restore(&mut kernel, &warm);
        std::mem::swap(&mut kernel.global_credit, &mut pipelines[0]);
        advance(&mut kernel, WARMUP, TIMED_CYCLES);
        let expected = capture_state(&kernel)?;
        std::mem::swap(&mut kernel.global_credit, &mut pipelines[0]);
        let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..ROUNDS {
            for offset in 0..pipelines.len() {
                let arm = (round + offset) % pipelines.len();
                restore(&mut kernel, &warm);
                if arm == 1 {
                    kernel
                        .queue
                        .write_buffer(&kernel.brain_state_buffer, 0, &transposed);
                    kernel.queue.submit([]);
                    kernel.poll_wait();
                }
                std::mem::swap(&mut kernel.global_credit, &mut pipelines[arm]);
                let start = Instant::now();
                advance(&mut kernel, WARMUP, TIMED_CYCLES);
                timings[arm].push(start.elapsed().as_secs_f64());
                let mut actual = capture_state(&kernel)?;
                if arm == 1 {
                    transpose_state(&kernel, &mut actual[BRAIN_BUFFER]);
                }
                assert_state_equal(&kernel, &expected, &actual);
                std::mem::swap(&mut kernel.global_credit, &mut pipelines[arm]);
            }
        }
        for samples in &mut timings {
            samples.sort_by(f64::total_cmp);
        }
        println!("PREDICTOR_TRANSPOSE_TIMING lanes={lanes} cycles={TIMED_CYCLES} rounds={ROUNDS} reference_seconds={:.9} transposed_seconds={:.9} speedup={:.3} decoded_exact_buffers=13 conversion_outside_timing=true", timings[0][ROUNDS / 2], timings[1][ROUNDS / 2], timings[0][ROUNDS / 2] / timings[1][ROUNDS / 2]);
    }
    Ok(())
}
