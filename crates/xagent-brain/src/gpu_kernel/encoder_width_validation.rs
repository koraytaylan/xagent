//! Hardware-only encoder lane-layout measurements with complete-cycle timing.
//! Candidates change FP32 association; state differences are reported and
//! repetition is checked independently for each arm.

use std::{error::Error, time::Instant};

use super::cached_combined_validation::prepare;
use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::global_credit::Pipelines;
use super::predictor_fusion::fuse_inline_predictor;
use super::rounding_validation::compare_rounding_state;
use super::*;

/// Four lanes retains production arithmetic; alternatives divide 256 threads.
const LANES: [u32; 4] = [4, 2, 8, 16];
const ARMS: usize = LANES.len();
/// Keep the measured production predictor configuration in every arm.
const PREFETCH: u32 = 8;
const PREDICTOR_LANES: u32 = 16;
/// Mature memory precedes the shared starting state.
const WARMUP: u32 = 256;
const TIMED_CYCLES: u32 = 100;
/// Rotating five trials yields a median without a fixed first arm.
const ROUNDS: usize = 5;
const WIDTH: u32 = 8;
const HEIGHT: u32 = 6;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn encoder_lanes(source: &str, lanes: u32) -> String {
    assert!(LANES.contains(&lanes));
    if lanes == LANES[0] {
        return source.to_owned();
    }
    let first = source.find("fn coop_encode(").unwrap();
    let last = first + source[first..].find("fn coop_habituate_homeo(").unwrap();
    let block = &source[first..last];
    let reduction_first = block.find("        // Lane-major scratch").unwrap();
    let reduction_last = block
        .find("        workgroupBarrier();   // REQUIRED")
        .unwrap();
    let output_tile = BRAIN_WORKGROUP_THREADS / lanes;
    let reduction = format!(
        r"        for (var stride = {lanes}u / 2u; stride > 0u; stride /= 2u) {{
            if (lane < stride) {{
                s_dense_partials[tid] += s_dense_partials[tid + stride * {output_tile}u];
            }}
            workgroupBarrier();
        }}
        if (lane == 0u) {{
            s_encoded[dim] = fast_tanh(s_dense_partials[tid]);
        }}
"
    );
    let changed = format!(
        "{}{}{}",
        &block[..reduction_first],
        reduction,
        &block[reduction_last..]
    )
    .replace("DENSE_OUTPUT_TILE", &format!("{output_tile}u"))
    .replace("DENSE_INNER_LANES", &format!("{lanes}u"))
    .replace("stride-four", "strided");
    format!("{}{}{}", &source[..first], changed, &source[last..])
}

fn arms(kernel: &GpuKernel) -> [Option<Pipelines>; ARMS] {
    let source = predictor_width::wider_predictor(
        &dense_prefetch::prefetch_passes(
            &fuse_inline_predictor(&compose_brain_passes(true)),
            PREFETCH,
        ),
        PREDICTOR_LANES,
    );
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    LANES.map(|lanes| {
        let candidate = encoder_lanes(&source, lanes);
        if lanes == LANES[0] {
            assert_eq!(candidate, source);
        } else {
            // An identity transform would otherwise satisfy same-arm replay.
            assert_ne!(candidate, source);
            let first = candidate.find("fn coop_encode(").unwrap();
            let last = first + candidate[first..].find("fn coop_habituate_homeo(").unwrap();
            let encode = &candidate[first..last];
            let output_tile = BRAIN_WORKGROUP_THREADS / lanes;
            assert!(encode.contains(&format!("let output_in_tile = tid % {output_tile}u;")));
            assert!(encode.contains(&format!("let lane = tid / {output_tile}u;")));
        }
        Some(Pipelines::new(kernel, &candidate, &constants).unwrap())
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

#[test]
#[ignore = "GPU benchmark; run in release mode with --ignored --nocapture"]
fn benchmark_encoder_lanes_with_global_credit() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare(WIDTH, HEIGHT, false);
    let mut pipelines = arms(&kernel);
    std::mem::swap(&mut kernel.global_credit, &mut pipelines[0]);
    advance(&mut kernel, 0, WARMUP);
    let warm = checkpoint(&kernel);
    std::mem::swap(&mut kernel.global_credit, &mut pipelines[0]);
    let mut timings: [Vec<f64>; ARMS] = std::array::from_fn(|_| Vec::new());
    let mut expected: Vec<Vec<Vec<u8>>> = Vec::new();
    for round in 0..ROUNDS {
        for offset in 0..ARMS {
            let arm = (round + offset) % ARMS;
            restore(&mut kernel, &warm);
            std::mem::swap(&mut kernel.global_credit, &mut pipelines[arm]);
            let start = Instant::now();
            advance(&mut kernel, WARMUP, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            let actual = capture_state(&kernel)?;
            if round == 0 {
                if arm > 0 {
                    compare_rounding_state(
                        &kernel,
                        &expected[0],
                        &actual,
                        &format!("encoder_lanes_{}", LANES[arm]),
                    );
                }
                expected.push(actual);
            } else {
                assert_state_equal(&kernel, &expected[arm], &actual);
            }
            std::mem::swap(&mut kernel.global_credit, &mut pipelines[arm]);
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let reference = timings[0][ROUNDS / 2];
    for (lanes, samples) in LANES.into_iter().zip(timings) {
        let seconds = samples[ROUNDS / 2];
        println!("ENCODER_LANES_TIMING lanes={lanes} cycles={TIMED_CYCLES} rounds={ROUNDS} seconds={seconds:.9} speedup={:.3} repeat_exact_buffers=13 global_credit=true", reference / seconds);
    }
    Ok(())
}
