//! Hardware-only context gathers prefetched before ordered accumulation.
//! Every arm includes production global encoder credit, cached vision,
//! cooperative whitening, dense prefetch8 and predictor16 explicitly.

use std::{error::Error, fmt::Write, time::Instant};

use super::cached_combined_validation::prepare;
use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::global_credit::Pipelines;
use super::predictor_fusion::fuse_inline_predictor;
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// The control keeps its scalar gather; candidates preload these many terms.
const FACTORS: [u32; 4] = [0, 4, 8, 16];
const ARMS: usize = FACTORS.len();
/// Match the current production dense-loop configuration.
const PREFETCH: u32 = 8;
const PREDICTOR_LANES: u32 = 16;
/// Visit death and whitening boundaries plus mature memory.
const CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const WARMUP: u32 = 256;
const TIMED_CYCLES: u32 = 100;
/// Rotating five rounds supports medians with different first arms.
const ROUNDS: usize = 5;
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
const INACTIVE_AGENT: u32 = 1;

/// Only the ordered context accumulation is replaced, leaving normalization
/// and tanh unchanged; this marker must occur exactly once in the source.
const CONTEXT_LOOP: &str = r"                for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                    let idx = u32(s_recall[k]);
                    let w = context_weight * max(s_recall_similarity[k], 0.0) / total_sim;
                    s_prediction[tid] +=
                        (pattern_buffer[pattern_base + tid * MEMORY_CAP + idx] + encoded_mean) * w;
                }
";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn prefetch_context(source: &str, factor: u32) -> String {
    assert!(FACTORS.contains(&factor));
    if factor == 0 {
        return source.to_owned();
    }
    assert_eq!(source.matches(CONTEXT_LOOP).count(), 1);
    let mut replacement =
        format!("                for (var k = 0u; k < recall_count; k += {factor}u) {{\n");
    for item in 0..factor {
        writeln!(replacement, "                    let recall_index_{item} = k + {item}u;\n                    var recalled_value_{item}: f32 = 0.0;\n                    var recalled_weight_{item}: f32 = 0.0;\n                    if (recall_index_{item} < recall_count) {{\n                        let idx = u32(s_recall[recall_index_{item}]);\n                        recalled_value_{item} = pattern_buffer[pattern_base + tid * MEMORY_CAP + idx] + encoded_mean;\n                        recalled_weight_{item} = context_weight * max(s_recall_similarity[recall_index_{item}], 0.0) / total_sim;\n                    }}").unwrap();
    }
    for item in 0..factor {
        writeln!(replacement, "                    if (recall_index_{item} < recall_count) {{\n                        s_prediction[tid] += recalled_value_{item} * recalled_weight_{item};\n                    }}").unwrap();
    }
    replacement.push_str("                }\n");
    let result = source.replacen(CONTEXT_LOOP, &replacement, 1);
    for marker in ["workgroupBarrier();", "storageBarrier();", "var<workgroup>"] {
        assert_eq!(
            result.matches(marker).count(),
            source.matches(marker).count()
        );
    }
    result
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
    std::array::from_fn(|arm| {
        Some(Pipelines::new(kernel, &prefetch_context(&source, FACTORS[arm]), &constants).unwrap())
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
    let mut states = Vec::new();
    for cycles in CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, cycle, cycles);
        cycle += cycles;
        states.push(capture_state(kernel)?);
    }
    Ok(states)
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn context_prefetch_preserves_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare(width, height, true);
        let mut pipelines = arms(&kernel);
        let initial = checkpoint(&kernel);
        let initial_state = capture_state(&kernel)?;
        std::mem::swap(&mut kernel.global_credit, &mut pipelines[0]);
        let expected = trajectory(&mut kernel)?;
        std::mem::swap(&mut kernel.global_credit, &mut pipelines[0]);
        for (arm, pipeline) in pipelines.iter_mut().enumerate().skip(1) {
            restore(&mut kernel, &initial);
            std::mem::swap(&mut kernel.global_credit, pipeline);
            let actual = trajectory(&mut kernel)?;
            for (expected, actual) in expected.iter().zip(&actual) {
                assert_state_equal(&kernel, expected, actual);
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial_state,
                    actual,
                    INACTIVE_AGENT,
                    "context prefetch",
                );
            }
            std::mem::swap(&mut kernel.global_credit, pipeline);
            println!("CONTEXT_PREFETCH_PARITY width={width} height={height} factor={} cycles=100 exact_buffers=13 death_refresh=true", FACTORS[arm]);
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run in release mode with --ignored --nocapture"]
fn benchmark_context_prefetch_with_global_credit() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare(FIELDS[0].0, FIELDS[0].1, false);
    let mut pipelines = arms(&kernel);
    std::mem::swap(&mut kernel.global_credit, &mut pipelines[0]);
    advance(&mut kernel, 0, WARMUP);
    let warm = checkpoint(&kernel);
    advance(&mut kernel, WARMUP, TIMED_CYCLES);
    let expected = capture_state(&kernel)?;
    std::mem::swap(&mut kernel.global_credit, &mut pipelines[0]);
    let mut timings: [Vec<f64>; ARMS] = std::array::from_fn(|_| Vec::new());
    for round in 0..ROUNDS {
        for offset in 0..ARMS {
            let arm = (round + offset) % ARMS;
            restore(&mut kernel, &warm);
            std::mem::swap(&mut kernel.global_credit, &mut pipelines[arm]);
            let start = Instant::now();
            advance(&mut kernel, WARMUP, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
            std::mem::swap(&mut kernel.global_credit, &mut pipelines[arm]);
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let reference = timings[0][ROUNDS / 2];
    for (factor, samples) in FACTORS.into_iter().zip(timings) {
        let seconds = samples[ROUNDS / 2];
        println!("CONTEXT_PREFETCH_TIMING factor={factor} cycles={TIMED_CYCLES} rounds={ROUNDS} seconds={seconds:.9} speedup={:.3} exact_buffers=13 global_credit=true", reference / seconds);
    }
    Ok(())
}
