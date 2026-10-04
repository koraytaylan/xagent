//! Hardware-only predictor lane-width experiments with FP32 tree reductions.
//! Encoder arithmetic and the number of production dispatches are unchanged.

use std::{error::Error, time::Instant};

use super::cycle_profile::{capture_state, checkpoint, restore};
use super::dense_prefetch_validation::make_pipeline;
use super::predictor_fusion::fuse_inline_predictor;
pub(super) use super::predictor_width::wider_predictor;
use super::rounding_validation::compare_rounding_state;
use super::whitening_validation::prepare_kernel;
use super::*;

/// Four lanes is the current fused predictor; wider arms retain 256 threads.
const LANE_WIDTHS: [u32; 4] = [4, 8, 16, 32];
/// Mature episodic memory precedes the common checkpoint.
const WARMUP_CYCLES: u32 = 256;
/// Five rotated rounds limit arm-order effects.
const ROUNDS: usize = 5;
/// One hundred full cycles amortize submission and completion overhead.
const TIMED_CYCLES: u32 = 100;
type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn advance(kernel: &mut GpuKernel, cycle: u32, cycles: u32) {
    kernel.dispatch_ticks(
        u64::from(cycle * kernel.brain_tick_stride),
        cycles * kernel.brain_tick_stride,
    );
    kernel.poll_wait();
}

#[test]
#[ignore = "requires a GPU; run in release mode with --ignored --nocapture"]
fn benchmark_predictor_lane_widths_with_rounding_report() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    let source = fuse_inline_predictor(&compose_brain_passes(true));
    let prefetch = std::env::var("XAGENT_WIDTH_PREFETCH").as_deref() == Ok("1");
    let source = if prefetch {
        // Match the production prefetch block before varying predictor lanes.
        const PREFETCH_FACTOR: u32 = 8;
        super::dense_prefetch::prefetch_passes(&source, PREFETCH_FACTOR)
    } else {
        source
    };
    let mut pipelines = LANE_WIDTHS
        .map(|lanes| make_pipeline(&kernel, &wider_predictor(&source, lanes), "predictor_width"));
    std::mem::swap(&mut kernel.kernel_pipeline, &mut pipelines[0]);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    advance(&mut kernel, WARMUP_CYCLES, 1);
    let reference = capture_state(&kernel)?;
    std::mem::swap(&mut kernel.kernel_pipeline, &mut pipelines[0]);
    for (arm, lanes) in LANE_WIDTHS.iter().enumerate() {
        restore(&mut kernel, &warm);
        std::mem::swap(&mut kernel.kernel_pipeline, &mut pipelines[arm]);
        advance(&mut kernel, WARMUP_CYCLES, 1);
        let candidate = capture_state(&kernel)?;
        compare_rounding_state(
            &kernel,
            &reference,
            &candidate,
            &format!("predictor_lanes_{lanes}"),
        );
        std::mem::swap(&mut kernel.kernel_pipeline, &mut pipelines[arm]);
    }
    let mut timings: [Vec<f64>; LANE_WIDTHS.len()] = std::array::from_fn(|_| Vec::new());
    for round in 0..ROUNDS {
        for offset in 0..LANE_WIDTHS.len() {
            let arm = (round + offset) % LANE_WIDTHS.len();
            restore(&mut kernel, &warm);
            std::mem::swap(&mut kernel.kernel_pipeline, &mut pipelines[arm]);
            let start = Instant::now();
            advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            std::mem::swap(&mut kernel.kernel_pipeline, &mut pipelines[arm]);
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let baseline = timings[0][ROUNDS / 2];
    for (lanes, samples) in LANE_WIDTHS.iter().zip(&timings) {
        let seconds = samples[ROUNDS / 2];
        println!("PREDICTOR_WIDTH lanes={lanes} prefetch={prefetch} cycles={TIMED_CYCLES} rounds={ROUNDS} seconds={seconds:.9} speedup={:.3}", baseline / seconds);
    }
    Ok(())
}
