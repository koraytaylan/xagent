//! Test-only reuse of recall cosines for memory reinforcement. The memory key,
//! pattern states, norms and active flags are unchanged between these phases.
//! Reusing the earlier dot and norm changes FP32 association, not the formula.
//! A third arm moves learning's original norm tree and two-lane pattern dots
//! into recall, retaining the cache through prediction without new storage.

use std::{error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::dense_prefetch::prefetch_passes;
use super::dense_prefetch_validation::make_pipeline;
use super::predictor_fusion::fuse_inline_predictor;
use super::predictor_width::wider_predictor;
use super::rounding_validation::{assert_inactive_agent_unchanged, compare_rounding_state};
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, REFRESH_CYCLES,
};
use super::*;

/// Explicit current optimized brain settings, independent of process flags.
const PREFETCH: u32 = 8;
const PREDICTOR_LANES: u32 = 16;
/// Mature memory is reached before paired throughput measurements.
const WARMUP_CYCLES: u32 = 256;
/// Five rotated three-arm trials, each advancing one hundred complete cycles.
const ROUNDS: usize = 5;
const TIMED_CYCLES: u32 = 100;
/// Compare current production, serial recall reuse, and cooperative recall reuse.
const ARM_COUNT: usize = 3;
/// Labels identify the same composed sources in parity and timing output.
const ARM_NAMES: [&str; ARM_COUNT] = [
    "optimized_baseline",
    "reuse_serial_recall",
    "reuse_cooperative_recall",
];
/// Death and refresh boundaries plus a longer continuation.
const CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// This fixture agent remains dead without a requested reset.
const INACTIVE_AGENT: u32 = 1;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

pub(super) fn cached_passes(source: &str) -> String {
    let recall_start = source
        .find("fn coop_recall_score(agent_id: u32, tid: u32) {")
        .unwrap();
    let recall_end = recall_start + source[recall_start..].find("\n}\n").unwrap();
    // Argmin scratch has no consumer until eviction, which overwrites it.
    let mut cached = format!("{}\n    if (tid < MEMORY_CAP) {{\n        s_argmin_val[tid] = s_similarities[tid];\n    }}{}", &source[..recall_end], &source[recall_end..]);
    let start = cached
        .find("    // ── 7c. Memory reinforcement + episodic credit:")
        .unwrap();
    let end = start + cached[start..].find("        // Episodic credit:").unwrap();
    let replacement = r"    // Recall saved the same cosine before top-K reordered similarities.
    let pattern = tid;
    if (tid < MEMORY_CAP) {
        let sim = s_argmin_val[tid];
        if (sim > 0.3 && pattern_buffer[pattern_base + O_PAT_ACTIVE + pattern] >= 0.5) {
            pattern_buffer[pattern_base + O_PAT_REINF + pattern] += sim * learning_rate * (1.0 - s_pred_td[S_PRED_ERROR]);
            pattern_buffer[pattern_base + O_PAT_REINF + pattern] = clamp(
                pattern_buffer[pattern_base + O_PAT_REINF + pattern], 0.0, 20.0);
        }

";
    cached.replace_range(start..end, replacement);
    assert_eq!(
        cached.matches("var<workgroup>").count(),
        source.matches("var<workgroup>").count()
    );
    assert!(cached.contains("s_argmin_val[tid] = s_similarities[tid];"));
    cached
}

pub(super) fn cooperative_cached_passes(source: &str) -> String {
    const NORM_MARKER: &str = "    // ── Compute memory-key norm ONCE";
    const REINFORCEMENT_MARKER: &str = "    // ── 7c. Memory reinforcement + episodic credit:";
    const DOT_MARKER: &str = "    let pattern = tid % MEMORY_CAP;";
    const NORMALIZATION_MARKER: &str = "    // Lane 0 (tid < MEMORY_CAP):";
    const RECALL_DEFINITION: &str = "fn coop_recall_score(agent_id: u32, tid: u32) {";
    for marker in [
        NORM_MARKER,
        REINFORCEMENT_MARKER,
        DOT_MARKER,
        NORMALIZATION_MARKER,
        RECALL_DEFINITION,
    ] {
        assert_eq!(
            source.matches(marker).count(),
            1,
            "unique source marker {marker}"
        );
    }
    let norm_start = source.find(NORM_MARKER).unwrap();
    let norm_end = source.find(REINFORCEMENT_MARKER).unwrap();
    let norm = &source[norm_start..norm_end];
    let dot_start = source.find(DOT_MARKER).unwrap();
    let dot_end = source.find(NORMALIZATION_MARKER).unwrap();
    let dot = &source[dot_start..dot_end];
    // Borrow the original learning implementation, including its fixed norm
    // tree and each lane's ascending stride-two additions, verbatim.
    assert!(norm.contains("wg_reduce_dense(tid);"));
    assert!(norm.contains("s_enc_norm = sqrt(s_dense_partials[0]);"));
    assert!(dot.contains("for (var d = lane; d < ENCODED_DIMENSION; d += 2u)"));
    assert!(dot.contains("s_reinf_dot[tid] = dot;"));
    let predict_start = source.find("fn coop_predict_and_act(").unwrap();
    let learn_start = source.find("fn coop_learn_and_store(").unwrap();
    for scratch in ["s_enc_norm", "s_argmin_val"] {
        assert!(
            !source[predict_start..learn_start].contains(scratch),
            "predict/act must leave the cached {scratch} untouched"
        );
    }
    let mut candidate = cached_passes(source);
    assert_eq!(candidate.matches(norm).count(), 1);
    candidate = candidate.replacen(norm, "", 1);
    let recall_start = candidate.find(RECALL_DEFINITION).unwrap();
    let recall_end =
        recall_start + candidate[recall_start..].find("\n}\n").unwrap() + "\n}\n".len();
    let recall = format!(
        r"{RECALL_DEFINITION}
    let pattern_base = agent_id * PATTERN_STRIDE;
{norm}{dot}
    if (tid < MEMORY_CAP) {{
        var similarity: f32 = -2.0;
        if (pattern_buffer[pattern_base + O_PAT_ACTIVE + pattern] >= 0.5) {{
            let dot_val = s_reinf_dot[tid] + s_reinf_dot[tid + MEMORY_CAP];
            let e_norm = s_enc_norm;
            let p_norm = pattern_buffer[pattern_base + O_PAT_NORMS + pattern];
            similarity = 0.0;
            if (e_norm >= 1e-8 && p_norm >= 1e-8) {{
                similarity = clamp(dot_val / (e_norm * p_norm), -1.0, 1.0);
            }}
        }}
        s_similarities[tid] = similarity;
        s_argmin_val[tid] = similarity;
    }}
}}
"
    );
    candidate.replace_range(recall_start..recall_end, &recall);
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        source.matches("var<workgroup>").count()
    );
    assert_eq!(
        candidate.matches("s_enc_norm =").count(),
        source.matches("s_enc_norm =").count()
    );
    assert_eq!(
        candidate.matches(norm).count(),
        1,
        "norm computation moved once"
    );
    assert_eq!(
        candidate.matches(dot).count(),
        1,
        "two-lane dot computation moved once"
    );
    candidate
}

fn prepare() -> (GpuKernel, [wgpu::ComputePipeline; ARM_COUNT]) {
    let mut kernel = prepare_kernel();
    let source = wider_predictor(
        &prefetch_passes(
            &fuse_inline_predictor(&compose_brain_passes(true)),
            PREFETCH,
        ),
        PREDICTOR_LANES,
    );
    let sources = [
        source.clone(),
        cached_passes(&source),
        cooperative_cached_passes(&source),
    ];
    let pipelines: [wgpu::ComputePipeline; ARM_COUNT] =
        std::array::from_fn(|arm| make_pipeline(&kernel, &sources[arm], ARM_NAMES[arm]));
    kernel.kernel_pipeline = pipelines[0].clone();
    (kernel, pipelines)
}

fn advance(kernel: &mut GpuKernel, cycle: u32, cycles: u32) {
    kernel.dispatch_ticks(
        u64::from(cycle * kernel.brain_tick_stride),
        cycles * kernel.brain_tick_stride,
    );
    kernel.poll_wait();
}

fn sequence(kernel: &mut GpuKernel) -> TestResult<Vec<Vec<Vec<u8>>>> {
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
fn cached_recall_reports_rounding_and_repeats() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, pipelines) = prepare();
    prepare_boundary_scene(&kernel);
    let initial = checkpoint(&kernel);
    let initial_state = capture_state(&kernel)?;
    let reference = sequence(&mut kernel)?;
    // The baseline is also repeated from identical bytes to distinguish
    // arithmetic drift between variants from same-source nondeterminism.
    restore(&mut kernel, &initial);
    for (first, repeated) in reference.iter().zip(sequence(&mut kernel)?) {
        assert_state_equal(&kernel, first, &repeated);
    }
    for (arm, pipeline) in pipelines.iter().enumerate().skip(1) {
        restore(&mut kernel, &initial);
        kernel.kernel_pipeline = pipeline.clone();
        let candidate = sequence(&mut kernel)?;
        for (stage, (expected, actual)) in reference.iter().zip(&candidate).enumerate() {
            compare_rounding_state(
                &kernel,
                expected,
                actual,
                &format!("recall_reuse/arm={}/stage={stage}", ARM_NAMES[arm]),
            );
            assert_inactive_agent_unchanged(
                &kernel,
                &initial_state,
                actual,
                INACTIVE_AGENT,
                ARM_NAMES[arm],
            );
        }
        restore(&mut kernel, &initial);
        for (first, repeated) in candidate.iter().zip(sequence(&mut kernel)?) {
            assert_state_equal(&kernel, first, &repeated);
        }
        println!("RECALL_REUSE_SEQUENCE arm={} cycles={} same_arm_repeat_all13_equal=true inactive_agent_unchanged=true", ARM_NAMES[arm], CHUNKS.iter().sum::<u32>());
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run in release mode with --ignored --nocapture"]
fn benchmark_cached_recall_against_optimized() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, pipelines) = prepare();
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    let mut timings: [Vec<f64>; ARM_COUNT] = std::array::from_fn(|_| Vec::new());
    let mut first_states: Option<Vec<Vec<Vec<u8>>>> = None;
    for round in 0..ROUNDS {
        let mut states = Vec::new();
        for offset in 0..pipelines.len() {
            let arm = (round + offset) % pipelines.len();
            restore(&mut kernel, &warm);
            kernel.kernel_pipeline = pipelines[arm].clone();
            let start = Instant::now();
            advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            states.push((arm, capture_state(&kernel)?));
        }
        states.sort_by_key(|state| state.0);
        for (arm, state) in states.iter().skip(1) {
            compare_rounding_state(
                &kernel,
                &states[0].1,
                state,
                &format!("recall_reuse/arm={}/round={round}", ARM_NAMES[*arm]),
            );
        }
        if let Some(first) = &first_states {
            for (arm, actual) in &states {
                assert_state_equal(&kernel, &first[*arm], actual);
            }
        } else {
            first_states = Some(states.into_iter().map(|(_, state)| state).collect());
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let baseline = timings[0][ROUNDS / 2];
    for (arm, samples) in timings.iter().enumerate() {
        let candidate = samples[ROUNDS / 2];
        println!("RECALL_REUSE arm={} cycles={TIMED_CYCLES} rounds={ROUNDS} baseline_seconds={baseline:.9} candidate_seconds={candidate:.9} speedup={:.3} precision=fp32 trajectories_may_differ=true same_arm_repeat_all13_equal=true", ARM_NAMES[arm], baseline / candidate);
    }
    Ok(())
}
