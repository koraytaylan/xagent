//! Hardware-only shared-cache experiments for prediction. All four arms use
//! production cooperative whitening, interleaved recall, and fused predictor
//! arithmetic. Candidates independently cache recalled-context normalization
//! and previous encoded inputs in scratch whose next consumer overwrites it.
//! Exact comparisons include all thirteen mutable simulation storage buffers.

use std::{error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::predictor_fusion::fuse_inline_predictor;
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, REFRESH_CYCLES,
};
use super::*;

/// Kernel push constants contain the starting tick and brain-pass limit.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Baseline, each independent cache, and the combined candidate.
const ARM_COUNT: usize = 4;
/// The reference retains the current optimized predictor's storage accesses.
const BASELINE_ARM: usize = 0;
/// Labels identify the same variants in parity and timing output.
const ARM_NAMES: [&str; ARM_COUNT] = [
    "optimized_baseline",
    "cached_context",
    "cached_previous_inputs",
    "cached_context_and_previous_inputs",
];
/// Context and previous-input transformations are independently selectable.
const ARM_OPTIONS: [(bool, bool); ARM_COUNT] =
    [(false, false), (true, false), (false, true), (true, true)];
/// Boundaries include refreshes, repeated death, and one hundred total cycles.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Mature episodic memory exercises prediction with populated recall.
const WARMUP_CYCLES: u32 = 256;
/// One hundred full cycles amortize submission and completion overhead.
const TIMED_CYCLES: u32 = 100;
/// Rotate the arm order and take the median of an odd number of measurements.
const TIMING_ROUNDS: usize = 5;
/// Every mutable simulation storage buffer participates in parity checks.
const MUTABLE_BUFFERS: usize = 13;
/// This fixture agent remains inactive without requesting a respawn.
const INACTIVE_AGENT: u32 = 1;

/// Exact original context block; source drift must fail before shader creation.
const CONTEXT_BLOCK: &str = r"    // ── Per-dimension context blend and tanh: threads 0..PREDICTOR_DIMENSION ──
    if (tid < PREDICTOR_DIMENSION) {
        // Context blend contribution (per-dimension)
        if (recall_count > 0u) {
            let context_weight = brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT];
            var total_sim: f32 = 0.0;
            for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                total_sim += max(s_recall_similarity[k], 0.0);
            }
            if (total_sim > 1e-8) {
                // Stored keys are centered; the running mean turns them back
                // into encoded states for the forward model.
                let encoded_mean = brain_state[brain_base + O_ENCODED_MEAN + tid];
                for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                    let idx = u32(s_recall[k]);
                    let w = context_weight * max(s_recall_similarity[k], 0.0) / total_sim;
                    s_prediction[tid] +=
                        (pattern_buffer[pattern_base + tid * MEMORY_CAP + idx] + encoded_mean) * w;
                }
            }
        }

        // Apply tanh to prediction
        s_prediction[tid] = fast_tanh(s_prediction[tid]);
    }
    workgroupBarrier();
";

/// Unique storage load in the fused predictor's ordered train/predict loop.
const PREVIOUS_INPUT_LOAD: &str =
    "let previous_input = brain_state[brain_base + O_PREV_ENCODED + j];";
/// All invocations reach this uniform scratch-prediction branch.
const PREDICTOR_BOUNDARY: &str = "    if (use_scratch_prediction) {\n";
/// Whitening consumes this scratch only after predictor computation completes.
const PREVIOUS_INPUT_CACHE: &str = r"    // Cache previous inputs before the inline predictor consumes them.
    if (tid < ENCODED_DIMENSION) {
        s_reinf_dot[tid] = brain_state[brain_base + O_PREV_ENCODED + tid];
    }
    workgroupBarrier();

";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn candidate_passes(context_cache: bool, previous_input_cache: bool) -> String {
    let mut source = fuse_inline_predictor(&compose_brain_passes(true));
    if context_cache {
        assert_eq!(source.matches(CONTEXT_BLOCK).count(), 1);
        assert!(source.contains("var<workgroup> s_credit: array<f32, ENCODED_DIMENSION>;"));
        assert!(source.contains("var<workgroup> s_enc_norm: f32;"));
        source = source.replacen(
            CONTEXT_BLOCK,
            "    blend_cached_recalled_context(brain_base, pattern_base, recall_count, tid);\n",
            1,
        );
        source.push('\n');
        source.push_str(include_str!("context_cached.wgsl"));
    }
    if previous_input_cache {
        assert_eq!(source.matches(PREVIOUS_INPUT_LOAD).count(), 1);
        assert_eq!(source.matches(PREDICTOR_BOUNDARY).count(), 1);
        assert!(source.contains("var<workgroup> s_reinf_dot: array<f32, 256>;"));
        source = source.replacen(
            PREVIOUS_INPUT_LOAD,
            "let previous_input = s_reinf_dot[j];",
            1,
        );
        source = source.replacen(
            PREDICTOR_BOUNDARY,
            &format!("{PREVIOUS_INPUT_CACHE}{PREDICTOR_BOUNDARY}"),
            1,
        );
    }
    source
}

fn make_pipeline(kernel: &GpuKernel, arm: usize) -> wgpu::ComputePipeline {
    let (context_cache, previous_input_cache) = ARM_OPTIONS[arm];
    let passes = candidate_passes(context_cache, previous_input_cache);
    let source = apply_subgroup_markers(
        &[
            include_str!("../shaders/kernel/common.wgsl"),
            &passes,
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            include_str!("../shaders/kernel/phase_food_claim.wgsl"),
            include_str!("../shaders/kernel/kernel_tick.wgsl"),
        ]
        .join("\n"),
        kernel.has_subgroup,
    );
    let label = ARM_NAMES[arm];
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    let constants = vision_override_constants(&kernel.layout);
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("kernel_tick"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        })
}

fn prepare_arms() -> (GpuKernel, [wgpu::ComputePipeline; ARM_COUNT]) {
    let mut kernel = prepare_kernel();
    let pipelines: [wgpu::ComputePipeline; ARM_COUNT] =
        std::array::from_fn(|arm| make_pipeline(&kernel, arm));
    // All arms use the same claim, global, and vision entries. The tested
    // main reference is explicitly composed, independent of environment flags.
    kernel.kernel_pipeline = pipelines[BASELINE_ARM].clone();
    (kernel, pipelines)
}

fn advance(kernel: &mut GpuKernel, start_cycle: u32, cycles: u32) {
    let stride = kernel.brain_tick_stride;
    kernel.dispatch_ticks(u64::from(start_cycle * stride), cycles * stride);
    kernel.poll_wait();
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn shared_prediction_caches_match_optimized_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, pipelines) = prepare_arms();
    prepare_boundary_scene(&kernel);
    let initial = checkpoint(&kernel);
    let inactive_before = kernel.read_agent_state(INACTIVE_AGENT);
    let deaths_before = kernel.read_full_state_blocking()[P_DEATH_COUNT];
    let mut expected = Vec::with_capacity(PARITY_CHUNKS.len());
    let mut cycle = 0;
    for cycles in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(&kernel);
        }
        advance(&mut kernel, cycle, cycles);
        expected.push(capture_state(&kernel)?);
        cycle += cycles;
    }
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] > deaths_before);
    for (arm, pipeline) in pipelines.iter().enumerate().skip(1) {
        restore(&mut kernel, &initial);
        kernel.kernel_pipeline = pipeline.clone();
        cycle = 0;
        for (cycles, expected) in PARITY_CHUNKS.into_iter().zip(&expected) {
            if cycle == REFRESH_CYCLES {
                force_death(&kernel);
            }
            advance(&mut kernel, cycle, cycles);
            assert_state_equal(&kernel, expected, &capture_state(&kernel)?);
            cycle += cycles;
        }
        let inactive_after = kernel.read_agent_state(INACTIVE_AGENT);
        assert_eq!(
            bytemuck::cast_slice::<f32, u32>(&inactive_before.brain_state),
            bytemuck::cast_slice::<f32, u32>(&inactive_after.brain_state),
        );
        println!(
            "PREDICTION_CACHE_PARITY arm={} cycles={cycle} exact_buffers={MUTABLE_BUFFERS} dead_agent_unchanged=true death_boundary=true",
            ARM_NAMES[arm],
        );
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_shared_prediction_caches() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, pipelines) = prepare_arms();
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    let mut timings: [Vec<f64>; ARM_COUNT] = std::array::from_fn(|_| Vec::new());
    for round in 0..TIMING_ROUNDS {
        let mut states: [Option<Vec<Vec<u8>>>; ARM_COUNT] = std::array::from_fn(|_| None);
        for offset in 0..ARM_COUNT {
            let arm = (round + offset) % ARM_COUNT;
            restore(&mut kernel, &warm);
            kernel.kernel_pipeline = pipelines[arm].clone();
            let start = Instant::now();
            advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            states[arm] = Some(capture_state(&kernel)?);
        }
        let reference = states[BASELINE_ARM].as_ref().unwrap();
        for candidate in states.iter().skip(1) {
            assert_state_equal(&kernel, reference, candidate.as_ref().unwrap());
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let reference = timings[BASELINE_ARM][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    for (arm, samples) in timings.iter().enumerate() {
        let elapsed = samples[TIMING_ROUNDS / 2];
        println!(
            "PREDICTION_CACHE_TIMING arm={} agents={} warmup_cycles={WARMUP_CYCLES} ticks={ticks} rounds={TIMING_ROUNDS} secs={elapsed:.9} tps={:.3} speedup={:.3} exact_buffers={MUTABLE_BUFFERS}",
            ARM_NAMES[arm], kernel.agent_count, f64::from(ticks) / elapsed, reference / elapsed,
        );
    }
    Ok(())
}
