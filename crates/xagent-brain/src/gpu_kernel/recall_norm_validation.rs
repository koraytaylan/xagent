//! Hardware-only comparison of repeated and shared ordered recall norms.
//! Both arms explicitly compose cooperative whitening, interleaved recall
//! and fused predictor updates. The candidate computes the identical scalar
//! query norm once per agent; every pattern retains its original dot order.
//! Complete state checks precede reports of whole-simulation timing.

use std::{error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::predictor_fusion::fuse_inline_predictor;
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, REFRESH_CYCLES,
};
use super::*;

/// Kernel entries accept a starting tick and the cooperative pass limit.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Includes each side of refreshes, repeated death, and a longer continuation.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Populate the episodic memory before timing repeated active-pattern queries.
const WARMUP_CYCLES: u32 = 256;
/// One hundred cycles amortize recording and GPU completion overhead.
const TIMED_CYCLES: u32 = 100;
/// Alternating arm order over an odd pair count yields one median per arm.
const TIMING_ROUNDS: usize = 5;
/// This agent stays inactive without requesting a respawn.
const INACTIVE_AGENT: u32 = 1;
/// Every mutable production simulation buffer participates in parity checks.
const MUTABLE_BUFFERS: usize = 13;
/// Force death initially and again at the first scheduled-refresh boundary.
const EXPECTED_FORCED_DEATHS: f32 = 2.0;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct KernelPipelines {
    claim: wgpu::ComputePipeline,
    main: wgpu::ComputePipeline,
}

fn shared_norm_passes(baseline: &str) -> String {
    const DEFINITION: &str = "fn coop_recall_score(agent_id: u32, tid: u32) {";
    const SCALAR_REFERENCE: &str = "fn scalar_recall_score_reference(agent_id: u32, tid: u32) {";
    const LEARNING_NORM: &str =
        "        if (tid == 0u) { s_enc_norm = sqrt(s_dense_partials[0]); }";
    assert_eq!(baseline.matches(DEFINITION).count(), 1);
    assert_eq!(baseline.matches(SCALAR_REFERENCE).count(), 1);
    assert_eq!(baseline.matches(LEARNING_NORM).count(), 1);
    assert!(baseline.contains("const BRAIN_WORKGROUP_SIZE: u32 = 256u;"));
    assert_eq!(MEMORY_CAP, ENCODED_DIMENSION);
    assert!(MEMORY_CAP < usize::try_from(BRAIN_WORKGROUP_THREADS).unwrap());
    // The scalar cortex norm has already been consumed before recall. No
    // later stage reads it before learning overwrites it with its tree norm.
    let recall_start = baseline.find(SCALAR_REFERENCE).unwrap();
    let learning_norm = baseline.find(LEARNING_NORM).unwrap();
    assert!(!baseline[recall_start..learning_norm].contains("s_enc_norm"));
    let renamed = baseline.replacen(
        DEFINITION,
        "fn repeated_recall_norm_reference(agent_id: u32, tid: u32) {",
        1,
    );
    [renamed.as_str(), include_str!("recall_shared_norm.wgsl")].join("\n")
}

fn make_pipelines(kernel: &GpuKernel, passes: &str, label: &str) -> KernelPipelines {
    let source = apply_subgroup_markers(
        &[
            include_str!("../shaders/kernel/common.wgsl"),
            passes,
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            include_str!("../shaders/kernel/phase_food_claim.wgsl"),
            include_str!("../shaders/kernel/kernel_tick.wgsl"),
        ]
        .join("\n"),
        kernel.has_subgroup,
    );
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
    let create = |entry| {
        kernel
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{label}_{entry}")),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
    };
    KernelPipelines {
        claim: create("kernel_claim_tick"),
        main: create("kernel_tick"),
    }
}

fn prepare_baseline_and_candidate() -> (GpuKernel, wgpu::ComputePipeline) {
    let mut kernel = prepare_kernel();
    let passes = fuse_inline_predictor(&compose_brain_passes(true));
    let baseline = make_pipelines(&kernel, &passes, "recall_norm_cooperative_fused_baseline");
    let candidate = make_pipelines(
        &kernel,
        &shared_norm_passes(&passes),
        "recall_norm_shared_query_candidate",
    );
    // Explicit pipelines prevent inherited production flags from selecting a
    // different brain baseline. Claim contains no reachable recall operation.
    kernel.kernel_claim_pipeline = baseline.claim;
    kernel.kernel_pipeline = baseline.main;
    (kernel, candidate.main)
}

fn advance(kernel: &mut GpuKernel, start_tick: u32, cycles: u32) {
    kernel.dispatch_ticks(u64::from(start_tick), cycles * kernel.brain_tick_stride);
    kernel.poll_wait();
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn shared_recall_norm_matches_complete_optimized_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut alternate_pipeline) = prepare_baseline_and_candidate();
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
        let tick = cycle * kernel.brain_tick_stride;
        advance(&mut kernel, tick, cycles);
        expected.push(capture_state(&kernel)?);
        cycle += cycles;
    }
    assert!(
        kernel.read_full_state_blocking()[P_DEATH_COUNT] >= deaths_before + EXPECTED_FORCED_DEATHS
    );
    restore(&mut kernel, &initial);
    std::mem::swap(&mut kernel.kernel_pipeline, &mut alternate_pipeline);
    cycle = 0;
    for (cycles, expected) in PARITY_CHUNKS.into_iter().zip(&expected) {
        if cycle == REFRESH_CYCLES {
            force_death(&kernel);
        }
        let tick = cycle * kernel.brain_tick_stride;
        advance(&mut kernel, tick, cycles);
        assert_state_equal(&kernel, expected, &capture_state(&kernel)?);
        cycle += cycles;
    }
    let inactive_after = kernel.read_agent_state(INACTIVE_AGENT);
    assert_eq!(
        bytemuck::cast_slice::<f32, u32>(&inactive_before.brain_state),
        bytemuck::cast_slice::<f32, u32>(&inactive_after.brain_state),
    );
    println!(
        "RECALL_NORM_PARITY baseline=cooperative_whitening_fused_predictor cycles={cycle} exact_buffers={MUTABLE_BUFFERS} dead_agent_unchanged=true death_boundaries=2"
    );
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_shared_recall_norm_against_optimized() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut alternate_pipeline) = prepare_baseline_and_candidate();
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    let tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    let mut current_arm = 0;
    let mut timings = [Vec::new(), Vec::new()];
    for round in 0..TIMING_ROUNDS {
        let mut states = [None, None];
        for offset in 0..timings.len() {
            let arm = (round + offset) % timings.len();
            restore(&mut kernel, &warm);
            if arm != current_arm {
                std::mem::swap(&mut kernel.kernel_pipeline, &mut alternate_pipeline);
                current_arm = arm;
            }
            let start = Instant::now();
            advance(&mut kernel, tick, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            states[arm] = Some(capture_state(&kernel)?);
        }
        assert_state_equal(
            &kernel,
            states[0].as_ref().unwrap(),
            states[1].as_ref().unwrap(),
        );
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let baseline = timings[0][TIMING_ROUNDS / 2];
    let shared_norm = timings[1][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    println!(
        "RECALL_NORM agents={} ticks={ticks} rounds={TIMING_ROUNDS} baseline=cooperative_whitening_fused_predictor baseline_secs={baseline:.9} shared_norm_secs={shared_norm:.9} baseline_tps={:.3} shared_norm_tps={:.3} speedup={:.3} exact_buffers={MUTABLE_BUFFERS}",
        kernel.agent_count, f64::from(ticks) / baseline, f64::from(ticks) / shared_norm, baseline / shared_norm,
    );
    Ok(())
}
