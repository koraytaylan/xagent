//! Hardware-only comparison of private and workgroup whitening matrices.
//! Only the fused main pipeline changes; dispatches and arithmetic retain
//! their production order, and all thirteen mutable buffers must match.

use std::{error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, PARITY_CHUNKS, REFRESH_CYCLES,
};
use super::*;

/// Kernel push constants contain the starting tick and the brain pass limit.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// More than a full memory window exercises the mature learning workload.
const WARMUP_CYCLES: u32 = 256;
/// Four command chunks amortize submission and completion overhead.
const TIMED_CYCLES: u32 = MAX_FUSED_BATCHES * 4;
/// Odd pair count gives one median while alternating arm order.
const TIMING_ROUNDS: usize = 5;
/// The fixture keeps this agent inactive without requesting a respawn.
const INACTIVE_AGENT: u32 = 1;
/// Full simulation snapshots include every mutable storage buffer.
const MUTABLE_BUFFERS: usize = 13;
/// The original local declarations are removed only inside the refresh body.
const PRIVATE_MATRICES: &str =
    "    var a: array<array<f32, 8>, 8>;\n    var v: array<array<f32, 8>, 8>;\n";
/// Two matrices share one resource, with the original row/column indexing.
const SHARED_MATRICES: &str = "\
// Only invocation zero calls refresh_vision_whitening. Both matrices are
// initialized completely before use, and no other invocation accesses them.
// The covariance and eigenvector matrices occupy one 512-byte resource.
const WHITENING_SCRATCH_MATRICES: u32 = 2u;
var<workgroup> whitening_scratch:
    array<array<array<f32, VISION_PATHWAY_INPUTS>, VISION_PATHWAY_INPUTS>, WHITENING_SCRATCH_MATRICES>;
\n";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

pub(super) fn workgroup_whitening_common() -> String {
    let common = include_str!("../shaders/kernel/common.wgsl");
    let begin = "fn refresh_vision_whitening(brain_base: u32) {";
    let end = "fn settle_recent_moments_at_death(brain_base: u32) {";
    assert_eq!(common.matches(begin).count(), 1);
    assert_eq!(common.matches(end).count(), 1);
    let start = common.find(begin).unwrap();
    let finish = common.find(end).unwrap();
    assert!(finish > start);
    let original = &common[start..finish];
    assert_eq!(original.matches(PRIVATE_MATRICES).count(), 1);
    assert!(original.contains("a[") && original.contains("v["));
    let shared = original
        .replacen(PRIVATE_MATRICES, "", 1)
        .replace("a[", "whitening_scratch[0u][")
        .replace("v[", "whitening_scratch[1u][");
    assert!(!shared.contains("a[") && !shared.contains("v["));
    [
        &common[..start],
        SHARED_MATRICES,
        shared.as_str(),
        &common[finish..],
    ]
    .concat()
}

fn make_shared_pipeline(kernel: &GpuKernel) -> wgpu::ComputePipeline {
    let common = workgroup_whitening_common();
    // The refresh's only caller is vision_pathway_step, itself called only
    // inside coop_predict_and_act's tid==0 exploration block. Each workgroup
    // serves one agent, so the matrices have one writer and reader. No new
    // barrier is required; the following existing pass barrier is unchanged.
    let passes = include_str!("../shaders/kernel/brain_passes.wgsl");
    assert_eq!(passes.matches("refresh_vision_whitening(").count(), 1);
    assert_eq!(passes.matches("vision_pathway_step(").count(), 2);
    let source = apply_subgroup_markers(
        &[
            common.as_str(),
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
            label: Some("workgroup_whitening_kernel_probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("workgroup_whitening_kernel_probe_layout"),
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
            label: Some("workgroup_whitening_kernel_probe"),
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

fn advance(kernel: &mut GpuKernel, start_tick: u32, cycles: u32) {
    kernel.dispatch_ticks(u64::from(start_tick), cycles * kernel.brain_tick_stride);
    kernel.poll_wait();
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn workgroup_whitening_matches_complete_serial_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    let mut alternate_pipeline = make_shared_pipeline(&kernel);
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
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] > deaths_before);
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
        "WHITENING_STORAGE_PARITY cycles={cycle} exact_buffers={MUTABLE_BUFFERS} dead_agent_unchanged=true death_boundary=true"
    );
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_workgroup_whitening_against_serial() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    let mut alternate_pipeline = make_shared_pipeline(&kernel);
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
    let serial = timings[0][TIMING_ROUNDS / 2];
    let shared = timings[1][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    println!(
        "WHITENING_STORAGE agents={} ticks={ticks} rounds={TIMING_ROUNDS} serial_secs={serial:.9} shared_secs={shared:.9} serial_tps={:.3} shared_tps={:.3} speedup={:.3} exact_buffers={MUTABLE_BUFFERS}",
        kernel.agent_count, f64::from(ticks) / serial, f64::from(ticks) / shared, serial / shared,
    );
    Ok(())
}
