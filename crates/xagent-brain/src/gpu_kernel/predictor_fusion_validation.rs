//! Hardware-only comparison of separate and fused predictor weight loops.
//! Production source composers supply the predictor-only and combined
//! whitening/recall candidates. An explicit untouched serial brain supplies
//! the oracle, independent of environment flags; all thirteen mutable buffers
//! must match before timing is reported.

use std::{error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::predictor_fusion::fuse_inline_predictor;
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, REFRESH_CYCLES,
};
use super::*;

/// Kernel push constants contain the starting tick and the brain pass limit.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Boundaries include refreshes, a repeated death, and one hundred cycles.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Mature episodic memory exercises prediction and learning with full recall.
const WARMUP_CYCLES: u32 = 256;
/// One hundred full cycles amortize submission and completion overhead.
const TIMED_CYCLES: u32 = 100;
/// Odd pair count gives one median while alternating arm order.
const TIMING_ROUNDS: usize = 5;
/// The fixture leaves this agent inactive without requesting a respawn.
const INACTIVE_AGENT: u32 = 1;
/// Every mutable simulation storage buffer participates in parity checks.
const MUTABLE_BUFFERS: usize = 13;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct KernelPipelines {
    claim: wgpu::ComputePipeline,
    main: wgpu::ComputePipeline,
}

fn make_kernel_pipelines(kernel: &GpuKernel, passes: &str, label: &str) -> KernelPipelines {
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

fn prepare_oracle_and_candidate(combined: bool) -> (GpuKernel, wgpu::ComputePipeline) {
    let mut kernel = prepare_kernel();
    let original_passes = include_str!("../shaders/kernel/brain_passes.wgsl");
    let oracle = make_kernel_pipelines(
        &kernel,
        original_passes,
        "predictor_untouched_serial_oracle",
    );
    kernel.kernel_claim_pipeline = oracle.claim;
    kernel.kernel_pipeline = oracle.main;
    let passes = if combined {
        compose_brain_passes(true)
    } else {
        original_passes.to_owned()
    };
    let passes = fuse_inline_predictor(&passes);
    let label = if combined {
        "combined_predictor_whitening_probe"
    } else {
        "fused_predictor_probe"
    };
    let candidate = make_kernel_pipelines(&kernel, &passes, label);
    // Claim has no reachable brain operations, so all arms use the oracle's
    // unchanged claim entry and swap only the main entry below.
    (kernel, candidate.main)
}

fn advance(kernel: &mut GpuKernel, start_tick: u32, cycles: u32) {
    kernel.dispatch_ticks(u64::from(start_tick), cycles * kernel.brain_tick_stride);
    kernel.poll_wait();
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn fused_predictor_matches_complete_serial_state() -> TestResult {
    check_fused_predictor(false)
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn combined_predictor_whitening_matches_complete_serial_state() -> TestResult {
    check_fused_predictor(true)
}

fn check_fused_predictor(combined: bool) -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut alternate_pipeline) = prepare_oracle_and_candidate(combined);
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
        "PREDICTOR_FUSION_PARITY combined={combined} cycles={cycle} exact_buffers={MUTABLE_BUFFERS} dead_agent_unchanged=true death_boundary=true"
    );
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_fused_predictor_against_serial() -> TestResult {
    benchmark_fused_predictor(false)
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_combined_predictor_whitening_against_serial() -> TestResult {
    benchmark_fused_predictor(true)
}

fn benchmark_fused_predictor(combined: bool) -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut alternate_pipeline) = prepare_oracle_and_candidate(combined);
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
    let fused = timings[1][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    println!(
        "PREDICTOR_FUSION combined={combined} agents={} ticks={ticks} rounds={TIMING_ROUNDS} serial_secs={serial:.9} fused_secs={fused:.9} serial_tps={:.3} fused_tps={:.3} speedup={:.3} exact_buffers={MUTABLE_BUFFERS}",
        kernel.agent_count, f64::from(ticks) / serial, f64::from(ticks) / fused, serial / fused,
    );
    Ok(())
}
