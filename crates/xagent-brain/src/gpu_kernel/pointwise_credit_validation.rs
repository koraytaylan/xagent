//! Hardware-only comparison of looped and pointwise encoder credit.
//! Both arms use the same exact 32-output cooperative tiled schedule. The
//! candidate changes only the credit pipeline and its workgroup count, giving
//! each weight one invocation while retaining the original update expression.

use std::{error::Error, time::Instant};

use super::cooperative_whitening_validation::make_pipeline as make_reference_pipeline;
use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::exact_tiled_validation::{advance, make_pipelines, TiledPipelines};
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, REFRESH_CYCLES,
};
use super::*;

/// Keep the better measured exact tile shape identical in both arms.
const TILE_OUTPUTS: u32 = 32;
/// Fill a portable workgroup with independent contiguous weight updates.
const POINTWISE_CREDIT_THREADS: u32 = 256;
/// All pipelines share the kernel's tick/pass-limit push-constant layout.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Includes refresh boundaries, two forced deaths and a longer continuation.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Populate episodic memory and run learning before timing weight updates.
const WARMUP_CYCLES: u32 = 256;
/// One hundred cycles amortize recording and completion overhead.
const TIMED_CYCLES: u32 = 100;
/// Alternating arm order over five pairs yields one median per arm.
const TIMING_ROUNDS: usize = 5;
/// The fixture leaves this agent inactive without requesting a respawn.
const INACTIVE_AGENT: u32 = 1;
/// Compare every mutable production storage buffer, including brain weights.
const MUTABLE_BUFFERS: usize = 13;
/// Force death initially and again at the first scheduled-refresh boundary.
const EXPECTED_FORCED_DEATHS: f32 = 2.0;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct CreditPipeline {
    pipeline: wgpu::ComputePipeline,
    workgroups_per_agent: u32,
}

impl CreditPipeline {
    fn swap_with(&mut self, pipelines: &mut TiledPipelines) {
        std::mem::swap(&mut pipelines.credit, &mut self.pipeline);
        std::mem::swap(
            &mut pipelines.credit_workgroups_per_agent,
            &mut self.workgroups_per_agent,
        );
    }
}

fn make_pointwise_credit(kernel: &GpuKernel) -> TestResult<CreditPipeline> {
    let fragment = include_str!("../shaders/kernel/exact_pointwise_encoder_credit.wgsl");
    assert!(fragment.contains(&format!(
        "const POINTWISE_CREDIT_THREADS: u32 = {POINTWISE_CREDIT_THREADS}u;"
    )));
    let weight_count = kernel
        .layout
        .feature_count
        .checked_mul(ENCODED_DIMENSION)
        .ok_or("Encoder weight count overflow")?;
    let weight_count = u32::try_from(weight_count)?;
    let workgroups_per_agent = weight_count.div_ceil(POINTWISE_CREDIT_THREADS);
    assert!(workgroups_per_agent <= kernel.device.limits().max_compute_workgroups_per_dimension);
    let label = "exact_pointwise_encoder_credit_probe";
    let source = [include_str!("../shaders/kernel/common.wgsl"), fragment].join("\n");
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
    let pipeline = kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("exact_pointwise_encoder_credit"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });
    Ok(CreditPipeline {
        pipeline,
        workgroups_per_agent,
    })
}

fn prepare() -> TestResult<(GpuKernel, TiledPipelines, CreditPipeline)> {
    let mut kernel = prepare_kernel();
    kernel.kernel_pipeline = make_reference_pipeline(&kernel, false);
    let pipelines = make_pipelines(&kernel, TILE_OUTPUTS, true);
    let candidate = make_pointwise_credit(&kernel)?;
    Ok((kernel, pipelines, candidate))
}

fn encoder_bits(kernel: &GpuKernel) -> Vec<u32> {
    let weights = kernel
        .layout
        .feature_count
        .checked_mul(ENCODED_DIMENSION)
        .unwrap();
    let end = O_ENC_WEIGHTS.checked_add(weights).unwrap();
    (0..kernel.agent_count)
        .flat_map(|agent| {
            kernel.read_agent_state(agent).brain_state[O_ENC_WEIGHTS..end]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn pointwise_encoder_credit_matches_complete_tiled_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut pipelines, mut candidate) = prepare()?;
    prepare_boundary_scene(&kernel);
    let initial = checkpoint(&kernel);
    let inactive_before = kernel.read_agent_state(INACTIVE_AGENT);
    let weights_before = encoder_bits(&kernel);
    let deaths_before = kernel.read_full_state_blocking()[P_DEATH_COUNT];
    let mut expected = Vec::with_capacity(PARITY_CHUNKS.len());
    let mut cycle = 0;
    for cycles in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(&kernel);
        }
        let tick = cycle * kernel.brain_tick_stride;
        advance(&mut kernel, Some(&pipelines), tick, cycles);
        expected.push(capture_state(&kernel)?);
        cycle += cycles;
    }
    assert!(
        kernel.read_full_state_blocking()[P_DEATH_COUNT] >= deaths_before + EXPECTED_FORCED_DEATHS
    );
    assert_ne!(
        encoder_bits(&kernel),
        weights_before,
        "fixture must update encoder weights"
    );
    restore(&mut kernel, &initial);
    candidate.swap_with(&mut pipelines);
    cycle = 0;
    for (cycles, expected) in PARITY_CHUNKS.into_iter().zip(&expected) {
        if cycle == REFRESH_CYCLES {
            force_death(&kernel);
        }
        let tick = cycle * kernel.brain_tick_stride;
        advance(&mut kernel, Some(&pipelines), tick, cycles);
        assert_state_equal(&kernel, expected, &capture_state(&kernel)?);
        cycle += cycles;
    }
    let inactive_after = kernel.read_agent_state(INACTIVE_AGENT);
    assert_eq!(
        bytemuck::cast_slice::<f32, u32>(&inactive_before.brain_state),
        bytemuck::cast_slice::<f32, u32>(&inactive_after.brain_state),
    );
    println!(
        "POINTWISE_CREDIT_PARITY baseline=exact_tiled32_cooperative credit_workgroups_per_agent={} cycles={cycle} exact_buffers={MUTABLE_BUFFERS} dead_agent_unchanged=true death_boundaries=2 encoder_weights_changed=true",
        pipelines.credit_workgroups_per_agent,
    );
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_pointwise_encoder_credit_against_tiled() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut pipelines, mut candidate) = prepare()?;
    advance(&mut kernel, Some(&pipelines), 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    let tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    let pointwise_workgroups = candidate.workgroups_per_agent;
    let mut current_arm = 0;
    let mut timings = [Vec::new(), Vec::new()];
    for round in 0..TIMING_ROUNDS {
        let mut states = [None, None];
        for offset in 0..timings.len() {
            let arm = (round + offset) % timings.len();
            restore(&mut kernel, &warm);
            if arm != current_arm {
                candidate.swap_with(&mut pipelines);
                current_arm = arm;
            }
            let start = Instant::now();
            advance(&mut kernel, Some(&pipelines), tick, TIMED_CYCLES);
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
    let pointwise = timings[1][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    println!(
        "POINTWISE_CREDIT agents={} ticks={ticks} rounds={TIMING_ROUNDS} baseline=exact_tiled32_cooperative pointwise_workgroups_per_agent={pointwise_workgroups} baseline_secs={baseline:.9} pointwise_secs={pointwise:.9} baseline_tps={:.3} pointwise_tps={:.3} speedup={:.3} exact_buffers={MUTABLE_BUFFERS}",
        kernel.agent_count, f64::from(ticks) / baseline, f64::from(ticks) / pointwise, baseline / pointwise,
    );
    Ok(())
}
