//! Hardware-only software-prefetch experiments for ordered dense brain loops.
//! Scalar locals hold four or eight independent inputs before arithmetic. Each
//! lane keeps its original stride-four addition order, and absent tail terms
//! execute no addition. All arms use production cooperative whitening and the
//! fused inline predictor, checked shaders, and unchanged workgroup storage.

use std::{error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::dense_prefetch::prefetch_passes;
use super::predictor_fusion::fuse_inline_predictor;
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, REFRESH_CYCLES,
};
use super::*;

/// Kernel push constants contain the starting tick and brain-pass limit.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Both block sizes expose independent loads without allocating shared arrays.
const PREFETCH_FACTORS: [u32; 2] = [4, 8];
/// The reference plus two candidate block sizes share the same checkpoint.
const ARM_COUNT: usize = 3;
/// The first arm keeps the production scalar loops unchanged.
const BASELINE_ARM: usize = 0;
/// Labels identify identical variants in parity, timings, and compiler output.
const ARM_NAMES: [&str; ARM_COUNT] = ["prefetch_baseline", "prefetch_4", "prefetch_8"];
/// Includes both sides of refreshes, repeated death, and a longer continuation.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Populate episodic memory before measuring dense work on evolving state.
const WARMUP_CYCLES: u32 = 256;
/// One hundred cycles amortize submission and GPU-completion overhead.
const TIMED_CYCLES: u32 = 100;
/// Rotate arm order for five complete trials of each candidate.
const TIMING_ROUNDS: usize = 5;
/// Every mutable production simulation buffer participates in exact parity.
const MUTABLE_BUFFERS: usize = 13;
/// The fixture leaves this agent inactive without requesting a respawn.
const INACTIVE_AGENT: u32 = 1;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

pub(super) fn make_pipeline(
    kernel: &GpuKernel,
    passes: &str,
    label: &str,
) -> wgpu::ComputePipeline {
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

struct PipelineArms {
    parked: [Option<wgpu::ComputePipeline>; ARM_COUNT],
    active: usize,
}

impl PipelineArms {
    fn prepare() -> (GpuKernel, Self) {
        let mut kernel = prepare_kernel();
        let baseline = fuse_inline_predictor(&compose_brain_passes(true));
        kernel.kernel_pipeline = make_pipeline(&kernel, &baseline, ARM_NAMES[BASELINE_ARM]);
        let candidates = PREFETCH_FACTORS.map(|factor| prefetch_passes(&baseline, factor));
        let parked = [
            None,
            Some(make_pipeline(&kernel, &candidates[0], ARM_NAMES[1])),
            Some(make_pipeline(&kernel, &candidates[1], ARM_NAMES[2])),
        ];
        (
            kernel,
            Self {
                parked,
                active: BASELINE_ARM,
            },
        )
    }

    fn activate(&mut self, kernel: &mut GpuKernel, arm: usize) {
        if arm != self.active {
            let next = self.parked[arm].take().unwrap();
            let previous = std::mem::replace(&mut kernel.kernel_pipeline, next);
            assert!(self.parked[self.active].replace(previous).is_none());
            self.active = arm;
        }
    }
}

fn advance(kernel: &mut GpuKernel, start_tick: u32, cycles: u32) {
    kernel.dispatch_ticks(u64::from(start_tick), cycles * kernel.brain_tick_stride);
    kernel.poll_wait();
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn prefetched_dense_loops_match_complete_optimized_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut pipelines) = PipelineArms::prepare();
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
    for (arm, name) in ARM_NAMES.iter().enumerate().skip(1) {
        restore(&mut kernel, &initial);
        pipelines.activate(&mut kernel, arm);
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
        println!("DENSE_PREFETCH_PARITY variant={name} cycles={cycle} exact_buffers={MUTABLE_BUFFERS} dead_agent_unchanged=true death_boundary=true");
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run in release mode with --ignored --nocapture"]
fn benchmark_prefetched_dense_loops_against_optimized() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut pipelines) = PipelineArms::prepare();
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    let tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    let mut timings: [Vec<f64>; ARM_COUNT] = std::array::from_fn(|_| Vec::new());
    for round in 0..TIMING_ROUNDS {
        let mut states: [Option<Vec<Vec<u8>>>; ARM_COUNT] = std::array::from_fn(|_| None);
        for offset in 0..ARM_COUNT {
            let arm = (round + offset) % ARM_COUNT;
            restore(&mut kernel, &warm);
            pipelines.activate(&mut kernel, arm);
            let start = Instant::now();
            advance(&mut kernel, tick, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            states[arm] = Some(capture_state(&kernel)?);
        }
        for state in states.iter().skip(1) {
            assert_state_equal(
                &kernel,
                states[BASELINE_ARM].as_ref().unwrap(),
                state.as_ref().unwrap(),
            );
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let baseline = timings[BASELINE_ARM][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    for (name, samples) in ARM_NAMES.iter().zip(timings) {
        let seconds = samples[TIMING_ROUNDS / 2];
        println!(
            "DENSE_PREFETCH variant={name} agents={} ticks={ticks} rounds={TIMING_ROUNDS} seconds={seconds:.9} tps={:.3} speedup={:.3} exact_buffers={MUTABLE_BUFFERS}",
            kernel.agent_count, f64::from(ticks) / seconds, baseline / seconds,
        );
    }
    Ok(())
}
