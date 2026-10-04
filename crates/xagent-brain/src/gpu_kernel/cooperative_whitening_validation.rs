//! Hardware-only cooperative whitening experiment. Jacobi rotations retain
//! their original order on invocation zero; separate invocations reconstruct
//! independent whitening cells with the original ordered sums. Interleaved
//! recall removes the other known long-lived private operand set.

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
/// Alternate the order of an odd number of matched timing pairs.
const TIMING_ROUNDS: usize = 5;
/// The fixture keeps this agent inactive without requesting a respawn.
const INACTIVE_AGENT: u32 = 1;
/// Full simulation snapshots include every mutable storage buffer.
const MUTABLE_BUFFERS: usize = 13;
/// Both forced deaths must execute, including the scheduled-refresh boundary.
const EXPECTED_FORCED_DEATHS: f32 = 2.0;
type TestResult<T = ()> = Result<T, Box<dyn Error>>;

pub(super) fn make_pipeline(kernel: &GpuKernel, cooperative: bool) -> wgpu::ComputePipeline {
    let common = include_str!("../shaders/kernel/common.wgsl");
    // The reference always compiles the untouched serial passes, independent
    // of any environment-selected production pipeline.
    let passes = if cooperative {
        compose_brain_passes(true)
    } else {
        include_str!("../shaders/kernel/brain_passes.wgsl").to_owned()
    };
    let source = apply_subgroup_markers(
        &[
            common,
            passes.as_str(),
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            include_str!("../shaders/kernel/phase_food_claim.wgsl"),
            include_str!("../shaders/kernel/kernel_tick.wgsl"),
        ]
        .join("\n"),
        kernel.has_subgroup,
    );
    let label = if cooperative {
        "cooperative_whitening_and_interleaved_recall"
    } else {
        "serial_whitening_and_recall_reference"
    };
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

fn advance(kernel: &mut GpuKernel, start_tick: u32, cycles: u32) {
    kernel.dispatch_ticks(u64::from(start_tick), cycles * kernel.brain_tick_stride);
    kernel.poll_wait();
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn cooperative_whitening_matches_complete_serial_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    kernel.kernel_pipeline = make_pipeline(&kernel, false);
    let mut alternate_pipeline = make_pipeline(&kernel, true);
    prepare_boundary_scene(&kernel);
    let initial = checkpoint(&kernel);
    let inactive_before = kernel.read_agent_state(INACTIVE_AGENT);
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
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= EXPECTED_FORCED_DEATHS);
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
        "COOPERATIVE_WHITENING_PARITY cycles={cycle} exact_buffers={MUTABLE_BUFFERS} dead_agent_unchanged=true death_boundary=true"
    );
    Ok(())
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn cooperative_brain_preserves_cortex_and_odd_vision_layouts() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    /// Odd dimensions produce a partial tile in the encoder's inner loop.
    const VISION_WIDTH: u32 = 13;
    const VISION_HEIGHT: u32 = 9;
    /// Keep cortex work bounded while exercising repeated learned-state reuse.
    const CYCLES: u32 = 8;
    const SEED: u64 = 42;
    const FOOD_COUNT: usize = 104;
    for cortex in [false, true] {
        let brain = BrainConfig {
            vision_width: VISION_WIDTH,
            vision_height: VISION_HEIGHT,
            visual_cortex_enabled: cortex,
            vision_stride: 1,
            ..BrainConfig::default()
        };
        let mut kernel = GpuKernel::new(1, FOOD_COUNT, &brain, &WorldConfig::default());
        kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
        kernel.set_brain_beside_vision(false);
        kernel.reset_agents_seeded(&brain, SEED);
        super::vision_validation::upload_random_scene(&kernel, SEED, true, false);
        kernel.kernel_pipeline = make_pipeline(&kernel, false);
        let candidate = make_pipeline(&kernel, true);
        let initial = checkpoint(&kernel);
        for cycle in 0..CYCLES {
            let tick = cycle * kernel.brain_tick_stride;
            advance(&mut kernel, tick, 1);
        }
        let expected = capture_state(&kernel)?;
        restore(&mut kernel, &initial);
        kernel.kernel_pipeline = candidate;
        for cycle in 0..CYCLES {
            let tick = cycle * kernel.brain_tick_stride;
            advance(&mut kernel, tick, 1);
        }
        assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
        println!("COOPERATIVE_LAYOUT_PARITY vision={VISION_WIDTH}x{VISION_HEIGHT} cortex={cortex} cycles={CYCLES} exact_buffers={MUTABLE_BUFFERS}");
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_cooperative_whitening_against_serial() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    kernel.kernel_pipeline = make_pipeline(&kernel, false);
    let mut alternate_pipeline = make_pipeline(&kernel, true);
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
    let cooperative = timings[1][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    println!(
        "COOPERATIVE_WHITENING agents={} ticks={ticks} rounds={TIMING_ROUNDS} serial_secs={serial:.9} cooperative_secs={cooperative:.9} serial_tps={:.3} cooperative_tps={:.3} speedup={:.3} exact_buffers={MUTABLE_BUFFERS}",
        kernel.agent_count, f64::from(ticks) / serial, f64::from(ticks) / cooperative, serial / cooperative,
    );
    Ok(())
}
