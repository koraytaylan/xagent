//! Hardware-only isolation of the visual whitening refresh. The normal
//! prefix precedes a small refresh dispatch and a cooperative brain dispatch;
//! all thirteen mutable buffers must match the unchanged serial schedule.

use std::{error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, make_kernel, restore};
use super::recall_validation::interleaved_brain_passes;
use super::*;

/// The full cooperative brain has seven passes.
const COMPLETE_BRAIN: u32 = 7;
/// Kernel push constants contain the starting tick and the brain pass limit.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Mirrors the schedule in common.wgsl; source assertions check the contract.
pub(super) const REFRESH_CYCLES: u32 = 20;
/// Checkpoints immediately before and after successive scheduled refreshes.
pub(super) const PARITY_CHUNKS: [u32; 6] = [1, 18, 1, 1, 19, 1];
/// A mature memory contains the entire 128-slot recent-experience window.
const WARMUP_CYCLES: u32 = 256;
/// Four production-sized command chunks amortize recording and waiting.
const TIMED_CYCLES: u32 = MAX_FUSED_BATCHES * 4;
/// Alternate execution order and report the median of an odd number of pairs.
const TIMING_ROUNDS: usize = 5;
/// A distinct agent remains inactive without requesting a respawn.
const INACTIVE_AGENT: u32 = 1;
/// Flat-world agents have full initial health, so zero energy forces death.
const NO_ENERGY: f32 = 0.0;
/// Symmetric off-diagonal entries ensure the refresh exercises Jacobi sweeps.
const COVARIANCE_DIAGONAL: f32 = 0.02;
/// Small off-diagonal covariance keeps the symmetric matrix positive definite.
const COVARIANCE_OFF_DIAGONAL: f32 = 0.001;
/// The fixture forces one death initially and another at the refresh boundary.
const EXPECTED_FORCED_DEATHS: f32 = 2.0;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct WhiteningPipelines {
    refresh: wgpu::ComputePipeline,
    brain: wgpu::ComputePipeline,
}

fn isolated_brain_source(interleaved_recall: bool) -> String {
    const REFRESH_BLOCK: &str = "    if (u32(tick) % VISION_WHITENING_REFRESH == 0u) {\n        refresh_vision_whitening(brain_base);\n    }\n";
    let passes = if interleaved_recall {
        interleaved_brain_passes()
    } else {
        include_str!("../shaders/kernel/brain_passes.wgsl").to_owned()
    };
    let common = include_str!("../shaders/kernel/common.wgsl");
    assert_eq!(passes.matches(REFRESH_BLOCK).count(), 1);
    assert_eq!(passes.matches("refresh_vision_whitening(").count(), 1);
    assert!(common.contains(&format!(
        "const VISION_WHITENING_REFRESH: u32 = {REFRESH_CYCLES}u;"
    )));
    let separated = passes.replacen(REFRESH_BLOCK, "", 1);
    assert!(!separated.contains("refresh_vision_whitening("));
    [
        common,
        separated.as_str(),
        include_str!("../shaders/kernel/brain_inner.wgsl"),
        include_str!("../shaders/kernel/whitening_refresh.wgsl"),
    ]
    .join("\n")
}

fn make_pipelines(kernel: &GpuKernel, interleaved_recall: bool) -> WhiteningPipelines {
    let source = apply_subgroup_markers(
        &isolated_brain_source(interleaved_recall),
        kernel.has_subgroup,
    );
    let label = if interleaved_recall {
        "whitening_recall_isolation_probe"
    } else {
        "whitening_isolation_probe"
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
    WhiteningPipelines {
        refresh: create("whitening_refresh"),
        brain: create("brain_without_whitening_refresh"),
    }
}

pub(super) fn prepare_kernel() -> GpuKernel {
    let mut kernel = make_kernel();
    kernel.set_brain_beside_vision(false);
    // Historical experiment baselines must remain the original serial brain
    // even when the surrounding process opts into production brain variants.
    kernel.kernel_pipeline = super::cooperative_whitening_validation::make_pipeline(&kernel, false);
    assert_eq!(kernel.vision_stride, 1);
    assert!(!kernel.layout.visual_cortex_enabled);
    kernel
}

fn record_isolated_cycle(
    kernel: &GpuKernel,
    pipelines: &WhiteningPipelines,
    pass: &mut wgpu::ComputePass<'_>,
    tick: u32,
) {
    kernel.record_kernel_cycle(pass, u64::from(tick), 0);
    pass.set_pipeline(&pipelines.refresh);
    pass.dispatch_workgroups(kernel.agent_count, 1, 1);
    pass.set_pipeline(&pipelines.brain);
    pass.set_push_constants(0, bytemuck::cast_slice(&[tick, COMPLETE_BRAIN]));
    pass.dispatch_workgroups(kernel.agent_count, 1, 1);
    let stride = kernel.brain_tick_stride;
    pass.set_pipeline(&kernel.global_pipeline);
    pass.set_push_constants(0, bytemuck::cast_slice(&[tick + stride, stride]));
    pass.dispatch_workgroups(1, 1, 1);
    pass.set_pipeline(&kernel.vision_pipeline);
    pass.dispatch_workgroups(kernel.vision_workgroups, 1, 1);
}

fn dispatch_isolated(
    kernel: &mut GpuKernel,
    pipelines: &WhiteningPipelines,
    start_tick: u32,
    cycles: u32,
) {
    let stride = kernel.brain_tick_stride;
    kernel.upload_world_config(u64::from(start_tick), stride);
    let mut completed = 0;
    while completed < cycles {
        let chunk = (cycles - completed).min(MAX_FUSED_BATCHES);
        let mut encoder = kernel.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &kernel.bind_groups[kernel.active_config_index], &[]);
            for cycle in completed..completed + chunk {
                record_isolated_cycle(kernel, pipelines, &mut pass, start_tick + cycle * stride);
            }
        }
        kernel.queue.submit([encoder.finish()]);
        completed += chunk;
    }
    kernel.active_config_index = 1 - kernel.active_config_index;
}

fn advance(
    kernel: &mut GpuKernel,
    isolated: Option<&WhiteningPipelines>,
    start_tick: u32,
    cycles: u32,
) {
    if let Some(pipelines) = isolated {
        dispatch_isolated(kernel, pipelines, start_tick, cycles);
    } else {
        kernel.dispatch_ticks(u64::from(start_tick), cycles * kernel.brain_tick_stride);
    }
    kernel.poll_wait();
}

pub(super) fn force_death(kernel: &GpuKernel) {
    kernel.write_agent_physics_fields(0, &[(P_ENERGY, NO_ENERGY)]);
}

pub(super) fn prepare_boundary_scene(kernel: &GpuKernel) {
    let fixed_base = fixed_tail_base(kernel.layout.brain_stride);
    let covariance = fixed_base + O_VISION_PATHWAY_COVARIANCE - O_PREDICTOR_CONTEXT_WEIGHT;
    let tick = fixed_base + O_TICK_COUNT - O_PREDICTOR_CONTEXT_WEIGHT;
    for agent in 0..kernel.agent_count {
        let mut state = kernel.read_agent_state(agent);
        state.brain_state[tick] = 0.0;
        for row in 0..VISION_PATHWAY_INPUTS {
            for column in 0..VISION_PATHWAY_INPUTS {
                state.brain_state[covariance + row * VISION_PATHWAY_INPUTS + column] =
                    if row == column {
                        COVARIANCE_DIAGONAL
                    } else {
                        COVARIANCE_OFF_DIAGONAL
                    };
            }
        }
        kernel.write_agent_state(agent, &state);
    }
    kernel.write_agent_physics_fields(INACTIVE_AGENT, &[(P_ALIVE, 0.0), (P_DIED_FLAG, 0.0)]);
    force_death(kernel);
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn isolated_whitening_matches_complete_serial_state() -> TestResult {
    check_isolated_whitening(false)
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn isolated_whitening_with_interleaved_recall_matches_complete_serial_state() -> TestResult {
    check_isolated_whitening(true)
}

fn check_isolated_whitening(interleaved_recall: bool) -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    let pipelines = make_pipelines(&kernel, interleaved_recall);
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
        advance(&mut kernel, None, tick, cycles);
        expected.push(capture_state(&kernel)?);
        cycle += cycles;
    }
    restore(&mut kernel, &initial);
    cycle = 0;
    for (cycles, expected) in PARITY_CHUNKS.into_iter().zip(&expected) {
        if cycle == REFRESH_CYCLES {
            force_death(&kernel);
        }
        let tick = cycle * kernel.brain_tick_stride;
        advance(&mut kernel, Some(&pipelines), tick, cycles);
        let actual = capture_state(&kernel)?;
        assert_state_equal(&kernel, expected, &actual);
        cycle += cycles;
    }
    let inactive_after = kernel.read_agent_state(INACTIVE_AGENT);
    assert_eq!(
        bytemuck::cast_slice::<f32, u32>(&inactive_before.brain_state),
        bytemuck::cast_slice::<f32, u32>(&inactive_after.brain_state)
    );
    let physics = kernel.read_full_state_blocking();
    assert!(physics[P_DEATH_COUNT] >= EXPECTED_FORCED_DEATHS);
    assert!(physics[P_ALIVE] >= 0.5);
    println!("WHITENING_PARITY interleaved_recall={interleaved_recall} cycles={cycle} exact_buffers=13 dead_agent_unchanged=true");
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_isolated_whitening_against_serial() -> TestResult {
    benchmark_isolated_whitening(false)
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_isolated_whitening_with_interleaved_recall_against_serial() -> TestResult {
    benchmark_isolated_whitening(true)
}

fn benchmark_isolated_whitening(interleaved_recall: bool) -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    let pipelines = make_pipelines(&kernel, interleaved_recall);
    advance(&mut kernel, None, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    let tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    let mut timings = [Vec::new(), Vec::new()];
    for round in 0..TIMING_ROUNDS {
        let mut states = [None, None];
        for offset in 0..timings.len() {
            let arm = (round + offset) % timings.len();
            restore(&mut kernel, &warm);
            let start = Instant::now();
            advance(
                &mut kernel,
                (arm == 1).then_some(&pipelines),
                tick,
                TIMED_CYCLES,
            );
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
    let isolated = timings[1][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    println!(
        "WHITENING_ISOLATION interleaved_recall={interleaved_recall} agents={} ticks={ticks} rounds={TIMING_ROUNDS} serial_secs={serial:.9} isolated_secs={isolated:.9} serial_tps={:.3} isolated_tps={:.3} speedup={:.3} exact_buffers=13",
        kernel.agent_count, f64::from(ticks) / serial, f64::from(ticks) / isolated, serial / isolated,
    );
    Ok(())
}
