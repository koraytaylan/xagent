//! Hardware-only FP32 reassociation experiment for tiled dense brain stages.
//! The baseline is the explicit cooperative-whitening/fused-predictor main.
//! An exact tiled control separates schedule cost from wider balanced sums.
//! Candidates preserve all phases, weight-update formulas and FP32 clamps.
//! Single-cycle numeric reports and longer behavioral reports intentionally
//! distinguish rounding differences from exact-control failures.

use std::{error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::exact_tiled_validation::{advance, make_pipelines, TiledPipelines};
use super::predictor_fusion::fuse_inline_predictor;
use super::rounding_validation::{assert_inactive_agent_unchanged, compare_rounding_state};
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, REFRESH_CYCLES,
};
use super::*;

/// Push constants retain the standard starting tick and complete pass limit.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// The measured exact tiled control uses thirty-two outputs and four lanes.
const CONTROL_OUTPUTS: u32 = 32;
/// Both candidates fill one portable 256-thread workgroup.
const BALANCED_SHAPES: [(u32, u32); 2] = [(16, 16), (8, 32)];
/// Monolithic, exact tiled, and two balanced tiled schedules.
const ARM_COUNT: usize = 4;
/// The first arm retains the current optimized monolithic brain.
const BASELINE_ARM: usize = 0;
/// This arm changes scheduling but retains the reference addition order.
const EXACT_TILED_ARM: usize = 1;
/// Stable labels distinguish pipeline statistics and numeric/timing reports.
const ARM_NAMES: [&str; ARM_COUNT] = [
    "optimized_monolithic",
    "exact_tiled_32x4",
    "balanced_tiled_16x16",
    "balanced_tiled_8x32",
];
/// Includes initial single-cycle rounding, refresh boundaries and two deaths.
const VALIDATION_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Populate the entire recent-history window before mature-state comparisons.
const WARMUP_CYCLES: u32 = 256;
/// Full-cycle trials include physics, global world work and production vision.
const TIMED_CYCLES: u32 = 100;
/// Rotate all four arms across five trials and report the median per arm.
const TIMING_ROUNDS: usize = 5;
/// The boundary fixture preserves this inactive agent without respawning it.
const INACTIVE_AGENT: u32 = 1;
/// The fixture introduces two deaths, including one on a refresh boundary.
const EXPECTED_FORCED_DEATHS: f32 = 2.0;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn make_monolithic_pipeline(kernel: &GpuKernel) -> wgpu::ComputePipeline {
    let passes = fuse_inline_predictor(&compose_brain_passes(true));
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
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("balanced_dense_monolithic_reference"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let layout = pipeline_layout(kernel);
    let constants = vision_override_constants(&kernel.layout);
    create_pipeline(kernel, &module, &layout, &constants, "kernel_tick")
}

fn pipeline_layout(kernel: &GpuKernel) -> wgpu::PipelineLayout {
    let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
    kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("balanced_dense_pipeline_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        })
}

fn create_pipeline(
    kernel: &GpuKernel,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    constants: &HashMap<String, f64>,
    entry: &str,
) -> wgpu::ComputePipeline {
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(layout),
            module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            cache: None,
        })
}

fn make_balanced_pipelines(kernel: &GpuKernel, outputs: u32, lanes: u32) -> TiledPipelines {
    assert!(BALANCED_SHAPES.contains(&(outputs, lanes)));
    assert!(lanes.is_power_of_two());
    assert_eq!(outputs * lanes, BRAIN_WORKGROUP_THREADS);
    assert_eq!(u32::try_from(ENCODED_DIMENSION).unwrap() % outputs, 0);
    let mut pipelines = make_pipelines(kernel, outputs, true);
    let source = [
        include_str!("../shaders/kernel/common.wgsl"),
        include_str!("../shaders/kernel/balanced_tiled_dense.wgsl"),
    ]
    .join("\n");
    let label = format!("balanced_dense_{outputs}x{lanes}");
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let layout = pipeline_layout(kernel);
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("BALANCED_TILE_OUTPUTS".into(), f64::from(outputs));
    constants.insert("BALANCED_INNER_LANES".into(), f64::from(lanes));
    pipelines.encode = create_pipeline(
        kernel,
        &module,
        &layout,
        &constants,
        "balanced_tiled_encode",
    );
    pipelines.predictor = create_pipeline(
        kernel,
        &module,
        &layout,
        &constants,
        "balanced_tiled_predictor",
    );
    pipelines
}

fn prepare() -> (GpuKernel, [Option<TiledPipelines>; ARM_COUNT]) {
    let mut kernel = prepare_kernel();
    kernel.kernel_pipeline = make_monolithic_pipeline(&kernel);
    let tiled = [
        None,
        Some(make_pipelines(&kernel, CONTROL_OUTPUTS, true)),
        Some(make_balanced_pipelines(
            &kernel,
            BALANCED_SHAPES[0].0,
            BALANCED_SHAPES[0].1,
        )),
        Some(make_balanced_pipelines(
            &kernel,
            BALANCED_SHAPES[1].0,
            BALANCED_SHAPES[1].1,
        )),
    ];
    (kernel, tiled)
}

fn compare_arm(
    kernel: &GpuKernel,
    reference: &[Vec<u8>],
    actual: &[Vec<u8>],
    arm: usize,
    case: &str,
) {
    let label = format!("{case}_{}", ARM_NAMES[arm]);
    if arm == EXACT_TILED_ARM {
        assert_state_equal(kernel, reference, actual);
    }
    compare_rounding_state(kernel, reference, actual, &label);
}

#[test]
#[ignore = "FP32 rounding and behavior diagnostic; requires a GPU"]
fn balanced_dense_reports_single_cycle_and_behavioral_error() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, pipelines) = prepare();
    prepare_boundary_scene(&kernel);
    let initial = checkpoint(&kernel);
    let initial_state = capture_state(&kernel)?;
    let deaths_before = kernel.read_full_state_blocking()[P_DEATH_COUNT];
    let mut expected = Vec::with_capacity(VALIDATION_CHUNKS.len());
    let mut cycle = 0;
    for cycles in VALIDATION_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(&kernel);
        }
        let tick = cycle * kernel.brain_tick_stride;
        advance(&mut kernel, None, tick, cycles);
        expected.push(capture_state(&kernel)?);
        cycle += cycles;
    }
    assert!(
        kernel.read_full_state_blocking()[P_DEATH_COUNT] >= deaths_before + EXPECTED_FORCED_DEATHS
    );
    for arm in 1..ARM_COUNT {
        restore(&mut kernel, &initial);
        cycle = 0;
        for (cycles, reference) in VALIDATION_CHUNKS.into_iter().zip(&expected) {
            if cycle == REFRESH_CYCLES {
                force_death(&kernel);
            }
            let tick = cycle * kernel.brain_tick_stride;
            advance(&mut kernel, pipelines[arm].as_ref(), tick, cycles);
            cycle += cycles;
            let actual = capture_state(&kernel)?;
            let case = format!("fresh_cycles_{cycle}");
            compare_arm(&kernel, reference, &actual, arm, &case);
            assert_inactive_agent_unchanged(
                &kernel,
                &initial_state,
                &actual,
                INACTIVE_AGENT,
                &case,
            );
        }
    }
    // A second restored checkpoint measures one cycle with mature recall and
    // trained weights, separately from accumulated trajectory divergence.
    restore(&mut kernel, &initial);
    advance(&mut kernel, None, 0, WARMUP_CYCLES);
    let mature = checkpoint(&kernel);
    let tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    advance(&mut kernel, None, tick, 1);
    let reference = capture_state(&kernel)?;
    for arm in 1..ARM_COUNT {
        restore(&mut kernel, &mature);
        advance(&mut kernel, pipelines[arm].as_ref(), tick, 1);
        compare_arm(
            &kernel,
            &reference,
            &capture_state(&kernel)?,
            arm,
            "mature_single_cycle",
        );
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run in release mode with --ignored --nocapture"]
fn benchmark_balanced_dense_against_fused_and_exact_tiled() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, pipelines) = prepare();
    advance(&mut kernel, None, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    let tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    let mut timings: [Vec<f64>; ARM_COUNT] = std::array::from_fn(|_| Vec::new());
    for round in 0..TIMING_ROUNDS {
        let mut states: [Option<Vec<Vec<u8>>>; ARM_COUNT] = std::array::from_fn(|_| None);
        for offset in 0..ARM_COUNT {
            let arm = (round + offset) % ARM_COUNT;
            restore(&mut kernel, &warm);
            let start = Instant::now();
            advance(&mut kernel, pipelines[arm].as_ref(), tick, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            states[arm] = Some(capture_state(&kernel)?);
        }
        for arm in 1..ARM_COUNT {
            let reference = states[BASELINE_ARM].as_ref().unwrap();
            let actual = states[arm].as_ref().unwrap();
            compare_arm(
                &kernel,
                reference,
                actual,
                arm,
                &format!("timed_continuation_round_{round}"),
            );
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let fused = timings[BASELINE_ARM][TIMING_ROUNDS / 2];
    let exact_tiled = timings[EXACT_TILED_ARM][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    for (name, samples) in ARM_NAMES.iter().zip(timings) {
        let seconds = samples[TIMING_ROUNDS / 2];
        println!(
            "BALANCED_DENSE variant={name} agents={} ticks={ticks} rounds={TIMING_ROUNDS} seconds={seconds:.9} tps={:.3} speedup_vs_fused={:.3} speedup_vs_exact_tiled={:.3} precision=fp32",
            kernel.agent_count, f64::from(ticks) / seconds, fused / seconds, exact_tiled / seconds,
        );
    }
    Ok(())
}
