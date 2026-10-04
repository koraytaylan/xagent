//! Hardware-only feasibility probe for exact multi-workgroup dense stages.
//! Canonical features and tail surround four-lane tiled encode/prediction.
//! Binding 13 uses private staging so all thirteen production state buffers
//! remain directly comparable with the fused serial schedule.
//! `XAGENT_EXACT_TILE_COOPERATIVE=1` combines tiled dense work with the
//! cooperative whitening and interleaved recall tail.

use std::{error::Error, time::Instant};

use super::cooperative_whitening_validation::make_pipeline as make_reference_pipeline;
use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, REFRESH_CYCLES,
};
use super::*;

/// Sixteen outputs and four logical lanes produce a 64-thread workgroup.
const DEFAULT_TILE_OUTPUTS: u32 = 16;
/// The optional 128-thread shape reduces the number of workgroups per agent.
const LARGE_TILE_OUTPUTS: u32 = 32;
/// The unchanged kernel prefix uses two push-constant words.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Refresh and death boundaries precede the hundred-cycle continuation.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// A full memory window is populated before timing begins.
const WARMUP_CYCLES: u32 = 256;
/// One hundred complete cycles keep this feasibility probe quick.
const TIMED_CYCLES: u32 = 100;
/// Alternating arm order and five pairs give an unambiguous median.
const TIMING_ROUNDS: usize = 5;
/// The boundary fixture preserves one inactive agent without respawning it.
const INACTIVE_AGENT: u32 = 1;
/// Includes every production mutable storage buffer and the original scratch.
const MUTABLE_BUFFERS: usize = 13;
/// The fixture forces death initially and again at the first refresh boundary.
const EXPECTED_FORCED_DEATHS: f32 = 2.0;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

pub(super) struct TiledPipelines {
    features: wgpu::ComputePipeline,
    pub(super) encode: wgpu::ComputePipeline,
    pub(super) predictor: wgpu::ComputePipeline,
    tail: wgpu::ComputePipeline,
    pub(super) credit: wgpu::ComputePipeline,
    bind_groups: [wgpu::BindGroup; 2],
    _transient: wgpu::Buffer,
    outputs_per_tile: u32,
    tiles_per_agent: u32,
    pub(super) credit_workgroups_per_agent: u32,
    cooperative: bool,
}

fn requested_tile_outputs() -> TestResult<u32> {
    let outputs = std::env::var("XAGENT_EXACT_TILE_OUTPUTS")
        .map_or(Ok(DEFAULT_TILE_OUTPUTS), |value| value.parse::<u32>())?;
    if !matches!(outputs, DEFAULT_TILE_OUTPUTS | LARGE_TILE_OUTPUTS) {
        return Err("XAGENT_EXACT_TILE_OUTPUTS must be 16 or 32".into());
    }
    Ok(outputs)
}

fn make_transient_bind_group(
    kernel: &GpuKernel,
    layout: &wgpu::BindGroupLayout,
    transient: &wgpu::Buffer,
    config_index: usize,
) -> wgpu::BindGroup {
    // Array position is the common.wgsl binding. Only brain scratch differs
    // from the production group; both world-config buffers remain available.
    let buffers = [
        &kernel.agent_phys_buffer,
        &kernel.decision_buffer,
        &kernel.heightmap_buffer,
        &kernel.biome_buffer,
        &kernel.world_config_bufs[config_index],
        &kernel.food_state_buffer,
        &kernel.food_flags_buffer,
        &kernel.food_grid_buffer,
        &kernel.agent_grid_buffer,
        &kernel.collision_scratch_buffer,
        &kernel.sensory_buffer,
        &kernel.brain_state_buffer,
        &kernel.pattern_buffer,
        transient,
        &kernel.brain_config_buffer,
        &kernel.dispatch_args_buffer,
        &kernel.trail_ring_buffer,
        &kernel._sensory_next_buffer,
    ];
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: u32::try_from(binding).unwrap(),
            resource: buffer.as_entire_binding(),
        })
        .collect();
    kernel.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("exact_tiled_transient_bind_group"),
        layout,
        entries: &entries,
    })
}

pub(super) fn make_pipelines(
    kernel: &GpuKernel,
    outputs: u32,
    cooperative: bool,
) -> TiledPipelines {
    let common = include_str!("../shaders/kernel/common.wgsl");
    let passes = compose_brain_passes(cooperative);
    assert!(passes.contains("const DENSE_INNER_LANES: u32 = 4u;"));
    let brain_source = apply_subgroup_markers(
        &[
            common,
            passes.as_str(),
            include_str!("../shaders/kernel/exact_tiled_brain.wgsl"),
        ]
        .join("\n"),
        kernel.has_subgroup,
    );
    let dense_source = [
        common,
        include_str!("../shaders/kernel/exact_tiled_dense.wgsl"),
    ]
    .join("\n");
    let make_module = |label, source: String| {
        kernel
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
    };
    let brain_label = if cooperative {
        "exact_tiled_cooperative_brain_probe"
    } else {
        "exact_tiled_brain_probe"
    };
    let brain_module = make_module(brain_label, brain_source);
    let dense_module = make_module("exact_tiled_dense_probe", dense_source);
    let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
    let pipeline_layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("exact_tiled_probe_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    let constants = vision_override_constants(&kernel.layout);
    let mut dense_constants = constants.clone();
    dense_constants.insert("EXACT_TILE_OUTPUTS".into(), f64::from(outputs));
    let create = |module, entry, constants| {
        kernel
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    ..Default::default()
                },
                cache: None,
            })
    };
    let transient = kernel.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("exact_tiled_transient_scratch"),
        size: kernel.brain_scratch_buffer.size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_groups = std::array::from_fn(|index| {
        make_transient_bind_group(kernel, &bind_layout, &transient, index)
    });
    let dimensions = u32::try_from(ENCODED_DIMENSION).unwrap();
    assert_eq!(dimensions % outputs, 0);
    TiledPipelines {
        features: create(&brain_module, "exact_tiled_features", &constants),
        encode: create(&dense_module, "exact_tiled_encode", &dense_constants),
        predictor: create(&dense_module, "exact_tiled_predictor", &dense_constants),
        tail: create(&brain_module, "exact_tiled_tail", &constants),
        credit: create(
            &dense_module,
            "exact_tiled_encoder_credit",
            &dense_constants,
        ),
        bind_groups,
        _transient: transient,
        outputs_per_tile: outputs,
        tiles_per_agent: dimensions / outputs,
        credit_workgroups_per_agent: dimensions / outputs,
        cooperative,
    }
}

fn record_tiled_cycle(
    kernel: &GpuKernel,
    pipelines: &TiledPipelines,
    pass: &mut wgpu::ComputePass<'_>,
    tick: u32,
) {
    kernel.record_kernel_cycle(pass, u64::from(tick), 0);
    // Dispatch boundaries publish storage writes across all workgroups.
    // Features use canonical extraction/cortex/adaptation exactly once.
    pass.set_pipeline(&pipelines.features);
    pass.dispatch_workgroups(kernel.agent_count, 1, 1);
    pass.set_pipeline(&pipelines.encode);
    pass.dispatch_workgroups(kernel.agent_count, pipelines.tiles_per_agent, 1);
    // Predictor inputs are unchanged by homeostasis/recall, and its weights
    // are disjoint from their writes, so it can precede the canonical tail.
    pass.set_pipeline(&pipelines.predictor);
    pass.dispatch_workgroups(kernel.agent_count, pipelines.tiles_per_agent, 1);
    pass.set_pipeline(&pipelines.tail);
    pass.dispatch_workgroups(kernel.agent_count, 1, 1);
    // Encoder weights are next consumed by the next cycle's encoder. Credit
    // and the current feature vector are unchanged by the intervening tail.
    pass.set_pipeline(&pipelines.credit);
    pass.dispatch_workgroups(kernel.agent_count, pipelines.credit_workgroups_per_agent, 1);
    let stride = kernel.brain_tick_stride;
    pass.set_pipeline(&kernel.global_pipeline);
    pass.set_push_constants(0, bytemuck::cast_slice(&[tick + stride, stride]));
    pass.dispatch_workgroups(1, 1, 1);
    pass.set_pipeline(&kernel.vision_pipeline);
    pass.dispatch_workgroups(kernel.vision_workgroups, 1, 1);
}

fn dispatch_tiled(
    kernel: &mut GpuKernel,
    pipelines: &TiledPipelines,
    start_tick: u32,
    cycles: u32,
) {
    assert_eq!(kernel.vision_stride, 1);
    let stride = kernel.brain_tick_stride;
    kernel.upload_world_config(u64::from(start_tick), stride);
    let mut completed = 0;
    while completed < cycles {
        let chunk = (cycles - completed).min(MAX_FUSED_BATCHES);
        let mut encoder = kernel.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &pipelines.bind_groups[kernel.active_config_index], &[]);
            for cycle in completed..completed + chunk {
                record_tiled_cycle(kernel, pipelines, &mut pass, start_tick + cycle * stride);
            }
        }
        kernel.queue.submit([encoder.finish()]);
        completed += chunk;
    }
    kernel.active_config_index = 1 - kernel.active_config_index;
}

pub(super) fn advance(
    kernel: &mut GpuKernel,
    tiled: Option<&TiledPipelines>,
    start_tick: u32,
    cycles: u32,
) {
    if let Some(pipelines) = tiled {
        dispatch_tiled(kernel, pipelines, start_tick, cycles);
    } else {
        kernel.dispatch_ticks(u64::from(start_tick), cycles * kernel.brain_tick_stride);
    }
    kernel.poll_wait();
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn exact_tiled_matches_complete_serial_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    // Explicitly compile the original brain even if production optimization
    // environment flags were inherited by GpuKernel construction.
    kernel.kernel_pipeline = make_reference_pipeline(&kernel, false);
    let cooperative = std::env::var("XAGENT_EXACT_TILE_COOPERATIVE").as_deref() == Ok("1");
    let pipelines = make_pipelines(&kernel, requested_tile_outputs()?, cooperative);
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
        advance(&mut kernel, None, tick, cycles);
        expected.push(capture_state(&kernel)?);
        cycle += cycles;
    }
    assert!(
        kernel.read_full_state_blocking()[P_DEATH_COUNT] >= deaths_before + EXPECTED_FORCED_DEATHS
    );
    restore(&mut kernel, &initial);
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
        "EXACT_TILED_PARITY outputs_per_tile={} cooperative={} reference=original_serial cycles={cycle} exact_buffers={MUTABLE_BUFFERS} dead_agent_unchanged=true death_boundaries=2",
        pipelines.outputs_per_tile, pipelines.cooperative,
    );
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_exact_tiled_against_serial() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    kernel.kernel_pipeline = make_reference_pipeline(&kernel, false);
    let cooperative = std::env::var("XAGENT_EXACT_TILE_COOPERATIVE").as_deref() == Ok("1");
    let pipelines = make_pipelines(&kernel, requested_tile_outputs()?, cooperative);
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
    let tiled = timings[1][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    println!(
        "EXACT_TILED agents={} outputs_per_tile={} workgroups_per_agent={} cooperative={} reference=original_serial ticks={ticks} rounds={TIMING_ROUNDS} serial_secs={serial:.9} tiled_secs={tiled:.9} serial_tps={:.3} tiled_tps={:.3} speedup={:.3} exact_buffers={MUTABLE_BUFFERS}",
        kernel.agent_count, pipelines.outputs_per_tile, pipelines.tiles_per_agent, pipelines.cooperative,
        f64::from(ticks) / serial, f64::from(ticks) / tiled, serial / tiled,
    );
    Ok(())
}
