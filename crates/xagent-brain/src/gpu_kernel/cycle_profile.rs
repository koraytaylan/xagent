//! Hardware-only profiling of the production fused cycle. Timestamp commands
//! surround actual dispatches within one compute pass. Checkpoint restoration
//! keeps every timing trial on identical state, and byte comparison against
//! `dispatch_ticks` checks the replay before its measurements are reported.

use std::{error::Error, time::Instant};

use glam::Vec3;
use rand::{rngs::StdRng, Rng, SeedableRng};

use super::*;

/// Matches the small-world population used in the latency investigation.
const PROFILE_AGENTS: u32 = 10;
/// Default-sized food population, spread through the full world.
const PROFILE_FOOD: usize = 104;
/// Enough brain cycles to populate memory before profiling the recall path.
const WARMUP_CYCLES: u32 = 256;
/// At the production command-buffer chunk limit, so the replay remains
/// one compute pass and one submit just like a production call of this size.
const PROFILE_CYCLES: u32 = MAX_FUSED_BATCHES;
/// An odd number of paired trials permits median reporting.
const PROFILE_ROUNDS: usize = 5;
/// Repeated configurations use exactly the same world and brain initial state.
const PROFILE_SEED: u64 = 42;
/// Full initial energy/integrity prevents an uninitialized inactive population.
const FULL_METER: f32 = 100.0;
/// Flat ground puts food at the usual height above the terrain.
const FOOD_HEIGHT: f32 = 0.35;
/// Spawn margin keeps initial agents and food clear of the world boundary.
const SPAWN_MARGIN: f32 = 5.0;
/// All brain phases execute in the production default.
const COMPLETE_BRAIN: u32 = 7;
/// All simulation phases execute in the production default.
const COMPLETE_PHASE_MASK: u32 = 7;
/// Converts wall seconds to the same unit as GPU timestamp durations.
const NANOS_PER_SECOND: f64 = 1_000_000_000.0;
/// Stage fractions are printed as percentages for readable bottleneck ranking.
const PERCENT_SCALE: f64 = 100.0;
const COMBINED_STAGES: [&str; 5] = [
    "claim",
    "kernel_prefix",
    "global",
    "brain_and_vision",
    "sensory_publish",
];
const STANDALONE_STAGES: [&str; 4] = ["claim", "kernel_and_brain", "global", "vision"];

type ProfileResult<T = ()> = Result<T, Box<dyn Error>>;

pub(super) struct Checkpoint {
    buffers: Vec<wgpu::Buffer>,
    active_config_index: usize,
}

struct Timings {
    stage_nanos: Vec<f64>,
    instrumented_wall_nanos: f64,
}

/// Every mutable GPU buffer touched by the fused cycle, including derived
/// scratch, registration, and trail state. Uniforms are regenerated per call.
fn state_buffers(kernel: &GpuKernel) -> [(&'static str, &wgpu::Buffer); 13] {
    [
        ("physics", &kernel.agent_phys_buffer),
        ("decisions", &kernel.decision_buffer),
        ("food", &kernel.food_state_buffer),
        ("food_flags_and_claims", &kernel.food_flags_buffer),
        ("food_grid", &kernel.food_grid_buffer),
        ("agent_grid_and_masks", &kernel.agent_grid_buffer),
        ("collision_scratch", &kernel.collision_scratch_buffer),
        ("sensory", &kernel.sensory_buffer),
        ("brain", &kernel.brain_state_buffer),
        ("brain_scratch", &kernel.brain_scratch_buffer),
        ("patterns", &kernel.pattern_buffer),
        ("trail_ring", &kernel.trail_ring_buffer),
        ("sensory_next", &kernel._sensory_next_buffer),
    ]
}

pub(super) fn checkpoint(kernel: &GpuKernel) -> Checkpoint {
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    let buffers = state_buffers(kernel)
        .iter()
        .map(|(label, source)| {
            let target = kernel.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: source.size(),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(source, 0, &target, 0, source.size());
            target
        })
        .collect();
    kernel.queue.submit([encoder.finish()]);
    kernel.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
    Checkpoint {
        buffers,
        active_config_index: kernel.active_config_index,
    }
}

pub(super) fn restore(kernel: &mut GpuKernel, checkpoint: &Checkpoint) {
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    for ((_, target), source) in state_buffers(kernel).iter().zip(&checkpoint.buffers) {
        encoder.copy_buffer_to_buffer(source, 0, target, 0, source.size());
    }
    kernel.queue.submit([encoder.finish()]);
    kernel.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
    kernel.active_config_index = checkpoint.active_config_index;
}

fn read_buffer(kernel: &GpuKernel, source: &wgpu::Buffer) -> ProfileResult<Vec<u8>> {
    let staging = kernel.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cycle_profile_readback"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    kernel.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    kernel.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
    receiver.recv()??;
    let bytes = slice.get_mapped_range().to_vec();
    staging.unmap();
    Ok(bytes)
}

pub(super) fn capture_state(kernel: &GpuKernel) -> ProfileResult<Vec<Vec<u8>>> {
    state_buffers(kernel)
        .iter()
        .map(|(_, buffer)| read_buffer(kernel, buffer))
        .collect()
}

pub(super) fn assert_state_equal(kernel: &GpuKernel, expected: &[Vec<u8>], actual: &[Vec<u8>]) {
    assert_eq!(actual.len(), expected.len());
    for (((label, _), expected), actual) in state_buffers(kernel).iter().zip(expected).zip(actual) {
        assert_eq!(actual.len(), expected.len(), "{label} buffer length");
        if let Some((byte, (actual, expected))) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .find(|(_, (actual, expected))| actual != expected)
        {
            panic!("profile replay changes {label} byte {byte}: actual {actual:02x} expected {expected:02x}");
        }
    }
}

pub(super) fn make_kernel() -> GpuKernel {
    let brain = BrainConfig {
        vision_stride: 1,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: PROFILE_SEED,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(PROFILE_AGENTS, PROFILE_FOOD, &brain, &world);
    // Only the fused production shape is replayed. Stage-skipping probes would
    // change the task being profiled and therefore are disabled explicitly.
    kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
    kernel.probe.skip_global = false;
    kernel.probe.skip_vision = false;
    kernel.probe.kernel_pass_limit = COMPLETE_BRAIN;
    kernel.reset_agents_seeded(&brain, PROFILE_SEED);
    let mut rng = StdRng::seed_from_u64(PROFILE_SEED);
    let half = world.world_size * 0.5 - SPAWN_MARGIN;
    let food: Vec<_> = (0..PROFILE_FOOD)
        .map(|_| {
            (
                rng.random_range(-half..half),
                FOOD_HEIGHT,
                rng.random_range(-half..half),
            )
        })
        .collect();
    kernel.upload_world(
        &vec![0.0; TERRAIN_VPS * TERRAIN_VPS],
        &vec![0; BIOME_GRID_RES * BIOME_GRID_RES],
        &food,
        &vec![false; PROFILE_FOOD],
        &vec![0.0; PROFILE_FOOD],
    );
    let agents: Vec<_> = (0..PROFILE_AGENTS)
        .map(|_| {
            (
                Vec3::new(
                    rng.random_range(-half..half),
                    1.0,
                    rng.random_range(-half..half),
                ),
                FULL_METER,
                FULL_METER,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agents);
    kernel
}

fn record_cycle(
    kernel: &GpuKernel,
    pass: &mut wgpu::ComputePass<'_>,
    queries: &wgpu::QuerySet,
    first_query: u32,
    tick: u64,
    combined: bool,
) -> ProfileResult {
    let pass_limit = if combined { 0 } else { COMPLETE_BRAIN };
    let kernel_push = [u32::try_from(tick)?, pass_limit];
    let mut query = first_query;
    pass.write_timestamp(queries, query);
    pass.set_pipeline(&kernel.kernel_claim_pipeline);
    pass.set_bind_group(0, &kernel.bind_groups[kernel.active_config_index], &[]);
    pass.set_push_constants(0, bytemuck::cast_slice(&kernel_push));
    pass.dispatch_workgroups(kernel.agent_count, 1, 1);
    query += 1;
    pass.write_timestamp(queries, query);

    let credit = kernel
        .global_credit
        .as_ref()
        .filter(|_| kernel.global_credit_active());
    if let Some(credit) = credit {
        assert!(!combined);
        pass.set_pipeline(&credit.main);
        pass.set_bind_group(0, &credit.bind_groups[kernel.active_config_index], &[]);
    } else {
        pass.set_pipeline(&kernel.kernel_pipeline);
    }
    pass.set_push_constants(0, bytemuck::cast_slice(&kernel_push));
    pass.dispatch_workgroups(kernel.agent_count, 1, 1);
    query += 1;
    pass.write_timestamp(queries, query);

    let batch_ticks = kernel.kernel_batch_size();
    let global_push = [u32::try_from(tick + u64::from(batch_ticks))?, batch_ticks];
    pass.set_pipeline(credit.map_or(&kernel.global_pipeline, |credit| &credit.global));
    pass.set_push_constants(0, bytemuck::cast_slice(&global_push));
    pass.dispatch_workgroups(credit.map_or(1, |credit| credit.global_workgroups), 1, 1);
    query += 1;
    pass.write_timestamp(queries, query);
    pass.set_bind_group(0, &kernel.bind_groups[kernel.active_config_index], &[]);

    if combined {
        let brain_push = [u32::try_from(tick)?, COMPLETE_BRAIN];
        pass.set_pipeline(&kernel.brain_vision_pipeline);
        pass.set_push_constants(0, bytemuck::cast_slice(&brain_push));
        pass.dispatch_workgroups(kernel.brain_vision_workgroups, 1, 1);
        query += 1;
        pass.write_timestamp(queries, query);
        pass.set_pipeline(&kernel.sensory_publish_pipeline);
        pass.dispatch_workgroups(kernel.agent_count, 1, 1);
    } else {
        pass.set_pipeline(&kernel.vision_pipeline);
        pass.dispatch_workgroups(kernel.vision_workgroups, 1, 1);
    }
    pass.write_timestamp(queries, query + 1);
    Ok(())
}

fn profile_cycles(
    kernel: &mut GpuKernel,
    start_tick: u64,
    combined: bool,
) -> ProfileResult<Timings> {
    assert_eq!(kernel.vision_stride, 1);
    assert!(PROFILE_CYCLES <= MAX_FUSED_BATCHES);
    let stages = if combined {
        COMBINED_STAGES.len()
    } else {
        STANDALONE_STAGES.len()
    };
    let queries_per_cycle = u32::try_from(stages + 1)?;
    let query_count = PROFILE_CYCLES.checked_mul(queries_per_cycle).unwrap();
    let queries = kernel.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("production_cycle_timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: query_count,
    });
    let results = kernel.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("production_cycle_timestamp_results"),
        size: u64::from(query_count) * u64::try_from(std::mem::size_of::<u64>())?,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
        mapped_at_creation: false,
    });
    let started = Instant::now();
    let batch_ticks = kernel.kernel_batch_size();
    kernel.upload_world_config_with_cycles(start_tick, batch_ticks, COMPLETE_PHASE_MASK, 1);
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &kernel.bind_groups[kernel.active_config_index], &[]);
        for cycle in 0..PROFILE_CYCLES {
            let tick = start_tick + u64::from(cycle) * u64::from(batch_ticks);
            record_cycle(
                kernel,
                &mut pass,
                &queries,
                cycle * queries_per_cycle,
                tick,
                combined,
            )?;
        }
    }
    encoder.resolve_query_set(&queries, 0..query_count, &results, 0);
    kernel.queue.submit([encoder.finish()]);
    kernel.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
    let instrumented_wall_nanos =
        started.elapsed().as_secs_f64() * NANOS_PER_SECOND / f64::from(PROFILE_CYCLES);
    kernel.active_config_index = 1 - kernel.active_config_index;
    let raw = read_buffer(kernel, &results)?;
    let timestamps: &[u64] = bytemuck::cast_slice(&raw);
    let mut stage_nanos = vec![0.0; stages];
    for cycle in timestamps.chunks_exact(usize::try_from(queries_per_cycle)?) {
        for (stage, &next) in cycle[1..].iter().enumerate() {
            assert!(next >= cycle[stage], "GPU timestamps must be monotonic");
            stage_nanos[stage] += (next - cycle[stage]) as f64
                * f64::from(kernel.queue.get_timestamp_period())
                / f64::from(PROFILE_CYCLES);
        }
    }
    Ok(Timings {
        stage_nanos,
        instrumented_wall_nanos,
    })
}

fn production_wall_time(kernel: &mut GpuKernel, start_tick: u64) -> f64 {
    let started = Instant::now();
    assert!(kernel.dispatch_ticks(start_tick, PROFILE_CYCLES * kernel.kernel_batch_size()));
    kernel.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
    started.elapsed().as_secs_f64() * NANOS_PER_SECOND / f64::from(PROFILE_CYCLES)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn profile_shape(mut kernel: GpuKernel, beside_requested: bool) -> ProfileResult {
    kernel.set_brain_beside_vision(beside_requested);
    let combined =
        beside_requested && !kernel.standalone_vision_required && !kernel.global_credit_active();
    let stages = if combined {
        &COMBINED_STAGES[..]
    } else {
        &STANDALONE_STAGES[..]
    };
    let warmup_ticks = WARMUP_CYCLES * kernel.kernel_batch_size();
    assert!(kernel.dispatch_ticks(0, warmup_ticks));
    kernel.poll_wait();
    let state = checkpoint(&kernel);
    let mut production_times = Vec::new();
    let mut instrumented_times = Vec::new();
    let mut stage_times = vec![Vec::new(); stages.len()];
    for round in 0..PROFILE_ROUNDS {
        let mut production_state = None;
        let mut profiled_state = None;
        // Alternate paired order while restoring identical warmed GPU state.
        for instrumented in if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            restore(&mut kernel, &state);
            if instrumented {
                let timings = profile_cycles(&mut kernel, u64::from(warmup_ticks), combined)?;
                instrumented_times.push(timings.instrumented_wall_nanos);
                for (samples, value) in stage_times.iter_mut().zip(timings.stage_nanos) {
                    samples.push(value);
                }
                if round == 0 {
                    profiled_state = Some(capture_state(&kernel)?);
                }
            } else {
                production_times.push(production_wall_time(&mut kernel, u64::from(warmup_ticks)));
                if round == 0 {
                    production_state = Some(capture_state(&kernel)?);
                }
            }
        }
        if let (Some(production), Some(profiled)) = (production_state, profiled_state) {
            assert_state_equal(&kernel, &production, &profiled);
        }
    }
    let shape = if combined { "combined" } else { "standalone" };
    let production_wall = median(&mut production_times);
    let instrumented_wall = median(&mut instrumented_times);
    let stage_medians: Vec<_> = stage_times
        .iter_mut()
        .map(|samples| median(samples))
        .collect();
    let gpu_cycle: f64 = stage_medians.iter().sum();
    assert!(gpu_cycle > 0.0, "the GPU interval must be positive");
    println!("CYCLE_PROFILE shape={shape} agents={} foods={} brain_tick_stride={} vision_stride={} warmed_cycles={WARMUP_CYCLES} timed_cycles={PROFILE_CYCLES} production_wall_ns={production_wall:.1} instrumented_wall_ns={instrumented_wall:.1} gpu_stage_sum_ns={gpu_cycle:.1} replay_all_buffers_equal=true", kernel.agent_count, kernel.food_count, kernel.brain_tick_stride, kernel.vision_stride);
    for (stage, nanos) in stages.iter().zip(stage_medians) {
        println!(
            "CYCLE_PROFILE_STAGE shape={shape} stage={stage} gpu_ns={nanos:.1} percent={:.2}",
            nanos / gpu_cycle * PERCENT_SCALE
        );
    }
    Ok(())
}

#[test]
#[ignore = "hardware GPU profile; run in release mode with --ignored --nocapture"]
fn profile_production_cycle_dispatches() -> ProfileResult {
    let _vulkan = vulkan_gate::enter();
    let kernel = make_kernel();
    assert!(
        kernel
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
        "dispatch-level profiling requires in-pass timestamp support"
    );
    let standalone_required = kernel.standalone_vision_required;
    println!("CYCLE_PROFILE_CONFIGURATION standalone_required={standalone_required} subgroup={} features={:?}", kernel.has_subgroup, kernel.device.features());
    profile_shape(kernel, !standalone_required)?;
    if !standalone_required {
        profile_shape(make_kernel(), false)?;
    }
    Ok(())
}

/// Measures one normal claim dispatch followed by one partially gated main
/// kernel. Callers must restore the warmed checkpoint before every invocation;
/// partial results are diagnostic state and must never feed another cycle.
fn profile_brain_prefix(
    kernel: &mut GpuKernel,
    tick: u64,
    limit: u32,
) -> ProfileResult<(f64, f64)> {
    /// Boundaries before claim, between dispatches, and after the main kernel.
    const QUERY_COUNT: u32 = 3;
    let queries = kernel.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("brain_prefix_timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: QUERY_COUNT,
    });
    let results = kernel.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brain_prefix_timestamp_results"),
        size: u64::from(QUERY_COUNT) * u64::try_from(std::mem::size_of::<u64>())?,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
        mapped_at_creation: false,
    });
    kernel.upload_world_config_with_cycles(
        tick,
        kernel.kernel_batch_size(),
        COMPLETE_PHASE_MASK,
        1,
    );
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &kernel.bind_groups[kernel.active_config_index], &[]);
        pass.write_timestamp(&queries, 0);
        pass.set_pipeline(&kernel.kernel_claim_pipeline);
        pass.set_push_constants(
            0,
            bytemuck::cast_slice(&[u32::try_from(tick)?, COMPLETE_BRAIN]),
        );
        pass.dispatch_workgroups(kernel.agent_count, 1, 1);
        pass.write_timestamp(&queries, 1);
        pass.set_pipeline(&kernel.kernel_pipeline);
        pass.set_push_constants(0, bytemuck::cast_slice(&[u32::try_from(tick)?, limit]));
        pass.dispatch_workgroups(kernel.agent_count, 1, 1);
        pass.write_timestamp(&queries, QUERY_COUNT - 1);
    }
    encoder.resolve_query_set(&queries, 0..QUERY_COUNT, &results, 0);
    kernel.queue.submit([encoder.finish()]);
    let bytes = read_buffer(kernel, &results)?;
    let timestamps: &[u64] = bytemuck::cast_slice(&bytes);
    assert!(
        timestamps.windows(2).all(|pair| pair[1] >= pair[0]),
        "GPU timestamps must be monotonic"
    );
    let period = f64::from(kernel.queue.get_timestamp_period());
    Ok((
        (timestamps[1] - timestamps[0]) as f64 * period,
        (timestamps[2] - timestamps[1]) as f64 * period,
    ))
}

#[test]
#[ignore = "measurement-only partial-brain diagnostic; run in release mode with --ignored --nocapture"]
fn profile_brain_pass_cumulative_costs() -> ProfileResult {
    let _vulkan = vulkan_gate::enter();
    /// Matches the seven cooperative gates in `brain_tick_inner`, plus its
    /// all-gates-disabled prefix containing the non-brain kernel work.
    const LIMIT_COUNT: usize = COMPLETE_BRAIN as usize + 1;
    /// Rotate limit ordering over seven trials to reduce clock/order bias.
    const PREFIX_ROUNDS: usize = 7;
    const NEW_STAGE: [&str; LIMIT_COUNT] = [
        "kernel_prefix",
        "features_cortex_adaptation",
        "encode",
        "habituation_homeostasis",
        "recall_score",
        "recall_topk",
        "predict_and_act",
        "learn_and_store",
    ];
    let mut kernel = make_kernel();
    // This diagnostic measures prefixes of the retained inline main shader.
    // The separate production-cycle profiler keeps the configured offload.
    kernel.global_credit = None;
    kernel.set_brain_beside_vision(false);
    assert!(
        kernel
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
        "prefix profiling requires in-pass timestamp support"
    );
    let warmup_ticks = WARMUP_CYCLES * kernel.kernel_batch_size();
    assert!(kernel.dispatch_ticks(0, warmup_ticks));
    kernel.poll_wait();
    let state = checkpoint(&kernel);
    let mut claims: Vec<_> = (0..LIMIT_COUNT)
        .map(|_| Vec::with_capacity(PREFIX_ROUNDS))
        .collect();
    let mut prefixes: Vec<_> = (0..LIMIT_COUNT)
        .map(|_| Vec::with_capacity(PREFIX_ROUNDS))
        .collect();
    for round in 0..PREFIX_ROUNDS {
        for offset in 0..LIMIT_COUNT {
            let limit = (round + offset) % LIMIT_COUNT;
            restore(&mut kernel, &state);
            let (claim, prefix) =
                profile_brain_prefix(&mut kernel, u64::from(warmup_ticks), u32::try_from(limit)?)?;
            claims[limit].push(claim);
            prefixes[limit].push(prefix);
        }
    }
    // Discard the final partial computation; no diagnostic result becomes
    // another cycle's input, including the caller-visible post-test state.
    restore(&mut kernel, &state);
    let mut previous = 0.0;
    for (limit, stage) in NEW_STAGE.iter().enumerate() {
        let claim = median(&mut claims[limit]);
        let prefix = median(&mut prefixes[limit]);
        let delta = prefix - previous;
        println!("BRAIN_PREFIX_PROFILE measurement_only=true restored_single_cycle=true agents={} pass_limit={limit} added_stage={stage} claim_gpu_ns={claim:.1} main_prefix_gpu_ns={prefix:.1} consecutive_delta_ns={delta:.1}", kernel.agent_count);
        previous = prefix;
    }
    Ok(())
}

/// Build an independent lane-ownership reference while keeping every logical
/// feature iteration, scalar operation, and left-associated reduction intact.
/// Exact occurrence checks make shader edits invalidate the reference loudly.
fn output_major_brain_reference() -> String {
    let mut source = include_str!("../shaders/kernel/brain_passes.wgsl").to_owned();
    for (coalesced, reference) in [
        (
            "let output_in_tile = tid % DENSE_OUTPUT_TILE;\n    let lane = tid / DENSE_OUTPUT_TILE;",
            "let output_in_tile = tid / DENSE_INNER_LANES;\n    let lane = tid % DENSE_INNER_LANES;",
        ),
        (
            "let base = output_in_tile;\n            let second_lane = base + DENSE_OUTPUT_TILE;\n            let third_lane = second_lane + DENSE_OUTPUT_TILE;\n            let fourth_lane = third_lane + DENSE_OUTPUT_TILE;\n            let reduced = s_dense_partials[base] + s_dense_partials[second_lane]\n                + s_dense_partials[third_lane] + s_dense_partials[fourth_lane];",
            "let base = tid;\n            let reduced = s_dense_partials[base] + s_dense_partials[base + 1u] + s_dense_partials[base + 2u] + s_dense_partials[base + 3u];",
        ),
        (
            "let output_in_tile_enc = tid % DENSE_OUTPUT_TILE;\n        let lane_enc = tid / DENSE_OUTPUT_TILE;",
            "let output_in_tile_enc = tid / DENSE_INNER_LANES;\n        let lane_enc = tid % DENSE_INNER_LANES;",
        ),
        (
            "override VC_SCRATCH_LEN: u32 = (1u - VISUAL_CORTEX_FEATURES_ACTIVE)\n    + VISUAL_CORTEX_FEATURES_ACTIVE * (3u * RETINA_PIXEL_COUNT + VISUAL_FEATURE_COUNT);",
            "override VC_SCRATCH_LEN: u32 = 3u * RETINA_PIXEL_COUNT + VISUAL_FEATURE_COUNT;",
        ),
        (
            "    if (VISUAL_CORTEX_FEATURES_ACTIVE == 0u) {\n        return;\n    }\n",
            "",
        ),
    ] {
        assert_eq!(source.matches(coalesced).count(), 1, "brain reference fragment must match exactly once: {coalesced}");
        source = source.replace(coalesced, reference);
    }
    source
}

fn reference_kernel_pipeline(kernel: &GpuKernel) -> wgpu::ComputePipeline {
    let brain_reference = output_major_brain_reference();
    let source = apply_subgroup_markers(
        &[
            include_str!("../shaders/kernel/common.wgsl"),
            brain_reference.as_str(),
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
            label: Some("output_major_brain_reference"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
    /// The kernel push constants are a tick and a cooperative-pass limit.
    const PUSH_WORDS: usize = 2;
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("output_major_kernel_reference_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..u32::try_from(std::mem::size_of::<[u32; PUSH_WORDS]>()).unwrap(),
            }],
        });
    let constants = vision_override_constants(&kernel.layout);
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("output_major_kernel_reference"),
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

fn assert_encoder_weights_changed(kernel: &GpuKernel, before: &[u8], after: &[u8]) {
    let before: &[u32] = bytemuck::cast_slice(before);
    let after: &[u32] = bytemuck::cast_slice(after);
    let weights = kernel
        .layout
        .feature_count
        .checked_mul(ENCODED_DIMENSION)
        .unwrap();
    let updated = before
        .chunks_exact(kernel.layout.brain_stride)
        .zip(after.chunks_exact(kernel.layout.brain_stride))
        .any(|(before, after)| {
            before[O_ENC_WEIGHTS..O_ENC_WEIGHTS + weights]
                != after[O_ENC_WEIGHTS..O_ENC_WEIGHTS + weights]
        });
    assert!(
        updated,
        "credit-update comparison must exercise actual encoder-weight changes"
    );
}

fn run_parity_cycles(kernel: &mut GpuKernel, start_tick: u64, cycles: u32, poll_each: bool) {
    let batch_ticks = kernel.kernel_batch_size();
    if poll_each {
        for cycle in 0..cycles {
            assert!(kernel.dispatch_ticks(
                start_tick + u64::from(cycle) * u64::from(batch_ticks),
                batch_ticks
            ));
            kernel.poll_wait();
        }
    } else {
        assert!(kernel.dispatch_ticks(start_tick, cycles * batch_ticks));
        kernel.poll_wait();
    }
}

#[test]
#[ignore = "hardware parity across complete cycles; run with --ignored --nocapture"]
fn coalesced_brain_matches_output_major_reference_all_state() -> ProfileResult {
    let _vulkan = vulkan_gate::enter();
    /// Default raw features have a remainder in the four-lane inner loop.
    const ODD_DEFAULT_FEATURES: usize = 267;
    /// Covers default vision and a larger field with odd row/column counts.
    const DIMENSIONS: [(u32, u32); 2] = [(8, 6), (13, 9)];
    /// Allows prediction credit and sensory adaptation to develop before the
    /// two independent lane mappings continue from one common checkpoint.
    const PARITY_WARMUP_CYCLES: u32 = 16;
    /// Multiple production chunks exercise repeated learning and state reuse.
    const PARITY_CYCLES: u32 = 2 * MAX_FUSED_BATCHES;
    /// Cortex kernels are much heavier; bound each driver job to one agent
    /// and one cycle while retaining several rounds of prediction and credit.
    const CORTEX_PARITY_CYCLES: u32 = 8;
    /// Establish previous sensory/prediction values before the cortex arms.
    const CORTEX_WARMUP_CYCLES: u32 = 2;
    for (width, height) in DIMENSIONS {
        for cortex in [false, true] {
            let brain = BrainConfig {
                vision_width: width,
                vision_height: height,
                visual_cortex_enabled: cortex,
                vision_stride: 1,
                ..BrainConfig::default()
            };
            let agents = if cortex { 1 } else { PROFILE_AGENTS };
            let mut kernel = GpuKernel::new(agents, PROFILE_FOOD, &brain, &WorldConfig::default());
            // This comparison swaps the inline main's encoder layout directly.
            kernel.global_credit = None;
            kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
            kernel.set_brain_beside_vision(false);
            kernel.probe.kernel_pass_limit = COMPLETE_BRAIN;
            kernel.probe.skip_global = false;
            kernel.probe.skip_vision = false;
            kernel.reset_agents_seeded(&brain, PROFILE_SEED);
            super::vision_validation::upload_random_scene(&kernel, PROFILE_SEED, true, false);
            if (width, height) == DIMENSIONS[0] && !cortex {
                assert_eq!(kernel.layout.feature_count, ODD_DEFAULT_FEATURES);
            }
            let reference = reference_kernel_pipeline(&kernel);
            let warmup_cycles = if cortex {
                CORTEX_WARMUP_CYCLES
            } else {
                PARITY_WARMUP_CYCLES
            };
            let test_cycles = if cortex {
                CORTEX_PARITY_CYCLES
            } else {
                PARITY_CYCLES
            };
            let warmup_ticks = warmup_cycles * kernel.kernel_batch_size();
            run_parity_cycles(&mut kernel, 0, warmup_cycles, cortex);
            let initial_brain = read_buffer(&kernel, &kernel.brain_state_buffer)?;
            let state = checkpoint(&kernel);
            run_parity_cycles(&mut kernel, u64::from(warmup_ticks), test_cycles, cortex);
            let coalesced_state = capture_state(&kernel)?;
            let learned_brain = read_buffer(&kernel, &kernel.brain_state_buffer)?;
            assert_encoder_weights_changed(&kernel, &initial_brain, &learned_brain);

            restore(&mut kernel, &state);
            let coalesced = std::mem::replace(&mut kernel.kernel_pipeline, reference);
            run_parity_cycles(&mut kernel, u64::from(warmup_ticks), test_cycles, cortex);
            let reference_state = capture_state(&kernel)?;
            assert_state_equal(&kernel, &reference_state, &coalesced_state);
            kernel.kernel_pipeline = coalesced;
            println!("BRAIN_LAYOUT_PARITY vision={width}x{height} cortex={cortex} agents={agents} feature_count={} cycles={test_cycles} all13buffers_equal=true encoder_weights_changed=true", kernel.layout.feature_count);
        }
    }
    Ok(())
}
