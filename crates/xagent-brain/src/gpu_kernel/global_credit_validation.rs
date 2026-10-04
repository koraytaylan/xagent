//! Hardware-only encoder-credit offload into the existing global dispatch.
//! The optimized monolithic brain publishes its actual adapted features and
//! omits only its independent encoder-weight updates. A global workgroup runs
//! unchanged world phases beside pointwise credit workgroups. Both schedules
//! retain four dispatches per full cycle. Transient binding 13 keeps every
//! production mutable buffer directly comparable, including brain scratch.

use std::{error::Error, time::Instant};

use super::cached_combined_validation::prepare;
use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::dense_prefetch_validation::make_pipeline;
use super::exact_tiled_validation::make_transient_bind_group;
use super::predictor_fusion::fuse_inline_predictor;
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// Raw visual fields exercise different feature strides and a partial tile.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// Match the current production optimized encoder and predictor loops.
const PREFETCH_FACTOR: u32 = 8;
const PREDICTOR_LANES: u32 = 16;
/// One invocation updates one weight; the portable workgroup has 256 lanes.
const CREDIT_THREADS: u32 = 256;
/// The first workgroup keeps all original global synchronization local.
const WORLD_WORKGROUPS: u32 = 1;
/// Every complete cycle executes all seven cooperative brain phases.
const COMPLETE_BRAIN: u32 = 7;
/// Tick and pass limit (or trail interval) use two push-constant words.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Checkpoints surround death and refresh boundaries through cycle 100.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Populate episodic memory before the common timing checkpoint.
const WARMUP_CYCLES: u32 = 256;
/// Full-cycle timing includes both world phases and cached vision.
const TIMED_CYCLES: u32 = 100;
/// Alternate arm order across an odd number of paired trials.
const TIMING_ROUNDS: usize = 5;
/// The boundary fixture keeps this agent inactive without a pending respawn.
const INACTIVE_AGENT: u32 = 1;
/// Two forced deaths must have reached the reset path during parity.
const FORCED_DEATHS: f32 = 2.0;
/// Every mutable production buffer participates in exact comparisons.
const MUTABLE_BUFFERS: usize = 13;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

struct CreditPipelines {
    main: wgpu::ComputePipeline,
    global: wgpu::ComputePipeline,
    bind_groups: [wgpu::BindGroup; 2],
    _transient: wgpu::Buffer,
    global_workgroups: u32,
}

fn replace_once(source: &str, old: &str, new: &str) -> String {
    assert_eq!(
        source.matches(old).count(),
        1,
        "unique source target: {old}"
    );
    source.replacen(old, new, 1)
}

fn offloaded_brain() -> String {
    let fused = fuse_inline_predictor(&compose_brain_passes(true));
    let prefetch = dense_prefetch::prefetch_passes(&fused, PREFETCH_FACTOR);
    let original = predictor_width::wider_predictor(&prefetch, PREDICTOR_LANES);
    let begin = "    if (run_encoder_credit) {\n";
    let end = "    // ── Compute memory-key norm ONCE (memory reinforcement tiling)";
    assert_eq!(original.matches(begin).count(), 1);
    assert_eq!(original.matches(end).count(), 1);
    let first = original.find(begin).unwrap();
    let last = original.find(end).unwrap();
    assert!(first < last);
    let original_credit = &original[first..last];
    assert_eq!(original_credit.matches("O_ENC_WEIGHTS").count(), 2);
    assert!(!original_credit.contains("Barrier"));
    // These are the actual s_features after the single sensory adaptation,
    // never reconstructed from raw sensory or the already-updated mean.
    let publish = r"    if (run_encoder_credit) {
        let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;
        for (var feature = tid; feature < FEATURE_COUNT; feature += BRAIN_WORKGROUP_SIZE) {
            brain_scratch[agent_scratch + SCRATCH_FEATURES + feature] = s_features[feature];
        }
    }

";
    let source = replace_once(&original, original_credit, publish);
    assert_eq!(
        source.matches("workgroupBarrier();").count(),
        original.matches("workgroupBarrier();").count()
    );
    assert_eq!(
        source.matches("storageBarrier();").count(),
        original.matches("storageBarrier();").count()
    );
    // Subsequent reinforcement/store/replay code does not consume weights.
    assert!(!source[first + publish.len()..].contains("O_ENC_WEIGHTS"));
    source
}

fn global_source() -> String {
    let global = replace_once(
        include_str!("../shaders/kernel/global_tick.wgsl"),
        "@compute @workgroup_size(256)\nfn global_tick(@builtin(local_invocation_id) lid: vec3u) {\n    let tid = lid.x;",
        "fn global_world_inner(tid: u32) {",
    );
    let credit = replace_once(
        include_str!("../shaders/kernel/exact_pointwise_encoder_credit.wgsl"),
        "@compute @workgroup_size(POINTWISE_CREDIT_THREADS)\nfn exact_pointwise_encoder_credit(\n    @builtin(workgroup_id) wgid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n)",
        "fn exact_pointwise_encoder_credit(wgid: vec3<u32>, lid: vec3<u32>)",
    );
    // World stages read no brain data and do not change P_ALIVE. Credit only
    // reads that stable flag and writes encoder weights, so the two branches
    // require no cross-workgroup communication or synchronization.
    let world_phases = [
        include_str!("../shaders/kernel/phase_clear.wgsl"),
        include_str!("../shaders/kernel/phase_food_grid.wgsl"),
        include_str!("../shaders/kernel/phase_food_respawn.wgsl"),
        include_str!("../shaders/kernel/phase_agent_grid.wgsl"),
        include_str!("../shaders/kernel/phase_grid_order.wgsl"),
        include_str!("../shaders/kernel/phase_collision.wgsl"),
        include_str!("../shaders/kernel/phase_trail_sample.wgsl"),
    ]
    .join("\n");
    assert!(!world_phases.contains("brain_state"));
    [
        include_str!("../shaders/kernel/common.wgsl"),
        world_phases.as_str(),
        global.as_str(),
        credit.as_str(),
        include_str!("../shaders/kernel/global_encoder_credit_tick.wgsl"),
    ]
    .join("\n")
}

fn make_pipelines(kernel: &GpuKernel) -> CreditPipelines {
    let main = make_pipeline(kernel, &offloaded_brain(), "credit_offloaded_main");
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("global_encoder_credit_tick"),
            source: wgpu::ShaderSource::Wgsl(global_source().into()),
        });
    let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("global_encoder_credit_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    let global = kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("global_encoder_credit_tick"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("global_encoder_credit_tick"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });
    let transient = kernel.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("global_credit_adapted_features"),
        size: kernel.brain_scratch_buffer.size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_groups = std::array::from_fn(|index| {
        make_transient_bind_group(kernel, &bind_layout, &transient, index)
    });
    let weights_per_agent = u32::try_from(
        kernel
            .layout
            .feature_count
            .checked_mul(ENCODED_DIMENSION)
            .unwrap(),
    )
    .unwrap();
    let global_workgroups = weights_per_agent
        .div_ceil(CREDIT_THREADS)
        .checked_mul(kernel.agent_count)
        .and_then(|groups| groups.checked_add(WORLD_WORKGROUPS))
        .unwrap();
    assert!(global_workgroups <= MAX_DISPATCH_WORKGROUPS);
    CreditPipelines {
        main,
        global,
        bind_groups,
        _transient: transient,
        global_workgroups,
    }
}

fn dispatch_offloaded(
    kernel: &mut GpuKernel,
    pipelines: &CreditPipelines,
    cycle: u32,
    cycles: u32,
) {
    assert_eq!(kernel.vision_stride, 1);
    let stride = kernel.brain_tick_stride;
    let start_tick = cycle * stride;
    kernel.upload_world_config(u64::from(start_tick), stride);
    let mut completed = 0;
    while completed < cycles {
        let chunk = (cycles - completed).min(MAX_FUSED_BATCHES);
        let mut encoder = kernel.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &pipelines.bind_groups[kernel.active_config_index], &[]);
            for offset in completed..completed + chunk {
                let tick = start_tick + offset * stride;
                pass.set_pipeline(&kernel.kernel_claim_pipeline);
                pass.set_push_constants(0, bytemuck::cast_slice(&[tick, COMPLETE_BRAIN]));
                pass.dispatch_workgroups(kernel.agent_count, 1, 1);
                pass.set_pipeline(&pipelines.main);
                // Separately created layouts clear push constants on a wgpu
                // pipeline switch, even when their ranges are compatible.
                pass.set_push_constants(0, bytemuck::cast_slice(&[tick, COMPLETE_BRAIN]));
                pass.dispatch_workgroups(kernel.agent_count, 1, 1);
                // The dispatch boundary publishes the adapted features and
                // decision credit to every pointwise workgroup.
                pass.set_pipeline(&pipelines.global);
                pass.set_push_constants(0, bytemuck::cast_slice(&[tick + stride, stride]));
                pass.dispatch_workgroups(pipelines.global_workgroups, 1, 1);
                pass.set_pipeline(&kernel.vision_pipeline);
                pass.dispatch_workgroups(kernel.vision_workgroups, 1, 1);
            }
        }
        kernel.queue.submit([encoder.finish()]);
        completed += chunk;
    }
    kernel.active_config_index = 1 - kernel.active_config_index;
}

fn advance(kernel: &mut GpuKernel, pipelines: Option<&CreditPipelines>, cycle: u32, cycles: u32) {
    if let Some(pipelines) = pipelines {
        dispatch_offloaded(kernel, pipelines, cycle, cycles);
    } else {
        kernel.dispatch_ticks(
            u64::from(cycle * kernel.brain_tick_stride),
            cycles * kernel.brain_tick_stride,
        );
    }
    kernel.poll_wait();
}

fn trajectory(
    kernel: &mut GpuKernel,
    pipelines: Option<&CreditPipelines>,
) -> TestResult<Vec<State>> {
    let mut states = Vec::with_capacity(PARITY_CHUNKS.len());
    let mut cycle = 0;
    for cycles in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, pipelines, cycle, cycles);
        states.push(capture_state(kernel)?);
        cycle += cycles;
    }
    Ok(states)
}

#[test]
#[ignore = "requires GPU; run explicitly with --ignored --nocapture"]
fn encoder_credit_beside_global_matches_all_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare(width, height, true);
        let pipelines = make_pipelines(&kernel);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let expected = trajectory(&mut kernel, None)?;
        restore(&mut kernel, &saved);
        let actual = trajectory(&mut kernel, Some(&pipelines))?;
        for (expected, actual) in expected.iter().zip(&actual) {
            assert_state_equal(&kernel, expected, actual);
            assert_inactive_agent_unchanged(
                &kernel,
                &initial,
                actual,
                INACTIVE_AGENT,
                "global encoder credit",
            );
        }
        assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= FORCED_DEATHS);
        restore(&mut kernel, &saved);
        let repeated = trajectory(&mut kernel, Some(&pipelines))?;
        for (actual, repeated) in actual.iter().zip(&repeated) {
            assert_state_equal(&kernel, actual, repeated);
        }
        println!("GLOBAL_CREDIT_PARITY width={width} height={height} cycles=100 exact_buffers={MUTABLE_BUFFERS} global_workgroups={} partial_weight_tile={} brain=coop_prefetch8_lanes16 death_refresh=true", pipelines.global_workgroups, kernel.layout.feature_count * ENCODED_DIMENSION % usize::try_from(CREDIT_THREADS)?);
    }
    Ok(())
}

#[test]
#[ignore = "requires GPU; run in release mode with --ignored --nocapture"]
fn benchmark_encoder_credit_beside_global() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let mut kernel = prepare(width, height, false);
    let pipelines = make_pipelines(&kernel);
    advance(&mut kernel, None, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    advance(&mut kernel, None, WARMUP_CYCLES, TIMED_CYCLES);
    let expected = capture_state(&kernel)?;
    restore(&mut kernel, &warm);
    advance(&mut kernel, Some(&pipelines), WARMUP_CYCLES, TIMED_CYCLES);
    assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
    let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..TIMING_ROUNDS {
        for offset in 0..timings.len() {
            let arm = (round + offset) % timings.len();
            restore(&mut kernel, &warm);
            let candidate = (arm != 0).then_some(&pipelines);
            let start = Instant::now();
            advance(&mut kernel, candidate, WARMUP_CYCLES, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let baseline = timings[0][TIMING_ROUNDS / 2];
    let candidate = timings[1][TIMING_ROUNDS / 2];
    println!("GLOBAL_CREDIT_TIMING cycles={TIMED_CYCLES} pairs={TIMING_ROUNDS} baseline_seconds={baseline:.9} candidate_seconds={candidate:.9} speedup={:.3} exact_buffers={MUTABLE_BUFFERS} dispatches_per_cycle=4 global_workgroups={}", baseline / candidate, pipelines.global_workgroups);
    Ok(())
}
