//! Opt-in feasibility measurement of a single-workgroup persistent world.
//! The production serial schedule supplies the reference; every mutable
//! simulation buffer with COPY_SRC is compared before timing is reported.

use std::error::Error;
use std::time::Instant;

use super::vision_validation::{make_kernel, read_buffer, upload_random_scene};
use super::*;

/// The target workload contains ten agents in one world.
const AGENTS: u32 = 10;
/// Matches the default scene's food count.
const FOOD_ITEMS: usize = 104;
/// Default visual field dimensions.
const VISION_WIDTH: u32 = 8;
const VISION_HEIGHT: u32 = 6;
/// All seven cooperative brain passes are required for valid simulation.
const FULL_BRAIN_PASS_LIMIT: u32 = 7;
/// Both push-constant words have the layout already used by the fused kernel.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Distinct cycle lengths exercise continuation across dispatch boundaries.
const PARITY_CHUNKS: [u32; 3] = [1, 7, 17];
/// More than the 128-slot episodic capacity warms the mature memory workload.
const WARMUP_CYCLES: u32 = 150;
/// Bounded dispatches keep a persistent kernel responsive on desktop GPUs.
const MAX_CYCLES_PER_DISPATCH: u32 = 25;
/// A hundred cycles amortize recording overhead without a lengthy probe.
const TIMED_CYCLES: u32 = 100;
/// Odd repeat count gives one median while rotating execution order.
const TIMING_ROUNDS: usize = 3;
/// Complete mutable storage includes depth, consumed flags, claims and grids.
const MUTABLE_BUFFERS: usize = 13;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct BufferSnapshot {
    name: &'static str,
    bytes: Vec<u8>,
}

fn state_buffers(kernel: &GpuKernel) -> [(&'static str, &wgpu::Buffer); MUTABLE_BUFFERS] {
    [
        ("physics", &kernel.agent_phys_buffer),
        ("decisions", &kernel.decision_buffer),
        ("food_state", &kernel.food_state_buffer),
        ("food_flags_and_claims", &kernel.food_flags_buffer),
        ("food_grid", &kernel.food_grid_buffer),
        ("agent_grid", &kernel.agent_grid_buffer),
        ("collision_scratch", &kernel.collision_scratch_buffer),
        ("sensory_including_depth", &kernel.sensory_buffer),
        ("brain", &kernel.brain_state_buffer),
        ("brain_scratch", &kernel.brain_scratch_buffer),
        ("patterns", &kernel.pattern_buffer),
        ("trail", &kernel.trail_ring_buffer),
        ("sensory_next", &kernel._sensory_next_buffer),
    ]
}

fn snapshot(kernel: &GpuKernel) -> TestResult<Vec<BufferSnapshot>> {
    state_buffers(kernel)
        .into_iter()
        .map(|(name, buffer)| {
            Ok(BufferSnapshot {
                name,
                bytes: read_buffer(kernel, buffer, buffer.size())?,
            })
        })
        .collect()
}

fn restore(kernel: &mut GpuKernel, snapshot: &[BufferSnapshot]) {
    assert_eq!(snapshot.len(), MUTABLE_BUFFERS);
    for ((name, buffer), saved) in state_buffers(kernel).into_iter().zip(snapshot) {
        assert_eq!(name, saved.name);
        assert_eq!(buffer.size(), u64::try_from(saved.bytes.len()).unwrap());
        kernel.queue.write_buffer(buffer, 0, &saved.bytes);
    }
    kernel.active_config_index = 0;
    // Flush reset transfers before a timed dispatch begins.
    kernel.queue.submit(std::iter::empty());
    kernel.poll_wait();
}

fn assert_snapshot_equal(expected: &[BufferSnapshot], actual: &[BufferSnapshot], label: &str) {
    assert_eq!(expected.len(), actual.len());
    for (expected, actual) in expected.iter().zip(actual) {
        assert_eq!(expected.name, actual.name);
        assert_eq!(expected.bytes.len(), actual.bytes.len());
        if let Some((offset, (expected_byte, actual_byte))) = expected
            .bytes
            .iter()
            .zip(&actual.bytes)
            .enumerate()
            .find(|(_, (expected_byte, actual_byte))| expected_byte != actual_byte)
        {
            panic!(
                "{label}: {} differs at byte {offset} (word {}): serial={expected_byte:#04x}, persistent={actual_byte:#04x}",
                expected.name,
                offset / std::mem::size_of::<u32>(),
            );
        }
    }
}

// The persistent module mutates grids, so its vision readers must use the
// atomic form of the same storage values. Only indexing expressions change;
// the serial ray and senses arithmetic remains the production source.
fn atomic_vision_source() -> String {
    let mut source = include_str!("../shaders/kernel/phase_vision.wgsl").to_owned();
    for buffer in ["food_flags", "food_grid", "agent_grid"] {
        let needle = format!("{buffer}[");
        let mut rewritten = String::with_capacity(source.len());
        let mut cursor = 0;
        while let Some(relative) = source[cursor..].find(&needle) {
            let start = cursor + relative;
            let opening = start + buffer.len();
            let mut nesting = 0_u32;
            let mut closing = None;
            for (index, byte) in source[opening..].bytes().enumerate() {
                match byte {
                    b'[' => nesting += 1,
                    b']' => {
                        nesting -= 1;
                        if nesting == 0 {
                            closing = Some(opening + index + 1);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let end = closing.expect("vision storage reads must have balanced brackets");
            rewritten.push_str(&source[cursor..start]);
            rewritten.push_str("atomicLoad(&");
            rewritten.push_str(&source[start..end]);
            rewritten.push(')');
            cursor = end;
        }
        rewritten.push_str(&source[cursor..]);
        source = rewritten;
    }
    source
}

fn make_persistent_pipeline(kernel: &GpuKernel) -> wgpu::ComputePipeline {
    let vision = atomic_vision_source();
    let source = apply_subgroup_markers(
        &[
            include_str!("../shaders/kernel/common.wgsl"),
            include_str!("../shaders/kernel/brain_passes.wgsl"),
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            include_str!("../shaders/kernel/phase_food_claim.wgsl"),
            include_str!("../shaders/kernel/kernel_tick.wgsl"),
            include_str!("../shaders/kernel/phase_clear.wgsl"),
            include_str!("../shaders/kernel/phase_food_grid.wgsl"),
            include_str!("../shaders/kernel/phase_food_respawn.wgsl"),
            include_str!("../shaders/kernel/phase_agent_grid.wgsl"),
            include_str!("../shaders/kernel/phase_grid_order.wgsl"),
            include_str!("../shaders/kernel/phase_collision.wgsl"),
            include_str!("../shaders/kernel/phase_trail_sample.wgsl"),
            vision.as_str(),
            include_str!("../shaders/kernel/persistent_tick.wgsl"),
        ]
        .join("\n"),
        kernel.has_subgroup,
    );
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("persistent_world_probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
    let pipeline_layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("persistent_world_probe_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    let mut constants = vision_override_constants(&kernel.layout);
    let parallel_requested = [
        "XAGENT_VISION_PARALLEL_STEPS",
        "XAGENT_VISION_OBJECT_QUERIES",
        "XAGENT_VISION_PARALLEL_SCENT",
    ]
    .iter()
    .any(|name| std::env::var(name).as_deref() == Ok("1"));
    let masks =
        parallel_requested && std::env::var("XAGENT_VISION_AGENT_MASKS").as_deref() == Ok("1");
    constants.insert("VISION_AGENT_MASKS".into(), f64::from(u32::from(masks)));
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("persistent_world_probe"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("persistent_tick"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        })
}

fn prepare_kernel(scene: u64) -> GpuKernel {
    let mut kernel = make_kernel(VISION_WIDTH, VISION_HEIGHT, AGENTS, FOOD_ITEMS);
    assert_eq!(kernel.vision_stride, 1);
    kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
    kernel.set_brain_beside_vision(false);
    kernel.probe.skip_global = false;
    kernel.probe.skip_vision = false;
    kernel.probe.kernel_pass_limit = FULL_BRAIN_PASS_LIMIT;
    upload_random_scene(&kernel, scene, true, false);
    // Close initial positions exercise collisions; low integrity exercises a
    // death reset without relying on the random terrain to kill an agent.
    for agent in 0..AGENTS {
        const SEPARATION: f32 = 1.5;
        kernel.write_agent_physics_fields(
            agent,
            &[(P_POS_X, agent as f32 * SEPARATION), (P_POS_Z, 0.0)],
        );
    }
    const LOW_INTEGRITY: f32 = 0.0001;
    kernel.write_agent_physics_fields(0, &[(P_INTEGRITY, LOW_INTEGRITY), (P_ENERGY, 0.0)]);
    kernel.poll_wait();
    kernel
}

fn dispatch_persistent(
    kernel: &mut GpuKernel,
    pipeline: &wgpu::ComputePipeline,
    start_tick: u32,
    ticks: u32,
) {
    assert_eq!(ticks % kernel.brain_tick_stride, 0);
    kernel.upload_world_config(u64::from(start_tick), ticks);
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &kernel.bind_groups[kernel.active_config_index], &[]);
        pass.set_push_constants(
            0,
            bytemuck::cast_slice(&[start_tick, FULL_BRAIN_PASS_LIMIT]),
        );
        pass.dispatch_workgroups(1, 1, 1);
    }
    kernel.queue.submit([encoder.finish()]);
    kernel.active_config_index = 1 - kernel.active_config_index;
}

fn advance(
    kernel: &mut GpuKernel,
    persistent: Option<&wgpu::ComputePipeline>,
    start_tick: u32,
    cycles: u32,
) {
    let mut completed = 0;
    while completed < cycles {
        let chunk = (cycles - completed).min(MAX_CYCLES_PER_DISPATCH);
        let tick = start_tick + completed * kernel.brain_tick_stride;
        let ticks = chunk * kernel.brain_tick_stride;
        if let Some(pipeline) = persistent {
            dispatch_persistent(kernel, pipeline, tick, ticks);
        } else {
            kernel.dispatch_ticks(u64::from(tick), ticks);
        }
        completed += chunk;
    }
    kernel.poll_wait();
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn persistent_world_matches_complete_serial_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel(0);
    let pipeline = make_persistent_pipeline(&kernel);
    let initial = snapshot(&kernel)?;
    let mut expected = Vec::with_capacity(PARITY_CHUNKS.len());
    let mut tick = 0;
    for cycles in PARITY_CHUNKS {
        advance(&mut kernel, None, tick, cycles);
        expected.push(snapshot(&kernel)?);
        tick += cycles * kernel.brain_tick_stride;
    }
    restore(&mut kernel, &initial);
    tick = 0;
    for (cycles, expected) in PARITY_CHUNKS.into_iter().zip(&expected) {
        advance(&mut kernel, Some(&pipeline), tick, cycles);
        tick += cycles * kernel.brain_tick_stride;
        assert_snapshot_equal(expected, &snapshot(&kernel)?, &format!("tick={tick}"));
    }
    println!("PERSISTENT_PARITY agents={AGENTS} ticks={tick} buffers={MUTABLE_BUFFERS} exact=true");
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_persistent_world_against_serial() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel(0);
    let pipeline = make_persistent_pipeline(&kernel);
    advance(&mut kernel, None, 0, WARMUP_CYCLES);
    let warm = snapshot(&kernel)?;
    let start_tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    let mut timings = [Vec::new(), Vec::new()];
    for round in 0..TIMING_ROUNDS {
        let mut snapshots = [None, None];
        for offset in 0..timings.len() {
            let arm = (round + offset) % timings.len();
            restore(&mut kernel, &warm);
            let start = Instant::now();
            advance(
                &mut kernel,
                (arm == 1).then_some(&pipeline),
                start_tick,
                TIMED_CYCLES,
            );
            timings[arm].push(start.elapsed().as_secs_f64());
            snapshots[arm] = Some(snapshot(&kernel)?);
        }
        assert_snapshot_equal(
            snapshots[0].as_ref().unwrap(),
            snapshots[1].as_ref().unwrap(),
            &format!("timing round={round}"),
        );
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let serial = timings[0][TIMING_ROUNDS / 2];
    let persistent = timings[1][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    println!(
        "PERSISTENT_WORLD agents={AGENTS} ticks={ticks} rounds={TIMING_ROUNDS} serial_secs={serial:.9} persistent_secs={persistent:.9} serial_tps={:.3} persistent_tps={:.3} speedup={:.3} exact_buffers={MUTABLE_BUFFERS}",
        f64::from(ticks) / serial, f64::from(ticks) / persistent, serial / persistent,
    );
    Ok(())
}
