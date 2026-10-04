//! Hardware-only cached object vision and ordered scent beside the brain.
//! Both schedules compile the same cooperative whitening, prefetch-eight and
//! sixteen-lane predictor. The standalone reference keeps its production
//! vision source unchanged. Its unused sensory_next allocation stays at the
//! checkpoint; the candidate must publish an exact copy of its next senses.

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::predictor_fusion::fuse_inline_predictor;
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::vision_validation::{make_kernel, upload_random_scene};
use super::whitening_validation::{force_death, prepare_boundary_scene, REFRESH_CYCLES};
use super::*;

/// The bounded object cache supports the default ten-agent population.
const AGENTS: u32 = 10;
/// The default food population fits the existing 256-item object cache.
const FOOD_ITEMS: usize = 104;
/// An odd field additionally exercises padding in the last eight-ray group.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// A complete ray workgroup contains eight groups of thirty-two invocations.
const RAYS_PER_GROUP: u32 = BRAIN_WORKGROUP_THREADS / PARALLEL_VISION_LANES;
/// Match the measured production FP32 brain configuration explicitly.
const PREFETCH_FACTOR: u32 = 8;
const PREDICTOR_LANES: u32 = 16;
/// The seven cooperative brain phases all execute in each full cycle.
const COMPLETE_BRAIN: u32 = 7;
/// Tick and pass limit occupy two four-byte push-constant words.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Scent aliases one scalar per item in the existing feature array.
const SCENT_CHUNK_SIZE: usize = 256;
/// Endpoints surround death and whitening refresh boundaries through cycle 100.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Populate episodic memory before timing either schedule.
const WARMUP_CYCLES: u32 = 256;
/// One hundred evolving cycles amortize host submission and waiting.
const TIMED_CYCLES: u32 = 100;
/// Alternate the first schedule across five paired timing trials.
const TIMING_ROUNDS: usize = 5;
/// State capture has thirteen mutable buffers; sensory_next is the last.
const MUTABLE_BUFFERS: usize = 13;
const SENSORY_BUFFER_INDEX: usize = 7;
const NEXT_SENSORY_BUFFER_INDEX: usize = MUTABLE_BUFFERS - 1;
/// This agent remains inactive without a pending respawn throughout parity.
const INACTIVE_AGENT: u32 = 1;
/// Nearby live agents exercise post-registration collision displacement.
const COLLISION_SEPARATION: f32 = 1.5;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn replace_once(source: &str, old: &str, new: &str) -> String {
    assert_eq!(
        source.matches(old).count(),
        1,
        "unique source target: {old}"
    );
    source.replacen(old, new, 1)
}

fn object_cache_in_brain_scratch() -> String {
    let mut source = replace_once(
        include_str!("../shaders/kernel/phase_vision_object_queries.wgsl"),
        "var<workgroup> vision_object_cache: array<vec4<f32>, VISION_OBJECT_CACHE_SIZE>;",
        "",
    );
    for (old, new) in [
        (
            "vision_object_cache[VISION_OBJECT_ORIGIN_BASE] = vec4<f32>(\n            physics_state[base + P_POS_X],\n            physics_state[base + P_POS_Y],\n            physics_state[base + P_POS_Z],\n            0.0,\n        );",
            "cached_object_store(VISION_OBJECT_ORIGIN_BASE, vec4<f32>(\n            physics_state[base + P_POS_X],\n            physics_state[base + P_POS_Y],\n            physics_state[base + P_POS_Z],\n            0.0,\n        ));",
        ),
        (
            "vision_object_cache[tid] = vision_object_load_food(\n            tid, wc_u32(WC_GRID_WIDTH), i32(wc_u32(WC_GRID_OFFSET)));",
            "cached_object_store(tid, vision_object_load_food(\n            tid, wc_u32(WC_GRID_WIDTH), i32(wc_u32(WC_GRID_OFFSET))));",
        ),
        (
            "vision_object_cache[VISION_OBJECT_AGENT_BASE + tid] = vec4<f32>(\n            physics_state[other_base + P_POS_X],\n            physics_state[other_base + P_POS_Y],\n            physics_state[other_base + P_POS_Z],\n            physics_state[other_base + P_ALIVE],\n        );",
            "cached_object_store(VISION_OBJECT_AGENT_BASE + tid, vec4<f32>(\n            physics_state[other_base + P_POS_X],\n            physics_state[other_base + P_POS_Y],\n            physics_state[other_base + P_POS_Z],\n            physics_state[other_base + P_ALIVE],\n        ));",
        ),
        (
            "vision_object_cache[VISION_OBJECT_DIRECTION_BASE + local_ray] =\n                vec4<f32>(vision_object_direction(agent_id, ray_idx), 0.0);",
            "cached_object_store(VISION_OBJECT_DIRECTION_BASE + local_ray,\n                vec4<f32>(vision_object_direction(agent_id, ray_idx), 0.0));",
        ),
    ] {
        source = replace_once(&source, old, new);
    }
    for index in [
        "VISION_OBJECT_ORIGIN_BASE",
        "VISION_OBJECT_DIRECTION_BASE + local_ray",
        "food_id",
        "VISION_OBJECT_AGENT_BASE + lane",
    ] {
        source = replace_once(
            &source,
            &format!("vision_object_cache[{index}]"),
            &format!("cached_object_load({index})"),
        );
    }
    assert!(!source.contains("vision_object_cache"));
    source
}

fn scent_in_brain_scratch() -> String {
    let mut source = include_str!("../shaders/kernel/phase_vision_scent_parallel.wgsl").to_owned();
    for declaration in [
        "var<workgroup> vision_scent_contributions: array<vec2<f32>, VISION_SCENT_CHUNK_SIZE>;",
        "var<workgroup> vision_scent_valid: array<u32, VISION_SCENT_CHUNK_SIZE>;",
        "var<workgroup> vision_prepared_scent: vec2<f32>;",
    ] {
        source = replace_once(&source, declaration, "");
    }
    for (old, new) in [
        ("vision_scent_contributions[slot] = contribution;", "s_dense_partials[slot] = contribution.x;\n            s_reinf_dot[slot] = contribution.y;"),
        ("vision_scent_valid[slot] = valid;", "s_features[slot] = f32(valid);"),
        ("let valid = vision_scent_valid[slot];", "let valid = u32(s_features[slot]);"),
        ("vision_scent_contributions[slot].x", "s_dense_partials[slot]"),
        ("vision_scent_contributions[slot].y", "s_reinf_dot[slot]"),
        ("vision_prepared_scent = vec2<f32>(1.0, 1.0) - exp(-strength * concentration);", "let prepared = vec2<f32>(1.0, 1.0) - exp(-strength * concentration);\n        s_explore[0u] = prepared.x;\n        s_explore[1u] = prepared.y;"),
    ] {
        source = replace_once(&source, old, new);
    }
    assert!(!source.contains("var<workgroup>"));
    // Masks are exactly 0..3, so their temporary FP32 representation is exact.
    source
}

fn optimized_brain() -> String {
    let fused = fuse_inline_predictor(&compose_brain_passes(true));
    let prefetched = dense_prefetch::prefetch_passes(&fused, PREFETCH_FACTOR);
    predictor_width::wider_predictor(&prefetched, PREDICTOR_LANES)
}

fn combined_source(kernel: &GpuKernel, brain: &str) -> String {
    let brain = replace_once(
        brain,
        "var<workgroup> s_visual: array<f32, VC_SCRATCH_LEN>;",
        "var<workgroup> s_visual: array<f32, CACHED_OBJECT_SCALAR_CAPACITY>;",
    );
    let phase = replace_once(
        &with_prepared_scent(include_str!("../shaders/kernel/phase_vision.wgsl")),
        "let scent = vision_prepared_scent;",
        "let scent = vec2<f32>(s_explore[0u], s_explore[1u]);",
    );
    let vision = [
        phase,
        include_str!("../shaders/kernel/phase_vision_parallel.wgsl").to_owned(),
        object_cache_in_brain_scratch(),
        scent_in_brain_scratch(),
    ]
    .join("\n")
    .replace("sensory_buffer[", "sensory_next[");
    assert!(!vision.contains("sensory_buffer"));
    let common = with_plain_grid_bindings(include_str!("../shaders/kernel/common.wgsl"));
    apply_subgroup_markers(
        &[
            common.as_str(),
            brain.as_str(),
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            vision.as_str(),
            include_str!("../shaders/kernel/cached_brain_vision_tick.wgsl"),
        ]
        .join("\n"),
        kernel.has_subgroup,
    )
}

fn pipeline(
    kernel: &GpuKernel,
    source: &str,
    entry: &str,
    constants: &HashMap<String, f64>,
) -> wgpu::ComputePipeline {
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            cache: None,
        })
}

fn install_pipelines(kernel: &mut GpuKernel) {
    assert!(!kernel.layout.visual_cortex_enabled);
    assert!(kernel.layout.feature_count >= SCENT_CHUNK_SIZE);
    assert!(kernel.agent_count <= u32::BITS);
    assert!(kernel.food_count <= VISION_OBJECT_FOOD_CAPACITY);
    let brain = optimized_brain();
    let common = include_str!("../shaders/kernel/common.wgsl");
    let kernel_source = apply_subgroup_markers(
        &[
            common,
            &brain,
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            include_str!("../shaders/kernel/phase_food_claim.wgsl"),
            include_str!("../shaders/kernel/kernel_tick.wgsl"),
        ]
        .join("\n"),
        kernel.has_subgroup,
    );
    let mut constants = vision_override_constants(&kernel.layout);
    kernel.kernel_pipeline = pipeline(kernel, &kernel_source, "kernel_tick", &constants);
    kernel.kernel_claim_pipeline =
        pipeline(kernel, &kernel_source, "kernel_claim_tick", &constants);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    let global_source = [
        common,
        include_str!("../shaders/kernel/phase_clear.wgsl"),
        include_str!("../shaders/kernel/phase_food_grid.wgsl"),
        include_str!("../shaders/kernel/phase_food_respawn.wgsl"),
        include_str!("../shaders/kernel/phase_agent_grid.wgsl"),
        include_str!("../shaders/kernel/phase_grid_order.wgsl"),
        include_str!("../shaders/kernel/phase_collision.wgsl"),
        include_str!("../shaders/kernel/phase_trail_sample.wgsl"),
        include_str!("../shaders/kernel/global_tick.wgsl"),
    ]
    .join("\n");
    kernel.global_pipeline = pipeline(kernel, &global_source, "global_tick", &constants);
    for name in [
        "VISION_PARALLEL_STEPS",
        "VISION_OBJECT_QUERIES",
        "VISION_PARALLEL_SCENT",
    ] {
        constants.insert(name.into(), 1.0);
    }
    constants.insert(
        "VISION_RAYS_PER_WORKGROUP".into(),
        f64::from(RAYS_PER_GROUP),
    );
    let vision_common = with_plain_grid_bindings(common);
    let standalone_source = compose_vision_source(&vision_common, true, true);
    kernel.vision_pipeline = pipeline(kernel, &standalone_source, "vision_tick", &constants);
    let combined = combined_source(kernel, &brain);
    kernel.brain_vision_pipeline =
        pipeline(kernel, &combined, "cached_brain_vision_tick", &constants);
    kernel.sensory_publish_pipeline =
        pipeline(kernel, &combined, "cached_sensory_publish", &constants);
    let rays = kernel.layout.vision_width * kernel.layout.vision_height;
    kernel.vision_workgroups = kernel.agent_count * rays.div_ceil(RAYS_PER_GROUP);
    kernel.brain_vision_workgroups = kernel.agent_count + kernel.vision_workgroups;
    assert!(kernel.brain_vision_workgroups <= MAX_DISPATCH_WORKGROUPS);
    // Only this test module bypasses the production cached-vision restriction.
    kernel.standalone_vision_required = false;
}

fn seed_next_sensory(kernel: &GpuKernel) {
    // A schedule switch starts with both sensory buffers representing the same
    // completed cycle, including slots retained for inactive agents. This is
    // fixture initialization before the shared checkpoint, never normalization
    // of a candidate result or extra work inside a timed cycle.
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(
        &kernel.sensory_buffer,
        0,
        &kernel._sensory_next_buffer,
        0,
        kernel.sensory_buffer.size(),
    );
    kernel.queue.submit([encoder.finish()]);
    kernel.poll_wait();
}

pub(super) fn prepare(width: u32, height: u32, boundary: bool) -> GpuKernel {
    let mut kernel = make_kernel(width, height, AGENTS, FOOD_ITEMS);
    // Both schedules below explicitly compile their own brain main. Preserve
    // that reference when the process enables another production route.
    kernel.global_credit = None;
    kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
    kernel.set_brain_beside_vision(false);
    kernel.probe.skip_global = false;
    kernel.probe.skip_vision = false;
    kernel.probe.kernel_pass_limit = COMPLETE_BRAIN;
    assert_eq!(kernel.vision_stride, 1);
    install_pipelines(&mut kernel);
    upload_random_scene(&kernel, 0, boundary, !boundary);
    if boundary {
        for agent in 0..AGENTS {
            kernel.write_agent_physics_fields(
                agent,
                &[
                    (P_POS_X, agent as f32 * COLLISION_SEPARATION),
                    (P_POS_Z, 0.0),
                ],
            );
        }
        prepare_boundary_scene(&kernel);
    }
    seed_next_sensory(&kernel);
    kernel
}

fn advance(kernel: &mut GpuKernel, cycle: u32, cycles: u32, combined: bool) {
    kernel.set_brain_beside_vision(combined);
    kernel.dispatch_ticks(
        u64::from(cycle * kernel.brain_tick_stride),
        cycles * kernel.brain_tick_stride,
    );
    kernel.poll_wait();
}

fn assert_schedule_state(
    kernel: &GpuKernel,
    initial: &State,
    reference: &State,
    candidate: &State,
) {
    for state in [initial, reference, candidate] {
        assert_eq!(state.len(), MUTABLE_BUFFERS);
    }
    assert_state_equal(
        kernel,
        &reference[..NEXT_SENSORY_BUFFER_INDEX],
        &candidate[..NEXT_SENSORY_BUFFER_INDEX],
    );
    assert_eq!(
        reference[NEXT_SENSORY_BUFFER_INDEX], initial[NEXT_SENSORY_BUFFER_INDEX],
        "standalone must leave its unused next-sensory allocation untouched"
    );
    assert_eq!(
        candidate[NEXT_SENSORY_BUFFER_INDEX], candidate[SENSORY_BUFFER_INDEX],
        "combined sensory publication must copy every slot exactly"
    );
}

fn boundary_trajectory(kernel: &mut GpuKernel, combined: bool) -> TestResult<Vec<State>> {
    let mut cycle = 0;
    let mut states = Vec::with_capacity(PARITY_CHUNKS.len());
    for cycles in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, cycle, cycles, combined);
        states.push(capture_state(kernel)?);
        cycle += cycles;
    }
    Ok(states)
}

#[test]
#[ignore = "requires GPU; run explicitly with --ignored --nocapture"]
fn cached_vision_beside_brain_matches_standalone_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare(width, height, true);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let reference = boundary_trajectory(&mut kernel, false)?;
        restore(&mut kernel, &saved);
        let candidate = boundary_trajectory(&mut kernel, true)?;
        for (reference, candidate) in reference.iter().zip(&candidate) {
            assert_schedule_state(&kernel, &initial, reference, candidate);
            assert_inactive_agent_unchanged(
                &kernel,
                &initial,
                candidate,
                INACTIVE_AGENT,
                "cached vision beside brain",
            );
        }
        restore(&mut kernel, &saved);
        let repeated = boundary_trajectory(&mut kernel, true)?;
        for (candidate, repeated) in candidate.iter().zip(&repeated) {
            assert_state_equal(&kernel, candidate, repeated);
        }
        println!("CACHED_COMBINED_PARITY width={width} height={height} cycles=100 persistent_buffers=12 auxiliary_publication=exact candidate_repeat_buffers=13 death_refresh=true brain=coop_prefetch8_lanes16");
    }
    Ok(())
}

#[test]
#[ignore = "requires GPU; run in release mode with --ignored --nocapture"]
fn benchmark_cached_vision_beside_brain() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let mut kernel = prepare(width, height, false);
    advance(&mut kernel, 0, WARMUP_CYCLES, false);
    seed_next_sensory(&kernel);
    let initial = capture_state(&kernel)?;
    let warm = checkpoint(&kernel);
    // Exact schedule parity is mandatory before the first timing result.
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES, false);
    let reference = capture_state(&kernel)?;
    restore(&mut kernel, &warm);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES, true);
    let candidate = capture_state(&kernel)?;
    assert_schedule_state(&kernel, &initial, &reference, &candidate);
    let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..TIMING_ROUNDS {
        for offset in 0..timings.len() {
            let arm = (round + offset) % timings.len();
            restore(&mut kernel, &warm);
            let start = Instant::now();
            advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES, arm != 0);
            timings[arm].push(start.elapsed().as_secs_f64());
            let expected = if arm == 0 { &reference } else { &candidate };
            assert_state_equal(&kernel, expected, &capture_state(&kernel)?);
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let separate = timings[0][TIMING_ROUNDS / 2];
    let combined = timings[1][TIMING_ROUNDS / 2];
    println!("CACHED_COMBINED_TIMING cycles={TIMED_CYCLES} pairs={TIMING_ROUNDS} separate_seconds={separate:.9} combined_seconds={combined:.9} speedup={:.3} persistent_buffers=12 auxiliary_publication=exact repeat_buffers=13 brain=coop_prefetch8_lanes16 extra_object_cache_bytes=4748", separate / combined);
    Ok(())
}
