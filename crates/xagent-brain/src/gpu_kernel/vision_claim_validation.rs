//! Hardware-only overlap of vision with the following cycle's food claim.
//! The global world's workgroup snapshots every physics word after collisions
//! into private binding 17. Vision reads that snapshot while the original next
//! claim mutates live physics and atomic claim slots. Claim does not modify
//! consumed-food flags, grids, sensory state, or the brain genes vision reads.
//!
//! Each submission finishes vision without starting another claim: C,M,G,
//! (V+C),M,G,...,V. All thirteen public state buffers remain comparable at API
//! boundaries, including the untouched public sensory_next. Snapshot and
//! adapted-feature storage are private; no normalization of reference state is
//! performed. Both arms use production global credit and the same explicitly
//! composed cooperative-whitening, prefetch-eight, sixteen-lane brain.

use std::{collections::HashMap, error::Error, fmt::Write, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::predictor_fusion::fuse_inline_predictor;
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// Default and odd ray counts exercise complete and partial eight-ray groups.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// The shared fixture uses the default small population.
const AGENTS: u32 = 10;
/// Current measured production brain arithmetic is explicit, independent of env.
const PREFETCH_FACTOR: u32 = 8;
const PREDICTOR_LANES: u32 = 16;
const COMPLETE_BRAIN: u32 = 7;
/// Both entries share the existing two-word push-constant range.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// One original claim workgroup and each object-ray workgroup have 256 lanes.
const RAYS_PER_GROUP: u32 = BRAIN_WORKGROUP_THREADS / PARALLEL_VISION_LANES;
/// Boundary checkpoints exercise singleton submissions, refresh, and death.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Populate memory before timing evolving complete cycles.
const WARMUP_CYCLES: u32 = 256;
const TIMED_CYCLES: u32 = 100;
const TIMING_ROUNDS: usize = 5;
const ARM_COUNT: usize = 2;
const MUTABLE_BUFFERS: usize = 13;
/// The fixture leaves one dead slot without requesting a respawn.
const INACTIVE_AGENT: u32 = 1;

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

fn function(source: &str, name: &str) -> String {
    let needle = format!("fn {name}(");
    assert_eq!(source.matches(&needle).count(), 1);
    let start = source.find(&needle).unwrap();
    let end = start + source[start..].find("\n}").unwrap() + "\n}".len();
    source[start..end].to_owned()
}

fn declaration(source: &str, name: &str) -> String {
    let matches: Vec<_> = source
        .lines()
        .filter(|line| line.starts_with(name))
        .collect();
    assert_eq!(matches.len(), 1, "unique declaration: {name}");
    matches[0].to_owned()
}

fn optimized_brain() -> String {
    let passes = fuse_inline_predictor(&compose_brain_passes(true));
    let passes = dense_prefetch::prefetch_passes(&passes, PREFETCH_FACTOR);
    predictor_width::wider_predictor(&passes, PREDICTOR_LANES)
}

fn claim_source() -> String {
    let original = include_str!("../shaders/kernel/kernel_tick.wgsl");
    let passes = include_str!("../shaders/kernel/brain_passes.wgsl");
    let inner = include_str!("../shaders/kernel/brain_inner.wgsl");
    let mut fragments = vec![inner[..inner.find("fn brain_tick_inner(").unwrap()].to_owned()];
    for name in [
        "s_similarities",
        "shared_sort_indices",
        "s_argmin_val",
        "s_argmin_idx",
    ] {
        fragments.push(declaration(passes, &format!("var<workgroup> {name}:")));
    }
    fragments.push(declaration(original, "var<workgroup> s_food_dist_sq:"));
    fragments.push(declaration(original, "const NO_FOOD:"));
    for name in ["food_precedes", "agent_physics", "agent_food_detect"] {
        fragments.push(function(original, name));
    }
    fragments.push(function(
        include_str!("../shaders/kernel/phase_food_claim.wgsl"),
        "claim_food",
    ));
    let entry = function(original, "kernel_claim_tick");
    let body = &entry[entry.find('{').unwrap() + 1..entry.len() - 1];
    let body = replace_once(
        body,
        "    let agent_id = wgid.x;\n    let tid = lid.x;\n",
        "",
    );
    fragments.push(format!(
        "fn vision_claim_next(agent_id: u32, tid: u32) {{{body}\n}}"
    ));
    let source = fragments.join("\n");
    assert!(!source.contains("brain_tick_inner("));
    assert!(!source.contains("sensory_buffer["));
    source
}

fn snapshot_vision_source() -> String {
    let phase = with_prepared_scent(include_str!("../shaders/kernel/phase_vision.wgsl"));
    let mut vision = [
        phase.as_str(),
        include_str!("../shaders/kernel/phase_vision_parallel.wgsl"),
        include_str!("../shaders/kernel/phase_vision_object_queries.wgsl"),
        include_str!("../shaders/kernel/phase_vision_scent_parallel.wgsl"),
    ]
    .join("\n")
    .replace("physics_state[", "vision_claim_physics[");
    assert!(!vision.contains("physics_state["));
    // Only consumed-flag reads occur in the vision fragments. Claim writes
    // atomic words in the separate claim half of the same backing buffer.
    for index in ["fidx", "f", "food_id", "food"] {
        vision = vision.replace(
            &format!("food_flags[{index}]"),
            &format!("atomicLoad(&food_flags[{index}])"),
        );
    }
    assert!(!vision
        .replace("atomicLoad(&food_flags[", "consumed_flag[")
        .contains("food_flags["));
    let entry = function(
        include_str!("../shaders/kernel/vision_tick.wgsl"),
        "vision_tick",
    );
    let body = &entry[entry.find('{').unwrap() + 1..entry.len() - 1];
    let body = body
        .replace("wgid.x", "vision_group")
        .replace("lid.x", "tid");
    writeln!(
        vision,
        "\nfn vision_claim_render(vision_group: u32, tid: u32) {{{body}\n}}"
    )
    .unwrap();
    vision
}

fn combined_source() -> String {
    let common = with_plain_grid_bindings(include_str!("../shaders/kernel/common.wgsl"));
    let (atomic, plain) = GRID_BINDING_DECLARATIONS[0];
    let common =
        replace_once(&common, plain, atomic).replace("sensory_next", "vision_claim_physics");
    [
        common,
        claim_source(),
        snapshot_vision_source(),
        include_str!("vision_claim_tick.wgsl").to_owned(),
    ]
    .join("\n")
}

fn snapshot_global_source() -> String {
    let common = include_str!("../shaders/kernel/common.wgsl")
        .replace("sensory_next", "vision_claim_physics");
    let global = replace_once(
        include_str!("../shaders/kernel/global_tick.wgsl"),
        "@compute @workgroup_size(256)\nfn global_tick(@builtin(local_invocation_id) lid: vec3u) {\n    let tid = lid.x;",
        "fn global_world_inner(tid: u32) {",
    );
    let end = global.rfind('}').unwrap();
    let snapshot = "\n    // The final collision barrier publishes every live and dead physics slot.\n    for (var word = tid; word < agent_count * PHYS_STRIDE; word += ENCODER_CREDIT_THREADS) {\n        vision_claim_physics[word] = physics_state[word];\n    }\n";
    let global = format!("{}{snapshot}}}{}", &global[..end], &global[end + 1..]);
    [
        common.as_str(),
        include_str!("../shaders/kernel/phase_clear.wgsl"),
        include_str!("../shaders/kernel/phase_food_grid.wgsl"),
        include_str!("../shaders/kernel/phase_food_respawn.wgsl"),
        include_str!("../shaders/kernel/phase_agent_grid.wgsl"),
        include_str!("../shaders/kernel/phase_grid_order.wgsl"),
        include_str!("../shaders/kernel/phase_collision.wgsl"),
        include_str!("../shaders/kernel/phase_trail_sample.wgsl"),
        global.as_str(),
        include_str!("../shaders/kernel/phase_encoder_credit.wgsl"),
        include_str!("../shaders/kernel/global_credit_tick.wgsl"),
    ]
    .join("\n")
}

fn pipeline(
    kernel: &GpuKernel,
    source: String,
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

struct Overlapped {
    global: wgpu::ComputePipeline,
    combined: wgpu::ComputePipeline,
    groups: [wgpu::BindGroup; 2],
    _features: wgpu::Buffer,
    _physics: wgpu::Buffer,
}

fn private_group(
    kernel: &GpuKernel,
    features: &wgpu::Buffer,
    physics: &wgpu::Buffer,
    config_index: usize,
) -> wgpu::BindGroup {
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
        features,
        &kernel.brain_config_buffer,
        &kernel.dispatch_args_buffer,
        &kernel.trail_ring_buffer,
        physics,
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
        label: Some("vision_claim_private_storage"),
        layout: &kernel.kernel_pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

fn prepare(width: u32, height: u32, boundary: bool) -> (GpuKernel, Overlapped) {
    let mut kernel = super::cached_combined_validation::prepare(width, height, boundary);
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    kernel.global_credit =
        Some(global_credit::Pipelines::new(&kernel, &optimized_brain(), &constants).unwrap());
    assert!(kernel.global_credit_active());
    assert_eq!(kernel.vision_stride, 1);
    assert_eq!(kernel.agent_count, AGENTS);
    let global = pipeline(
        &kernel,
        snapshot_global_source(),
        "global_credit_tick",
        &constants,
    );
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
    let combined = pipeline(&kernel, combined_source(), "vision_claim_tick", &constants);
    let buffer = |size, label| {
        kernel.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let features = buffer(kernel.brain_scratch_buffer.size(), "vision_claim_features");
    // Binding 17's runtime array has no host-side sensory-layout size minimum.
    let physics = buffer(kernel.agent_phys_buffer.size(), "vision_claim_physics");
    let groups = std::array::from_fn(|index| private_group(&kernel, &features, &physics, index));
    assert!(kernel.agent_count + kernel.vision_workgroups <= MAX_DISPATCH_WORKGROUPS);
    (
        kernel,
        Overlapped {
            global,
            combined,
            groups,
            _features: features,
            _physics: physics,
        },
    )
}

fn advance(
    kernel: &mut GpuKernel,
    candidate: &Overlapped,
    start_cycle: u32,
    cycles: u32,
    overlap: bool,
) {
    assert!(cycles > 0);
    let start_tick = start_cycle.checked_mul(kernel.brain_tick_stride).unwrap();
    if !overlap {
        kernel.dispatch_ticks(
            u64::from(start_tick),
            cycles.checked_mul(kernel.brain_tick_stride).unwrap(),
        );
        kernel.poll_wait();
        return;
    }
    assert!(kernel.global_credit_active() && !kernel.probe.skip_vision);
    assert_eq!(kernel.vision_stride, 1);
    kernel.upload_world_config_with_cycles(
        u64::from(start_tick),
        kernel.brain_tick_stride,
        COMPLETE_BRAIN,
        1,
    );
    let credit = kernel.global_credit.as_ref().unwrap();
    let mut completed = 0;
    while completed < cycles {
        let end = (completed + MAX_FUSED_BATCHES).min(cycles);
        let mut encoder = kernel.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &candidate.groups[kernel.active_config_index], &[]);
            let tick = start_tick + completed * kernel.brain_tick_stride;
            pass.set_pipeline(&kernel.kernel_claim_pipeline);
            pass.set_push_constants(0, bytemuck::cast_slice(&[tick, COMPLETE_BRAIN]));
            pass.dispatch_workgroups(kernel.agent_count, 1, 1);
            for cycle in completed..end {
                let tick = start_tick + cycle * kernel.brain_tick_stride;
                pass.set_pipeline(&credit.main);
                pass.set_push_constants(0, bytemuck::cast_slice(&[tick, COMPLETE_BRAIN]));
                pass.dispatch_workgroups(kernel.agent_count, 1, 1);
                let next_tick = tick + kernel.brain_tick_stride;
                pass.set_pipeline(&candidate.global);
                pass.set_push_constants(
                    0,
                    bytemuck::cast_slice(&[next_tick, kernel.kernel_batch_size()]),
                );
                pass.dispatch_workgroups(credit.global_workgroups, 1, 1);
                if cycle + 1 == end {
                    pass.set_pipeline(&kernel.vision_pipeline);
                    pass.dispatch_workgroups(kernel.vision_workgroups, 1, 1);
                } else {
                    pass.set_pipeline(&candidate.combined);
                    pass.set_push_constants(0, bytemuck::cast_slice(&[next_tick, COMPLETE_BRAIN]));
                    pass.dispatch_workgroups(kernel.agent_count + kernel.vision_workgroups, 1, 1);
                }
            }
        }
        kernel.queue.submit([encoder.finish()]);
        completed = end;
    }
    kernel.active_config_index = 1 - kernel.active_config_index;
    kernel.poll_wait();
}

fn trajectory(
    kernel: &mut GpuKernel,
    pipelines: &Overlapped,
    overlap: bool,
) -> TestResult<Vec<State>> {
    let mut states = Vec::new();
    let mut cycle = 0;
    for cycles in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, pipelines, cycle, cycles, overlap);
        states.push(capture_state(kernel)?);
        cycle += cycles;
    }
    Ok(states)
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn vision_beside_next_claim_preserves_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let (mut kernel, pipelines) = prepare(width, height, true);
        let initial_state = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let reference = trajectory(&mut kernel, &pipelines, false)?;
        restore(&mut kernel, &saved);
        let candidate = trajectory(&mut kernel, &pipelines, true)?;
        for (reference, candidate) in reference.iter().zip(&candidate) {
            assert_eq!(candidate.len(), MUTABLE_BUFFERS);
            assert_state_equal(&kernel, reference, candidate);
            assert_inactive_agent_unchanged(
                &kernel,
                &initial_state,
                candidate,
                INACTIVE_AGENT,
                "vision beside next claim",
            );
        }
        restore(&mut kernel, &saved);
        let repeated = trajectory(&mut kernel, &pipelines, true)?;
        for (candidate, repeated) in candidate.iter().zip(&repeated) {
            assert_state_equal(&kernel, candidate, repeated);
        }
        println!("VISION_CLAIM_PARITY width={width} height={height} cycles=100 exact_buffers={MUTABLE_BUFFERS} repeat_buffers={MUTABLE_BUFFERS} death_refresh=true private_physics_snapshot=true no_pending_claim_at_boundary=true");
    }
    Ok(())
}

#[test]
#[ignore = "GPU full-cycle benchmark; run explicitly in release mode"]
fn benchmark_vision_beside_next_claim() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let (mut kernel, pipelines) = prepare(width, height, false);
    advance(&mut kernel, &pipelines, 0, WARMUP_CYCLES, false);
    let warm = checkpoint(&kernel);
    advance(&mut kernel, &pipelines, WARMUP_CYCLES, TIMED_CYCLES, false);
    let expected = capture_state(&kernel)?;
    restore(&mut kernel, &warm);
    advance(&mut kernel, &pipelines, WARMUP_CYCLES, TIMED_CYCLES, true);
    assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
    let mut timings: [Vec<f64>; ARM_COUNT] = std::array::from_fn(|_| Vec::new());
    for round in 0..TIMING_ROUNDS {
        for position in 0..ARM_COUNT {
            let arm = (round + position) % ARM_COUNT;
            restore(&mut kernel, &warm);
            let start = Instant::now();
            advance(
                &mut kernel,
                &pipelines,
                WARMUP_CYCLES,
                TIMED_CYCLES,
                arm != 0,
            );
            timings[arm].push(start.elapsed().as_secs_f64());
            assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
        }
    }
    for times in &mut timings {
        times.sort_by(f64::total_cmp);
    }
    let reference = timings[0][TIMING_ROUNDS / 2];
    let candidate = timings[1][TIMING_ROUNDS / 2];
    let submissions = TIMED_CYCLES.div_ceil(MAX_FUSED_BATCHES);
    println!("VISION_CLAIM_TIMING cycles={TIMED_CYCLES} pairs={TIMING_ROUNDS} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} reference_dispatches={} candidate_dispatches={} exact_buffers={MUTABLE_BUFFERS} physics_snapshot_bytes={} brain=coop_prefetch8_lanes16_global_credit full_simulation=true", reference/candidate, TIMED_CYCLES*4, TIMED_CYCLES*3+submissions, kernel.agent_phys_buffer.size());
    Ok(())
}
