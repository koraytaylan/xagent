//! Test-only complete cycle with claim, prefix, brain beside world, then
//! vision beside packed encoder credit. The production 128-thread main is
//! the independent reference, including its canonical claim and prefix.
//! Public brain storage remains authoritative; both schedules share one
//! private packed cache and its unchanged import/invalidation lifecycle.

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::main_width::MAIN_THREADS;
use super::packed_store_validation::{
    advance, assert_mirror, cache, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// Raw fields exercise complete and partial feature/ray tiles.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// Capture both sides of refresh and a second forced death through cycle100.
const CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const REPLAYS: usize = 2;
const WARMUP_CYCLES: u32 = 1_000;
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;
const COMPLETE_BRAIN: u32 = 7;
const PHASE_MASK: u32 = 7;
const PUSH_CONSTANT_BYTES: u32 = 8;
const MUTABLE_BUFFERS: usize = 13;
const INACTIVE_AGENT: u32 = 1;
const FORCED_DEATHS: f32 = 2.0;
/// A ray group retains its original thirty-two sample invocations.
const RAYS_PER_GROUP: u32 = BRAIN_WORKGROUP_THREADS / PARALLEL_VISION_LANES;
/// The world branch handles one agent per lane without virtual agent phases.
const WORLD_THREADS: u32 = MAIN_THREADS;
/// Conservatively round every shared resource to Metal's sixteen-byte boundary.
const SHARED_ALIGNMENT: usize = 16;
const WORD_BYTES: usize = size_of::<f32>();
const SCENT_ITEMS: usize = 256;
const OBJECT_AGENTS: usize = 32;
const ENCODER_PARTIAL_LANES: usize = 4;
/// Six independent loops cover clear, food construction/respawn and grid sort.
const WORLD_STRIDED_LOOPS: usize = 6;
const RAW_MAIN_SHARED_BOUND: u64 = 9_200;
const VISION_SHARED_BOUND: u64 = 7_872;
const FOOD_GRID_STATE: usize = 4;
const AGENT_GRID_STATE: usize = 5;

const VISION_ENTRY: &str = "@compute @workgroup_size(VISION_WORKGROUP_SIZE)\nfn vision_tick(\n    @builtin(local_invocation_id) lid: vec3u,\n    @builtin(workgroup_id) wgid: vec3u,\n)";
const WORLD_ENTRY: &str = "@compute @workgroup_size(256)\nfn global_tick(@builtin(local_invocation_id) lid: vec3u) {\n    let tid = lid.x;";

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

fn world128_source() -> String {
    let phases = [
        include_str!("../shaders/kernel/phase_clear.wgsl"),
        include_str!("../shaders/kernel/phase_food_grid.wgsl"),
        include_str!("../shaders/kernel/phase_food_respawn.wgsl"),
        include_str!("../shaders/kernel/phase_agent_grid.wgsl"),
        include_str!("../shaders/kernel/phase_grid_order.wgsl"),
        include_str!("../shaders/kernel/phase_collision.wgsl"),
        include_str!("../shaders/kernel/phase_trail_sample.wgsl"),
    ]
    .join("\n");
    assert_eq!(phases.matches("+= 256u").count(), WORLD_STRIDED_LOOPS);
    assert!(!phases.contains("brain_state"));
    assert!(!phases.contains("sensory_buffer"));
    let phases = phases.replace("+= 256u", "+= FOUR_STAGE_WORLD_THREADS");
    let original = include_str!("../shaders/kernel/global_tick.wgsl");
    assert_eq!(original.matches(WORLD_ENTRY).count(), 1);
    // Only the canonical body is retained: no second push-constant variable.
    let body = &original[original.find(WORLD_ENTRY).unwrap() + WORLD_ENTRY.len()..];
    let body = body
        .replace("gpc.tick", "world_tick")
        .replace("gpc.trail_interval", "trail_interval");
    let source = format!(
        "const FOUR_STAGE_WORLD_THREADS: u32 = {WORLD_THREADS}u;\n{phases}\nfn four_stage_world(tid: u32, world_tick: u32, trail_interval: u32) {{{body}"
    );
    assert!(!source.contains("gpc"));
    assert!(!source.contains("var<push_constant>"));
    assert!(shared_declarations(&source).is_empty());
    source
}

fn brain_world_source(kernel: &GpuKernel) -> String {
    let original =
        global_credit::main_source(&optimized_brain(), kernel.has_subgroup, Some(cache(kernel)));
    let main = main_width::try_transform(&original).unwrap();
    let entry = include_str!("../shaders/kernel/four_stage_brain_world.wgsl")
        .replace(
            "// FOUR_STAGE_SUBGROUP_PARAMS",
            if kernel.has_subgroup {
                "@builtin(subgroup_invocation_id) sgid: u32,"
            } else {
                ""
            },
        )
        .replace(
            " /* FOUR_STAGE_SUBGROUP_ARGS */",
            if kernel.has_subgroup { ", sgid" } else { "" },
        );
    let candidate = [main.as_str(), &world128_source(), &entry].join("\n");
    assert_eq!(shared_declarations(&candidate), shared_declarations(&main));
    assert_eq!(candidate.matches("var<push_constant>").count(), 1);
    candidate
}

fn vision_credit_source(kernel: &GpuKernel) -> String {
    let common = with_plain_grid_bindings(&cache(kernel).common_source());
    let vision = compose_vision_source(&common, true, true);
    let helper = replace_once(
        &vision,
        VISION_ENTRY,
        "fn four_stage_vision_inner(lid: vec3u, wgid: vec3u)",
    );
    let credit = packed_encoder::credit_source(true);
    assert!(!credit.contains("sensory_buffer"));
    assert!(!credit.contains("Barrier"));
    let candidate = [
        helper.as_str(),
        credit.as_str(),
        include_str!("../shaders/kernel/four_stage_vision_credit.wgsl"),
    ]
    .join("\n");
    assert_eq!(
        shared_declarations(&candidate),
        shared_declarations(&vision)
    );
    candidate
}

fn shared_declarations(source: &str) -> Vec<String> {
    let mut declarations = Vec::new();
    let mut pending = String::new();
    for line in source.lines().map(str::trim) {
        if line.starts_with("var<workgroup>") || !pending.is_empty() {
            pending.push_str(line);
            if line.ends_with(';') {
                declarations.push(std::mem::take(&mut pending));
            }
        }
    }
    assert!(pending.is_empty());
    declarations
}

/// Count the actual composed globals, rejecting unknown shapes rather than
/// silently letting a source change invalidate the explicit resource bound.
fn shared_bound(source: &str, layout: &BrainLayout) -> u64 {
    assert!(!layout.visual_cortex_enabled);
    let words_for = |count: &str| match count.trim() {
        "FEATURE_COUNT" => layout.feature_count,
        "ENCODED_DIMENSION" | "PREDICTOR_DIMENSION" => ENCODED_DIMENSION,
        "MEMORY_CAP" => MEMORY_CAP,
        "RECALL_K" => crate::buffers::RECALL_K,
        "PACKED_ENCODER_PARTIAL_WORDS" => ENCODED_DIMENSION * ENCODER_PARTIAL_LANES,
        "VC_SCRATCH_LEN" => 1,
        "VISION_PARALLEL_MAX_RAYS_PER_WORKGROUP" => usize::try_from(RAYS_PER_GROUP).unwrap(),
        "VISION_OBJECT_CACHE_SIZE" => VISION_OBJECT_FOOD_CAPACITY
            .checked_add(OBJECT_AGENTS)
            .unwrap()
            .checked_add(usize::try_from(RAYS_PER_GROUP).unwrap())
            .unwrap()
            .checked_add(1)
            .unwrap(),
        "VISION_SCENT_CHUNK_SIZE" => SCENT_ITEMS,
        literal => literal.parse().unwrap(),
    };
    let mut total = 0usize;
    for declaration in shared_declarations(source) {
        let kind = declaration
            .split_once(':')
            .unwrap()
            .1
            .trim()
            .trim_end_matches(';');
        let (element, count) = if let Some(array) = kind.strip_prefix("array<") {
            let (element, count) = array.strip_suffix('>').unwrap().rsplit_once(',').unwrap();
            (element.trim(), words_for(count))
        } else {
            (kind, 1)
        };
        let element_words = match element {
            "f32" | "u32" | "atomic<u32>" => 1,
            "vec2<f32>" => 2,
            "vec4<f32>" => 4,
            other => panic!("unaccounted shared element: {other}"),
        };
        let bytes = count
            .checked_mul(element_words)
            .unwrap()
            .checked_mul(WORD_BYTES)
            .unwrap();
        let rounded =
            bytes.checked_add(SHARED_ALIGNMENT - 1).unwrap() / SHARED_ALIGNMENT * SHARED_ALIGNMENT;
        total = total.checked_add(rounded).unwrap();
    }
    u64::try_from(total).unwrap()
}

fn constants(kernel: &GpuKernel) -> HashMap<String, f64> {
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    constants
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
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: &[&binding],
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

fn prepare(width: u32, height: u32, boundary: bool) -> GpuKernel {
    let mut kernel = prepare_kernel_with_store_suppression(width, height, boundary, true);
    kernel.global_credit = global_credit::Pipelines::new_packed_with_main128(
        &kernel,
        &optimized_brain(),
        &constants(&kernel),
        true,
    );
    assert_eq!(
        kernel.global_credit.as_ref().unwrap().main_threads,
        MAIN_THREADS
    );
    assert!(!cache(&kernel).is_valid());
    kernel
}

struct Candidate {
    brain_world: wgpu::ComputePipeline,
    vision_credit: wgpu::ComputePipeline,
    brain_world_groups: u32,
    vision_credit_groups: u32,
    brain_shared_bytes: u64,
    vision_shared_bytes: u64,
}

impl Candidate {
    fn new(kernel: &GpuKernel) -> Self {
        assert!(kernel.global_credit_active());
        assert_eq!(
            kernel.global_credit.as_ref().unwrap().main_threads,
            MAIN_THREADS
        );
        assert!(!kernel.layout.visual_cortex_enabled);
        assert!(kernel.agent_count > 0 && kernel.agent_count <= WORLD_THREADS);
        assert!(kernel.agent_count <= u32::try_from(OBJECT_AGENTS).unwrap());
        assert!(kernel.food_count <= VISION_OBJECT_FOOD_CAPACITY);
        assert_eq!(
            kernel.vision_workgroups,
            kernel
                .agent_count
                .checked_mul(
                    kernel
                        .layout
                        .vision_width
                        .checked_mul(kernel.layout.vision_height)
                        .unwrap()
                        .div_ceil(RAYS_PER_GROUP)
                )
                .unwrap()
        );
        let brain_source = brain_world_source(kernel);
        let vision_source = vision_credit_source(kernel);
        let brain_shared_bytes = shared_bound(&brain_source, &kernel.layout);
        let vision_shared_bytes = shared_bound(&vision_source, &kernel.layout);
        for bytes in [brain_shared_bytes, vision_shared_bytes] {
            assert!(bytes <= u64::from(kernel.device.limits().max_compute_workgroup_storage_size));
        }
        let brain_world_groups = kernel.agent_count.checked_add(1).unwrap();
        let vision_credit_groups = kernel
            .vision_workgroups
            .checked_add(
                kernel
                    .agent_count
                    .checked_mul(cache(kernel).groups_per_agent())
                    .unwrap(),
            )
            .unwrap();
        for groups in [brain_world_groups, vision_credit_groups] {
            assert!(groups <= MAX_DISPATCH_WORKGROUPS);
            assert!(groups <= kernel.device.limits().max_compute_workgroups_per_dimension);
        }
        let mut overrides = constants(kernel);
        let brain_world = pipeline(kernel, brain_source, "four_stage_brain_world", &overrides);
        for name in [
            "VISION_PARALLEL_STEPS",
            "VISION_OBJECT_QUERIES",
            "VISION_PARALLEL_SCENT",
        ] {
            overrides.insert(name.into(), 1.0);
        }
        overrides.insert(
            "VISION_RAYS_PER_WORKGROUP".into(),
            f64::from(RAYS_PER_GROUP),
        );
        overrides.insert(
            "FOUR_STAGE_CREDIT_GROUPS_PER_AGENT".into(),
            f64::from(cache(kernel).groups_per_agent()),
        );
        let vision_credit = pipeline(
            kernel,
            vision_source,
            "four_stage_vision_credit",
            &overrides,
        );
        Self {
            brain_world,
            vision_credit,
            brain_world_groups,
            vision_credit_groups,
            brain_shared_bytes,
            vision_shared_bytes,
        }
    }

    fn record_cycle(
        &self,
        kernel: &GpuKernel,
        pass: &mut wgpu::ComputePass<'_>,
        tick: u32,
        candidate: bool,
    ) {
        let credit = kernel.global_credit.as_ref().unwrap();
        pass.set_pipeline(&kernel.kernel_claim_pipeline);
        pass.set_push_constants(0, bytemuck::cast_slice(&[tick, COMPLETE_BRAIN]));
        pass.dispatch_workgroups(kernel.agent_count, 1, 1);
        pass.set_pipeline(&credit.main);
        pass.set_push_constants(
            0,
            bytemuck::cast_slice(&[tick, if candidate { 0 } else { COMPLETE_BRAIN }]),
        );
        pass.dispatch_workgroups(kernel.agent_count, 1, 1);
        if candidate {
            pass.set_pipeline(&self.brain_world);
            pass.set_push_constants(0, bytemuck::cast_slice(&[tick, COMPLETE_BRAIN]));
            pass.dispatch_workgroups(self.brain_world_groups, 1, 1);
            pass.set_pipeline(&self.vision_credit);
            pass.dispatch_workgroups(self.vision_credit_groups, 1, 1);
        } else {
            let stride = kernel.brain_tick_stride;
            pass.set_pipeline(&credit.global);
            pass.set_push_constants(
                0,
                bytemuck::cast_slice(&[tick.checked_add(stride).unwrap(), stride]),
            );
            pass.dispatch_workgroups(credit.global_workgroups, 1, 1);
            pass.set_pipeline(&kernel.vision_pipeline);
            pass.dispatch_workgroups(kernel.vision_workgroups, 1, 1);
        }
    }

    fn advance(&self, kernel: &mut GpuKernel, cycle: u32, cycles: u32, candidate: bool) {
        assert!(kernel.global_credit_active() && !kernel.probe.skip_vision);
        assert_eq!(kernel.vision_stride, 1);
        assert_eq!(
            kernel.global_credit.as_ref().unwrap().main_threads,
            MAIN_THREADS
        );
        let stride = kernel.brain_tick_stride;
        let mut tick = cycle.checked_mul(stride).unwrap();
        kernel.upload_world_config_with_cycles(u64::from(tick), stride, PHASE_MASK, 1);
        let mut batch = 0u32;
        while batch < cycles {
            let end = batch.checked_add(MAX_FUSED_BATCHES).unwrap().min(cycles);
            let mut encoder = kernel.device.create_command_encoder(&Default::default());
            cache(kernel).record_import(kernel, &mut encoder);
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(
                    0,
                    &kernel.global_credit.as_ref().unwrap().bind_groups[kernel.active_config_index],
                    &[],
                );
                for _ in batch..end {
                    self.record_cycle(kernel, &mut pass, tick, candidate);
                    tick = tick.checked_add(stride).unwrap();
                }
            }
            kernel.queue.submit([encoder.finish()]);
            batch = end;
        }
        kernel.active_config_index = 1 - kernel.active_config_index;
        kernel.poll_wait();
    }
}

fn assert_grid_counts(kernel: &GpuKernel, food: &[f32], agents: &[f32]) {
    assert_eq!(food.len() % FOOD_GRID_CELL_STRIDE, 0);
    let cells = food.len() / FOOD_GRID_CELL_STRIDE;
    let agent_words = cells.checked_mul(AGENT_GRID_CELL_STRIDE).unwrap();
    assert!(agent_words <= agents.len());
    assert!(kernel.agent_count <= WORLD_THREADS);
    for cell in 0..cells {
        let food_count = food[cell * FOOD_GRID_CELL_STRIDE].to_bits();
        let agent_count = agents[cell * AGENT_GRID_CELL_STRIDE].to_bits();
        assert!(
            food_count <= u32::try_from(FOOD_GRID_MAX_PER_CELL).unwrap(),
            "food grid overflow in cell {cell}: {food_count}"
        );
        assert!(
            agent_count <= u32::try_from(AGENT_GRID_MAX_PER_CELL).unwrap(),
            "agent grid overflow in cell {cell}: {agent_count}"
        );
    }
}

/// Untimed per-cycle inspection detects an overflow even if later grid clears
/// hide it. No diagnostic work or counter is added to either timed shader.
fn assert_no_overflow(kernel: &GpuKernel) {
    let mut food = Vec::new();
    let mut agents = Vec::new();
    kernel.read_buffer_range(
        &kernel.food_grid_buffer,
        0,
        kernel.food_grid_buffer.size(),
        &mut food,
    );
    kernel.read_buffer_range(
        &kernel.agent_grid_buffer,
        0,
        kernel.agent_grid_buffer.size(),
        &mut agents,
    );
    assert_grid_counts(kernel, &food, &agents);
}

fn checked_state(kernel: &GpuKernel) -> TestResult<State> {
    let state = capture_state(kernel)?;
    assert_eq!(state.len(), MUTABLE_BUFFERS);
    assert_grid_counts(
        kernel,
        bytemuck::cast_slice(&state[FOOD_GRID_STATE]),
        bytemuck::cast_slice(&state[AGENT_GRID_STATE]),
    );
    assert_mirror(kernel, &state)?;
    Ok(state)
}

fn trajectory(kernel: &mut GpuKernel, candidate: &Candidate, arm: usize) -> TestResult<Vec<State>> {
    let mut states = Vec::new();
    let mut cycle = 0;
    for count in CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        for _ in 0..count {
            if arm == 0 {
                advance(kernel, cycle, 1);
            } else {
                candidate.advance(kernel, cycle, 1, arm == 2);
            }
            assert_no_overflow(kernel);
            cycle += 1;
        }
        states.push(checked_state(kernel)?);
    }
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= FORCED_DEATHS);
    Ok(states)
}

#[test]
fn four_dispatch_world_preserves_phase_order_and_covers_cells() {
    let source = world128_source();
    assert_eq!(
        source.matches("+= FOUR_STAGE_WORLD_THREADS").count(),
        WORLD_STRIDED_LOOPS
    );
    assert!(!source.contains("+= 256u"));
    assert!(source.contains("phase_food_respawn(tid, world_tick)"));
    assert!(source.contains("phase_trail_sample(tid, world_tick / trail_interval)"));
    // An odd length spans several 128-lane trips and leaves a short tail.
    let count = WORLD_THREADS
        .checked_mul(3)
        .unwrap()
        .checked_add(1)
        .unwrap();
    let mut owners = vec![0; usize::try_from(count).unwrap()];
    for thread in 0..WORLD_THREADS {
        for item in (thread..count).step_by(usize::try_from(WORLD_THREADS).unwrap()) {
            owners[usize::try_from(item).unwrap()] += 1;
        }
    }
    assert!(owners.into_iter().all(|owners| owners == 1));
    #[cfg(not(target_arch = "wasm32"))]
    {
        let complete = format!("{}\n{source}\n@compute @workgroup_size(128) fn world_probe(@builtin(local_invocation_index) tid: u32) {{ four_stage_world(tid, 1u, 1u); }}", include_str!("../shaders/kernel/common.wgsl"));
        let module = wgpu::naga::front::wgsl::parse_str(&complete)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&complete)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&complete)));
    }
}

#[test]
fn four_dispatch_shared_bounds_follow_composed_declarations() {
    let layout = BrainLayout::new(FIELDS[0].0, FIELDS[0].1);
    let main =
        packed_encoder::packed_passes(&global_credit::main_source(&optimized_brain(), false, None));
    let vision = compose_vision_source(
        &with_plain_grid_bindings(include_str!("../shaders/kernel/common.wgsl")),
        true,
        true,
    );
    assert_eq!(shared_bound(&main, &layout), RAW_MAIN_SHARED_BOUND);
    assert_eq!(shared_bound(&vision, &layout), VISION_SHARED_BOUND);
    for constant in [
        "const VISION_OBJECT_FOOD_CAPACITY: u32 = 256u;",
        "const VISION_OBJECT_AGENT_CAPACITY: u32 = 32u;",
        "const VISION_PARALLEL_MAX_RAYS_PER_WORKGROUP: u32 = 8u;",
        "const VISION_SCENT_CHUNK_SIZE: u32 = 256u;",
    ] {
        assert!(vision.contains(constant));
    }
}

#[test]
#[ignore = "requires GPU; exact complete schedule, production recorder and private mirror"]
fn four_dispatch_preserves_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare(width, height, true);
        let candidate = Candidate::new(&kernel);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let expected = trajectory(&mut kernel, &candidate, 0)?;
        for arm in 1..=2 {
            for replay in 0..REPLAYS {
                restore(&mut kernel, &saved);
                let actual = trajectory(&mut kernel, &candidate, arm)?;
                for (expected, actual) in expected.iter().zip(&actual) {
                    assert_state_equal(&kernel, expected, actual);
                    assert_inactive_agent_unchanged(
                        &kernel,
                        &initial,
                        actual,
                        INACTIVE_AGENT,
                        "four dispatch",
                    );
                }
                println!("FOUR_DISPATCH_PARITY width={width} height={height} arm={arm} replay={replay} cycles=100 exact_buffers={MUTABLE_BUFFERS} death_refresh=true private_mirror_exact=true production_reference=true no_grid_overflow_every_cycle=true brain_shared_bytes={} vision_shared_bytes={}", candidate.brain_shared_bytes, candidate.vision_shared_bytes);
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU complete-cycle benchmark; run explicitly in release mode"]
fn benchmark_four_dispatch() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let mut kernel = prepare(width, height, false);
    let candidate = Candidate::new(&kernel);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let saved = checkpoint(&kernel);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let expected = checked_state(&kernel)?;
    // Inspect every cycle before timing, including cells that later clear.
    // Both arms must reach the independently recorded production endpoint.
    for arm in [false, true] {
        restore(&mut kernel, &saved);
        for offset in 0..TIMED_CYCLES {
            candidate.advance(&mut kernel, WARMUP_CYCLES + offset, 1, arm);
            assert_no_overflow(&kernel);
        }
        assert_state_equal(&kernel, &expected, &checked_state(&kernel)?);
    }
    let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for pair in 0..TIMING_PAIRS {
        for arm in [pair % 2, 1 - pair % 2] {
            restore(&mut kernel, &saved);
            let start = Instant::now();
            candidate.advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES, arm != 0);
            timings[arm].push(start.elapsed().as_secs_f64());
            assert_state_equal(&kernel, &expected, &checked_state(&kernel)?);
        }
    }
    for values in &mut timings {
        values.sort_by(f64::total_cmp);
    }
    let reference = timings[0][TIMING_PAIRS / 2];
    let changed = timings[1][TIMING_PAIRS / 2];
    println!("FOUR_DISPATCH_TIMING warmup_cycles={WARMUP_CYCLES} cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} reference_seconds={reference:.9} candidate_seconds={changed:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true full_simulation=true same_chunking=true dispatches_per_cycle=4 cold_import_timed=true readback_timed=false no_overflow_preflight_every_cycle=true brain_shared_bytes={} vision_shared_bytes={}", reference/changed, candidate.brain_shared_bytes, candidate.vision_shared_bytes);
    Ok(())
}
