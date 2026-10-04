//! Isolate per-entry binding layouts without changing any WGSL source or
//! access qualifier. Naga's transitive entry-point usage removes only globals
//! the selected entry never accesses. The full-layout control retains the
//! production pipelines, and its custom recorder is checked independently
//! against production before comparing the candidate's thirteen buffers.

#![cfg(not(target_arch = "wasm32"))]

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::main_width::MAIN_THREADS;
use super::packed_store_validation::{
    advance, assert_mirror, cache, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
const CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const REPLAYS: usize = 2;
const WARMUP_CYCLES: u32 = 1_000;
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;
const COMPLETE_BRAIN: u32 = 7;
const PHASE_MASK: u32 = 7;
const PUSH_CONSTANT_BYTES: u32 = 8;
const FULL_BINDINGS: usize = 18;
const MUTABLE_BUFFERS: usize = 13;
const INACTIVE_AGENT: u32 = 1;
const FORCED_DEATHS: f32 = 2.0;
const RAYS_PER_GROUP: u32 = BRAIN_WORKGROUP_THREADS / PARALLEL_VISION_LANES;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

#[derive(Clone, Debug)]
struct UsedBinding {
    binding: u32,
    kind: wgpu::BufferBindingType,
    usage: wgpu::naga::valid::GlobalUse,
}

/// Keep declarations unchanged: a read-only use of an RW declaration stays RW.
/// QUERY and ATOMIC count as usage, including through reachable callees.
fn used_bindings(source: &str, entry: &str) -> Vec<UsedBinding> {
    use wgpu::naga::{valid, AddressSpace, StorageAccess};
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    let index = module
        .entry_points
        .iter()
        .position(|point| point.name == entry)
        .expect("selected entry point");
    let function = info.get_entry_point(index);
    let mut bindings = Vec::new();
    for (handle, variable) in module.global_variables.iter() {
        let Some(resource) = &variable.binding else {
            continue;
        };
        let usage = function[handle];
        if usage.is_empty() {
            continue;
        }
        assert_eq!(
            resource.group, 0,
            "only the canonical bind group is supported"
        );
        let kind = match variable.space {
            AddressSpace::Uniform => wgpu::BufferBindingType::Uniform,
            AddressSpace::Storage { access } => wgpu::BufferBindingType::Storage {
                read_only: !access.contains(StorageAccess::STORE),
            },
            other => panic!("unexpected bound address space: {other:?}"),
        };
        bindings.push(UsedBinding {
            binding: resource.binding,
            kind,
            usage,
        });
    }
    bindings.sort_by_key(|resource| resource.binding);
    assert!(!bindings.is_empty());
    assert!(bindings
        .windows(2)
        .all(|pair| pair[0].binding != pair[1].binding));
    bindings
}

/// This is the canonical eighteen-binding ordering from private_bind_group.
/// Binding thirteen remains the SAME production packed cache in every arm.
fn buffers(kernel: &GpuKernel, config: usize) -> [&wgpu::Buffer; FULL_BINDINGS] {
    [
        &kernel.agent_phys_buffer,
        &kernel.decision_buffer,
        &kernel.heightmap_buffer,
        &kernel.biome_buffer,
        &kernel.world_config_bufs[config],
        &kernel.food_state_buffer,
        &kernel.food_flags_buffer,
        &kernel.food_grid_buffer,
        &kernel.agent_grid_buffer,
        &kernel.collision_scratch_buffer,
        &kernel.sensory_buffer,
        &kernel.brain_state_buffer,
        &kernel.pattern_buffer,
        cache(kernel).buffer(),
        &kernel.brain_config_buffer,
        &kernel.dispatch_args_buffer,
        &kernel.trail_ring_buffer,
        &kernel._sensory_next_buffer,
    ]
}

fn constants(kernel: &GpuKernel) -> HashMap<String, f64> {
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    constants
}

/// Match cached_combined_validation::install_pipelines, which establishes the
/// explicit claim reference before the packed main is installed. Unreachable
/// brain helpers are retained, so even shader-module text remains unchanged.
fn claim_source(kernel: &GpuKernel) -> String {
    let fused = predictor_fusion::fuse_inline_predictor(&compose_brain_passes(true));
    let prefetched = dense_prefetch::prefetch_passes(&fused, 8);
    let brain = predictor_width::wider_predictor(&prefetched, 16);
    apply_subgroup_markers(
        &[
            include_str!("../shaders/kernel/common.wgsl"),
            &brain,
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            include_str!("../shaders/kernel/phase_food_claim.wgsl"),
            include_str!("../shaders/kernel/kernel_tick.wgsl"),
        ]
        .join("\n"),
        kernel.has_subgroup,
    )
}

fn prepare(width: u32, height: u32, boundary: bool) -> GpuKernel {
    let mut kernel = prepare_kernel_with_store_suppression(width, height, boundary, true);
    kernel.global_credit = global_credit::Pipelines::new_packed_with_main128(
        &kernel,
        &optimized_brain(),
        &constants(&kernel),
        true,
    );
    assert!(kernel.global_credit_active());
    assert_eq!(
        kernel.global_credit.as_ref().unwrap().main_threads,
        MAIN_THREADS
    );
    assert!(!cache(&kernel).is_valid());
    kernel
}

struct Stage {
    pipeline: wgpu::ComputePipeline,
    groups: [wgpu::BindGroup; 2],
}

impl Stage {
    fn new(
        kernel: &GpuKernel,
        source: String,
        entry: &str,
        constants: &HashMap<String, f64>,
    ) -> Self {
        let used = used_bindings(&source, entry);
        assert!(
            used.len() < FULL_BINDINGS,
            "the candidate must omit a binding"
        );
        let entries: Vec<_> = used
            .iter()
            .map(|resource| wgpu::BindGroupLayoutEntry {
                binding: resource.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: resource.kind,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let binding = kernel
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(entry),
                entries: &entries,
            });
        let groups = std::array::from_fn(|config| {
            let resources = buffers(kernel, config);
            let entries: Vec<_> = used
                .iter()
                .map(|resource| wgpu::BindGroupEntry {
                    binding: resource.binding,
                    resource: resources[usize::try_from(resource.binding).unwrap()]
                        .as_entire_binding(),
                })
                .collect();
            kernel.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(entry),
                layout: &binding,
                entries: &entries,
            })
        });
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
        // The exact original string is handed to wgpu. Usage analysis above
        // does not transform, prune or re-emit the shader module.
        let module = kernel
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = kernel
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
            });
        println!("MINIMAL_BINDING_LAYOUT entry={entry} full_bindings={FULL_BINDINGS} retained_bindings={} source_unchanged=true qualifiers_unchanged=true resources={used:?}", used.len());
        Self { pipeline, groups }
    }
}

struct Candidate {
    claim: Stage,
    main: Stage,
    global: Stage,
    vision: Stage,
}

impl Candidate {
    fn new(kernel: &GpuKernel) -> Self {
        assert_eq!(kernel.vision_stride, 1);
        assert!(!kernel.layout.visual_cortex_enabled);
        let claim = Stage::new(
            kernel,
            claim_source(kernel),
            "kernel_claim_tick",
            &vision_override_constants(&kernel.layout),
        );
        let mut overrides = constants(kernel);
        overrides.insert(
            "GLOBAL_CREDIT_GROUPS_PER_AGENT".into(),
            f64::from(cache(kernel).groups_per_agent()),
        );
        let main_source = main_width::try_transform(&global_credit::main_source(
            &optimized_brain(),
            kernel.has_subgroup,
            Some(cache(kernel)),
        ))
        .unwrap();
        let main = Stage::new(kernel, main_source, "kernel_tick", &overrides);
        let global = Stage::new(
            kernel,
            global_credit::global_source_with_store_suppression(Some(cache(kernel)), true),
            "global_credit_tick",
            &overrides,
        );
        let mut vision_constants = constants(kernel);
        for name in [
            "VISION_PARALLEL_STEPS",
            "VISION_OBJECT_QUERIES",
            "VISION_PARALLEL_SCENT",
        ] {
            vision_constants.insert(name.into(), 1.0);
        }
        vision_constants.insert(
            "VISION_RAYS_PER_WORKGROUP".into(),
            f64::from(RAYS_PER_GROUP),
        );
        let common = with_plain_grid_bindings(include_str!("../shaders/kernel/common.wgsl"));
        let vision = Stage::new(
            kernel,
            compose_vision_source(&common, true, true),
            "vision_tick",
            &vision_constants,
        );
        assert_eq!(
            kernel.vision_workgroups,
            kernel.agent_count
                * (kernel.layout.vision_width * kernel.layout.vision_height)
                    .div_ceil(RAYS_PER_GROUP)
        );
        Self {
            claim,
            main,
            global,
            vision,
        }
    }

    fn record_cycle(
        &self,
        kernel: &GpuKernel,
        pass: &mut wgpu::ComputePass<'_>,
        tick: u32,
        minimal: bool,
    ) {
        let credit = kernel.global_credit.as_ref().unwrap();
        let config = kernel.active_config_index;
        let full = &credit.bind_groups[config];
        let stride = kernel.brain_tick_stride;
        for (stage, original, push, groups) in [
            (
                &self.claim,
                &kernel.kernel_claim_pipeline,
                [tick, COMPLETE_BRAIN],
                kernel.agent_count,
            ),
            (
                &self.main,
                &credit.main,
                [tick, COMPLETE_BRAIN],
                kernel.agent_count,
            ),
            (
                &self.global,
                &credit.global,
                [tick.checked_add(stride).unwrap(), stride],
                credit.global_workgroups,
            ),
            (
                &self.vision,
                &kernel.vision_pipeline,
                [tick.checked_add(stride).unwrap(), stride],
                kernel.vision_workgroups,
            ),
        ] {
            pass.set_pipeline(if minimal { &stage.pipeline } else { original });
            pass.set_bind_group(0, if minimal { &stage.groups[config] } else { full }, &[]);
            pass.set_push_constants(0, bytemuck::cast_slice(&push));
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }

    fn advance(&self, kernel: &mut GpuKernel, cycle: u32, cycles: u32, minimal: bool) {
        assert!(kernel.global_credit_active() && !kernel.probe.skip_vision);
        assert_eq!(kernel.vision_stride, 1);
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
                for _ in batch..end {
                    self.record_cycle(kernel, &mut pass, tick, minimal);
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

fn checked_state(kernel: &GpuKernel) -> TestResult<State> {
    let state = capture_state(kernel)?;
    assert_eq!(state.len(), MUTABLE_BUFFERS);
    assert_mirror(kernel, &state)?;
    Ok(state)
}

fn trajectory(kernel: &mut GpuKernel, candidate: &Candidate, arm: usize) -> TestResult<Vec<State>> {
    let mut cycle = 0;
    let mut states = Vec::new();
    for count in CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        match arm {
            0 => advance(kernel, cycle, count),
            1 | 2 => candidate.advance(kernel, cycle, count, arm == 2),
            _ => panic!("unknown arm"),
        }
        states.push(checked_state(kernel)?);
        cycle += count;
    }
    let physics: &[f32] = bytemuck::cast_slice(&states.last().unwrap()[0]);
    assert!(
        physics[P_DEATH_COUNT] >= FORCED_DEATHS,
        "the forced-death fixture must execute"
    );
    Ok(states)
}

#[test]
#[ignore = "requires a GPU; original versus minimal per-entry binding layouts"]
fn minimal_bindings_preserve_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare(width, height, true);
        let candidate = Candidate::new(&kernel);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let expected = trajectory(&mut kernel, &candidate, 0)?;
        for arm in [1, 2] {
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
                        "minimal bindings",
                    );
                }
                println!("MINIMAL_BINDING_PARITY width={width} height={height} arm={arm} replay={replay} cycles=100 exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true death_refresh=true custom_reference_matches_production=true source_unchanged=true access_qualifiers_unchanged=true");
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU complete-cycle benchmark; run explicitly in release mode"]
fn benchmark_minimal_bindings() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let mut kernel = prepare(width, height, false);
    let candidate = Candidate::new(&kernel);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let saved = checkpoint(&kernel);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let expected = checked_state(&kernel)?;
    for minimal in [false, true] {
        restore(&mut kernel, &saved);
        candidate.advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES, minimal);
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
    println!("MINIMAL_BINDING_TIMING warmup_cycles={WARMUP_CYCLES} cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} reference_seconds={reference:.9} candidate_seconds={changed:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true full_simulation=true source_unchanged=true qualifiers_unchanged=true custom_reference_matches_production=true same_chunking=true same_bind_calls=true cold_import_timed=true completion_timed=true readback_timed=false", reference / changed);
    Ok(())
}

#[test]
fn entry_usage_drops_unused_bindings_and_preserves_access() {
    use wgpu::naga::valid::GlobalUse;
    let source = r"
@group(0) @binding(0) var<storage, read_write> unused: array<f32>;
@group(0) @binding(1) var<storage, read_write> declared_rw: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> read_values: array<f32>;
@group(0) @binding(4) var<storage, read_write> queried: array<f32>;
@group(0) @binding(5) var<storage, read_write> atomic_input: atomic<u32>;
fn load_input() -> f32 { return declared_rw[0]; }
@compute @workgroup_size(1)
fn probe() {
    output[0] = load_input() + read_values[0] + f32(arrayLength(&queried)) + f32(atomicLoad(&atomic_input));
}
@compute @workgroup_size(1)
fn unrelated() { unused[0] = 1.0; }
";
    let bindings = used_bindings(source, "probe");
    assert_eq!(
        bindings.iter().map(|item| item.binding).collect::<Vec<_>>(),
        [1, 2, 3, 4, 5]
    );
    assert!(bindings[0].usage.contains(GlobalUse::READ));
    assert!(!bindings[0].usage.contains(GlobalUse::WRITE));
    assert_eq!(
        bindings[0].kind,
        wgpu::BufferBindingType::Storage { read_only: false }
    );
    assert!(bindings[1].usage.contains(GlobalUse::WRITE));
    assert_eq!(
        bindings[1].kind,
        wgpu::BufferBindingType::Storage { read_only: false }
    );
    assert_eq!(
        bindings[2].kind,
        wgpu::BufferBindingType::Storage { read_only: true }
    );
    assert!(bindings[3].usage.contains(GlobalUse::QUERY));
    assert!(!bindings[4].usage.is_empty());
    assert_eq!(
        bindings[4].kind,
        wgpu::BufferBindingType::Storage { read_only: false }
    );
}
