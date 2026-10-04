//! Test-only removal of compiler-generated workgroup initialization on the
//! complete raw-vision packed/main128 route. Storage bounds checks, ordinary
//! storage initialization, WGSL arithmetic, and all source barriers stay intact.
//! The production global/world/credit pipeline is retained in every arm.
//!
//! Written-before-read audit for this deliberately bounded route:
//! - Claim writes `s_alive` before its broadcast barrier. Every lower scan slot
//!   in similarities, indices, food distance and bearing arrays is initialized
//!   before the upper-half merge, including inactive agents and empty scans.
//! - Main initializes alive and danger scratch before their reads. Feature
//!   extraction covers every raw visual and nonvisual input; encode fills all
//!   512 packed partials and every encoded value. Homeostasis fills the memory
//!   key and homeostatic fields; the later predictor fills homeostasis slot7.
//! - Recall initializes every similarity/index and all seventeen recall words.
//!   Prediction fills every output and recalled-similarity slot. Each reduction
//!   initializes its consumed lower128 cells and publishes its scalar before
//!   reads. All five prediction/TD fields, credit and exploration values are
//!   written on every live complete-pass path before use.
//! - Context fills both consumed scratch regions each tile. Whitening fills
//!   its two matrices on refresh, and never reads them otherwise. Main128
//!   reinforcement fills both128-word halves before its sum; decay initializes
//!   every argmin input. Inactive agents skip the entire brain uniformly.
//! - Vision initializes all eight event atomics before reduction; cached origin,
//!   live food/agent slots and valid-ray directions precede their barrier.
//!   Padding/dead rays do not read an unwritten direction. Scent initializes
//!   every consumed chunk slot, and publishes its result even for zero food.
//!
//! Partial cortex stage limits are excluded: they return before producing the
//! visual feature head, whose implicit zeros are currently observable. This is
//! not an audit of every execution mode, cortex configuration or future shader.
//! An untimed poison arm explicitly fills every declared workgroup object with
//! varying finite/nonzero words before entry execution, then barriers. It makes
//! accidental dependence on initial zeros falsifiable even on drivers that
//! happen to return zeroed memory when automatic initialization is disabled.

use std::{collections::HashMap, error::Error, fmt::Write, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::packed_store_validation::{
    advance, assert_mirror, cache, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// Odd dimensions exercise the final ray and feature tiles.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// Checkpoints surround death, whitening refresh and submission boundaries.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Mature matrices and episodic memory are restored identically for each arm.
const WARMUP_CYCLES: u32 = 1_000;
/// Complete evolving cycles per timing arm.
const TIMED_CYCLES: u32 = 100;
/// Alternating first arms over an odd pair count gives an unambiguous median.
const TIMING_PAIRS: usize = 5;
/// Two independent replays detect stale scratch or hidden initial-state input.
const REPLAYS: usize = 2;
/// Production reference, explicit initialized source control, poison, no init.
const ARM_COUNT: usize = 4;
const REFERENCE: usize = 0;
const SOURCE_CONTROL: usize = 1;
const POISON: usize = 2;
const NO_INITIALIZATION: usize = 3;
const LABELS: [&str; ARM_COUNT] = ["production", "explicit_zero", "poison", "no_init"];
/// The persistent inactive fixture agent has no pending respawn.
const INACTIVE_AGENT: u32 = 1;
/// Initial and refresh-boundary forced deaths must both reach the trajectory.
const EXPECTED_DEATHS: f32 = 2.0;
/// Snapshot covers every mutable public simulation buffer.
const MUTABLE_BUFFERS: usize = 13;
/// Tick and pass-limit words match the production layouts.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Complete production parallel-ray group: eight rays, thirty-two lanes each.
const RAYS_PER_GROUP: u32 = BRAIN_WORKGROUP_THREADS / PARALLEL_VISION_LANES;
/// Source inventories make additions to scratch require a renewed audit.
const MAIN_SCRATCH_OBJECTS: usize = 31;
const VISION_SCRATCH_OBJECTS: usize = 5;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn constants(kernel: &GpuKernel) -> HashMap<String, f64> {
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    constants
}

fn prepare(width: u32, height: u32, boundary: bool) -> GpuKernel {
    let mut kernel = prepare_kernel_with_store_suppression(width, height, boundary, true);
    kernel.global_credit = global_credit::Pipelines::new_packed_with_main128(
        &kernel,
        &optimized_brain(),
        &constants(&kernel),
        true,
    );
    assert!(!kernel.layout.visual_cortex_enabled);
    assert!(kernel.global_credit_active());
    assert_eq!(
        kernel.global_credit.as_ref().unwrap().main_threads,
        main_width::MAIN_THREADS
    );
    assert_eq!(kernel.probe.kernel_pass_limit, 7);
    kernel
}

/// Read only declarations, including the two-line atomic event declaration.
/// Restrict supported types so a new scratch shape cannot be silently skipped.
fn declarations(source: &str) -> Vec<(String, String)> {
    let mut declarations = Vec::new();
    let mut lines = source.lines();
    while let Some(line) = lines.next() {
        let Some(rest) = line.trim().strip_prefix("var<workgroup> ") else {
            continue;
        };
        let mut declaration = rest.to_owned();
        while !declaration.contains(';') {
            declaration.push(' ');
            declaration.push_str(lines.next().expect("complete scratch declaration").trim());
        }
        let declaration = declaration.strip_suffix(';').unwrap();
        let (name, value_type) = declaration.split_once(':').unwrap();
        assert!(
            !value_type.contains('='),
            "explicit initializers need an audit"
        );
        declarations.push((name.trim().to_owned(), value_type.trim().to_owned()));
    }
    declarations
}

fn poison_store(destination: &str, value_type: &str, integer: &str) -> String {
    match value_type {
        "f32" => format!("{destination} = f32({integer});"),
        "u32" => format!("{destination} = {integer};"),
        "atomic<u32>" => format!("atomicStore(&{destination}, {integer});"),
        "vec2<f32>" | "vec4<f32>" => {
            format!("{destination} = {value_type}(f32({integer}));")
        }
        _ => panic!("scratch type needs an explicit audit: {value_type}"),
    }
}

/// Every poison value is finite, positive and nonzero; both object and element
/// vary, so false identity/zero assumptions do not receive neutral inputs.
fn poison_source(source: &str, entry: &str, threads: u32, expected_objects: usize) -> String {
    let objects = declarations(source);
    assert_eq!(objects.len(), expected_objects);
    let mut helper = String::from("\nfn poison_workgroup(tid: u32, group: u32, threads: u32) {\n");
    for (object, (name, value_type)) in objects.iter().enumerate() {
        let salt = object.checked_add(1).unwrap();
        if let Some(array) = value_type.strip_prefix("array<") {
            let (element, length) = array.strip_suffix('>').unwrap().rsplit_once(',').unwrap();
            let integer = format!("1u + ((i + group + {salt}u) % 97u)");
            let store = poison_store(&format!("{name}[i]"), element.trim(), &integer);
            writeln!(
                helper,
                "    for (var i = tid; i < u32({}); i += threads) {{ {store} }}",
                length.trim()
            )
            .unwrap();
        } else {
            let integer = format!("1u + ((group + {salt}u) % 97u)");
            let store = poison_store(name, value_type, &integer);
            writeln!(helper, "    if (tid == 0u) {{ {store} }}").unwrap();
        }
    }
    helper.push_str("    workgroupBarrier();\n}\n");
    let signature = format!("fn {entry}(");
    assert_eq!(source.matches(&signature).count(), 1);
    let begin = source.find(&signature).unwrap();
    let body = begin + source[begin..].find('{').unwrap() + 1;
    let mut poisoned = source.to_owned();
    poisoned.insert_str(
        body,
        &format!("\n    poison_workgroup(lid.x, wgid.x, {threads}u);\n"),
    );
    poisoned.push_str(&helper);
    assert_ne!(poisoned, source);
    poisoned
}

fn pipeline(
    kernel: &GpuKernel,
    source: &str,
    entry: &str,
    constants: &HashMap<String, f64>,
    initialize: bool,
) -> wgpu::ComputePipeline {
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("workgroup_initialization_probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("workgroup_initialization_probe"),
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
                zero_initialize_workgroup_memory: initialize,
            },
            cache: None,
        })
}

struct Variant {
    main: wgpu::ComputePipeline,
    claim: wgpu::ComputePipeline,
    vision: wgpu::ComputePipeline,
}

impl Variant {
    fn compile(kernel: &GpuKernel, initialize: bool, poison: bool) -> Self {
        assert!(!initialize || !poison);
        let passes = optimized_brain();
        let main = global_credit::main_source(&passes, kernel.has_subgroup, Some(cache(kernel)));
        let mut main = main_width::try_transform(&main).unwrap();
        // The independent claim keeps256 threads and the ordinary binding13.
        // Brain-only source changes are unreachable from this entry point.
        let mut claim = global_credit::main_source(&passes, kernel.has_subgroup, None);
        let common = with_plain_grid_bindings(include_str!("../shaders/kernel/common.wgsl"));
        let mut vision = compose_vision_source(&common, true, true);
        assert_eq!(declarations(&main).len(), MAIN_SCRATCH_OBJECTS);
        assert_eq!(declarations(&claim).len(), MAIN_SCRATCH_OBJECTS);
        assert_eq!(declarations(&vision).len(), VISION_SCRATCH_OBJECTS);
        if poison {
            main = poison_source(
                &main,
                "kernel_tick",
                main_width::MAIN_THREADS,
                MAIN_SCRATCH_OBJECTS,
            );
            claim = poison_source(
                &claim,
                "kernel_claim_tick",
                BRAIN_WORKGROUP_THREADS,
                MAIN_SCRATCH_OBJECTS,
            );
            vision = poison_source(
                &vision,
                "vision_tick",
                BRAIN_WORKGROUP_THREADS,
                VISION_SCRATCH_OBJECTS,
            );
        }
        let mut constants = constants(kernel);
        let main = pipeline(kernel, &main, "kernel_tick", &constants, initialize);
        let claim = pipeline(kernel, &claim, "kernel_claim_tick", &constants, initialize);
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
        let vision = pipeline(kernel, &vision, "vision_tick", &constants, initialize);
        Self {
            main,
            claim,
            vision,
        }
    }
}

struct Arms {
    parked: [Option<Variant>; ARM_COUNT],
    current: usize,
}

impl Arms {
    fn new(kernel: &GpuKernel) -> Self {
        Self {
            parked: [
                None,
                Some(Variant::compile(kernel, true, false)),
                Some(Variant::compile(kernel, false, true)),
                Some(Variant::compile(kernel, false, false)),
            ],
            current: REFERENCE,
        }
    }

    fn activate(&mut self, kernel: &mut GpuKernel, arm: usize) {
        if arm != self.current {
            let next = self.parked[arm].take().unwrap();
            self.parked[self.current] = Some(Variant {
                main: std::mem::replace(
                    &mut kernel.global_credit.as_mut().unwrap().main,
                    next.main,
                ),
                claim: std::mem::replace(&mut kernel.kernel_claim_pipeline, next.claim),
                vision: std::mem::replace(&mut kernel.vision_pipeline, next.vision),
            });
            self.current = arm;
        }
        // Every arm shares the same private cache and production global pipeline.
        // Checkpoint restore handles its invalidation; no parked cache exists.
        assert!(kernel.global_credit_active());
    }
}

fn trajectory(kernel: &mut GpuKernel) -> TestResult<Vec<State>> {
    let mut cycle = 0;
    let mut states = Vec::new();
    for count in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, cycle, count);
        let state = capture_state(kernel)?;
        assert_mirror(kernel, &state)?;
        states.push(state);
        cycle += count;
    }
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= EXPECTED_DEATHS);
    Ok(states)
}

#[test]
#[ignore = "requires a GPU; source control, poisoned scratch and explicit no-init"]
fn workgroup_initialization_preserves_complete_raw_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare(width, height, true);
        let mut arms = Arms::new(&kernel);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let expected = trajectory(&mut kernel)?;
        for arm in [SOURCE_CONTROL, POISON, NO_INITIALIZATION] {
            for replay in 0..REPLAYS {
                restore(&mut kernel, &saved);
                arms.activate(&mut kernel, arm);
                let actual = trajectory(&mut kernel)?;
                for (expected, actual) in expected.iter().zip(&actual) {
                    assert_state_equal(&kernel, expected, actual);
                    assert_inactive_agent_unchanged(
                        &kernel,
                        &initial,
                        actual,
                        INACTIVE_AGENT,
                        LABELS[arm],
                    );
                }
                println!(
                    "WORKGROUP_INIT_PARITY width={width} height={height} arm={} cycles=100 replay={replay} exact_buffers={MUTABLE_BUFFERS} death_refresh=true inactive_exact=true private_mirror_exact=true global_unchanged=true cortex=false",
                    LABELS[arm]
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU full-cycle benchmark; run explicitly in release mode"]
fn benchmark_workgroup_initialization() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let mut kernel = prepare(width, height, false);
    let mut arms = Arms::new(&kernel);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let expected = capture_state(&kernel)?;
    // Poison and source controls are correctness gates, never timed candidates.
    for arm in [SOURCE_CONTROL, POISON, NO_INITIALIZATION] {
        restore(&mut kernel, &warm);
        arms.activate(&mut kernel, arm);
        advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
        let actual = capture_state(&kernel)?;
        assert_state_equal(&kernel, &expected, &actual);
        assert_mirror(&kernel, &actual)?;
    }
    let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for pair in 0..TIMING_PAIRS {
        for slot in [pair % 2, 1 - pair % 2] {
            let arm = if slot == 0 {
                REFERENCE
            } else {
                NO_INITIALIZATION
            };
            restore(&mut kernel, &warm);
            arms.activate(&mut kernel, arm);
            let start = Instant::now();
            advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
            timings[slot].push(start.elapsed().as_secs_f64());
            let actual = capture_state(&kernel)?;
            assert_state_equal(&kernel, &expected, &actual);
            assert_mirror(&kernel, &actual)?;
        }
    }
    for values in &mut timings {
        values.sort_by(f64::total_cmp);
    }
    let reference = timings[0][TIMING_PAIRS / 2];
    let candidate = timings[1][TIMING_PAIRS / 2];
    println!(
        "WORKGROUP_INIT_TIMING warmup_cycles={WARMUP_CYCLES} cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true full_simulation=true same_production_recorder=true cold_import_timed=true global_unchanged=true initialization_disabled=claim,main128,vision poison_untimed=true cortex=false",
        reference / candidate
    );
    Ok(())
}

#[test]
fn workgroup_poison_covers_every_declared_shape() {
    let source = r"
// var<workgroup> ignored_comment: f32;
var<workgroup> floats: array<f32, 8>;
var<workgroup> indices: array<u32, 8>;
var<workgroup> events:
    array<atomic<u32>, 8>;
var<workgroup> vectors: array<vec4<f32>, 8>;
var<workgroup> scalar: f32;
var<workgroup> alive: u32;
var<workgroup> result: vec2<f32>;
@compute @workgroup_size(256)
fn probe(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wgid: vec3u) {
    scalar = 0.0;
}
";
    let changed = poison_source(source, "probe", BRAIN_WORKGROUP_THREADS, 7);
    assert_ne!(changed, source);
    assert_eq!(declarations(&changed), declarations(source));
    assert!(changed.contains("atomicStore(&events[i],"));
    assert!(changed.contains("vectors[i] = vec4<f32>(f32("));
    assert!(changed.contains("result = vec2<f32>(f32("));
    assert_eq!(
        changed
            .matches("poison_workgroup(lid.x, wgid.x, 256u);")
            .count(),
        1
    );
    assert_eq!(changed.matches("workgroupBarrier();").count(), 1);
    assert!(
        changed.find("poison_workgroup(lid.x").unwrap() < changed.find("scalar = 0.0;").unwrap()
    );
}
