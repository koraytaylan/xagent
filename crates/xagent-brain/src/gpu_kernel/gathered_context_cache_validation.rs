//! Cache normalization once for the production gathered-context main128.
//! Sixteen coefficients and their ordered total reuse released packed encoder
//! scratch. Raw pattern staging, blend order, guards and tanh remain unchanged.
//! The comparison uses the actual production packed/store-suppression baseline;
//! no recorder, allocation, production route or shader counter is introduced.

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::main_width::MAIN_THREADS;
use super::packed_store_validation::{
    advance, assert_mirror, cache, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::vision_validation::read_buffer;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// Raw fields include an odd width and partial packed-credit tiles.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// Boundaries include two deaths, whitening refresh and one hundred cycles.
const CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const REPLAYS: usize = 2;
const FORCED_DEATHS: f32 = 2.0;
const INACTIVE_AGENT: u32 = 1;
const MUTABLE_BUFFERS: usize = 13;
/// Match the mature current-production benchmark fixture.
const WARMUP_CYCLES: u32 = 1_000;
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;
const PUSH_BYTES: u32 = 8;
const WORD_BYTES: usize = size_of::<f32>();
/// Main128 gathers sixteen outputs per tile with eight loaders per output.
const GATHER_LANES: u32 = 8;
/// The packed encoder allocates four partials for each encoded dimension.
const PACKED_PARTIALS: usize = ENCODED_DIMENSION * 4;
const COEFFICIENT_BASE: usize = ENCODED_DIMENSION * 2;
const TOTAL_SLOT: usize = COEFFICIENT_BASE + RECALL_K;
/// The canonical context blend requires a strictly positive total above this.
const TOTAL_GUARD: f32 = 1e-8;
const RAW_CASES: usize = 10;
const PARTIAL_RECALL: usize = 7;
const CONTEXT_WEIGHT: f32 = 0.3;
const INPUT_SCALE: f32 = 0.25;
const TINY_SIMILARITY: f32 = 1e-10;
/// Finite poison exposes reading coefficients outside the initialized range.
const UNUSED_SIMILARITY: f32 = 100.0;
const HELPER: &str = include_str!("../shaders/kernel/brain_gathered_context_cache.wgsl");
const GATHER_DECLARATION: &str = "const CONTEXT_GATHER_LANES: u32 = RECALL_K;";
const PACKED_DECLARATION: &str =
    "var<workgroup> s_dense_partials: array<f32, PACKED_ENCODER_PARTIAL_WORDS>;";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn replace_once(source: &str, before: &str, after: &str) -> String {
    assert_eq!(
        source.matches(before).count(),
        1,
        "unique source target: {before}"
    );
    source.replacen(before, after, 1)
}

fn original_helper() -> String {
    replace_once(
        include_str!("../shaders/kernel/context_gather.wgsl"),
        GATHER_DECLARATION,
        &format!("const CONTEXT_GATHER_LANES: u32 = {GATHER_LANES}u;"),
    )
}

/// Apply only after the production packed main128 source has been composed.
/// Exact helper matching protects the staging/ordering contract from drift.
pub(super) fn cached_context_source(source: &str) -> String {
    for required in [
        "const BRAIN_WORKGROUP_SIZE: u32 = 128u;",
        "@compute @workgroup_size(128)\nfn kernel_tick(",
        "const ENCODED_DIMENSION: u32 = 128u;",
        "const RECALL_K: u32 = 16u;",
        "const PACKED_ENCODER_INNER_LANES: u32 = 4u;",
        "const PACKED_ENCODER_PARTIAL_WORDS: u32 = ENCODED_DIMENSION * PACKED_ENCODER_INNER_LANES;",
        PACKED_DECLARATION,
        "    blend_gathered_recalled_context(brain_base, pattern_base, recall_count, tid);",
    ] {
        assert_eq!(
            source.matches(required).count(),
            1,
            "source contract: {required}"
        );
    }
    assert!(!source.contains("CONTEXT_COEFFICIENT_BASE"));
    assert!(TOTAL_SLOT < PACKED_PARTIALS);
    assert!(COEFFICIENT_BASE >= usize::try_from(MAIN_THREADS).unwrap());
    let candidate = replace_once(source, &original_helper(), HELPER);
    assert_ne!(candidate, source);
    for token in ["var<workgroup>", "storageBarrier();"] {
        assert_eq!(
            candidate.matches(token).count(),
            source.matches(token).count()
        );
    }
    assert_eq!(
        candidate.matches("workgroupBarrier();").count(),
        source.matches("workgroupBarrier();").count() + 1
    );
    // Motor blending consumes these original similarities after context.
    assert_eq!(
        candidate.matches("s_recall_similarity[").count(),
        source.matches("s_recall_similarity[").count()
    );
    candidate
}

fn constants(kernel: &GpuKernel) -> HashMap<String, f64> {
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    constants
}

fn reference_source(kernel: &GpuKernel) -> String {
    main_width::try_transform(&global_credit::main_source(
        &optimized_brain(),
        kernel.has_subgroup,
        Some(cache(kernel)),
    ))
    .unwrap()
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
    kernel
}

fn pipeline(kernel: &GpuKernel, source: String, entry: &str) -> wgpu::ComputePipeline {
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gathered_context_cache"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gathered_context_cache"),
            bind_group_layouts: &[&binding],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_BYTES,
            }],
        });
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gathered_context_cache"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants(kernel),
                ..Default::default()
            },
            cache: None,
        })
}

struct Arms {
    parked: wgpu::ComputePipeline,
    candidate: bool,
}

impl Arms {
    fn new(kernel: &GpuKernel) -> Self {
        Self {
            parked: pipeline(
                kernel,
                cached_context_source(&reference_source(kernel)),
                "kernel_tick",
            ),
            candidate: false,
        }
    }

    fn activate(&mut self, kernel: &mut GpuKernel, candidate: bool) {
        if self.candidate != candidate {
            // Both main handles use the same attached cache; restore already
            // invalidates it before the production recorder imports weights.
            std::mem::swap(
                &mut kernel.global_credit.as_mut().unwrap().main,
                &mut self.parked,
            );
            self.candidate = candidate;
        }
        assert!(kernel.global_credit_active());
    }
}

fn trajectory(kernel: &mut GpuKernel) -> TestResult<Vec<State>> {
    let mut cycle = 0;
    let mut states = Vec::new();
    for count in CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, cycle, count);
        let state = capture_state(kernel)?;
        assert_mirror(kernel, &state)?;
        states.push(state);
        cycle += count;
    }
    assert_eq!(cycle, TIMED_CYCLES);
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= FORCED_DEATHS);
    Ok(states)
}

#[test]
#[ignore = "GPU exact gathered-context coefficient cache state and private mirror"]
fn gathered_context_coefficients_preserve_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare(width, height, true);
        let mut arms = Arms::new(&kernel);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let reference = trajectory(&mut kernel)?;
        for replay in 0..REPLAYS {
            for candidate in [false, true] {
                restore(&mut kernel, &saved);
                arms.activate(&mut kernel, candidate);
                let actual = trajectory(&mut kernel)?;
                let mut through_cycle = 0;
                for ((expected, actual), count) in reference.iter().zip(&actual).zip(CHUNKS) {
                    through_cycle += count;
                    println!("GATHERED_CONTEXT_CACHE_CHECK width={width} height={height} candidate={candidate} replay={replay} through_cycle={through_cycle}");
                    assert_state_equal(&kernel, expected, actual);
                    assert_inactive_agent_unchanged(
                        &kernel,
                        &initial,
                        actual,
                        INACTIVE_AGENT,
                        "gathered context cache",
                    );
                }
                println!("GATHERED_CONTEXT_CACHE_PARITY width={width} height={height} candidate={candidate} replay={replay} cycles={TIMED_CYCLES} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true death_refresh=true main_threads={MAIN_THREADS}");
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU full-cycle benchmark; run in release mode"]
fn benchmark_gathered_context_coefficients() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let mut kernel = prepare(width, height, false);
    let mut arms = Arms::new(&kernel);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let saved = checkpoint(&kernel);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let expected = capture_state(&kernel)?;
    assert_mirror(&kernel, &expected)?;
    let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for pair in 0..TIMING_PAIRS {
        for arm in [pair % 2, 1 - pair % 2] {
            restore(&mut kernel, &saved);
            arms.activate(&mut kernel, arm != 0);
            let start = Instant::now();
            advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            let actual = capture_state(&kernel)?;
            assert_state_equal(&kernel, &expected, &actual);
            assert_mirror(&kernel, &actual)?;
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let reference = timings[0][TIMING_PAIRS / 2];
    let candidate = timings[1][TIMING_PAIRS / 2];
    println!("GATHERED_CONTEXT_CACHE_TIMING warmup_cycles={WARMUP_CYCLES} cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true full_simulation=true main_threads={MAIN_THREADS} context_lanes={GATHER_LANES} extra_barriers=1 extra_shared_bytes=0 cold_import_timed=true state_comparison_timed=false", reference / candidate);
    Ok(())
}

/// The actual helper writes its predictions into private scratch, leaving all
/// public state untouched. Input slots are explicit fixture-only staging.
const RAW_ENTRY: &str = r"
@compute @workgroup_size(128)
fn gathered_context_probe(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let agent_id = group.x;
    let brain_base = agent_id * BRAIN_STRIDE;
    let pattern_base = agent_id * PATTERN_STRIDE;
    let decision_base = agent_id * DECISION_STRIDE;
    let recall_count = u32(physics_state[agent_id * PHYS_STRIDE + P_PROCESSING_SLOTS]);
    s_prediction[tid] = decision_buffer[decision_base + DECISION_PREDICTION + tid];
    if (tid < RECALL_K) {
        s_recall[tid] = f32(tid);
        s_recall_similarity[tid] = decision_buffer[decision_base + DECISION_CREDIT + tid];
    }
    workgroupBarrier();
    blend_gathered_recalled_context(brain_base, pattern_base, recall_count, tid);
    packed_encoder.scratch[agent_id * BRAIN_SCRATCH_STRIDE + tid] = s_prediction[tid];
}
";

fn raw_case(agent: usize) -> (usize, Vec<f32>, f32) {
    let mut similarities = vec![UNUSED_SIMILARITY; RECALL_K];
    let count = match agent {
        0 => 0,
        2..=4 => 1,
        6 => PARTIAL_RECALL,
        _ => RECALL_K,
    };
    for (index, similarity) in similarities.iter_mut().enumerate().take(count) {
        *similarity = match agent {
            1 => -1.0,
            2 => f32::from_bits(TOTAL_GUARD.to_bits() - 1),
            3 => TOTAL_GUARD,
            4 => f32::from_bits(TOTAL_GUARD.to_bits() + 1),
            7 => TINY_SIMILARITY,
            _ => {
                if index.is_multiple_of(2) {
                    INPUT_SCALE
                } else {
                    -INPUT_SCALE
                }
            }
        };
    }
    let weight = if agent == 8 { 0.0 } else { CONTEXT_WEIGHT };
    (count, similarities, weight)
}

fn upload_raw_fixture(kernel: &GpuKernel) {
    let agents = usize::try_from(kernel.agent_count).unwrap();
    assert_eq!(agents, RAW_CASES);
    let mut brain = vec![0.0_f32; agents.checked_mul(kernel.layout.brain_stride).unwrap()];
    let mut patterns = vec![0.0_f32; agents.checked_mul(PATTERN_STRIDE).unwrap()];
    let mut decisions = vec![0.0_f32; agents.checked_mul(DECISION_STRIDE).unwrap()];
    let fixed = fixed_tail_base(kernel.layout.brain_stride);
    for agent in 0..agents {
        let (count, similarities, weight) = raw_case(agent);
        let count_float = f32::from(u16::try_from(count).unwrap());
        kernel.write_agent_physics_fields(
            u32::try_from(agent).unwrap(),
            &[(P_PROCESSING_SLOTS, count_float)],
        );
        let brain_base = agent * kernel.layout.brain_stride;
        brain[brain_base + fixed] = weight;
        let decision_base = agent * DECISION_STRIDE;
        decisions[decision_base + DECISION_CREDIT..decision_base + DECISION_CREDIT + RECALL_K]
            .copy_from_slice(&similarities);
        for dim in 0..ENCODED_DIMENSION {
            let positive = dim.is_multiple_of(2);
            decisions[decision_base + dim] = if positive { 0.0 } else { -0.0 };
            brain[brain_base + fixed + O_ENCODED_MEAN - O_PREDICTOR_CONTEXT_WEIGHT + dim] =
                if positive { INPUT_SCALE } else { -INPUT_SCALE };
            for pattern in 0..RECALL_K {
                patterns[agent * PATTERN_STRIDE + dim * MEMORY_CAP + pattern] =
                    if (dim + pattern).is_multiple_of(2) {
                        INPUT_SCALE
                    } else {
                        -INPUT_SCALE
                    };
            }
        }
    }
    kernel
        .queue
        .write_buffer(&kernel.brain_state_buffer, 0, bytemuck::cast_slice(&brain));
    kernel
        .queue
        .write_buffer(&kernel.pattern_buffer, 0, bytemuck::cast_slice(&patterns));
    kernel
        .queue
        .write_buffer(&kernel.decision_buffer, 0, bytemuck::cast_slice(&decisions));
}

fn execute_raw(kernel: &GpuKernel, pipeline: &wgpu::ComputePipeline) -> TestResult<Vec<f32>> {
    assert!(kernel.layout.brain_scratch_stride >= ENCODED_DIMENSION);
    let words = usize::try_from(kernel.agent_count)
        .unwrap()
        .checked_mul(kernel.layout.brain_scratch_stride)
        .unwrap();
    let bytes = u64::try_from(words.checked_mul(WORD_BYTES).unwrap()).unwrap();
    kernel.queue.write_buffer(
        cache(kernel).buffer(),
        0,
        bytemuck::cast_slice(&vec![f32::NAN; words]),
    );
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(
            0,
            &kernel.global_credit.as_ref().unwrap().bind_groups[kernel.active_config_index],
            &[],
        );
        pass.set_push_constants(0, bytemuck::cast_slice(&[0_u32, 0_u32]));
        pass.dispatch_workgroups(kernel.agent_count, 1, 1);
    }
    kernel.queue.submit([encoder.finish()]);
    let bytes = read_buffer(kernel, cache(kernel).buffer(), bytes)?;
    let (words, remainder) = bytes.as_chunks::<WORD_BYTES>();
    assert!(remainder.is_empty());
    Ok(words.iter().map(|word| f32::from_le_bytes(*word)).collect())
}

#[test]
#[ignore = "GPU actual gathered-context helpers, guard boundaries and signed zero"]
fn gathered_context_coefficients_preserve_raw_guards() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let kernel = prepare(width, height, false);
        let source = reference_source(&kernel);
        let pipelines = [source.clone(), cached_context_source(&source)].map(|source| {
            pipeline(
                &kernel,
                format!("{source}\n{RAW_ENTRY}"),
                "gathered_context_probe",
            )
        });
        upload_raw_fixture(&kernel);
        let before = capture_state(&kernel)?;
        let expected = execute_raw(&kernel, &pipelines[0])?;
        let mut changed = 0;
        for replay in 0..REPLAYS {
            let actual = execute_raw(&kernel, &pipelines[1])?;
            assert_state_equal(&kernel, &before, &capture_state(&kernel)?);
            for agent in 0..RAW_CASES {
                let (count, similarities, _) = raw_case(agent);
                let total = similarities
                    .iter()
                    .take(count)
                    .fold(0.0_f32, |total, value| total + value.max(0.0));
                let skipped = count == 0 || total <= TOTAL_GUARD;
                for dim in 0..ENCODED_DIMENSION {
                    let index = agent * kernel.layout.brain_scratch_stride + dim;
                    assert!(actual[index].is_finite());
                    assert_eq!(
                        actual[index].to_bits(),
                        expected[index].to_bits(),
                        "context agent={agent} dim={dim} replay={replay}"
                    );
                    if skipped {
                        // Either input zero remains zero; the preceding exact
                        // comparison also detects a changed baseline zero sign.
                        assert_eq!(actual[index], 0.0);
                    } else if actual[index] != 0.0 {
                        changed += 1;
                    }
                }
            }
        }
        assert!(
            changed > 0,
            "raw fixtures must exercise a nonzero context contribution"
        );
        println!("GATHERED_CONTEXT_CACHE_RAW width={width} height={height} cases={RAW_CASES} replays={REPLAYS} outputs_per_case={ENCODED_DIMENSION} exact_bits=true zero_negative_threshold_partial_recall=true public_buffers_unchanged={MUTABLE_BUFFERS}");
    }
    Ok(())
}

/// Parse-only aligned prefix: no GPU storage is allocated by the CPU test.
fn cpu_source(subgroup: bool) -> String {
    let source = global_credit::main_source(&optimized_brain(), subgroup, None);
    let source = packed_encoder::packed_passes(&source);
    let source = replace_once(&source,
        "@group(0) @binding(13) var<storage, read_write> brain_scratch:       array<f32>;",
        "struct PackedEncoder { scratch: array<f32, 4>, weights: array<vec4<f32>>, }\n@group(0) @binding(13) var<storage, read_write> packed_encoder: PackedEncoder;");
    let source = format!("{source}\nconst PACKED_ENCODER_WIDTH: u32 = 4u;\nconst PACKED_ENCODER_OUTPUT_VECTORS: u32 = ENCODED_DIMENSION / PACKED_ENCODER_WIDTH;\nconst PACKED_ENCODER_PREFETCH: u32 = 2u;\nconst PACKED_ENCODER_INNER_LANES: u32 = 4u;\nconst PACKED_ENCODER_PARTIAL_WORDS: u32 = ENCODED_DIMENSION * PACKED_ENCODER_INNER_LANES;\n");
    main_width::try_transform(&source).unwrap()
}

#[test]
fn gathered_context_cache_requires_packed_scratch_and_preserves_barriers() {
    for subgroup in [false, true] {
        let source = cpu_source(subgroup);
        let candidate = cached_context_source(&source);
        assert_ne!(candidate, source);
        assert!(candidate.contains("let w = s_dense_partials[CONTEXT_COEFFICIENT_BASE + k];"));
        assert!(!candidate.contains(&original_helper()));
        for required in [
            PACKED_DECLARATION,
            "const BRAIN_WORKGROUP_SIZE: u32 = 128u;",
        ] {
            let missing = replace_once(&source, required, "");
            assert!(std::panic::catch_unwind(|| cached_context_source(&missing)).is_err());
        }
        assert!(std::panic::catch_unwind(|| cached_context_source(&candidate)).is_err());
        #[cfg(not(target_arch = "wasm32"))]
        for tested in [
            candidate.clone(),
            format!("{candidate}\n{RAW_ENTRY}"),
            format!("{source}\n{RAW_ENTRY}"),
        ] {
            let module = wgpu::naga::front::wgsl::parse_str(&tested)
                .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&tested)));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&tested)));
        }
    }
}
