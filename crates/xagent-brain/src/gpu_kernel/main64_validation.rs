//! Raw-vision-only 64-thread main, compared with the production 128-thread main.
//! Each barrier-free 128-owner block visits two logical owners before the next
//! original barrier. Tree reductions retain their 128 inputs and original
//! addition order; no barrier-containing function is called twice per phase.
//! The packed encoder retains all four feature lanes, and the subgroup sorter
//! processes two independent halves before its shared-memory merge stages.
//!
//! Claim, global credit, vision, packed storage and recording remain production
//! implementations. Cortex is explicitly unsupported by this experiment. No
//! workgroup arrays, CPU per-cycle computation or production routes are added.

use std::{collections::HashMap, error::Error, ops::Range, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::packed_store_validation::{
    advance, assert_mirror, cache, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// Two logical output/pattern owners share one physical invocation.
const MAIN_THREADS: u32 = 64;
const LOGICAL_THREADS: u32 = 128;
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const REPLAYS: usize = 2;
const WARMUP_CYCLES: u32 = 1_000;
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;
const MUTABLE_BUFFERS: usize = 13;
const INACTIVE_AGENT: u32 = 1;
const EXPECTED_DEATHS: f32 = 2.0;
const PUSH_CONSTANT_BYTES: u32 = 8;

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

/// Keep byte offsets while ignoring braces and tokens inside WGSL comments.
fn code_mask(source: &str) -> String {
    let mut bytes = source.as_bytes().to_vec();
    let mut cursor = 0;
    let mut block_depth = 0u32;
    let mut line_comment = false;
    while cursor < bytes.len() {
        let pair = source.as_bytes().get(cursor..cursor + 2);
        if line_comment {
            if bytes[cursor] == b'\n' {
                line_comment = false;
            } else {
                bytes[cursor] = b' ';
            }
        } else if pair == Some(b"/*") {
            block_depth += 1;
            bytes[cursor..cursor + 2].fill(b' ');
            cursor += 1;
        } else if block_depth != 0 && pair == Some(b"*/") {
            block_depth -= 1;
            bytes[cursor..cursor + 2].fill(b' ');
            cursor += 1;
        } else if block_depth != 0 {
            bytes[cursor] = b' ';
        } else if pair == Some(b"//") {
            line_comment = true;
            bytes[cursor..cursor + 2].fill(b' ');
            cursor += 1;
        }
        cursor += 1;
    }
    assert_eq!(block_depth, 0);
    String::from_utf8(bytes).unwrap()
}

fn matching_brace(mask: &str, open: usize) -> usize {
    assert_eq!(mask.as_bytes()[open], b'{');
    let mut depth = 0u32;
    for (offset, byte) in mask.as_bytes()[open..].iter().enumerate() {
        if *byte == b'{' {
            depth += 1;
        } else if *byte == b'}' {
            depth -= 1;
            if depth == 0 {
                return open + offset + 1;
            }
        }
    }
    panic!("unclosed WGSL block");
}

fn function_range(source: &str, name: &str) -> Range<usize> {
    let marker = format!("fn {name}(");
    assert_eq!(source.matches(&marker).count(), 1);
    let start = source.find(&marker).unwrap();
    let open = start + source[start..].find('{').unwrap();
    start..matching_brace(&code_mask(source), open)
}

fn identifier_replace(source: &str, old: &str, new: &str) -> String {
    let identifier_byte = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    let mut result = String::new();
    let mut consumed = 0;
    for (start, _) in source.match_indices(old) {
        let end = start + old.len();
        if start > 0 && identifier_byte(source.as_bytes()[start - 1])
            || end < source.len() && identifier_byte(source.as_bytes()[end])
        {
            continue;
        }
        result.push_str(&source[consumed..start]);
        result.push_str(new);
        consumed = end;
    }
    result.push_str(&source[consumed..]);
    result
}

/// Both owners finish before any existing cooperative barrier. Reject calls
/// with barriers and nonlocal exits instead of trying to virtualize them.
fn paired_block(source: &str) -> String {
    let code = code_mask(source);
    for forbidden in ["Barrier(", "wg_reduce_dense(", "cooperative_", "return;"] {
        assert!(
            !code.contains(forbidden),
            "paired owner block contains {forbidden}"
        );
    }
    let paired = identifier_replace(source, "tid", "main64_owner");
    format!(
        "for (var main64_base = 0u; main64_base < MAIN64_LOGICAL_THREADS; main64_base += BRAIN_WORKGROUP_SIZE) {{\n    let main64_owner = tid + main64_base;\n{paired}\n}}"
    )
}

/// Include a directly attached else branch, so disabled feature paths still
/// initialize both logical halves rather than leaving upper scratch stale.
fn if_range(source: &str, start: usize) -> Range<usize> {
    let mask = code_mask(source);
    let open = start + mask[start..].find('{').unwrap();
    let mut end = matching_brace(&mask, open);
    let suffix = mask[end..].trim_start();
    if suffix.starts_with("else") {
        assert!(
            suffix.starts_with("else {"),
            "owner guard has an unsupported else-if"
        );
        let next = end + mask[end..].find("else").unwrap();
        let open = next + mask[next..].find('{').unwrap();
        end = matching_brace(&mask, open);
    }
    start..end
}

fn pair_owner_guards(source: &str) -> String {
    let mut candidate = source.to_owned();
    for guard in [
        "if (homeo_pred_enabled && tid < ENCODED_DIMENSION)",
        "if (tid < ENCODED_DIMENSION)",
        "if (tid < PREDICTOR_DIMENSION)",
        "if (tid < MEMORY_CAP)",
        "if (tid < RECENT_CAP)",
    ] {
        while let Some(start) = code_mask(&candidate).find(guard) {
            let range = if_range(&candidate, start);
            let mut block = candidate[range.clone()].to_owned();
            if block.contains("let dot_val = s_reinf_dot[tid] + s_reinf_dot[tid + MEMORY_CAP];") {
                // Reinforcement's pattern index must follow this block's
                // logical owner, not the physical invocation's lower pattern.
                block = identifier_replace(&block, "pattern", "tid");
            }
            candidate.replace_range(range, &paired_block(&block));
        }
    }
    candidate
}

fn pair_encoder(source: &str) -> String {
    let range = function_range(source, "coop_encode");
    let original = &source[range.clone()];
    let start = original
        .find("    let output_vector = tid % PACKED_ENCODER_OUTPUT_VECTORS;")
        .unwrap();
    let end = original.find("    workgroupBarrier();").unwrap();
    let arithmetic = &original[start..end];
    assert!(arithmetic.contains("let slot = tid * PACKED_ENCODER_WIDTH;"));
    let paired = format!(
        "{}\n    let output_vector = tid % PACKED_ENCODER_OUTPUT_VECTORS;\n    let lane = tid / PACKED_ENCODER_OUTPUT_VECTORS;\n",
        paired_block(arithmetic),
    );
    let mut changed = original.to_owned();
    changed.replace_range(start..end, &paired);
    let mut candidate = source.to_owned();
    candidate.replace_range(range, &changed);
    candidate
}

fn pair_topk(source: &str) -> String {
    let range = function_range(source, "coop_recall_topk");
    let mut changed = source[range.clone()].to_owned();
    if changed.contains("subgroupShuffle(") {
        let start = changed.find("    var my_val: f32 = -3.0;").unwrap();
        let end = start + changed[start..].find("    workgroupBarrier();").unwrap();
        let block = &changed[start..end];
        assert!(block.contains("stage < 5u"));
        assert!(!block.contains("stage < 7u"));
        // Stages0–4 never exchange across a32-value region. Processing each
        // logical64-value half retains the same sgid partners and tie rules.
        let paired = paired_block(block);
        changed.replace_range(start..end, &paired);
    }
    changed = pair_owner_guards(&changed);
    let mut candidate = source.to_owned();
    candidate.replace_range(range, &changed);
    candidate
}

fn pair_predictor(source: &str) -> String {
    let range = function_range(source, "coop_predict_and_act");
    let mut changed = source[range.clone()].to_owned();
    changed = replace_once(
        &changed,
        "for (var tile = 0u; tile < PREDICTOR_DIMENSION; tile += 8u)",
        "for (var tile = 0u; tile < PREDICTOR_DIMENSION; tile += 4u)",
    );
    let start = changed
        .find("    // Sub-step 1: parallel dot product")
        .unwrap();
    let end = start + changed[start..].find("    workgroupBarrier();").unwrap();
    let guarded = &changed[start..end];
    assert!(guarded.contains("if (homeo_pred_enabled)"));
    let paired = paired_block(guarded);
    changed.replace_range(start..end, &paired);
    changed = pair_owner_guards(&changed);
    let mut candidate = source.to_owned();
    candidate.replace_range(range, &changed);
    candidate
}

fn pair_learning(source: &str) -> String {
    let range = function_range(source, "coop_learn_and_store");
    let mut changed = source[range.clone()].to_owned();
    let start = changed.find("    let pattern = tid;\n").unwrap();
    let end = start + changed[start..].find("    workgroupBarrier();").unwrap();
    let arithmetic = &changed[start..end];
    assert!(arithmetic.contains("s_reinf_dot[tid + MEMORY_CAP] = odd_dot;"));
    let paired = paired_block(arithmetic);
    changed.replace_range(start..end, &paired);
    changed = pair_owner_guards(&changed);
    let mut candidate = source.to_owned();
    candidate.replace_range(range, &changed);
    candidate
}

/// Input is the production128 main. This experiment refuses cortex layouts;
/// only raw vision's feature loop and the fixed128 neural dimensions qualify.
fn main64_source(source: &str, cortex: bool) -> Option<String> {
    if cortex {
        return None;
    }
    assert!(source.contains("const ENCODED_DIMENSION: u32 = 128u;"));
    assert!(source.contains("const MEMORY_CAP: u32 = 128u;"));
    assert!(source.contains("const RECENT_CAP: u32 = 128u;"));
    assert!(source.contains("const CONTEXT_GATHER_LANES: u32 = 8u;"));
    let mut candidate = replace_once(
        source,
        "const BRAIN_WORKGROUP_SIZE: u32 = 128u;",
        "const BRAIN_WORKGROUP_SIZE: u32 = 64u;\nconst MAIN64_LOGICAL_THREADS: u32 = 128u;",
    );
    candidate = replace_once(
        &candidate,
        "@compute @workgroup_size(128)\nfn kernel_tick(",
        "@compute @workgroup_size(64)\nfn kernel_tick(",
    );
    candidate = pair_encoder(&candidate);
    candidate = pair_topk(&candidate);
    candidate = pair_predictor(&candidate);
    candidate = pair_learning(&candidate);
    for function in ["coop_habituate_homeo", "coop_recall_score"] {
        let range = function_range(&candidate, function);
        let changed = pair_owner_guards(&candidate[range.clone()]);
        candidate.replace_range(range, &changed);
    }
    // The physical64 scan visits all cells at stride64. Only initialized
    // lower slots participate in its exact distance/cell-index minimum.
    let range = function_range(&candidate, "agent_danger_detect");
    let changed = replace_once(
        &candidate[range.clone()],
        "for (var i = 0u; i < MEMORY_CAP; i++)",
        "for (var i = 0u; i < BRAIN_WORKGROUP_SIZE; i++)",
    );
    candidate.replace_range(range, &changed);
    let declarations = |text: &str| {
        text.lines()
            .filter(|line| line.starts_with("var<workgroup>"))
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    assert_eq!(declarations(&candidate), declarations(source));
    assert_eq!(
        candidate.matches("workgroupBarrier();").count(),
        source.matches("workgroupBarrier();").count()
    );
    assert_eq!(
        candidate.matches("storageBarrier();").count(),
        source.matches("storageBarrier();").count()
    );
    Some(candidate)
}

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
    assert_eq!(
        kernel.global_credit.as_ref().unwrap().main_threads,
        LOGICAL_THREADS
    );
    assert!(kernel.global_credit_active());
    assert!(!kernel.layout.visual_cortex_enabled);
    kernel
}

fn candidate_pipeline(kernel: &GpuKernel) -> wgpu::ComputePipeline {
    let source =
        global_credit::main_source(&optimized_brain(), kernel.has_subgroup, Some(cache(kernel)));
    let source = main_width::try_transform(&source).unwrap();
    let source = main64_source(&source, kernel.layout.visual_cortex_enabled).unwrap();
    eprintln!(
        "MAIN64_COMPILE_BEGIN width={} height={} subgroup={}",
        kernel.layout.vision_width, kernel.layout.vision_height, kernel.has_subgroup
    );
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("main64_probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main64_probe"),
            bind_group_layouts: &[&binding],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    let pipeline = kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("main64_probe"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("kernel_tick"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants(kernel),
                ..Default::default()
            },
            cache: None,
        });
    eprintln!("MAIN64_COMPILE_END");
    pipeline
}

struct Arms {
    parked: wgpu::ComputePipeline,
    candidate: bool,
}

impl Arms {
    fn new(kernel: &GpuKernel) -> Self {
        Self {
            parked: candidate_pipeline(kernel),
            candidate: false,
        }
    }

    fn activate(&mut self, kernel: &mut GpuKernel, candidate: bool) {
        if candidate != self.candidate {
            std::mem::swap(
                &mut kernel.global_credit.as_mut().unwrap().main,
                &mut self.parked,
            );
            self.candidate = candidate;
        }
        assert!(kernel.global_credit_active());
        // Both pipeline handles share the same cache. Restoring public state
        // invalidates it before the production recorder's existing import.
        kernel.global_credit.as_mut().unwrap().main_threads = if candidate {
            MAIN_THREADS
        } else {
            LOGICAL_THREADS
        };
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
#[ignore = "requires a GPU; complete raw state and packed mirror parity"]
fn main64_preserves_raw_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare(width, height, true);
        let mut arms = Arms::new(&kernel);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let expected = trajectory(&mut kernel)?;
        for replay in 0..REPLAYS {
            restore(&mut kernel, &saved);
            arms.activate(&mut kernel, true);
            let actual = trajectory(&mut kernel)?;
            for (expected, actual) in expected.iter().zip(&actual) {
                assert_state_equal(&kernel, expected, actual);
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial,
                    actual,
                    INACTIVE_AGENT,
                    "main64",
                );
            }
            println!("MAIN64_PARITY width={width} height={height} cycles=100 replay={replay} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true death_refresh=true cortex=false reference_threads={LOGICAL_THREADS} candidate_threads={MAIN_THREADS}");
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU full-cycle benchmark; run explicitly in release mode"]
fn benchmark_main64_complete_cycles() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let mut kernel = prepare(width, height, false);
    let mut arms = Arms::new(&kernel);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let saved = checkpoint(&kernel);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let expected = capture_state(&kernel)?;
    assert_mirror(&kernel, &expected)?;
    restore(&mut kernel, &saved);
    arms.activate(&mut kernel, true);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let preflight = capture_state(&kernel)?;
    assert_state_equal(&kernel, &expected, &preflight);
    assert_mirror(&kernel, &preflight)?;
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
    for values in &mut timings {
        values.sort_by(f64::total_cmp);
    }
    let reference = timings[0][TIMING_PAIRS / 2];
    let candidate = timings[1][TIMING_PAIRS / 2];
    println!("MAIN64_TIMING warmup_cycles={WARMUP_CYCLES} cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true full_simulation=true same_production_recorder=true cold_import_timed=true state_comparison_timed=false shared_storage_unchanged=true reference_threads={LOGICAL_THREADS} candidate_threads={MAIN_THREADS}", reference/candidate);
    Ok(())
}

/// A type-valid CPU-only storage prefix; no buffer is allocated or dispatched.
const CPU_PACKED_BINDING: &str = r"struct PackedEncoder {
    scratch: array<f32, 4>,
    weights: array<vec4<f32>>,
}
@group(0) @binding(13) var<storage, read_write> packed_encoder: PackedEncoder;";
const CPU_PACKED_CONSTANTS: &str = r"
const PACKED_ENCODER_WIDTH: u32 = 4u;
const PACKED_ENCODER_OUTPUT_VECTORS: u32 = ENCODED_DIMENSION / PACKED_ENCODER_WIDTH;
const PACKED_ENCODER_PREFETCH: u32 = 2u;
const PACKED_ENCODER_INNER_LANES: u32 = 4u;
const PACKED_ENCODER_PARTIAL_WORDS: u32 = ENCODED_DIMENSION * PACKED_ENCODER_INNER_LANES;
";

fn cpu_source(subgroup: bool) -> String {
    let source = global_credit::main_source(&optimized_brain(), subgroup, None);
    let source = packed_encoder::packed_passes(&source);
    let mut source = replace_once(
        &source,
        "@group(0) @binding(13) var<storage, read_write> brain_scratch:       array<f32>;",
        CPU_PACKED_BINDING,
    );
    source.push_str(CPU_PACKED_CONSTANTS);
    main_width::try_transform(&source).unwrap()
}

#[test]
fn main64_composition_preserves_barrier_phases_and_refuses_cortex() {
    for subgroup in [false, true] {
        let original = cpu_source(subgroup);
        assert!(main64_source(&original, true).is_none());
        let candidate = main64_source(&original, false).unwrap();
        assert!(candidate.contains("@compute @workgroup_size(64)\nfn kernel_tick("));
        assert!(candidate.contains("@compute @workgroup_size(256)\nfn kernel_claim_tick("));
        assert!(candidate.contains("var stride: u32 = ENCODED_DIMENSION / 2u;"));
        assert!(candidate.contains("for (var tile = 0u; tile < PREDICTOR_DIMENSION; tile += 4u)"));
        for function in [
            "coop_habituate_homeo",
            "coop_recall_score",
            "coop_recall_topk",
            "coop_predict_and_act",
            "coop_learn_and_store",
        ] {
            let body = &candidate[function_range(&candidate, function)];
            for unpaired in [
                "if (tid < ENCODED_DIMENSION)",
                "if (tid < PREDICTOR_DIMENSION)",
                "if (tid < MEMORY_CAP)",
                "if (tid < RECENT_CAP)",
            ] {
                assert!(
                    !code_mask(body).contains(unpaired),
                    "unpaired owner in {function}: {unpaired}"
                );
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let module = wgpu::naga::front::wgsl::parse_str(&candidate)
                .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&candidate)));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&candidate)));
        }
    }
}

#[test]
fn main64_owner_mapping_covers_every_logical_partial_once() {
    /// The encoder packs four adjacent output components per logical owner.
    const COMPONENTS: usize = 4;
    /// Reinforcement retains one even and one odd chain per memory pattern.
    const REINFORCEMENT_LANES: usize = 2;
    let logical = usize::try_from(LOGICAL_THREADS).unwrap();
    let physical = usize::try_from(MAIN_THREADS).unwrap();
    let mut encoder = vec![0u32; logical * COMPONENTS];
    let mut reinforcement = vec![0u32; logical * REINFORCEMENT_LANES];
    for tid in 0..physical {
        for owner in [tid, tid + physical] {
            for component in 0..COMPONENTS {
                encoder[owner * COMPONENTS + component] += 1;
            }
            for lane in 0..REINFORCEMENT_LANES {
                reinforcement[owner + lane * logical] += 1;
            }
        }
    }
    assert!(encoder.iter().all(|count| *count == 1));
    assert!(reinforcement.iter().all(|count| *count == 1));
    assert!(std::panic::catch_unwind(|| paired_block("wg_reduce_dense(tid);")).is_err());
    assert!(std::panic::catch_unwind(|| paired_block("workgroupBarrier();")).is_err());
}
