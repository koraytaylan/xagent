//! Optional 128-thread main for the packed encoder and sixteen-lane predictor.
//! Claim, global credit/world, vision, and workgroup storage keep their original
//! shapes. Predictor/context tiles shrink, while reinforcement preserves both
//! ascending stride-two chains and cortex preserves its logical 256-lane tree.
//! Unsupported composed sources retain the ordinary main unchanged.

use std::ops::Range;

/// One invocation owns each of the 128 encoded outputs and memory patterns.
pub(super) const MAIN_THREADS: u32 = 128;

const ORIGINAL_MAIN: &str = "@compute @workgroup_size(256)\nfn kernel_tick(";
const COMPACT_MAIN: &str = "@compute @workgroup_size(128)\nfn kernel_tick(";
const CLAIM_ENTRY: &str = "@compute @workgroup_size(256)\nfn kernel_claim_tick(";
const ORIGINAL_GROUP: &str = "const BRAIN_WORKGROUP_SIZE: u32 = 256u;";
const COMPACT_GROUP: &str =
    "const BRAIN_WORKGROUP_SIZE: u32 = 128u;\nconst MAIN_LOGICAL_WORKGROUP_SIZE: u32 = 256u;";
const ORIGINAL_CORTEX_TREE: &str = "var stride: u32 = BRAIN_WORKGROUP_SIZE / 2u;";
const LOGICAL_CORTEX_TREE: &str = "var stride: u32 = MAIN_LOGICAL_WORKGROUP_SIZE / 2u;";
const PREDICTOR_BEGIN: &str = "fn coop_predict_and_act(";
const PREDICTOR_END: &str = "    // ── Recalled cosine similarities:";
const ORIGINAL_PREDICTOR_TILE: &str =
    "for (var tile = 0u; tile < PREDICTOR_DIMENSION; tile += 16u)";
const COMPACT_PREDICTOR_TILE: &str = "for (var tile = 0u; tile < PREDICTOR_DIMENSION; tile += 8u)";
const PACKED_ENCODER: &str = include_str!("../shaders/kernel/brain_packed_encoder.wgsl");

const ORIGINAL_REINFORCEMENT: &str = r"    let pattern = tid % MEMORY_CAP;
    let lane = tid / MEMORY_CAP;

    // All 256 threads compute their partial dot product over their stride-2 dimension range
    {
        var dot: f32 = 0.0;
        for (var d = lane; d < ENCODED_DIMENSION; d += 2u) {
            dot += s_memory_key[d] * pattern_buffer[pattern_base + d * MEMORY_CAP + pattern];
        }
        s_reinf_dot[tid] = dot;
    }
";

const PAIRED_REINFORCEMENT: &str = r"    let pattern = tid;

    // Each physical thread preserves both ascending stride-two FP32 chains.
    // The existing barrier and sum combine the same even and odd partials.
    {
        var even_dot: f32 = 0.0;
        var odd_dot: f32 = 0.0;
        for (var d = 0u; d < ENCODED_DIMENSION; d += 2u) {
            even_dot += s_memory_key[d] * pattern_buffer[pattern_base + d * MEMORY_CAP + pattern];
            odd_dot += s_memory_key[d + 1u] * pattern_buffer[pattern_base + (d + 1u) * MEMORY_CAP + pattern];
        }
        s_reinf_dot[tid] = even_dot;
        s_reinf_dot[tid + MEMORY_CAP] = odd_dot;
    }
";

const CORTEX_PARTIALS: &str = r"    if (tid < VISUAL_FEATURE_COUNT) {
        let v = s_visual[VC_COMPLEX_BASE + tid];
        s_dense_partials[tid] = v * v;
    } else {
        s_dense_partials[tid] = 0.0;
    }
";

/// Every shape assumption has an explicit source guard before any mutation.
const REQUIRED: &[&str] = &[
    ORIGINAL_MAIN,
    CLAIM_ENTRY,
    ORIGINAL_GROUP,
    ORIGINAL_CORTEX_TREE,
    ORIGINAL_REINFORCEMENT,
    CORTEX_PARTIALS,
    PACKED_ENCODER,
    PREDICTOR_BEGIN,
    PREDICTOR_END,
    "const ENCODED_DIMENSION: u32 = 128u;",
    "const MEMORY_CAP: u32 = 128u;",
    "const DENSE_INNER_LANES: u32 = 4u;",
    "const CONTEXT_GATHER_LANES: u32 = 8u;",
    "const CONTEXT_GATHER_OUTPUT_TILE: u32 = BRAIN_WORKGROUP_SIZE / CONTEXT_GATHER_LANES;",
    "const PACKED_ENCODER_WIDTH: u32 = 4u;",
    "const PACKED_ENCODER_OUTPUT_VECTORS: u32 = ENCODED_DIMENSION / PACKED_ENCODER_WIDTH;",
    "const PACKED_ENCODER_PREFETCH: u32 = 2u;",
    "const PACKED_ENCODER_INNER_LANES: u32 = 4u;",
    "const PACKED_ENCODER_PARTIAL_WORDS: u32 = ENCODED_DIMENSION * PACKED_ENCODER_INNER_LANES;",
    "var<workgroup> s_dense_partials: array<f32, PACKED_ENCODER_PARTIAL_WORDS>;",
    "var<workgroup> s_reinf_dot: array<f32, 256>;",
    "let dot_val = s_reinf_dot[tid] + s_reinf_dot[tid + MEMORY_CAP];",
];

fn predictor_range(source: &str) -> Option<Range<usize>> {
    // Cortex outputs must all belong to the lower physical half. Otherwise
    // initializing the upper logical half with zero would discard real terms.
    if crate::complex::VISUAL_FEATURE_COUNT > usize::try_from(MAIN_THREADS).ok()? {
        return None;
    }
    if REQUIRED
        .iter()
        .any(|required| source.matches(*required).count() != 1)
    {
        return None;
    }
    let start = source.find(PREDICTOR_BEGIN)?;
    let end = start.checked_add(source[start..].find(PREDICTOR_END)?)?;
    let predictor = &source[start..end];
    for required in [
        "let output_in_tile = tid / 16u;",
        "let lane = tid % 16u;",
        "for (var stride = 16u / 2u; stride > 0u; stride /= 2u)",
        ORIGINAL_PREDICTOR_TILE,
    ] {
        if predictor.matches(required).count() != 1 {
            return None;
        }
    }
    Some(start..end)
}

fn replace_once(source: &str, old: &str, new: &str) -> String {
    assert_eq!(
        source.matches(old).count(),
        1,
        "checked source target: {old}"
    );
    source.replacen(old, new, 1)
}

fn workgroup_declarations(source: &str) -> Vec<&str> {
    source
        .lines()
        .filter(|line| line.starts_with("var<workgroup>"))
        .collect()
}

/// Reduce only a supported packed main's physical invocation count. The caller
/// keeps the original source when this returns `None`; no device-specific
/// behavior or optional brain feature is enabled implicitly.
///
/// All arithmetic reductions keep their original order. The danger scan uses
/// the smaller group stride and retains its distance/cell-index tie breaker.
/// Shared storage is unchanged, so the packed cache's existing device-limit
/// check remains applicable. This must only compile the `kernel_tick` entry.
///
/// # Panics
///
/// Panics if an internal rewrite violates an already-checked target or changes
/// workgroup declarations or the independent claim entry.
pub(super) fn try_transform(source: &str) -> Option<String> {
    let predictor = predictor_range(source)?;
    let changed = replace_once(
        &source[predictor.clone()],
        ORIGINAL_PREDICTOR_TILE,
        COMPACT_PREDICTOR_TILE,
    );
    let mut candidate = source.to_owned();
    candidate.replace_range(predictor, &changed);
    for (original, replacement) in [
        (ORIGINAL_GROUP, COMPACT_GROUP),
        (ORIGINAL_MAIN, COMPACT_MAIN),
        (ORIGINAL_REINFORCEMENT, PAIRED_REINFORCEMENT),
        (ORIGINAL_CORTEX_TREE, LOGICAL_CORTEX_TREE),
    ] {
        candidate = replace_once(&candidate, original, replacement);
    }
    // All 256 logical partials remain initialized even with 128 physical
    // invocations, preserving the cortex's leading positive-zero additions.
    candidate = replace_once(
        &candidate,
        CORTEX_PARTIALS,
        &format!("{CORTEX_PARTIALS}    s_dense_partials[tid + BRAIN_WORKGROUP_SIZE] = 0.0;\n"),
    );
    assert_eq!(
        workgroup_declarations(&candidate),
        workgroup_declarations(source)
    );
    assert_eq!(candidate.matches(CLAIM_ENTRY).count(), 1);
    assert_eq!(
        candidate.matches("tick * 256u").count(),
        source.matches("tick * 256u").count()
    );
    Some(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_kernel::{
        compose_brain_passes, context_gather, dense_prefetch, global_credit, packed_encoder,
        predictor_fusion, predictor_width,
    };

    /// Use the production dense prefetch and valid/invalid reduction shapes.
    const PREFETCH_FACTOR: u32 = 8;
    const PREDICTOR_LANES: u32 = 16;
    const CONTEXT_LANES: u32 = 8;
    const OTHER_PREDICTOR_LANES: [u32; 3] = [4, 8, 32];
    const OTHER_CONTEXT_LANES: u32 = 16;

    // Only type/source validation uses this fixed aligned prefix. No buffer
    // is allocated and no runtime index is exercised by these CPU tests.
    const PACKED_HEADER: &str = r"
const PACKED_ENCODER_WIDTH: u32 = 4u;
const PACKED_ENCODER_OUTPUT_VECTORS: u32 = ENCODED_DIMENSION / PACKED_ENCODER_WIDTH;
const PACKED_ENCODER_PREFETCH: u32 = 2u;
const PACKED_ENCODER_INNER_LANES: u32 = 4u;
const PACKED_ENCODER_PARTIAL_WORDS: u32 = ENCODED_DIMENSION * PACKED_ENCODER_INNER_LANES;
";
    const PACKED_BINDING: &str = r"struct PackedEncoder {
    scratch: array<f32, 4>,
    weights: array<vec4<f32>>,
}
@group(0) @binding(13) var<storage, read_write> packed_encoder: PackedEncoder;";

    fn fixture(packed: bool, lanes: u32, context: Option<u32>, cooperative: bool) -> String {
        let fused = predictor_fusion::fuse_inline_predictor(&compose_brain_passes(cooperative));
        let prefetched = dense_prefetch::prefetch_passes(&fused, PREFETCH_FACTOR);
        let mut passes = predictor_width::wider_predictor(&prefetched, lanes);
        if let Some(context) = context {
            passes = context_gather::gather_context(&passes, context);
        }
        let mut source = global_credit::main_source(&passes, false, None);
        if packed {
            source = packed_encoder::packed_passes(&source);
            source = replace_once(
                &source,
                "@group(0) @binding(13) var<storage, read_write> brain_scratch:       array<f32>;",
                PACKED_BINDING,
            );
            source.push_str(PACKED_HEADER);
        }
        source
    }

    #[test]
    fn main128_supported_shapes_preserve_entries_storage_and_reductions() {
        for cooperative in [false, true] {
            let source = fixture(true, PREDICTOR_LANES, Some(CONTEXT_LANES), cooperative);
            let candidate = try_transform(&source).unwrap();
            assert!(candidate.contains(&format!(
                "@compute @workgroup_size({MAIN_THREADS})\nfn kernel_tick("
            )));
            assert!(candidate.contains(CLAIM_ENTRY));
            assert!(candidate.contains("for (var f = tid; f < food_count; f += 256u)"));
            assert!(candidate.contains("s_reinf_dot[tid + MEMORY_CAP] = odd_dot;"));
            assert!(candidate.contains("var stride: u32 = ENCODED_DIMENSION / 2u;"));
            assert!(candidate.contains(LOGICAL_CORTEX_TREE));
            assert!(candidate.contains("s_dense_partials[tid + BRAIN_WORKGROUP_SIZE] = 0.0;"));
            assert_eq!(
                candidate.matches("workgroupBarrier();").count(),
                source.matches("workgroupBarrier();").count()
            );
            assert_eq!(
                candidate.matches("storageBarrier();").count(),
                source.matches("storageBarrier();").count()
            );
            assert!(
                try_transform(&candidate).is_none(),
                "repeated transformation must fall back"
            );
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
    fn main128_unsupported_optional_configurations_fall_back() {
        assert!(
            try_transform(&fixture(false, PREDICTOR_LANES, Some(CONTEXT_LANES), true)).is_none()
        );
        for lanes in OTHER_PREDICTOR_LANES {
            assert!(try_transform(&fixture(true, lanes, Some(CONTEXT_LANES), true)).is_none());
        }
        for context in [None, Some(OTHER_CONTEXT_LANES)] {
            assert!(try_transform(&fixture(true, PREDICTOR_LANES, context, true)).is_none());
        }
    }

    #[test]
    fn main128_missing_or_duplicate_shape_contracts_fall_back() {
        let source = fixture(true, PREDICTOR_LANES, Some(CONTEXT_LANES), true);
        for &required in REQUIRED {
            assert!(
                try_transform(&source.replacen(required, "", 1)).is_none(),
                "missing {required}"
            );
            assert!(
                try_transform(&format!("{source}\n{required}")).is_none(),
                "duplicate {required}"
            );
        }
        for required in [
            "let output_in_tile = tid / 16u;",
            "let lane = tid % 16u;",
            "for (var stride = 16u / 2u; stride > 0u; stride /= 2u)",
            ORIGINAL_PREDICTOR_TILE,
        ] {
            assert!(
                try_transform(&source.replacen(required, "", 1)).is_none(),
                "missing predictor contract {required}"
            );
        }
    }
}
