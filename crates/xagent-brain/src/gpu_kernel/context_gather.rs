//! Ordered context blending with parallel loads of recalled pattern values.
//! Eight- or sixteen-invocation teams stage raw values in existing scratch;
//! one invocation retains each output's original FP32 accumulation order.
//! Scratch is released before whitening and memory reinforcement consume it.

/// Both team sizes fit all recalled values in existing workgroup arrays.
const SUPPORTED_LANES: [u32; 2] = [8, 16];
/// Specialize the shared WGSL fragment without changing reduction arithmetic.
const LANE_DECLARATION: &str = "const CONTEXT_GATHER_LANES: u32 = RECALL_K;";

/// Keep the original arithmetic and its guards visible in the rewrite contract.
const CONTEXT_BLOCK: &str = r"    // ── Per-dimension context blend and tanh: threads 0..PREDICTOR_DIMENSION ──
    if (tid < PREDICTOR_DIMENSION) {
        // Context blend contribution (per-dimension)
        if (recall_count > 0u) {
            let context_weight = brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT];
            var total_sim: f32 = 0.0;
            for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                total_sim += max(s_recall_similarity[k], 0.0);
            }
            if (total_sim > 1e-8) {
                // Stored keys are centered; the running mean turns them back
                // into encoded states for the forward model.
                let encoded_mean = brain_state[brain_base + O_ENCODED_MEAN + tid];
                for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                    let idx = u32(s_recall[k]);
                    let w = context_weight * max(s_recall_similarity[k], 0.0) / total_sim;
                    s_prediction[tid] +=
                        (pattern_buffer[pattern_base + tid * MEMORY_CAP + idx] + encoded_mean) * w;
                }
            }
        }

        // Apply tanh to prediction
        s_prediction[tid] = fast_tanh(s_prediction[tid]);
    }
    workgroupBarrier();
";

/// Guard the scratch lifetime against changes to the composed brain passes.
fn assert_scratch_available(source: &str) {
    assert!(source.contains("var<workgroup> s_dense_partials: array<f32, BRAIN_WORKGROUP_SIZE>;"));
    assert!(source.contains("var<workgroup> s_reinf_dot: array<f32, 256>;"));
    let predict_start = source.find("fn coop_predict_and_act(").unwrap();
    let learn_start = source.find("fn coop_learn_and_store(").unwrap();
    let context_start = source.find(CONTEXT_BLOCK).unwrap();
    assert!(predict_start < context_start && context_start < learn_start);
    assert!(!source[predict_start..learn_start].contains("s_reinf_dot["));
    if let Some(whitening_start) =
        source.find("    cooperative_refresh_vision_whitening(brain_base, tid);")
    {
        // The helper initializes both matrix regions before reading them.
        assert!(context_start < whitening_start && whitening_start < learn_start);
    }
    // Reinforcement initializes every entry even when whitening was skipped.
    assert!(source.contains("        s_reinf_dot[tid] = dot;\n    }\n    workgroupBarrier();"));
}

/// Stage raw recalled values while preserving scalar normalization and blend.
/// The caller selects whether this transformation is enabled; accepted team
/// sizes use one or two existing scratch arrays without allocating resources.
pub(super) fn gather_context(source: &str, lanes: u32) -> String {
    assert!(SUPPORTED_LANES.contains(&lanes));
    assert_eq!(source.matches(CONTEXT_BLOCK).count(), 1);
    assert_scratch_available(source);
    let helper = include_str!("../shaders/kernel/context_gather.wgsl");
    assert_eq!(helper.matches(LANE_DECLARATION).count(), 1);
    let helper = helper.replacen(
        LANE_DECLARATION,
        &format!("const CONTEXT_GATHER_LANES: u32 = {lanes}u;"),
        1,
    );
    let mut candidate = source.replacen(
        CONTEXT_BLOCK,
        "    blend_gathered_recalled_context(brain_base, pattern_base, recall_count, tid);\n",
        1,
    );
    candidate.push('\n');
    candidate.push_str(&helper);
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        source.matches("var<workgroup>").count(),
        "the gather must reuse existing workgroup storage"
    );
    candidate
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_kernel::{
        compose_brain_passes, dense_prefetch, predictor_fusion::fuse_inline_predictor,
        predictor_width,
    };

    /// Exercise unsupported disabling, undersized and oversized gather teams.
    const UNSUPPORTED_LANES: [u32; 5] = [0, 1, 4, 32, 256];
    /// Use the complete production prefetch and predictor configuration.
    const PREFETCH_FACTOR: u32 = 8;
    const PREDICTOR_LANES: u32 = 16;
    /// Removing either declaration must fail before shader compilation.
    const SCRATCH_DECLARATIONS: [&str; 2] = [
        "var<workgroup> s_dense_partials: array<f32, BRAIN_WORKGROUP_SIZE>;",
        "var<workgroup> s_reinf_dot: array<f32, 256>;",
    ];

    fn assert_rejected(source: &str, lanes: u32) {
        assert!(
            std::panic::catch_unwind(|| gather_context(source, lanes)).is_err(),
            "unsupported source or lane count must fail before compilation"
        );
    }

    // Native wgpu exposes the same parser and validator used for shader
    // creation. Browser-only wgpu builds do not expose this native dependency.
    #[cfg(not(target_arch = "wasm32"))]
    fn validate_brain_source(passes: &str) {
        let source = crate::gpu_kernel::apply_subgroup_markers(
            &[
                include_str!("../shaders/kernel/common.wgsl"),
                passes,
                include_str!("../shaders/kernel/brain_tick.wgsl"),
            ]
            .join("\n"),
            false,
        );
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    }

    #[test]
    fn context_gather_composes_with_whitening_and_dense_options() {
        for whitening in [false, true] {
            let original = compose_brain_passes(whitening);
            let fused = fuse_inline_predictor(&original);
            let prefetched = dense_prefetch::prefetch_passes(&fused, PREFETCH_FACTOR);
            let wider = predictor_width::wider_predictor(&prefetched, PREDICTOR_LANES);
            for source in [original, fused, prefetched, wider] {
                let original_declarations: Vec<_> = source
                    .lines()
                    .filter(|line| line.starts_with("var<workgroup>"))
                    .collect();
                for lanes in SUPPORTED_LANES {
                    let candidate = gather_context(&source, lanes);
                    assert_ne!(candidate, source);
                    assert!(!candidate.contains(CONTEXT_BLOCK));
                    assert_eq!(
                        candidate
                            .matches("fn blend_gathered_recalled_context(")
                            .count(),
                        1
                    );
                    assert_eq!(candidate.matches("    blend_gathered_recalled_context(brain_base, pattern_base, recall_count, tid);").count(), 1);
                    assert!(
                        candidate.contains(&format!("const CONTEXT_GATHER_LANES: u32 = {lanes}u;"))
                    );
                    let gathered_declarations: Vec<_> = candidate
                        .lines()
                        .filter(|line| line.starts_with("var<workgroup>"))
                        .collect();
                    assert_eq!(gathered_declarations, original_declarations);
                    #[cfg(not(target_arch = "wasm32"))]
                    validate_brain_source(&candidate);
                }
            }
        }
    }

    #[test]
    fn context_gather_rejects_unsupported_lanes() {
        let source = compose_brain_passes(false);
        for lanes in UNSUPPORTED_LANES {
            assert_rejected(&source, lanes);
        }
    }

    #[test]
    fn context_gather_requires_existing_scratch() {
        let source = compose_brain_passes(true);
        for declaration in SCRATCH_DECLARATIONS {
            assert_eq!(source.matches(declaration).count(), 1);
            let missing = source.replacen(declaration, "", 1);
            for lanes in SUPPORTED_LANES {
                assert_rejected(&missing, lanes);
            }
        }
    }

    #[test]
    fn context_gather_rejects_conflicting_scratch_lifetimes() {
        let source = compose_brain_passes(true);
        let conflict = source.replacen(
            CONTEXT_BLOCK,
            &format!("    s_reinf_dot[tid] = 0.0;\n{CONTEXT_BLOCK}"),
            1,
        );
        let missing_initialization = source.replacen("        s_reinf_dot[tid] = dot;", "", 1);
        assert_ne!(missing_initialization, source);
        for lanes in SUPPORTED_LANES {
            assert_rejected(&conflict, lanes);
            assert_rejected(&missing_initialization, lanes);
        }
    }

    #[test]
    fn context_gather_requires_a_unique_original_context() {
        let source = compose_brain_passes(false);
        let duplicate = format!("{source}\n{CONTEXT_BLOCK}");
        for lanes in SUPPORTED_LANES {
            assert_rejected(&duplicate, lanes);
            let already_gathered = gather_context(&source, lanes);
            assert_rejected(&already_gathered, lanes);
        }
    }
}
