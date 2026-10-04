//! Optional source composition for the inline predictor. Each invocation
//! trains and consumes its own weight columns in their existing order.
//! The checked-in brain fragment remains the separate-loop reference.

/// Exact original block shared by fused and separate serial brain entries.
const SEPARATE_LOOPS: &str = r"            for (var j = lane; j < ENCODED_DIMENSION; j += DENSE_INNER_LANES) {
                let previous_input = brain_state[brain_base + O_PREV_ENCODED + j];
                let grad = clamp(transition_error * tanh_derivative * previous_input, -1.0, 1.0);
                var w = brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j] - predictor_learning_rate * grad;
                w = clamp(w, -3.0, 3.0);
                brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j] = w;
            }
            workgroupBarrier(); // Weight writes must be visible before predict step

            // PREDICT sub-step: each lane accumulates, lane 0 reduces
            var partial: f32 = 0.0;
            for (var j = lane; j < ENCODED_DIMENSION; j += DENSE_INNER_LANES) {
                partial += s_encoded[j] * brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j];
            }
";

/// Weight update/clamp and the ascending stride-four sum retain their order.
const FUSED_LOOP: &str = r"            // Each lane predicts from the exact clamped f32 weight it stores.
            var partial: f32 = 0.0;
            for (var j = lane; j < ENCODED_DIMENSION; j += DENSE_INNER_LANES) {
                let previous_input = brain_state[brain_base + O_PREV_ENCODED + j];
                let grad = clamp(transition_error * tanh_derivative * previous_input, -1.0, 1.0);
                var w = brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j] - predictor_learning_rate * grad;
                w = clamp(w, -3.0, 3.0);
                brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j] = w;
                partial += s_encoded[j] * w;
            }
";

/// Fuse the inline predictor while preserving its four-lane arithmetic.
///
/// Every invocation owns the columns that it subsequently reads, so the
/// removed training barrier has no cross-invocation dependency. The existing
/// partial-sum and tile barriers remain, as does the four-partial reduction.
/// Cooperative whitening and interleaved recall modify disjoint source blocks.
///
/// Apply this only to a brain-pass fragment, before composing its entry point.
/// The ParallelTiled tail uses the scratch-prediction branch, so this inline
/// transformation does not change its separate tiled predictor or sum order.
///
/// # Panics
///
/// Panics if the original block is missing or repeated, preventing a changed
/// shader or a second composition from silently bypassing the requested mode.
pub(super) fn fuse_inline_predictor(source: &str) -> String {
    assert_eq!(
        source.matches(SEPARATE_LOOPS).count(),
        1,
        "inline predictor source must contain exactly one separate train/predict block"
    );
    let fused = source.replacen(SEPARATE_LOOPS, FUSED_LOOP, 1);
    assert_eq!(fused.matches(FUSED_LOOP).count(), 1);
    assert_eq!(
        source.matches("workgroupBarrier();").count(),
        fused.matches("workgroupBarrier();").count() + 1,
        "inline predictor fusion removes only the training barrier"
    );
    fused
}
