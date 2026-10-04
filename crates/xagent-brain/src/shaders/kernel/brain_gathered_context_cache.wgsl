// Packed encoding releases all 512 partials before prediction. The gathered
// context uses only the lower 128 entries; coefficients and their ordered
// total remain in the upper half until every output tile has consumed them.
const CONTEXT_GATHER_LANES: u32 = 8u;
const CONTEXT_GATHER_OUTPUT_TILE: u32 = BRAIN_WORKGROUP_SIZE / CONTEXT_GATHER_LANES;
const CONTEXT_COEFFICIENT_BASE: u32 = 2u * ENCODED_DIMENSION;
const CONTEXT_TOTAL_SLOT: u32 = CONTEXT_COEFFICIENT_BASE + RECALL_K;

fn blend_gathered_recalled_context(
    brain_base: u32,
    pattern_base: u32,
    recall_count: u32,
    tid: u32,
) {
    if (tid == 0u) {
        var total_sim: f32 = 0.0;
        for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
            total_sim += max(s_recall_similarity[k], 0.0);
        }
        s_dense_partials[CONTEXT_TOTAL_SLOT] = total_sim;
    }
    workgroupBarrier();
    if (tid < RECALL_K) {
        var coefficient = 0.0;
        let total_sim = s_dense_partials[CONTEXT_TOTAL_SLOT];
        if (tid < recall_count && total_sim > 1e-8) {
            let context_weight = brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT];
            coefficient = context_weight * max(s_recall_similarity[tid], 0.0) / total_sim;
        }
        s_dense_partials[CONTEXT_COEFFICIENT_BASE + tid] = coefficient;
    }

    let output_in_tile = tid / CONTEXT_GATHER_LANES;
    let lane = tid % CONTEXT_GATHER_LANES;
    for (var tile = 0u; tile < PREDICTOR_DIMENSION; tile += CONTEXT_GATHER_OUTPUT_TILE) {
        let dim = tile + output_in_tile;
        var recalled_value = 0.0;
        if (dim < PREDICTOR_DIMENSION && lane < recall_count) {
            let idx = u32(s_recall[lane]);
            recalled_value = pattern_buffer[pattern_base + dim * MEMORY_CAP + idx];
        }
        s_dense_partials[tid] = recalled_value;
        if (CONTEXT_GATHER_LANES < RECALL_K) {
            let upper_lane = lane + CONTEXT_GATHER_LANES;
            var upper_value = 0.0;
            if (dim < PREDICTOR_DIMENSION && upper_lane < recall_count) {
                let idx = u32(s_recall[upper_lane]);
                upper_value = pattern_buffer[pattern_base + dim * MEMORY_CAP + idx];
            }
            s_reinf_dot[tid] = upper_value;
        }
        // The first tile also publishes all sixteen cached coefficients.
        workgroupBarrier();

        if (lane == 0u && dim < PREDICTOR_DIMENSION) {
            if (recall_count > 0u) {
                let total_sim = s_dense_partials[CONTEXT_TOTAL_SLOT];
                if (total_sim > 1e-8) {
                    let encoded_mean = brain_state[brain_base + O_ENCODED_MEAN + dim];
                    for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                        let w = s_dense_partials[CONTEXT_COEFFICIENT_BASE + k];
                        var raw_value: f32;
                        if (CONTEXT_GATHER_LANES == RECALL_K) {
                            raw_value = s_dense_partials[tid + k];
                        } else if (k < CONTEXT_GATHER_LANES) {
                            raw_value = s_dense_partials[tid + k];
                        } else {
                            raw_value = s_reinf_dot[tid + k - CONTEXT_GATHER_LANES];
                        }
                        s_prediction[dim] +=
                            (raw_value + encoded_mean) * w;
                    }
                }
            }
            s_prediction[dim] = fast_tanh(s_prediction[dim]);
        }
        workgroupBarrier();
    }
}
