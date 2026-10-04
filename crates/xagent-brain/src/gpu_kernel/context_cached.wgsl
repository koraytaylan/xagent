// Test-only context normalization cache. The encoder/cortex no longer needs
// s_enc_norm here, and TD credit overwrites all s_credit entries before reading
// them. No additional workgroup storage is allocated.
fn blend_cached_recalled_context(brain_base: u32, pattern_base: u32, recall_count: u32, tid: u32) {
    if (tid == 0u) {
        var total_sim: f32 = 0.0;
        for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
            total_sim += max(s_recall_similarity[k], 0.0);
        }
        s_enc_norm = total_sim;
    }
    workgroupBarrier();

    if (tid < RECALL_K && tid < recall_count && s_enc_norm > 1e-8) {
        let context_weight = brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT];
        s_credit[tid] = context_weight * max(s_recall_similarity[tid], 0.0) / s_enc_norm;
    }
    workgroupBarrier();

    if (tid < PREDICTOR_DIMENSION) {
        if (recall_count > 0u && s_enc_norm > 1e-8) {
            let encoded_mean = brain_state[brain_base + O_ENCODED_MEAN + tid];
            for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                let idx = u32(s_recall[k]);
                let w = s_credit[k];
                s_prediction[tid] +=
                    (pattern_buffer[pattern_base + tid * MEMORY_CAP + idx] + encoded_mean) * w;
            }
        }
        s_prediction[tid] = fast_tanh(s_prediction[tid]);
    }
    workgroupBarrier();
}
