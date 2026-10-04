// Test-only recall variant. Each accumulator visits dimensions in the same
// ascending order as the two original loops. Both key uses share one loop
// iteration, so their source-level lifetime does not span separate loops.
fn coop_recall_score(agent_id: u32, tid: u32) {
    if tid < MEMORY_CAP {
        let pattern_base = agent_id * PATTERN_STRIDE;
        let is_active = pattern_buffer[pattern_base + O_PAT_ACTIVE + tid];
        if is_active < 0.5 {
            // The original norm calculation has no observable result here.
            s_similarities[tid] = -2.0;
        } else {
            var q_norm_sq: f32 = 0.0;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
                let key = s_memory_key[d];
                q_norm_sq += key * key;
                dot += key * pattern_buffer[pattern_base + d * MEMORY_CAP + tid];
            }
            let q_norm = sqrt(q_norm_sq);
            let p_norm = pattern_buffer[pattern_base + O_PAT_NORMS + tid];
            if q_norm < 1e-8 || p_norm < 1e-8 {
                s_similarities[tid] = 0.0;
            } else {
                s_similarities[tid] = clamp(dot / (q_norm * p_norm), -1.0, 1.0);
            }
        }
    }
}
