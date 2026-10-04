// Every active pattern queries the same centered encoding. One otherwise
// unused invocation computes its original ascending norm while the pattern
// invocations retain their original ascending dot products.
fn coop_recall_score(agent_id: u32, tid: u32) {
    let pattern_base = agent_id * PATTERN_STRIDE;
    var is_active: f32 = 0.0;
    var dot: f32 = 0.0;

    if (tid == MEMORY_CAP) {
        var q_norm_sq: f32 = 0.0;
        for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
            let key = s_memory_key[d];
            q_norm_sq += key * key;
        }
        // Cortex has already consumed this scratch. Learning later replaces
        // it with its own differently ordered tree norm before every read.
        s_enc_norm = sqrt(q_norm_sq);
    }
    if (tid < MEMORY_CAP) {
        is_active = pattern_buffer[pattern_base + O_PAT_ACTIVE + tid];
        if (!(is_active < 0.5)) {
            for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
                let key = s_memory_key[d];
                dot += key * pattern_buffer[pattern_base + d * MEMORY_CAP + tid];
            }
        }
    }

    // The caller's alive/pass guards are uniform. All invocations reach this
    // barrier, including the norm writer and the inactive-pattern lanes.
    workgroupBarrier();
    if (tid < MEMORY_CAP) {
        if (is_active < 0.5) {
            s_similarities[tid] = -2.0;
        } else {
            let q_norm = s_enc_norm;
            let p_norm = pattern_buffer[pattern_base + O_PAT_NORMS + tid];
            if (q_norm < 1e-8 || p_norm < 1e-8) {
                s_similarities[tid] = 0.0;
            } else {
                s_similarities[tid] = clamp(dot / (q_norm * p_norm), -1.0, 1.0);
            }
        }
    }
}
