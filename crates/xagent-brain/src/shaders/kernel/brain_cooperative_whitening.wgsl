// Optional cooperative brain helpers, composed with common + brain_passes.
// One workgroup owns one agent. Whitening borrows the first two matrix-sized
// regions of s_reinf_dot after context blending releases its scratch. The
// learning pass overwrites every s_reinf_dot[tid], then barriers, before its
// first read. If an agent is dead or learning is skipped, no later pass reads
// these temporary values. No additional workgroup resource is allocated.
const WHITENING_MATRIX_CELLS: u32 = MAX_SENSE_CHANNELS * MAX_SENSE_CHANNELS;

fn whitening_covariance_index(row: u32, column: u32, n: u32) -> u32 {
    return row * n + column;
}

fn whitening_eigenvector_index(row: u32, column: u32, n: u32) -> u32 {
    return WHITENING_MATRIX_CELLS + row * n + column;
}

// Invocation zero preserves the scalar Jacobi rotations and their order.
// The output cells are independent, and each retains the scalar trace and
// eigenvector sum order. Senses run one after another, each closed by its
// barriers, so the shared scratch is reused serially. Every invocation reaches
// every barrier, including ticks where a refresh is disabled or the cortex
// supplies visual features.
fn cooperative_refresh_vision_whitening(brain_base: u32, tid: u32) {
    for (var sense = 0u; sense < SENSE_COUNT; sense++) {
        let n = sense_channels(sense);
        let refresh = sense_whitening_kind(sense) == SENSE_WHITENING_KIND_JACOBI
            && !(bc_f32(CFG_VISUAL_CORTEX_ENABLED) != 0.0
                && sense_input_kind(sense) == SENSE_INPUT_KIND_VISUAL_HEMIFIELDS)
            && u32(brain_state[brain_base + O_TICK_COUNT]) % sense_refresh_ticks(sense) == 0u;
        if (refresh && tid == 0u) {
            for (var i = 0u; i < n; i++) {
                for (var j = 0u; j < n; j++) {
                    s_reinf_dot[whitening_covariance_index(i, j, n)] = brain_state[brain_base + sense_covariance(sense) + i * n + j];
                    s_reinf_dot[whitening_eigenvector_index(i, j, n)] = select(0.0, 1.0, i == j);
                }
            }
            for (var sweep = 0u; sweep < JACOBI_SWEEPS; sweep++) {
                var off = 0.0;
                for (var p = 0u; p < n; p++) {
                    for (var q = p + 1u; q < n; q++) {
                        off += s_reinf_dot[whitening_covariance_index(p, q, n)] * s_reinf_dot[whitening_covariance_index(p, q, n)];
                    }
                }
                if (off < 1e-30) { break; }
                for (var p = 0u; p < n; p++) {
                    for (var q = p + 1u; q < n; q++) {
                        let apq = s_reinf_dot[whitening_covariance_index(p, q, n)];
                        if (abs(apq) < 1e-30) { continue; }
                        let theta = (s_reinf_dot[whitening_covariance_index(q, q, n)] - s_reinf_dot[whitening_covariance_index(p, p, n)]) / (2.0 * apq);
                        var t = sign(theta) / (abs(theta) + sqrt(theta * theta + 1.0));
                        if (theta == 0.0) { t = 1.0; }
                        let c = 1.0 / sqrt(t * t + 1.0);
                        let s = t * c;
                        for (var k = 0u; k < n; k++) {
                            let akp = s_reinf_dot[whitening_covariance_index(k, p, n)];
                            let akq = s_reinf_dot[whitening_covariance_index(k, q, n)];
                            s_reinf_dot[whitening_covariance_index(k, p, n)] = c * akp - s * akq;
                            s_reinf_dot[whitening_covariance_index(k, q, n)] = s * akp + c * akq;
                        }
                        for (var k = 0u; k < n; k++) {
                            let apk = s_reinf_dot[whitening_covariance_index(p, k, n)];
                            let aqk = s_reinf_dot[whitening_covariance_index(q, k, n)];
                            s_reinf_dot[whitening_covariance_index(p, k, n)] = c * apk - s * aqk;
                            s_reinf_dot[whitening_covariance_index(q, k, n)] = s * apk + c * aqk;
                        }
                        for (var k = 0u; k < n; k++) {
                            let vkp = s_reinf_dot[whitening_eigenvector_index(k, p, n)];
                            let vkq = s_reinf_dot[whitening_eigenvector_index(k, q, n)];
                            s_reinf_dot[whitening_eigenvector_index(k, p, n)] = c * vkp - s * vkq;
                            s_reinf_dot[whitening_eigenvector_index(k, q, n)] = s * vkp + c * vkq;
                        }
                    }
                }
            }
        }
        workgroupBarrier();
        if (refresh && tid < n * n) {
            let i = tid / n;
            let j = tid % n;
            var trace = 0.0;
            for (var k = 0u; k < n; k++) {
                trace += s_reinf_dot[whitening_covariance_index(k, k, n)];
            }
            let floor_value = WHITENING_RELATIVE_FLOOR * trace / f32(n) + 1e-14;
            var w = 0.0;
            for (var k = 0u; k < n; k++) {
                w += s_reinf_dot[whitening_eigenvector_index(i, k, n)] * s_reinf_dot[whitening_eigenvector_index(j, k, n)] / sqrt(max(s_reinf_dot[whitening_covariance_index(k, k, n)], floor_value));
            }
            brain_state[brain_base + sense_whitening(sense) + i * n + j] = w;
        }
        storageBarrier();
        workgroupBarrier();
    }
}

// Sharing a key load between the norm and dot updates shortens its lifetime.
// Each accumulator still visits every dimension in ascending order. Inactive
// patterns retain their sentinel without computing an unobserved query norm.
fn coop_recall_score(agent_id: u32, tid: u32) {
    if (tid < MEMORY_CAP) {
        let pattern_base = agent_id * PATTERN_STRIDE;
        let is_active = pattern_buffer[pattern_base + O_PAT_ACTIVE + tid];
        if (is_active < 0.5) {
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
            if (q_norm < 1e-8 || p_norm < 1e-8) {
                s_similarities[tid] = 0.0;
            } else {
                s_similarities[tid] = clamp(dot / (q_norm * p_norm), -1.0, 1.0);
            }
        }
    }
}
