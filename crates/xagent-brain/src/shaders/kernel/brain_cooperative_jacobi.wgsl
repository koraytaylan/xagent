// One agent owns the workgroup. Context gathering releases s_reinf_dot before
// this call; learning initializes every element before its next read. The two
// matrices occupy [0,128), and the five publication words follow them. Encoder
// packing changes only s_dense_partials, leaving this ownership unchanged.
const JACOBI_REFRESH_FLAG: u32 = 2u * WHITENING_MATRIX_CELLS;
const JACOBI_SWEEP_FLAG: u32 = JACOBI_REFRESH_FLAG + 1u;
const JACOBI_ROTATION_FLAG: u32 = JACOBI_SWEEP_FLAG + 1u;
const JACOBI_COSINE: u32 = JACOBI_ROTATION_FLAG + 1u;
const JACOBI_SINE: u32 = JACOBI_COSINE + 1u;

fn cooperative_refresh_vision_whitening(brain_base: u32, tid: u32) {
    for (var sense = 0u; sense < SENSE_COUNT; sense++) {
    let n = sense_channels(sense);
    if (tid == 0u) {
        let refresh = sense_whitening_kind(sense) == SENSE_WHITENING_KIND_JACOBI
            && !(bc_f32(CFG_VISUAL_CORTEX_ENABLED) != 0.0
                && sense_input_kind(sense) == SENSE_INPUT_KIND_VISUAL_HEMIFIELDS)
            && u32(brain_state[brain_base + O_TICK_COUNT]) % sense_refresh_ticks(sense) == 0u;
        s_reinf_dot[JACOBI_REFRESH_FLAG] = select(0.0, 1.0, refresh);
    }
    // The builtin publishes a uniform value and synchronizes its readers.
    // Ordinary and cortex ticks never enter any rotation-loop barriers.
    let refresh = workgroupUniformLoad(&s_reinf_dot[JACOBI_REFRESH_FLAG]) != 0.0;
    if (refresh) {
        if (tid < n * n) {
            let i = tid / n;
            let j = tid % n;
            s_reinf_dot[whitening_covariance_index(i, j, n)] = brain_state[brain_base + sense_covariance(sense) + i * n + j];
            s_reinf_dot[whitening_eigenvector_index(i, j, n)] = select(0.0, 1.0, i == j);
        }
        workgroupBarrier();
        for (var sweep = 0u; sweep < JACOBI_SWEEPS; sweep++) {
            if (tid == 0u) {
                var off = 0.0;
                for (var p = 0u; p < n; p++) {
                    for (var q = p + 1u; q < n; q++) {
                        off += s_reinf_dot[whitening_covariance_index(p, q, n)] * s_reinf_dot[whitening_covariance_index(p, q, n)];
                    }
                }
                s_reinf_dot[JACOBI_SWEEP_FLAG] = select(1.0, 0.0, off < 1e-30);
            }
            let continue_sweep = workgroupUniformLoad(&s_reinf_dot[JACOBI_SWEEP_FLAG]) != 0.0;
            if (!continue_sweep) { break; }
            for (var p = 0u; p < n; p++) {
                for (var q = p + 1u; q < n; q++) {
                    if (tid == 0u) {
                        let apq = s_reinf_dot[whitening_covariance_index(p, q, n)];
                        s_reinf_dot[JACOBI_ROTATION_FLAG] = select(1.0, 0.0, abs(apq) < 1e-30);
                        if (!(abs(apq) < 1e-30)) {
                            let theta = (s_reinf_dot[whitening_covariance_index(q, q, n)] - s_reinf_dot[whitening_covariance_index(p, p, n)]) / (2.0 * apq);
                            var t = sign(theta) / (abs(theta) + sqrt(theta * theta + 1.0));
                            if (theta == 0.0) { t = 1.0; }
                            let c = 1.0 / sqrt(t * t + 1.0);
                            let s = t * c;
                            s_reinf_dot[JACOBI_COSINE] = c;
                            s_reinf_dot[JACOBI_SINE] = s;
                        }
                    }
                    workgroupBarrier();
                    // Each lane owns two cells in one row. Eigenvectors are
                    // independent of covariance, so their column update can
                    // share this phase without changing any dependency.
                    if (s_reinf_dot[JACOBI_ROTATION_FLAG] != 0.0 && tid < n) {
                        let k = tid;
                        let c = s_reinf_dot[JACOBI_COSINE];
                        let s = s_reinf_dot[JACOBI_SINE];
                        let akp = s_reinf_dot[whitening_covariance_index(k, p, n)];
                        let akq = s_reinf_dot[whitening_covariance_index(k, q, n)];
                        s_reinf_dot[whitening_covariance_index(k, p, n)] = c * akp - s * akq;
                        s_reinf_dot[whitening_covariance_index(k, q, n)] = s * akp + c * akq;
                        let vkp = s_reinf_dot[whitening_eigenvector_index(k, p, n)];
                        let vkq = s_reinf_dot[whitening_eigenvector_index(k, q, n)];
                        s_reinf_dot[whitening_eigenvector_index(k, p, n)] = c * vkp - s * vkq;
                        s_reinf_dot[whitening_eigenvector_index(k, q, n)] = s * vkp + c * vkq;
                    }
                    workgroupBarrier();
                    // The column phase must finish before any lane reads the
                    // intersecting row cells. Both operands precede stores.
                    if (s_reinf_dot[JACOBI_ROTATION_FLAG] != 0.0 && tid < n) {
                        let k = tid;
                        let c = s_reinf_dot[JACOBI_COSINE];
                        let s = s_reinf_dot[JACOBI_SINE];
                        let apk = s_reinf_dot[whitening_covariance_index(p, k, n)];
                        let aqk = s_reinf_dot[whitening_covariance_index(q, k, n)];
                        s_reinf_dot[whitening_covariance_index(p, k, n)] = c * apk - s * aqk;
                        s_reinf_dot[whitening_covariance_index(q, k, n)] = s * apk + c * aqk;
                    }
                    // Skipped rotations also reach this barrier: no wave may
                    // overwrite the next pair's flag while another reads it.
                    workgroupBarrier();
                }
            }
        }
        if (tid < n * n) {
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
    }
    storageBarrier();
    workgroupBarrier();
    }
}
