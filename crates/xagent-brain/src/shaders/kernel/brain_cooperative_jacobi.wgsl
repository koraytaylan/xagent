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
    if (tid == 0u) {
        let refresh = bc_f32(CFG_VISUAL_CORTEX_ENABLED) == 0.0
            && u32(brain_state[brain_base + O_TICK_COUNT]) % VISION_WHITENING_REFRESH == 0u;
        s_reinf_dot[JACOBI_REFRESH_FLAG] = select(0.0, 1.0, refresh);
    }
    // The builtin publishes a uniform value and synchronizes its readers.
    // Ordinary and cortex ticks never enter any rotation-loop barriers.
    let refresh = workgroupUniformLoad(&s_reinf_dot[JACOBI_REFRESH_FLAG]) != 0.0;
    if (refresh) {
        if (tid < WHITENING_MATRIX_CELLS) {
            let i = tid / VISION_PATHWAY_INPUTS;
            let j = tid % VISION_PATHWAY_INPUTS;
            s_reinf_dot[whitening_covariance_index(i, j)] = brain_state[brain_base + O_VISION_PATHWAY_COVARIANCE + i * VISION_PATHWAY_INPUTS + j];
            s_reinf_dot[whitening_eigenvector_index(i, j)] = select(0.0, 1.0, i == j);
        }
        workgroupBarrier();
        for (var sweep = 0u; sweep < JACOBI_SWEEPS; sweep++) {
            if (tid == 0u) {
                var off = 0.0;
                for (var p = 0u; p < VISION_PATHWAY_INPUTS; p++) {
                    for (var q = p + 1u; q < VISION_PATHWAY_INPUTS; q++) {
                        off += s_reinf_dot[whitening_covariance_index(p, q)] * s_reinf_dot[whitening_covariance_index(p, q)];
                    }
                }
                s_reinf_dot[JACOBI_SWEEP_FLAG] = select(1.0, 0.0, off < 1e-30);
            }
            let continue_sweep = workgroupUniformLoad(&s_reinf_dot[JACOBI_SWEEP_FLAG]) != 0.0;
            if (!continue_sweep) { break; }
            for (var p = 0u; p < VISION_PATHWAY_INPUTS; p++) {
                for (var q = p + 1u; q < VISION_PATHWAY_INPUTS; q++) {
                    if (tid == 0u) {
                        let apq = s_reinf_dot[whitening_covariance_index(p, q)];
                        s_reinf_dot[JACOBI_ROTATION_FLAG] = select(1.0, 0.0, abs(apq) < 1e-30);
                        if (!(abs(apq) < 1e-30)) {
                            let theta = (s_reinf_dot[whitening_covariance_index(q, q)] - s_reinf_dot[whitening_covariance_index(p, p)]) / (2.0 * apq);
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
                    if (s_reinf_dot[JACOBI_ROTATION_FLAG] != 0.0 && tid < VISION_PATHWAY_INPUTS) {
                        let k = tid;
                        let c = s_reinf_dot[JACOBI_COSINE];
                        let s = s_reinf_dot[JACOBI_SINE];
                        let akp = s_reinf_dot[whitening_covariance_index(k, p)];
                        let akq = s_reinf_dot[whitening_covariance_index(k, q)];
                        s_reinf_dot[whitening_covariance_index(k, p)] = c * akp - s * akq;
                        s_reinf_dot[whitening_covariance_index(k, q)] = s * akp + c * akq;
                        let vkp = s_reinf_dot[whitening_eigenvector_index(k, p)];
                        let vkq = s_reinf_dot[whitening_eigenvector_index(k, q)];
                        s_reinf_dot[whitening_eigenvector_index(k, p)] = c * vkp - s * vkq;
                        s_reinf_dot[whitening_eigenvector_index(k, q)] = s * vkp + c * vkq;
                    }
                    workgroupBarrier();
                    // The column phase must finish before any lane reads the
                    // intersecting row cells. Both operands precede stores.
                    if (s_reinf_dot[JACOBI_ROTATION_FLAG] != 0.0 && tid < VISION_PATHWAY_INPUTS) {
                        let k = tid;
                        let c = s_reinf_dot[JACOBI_COSINE];
                        let s = s_reinf_dot[JACOBI_SINE];
                        let apk = s_reinf_dot[whitening_covariance_index(p, k)];
                        let aqk = s_reinf_dot[whitening_covariance_index(q, k)];
                        s_reinf_dot[whitening_covariance_index(p, k)] = c * apk - s * aqk;
                        s_reinf_dot[whitening_covariance_index(q, k)] = s * apk + c * aqk;
                    }
                    // Skipped rotations also reach this barrier: no wave may
                    // overwrite the next pair's flag while another reads it.
                    workgroupBarrier();
                }
            }
        }
        if (tid < WHITENING_MATRIX_CELLS) {
            let i = tid / VISION_PATHWAY_INPUTS;
            let j = tid % VISION_PATHWAY_INPUTS;
            var trace = 0.0;
            for (var k = 0u; k < VISION_PATHWAY_INPUTS; k++) {
                trace += s_reinf_dot[whitening_covariance_index(k, k)];
            }
            let floor_value = WHITENING_RELATIVE_FLOOR * trace / f32(VISION_PATHWAY_INPUTS) + 1e-14;
            var w = 0.0;
            for (var k = 0u; k < VISION_PATHWAY_INPUTS; k++) {
                w += s_reinf_dot[whitening_eigenvector_index(i, k)] * s_reinf_dot[whitening_eigenvector_index(j, k)] / sqrt(max(s_reinf_dot[whitening_covariance_index(k, k)], floor_value));
            }
            brain_state[brain_base + O_VISION_PATHWAY_WHITENING + i * VISION_PATHWAY_INPUTS + j] = w;
        }
    }
    storageBarrier();
    workgroupBarrier();
}
