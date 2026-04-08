// Pass 4: Cosine similarity of encoded state vs all memory patterns
// Uses encoded (pre-habituation) state as the memory query so that
// sustained-input attenuation does not silence the memory system.

@group(0) @binding(0) var<storage, read> encoded: array<f32>;
@group(0) @binding(1) var<storage, read> patterns: array<f32>;
@group(0) @binding(2) var<storage, read_write> similarities: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent = gid.x;
    let e_base = agent * DIM;
    let p_base = agent * PATTERN_STRIDE;
    let s_base = agent * MEMORY_CAP;

    // Compute query norm
    var q_norm_sq: f32 = 0.0;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let v = encoded[e_base + d];
        q_norm_sq += v * v;
    }
    let q_norm = sqrt(q_norm_sq);

    // Score each pattern slot
    for (var j: u32 = 0u; j < MEMORY_CAP; j = j + 1u) {
        let is_active = patterns[p_base + O_PAT_ACTIVE + j];
        if (is_active < 0.5) {
            similarities[s_base + j] = -2.0;
            continue;
        }

        // Dot product
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            dot += encoded[e_base + d] * patterns[p_base + d * MEMORY_CAP + j];
        }

        let p_norm = patterns[p_base + O_PAT_NORMS + j];

        if (q_norm < 1e-8 || p_norm < 1e-8) {
            similarities[s_base + j] = 0.0;
        } else {
            similarities[s_base + j] = clamp(dot / (q_norm * p_norm), -1.0, 1.0);
        }
    }
}
