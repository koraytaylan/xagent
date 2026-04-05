// Pass 4: Cosine similarity of habituated state vs all memory patterns

@group(0) @binding(0) var<storage, read> habituated: array<f32>;
@group(0) @binding(1) var<storage, read> patterns: array<f32>;
@group(0) @binding(2) var<storage, read_write> similarities: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent = gid.x;
    let h_base = agent * DIM;
    let p_base = agent * PATTERN_STRIDE;
    let s_base = agent * MEMORY_CAP;

    // Compute query norm
    var q_norm_sq: f32 = 0.0;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let v = habituated[h_base + d];
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
            dot += habituated[h_base + d] * patterns[p_base + O_PAT_STATES + j * DIM + d];
        }

        let p_norm = patterns[p_base + O_PAT_NORMS + j];

        if (q_norm < 1e-8 || p_norm < 1e-8) {
            similarities[s_base + j] = 0.0;
        } else {
            similarities[s_base + j] = clamp(dot / (q_norm * p_norm), -1.0, 1.0);
        }
    }
}
