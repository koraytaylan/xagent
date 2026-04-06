// Pass 2: Features x encoder_weights + biases -> encoded state
// Matrix multiply: for each dim d, encoded[d] = fast_tanh(sum_f(features[f] * weights[f*DIM+d]) + biases[d])

@group(0) @binding(0) var<storage, read> features: array<f32>;
@group(0) @binding(1) var<storage, read> brain_state: array<f32>;
@group(0) @binding(2) var<storage, read_write> encoded: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent = gid.x;
    let f_base = agent * FEATURES_STRIDE;
    let b_base = agent * BRAIN_STRIDE;
    let e_base = agent * DIM;

    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        var sum: f32 = brain_state[b_base + O_ENC_BIASES + d];
        for (var f: u32 = 0u; f < FEATURE_COUNT; f = f + 1u) {
            sum += features[f_base + f] * brain_state[b_base + O_ENC_WEIGHTS + f * DIM + d];
        }
        encoded[e_base + d] = fast_tanh(sum);
    }
}
