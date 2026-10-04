// Every encoder weight update is independent after the tail publishes credit.
// One invocation owns one feature/output pair; the threshold, scale, update
// and clamp retain the tiled credit stage's original scalar expressions.
const POINTWISE_CREDIT_THREADS: u32 = 256u;

@compute @workgroup_size(POINTWISE_CREDIT_THREADS)
fn exact_pointwise_encoder_credit(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let agent_id = wgid.x;
    let weight = wgid.y * POINTWISE_CREDIT_THREADS + lid.x;
    if (weight >= FEATURE_COUNT * ENCODED_DIMENSION) { return; }
    if (physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }
    let feature = weight / ENCODED_DIMENSION;
    let dim = weight % ENCODED_DIMENSION;
    let brain_base = agent_id * BRAIN_STRIDE;
    let decision_base = agent_id * DECISION_STRIDE;
    let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;

    let action_credit = decision_buffer[decision_base + DECISION_CREDIT + dim];
    if (abs(action_credit) >= CREDIT_EPSILON) {
        let learning_rate = brain_config[1].x;
        let scale = learning_rate * action_credit * ENCODER_CREDIT_SCALE;
        var w = brain_state[brain_base + O_ENC_WEIGHTS + feature * ENCODED_DIMENSION + dim]
            + scale * brain_scratch[agent_scratch + SCRATCH_FEATURES + feature];
        w = clamp(w, -2.0, 2.0);
        brain_state[brain_base + O_ENC_WEIGHTS + feature * ENCODED_DIMENSION + dim] = w;
    }
}
