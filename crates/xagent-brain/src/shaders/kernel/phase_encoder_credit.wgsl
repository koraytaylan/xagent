// Every encoder weight update is independent once the main dispatch has
// published its adapted features and decision credit. One invocation owns
// one feature/output pair. Each update retains the scalar threshold, scale,
// multiply/add and clamp used by the inline brain path.
const ENCODER_CREDIT_THREADS: u32 = 256u;

fn phase_encoder_credit(agent_id: u32, weight: u32) {
    if weight >= FEATURE_COUNT * ENCODED_DIMENSION { return; }
    if physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5 { return; }
    let feature = weight / ENCODED_DIMENSION;
    let dim = weight % ENCODED_DIMENSION;
    let brain_base = agent_id * BRAIN_STRIDE;
    let decision_base = agent_id * DECISION_STRIDE;
    let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;

    let action_credit = decision_buffer[decision_base + DECISION_CREDIT + dim];
    if abs(action_credit) >= CREDIT_EPSILON {
        let learning_rate = brain_config[1].x;
        let scale = learning_rate * action_credit * ENCODER_CREDIT_SCALE;
        var weight_value = brain_state[brain_base + O_ENC_WEIGHTS + feature * ENCODED_DIMENSION + dim]
            + scale * brain_scratch[agent_scratch + SCRATCH_FEATURES + feature];
        weight_value = clamp(weight_value, -2.0, 2.0);
        brain_state[brain_base + O_ENC_WEIGHTS + feature * ENCODED_DIMENSION + dim] = weight_value;
    }
}
