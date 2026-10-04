// One invocation exclusively owns the complete storage vector. Component
// assignments below affect a private local, followed by one whole-vector
// store, so no invocation can race through a component-store read/modify/write.
const ENCODER_CREDIT_THREADS: u32 = 256u;

fn phase_encoder_credit(agent_id: u32, weight_vector: u32) {
    if weight_vector >= FEATURE_COUNT * PACKED_ENCODER_OUTPUT_VECTORS { return; }
    if physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5 { return; }
    let feature = weight_vector / PACKED_ENCODER_OUTPUT_VECTORS;
    let dimension = (weight_vector % PACKED_ENCODER_OUTPUT_VECTORS) * PACKED_ENCODER_WIDTH;
    let decision_base = agent_id * DECISION_STRIDE + DECISION_CREDIT + dimension;
    let credits = vec4<f32>(
        decision_buffer[decision_base],
        decision_buffer[decision_base + 1u],
        decision_buffer[decision_base + 2u],
        decision_buffer[decision_base + 3u],
    );
    let credit_enabled = abs(credits) >= vec4<f32>(CREDIT_EPSILON);
    if !any(credit_enabled) { return; }
    let address = agent_id * FEATURE_COUNT * PACKED_ENCODER_OUTPUT_VECTORS + weight_vector;
    let input = packed_encoder.scratch[agent_id * BRAIN_SCRATCH_STRIDE + SCRATCH_FEATURES + feature];
    let learning_rate = brain_config[1].x;
    var weight = packed_encoder.weights[address];
    // Skipped components execute neither addition nor clamp, preserving their
    // exact bit pattern, including signed zero and values outside the clamp.
    if credit_enabled.x {
        let scale = learning_rate * credits.x * ENCODER_CREDIT_SCALE;
        weight.x = clamp(weight.x + scale * input, -2.0, 2.0);
    }
    if credit_enabled.y {
        let scale = learning_rate * credits.y * ENCODER_CREDIT_SCALE;
        weight.y = clamp(weight.y + scale * input, -2.0, 2.0);
    }
    if credit_enabled.z {
        let scale = learning_rate * credits.z * ENCODER_CREDIT_SCALE;
        weight.z = clamp(weight.z + scale * input, -2.0, 2.0);
    }
    if credit_enabled.w {
        let scale = learning_rate * credits.w * ENCODER_CREDIT_SCALE;
        weight.w = clamp(weight.w + scale * input, -2.0, 2.0);
    }
    packed_encoder.weights[address] = weight;
    // The scalar matrix remains authoritative for readback and every fallback
    // schedule. Each invocation owns these same four scalar locations.
    let scalar_address = agent_id * BRAIN_STRIDE + O_ENC_WEIGHTS
        + weight_vector * PACKED_ENCODER_WIDTH;
    brain_state[scalar_address] = weight.x;
    brain_state[scalar_address + 1u] = weight.y;
    brain_state[scalar_address + 2u] = weight.z;
    brain_state[scalar_address + 3u] = weight.w;
}
