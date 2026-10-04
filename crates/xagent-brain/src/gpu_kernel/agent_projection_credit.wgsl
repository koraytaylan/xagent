// One workgroup owns all weights and predicted visual inputs for one agent.
// Staging q once ensures the dot and its later equality gate use identical bits.
var<workgroup> agent_projection_inputs: array<f32, PROJECTION_VISUAL_COUNT>;

fn projection_credit_agent(agent_id: u32, tid: u32) {
    let alive = physics_state[agent_id * PHYS_STRIDE + P_ALIVE] >= 0.5;
    let private_base = PROJECTION_STORAGE_BASE + agent_id * PROJECTION_AGENT_STRIDE;
    let brain_base = agent_id * BRAIN_STRIDE;
    for (var feature = tid; feature < PROJECTION_VISUAL_COUNT; feature += ENCODER_CREDIT_THREADS) {
        var predicted = 0.0;
        if (alive) {
            predicted = sensory_buffer[agent_id * SENSORY_STRIDE + feature]
                - brain_state[brain_base + O_SENSORY_MEAN + feature];
        }
        agent_projection_inputs[feature] = predicted;
    }
    workgroupBarrier();

    let output_vector = tid % PACKED_ENCODER_OUTPUT_VECTORS;
    let lane = tid / PACKED_ENCODER_OUTPUT_VECTORS;
    if (alive && lane < PACKED_ENCODER_INNER_LANES) {
        let dimension = output_vector * PACKED_ENCODER_WIDTH;
        var partial = vec4<f32>(0.0);
        if (lane == 0u) {
            let bias = brain_base + O_ENC_BIASES + dimension;
            partial = vec4<f32>(brain_state[bias], brain_state[bias + 1u], brain_state[bias + 2u], brain_state[bias + 3u]);
        }
        // One owner per whole vec4; each of the four feature lanes retains its
        // ascending sequence. Nonvisual weights are updated but not projected.
        for (var feature = lane; feature < FEATURE_COUNT; feature += PACKED_ENCODER_INNER_LANES) {
            let weight_vector = feature * PACKED_ENCODER_OUTPUT_VECTORS + output_vector;
            let weight = projection_credit_weight(agent_id, weight_vector);
            if (feature < PROJECTION_VISUAL_COUNT) {
                partial += agent_projection_inputs[feature] * weight;
            }
        }
        let base = private_base + PROJECTION_PARTIAL_OFFSET
            + lane * ENCODED_DIMENSION + dimension;
        packed_encoder.scratch[base] = partial.x;
        packed_encoder.scratch[base + 1u] = partial.y;
        packed_encoder.scratch[base + 2u] = partial.z;
        packed_encoder.scratch[base + 3u] = partial.w;
    }

    // Every old-feature read is complete before any owner replaces that
    // feature with q. Dead agents also reach both unconditional barriers.
    storageBarrier();
    workgroupBarrier();
    if (alive) {
        for (var feature = tid; feature < PROJECTION_VISUAL_COUNT; feature += ENCODER_CREDIT_THREADS) {
            packed_encoder.scratch[agent_id * BRAIN_SCRATCH_STRIDE + SCRATCH_FEATURES + feature]
                = agent_projection_inputs[feature];
        }
    }
    if (tid == 0u) {
        // The next main dispatch waits for this complete global dispatch.
        // Validity describes this agent only; no other workgroup publishes it.
        packed_encoder.scratch[private_base + PROJECTION_VALID_OFFSET] = select(0.0, 1.0, alive);
        packed_encoder.scratch[private_base + PROJECTION_DEATH_OFFSET]
            = physics_state[agent_id * PHYS_STRIDE + P_DEATH_COUNT];
    }
}
