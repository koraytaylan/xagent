// Dense stages preserve the fused brain's four logical accumulation lanes.
// Output tiles vary independently of that arithmetic; each weight has one
// writer and each output keeps its original stride-four reduction order.

override EXACT_TILE_OUTPUTS: u32 = 16u;
// Four partials are part of the fused arithmetic contract.
const EXACT_TILE_INNER_LANES: u32 = 4u;
override EXACT_TILE_THREADS: u32 = EXACT_TILE_OUTPUTS * EXACT_TILE_INNER_LANES;
var<workgroup> exact_tile_partials: array<f32, EXACT_TILE_THREADS>;

@compute @workgroup_size(EXACT_TILE_THREADS)
fn exact_tiled_encode(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let agent_id = wgid.x;
    // The preceding dispatch fixes liveness for the whole agent; every
    // invocation reads the same slot before reaching any barrier.
    if (physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }
    let tid = lid.x;
    let output_in_tile = tid % EXACT_TILE_OUTPUTS;
    let lane = tid / EXACT_TILE_OUTPUTS;
    let dim = wgid.y * EXACT_TILE_OUTPUTS + output_in_tile;
    let brain_base = agent_id * BRAIN_STRIDE;
    let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;

    var partial: f32 = 0.0;
    if (lane == 0u) {
        partial = brain_state[brain_base + O_ENC_BIASES + dim];
    }
    for (var feature = lane; feature < FEATURE_COUNT; feature += EXACT_TILE_INNER_LANES) {
        partial += brain_scratch[agent_scratch + SCRATCH_FEATURES + feature]
            * brain_state[brain_base + O_ENC_WEIGHTS + feature * ENCODED_DIMENSION + dim];
    }
    exact_tile_partials[tid] = partial;
    workgroupBarrier();
    if (lane == 0u) {
        let first = output_in_tile;
        let second = first + EXACT_TILE_OUTPUTS;
        let third = second + EXACT_TILE_OUTPUTS;
        let fourth = third + EXACT_TILE_OUTPUTS;
        let reduced = exact_tile_partials[first] + exact_tile_partials[second]
            + exact_tile_partials[third] + exact_tile_partials[fourth];
        brain_scratch[agent_scratch + SCRATCH_ENCODED + dim] = fast_tanh(reduced);
    }
}

@compute @workgroup_size(EXACT_TILE_THREADS)
fn exact_tiled_predictor(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let agent_id = wgid.x;
    if (physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }
    let tid = lid.x;
    let output_in_tile = tid / EXACT_TILE_INNER_LANES;
    let lane = tid % EXACT_TILE_INNER_LANES;
    let dim = wgid.y * EXACT_TILE_OUTPUTS + output_in_tile;
    let brain_base = agent_id * BRAIN_STRIDE;
    let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;
    let predictor_learning_rate = bc_f32(CFG_LEARNING_RATE);

    let previous_prediction = brain_state[brain_base + O_PREV_PREDICTION + dim];
    let transition_error = previous_prediction - brain_scratch[agent_scratch + SCRATCH_ENCODED + dim];
    let tanh_derivative = 1.0 - previous_prediction * previous_prediction;
    for (var j = lane; j < ENCODED_DIMENSION; j += EXACT_TILE_INNER_LANES) {
        let previous_input = brain_state[brain_base + O_PREV_ENCODED + j];
        let grad = clamp(transition_error * tanh_derivative * previous_input, -1.0, 1.0);
        var w = brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j]
            - predictor_learning_rate * grad;
        w = clamp(w, -3.0, 3.0);
        brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j] = w;
    }
    workgroupBarrier();

    var partial: f32 = 0.0;
    for (var j = lane; j < ENCODED_DIMENSION; j += EXACT_TILE_INNER_LANES) {
        partial += brain_scratch[agent_scratch + SCRATCH_ENCODED + j]
            * brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j];
    }
    exact_tile_partials[tid] = partial;
    workgroupBarrier();
    if (lane == 0u) {
        let first = tid;
        let second = first + 1u;
        let third = second + 1u;
        let fourth = third + 1u;
        let reduced = exact_tile_partials[first] + exact_tile_partials[second]
            + exact_tile_partials[third] + exact_tile_partials[fourth];
        brain_scratch[agent_scratch + SCRATCH_PREDICTION + dim] = reduced;
    }
}

@compute @workgroup_size(EXACT_TILE_THREADS)
fn exact_tiled_encoder_credit(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let agent_id = wgid.x;
    if (physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }
    let tid = lid.x;
    let output_in_tile = tid % EXACT_TILE_OUTPUTS;
    let lane = tid / EXACT_TILE_OUTPUTS;
    let dim = wgid.y * EXACT_TILE_OUTPUTS + output_in_tile;
    let brain_base = agent_id * BRAIN_STRIDE;
    let decision_base = agent_id * DECISION_STRIDE;
    let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;

    let action_credit = decision_buffer[decision_base + DECISION_CREDIT + dim];
    if (abs(action_credit) >= CREDIT_EPSILON) {
        let learning_rate = brain_config[1].x;
        let scale = learning_rate * action_credit * ENCODER_CREDIT_SCALE;
        for (var feature = lane; feature < FEATURE_COUNT; feature += EXACT_TILE_INNER_LANES) {
            var w = brain_state[brain_base + O_ENC_WEIGHTS + feature * ENCODED_DIMENSION + dim]
                + scale * brain_scratch[agent_scratch + SCRATCH_FEATURES + feature];
            w = clamp(w, -2.0, 2.0);
            brain_state[brain_base + O_ENC_WEIGHTS + feature * ENCODED_DIMENSION + dim] = w;
        }
    }
}
