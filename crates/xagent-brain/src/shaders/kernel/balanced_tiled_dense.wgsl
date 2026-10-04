// Test-only FP32 dense stages with wider, balanced inner reductions.
// Each workgroup owns complete output rows, and each predictor weight has
// one writer. The feature/weight formulas and clamps are unchanged; only
// the association of encoder and predictor dot-product additions differs.

override BALANCED_TILE_OUTPUTS: u32 = 16u;
override BALANCED_INNER_LANES: u32 = 16u;
override BALANCED_TILE_THREADS: u32 = BALANCED_TILE_OUTPUTS * BALANCED_INNER_LANES;
var<workgroup> balanced_partials: array<f32, BALANCED_TILE_THREADS>;
// A binary reduction halves the active lanes at each barrier.
const BALANCED_REDUCTION_RADIX: u32 = 2u;
// Matches the original FP32 predictor weight clamp exactly.
const BALANCED_PREDICTOR_WEIGHT_LIMIT: f32 = 3.0;

// Both layouts use the same power-of-two tree. Encoder lanes are separated
// by an output tile; predictor lanes are adjacent within one output row.
fn balanced_dense_reduce(tid: u32, lane: u32, lane_stride: u32) {
    for (var width = BALANCED_INNER_LANES / BALANCED_REDUCTION_RADIX; width > 0u; width /= BALANCED_REDUCTION_RADIX) {
        if (lane < width) {
            balanced_partials[tid] += balanced_partials[tid + width * lane_stride];
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(BALANCED_TILE_THREADS)
fn balanced_tiled_encode(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let agent = wgid.x;
    // The preceding dispatch fixes liveness before any dense group starts.
    if (physics_state[agent * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }
    let tid = lid.x;
    let output = tid % BALANCED_TILE_OUTPUTS;
    let lane = tid / BALANCED_TILE_OUTPUTS;
    let dimension = wgid.y * BALANCED_TILE_OUTPUTS + output;
    let brain_base = agent * BRAIN_STRIDE;
    let scratch = agent * BRAIN_SCRATCH_STRIDE;

    var partial = 0.0;
    if (lane == 0u) {
        partial = brain_state[brain_base + O_ENC_BIASES + dimension];
    }
    for (var feature = lane; feature < FEATURE_COUNT; feature += BALANCED_INNER_LANES) {
        partial += brain_scratch[scratch + SCRATCH_FEATURES + feature]
            * brain_state[brain_base + O_ENC_WEIGHTS + feature * ENCODED_DIMENSION + dimension];
    }
    balanced_partials[tid] = partial;
    workgroupBarrier();
    balanced_dense_reduce(tid, lane, BALANCED_TILE_OUTPUTS);
    if (lane == 0u) {
        brain_scratch[scratch + SCRATCH_ENCODED + dimension] = fast_tanh(balanced_partials[tid]);
    }
}

@compute @workgroup_size(BALANCED_TILE_THREADS)
fn balanced_tiled_predictor(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let agent = wgid.x;
    if (physics_state[agent * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }
    let tid = lid.x;
    let output = tid / BALANCED_INNER_LANES;
    let lane = tid % BALANCED_INNER_LANES;
    let dimension = wgid.y * BALANCED_TILE_OUTPUTS + output;
    let brain_base = agent * BRAIN_STRIDE;
    let scratch = agent * BRAIN_SCRATCH_STRIDE;
    let learning_rate = bc_f32(CFG_LEARNING_RATE);
    let previous_prediction = brain_state[brain_base + O_PREV_PREDICTION + dimension];
    let transition_error = previous_prediction - brain_scratch[scratch + SCRATCH_ENCODED + dimension];
    let tanh_derivative = 1.0 - previous_prediction * previous_prediction;

    var partial = 0.0;
    for (var input = lane; input < ENCODED_DIMENSION; input += BALANCED_INNER_LANES) {
        let previous_input = brain_state[brain_base + O_PREV_ENCODED + input];
        let gradient = clamp(transition_error * tanh_derivative * previous_input, -1.0, 1.0);
        var weight = brain_state[brain_base + O_PREDICTOR_WEIGHTS + dimension * ENCODED_DIMENSION + input]
            - learning_rate * gradient;
        weight = clamp(weight, -BALANCED_PREDICTOR_WEIGHT_LIMIT, BALANCED_PREDICTOR_WEIGHT_LIMIT);
        brain_state[brain_base + O_PREDICTOR_WEIGHTS + dimension * ENCODED_DIMENSION + input] = weight;
        partial += brain_scratch[scratch + SCRATCH_ENCODED + input] * weight;
    }
    balanced_partials[tid] = partial;
    workgroupBarrier();
    balanced_dense_reduce(tid, lane, 1u);
    if (lane == 0u) {
        brain_scratch[scratch + SCRATCH_PREDICTION + dimension] = balanced_partials[tid];
    }
}
