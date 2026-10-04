// Every feature occupies exactly 32 adjacent vectors, so one 256-thread
// credit workgroup owns eight complete features. The workgroup barrier below
// finishes all old-feature reads before its sole writer publishes predicted q.
var<workgroup> projection_credit_partials: array<vec4<f32>, ENCODER_CREDIT_THREADS>;

fn projection_credit_group(agent_id: u32, tile: u32, tid: u32) {
    let weight_vector = tile * ENCODER_CREDIT_THREADS + tid;
    let feature = weight_vector / PACKED_ENCODER_OUTPUT_VECTORS;
    let output_vector = tid % PACKED_ENCODER_OUTPUT_VECTORS;
    // This helper returns actual updated weights, including disabled and
    // unchanged vectors. Its early returns never bypass the outer barriers.
    let weight = projection_credit_weight(agent_id, weight_vector);
    var predicted = 0.0;
    if (feature < PROJECTION_VISUAL_COUNT) {
        predicted = sensory_buffer[agent_id * SENSORY_STRIDE + feature]
            - brain_state[agent_id * BRAIN_STRIDE + O_SENSORY_MEAN + feature];
    }
    projection_credit_partials[tid] = weight * predicted;
    storageBarrier();
    workgroupBarrier();
    if (feature < PROJECTION_VISUAL_COUNT && output_vector == 0u) {
        packed_encoder.scratch[agent_id * BRAIN_SCRATCH_STRIDE + SCRATCH_FEATURES + feature] = predicted;
    }
    // The same output vector appears in eight feature lanes. Reduce those
    // lanes in a fixed tree; no floating atomics or cross-group sums occur.
    if (tid < ENCODER_CREDIT_THREADS / 2u) {
        projection_credit_partials[tid] += projection_credit_partials[tid + ENCODER_CREDIT_THREADS / 2u];
    }
    workgroupBarrier();
    if (tid < ENCODER_CREDIT_THREADS / 4u) {
        projection_credit_partials[tid] += projection_credit_partials[tid + ENCODER_CREDIT_THREADS / 4u];
    }
    workgroupBarrier();
    if (tid < PACKED_ENCODER_OUTPUT_VECTORS && tile < PROJECTION_GROUP_COUNT) {
        let reduced = projection_credit_partials[tid]
            + projection_credit_partials[tid + PACKED_ENCODER_OUTPUT_VECTORS];
        let base = PROJECTION_STORAGE_BASE + agent_id * PROJECTION_AGENT_STRIDE
            + PROJECTION_PARTIAL_OFFSET + tile * ENCODED_DIMENSION + tid * PACKED_ENCODER_WIDTH;
        packed_encoder.scratch[base] = reduced.x;
        packed_encoder.scratch[base + 1u] = reduced.y;
        packed_encoder.scratch[base + 2u] = reduced.z;
        packed_encoder.scratch[base + 3u] = reduced.w;
    }
    if (tile == 0u && tid == 0u) {
        // The next main dispatch follows completion of every credit group.
        packed_encoder.scratch[PROJECTION_STORAGE_BASE + agent_id * PROJECTION_AGENT_STRIDE + PROJECTION_VALID_OFFSET] = 1.0;
        packed_encoder.scratch[PROJECTION_STORAGE_BASE + agent_id * PROJECTION_AGENT_STRIDE + PROJECTION_DEATH_OFFSET] = physics_state[agent_id * PHYS_STRIDE + P_DEATH_COUNT];
    }
}
