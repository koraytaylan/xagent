// The host supplies current encoded inputs and the previous training state.
// Both calls execute the actual update/dot implementation before context/tanh.
@compute @workgroup_size(BRAIN_WORKGROUP_SIZE)
fn packed_predictor_dot_probe(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let agent_id = group.x;
    let scratch_base = agent_id * BRAIN_SCRATCH_STRIDE;
    if (tid < ENCODED_DIMENSION) {
        s_encoded[tid] = brain_scratch[scratch_base + SCRATCH_FEATURES + tid];
    }
    workgroupBarrier();
    // RAW_PREDICTOR_CALL
    workgroupBarrier();
    if (tid < PREDICTOR_DIMENSION) {
        brain_scratch[scratch_base + SCRATCH_PREDICTION + tid] = s_prediction[tid];
    }
}
