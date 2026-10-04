// Only the extracted production predictor prefix is reachable. Its training
// inputs are zero, so the original train/clamp/store expression leaves each
// supplied in-range weight unchanged. Raw row sums are published before any
// recalled-context addition or tanh evaluation.
@compute @workgroup_size(BRAIN_WORKGROUP_SIZE)
fn predictor_dot_probe(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let agent_id = workgroup.x;
    let scratch_base = agent_id * BRAIN_SCRATCH_STRIDE;
    if (tid < ENCODED_DIMENSION) {
        s_encoded[tid] = brain_scratch[scratch_base + SCRATCH_FEATURES + tid];
    }
    workgroupBarrier();
    coop_predict_and_act(agent_id, tid, false);
    workgroupBarrier();
    if (tid < PREDICTOR_DIMENSION) {
        brain_scratch[scratch_base + SCRATCH_PREDICTION + tid] = s_prediction[tid];
    }
}
