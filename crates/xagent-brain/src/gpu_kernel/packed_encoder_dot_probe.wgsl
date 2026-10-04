// Feed the actual encoder function, then publish its raw bias-plus-dot sums.
// The host removes only the terminal tanh calls from that function; every
// accumulation, feature guard, lane mapping and reduction remains unchanged.
@compute @workgroup_size(BRAIN_WORKGROUP_SIZE)
fn packed_encoder_dot_probe(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let agent_id = group.x;
    let scratch_base = agent_id * BRAIN_SCRATCH_STRIDE;
    for (var feature = tid; feature < FEATURE_COUNT; feature += BRAIN_WORKGROUP_SIZE) {
        s_features[feature] = brain_scratch[scratch_base + SCRATCH_FEATURES + feature];
    }
    workgroupBarrier();
    coop_encode(agent_id, tid);
    workgroupBarrier();
    if (tid < ENCODED_DIMENSION) {
        brain_scratch[scratch_base + SCRATCH_ENCODED + tid] = s_encoded[tid];
    }
}
