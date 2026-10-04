// Feature publication and cooperative tail for an exact tiled brain probe.
// Requires common.wgsl and the unchanged brain_passes.wgsl.

var<workgroup> exact_tiled_alive: u32;

@compute @workgroup_size(BRAIN_WORKGROUP_SIZE)
fn exact_tiled_features(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let agent_id = wgid.x;
    let tid = lid.x;
    if (tid == 0u) {
        exact_tiled_alive = select(0u, 1u, physics_state[agent_id * PHYS_STRIDE + P_ALIVE] >= 0.5);
    }
    workgroupBarrier();
    let alive = exact_tiled_alive != 0u;

    if (alive) { coop_feature_extract(agent_id, tid); }
    workgroupBarrier();
    if (alive) { coop_visual_cortex(agent_id, tid); }
    workgroupBarrier();
    if (alive) { coop_sensory_adapt(agent_id, tid); }
    workgroupBarrier();

    if (alive) {
        let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;
        for (var feature = tid; feature < FEATURE_COUNT; feature += BRAIN_WORKGROUP_SIZE) {
            brain_scratch[agent_scratch + SCRATCH_FEATURES + feature] = s_features[feature];
        }
    }
}

@compute @workgroup_size(BRAIN_WORKGROUP_SIZE)
fn exact_tiled_tail(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    // SUBGROUP_ENTRY_PARAMS
) {
    let agent_id = wgid.x;
    let tid = lid.x;
    if (tid == 0u) {
        exact_tiled_alive = select(0u, 1u, physics_state[agent_id * PHYS_STRIDE + P_ALIVE] >= 0.5);
    }
    workgroupBarrier();
    let alive = exact_tiled_alive != 0u;

    if (alive) {
        let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;
        // Scent and the visual pathway consume the adapted feature vector,
        // even when encoder credit is performed in a separate dispatch.
        for (var feature = tid; feature < FEATURE_COUNT; feature += BRAIN_WORKGROUP_SIZE) {
            s_features[feature] = brain_scratch[agent_scratch + SCRATCH_FEATURES + feature];
        }
        for (var dimension = tid; dimension < ENCODED_DIMENSION; dimension += BRAIN_WORKGROUP_SIZE) {
            s_encoded[dimension] = brain_scratch[agent_scratch + SCRATCH_ENCODED + dimension];
        }
    }
    workgroupBarrier();

    if (alive) { coop_habituate_homeo(agent_id, tid); }
    storageBarrier(); workgroupBarrier();
    if (alive) { coop_recall_score(agent_id, tid); }
    workgroupBarrier();
    if (alive) { coop_recall_topk(agent_id, tid /* SUBGROUP_TOPK_ARGS */); }
    storageBarrier(); workgroupBarrier();
    if (alive) { coop_predict_and_act(agent_id, tid, true); }
    storageBarrier(); workgroupBarrier();
    if (alive) { coop_learn_and_store(agent_id, tid, false); }
}
