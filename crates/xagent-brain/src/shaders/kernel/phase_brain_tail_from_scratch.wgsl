// ParallelTiled brain tail: remaining brain passes after tiled feature extract.
//
// One workgroup per agent (workgroup_size 256). feature_extract (pass 0) and
// encode (pass 1) have already run as separate dispatches writing SCRATCH_*; this
// tail loads SCRATCH_ENCODED into the workgroup s_encoded array and then runs the
// remaining cooperative passes 2..6 exactly as the fused brain_tick does —
// habituate_homeo, recall_score, recall_topk, predict_and_act, and
// learn_and_store — but with encoder-credit (7b) SKIPPED, because the tiled
// encoder-credit phase performs it. None of these passes read s_features, so only
// s_encoded must be reloaded. Concatenated with common.wgsl + brain_passes.wgsl.
//
// SAFETY: `alive` is broadcast by thread 0 into the workgroup-shared s_alive_tail
// before the first barrier, so `alive && ...` is workgroup-uniform and the
// cooperative passes' internal barriers are reached all-or-none. Inter-pass
// barriers stay UNCONDITIONAL (outside the alive guards), preserving barrier
// uniformity — identical to brain_tick_inner.

var<workgroup> s_alive_tail: u32;

@compute @workgroup_size(256)
fn phase_brain_tail_from_scratch(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    // SUBGROUP_ENTRY_PARAMS
) {
    let agent_id = wgid.x;
    let tid = lid.x;

    if (tid == 0u) {
        s_alive_tail = select(0u, 1u, physics_state[agent_id * PHYS_STRIDE + P_ALIVE] >= 0.5);
    }
    workgroupBarrier();
    let alive = s_alive_tail != 0u;

    // Load the tiled-encode result from scratch into the workgroup array the
    // cooperative passes read.
    let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;
    for (var i = tid; i < ENCODED_DIMENSION; i += 256u) {
        s_encoded[i] = brain_scratch[agent_scratch + SCRATCH_ENCODED + i];
    }
    workgroupBarrier();

    if (alive) { coop_habituate_homeo(agent_id, tid); }
    storageBarrier();
    workgroupBarrier();

    if (alive) { coop_recall_score(agent_id, tid); }
    workgroupBarrier();

    if (alive) { coop_recall_topk(agent_id, tid /* SUBGROUP_TOPK_ARGS */); }
    storageBarrier();
    workgroupBarrier();

    if (alive) { coop_predict_and_act(agent_id, tid, true); }
    storageBarrier();
    workgroupBarrier();

    if (alive) { coop_learn_and_store(agent_id, tid, false); }
}
