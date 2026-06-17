// ── Brain tick entry point ──────────────────────────────────────────────────
// Requires: common.wgsl + brain_passes.wgsl (concatenated by Rust host).
//
// dispatch(agent_count, 1, 1) — one workgroup per agent.
// 256 threads cooperate on the 7 brain passes defined in brain_passes.wgsl.
//
// Subgroup builtins are spliced in via the stable markers documented in
// `gpu_kernel.rs`. With subgroups: `sgid` is passed through to the top-K
// pass so a subgroup-shuffle bitonic sort (stages 0–4) can run barrier-free.
// Without subgroups: the markers collapse to no-ops and the top-K pass uses
// the shared-memory bitonic sort defined in brain_passes.wgsl.

@compute @workgroup_size(256)
fn brain_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
    // SUBGROUP_ENTRY_PARAMS
) {
    let agent_id = wgid.x;
    let tid = lid.x;

    // All threads check same agent — uniform control flow, safe for barriers
    if (physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }

    coop_feature_extract(agent_id, tid);
    workgroupBarrier();

    // Visual cortex (plan 0008): no-op passthrough until the DoG → Gabor →
    // complex stages land, gated on CFG_VISUAL_CORTEX_ENABLED. Kept in sync with
    // the fused kernel (kernel_tick.wgsl) so both pipelines run the same passes.
    coop_visual_cortex(agent_id, tid);
    workgroupBarrier();

    coop_encode(agent_id, tid);
    workgroupBarrier();

    coop_habituate_homeo(agent_id, tid);
    storageBarrier(); workgroupBarrier();

    coop_recall_score(agent_id, tid);
    workgroupBarrier();

    coop_recall_topk(agent_id, tid /* SUBGROUP_TOPK_ARGS */);
    storageBarrier(); workgroupBarrier();

    coop_predict_and_act(agent_id, tid, false);
    storageBarrier(); workgroupBarrier();

    coop_learn_and_store(agent_id, tid, true);
}
