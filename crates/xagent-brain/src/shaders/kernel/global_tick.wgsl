// ── Global pass: grid rebuild, food respawn, collisions ─────────────────────
// dispatch(1, 1, 1) — single workgroup of 256 threads.
// Runs once per vision_stride kernel cycles.
// Requires: common.wgsl + phase_clear + phase_food_grid + phase_food_respawn
//           + phase_agent_grid + phase_collision (concatenated by Rust).

struct GlobalPushConstants {
    tick: u32,
    _pad: u32,
}
var<push_constant> gpc: GlobalPushConstants;

@compute @workgroup_size(256)
fn global_tick(@builtin(local_invocation_id) lid: vec3u) {
    let tid = lid.x;
    let agent_count = wc_u32(WC_AGENT_COUNT);

    phase_clear(tid);
    storageBarrier(); workgroupBarrier();

    phase_food_grid(tid);
    storageBarrier(); workgroupBarrier();

    phase_food_respawn(tid, gpc.tick);
    storageBarrier(); workgroupBarrier();

    if (tid < agent_count) { phase_agent_grid(tid); }
    storageBarrier(); workgroupBarrier();

    for (var c = 0u; c < 3u; c++) {
        if (tid < agent_count) { phase_collision_accumulate(tid); }
        storageBarrier(); workgroupBarrier();
        if (tid < agent_count) { phase_collision_apply(tid); }
        storageBarrier(); workgroupBarrier();
    }
}
