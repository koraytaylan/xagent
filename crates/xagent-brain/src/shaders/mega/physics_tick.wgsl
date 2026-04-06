// ── Physics dispatch ────────────────────────────────────────────────────────
// Single workgroup of 256 threads. Runs N physics ticks.
// Vision and brain run as separate multi-workgroup dispatches.
// Phase mask bit 0 = physics.

struct PushConstants {
    start_tick: u32,
    ticks_to_run: u32,
}
var<push_constant> pc: PushConstants;

fn run_physics_tick(tid: u32, tick: u32, agent_count: u32) {
    phase_clear(tid);
    storageBarrier(); workgroupBarrier();

    phase_food_grid(tid);
    storageBarrier(); workgroupBarrier();

    if (tid < agent_count) { phase_physics(tid, tick); }
    storageBarrier(); workgroupBarrier();

    if (tid < agent_count) { phase_death_respawn(tid, tick); }
    storageBarrier(); workgroupBarrier();

    if (tid < agent_count) { phase_food_detect(tid); }
    storageBarrier(); workgroupBarrier();

    phase_food_respawn(tid, tick);
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

fn run_empty_tick(tid: u32) {
    storageBarrier(); workgroupBarrier();
    storageBarrier(); workgroupBarrier();
    storageBarrier(); workgroupBarrier();
    storageBarrier(); workgroupBarrier();
    storageBarrier(); workgroupBarrier();
    storageBarrier(); workgroupBarrier();
    storageBarrier(); workgroupBarrier();
    for (var c = 0u; c < 3u; c++) {
        storageBarrier(); workgroupBarrier();
        storageBarrier(); workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn physics_tick(@builtin(local_invocation_id) lid: vec3u) {
    let tid = lid.x;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    let mask = wc_u32(WC_PHASE_MASK);

    for (var t = 0u; t < pc.ticks_to_run; t = t + 1u) {
        if ((mask & 1u) != 0u) {
            run_physics_tick(tid, pc.start_tick + t, agent_count);
        } else {
            run_empty_tick(tid);
        }
    }
}
