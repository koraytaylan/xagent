// ── Entry points ───────────────────────────────────────────────────────────
// Three kernels dispatched in a cycle:
//   physics_tick  — single workgroup (256), N ticks of phases 0-9 with barriers
//   vision_tick   — multi-workgroup, 1 ray per thread, ceil(agents*48/256) WGs
//   brain_tick    — per-agent senses + 7-pass brain

// ── Physics mini-kernel ──────────────────────────────────────────────────
@compute @workgroup_size(256)
fn physics_tick(@builtin(local_invocation_id) lid: vec3u) {
    let tid = lid.x;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    let start_tick = wc_u32(WC_TICK);
    let ticks_to_run = wc_u32(WC_TICKS_TO_RUN);

    for (var t = 0u; t < ticks_to_run; t++) {
        let tick = start_tick + t;

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
}

// ── Vision kernel (multi-workgroup) ──────────────────────────────────────
@compute @workgroup_size(256)
fn vision_tick(@builtin(global_invocation_id) gid: vec3u) {
    let global_id = gid.x;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    let total_rays = agent_count * VISION_RAYS;
    if (global_id >= total_rays) { return; }

    let agent_id = global_id / VISION_RAYS;
    let ray_idx = global_id % VISION_RAYS;

    vision_single_ray(agent_id, ray_idx);
}

// ── Brain kernel ─────────────────────────────────────────────────────────
@compute @workgroup_size(256)
fn brain_tick(@builtin(global_invocation_id) gid: vec3u) {
    let tid = gid.x;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    if (tid >= agent_count) { return; }

    phase_vision_senses(tid);
    phase_brain(tid);
}
