// ── Mega-kernel entry point ─────────────────────────────────────────────────
// Single workgroup (256 threads) runs N ticks of the full simulation loop.
// All phase functions are defined in the fragment files concatenated before this.

@compute @workgroup_size(256)
fn mega_tick(@builtin(local_invocation_id) lid: vec3u) {
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

        if (tick % 4u == 0u) {
            phase_vision_rays(tid);   // cooperative — all 256 threads share rays
            storageBarrier(); workgroupBarrier();
            if (tid < agent_count) {
                phase_vision_senses(tid); // per-agent proprioception/touch
                phase_brain(tid);
            }
        }
        storageBarrier(); workgroupBarrier();
    }
}
