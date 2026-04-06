// ── Physics + Vision dispatch ───────────────────────────────────────────────
// Single workgroup of 256 threads. Runs N physics ticks then cooperative vision.
// Brain runs as a separate multi-workgroup dispatch.
// Phase mask: bit 0 = physics, bit 1 = vision.

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
fn physics_vision_tick(@builtin(local_invocation_id) lid: vec3u) {
    let tid = lid.x;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    let mask = wc_u32(WC_PHASE_MASK);

    // Physics ticks
    for (var t = 0u; t < pc.ticks_to_run; t = t + 1u) {
        if ((mask & 1u) != 0u) {
            run_physics_tick(tid, pc.start_tick + t, agent_count);
        } else {
            run_empty_tick(tid);
        }
    }

    // Cooperative vision + senses (once per batch)
    if ((mask & 2u) != 0u) {
        let total_rays = agent_count * VISION_RAYS;
        let rays_per_thread = (total_rays + 255u) / 256u;
        for (var r = 0u; r < rays_per_thread; r = r + 1u) {
            let ray_id = tid + r * 256u;
            if (ray_id < total_rays) {
                let agent_id = ray_id / VISION_RAYS;
                let ray_idx = ray_id % VISION_RAYS;
                vision_single_ray(agent_id, ray_idx);
            }
        }
        storageBarrier(); workgroupBarrier();

        if (tid < agent_count) {
            phase_vision_senses(tid);
        }
        storageBarrier(); workgroupBarrier();
    }
}
