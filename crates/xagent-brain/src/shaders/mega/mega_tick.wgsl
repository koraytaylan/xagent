// ── Mega-kernel: single dispatch, all cycles ─────────────────────────────
// One workgroup of 256 threads runs the entire simulation.
// Phase mask (WC_PHASE_MASK) controls which phases run:
//   bit 0 = physics+grids, bit 1 = vision, bit 2 = brain

const BRAIN_TICK_STRIDE: u32 = 4u;

struct PushConstants {
    start_tick: u32,
    ticks_to_run: u32,
}
var<push_constant> pc: PushConstants;

// ── Physics tick helper ──────────────────────────────────────────────────
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

// ── Barriers-only tick (same barrier count, no compute) ──────────────────
fn run_empty_tick(tid: u32) {
    storageBarrier(); workgroupBarrier(); // clear
    storageBarrier(); workgroupBarrier(); // food_grid
    storageBarrier(); workgroupBarrier(); // physics
    storageBarrier(); workgroupBarrier(); // death
    storageBarrier(); workgroupBarrier(); // food_detect
    storageBarrier(); workgroupBarrier(); // food_respawn
    storageBarrier(); workgroupBarrier(); // agent_grid
    for (var c = 0u; c < 3u; c++) {
        storageBarrier(); workgroupBarrier(); // collision_acc
        storageBarrier(); workgroupBarrier(); // collision_apply
    }
}

// ── Main entry point ─────────────────────────────────────────────────────
@compute @workgroup_size(256)
fn mega_tick(@builtin(local_invocation_id) lid: vec3u) {
    let tid = lid.x;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    let num_cycles = pc.ticks_to_run / BRAIN_TICK_STRIDE;
    let remainder = pc.ticks_to_run % BRAIN_TICK_STRIDE;
    let mask = wc_u32(WC_PHASE_MASK);

    for (var cycle = 0u; cycle < num_cycles; cycle++) {
        let base_tick = pc.start_tick + cycle * BRAIN_TICK_STRIDE;

        // ── Physics ticks ────────────────────────────────────────────
        for (var t = 0u; t < BRAIN_TICK_STRIDE; t++) {
            if ((mask & 1u) != 0u) {
                run_physics_tick(tid, base_tick + t, agent_count);
            } else {
                run_empty_tick(tid);
            }
        }

        // ── Cooperative vision ───────────────────────────────────────
        if ((mask & 2u) != 0u) {
            let total_rays = agent_count * VISION_RAYS;
            let rays_per_thread = (total_rays + 255u) / 256u;
            for (var r = 0u; r < rays_per_thread; r++) {
                let ray_id = tid + r * 256u;
                if (ray_id < total_rays) {
                    let agent_id = ray_id / VISION_RAYS;
                    let ray_idx = ray_id % VISION_RAYS;
                    vision_single_ray(agent_id, ray_idx);
                }
            }
        }
        storageBarrier(); workgroupBarrier();

        // ── Brain ────────────────────────────────────────────────────
        if ((mask & 4u) != 0u) {
            if (tid < agent_count) {
                phase_vision_senses(tid);
                phase_brain(tid);
            }
        }
        storageBarrier(); workgroupBarrier();
    }

    // ── Remainder physics-only ticks ─────────────────────────────────
    let rem_base = pc.start_tick + num_cycles * BRAIN_TICK_STRIDE;
    for (var t = 0u; t < remainder; t++) {
        if ((mask & 1u) != 0u) {
            run_physics_tick(tid, rem_base + t, agent_count);
        } else {
            run_empty_tick(tid);
        }
    }
}
