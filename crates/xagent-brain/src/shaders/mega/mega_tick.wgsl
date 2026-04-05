// ── Mega-kernel: single dispatch, all cycles ─────────────────────────────
// One workgroup of 256 threads runs the entire simulation:
//   - Physics phases with barriers (BRAIN_TICK_STRIDE ticks per cycle)
//   - Cooperative vision (rays distributed across all 256 threads)
//   - Per-agent brain (7-pass neural network)
// Loops over all cycles internally — zero dispatch overhead.

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

// ── Main entry point ─────────────────────────────────────────────────────
@compute @workgroup_size(256)
fn mega_tick(@builtin(local_invocation_id) lid: vec3u) {
    let tid = lid.x;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    let num_cycles = pc.ticks_to_run / BRAIN_TICK_STRIDE;
    let remainder = pc.ticks_to_run % BRAIN_TICK_STRIDE;

    for (var cycle = 0u; cycle < num_cycles; cycle++) {
        let base_tick = pc.start_tick + cycle * BRAIN_TICK_STRIDE;

        // ── Physics: BRAIN_TICK_STRIDE ticks with barriers ───────────
        for (var t = 0u; t < BRAIN_TICK_STRIDE; t++) {
            run_physics_tick(tid, base_tick + t, agent_count);
        }

        // ── Cooperative vision: rays distributed across 256 threads ──
        let total_rays = agent_count * VISION_RAYS;
        let rays_per_thread = (total_rays + 255u) / 256u;
        for (var r = 0u; r < rays_per_thread; r++) {
            let ray_id = tid + r * 256u; // interleaved for cache locality
            if (ray_id < total_rays) {
                let agent_id = ray_id / VISION_RAYS;
                let ray_idx = ray_id % VISION_RAYS;
                vision_single_ray(agent_id, ray_idx);
            }
        }
        storageBarrier(); workgroupBarrier();

        // ── Brain: per-agent senses + 7-pass neural network ─────────
        if (tid < agent_count) {
            phase_vision_senses(tid);
            phase_brain(tid);
        }
        storageBarrier(); workgroupBarrier();
    }

    // ── Remainder physics-only ticks ─────────────────────────────────
    let rem_base = pc.start_tick + num_cycles * BRAIN_TICK_STRIDE;
    for (var t = 0u; t < remainder; t++) {
        run_physics_tick(tid, rem_base + t, agent_count);
    }
}
