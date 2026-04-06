// ── Vision dispatch: multi-workgroup, one workgroup per agent ───────────────
// dispatch(agent_count, 1, 1) — threads 0..47 each cast one ray,
// then thread 0 packs proprioception/interoception/touch into sensory_buf.

@compute @workgroup_size(256)
fn vision_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
) {
    let agent_id = wgid.x;
    let tid = lid.x;

    if (agent_phys[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }

    // Each thread casts one ray (VISION_RAYS = 48, threads 0..47 active)
    if (tid < VISION_RAYS) {
        vision_single_ray(agent_id, tid);
    }
    storageBarrier(); workgroupBarrier();

    // Thread 0 packs sensory data (proprioception, interoception, touch)
    if (tid == 0u) {
        phase_vision_senses(agent_id);
    }
}
