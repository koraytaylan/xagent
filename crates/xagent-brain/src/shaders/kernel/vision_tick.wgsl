// ── Vision dispatch: multi-workgroup, one workgroup per agent ───────────────
// dispatch(agent_count, 1, 1) — each workgroup's WORKGROUP_SIZE threads
// cooperatively cast all VISION_RAYS rays (looping when VISION_RAYS >
// WORKGROUP_SIZE), then thread 0 packs proprioception/interoception/touch
// into sensory_buffer.

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn vision_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
) {
    let agent_id = wgid.x;
    let tid = lid.x;

    if (physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }

    // Each thread casts rays in a strided loop (handles VISION_RAYS > WORKGROUP_SIZE)
    for (var ray = tid; ray < VISION_RAYS; ray += WORKGROUP_SIZE) {
        vision_single_ray(agent_id, ray);
    }
    storageBarrier(); workgroupBarrier();

    // Thread 0 packs sensory data (proprioception, interoception, touch)
    if (tid == 0u) {
        phase_vision_senses(agent_id);
    }
}
