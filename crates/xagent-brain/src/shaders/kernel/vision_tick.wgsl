// ── Vision dispatch: VISION_GROUPS_PER_AGENT workgroups per agent ───────────
// dispatch(agent_count * VISION_GROUPS_PER_AGENT, 1, 1) — each workgroup of
// VISION_RAYS_PER_WORKGROUP threads casts one ray per thread, and thread 0 of
// an agent's first workgroup packs its proprioception / interoception / touch
// / smell into sensory_buffer. The senses write only the non-visual slots and
// read nothing the rays write, so they need no barrier after the rays.

@compute @workgroup_size(VISION_RAYS_PER_WORKGROUP)
fn vision_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
) {
    let agent_id = wgid.x / VISION_GROUPS_PER_AGENT;
    let group = wgid.x % VISION_GROUPS_PER_AGENT;

    if (physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }

    let ray = group * VISION_RAYS_PER_WORKGROUP + lid.x;
    if (ray < VISION_RAYS) {
        vision_single_ray(agent_id, ray);
    }

    if (group == 0u && lid.x == 0u) {
        phase_vision_senses(agent_id);
    }
}
