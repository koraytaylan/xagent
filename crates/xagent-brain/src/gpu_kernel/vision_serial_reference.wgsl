// Standalone serial reference excludes every cooperative fragment and its
// workgroup storage from both parity and timing baseline pipelines.

@compute @workgroup_size(VISION_RAYS_PER_WORKGROUP)
fn vision_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
) {
    let agent_id = wgid.x / VISION_GROUPS_PER_AGENT;
    let group = wgid.x % VISION_GROUPS_PER_AGENT;
    if physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5 { return; }

    let ray = group * VISION_RAYS_PER_WORKGROUP + lid.x;
    if ray < VISION_RAYS {
        vision_single_ray(agent_id, ray);
    }
    if group == 0u && lid.x == 0u {
        phase_vision_senses(agent_id);
    }
}
