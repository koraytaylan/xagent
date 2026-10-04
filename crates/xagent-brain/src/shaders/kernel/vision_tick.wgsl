// ── Vision dispatch: VISION_GROUPS_PER_AGENT workgroups per agent ───────────
// dispatch(agent_count * VISION_GROUPS_PER_AGENT, 1, 1) — each workgroup casts
// VISION_RAYS_PER_WORKGROUP rays, using either one serial invocation per ray
// or 32 invocations evaluating its samples in parallel. Thread 0 of an agent's
// first workgroup packs proprioception / interoception / touch / smell into
// sensory_buffer. The senses write only non-visual slots and read nothing the
// rays write, so they need no additional barrier after the ray reduction.

@compute @workgroup_size(VISION_WORKGROUP_SIZE)
fn vision_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
) {
    let agent_id = wgid.x / VISION_GROUPS_PER_AGENT;
    let group = wgid.x % VISION_GROUPS_PER_AGENT;

    // BEGIN_VISION_OBJECT_QUERIES
    if (VISION_OBJECT_QUERIES) {
        vision_object_rays(agent_id, group * VISION_RAYS_PER_WORKGROUP, lid.x);
    } else
    // END_VISION_OBJECT_QUERIES
    if (VISION_PARALLEL_STEPS) {
        vision_parallel_rays(agent_id, group * VISION_RAYS_PER_WORKGROUP, lid.x);
    } else {
        let ray = group * VISION_RAYS_PER_WORKGROUP + lid.x;
        if (ray < VISION_RAYS) {
            vision_single_ray(agent_id, ray);
        }
    }

    // BEGIN_VISION_PARALLEL_SCENT
    if (VISION_PARALLEL_SCENT && group == 0u) {
        vision_prepare_scent(agent_id, lid.x, VISION_WORKGROUP_SIZE);
    }
    // END_VISION_PARALLEL_SCENT
    if (group == 0u && lid.x == 0u) {
        phase_vision_senses(agent_id);
    }
}
