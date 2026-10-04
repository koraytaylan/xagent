// Test-only cached vision beside the brain. Every workgroup executes exactly
// one branch, so vision may reuse brain scratch without any cross-branch
// synchronization. The host restricts this entry to cortex-off layouts whose
// feature vector contains at least VISION_SCENT_CHUNK_SIZE elements.

const CACHED_OBJECT_COMPONENTS: u32 = 4u;
const CACHED_OBJECT_SCALAR_CAPACITY: u32 =
    VISION_OBJECT_CACHE_SIZE * CACHED_OBJECT_COMPONENTS;
override CACHED_VISION_GROUPS_PER_AGENT: u32 =
    (VISION_RAYS + VISION_PARALLEL_MAX_RAYS_PER_WORKGROUP - 1u)
    / VISION_PARALLEL_MAX_RAYS_PER_WORKGROUP;

// The cortex is inactive in this fixture. Resizing its existing allocation
// supplies the complete object cache without another Metal threadgroup slot.
fn cached_object_load(slot: u32) -> vec4<f32> {
    let base = slot * CACHED_OBJECT_COMPONENTS;
    return vec4<f32>(s_visual[base], s_visual[base + 1u],
        s_visual[base + 2u], s_visual[base + 3u]);
}

fn cached_object_store(slot: u32, value: vec4<f32>) {
    let base = slot * CACHED_OBJECT_COMPONENTS;
    s_visual[base] = value.x;
    s_visual[base + 1u] = value.y;
    s_visual[base + 2u] = value.z;
    s_visual[base + 3u] = value.w;
}

@compute @workgroup_size(256)
fn cached_brain_vision_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
    // KERNEL_SUBGROUP_ENTRY_PARAMS
) {
    let tid = lid.x;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    if wgid.x < agent_count {
        let agent_id = wgid.x;
        if tid == 0u {
            s_alive = select(0u, 1u, physics_state[agent_id * PHYS_STRIDE + P_ALIVE] >= 0.5);
        }
        workgroupBarrier();
        brain_tick_inner(agent_id, tid /* KERNEL_SUBGROUP_TOPK_INNER_ARGS */);
    } else {
        let vision_group = wgid.x - agent_count;
        let agent_id = vision_group / CACHED_VISION_GROUPS_PER_AGENT;
        let group = vision_group % CACHED_VISION_GROUPS_PER_AGENT;
        vision_object_rays(agent_id,
            group * VISION_PARALLEL_MAX_RAYS_PER_WORKGROUP, tid);
        if group == 0u {
            // All invocations, including dead observers, reach every scent
            // barrier. Object writes finish before the disjoint scent reuse.
            vision_prepare_scent(agent_id, tid, BRAIN_WORKGROUP_SIZE);
            if tid == 0u {
                phase_vision_senses(agent_id);
            }
        }
    }
}

@compute @workgroup_size(256)
fn cached_sensory_publish(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
) {
    let base = wgid.x * SENSORY_STRIDE;
    for (var slot = lid.x; slot < SENSORY_STRIDE; slot += BRAIN_WORKGROUP_SIZE) {
        sensory_buffer[base + slot] = sensory_next[base + slot];
    }
}
