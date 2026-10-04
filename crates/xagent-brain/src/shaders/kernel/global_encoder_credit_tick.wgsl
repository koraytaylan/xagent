// Encoder credit is independent of the world update. Workgroup zero executes
// the original global phases; the remaining workgroups each update at most
// POINTWISE_CREDIT_THREADS weights for one agent. The preceding main dispatch
// has published that agent's adapted features and decision credit. The next
// dispatch waits for both branches through its storage dependency.
const GLOBAL_WORLD_WORKGROUPS: u32 = 1u;
override GLOBAL_CREDIT_GROUPS_PER_AGENT: u32 =
    (FEATURE_COUNT * ENCODED_DIMENSION + POINTWISE_CREDIT_THREADS - 1u)
    / POINTWISE_CREDIT_THREADS;

@compute @workgroup_size(POINTWISE_CREDIT_THREADS)
fn global_encoder_credit_tick(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    if wgid.x < GLOBAL_WORLD_WORKGROUPS {
        global_world_inner(lid.x);
    } else {
        let credit_group = wgid.x - GLOBAL_WORLD_WORKGROUPS;
        let agent_id = credit_group / GLOBAL_CREDIT_GROUPS_PER_AGENT;
        let tile = credit_group % GLOBAL_CREDIT_GROUPS_PER_AGENT;
        // The helper retains its own short-tile and inactive-agent guards.
        // Its writes are disjoint from the world branch and all other tiles.
        exact_pointwise_encoder_credit(vec3<u32>(agent_id, tile, 0u), lid);
    }
}
