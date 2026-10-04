// Each workgroup executes either the next cycle's original claim phase or
// this cycle's vision. Vision reads the immutable post-global physics copy;
// claim writes live physics and disjoint atomic food-claim slots. Consumed
// food flags, food positions, grids, and brain genes remain unchanged here.
@compute @workgroup_size(256)
fn vision_claim_tick(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let agents = wc_u32(WC_AGENT_COUNT);
    if (group.x < agents) {
        vision_claim_next(group.x, local.x);
    } else {
        vision_claim_render(group.x - agents, local.x);
    }
}
