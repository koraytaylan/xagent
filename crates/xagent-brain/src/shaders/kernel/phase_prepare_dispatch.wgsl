// Phase: prepare indirect dispatch arguments.
// Reads agent_count from world config, writes dispatch triplets.
// dispatch_args layout: [vision_x, 1, 1, brain_x, 1, 1]; vision_x covers
// VISION_GROUPS_PER_AGENT workgroups per agent.

@compute @workgroup_size(1)
fn prepare_dispatch() {
    let agent_count = wc_u32(WC_AGENT_COUNT);
    // Vision: VISION_GROUPS_PER_AGENT workgroups per agent
    dispatch_args[0] = agent_count * VISION_GROUPS_PER_AGENT;
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
    // Brain: 1 workgroup per agent
    dispatch_args[3] = agent_count;
    dispatch_args[4] = 1u;
    dispatch_args[5] = 1u;
}
