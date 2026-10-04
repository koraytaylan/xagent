// Prefix completion publishes alive, interoception and the saved pre-collision
// position. Brain reads those stable fields while world changes only position,
// food, grids and trails. Branches are uniform across each complete workgroup.
@compute @workgroup_size(128)
fn four_stage_brain_world(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    // FOUR_STAGE_SUBGROUP_PARAMS
) {
    let agent_count = wc_u32(WC_AGENT_COUNT);
    if group.x < agent_count {
        let agent_id = group.x;
        if tid == 0u {
            s_alive = select(0u, 1u,
                physics_state[agent_id * PHYS_STRIDE + P_ALIVE] >= 0.5);
        }
        workgroupBarrier();
        brain_tick_inner(agent_id, tid /* FOUR_STAGE_SUBGROUP_ARGS */);
    } else {
        let stride = wc_u32(WC_BRAIN_TICK_STRIDE);
        four_stage_world(tid, kpc.start_tick + stride, stride);
    }
}
