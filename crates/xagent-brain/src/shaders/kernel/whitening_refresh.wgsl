// Test-only separation of the private Jacobi matrices from the cooperative
// brain. The host runs the normal kernel prefix before either entry point.
// That prefix has already settled food claims and completed death/respawn.

@compute @workgroup_size(1)
fn whitening_refresh(@builtin(workgroup_id) workgroup: vec3u) {
    let agent = workgroup.x;
    if agent >= wc_u32(WC_AGENT_COUNT) { return; }
    if !(physics_state[agent * PHYS_STRIDE + P_ALIVE] >= 0.5) { return; }
    if bc_f32(CFG_VISUAL_CORTEX_ENABLED) != 0.0 { return; }
    let brain_base = agent * BRAIN_STRIDE;
    let tick = brain_state[brain_base + O_TICK_COUNT];
    if u32(tick) % VISION_WHITENING_REFRESH == 0u {
        refresh_vision_whitening(brain_base);
    }
}

// brain_passes is composed with only the scheduled refresh block removed.
// Every other operation, lane mapping and arithmetic expression is shared
// with the normal fused kernel through brain_tick_inner.
@compute @workgroup_size(256)
fn brain_without_whitening_refresh(
    @builtin(local_invocation_id) local: vec3u,
    @builtin(workgroup_id) workgroup: vec3u,
    // KERNEL_SUBGROUP_ENTRY_PARAMS
) {
    let agent = workgroup.x;
    let tid = local.x;
    if tid == 0u {
        s_alive = select(0u, 1u, physics_state[agent * PHYS_STRIDE + P_ALIVE] >= 0.5);
    }
    workgroupBarrier();
    brain_tick_inner(agent, tid /* KERNEL_SUBGROUP_TOPK_INNER_ARGS */);
}
