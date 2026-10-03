// ── Brain beside vision (fused path, vision_stride 1) ──────────────────────
// Within a cycle the brain reads the sensory buffer the previous cycle's
// vision wrote, and vision reads nothing the brain writes (positions, grids,
// food and the static field-of-view and smell genes; the brain only writes
// its own state, the decision buffer and the P_*_OUT telemetry slots). So
// after the kernel prefix (physics, food and danger detection, death/respawn)
// and the global pass, one dispatch runs both: workgroups [0, agent_count)
// run each agent's brain, and the next agent_count *
// COMBINED_VISION_GROUPS_PER_AGENT workgroups cast its rays and pack its
// senses. Two things keep the results those of the serial order:
//
//   * The brain's staleness ring reads P_BRAIN_POS_X/Z, the position the
//     prefix saved before the global pass's collisions moved the agents.
//   * Vision writes `sensory_next`, not the buffer the brain is reading, and
//     `sensory_publish` copies it over once both are done, as the serial
//     vision pass would have left it.

override COMBINED_VISION_GROUPS_PER_AGENT: u32 =
    (VISION_RAYS + BRAIN_WORKGROUP_SIZE - 1u) / BRAIN_WORKGROUP_SIZE;

@compute @workgroup_size(256)
fn brain_vision_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
    // KERNEL_SUBGROUP_ENTRY_PARAMS
) {
    let tid = lid.x;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    if (wgid.x < agent_count) {
        // Brain workgroup. `wgid` and the uniform agent count make this branch
        // workgroup-uniform, so the passes' barriers are reached by all
        // threads. Nothing between death/respawn and here changes P_ALIVE.
        let agent_id = wgid.x;
        if (tid == 0u) {
            s_alive = select(0u, 1u, physics_state[agent_id * PHYS_STRIDE + P_ALIVE] >= 0.5);
        }
        workgroupBarrier();
        brain_tick_inner(agent_id, tid /* KERNEL_SUBGROUP_TOPK_INNER_ARGS */);
    } else {
        // Vision workgroup: one thread per ray, as the vision pass casts them.
        let vision_group = wgid.x - agent_count;
        let agent_id = vision_group / COMBINED_VISION_GROUPS_PER_AGENT;
        let group = vision_group % COMBINED_VISION_GROUPS_PER_AGENT;
        if (physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }
        let ray = group * BRAIN_WORKGROUP_SIZE + tid;
        if (ray < VISION_RAYS) {
            vision_single_ray(agent_id, ray);
        }
        if (group == 0u && tid == 0u) {
            phase_vision_senses(agent_id);
        }
    }
}

// Copy each agent's freshly written senses from `sensory_next` into the
// sensory buffer the brain and telemetry read. Dispatched once per cycle
// after `brain_vision_tick`, one workgroup per agent.
@compute @workgroup_size(256)
fn sensory_publish(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
) {
    let base = wgid.x * SENSORY_STRIDE;
    for (var i = lid.x; i < SENSORY_STRIDE; i += BRAIN_WORKGROUP_SIZE) {
        sensory_buffer[base + i] = sensory_next[base + i];
    }
}
