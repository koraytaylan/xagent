// One workgroup owns the complete world while it advances whole brain cycles.
// Per-agent brain work runs in agent order, reusing the cooperative scratch.
// Global phases keep their existing order and every sensory frame is consumed
// by the following cycle, exactly as in the separate serial dispatch schedule.

// Three accumulated collision corrections match the global entry point.
const PERSISTENT_COLLISION_ITERATIONS: u32 = 3u;

fn persistent_global(tid: u32, tick: u32, trail_interval: u32) {
    let agent_count = wc_u32(WC_AGENT_COUNT);
    phase_clear(tid);
    storageBarrier(); workgroupBarrier();
    phase_food_grid(tid);
    storageBarrier(); workgroupBarrier();
    phase_food_respawn(tid, tick);
    storageBarrier(); workgroupBarrier();
    if tid < agent_count { phase_agent_grid(tid); }
    storageBarrier(); workgroupBarrier();
    phase_sort_grid_cells(tid);
    storageBarrier(); workgroupBarrier();
    for (var iteration = 0u; iteration < PERSISTENT_COLLISION_ITERATIONS; iteration++) {
        if tid < agent_count { phase_collision_accumulate(tid); }
        storageBarrier(); workgroupBarrier();
        if tid < agent_count { phase_collision_apply(tid); }
        storageBarrier(); workgroupBarrier();
    }
    if trail_interval != 0u && tick % trail_interval == 0u {
        phase_trail_sample(tid, tick / trail_interval);
    }
}

@compute @workgroup_size(256)
fn persistent_tick(
    @builtin(local_invocation_id) lid: vec3u,
    // KERNEL_SUBGROUP_ENTRY_PARAMS
) {
    let tid = lid.x;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    let stride = wc_u32(WC_BRAIN_TICK_STRIDE);
    let cycles = wc_u32(WC_TICKS_TO_RUN) / max(stride, 1u);
    for (var cycle = 0u; cycle < cycles; cycle++) {
        let cycle_tick = kpc.start_tick + cycle * stride;

        // Each agent's sub-ticks depend only on its own current state and the
        // previous cycle's decision; all physics writers finish before scans.
        if tid < agent_count {
            for (var step = 0u; step < stride; step++) {
                agent_physics(tid, cycle_tick + step);
            }
        }
        storageBarrier(); workgroupBarrier();

        // Every food scan observes the same consumed flags. Claims are settled
        // only after all agents have claimed, preserving the lowest-ID winner.
        for (var agent = 0u; agent < agent_count; agent++) {
            if tid == 0u {
                s_alive = select(0u, 1u, physics_state[agent * PHYS_STRIDE + P_ALIVE] >= 0.5);
            }
            workgroupBarrier();
            agent_food_detect(agent, tid);
            storageBarrier(); workgroupBarrier();
        }
        if tid < agent_count { resolve_food_claim(tid); }
        storageBarrier(); workgroupBarrier();

        // These phases read no mutable state belonging to another agent.
        // Uniform agent loops keep every invocation at each helper barrier.
        for (var agent = 0u; agent < agent_count; agent++) {
            if tid == 0u {
                s_alive = select(0u, 1u, physics_state[agent * PHYS_STRIDE + P_ALIVE] >= 0.5);
            }
            workgroupBarrier();
            agent_danger_detect(agent, tid);
            workgroupBarrier();
            if tid == 0u {
                let motor_turn = decision_buffer[agent * DECISION_STRIDE + DECISION_MOTOR + 1u];
                agent_avoidance_accumulate(agent, motor_turn);
                agent_approach_accumulate(agent, motor_turn);
                agent_death_respawn(agent, cycle_tick);
                let base = agent * PHYS_STRIDE;
                physics_state[base + P_BRAIN_POS_X] = physics_state[base + P_POS_X];
                physics_state[base + P_BRAIN_POS_Z] = physics_state[base + P_POS_Z];
                s_alive = select(0u, 1u, physics_state[base + P_ALIVE] >= 0.5);
            }
            storageBarrier(); workgroupBarrier();
            brain_tick_inner(agent, tid /* KERNEL_SUBGROUP_TOPK_INNER_ARGS */);
            storageBarrier(); workgroupBarrier();
        }

        persistent_global(tid, cycle_tick + stride, stride);
        // Ray outputs and nonvisual outputs occupy disjoint sensory slots.
        for (var ray = tid; ray < agent_count * VISION_RAYS; ray += BRAIN_WORKGROUP_SIZE) {
            vision_single_ray(ray / VISION_RAYS, ray % VISION_RAYS);
        }
        if tid < agent_count { phase_vision_senses(tid); }
        storageBarrier(); workgroupBarrier();
    }
}
