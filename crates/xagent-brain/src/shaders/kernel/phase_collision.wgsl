// Phase: collision detection and resolution.
// Ported from collision_accumulate.wgsl and collision_apply.wgsl.
// 1 thread per agent, tid = agent index.

fn phase_collision_accumulate(tid: u32) {
    let agent = tid;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    if agent >= agent_count { return; }

    let b = agent * PHYS_STRIDE;

    // Skip dead agents
    if physics_state[b + P_ALIVE] < 0.5 { return; }

    // Read self position
    let self_x = physics_state[b + P_POS_X];
    let self_y = physics_state[b + P_POS_Y];
    let self_z = physics_state[b + P_POS_Z];

    // Compute cell with offset
    let grid_offset = i32(wc_u32(WC_GRID_OFFSET));
    let grid_w = wc_u32(WC_GRID_WIDTH);
    let self_cx = cell_coord(self_x) + grid_offset;
    let self_cz = cell_coord(self_z) + grid_offset;

    // Iterate 3x3 neighborhood
    for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let cx = self_cx + dx;
            let cz = self_cz + dz;

            // Bounds check
            if cx < 0 || cz < 0 { continue; }
            let ucx = u32(cx);
            let ucz = u32(cz);
            if ucx >= grid_w || ucz >= grid_w { continue; }

            let cell_idx = ucx * grid_w + ucz;
            let cell_base = cell_idx * AGENT_GRID_CELL_STRIDE;
            let count = atomicLoad(&agent_grid[cell_base]);
            let clamped_count = min(count, AGENT_GRID_MAX_PER_CELL);

            for (var s: u32 = 0u; s < clamped_count; s++) {
                let other = atomicLoad(&agent_grid[cell_base + 1u + s]);

                // Only process pair once (lower index pushes higher index)
                if other <= agent { continue; }

                let ob = other * PHYS_STRIDE;

                // Skip dead other
                if physics_state[ob + P_ALIVE] < 0.5 { continue; }

                // Compute displacement from self to other
                let diff_x = physics_state[ob + P_POS_X] - self_x;
                let diff_y = physics_state[ob + P_POS_Y] - self_y;
                let diff_z = physics_state[ob + P_POS_Z] - self_z;
                let dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

                if dist_sq < COLLISION_MIN_DIST_SQ && dist_sq > 0.001 {
                    let dist = sqrt(dist_sq);
                    let overlap = COLLISION_MIN_DIST - dist;
                    let inv_dist = 1.0 / dist;
                    let push_x = diff_x * inv_dist * overlap * 0.5;
                    let push_y = diff_y * inv_dist * overlap * 0.5;
                    let push_z = diff_z * inv_dist * overlap * 0.5;

                    // Fixed-point encode
                    let ipx = i32(push_x * COLLISION_FIXED_SCALE);
                    let ipy = i32(push_y * COLLISION_FIXED_SCALE);
                    let ipz = i32(push_z * COLLISION_FIXED_SCALE);

                    // Self pushed away from other (negative push)
                    atomicAdd(&collision_scratch[agent * 3u],      -ipx);
                    atomicAdd(&collision_scratch[agent * 3u + 1u], -ipy);
                    atomicAdd(&collision_scratch[agent * 3u + 2u], -ipz);

                    // Other pushed toward self (positive push)
                    atomicAdd(&collision_scratch[other * 3u],      ipx);
                    atomicAdd(&collision_scratch[other * 3u + 1u], ipy);
                    atomicAdd(&collision_scratch[other * 3u + 2u], ipz);
                }
            }
        }
    }
}

fn phase_collision_apply(tid: u32) {
    let agent = tid;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    if agent >= agent_count { return; }

    // Read and clear scratch atomically (keeps scratch clean even for dead agents)
    let ipx = atomicExchange(&collision_scratch[agent * 3u],      0);
    let ipy = atomicExchange(&collision_scratch[agent * 3u + 1u], 0);
    let ipz = atomicExchange(&collision_scratch[agent * 3u + 2u], 0);

    let b = agent * PHYS_STRIDE;

    // Skip dead agents (but scratch was already cleared above)
    if physics_state[b + P_ALIVE] < 0.5 { return; }

    // Apply accumulated push
    physics_state[b + P_POS_X] += f32(ipx) / COLLISION_FIXED_SCALE;
    physics_state[b + P_POS_Y] += f32(ipy) / COLLISION_FIXED_SCALE;
    physics_state[b + P_POS_Z] += f32(ipz) / COLLISION_FIXED_SCALE;
}
