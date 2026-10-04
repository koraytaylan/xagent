// Phase: build agent grid.
// Ported from agent_grid_build.wgsl. 1 thread per agent, tid = agent index.

fn phase_agent_grid(tid: u32) {
    let agent = tid;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    if agent >= agent_count { return; }

    let b = agent * PHYS_STRIDE;

    // Skip dead agents
    if physics_state[b + P_ALIVE] < 0.5 { return; }

    // Read position
    let px = physics_state[b + P_POS_X];
    let pz = physics_state[b + P_POS_Z];

    // Compute cell with offset
    let grid_offset = i32(wc_u32(WC_GRID_OFFSET));
    let grid_w = wc_u32(WC_GRID_WIDTH);
    let cx = cell_coord(px) + grid_offset;
    let cz = cell_coord(pz) + grid_offset;

    // Bounds check
    if cx < 0 || cz < 0 { return; }
    let ucx = u32(cx);
    let ucz = u32(cz);
    if ucx >= grid_w || ucz >= grid_w { return; }

    // Cell index and base in flat grid buffer
    let cell_idx = ucx * grid_w + ucz;
    let cell_base = cell_idx * AGENT_GRID_CELL_STRIDE;

    // Atomically claim a slot (index 0 is the count)
    let slot = atomicAdd(&agent_grid[cell_base], 1u);
    if slot < AGENT_GRID_MAX_PER_CELL {
        atomicStore(&agent_grid[cell_base + 1u + slot], agent);
        insert_agent_visibility_mask(agent, cx, cz, grid_w);
    }
}

// A retained registration is visible from each neighboring centre cell.
// Keeping registered coordinates preserves candidate membership after
// collisions move the agent; current positions remain the hit-test inputs.
fn insert_agent_visibility_mask(agent_id: u32, cell_x: i32, cell_z: i32, grid_width: u32) {
    if !VISION_AGENT_MASKS || wc_u32(WC_AGENT_COUNT) > AGENT_VISIBILITY_MASK_BITS {
        return;
    }
    let mask_offset = agent_visibility_mask_offset(grid_width);
    let agent_bit = 1u << agent_id;
    for (var offset_x: i32 = -1; offset_x <= 1; offset_x++) {
        for (var offset_z: i32 = -1; offset_z <= 1; offset_z++) {
            let center_x = cell_x + offset_x;
            let center_z = cell_z + offset_z;
            if center_x < 0 || center_z < 0 { continue; }
            let grid_x = u32(center_x);
            let grid_z = u32(center_z);
            if grid_x >= grid_width || grid_z >= grid_width { continue; }
            let center_cell = grid_x * grid_width + grid_z;
            atomicOr(&agent_grid[mask_offset + center_cell], agent_bit);
        }
    }
}
