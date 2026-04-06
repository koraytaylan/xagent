// Phase: build agent grid.
// Ported from agent_grid_build.wgsl. 1 thread per agent, tid = agent index.

fn phase_agent_grid(tid: u32) {
    let agent = tid;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    if agent >= agent_count { return; }

    let b = agent * PHYS_STRIDE;

    // Skip dead agents
    if agent_phys[b + P_ALIVE] < 0.5 { return; }

    // Read position
    let px = agent_phys[b + P_POS_X];
    let pz = agent_phys[b + P_POS_Z];

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
    }
}
