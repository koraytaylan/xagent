// Phase: clear grids and collision scratch.
// All 256 threads cooperate to zero food_grid, agent_grid, collision_scratch.

fn phase_clear(tid: u32) {
    let grid_w = wc_u32(WC_GRID_WIDTH);
    let total_cells = grid_w * grid_w;

    // Zero food_grid: total_cells * FOOD_GRID_CELL_STRIDE elements
    let food_grid_size = total_cells * FOOD_GRID_CELL_STRIDE;
    for (var i = tid; i < food_grid_size; i += 256u) {
        atomicStore(&food_grid[i], 0u);
    }

    // Zero agent_grid: total_cells * AGENT_GRID_CELL_STRIDE elements
    let agent_grid_size = total_cells * AGENT_GRID_CELL_STRIDE;
    for (var i = tid; i < agent_grid_size; i += 256u) {
        atomicStore(&agent_grid[i], 0u);
    }

    // Zero collision_scratch: agent_count * 3 elements
    let agent_count = wc_u32(WC_AGENT_COUNT);
    let scratch_size = agent_count * 3u;
    for (var i = tid; i < scratch_size; i += 256u) {
        atomicStore(&collision_scratch[i], 0);
    }
}
