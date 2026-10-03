// Phase: put every grid cell's entries in index order.
// The food grid (phase_food_grid, phase_food_respawn) and the agent grid
// (phase_agent_grid) are filled by threads claiming slots with atomicAdd, so
// the order of the entries within a cell depends on thread timing. Readers
// visit entries in slot order — touch reports its first contacts in that
// order, and the vision ray stops at the first hit — so that order reaches
// the brain and runs with the same seed diverged. Sorting each cell's entries
// by index after all insertions makes the grids, and every run, the same.
// Each thread sorts a share of the cells by insertion sort (cells hold a few
// entries at most). A cell that overflowed its slots still keeps whichever
// entries won their slots.

fn phase_sort_grid_cells(tid: u32) {
    let grid_w = wc_u32(WC_GRID_WIDTH);
    let total_cells = grid_w * grid_w;
    for (var cell = tid; cell < total_cells; cell += 256u) {
        let food_base = cell * FOOD_GRID_CELL_STRIDE;
        let food_count = min(atomicLoad(&food_grid[food_base]), FOOD_GRID_MAX_PER_CELL);
        for (var i = 1u; i < food_count; i++) {
            let key = atomicLoad(&food_grid[food_base + 1u + i]);
            var j = i;
            while (j > 0u && atomicLoad(&food_grid[food_base + j]) > key) {
                atomicStore(&food_grid[food_base + 1u + j], atomicLoad(&food_grid[food_base + j]));
                j--;
            }
            atomicStore(&food_grid[food_base + 1u + j], key);
        }

        let agent_base = cell * AGENT_GRID_CELL_STRIDE;
        let agent_count = min(atomicLoad(&agent_grid[agent_base]), AGENT_GRID_MAX_PER_CELL);
        for (var i = 1u; i < agent_count; i++) {
            let key = atomicLoad(&agent_grid[agent_base + 1u + i]);
            var j = i;
            while (j > 0u && atomicLoad(&agent_grid[agent_base + j]) > key) {
                atomicStore(&agent_grid[agent_base + 1u + j], atomicLoad(&agent_grid[agent_base + j]));
                j--;
            }
            atomicStore(&agent_grid[agent_base + 1u + j], key);
        }
    }
}
