// Phase: build food grid.
// Ported from food_grid_build.wgsl. All 256 threads cooperate to process all food items.

fn phase_food_grid(tid: u32) {
    let food_count = wc_u32(WC_FOOD_COUNT);
    let grid_offset = i32(wc_u32(WC_GRID_OFFSET));
    let grid_w = wc_u32(WC_GRID_WIDTH);

    for (var i = tid; i < food_count; i += 256u) {
        // Skip consumed food
        if atomicLoad(&food_flags[i]) != 0u { continue; }

        // Read position
        let base = i * FOOD_STATE_STRIDE;
        let fx = food_state[base + F_POS_X];
        let fz = food_state[base + F_POS_Z];

        // Compute grid cell with offset
        let cx = cell_coord(fx) + grid_offset;
        let cz = cell_coord(fz) + grid_offset;

        // Bounds check
        if cx < 0 || cz < 0 { continue; }
        let ucx = u32(cx);
        let ucz = u32(cz);
        if ucx >= grid_w || ucz >= grid_w { continue; }

        // Cell index and base in flat grid buffer
        let cell_idx = ucx * grid_w + ucz;
        let cell_base = cell_idx * FOOD_GRID_CELL_STRIDE;

        // Atomically claim a slot (index 0 is the count)
        let slot = atomicAdd(&food_grid[cell_base], 1u);
        if slot < FOOD_GRID_MAX_PER_CELL {
            atomicStore(&food_grid[cell_base + 1u + slot], i);
        }
    }
}
