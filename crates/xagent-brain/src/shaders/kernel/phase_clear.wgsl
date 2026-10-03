// Phase: clear grids and collision scratch.
// All 256 threads cooperate to empty food_grid and agent_grid and zero
// collision_scratch.

fn phase_clear(tid: u32) {
    let grid_w = wc_u32(WC_GRID_WIDTH);
    let total_cells = grid_w * grid_w;

    // Empty every grid cell by zeroing its count (slot 0). Every reader stops
    // at the clamped count and every insert writes the slot it claims, so the
    // stale entries left past the count are never read.
    for (var cell = tid; cell < total_cells; cell += 256u) {
        atomicStore(&food_grid[cell * FOOD_GRID_CELL_STRIDE], 0u);
        atomicStore(&agent_grid[cell * AGENT_GRID_CELL_STRIDE], 0u);
    }

    // No food is claimed until the next claim step (see resolve_food_claim).
    let food_count = wc_u32(WC_FOOD_COUNT);
    for (var item = tid; item < food_count; item += 256u) {
        atomicStore(&food_flags[food_claim_slot(item)], FOOD_UNCLAIMED);
    }

    // Zero collision_scratch: agent_count * 3 elements
    let agent_count = wc_u32(WC_AGENT_COUNT);
    let scratch_size = agent_count * 3u;
    for (var i = tid; i < scratch_size; i += 256u) {
        atomicStore(&collision_scratch[i], 0);
    }
}
