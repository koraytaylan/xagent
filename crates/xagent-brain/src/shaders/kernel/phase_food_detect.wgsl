// Phase: food detection and consumption.
// Ported from food_detect.wgsl. 1 thread per agent, tid = agent index.

fn phase_food_detect(tid: u32) {
    let agent = tid;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    if agent >= agent_count { return; }

    let b = agent * PHYS_STRIDE;

    // Skip dead agents
    let alive = physics_state[b + P_ALIVE];
    if alive < 0.5 { return; }

    // Agent position
    let ax = physics_state[b + P_POS_X];
    let az = physics_state[b + P_POS_Z];

    // Search 3x3 neighborhood
    let grid_offset = i32(wc_u32(WC_GRID_OFFSET));
    let grid_w = wc_u32(WC_GRID_WIDTH);
    let center_cx = cell_coord(ax) + grid_offset;
    let center_cz = cell_coord(az) + grid_offset;

    let food_radius = wc_f32(WC_FOOD_RADIUS);
    let food_radius_sq = food_radius * food_radius;

    var best_idx: u32 = 0xFFFFFFFFu;
    var best_dist_sq: f32 = food_radius_sq;

    for (var di: i32 = -1; di <= 1; di++) {
        for (var dj: i32 = -1; dj <= 1; dj++) {
            let ncx = center_cx + di;
            let ncz = center_cz + dj;

            // Bounds check
            if ncx < 0 || ncz < 0 { continue; }
            let uncx = u32(ncx);
            let uncz = u32(ncz);
            if uncx >= grid_w || uncz >= grid_w { continue; }

            let cell_idx = uncx * grid_w + uncz;
            let cell_base = cell_idx * FOOD_GRID_CELL_STRIDE;

            let count = min(atomicLoad(&food_grid[cell_base]), FOOD_GRID_MAX_PER_CELL);
            for (var s: u32 = 0u; s < count; s++) {
                let fidx = atomicLoad(&food_grid[cell_base + 1u + s]);

                // Skip consumed
                if atomicLoad(&food_flags[fidx]) != 0u { continue; }

                let fbase = fidx * FOOD_STATE_STRIDE;
                let fx = food_state[fbase + F_POS_X];
                let fz = food_state[fbase + F_POS_Z];

                let dx = ax - fx;
                let dz = az - fz;
                let dist_sq = dx * dx + dz * dz;

                if dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    best_idx = fidx;
                }
            }
        }
    }

    // Try to claim the nearest food
    if best_idx != 0xFFFFFFFFu {
        let result = atomicCompareExchangeWeak(&food_flags[best_idx], 0u, 1u);
        if result.exchanged {
            // Winner: award energy and increment food count
            let food_energy = wc_f32(WC_FOOD_ENERGY);
            physics_state[b + P_ENERGY] = physics_state[b + P_ENERGY] + food_energy;
            physics_state[b + P_FOOD_COUNT] = physics_state[b + P_FOOD_COUNT] + 1.0;
        }
    }
}
