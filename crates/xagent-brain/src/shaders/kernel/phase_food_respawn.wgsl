// Phase: food respawn — count down timers, pick new positions via GPU RNG.
// Ported from food_respawn.wgsl. All 256 threads cooperate to process all food items.

fn phase_food_respawn(tid: u32, tick: u32) {
    let food_count = wc_u32(WC_FOOD_COUNT);
    let grid_offset = i32(wc_u32(WC_GRID_OFFSET));
    let grid_w = wc_u32(WC_GRID_WIDTH);

    for (var i = tid; i < food_count; i += 256u) {
        // Only process consumed (unavailable) food
        if atomicLoad(&food_flags[i]) == 0u { continue; }

        let base = i * FOOD_STATE_STRIDE;
        var timer = food_state[base + F_RESPAWN_TIMER];

        // Initialize timer on first tick after consumption
        if timer <= 0.0 {
            food_state[base + F_RESPAWN_TIMER] = FOOD_RESPAWN_TIME;
            continue;
        }

        // Decrement timer
        let dt = wc_f32(WC_DT);
        timer -= dt;
        if timer > 0.0 {
            food_state[base + F_RESPAWN_TIMER] = timer;
            continue;
        }

        // Timer expired — find a new spawn position using GPU RNG
        let seed = wc_u32(WC_RNG_SEED);
        let terrain_half = wc_f32(WC_TERRAIN_HALF);
        let spawn_half = terrain_half - 5.0;

        var new_x = food_state[base + F_POS_X];
        var new_z = food_state[base + F_POS_Z];
        var found = false;

        for (var attempt: u32 = 0u; attempt < FOOD_RESPAWN_ATTEMPTS; attempt++) {
            let hash1 = pcg_hash(i ^ seed ^ tick ^ (attempt * 7919u));
            let hash2 = pcg_hash(hash1 + 1u);

            let cx = (hash_to_float(hash1) * 2.0 - 1.0) * spawn_half;
            let cz = (hash_to_float(hash2) * 2.0 - 1.0) * spawn_half;

            // Check biome at candidate position
            let biome_type = sample_biome(cx, cz);

            if biome_type == BIOME_FOOD_RICH {
                new_x = cx;
                new_z = cz;
                found = true;
                break;
            }
        }

        // Compute terrain height at new position (use sample_height from common.wgsl)
        let new_y = sample_height(new_x, new_z) + FOOD_HEIGHT_OFFSET;

        // Update food state
        food_state[base + F_POS_X] = new_x;
        food_state[base + F_POS_Y] = new_y;
        food_state[base + F_POS_Z] = new_z;
        food_state[base + F_RESPAWN_TIMER] = 0.0;

        // Mark food as available
        atomicStore(&food_flags[i], 0u);

        // Insert into food_grid
        let cx_grid = cell_coord(new_x) + grid_offset;
        let cz_grid = cell_coord(new_z) + grid_offset;
        if cx_grid >= 0 && cz_grid >= 0 {
            let ucx = u32(cx_grid);
            let ucz = u32(cz_grid);
            if ucx < grid_w && ucz < grid_w {
                let cell_idx = ucx * grid_w + ucz;
                let cell_base = cell_idx * FOOD_GRID_CELL_STRIDE;
                let slot = atomicAdd(&food_grid[cell_base], 1u);
                if slot < FOOD_GRID_MAX_PER_CELL {
                    atomicStore(&food_grid[cell_base + 1u + slot], i);
                }
            }
        }
    }
}
