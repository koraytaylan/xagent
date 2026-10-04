// Test-only sparse maintenance of registered grid cells. Codes are cell+1,
// with zero meaning no registration; the checked host layout bounds them to
// exactly representable positive f32 integers. Metadata records insertion
// coordinates before collision movement, never positions read after collision.

fn sparse_grid_usable() -> bool {
    return packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_VALID] == 1.0
        && packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_WIDTH] == f32(wc_u32(WC_GRID_WIDTH))
        && packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_OFFSET] == f32(wc_u32(WC_GRID_OFFSET))
        && packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_FOODS] == f32(wc_u32(WC_FOOD_COUNT))
        && packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_AGENTS] == f32(wc_u32(WC_AGENT_COUNT));
}

fn phase_sparse_clear(tid: u32) {
    if !sparse_grid_usable() {
        phase_clear(tid);
        return;
    }
    let width = wc_u32(WC_GRID_WIDTH);
    if tid < wc_u32(WC_FOOD_COUNT) {
        let code = u32(packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_FOOD_CELLS + tid]);
        if code != 0u {
            atomicStore(&food_grid[(code - 1u) * FOOD_GRID_CELL_STRIDE], 0u);
        }
    }
    if tid < wc_u32(WC_AGENT_COUNT) {
        let code = u32(packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_AGENT_CELLS + tid]);
        if code != 0u {
            let cell = code - 1u;
            atomicStore(&agent_grid[cell * AGENT_GRID_CELL_STRIDE], 0u);
            if VISION_AGENT_MASKS && wc_u32(WC_AGENT_COUNT) <= AGENT_VISIBILITY_MASK_BITS {
                let cell_x = i32(cell / width);
                let cell_z = i32(cell % width);
                let mask_offset = agent_visibility_mask_offset(width);
                for (var offset_x: i32 = -1; offset_x <= 1; offset_x++) {
                    for (var offset_z: i32 = -1; offset_z <= 1; offset_z++) {
                        let x = cell_x + offset_x;
                        let z = cell_z + offset_z;
                        if x < 0 || z < 0 || u32(x) >= width || u32(z) >= width { continue; }
                        atomicStore(&agent_grid[mask_offset + u32(x) * width + u32(z)], 0u);
                    }
                }
            }
        }
    }
    // Duplicated zero stores to shared occupied cells are atomic and harmless.
    // The canonical barrier after this function completes all clears before
    // any new insertion or visibility-mask OR.
    sparse_clear_remainder(tid);
}

fn phase_sparse_sort_grid_cells(tid: u32) {
    // The cold route retains the complete canonical cell scan. Local ownership
    // decisions are all formed before the unconditional barrier below: the
    // insertion sort changes the first retained ID, so checking that ID
    // while another owner sorts would permit duplicate owners.
    let warm = sparse_grid_usable();
    var food_code = 0u;
    var agent_code = 0u;
    var food_owner = false;
    var agent_owner = false;
    if warm && tid < wc_u32(WC_FOOD_COUNT) {
        food_code = u32(packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_FOOD_CELLS + tid]);
        if food_code != 0u {
            let base = (food_code - 1u) * FOOD_GRID_CELL_STRIDE;
            food_owner = atomicLoad(&food_grid[base]) != 0u
                && atomicLoad(&food_grid[base + 1u]) == tid;
        }
    }
    if warm && tid < wc_u32(WC_AGENT_COUNT) {
        agent_code = u32(packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_AGENT_CELLS + tid]);
        if agent_code != 0u {
            let base = (agent_code - 1u) * AGENT_GRID_CELL_STRIDE;
            agent_owner = atomicLoad(&agent_grid[base]) != 0u
                && atomicLoad(&agent_grid[base + 1u]) == tid;
        }
    }
    storageBarrier(); workgroupBarrier();
    if warm {
        if food_owner { sparse_sort_food(food_code - 1u); }
        if agent_owner { sparse_sort_agents(agent_code - 1u); }
    } else {
        phase_sort_grid_cells(tid);
    }
}

fn sparse_grid_finish(tid: u32) {
    if tid == 0u {
        let warm = sparse_grid_usable();
        packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_USED] = select(0.0, 1.0, warm);
        packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_WIDTH] = f32(wc_u32(WC_GRID_WIDTH));
        packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_OFFSET] = f32(wc_u32(WC_GRID_OFFSET));
        packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_FOODS] = f32(wc_u32(WC_FOOD_COUNT));
        packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_AGENTS] = f32(wc_u32(WC_AGENT_COUNT));
        packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_VALID] = 1.0;
    }
}
