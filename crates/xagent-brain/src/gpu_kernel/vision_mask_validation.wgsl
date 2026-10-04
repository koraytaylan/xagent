// Test entries call production grid and visibility-mask construction helpers.

@compute @workgroup_size(256)
fn masks_from_retained_grid(@builtin(local_invocation_index) thread: u32) {
    let width = wc_u32(WC_GRID_WIDTH);
    let cells = width * width;
    let mask_base = agent_visibility_mask_offset(width);
    for (var cell = thread; cell < cells; cell += 256u) {
        atomicStore(&agent_grid[mask_base + cell], 0u);
    }
    storageBarrier();
    workgroupBarrier();
    for (var cell = thread; cell < cells; cell += 256u) {
        let base = cell * AGENT_GRID_CELL_STRIDE;
        let count = min(atomicLoad(&agent_grid[base]), AGENT_GRID_MAX_PER_CELL);
        for (var slot = 0u; slot < count; slot++) {
            let agent = atomicLoad(&agent_grid[base + 1u + slot]);
            insert_agent_visibility_mask(agent, i32(cell / width), i32(cell % width), width);
        }
    }
}

@compute @workgroup_size(256)
fn masks_from_production_grid(@builtin(local_invocation_index) thread: u32) {
    phase_clear(thread);
    storageBarrier();
    workgroupBarrier();
    phase_agent_grid(thread);
}
