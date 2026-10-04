// Test-only metadata uses the first agent's otherwise-unused tiled prediction
// scratch. Codes are exact nonnegative integers represented as f32, not NaNs.
const FOOD_CACHE_VALID: u32 = 0u;
const FOOD_CACHE_OVERFLOW: u32 = 1u;
const FOOD_CACHE_REUSED: u32 = 2u;
const FOOD_CACHE_EXPIRY: u32 = 3u;
const FOOD_CACHE_HEADER: u32 = 4u;
const FOOD_CACHE_DIRTY_FLAG: u32 = 0u;
const FOOD_CACHE_EXPIRY_FLAG: u32 = 1u;
const FOOD_CACHE_OVERFLOW_FLAG: u32 = 2u;
var<workgroup> food_cache_flags: array<atomic<u32>, 3>;

fn food_cache_cell(item: u32) -> u32 {
    if atomicLoad(&food_flags[item]) != 0u { return 0u; }
    let base = item * FOOD_STATE_STRIDE;
    let grid_offset = i32(wc_u32(WC_GRID_OFFSET));
    let width = wc_u32(WC_GRID_WIDTH);
    let x = cell_coord(food_state[base + FOOD_POSITION_X]) + grid_offset;
    let z = cell_coord(food_state[base + FOOD_POSITION_Z]) + grid_offset;
    if x < 0 || z < 0 || u32(x) >= width || u32(z) >= width { return 0u; }
    return u32(x) * width + u32(z) + 1u;
}

fn food_cache_rebuild() -> bool {
    return atomicLoad(&food_cache_flags[FOOD_CACHE_DIRTY_FLAG]) != 0u;
}

fn food_cache_begin(tid: u32) {
    if tid == 0u {
        let valid = packed_encoder.scratch[SCRATCH_PREDICTION + FOOD_CACHE_VALID] == 1.0;
        let overflow = packed_encoder.scratch[SCRATCH_PREDICTION + FOOD_CACHE_OVERFLOW] != 0.0;
        atomicStore(&food_cache_flags[FOOD_CACHE_DIRTY_FLAG], select(1u, 0u, valid && !overflow));
        atomicStore(&food_cache_flags[FOOD_CACHE_EXPIRY_FLAG], 0u);
        atomicStore(&food_cache_flags[FOOD_CACHE_OVERFLOW_FLAG], 0u);
    }
    workgroupBarrier();
    for (var item = tid; item < wc_u32(WC_FOOD_COUNT); item += 256u) {
        let code = food_cache_cell(item);
        let slot = SCRATCH_PREDICTION + FOOD_CACHE_HEADER + item;
        if packed_encoder.scratch[slot] != f32(code) {
            atomicStore(&food_cache_flags[FOOD_CACHE_DIRTY_FLAG], 1u);
        }
        packed_encoder.scratch[slot] = f32(code);

        // Use the original respawn branch arithmetic. Expiry forces a full
        // rebuild now and invalidates this pre-respawn membership snapshot for
        // the next cycle. The canonical respawn function remains untouched.
        if atomicLoad(&food_flags[item]) != 0u {
            let timer = food_state[item * FOOD_STATE_STRIDE + FOOD_RESPAWN_TIMER];
            if !(timer <= 0.0) && !(timer - wc_f32(WC_DT) > 0.0) {
                atomicStore(&food_cache_flags[FOOD_CACHE_DIRTY_FLAG], 1u);
                atomicStore(&food_cache_flags[FOOD_CACHE_EXPIRY_FLAG], 1u);
            }
        }
    }
    workgroupBarrier();
}

fn food_cache_finish(tid: u32) {
    if tid == 0u {
        let expiry = atomicLoad(&food_cache_flags[FOOD_CACHE_EXPIRY_FLAG]);
        packed_encoder.scratch[SCRATCH_PREDICTION + FOOD_CACHE_VALID] = select(1.0, 0.0, expiry != 0u);
        packed_encoder.scratch[SCRATCH_PREDICTION + FOOD_CACHE_OVERFLOW] = f32(atomicLoad(&food_cache_flags[FOOD_CACHE_OVERFLOW_FLAG]));
        packed_encoder.scratch[SCRATCH_PREDICTION + FOOD_CACHE_REUSED] = select(1.0, 0.0, food_cache_rebuild());
        packed_encoder.scratch[SCRATCH_PREDICTION + FOOD_CACHE_EXPIRY] = f32(expiry);
    }
}
