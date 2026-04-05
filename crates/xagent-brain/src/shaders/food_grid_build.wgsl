@group(0) @binding(0) var<storage, read> food_state: array<f32>;
@group(0) @binding(1) var<storage, read> food_flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> food_grid: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let food_idx = gid.x;
    if food_idx >= FOOD_COUNT_VAL { return; }

    // Skip consumed food
    if food_flags[food_idx] != 0u { return; }

    // Read position
    let base = food_idx * FOOD_STATE_STRIDE;
    let fx = food_state[base + F_POS_X];
    let fz = food_state[base + F_POS_Z];

    // Compute grid cell with offset
    let cx = cell_coord(fx) + GRID_OFFSET;
    let cz = cell_coord(fz) + GRID_OFFSET;

    // Bounds check
    if cx < 0 || cz < 0 { return; }
    let ucx = u32(cx);
    let ucz = u32(cz);
    if ucx >= GRID_WIDTH || ucz >= GRID_WIDTH { return; }

    // Cell index and base in flat grid buffer
    let cell_idx = ucx * GRID_WIDTH + ucz;
    let cell_base = cell_idx * FOOD_GRID_CELL_STRIDE;

    // Atomically claim a slot (index 0 is the count)
    let slot = atomicAdd(&food_grid[cell_base], 1u);
    if slot < FOOD_GRID_MAX_PER_CELL {
        atomicStore(&food_grid[cell_base + 1u + slot], food_idx);
    }
}
