@group(0) @binding(0) var<storage, read> agent_phys: array<f32>;
@group(0) @binding(1) var<storage, read_write> agent_grid: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent = gid.x;
    if agent >= AGENT_COUNT_VAL { return; }

    let b = agent * PHYS_STRIDE;

    // Skip dead agents
    if agent_phys[b + P_ALIVE] < 0.5 { return; }

    // Read position
    let px = agent_phys[b + P_POS_X];
    let pz = agent_phys[b + P_POS_Z];

    // Compute cell with offset
    let cx = cell_coord(px) + GRID_OFFSET;
    let cz = cell_coord(pz) + GRID_OFFSET;

    // Bounds check
    if cx < 0 || cz < 0 { return; }
    let ucx = u32(cx);
    let ucz = u32(cz);
    if ucx >= GRID_WIDTH || ucz >= GRID_WIDTH { return; }

    // Cell index and base in flat grid buffer
    let cell_idx = ucx * GRID_WIDTH + ucz;
    let cell_base = cell_idx * AGENT_GRID_CELL_STRIDE;

    // Atomically claim a slot (index 0 is the count)
    let slot = atomicAdd(&agent_grid[cell_base], 1u);
    if slot < AGENT_GRID_MAX_PER_CELL {
        atomicStore(&agent_grid[cell_base + 1u + slot], agent);
    }
}
