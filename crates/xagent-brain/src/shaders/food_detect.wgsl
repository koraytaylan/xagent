@group(0) @binding(0) var<storage, read_write> agent_phys: array<f32>;
@group(0) @binding(1) var<storage, read> food_state: array<f32>;
@group(0) @binding(2) var<storage, read_write> food_flags: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> food_grid: array<u32>;
@group(0) @binding(4) var<uniform> wconfig: array<vec4<f32>, 6>;

fn wc(idx: u32) -> f32 {
    return wconfig[idx / 4u][idx % 4u];
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent = gid.x;
    if agent >= AGENT_COUNT_VAL { return; }

    let b = agent * PHYS_STRIDE;

    // Skip dead agents
    let alive = agent_phys[b + P_ALIVE];
    if alive < 0.5 { return; }

    // Agent position
    let ax = agent_phys[b + P_POS_X];
    let az = agent_phys[b + P_POS_Z];

    // Search 3×3 neighborhood
    let center_cx = cell_coord(ax) + GRID_OFFSET;
    let center_cz = cell_coord(az) + GRID_OFFSET;

    let food_radius = wc(WC_FOOD_RADIUS);
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
            if uncx >= GRID_WIDTH || uncz >= GRID_WIDTH { continue; }

            let cell_idx = uncx * GRID_WIDTH + uncz;
            let cell_base = cell_idx * FOOD_GRID_CELL_STRIDE;

            let count = min(food_grid[cell_base], FOOD_GRID_MAX_PER_CELL);
            for (var s: u32 = 0u; s < count; s++) {
                let fidx = food_grid[cell_base + 1u + s];

                // Skip consumed (non-atomic read is fine, grid was built before this runs)
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
            let food_energy = wc(6u); // WC_FOOD_ENERGY
            agent_phys[b + P_ENERGY] = agent_phys[b + P_ENERGY] + food_energy;
            agent_phys[b + P_FOOD_COUNT] = agent_phys[b + P_FOOD_COUNT] + 1.0;
        }
    }
}
