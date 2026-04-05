@group(0) @binding(0) var<storage, read> agent_phys: array<f32>;
@group(0) @binding(1) var<storage, read> agent_grid: array<u32>;
@group(0) @binding(2) var<storage, read_write> scratch: array<atomic<i32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent = gid.x;
    if agent >= AGENT_COUNT_VAL { return; }

    let b = agent * PHYS_STRIDE;

    // Skip dead agents
    if agent_phys[b + P_ALIVE] < 0.5 { return; }

    // Read self position
    let self_x = agent_phys[b + P_POS_X];
    let self_y = agent_phys[b + P_POS_Y];
    let self_z = agent_phys[b + P_POS_Z];

    // Compute cell with offset
    let self_cx = cell_coord(self_x) + GRID_OFFSET;
    let self_cz = cell_coord(self_z) + GRID_OFFSET;

    // Iterate 3x3 neighborhood
    for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let cx = self_cx + dx;
            let cz = self_cz + dz;

            // Bounds check
            if cx < 0 || cz < 0 { continue; }
            let ucx = u32(cx);
            let ucz = u32(cz);
            if ucx >= GRID_WIDTH || ucz >= GRID_WIDTH { continue; }

            let cell_idx = ucx * GRID_WIDTH + ucz;
            let cell_base = cell_idx * AGENT_GRID_CELL_STRIDE;
            let count = agent_grid[cell_base];
            let clamped_count = min(count, AGENT_GRID_MAX_PER_CELL);

            for (var s: u32 = 0u; s < clamped_count; s++) {
                let other = agent_grid[cell_base + 1u + s];

                // Only process pair once (lower index pushes higher index)
                if other <= agent { continue; }

                let ob = other * PHYS_STRIDE;

                // Skip dead other
                if agent_phys[ob + P_ALIVE] < 0.5 { continue; }

                // Compute displacement from self to other
                let diff_x = agent_phys[ob + P_POS_X] - self_x;
                let diff_y = agent_phys[ob + P_POS_Y] - self_y;
                let diff_z = agent_phys[ob + P_POS_Z] - self_z;
                let dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

                if dist_sq < COLLISION_MIN_DIST_SQ && dist_sq > 0.001 {
                    let dist = sqrt(dist_sq);
                    let overlap = COLLISION_MIN_DIST - dist;
                    let inv_dist = 1.0 / dist;
                    let push_x = diff_x * inv_dist * overlap * 0.5;
                    let push_y = diff_y * inv_dist * overlap * 0.5;
                    let push_z = diff_z * inv_dist * overlap * 0.5;

                    // Fixed-point encode
                    let ipx = i32(push_x * COLLISION_FIXED_SCALE);
                    let ipy = i32(push_y * COLLISION_FIXED_SCALE);
                    let ipz = i32(push_z * COLLISION_FIXED_SCALE);

                    // Self pushed away from other (negative push)
                    atomicAdd(&scratch[agent * 3u],      -ipx);
                    atomicAdd(&scratch[agent * 3u + 1u], -ipy);
                    atomicAdd(&scratch[agent * 3u + 2u], -ipz);

                    // Other pushed toward self (positive push)
                    atomicAdd(&scratch[other * 3u],      ipx);
                    atomicAdd(&scratch[other * 3u + 1u], ipy);
                    atomicAdd(&scratch[other * 3u + 2u], ipz);
                }
            }
        }
    }
}
