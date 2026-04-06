// Food respawn shader: counts down respawn timers for consumed food items,
// then picks a new position using GPU RNG (pcg_hash) once the timer expires.
// 1 thread per food item.

@group(0) @binding(0) var<storage, read_write> food_state: array<f32>;
@group(0) @binding(1) var<storage, read_write> food_flags: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> food_grid: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> heightmap: array<f32>;
@group(0) @binding(4) var<storage, read> biome: array<u32>;
@group(0) @binding(5) var<uniform> wconfig: array<vec4<f32>, 6>;

fn wc(idx: u32) -> f32 {
    return wconfig[idx / 4u][idx % 4u];
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let food_idx = gid.x;
    if food_idx >= FOOD_COUNT_VAL { return; }

    // Only process consumed (unavailable) food
    if atomicLoad(&food_flags[food_idx]) == 0u { return; }

    let base = food_idx * FOOD_STATE_STRIDE;
    var timer = food_state[base + F_RESPAWN_TIMER];

    // Initialize timer on first tick after consumption
    if timer <= 0.0 {
        food_state[base + F_RESPAWN_TIMER] = FOOD_RESPAWN_TIME;
        return;
    }

    // Decrement timer
    let dt = wc(1u);
    timer -= dt;
    if timer > 0.0 {
        food_state[base + F_RESPAWN_TIMER] = timer;
        return;
    }

    // Timer expired — find a new spawn position using GPU RNG
    let seed = u32(wc(15u)); // WC_RNG_SEED
    let tick = u32(wc(14u)); // WC_TICK
    let spawn_half = TERRAIN_HALF - 5.0;

    var new_x = food_state[base + F_POS_X];
    var new_z = food_state[base + F_POS_Z];
    var found = false;

    for (var attempt: u32 = 0u; attempt < FOOD_RESPAWN_ATTEMPTS; attempt++) {
        let hash1 = pcg_hash(food_idx ^ seed ^ tick ^ (attempt * 7919u));
        let hash2 = pcg_hash(hash1 + 1u);

        let cx = (rand_f32(hash1) * 2.0 - 1.0) * spawn_half;
        let cz = (rand_f32(hash2) * 2.0 - 1.0) * spawn_half;

        // Check biome at candidate position
        let biome_half = wc(10u); // WC_TERRAIN_HALF / terrain_half
        let biome_inv = wc(11u);  // WC_BIOME_INV_CELL
        let bcol = min(u32((cx + biome_half) * biome_inv), 255u);
        let brow = min(u32((cz + biome_half) * biome_inv), 255u);
        let biome_type = biome[brow * 256u + bcol];

        if biome_type == 0u { // FoodRich
            new_x = cx;
            new_z = cz;
            found = true;
            break;
        }
    }

    // Inline bilinear terrain height calculation
    let gx = clamp((new_x + TERRAIN_HALF) * TERRAIN_INV_STEP, 0.0, TERRAIN_MAX_COORD);
    let gz = clamp((new_z + TERRAIN_HALF) * TERRAIN_INV_STEP, 0.0, TERRAIN_MAX_COORD);
    let ix = min(u32(gx), TERRAIN_MAX_IDX);
    let iz = min(u32(gz), TERRAIN_MAX_IDX);
    let fx = gx - f32(ix);
    let fz = gz - f32(iz);
    let h00 = heightmap[iz * TERRAIN_VPS + ix];
    let h10 = heightmap[iz * TERRAIN_VPS + ix + 1u];
    let h01 = heightmap[(iz + 1u) * TERRAIN_VPS + ix];
    let h11 = heightmap[(iz + 1u) * TERRAIN_VPS + ix + 1u];
    let new_y = mix(mix(h00, h10, fx), mix(h01, h11, fx), fz) + FOOD_HEIGHT_OFFSET;

    // Update food state
    food_state[base + F_POS_X] = new_x;
    food_state[base + F_POS_Y] = new_y;
    food_state[base + F_POS_Z] = new_z;
    food_state[base + F_RESPAWN_TIMER] = 0.0;

    // Mark food as available
    atomicStore(&food_flags[food_idx], 0u);

    // Insert into food_grid
    let cx_grid = cell_coord(new_x) + GRID_OFFSET;
    let cz_grid = cell_coord(new_z) + GRID_OFFSET;
    if cx_grid >= 0 && cz_grid >= 0 {
        let ucx = u32(cx_grid);
        let ucz = u32(cz_grid);
        if ucx < GRID_WIDTH && ucz < GRID_WIDTH {
            let cell_idx = ucx * GRID_WIDTH + ucz;
            let cell_base = cell_idx * FOOD_GRID_CELL_STRIDE;
            let slot = atomicAdd(&food_grid[cell_base], 1u);
            if slot < FOOD_GRID_MAX_PER_CELL {
                atomicStore(&food_grid[cell_base + 1u + slot], food_idx);
            }
        }
    }
}
