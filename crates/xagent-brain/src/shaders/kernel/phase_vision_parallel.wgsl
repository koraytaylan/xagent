// Independent samples of each ray share an integer first-event reduction.
// Every sample retains the serial ray's arithmetic and hit precedence.

override VISION_PARALLEL_STEPS: bool = false;
// Twenty-five samples fit in one 32-lane ray group.
const VISION_PARALLEL_LANES_PER_RAY: u32 = 32u;
// The largest entry point has 256 invocations, or eight ray groups.
const VISION_PARALLEL_MAX_RAYS_PER_WORKGROUP: u32 = 8u;
override VISION_WORKGROUP_SIZE: u32 = VISION_RAYS_PER_WORKGROUP *
    (1u + (VISION_PARALLEL_LANES_PER_RAY - 1u) * u32(VISION_PARALLEL_STEPS));

// Eight keys per step keep event precedence below the next sample's key.
const VISION_EVENT_STEP_STRIDE: u32 = 8u;
const VISION_EVENT_FOOD: u32 = 0u;
const VISION_EVENT_AGENT: u32 = 1u;
const VISION_EVENT_TERRAIN_GREEN: u32 = 2u;
const VISION_EVENT_TERRAIN_BROWN: u32 = 3u;
const VISION_EVENT_TERRAIN_RED: u32 = 4u;
const VISION_EVENT_SKY: u32 = 5u;
const VISION_EVENT_NONE: u32 = 0xffffffffu;

var<workgroup> vision_first_events:
    array<atomic<u32>, VISION_PARALLEL_MAX_RAYS_PER_WORKGROUP>;

// Returns the first event at a single unchanged discrete sample. A sky
// event terminates the ray even if a later sample would encounter an object.
fn vision_parallel_sample(agent_id: u32, ray_idx: u32, step: u32) -> u32 {
    let base = agent_id * PHYS_STRIDE;
    let pos = vec3<f32>(
        physics_state[base + P_POS_X],
        physics_state[base + P_POS_Y],
        physics_state[base + P_POS_Z],
    );
    let facing = vec3<f32>(
        physics_state[base + P_FACING_X],
        physics_state[base + P_FACING_Y],
        physics_state[base + P_FACING_Z],
    );
    let grid_width = wc_u32(WC_GRID_WIDTH);
    let grid_offset = i32(wc_u32(WC_GRID_OFFSET));

    let col = ray_idx % VISION_W;
    let row = ray_idx / VISION_W;
    let u = (f32(col) / f32(VISION_W - 1u)) * 2.0 - 1.0;
    let v = (f32(row) / f32(VISION_H - 1u)) * 2.0 - 1.0;
    let gene_base = agent_id * BRAIN_STRIDE;
    let horizontal_fov = clamp(
        brain_state[gene_base + O_HORIZONTAL_FOV], HORIZONTAL_FOV_MIN, HORIZONTAL_FOV_MAX);
    let vertical_fov = clamp(
        brain_state[gene_base + O_VERTICAL_FOV], VERTICAL_FOV_MIN, VERTICAL_FOV_MAX);
    let tan_half_horizontal = tan(radians(horizontal_fov) * 0.5);
    let tan_half_vertical = tan(radians(vertical_fov) * 0.5);
    let right = vec3<f32>(facing.z, 0.0, -facing.x);
    let ray_dir = normalize(
        facing + right * u * tan_half_horizontal + vec3<f32>(0.0, -v * tan_half_vertical, 0.0));
    let t = f32(step + 1u) * VISION_STEP_SIZE;
    let ray_pos = pos + ray_dir * t;

    let food_cx_low = cell_coord(ray_pos.x - FOOD_PROBE_HALF_WIDTH) + grid_offset;
    let food_cx_high = cell_coord(ray_pos.x + FOOD_PROBE_HALF_WIDTH) + grid_offset;
    let food_cz_low = cell_coord(ray_pos.z - FOOD_PROBE_HALF_WIDTH) + grid_offset;
    let food_cz_high = cell_coord(ray_pos.z + FOOD_PROBE_HALF_WIDTH) + grid_offset;
    for (var ncx = food_cx_low; ncx <= food_cx_high; ncx++) {
        for (var ncz = food_cz_low; ncz <= food_cz_high; ncz++) {
            if ncx < 0 || ncz < 0 { continue; }
            let uncx = u32(ncx);
            let uncz = u32(ncz);
            if uncx >= grid_width || uncz >= grid_width { continue; }
            let cell_idx = uncx * grid_width + uncz;
            let cell_base = cell_idx * FOOD_GRID_CELL_STRIDE;
            let count = min(u32(food_grid[cell_base]), FOOD_GRID_MAX_PER_CELL);
            for (var s: u32 = 0u; s < count; s++) {
                let fidx = u32(food_grid[cell_base + 1u + s]);
                if food_flags[fidx] != 0u { continue; }
                let fbase = fidx * FOOD_STATE_STRIDE;
                let fx = food_state[fbase + FOOD_POSITION_X];
                let fy = food_state[fbase + FOOD_POSITION_Y];
                let fz = food_state[fbase + FOOD_POSITION_Z];
                let dx = ray_pos.x - fx;
                let dy = ray_pos.y - fy;
                let dz = ray_pos.z - fz;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                if dist_sq < FOOD_RAY_RADIUS_SQ {
                    return VISION_EVENT_FOOD;
                }
            }
        }
    }

    if vision_parallel_agent_sample(agent_id, ray_pos, grid_width, grid_offset) {
        return VISION_EVENT_AGENT;
    }

    let ground_h = sample_height(ray_pos.x, ray_pos.z);
    if ray_pos.y <= ground_h {
        let biome_type = sample_biome(ray_pos.x, ray_pos.z);
        if biome_type == 0u { return VISION_EVENT_TERRAIN_GREEN; }
        if biome_type == 1u { return VISION_EVENT_TERRAIN_BROWN; }
        return VISION_EVENT_TERRAIN_RED;
    }
    if ray_dir.y > 0.3 && ray_pos.y > ground_h + 5.0 {
        return VISION_EVENT_SKY;
    }
    return VISION_EVENT_NONE;
}

// Candidate membership is the registered 3x3 block, including registrations
// whose agents have crossed a cell boundary during collision resolution.
fn vision_parallel_agent_sample(
    agent_id: u32, ray_pos: vec3<f32>, grid_width: u32, grid_offset: i32,
) -> bool {
    let center_x = cell_coord(ray_pos.x) + grid_offset;
    let center_z = cell_coord(ray_pos.z) + grid_offset;
    if VISION_AGENT_MASKS && wc_u32(WC_AGENT_COUNT) <= AGENT_VISIBILITY_MASK_BITS
        && center_x >= 0 && center_z >= 0
        && u32(center_x) < grid_width && u32(center_z) < grid_width {
        let center_cell = u32(center_x) * grid_width + u32(center_z);
        let mask = agent_grid[agent_visibility_mask_offset(grid_width) + center_cell];
        var candidates = mask & ~(1u << agent_id);
        while candidates != 0u {
            let other = firstTrailingBit(candidates);
            candidates = candidates & (candidates - 1u);
            if vision_parallel_agent_hit(ray_pos, other) { return true; }
        }
        return false;
    }

    // Outside-grid centre cells can still see registrations in an edge cell.
    for (var offset_x: i32 = -1; offset_x <= 1; offset_x++) {
        for (var offset_z: i32 = -1; offset_z <= 1; offset_z++) {
            let cell_x = center_x + offset_x;
            let cell_z = center_z + offset_z;
            if cell_x < 0 || cell_z < 0 { continue; }
            let grid_x = u32(cell_x);
            let grid_z = u32(cell_z);
            if grid_x >= grid_width || grid_z >= grid_width { continue; }
            let cell_base = (grid_x * grid_width + grid_z) * AGENT_GRID_CELL_STRIDE;
            let count = min(u32(agent_grid[cell_base]), AGENT_GRID_MAX_PER_CELL);
            for (var slot = 0u; slot < count; slot++) {
                let other = u32(agent_grid[cell_base + 1u + slot]);
                if other == agent_id { continue; }
                if vision_parallel_agent_hit(ray_pos, other) { return true; }
            }
        }
    }
    return false;
}

fn vision_parallel_agent_hit(ray_pos: vec3<f32>, other: u32) -> bool {
    let ob = other * PHYS_STRIDE;
    if physics_state[ob + P_ALIVE] < 0.5 { return false; }
    let ox = physics_state[ob + P_POS_X];
    let oy = physics_state[ob + P_POS_Y];
    let oz = physics_state[ob + P_POS_Z];
    let dx = ray_pos.x - ox;
    let dy = ray_pos.y - oy;
    let dz = ray_pos.z - oz;
    let dist_sq = dx * dx + dy * dy + dz * dz;
    return dist_sq < AGENT_RAY_RADIUS_SQ;
}

// All workgroup invocations must call this function, including padding rays
// and dead agents, so both reduction barriers are reached uniformly.
fn vision_parallel_rays(agent_id: u32, first_ray: u32, tid: u32) {
    let local_ray = tid / VISION_PARALLEL_LANES_PER_RAY;
    let step = tid % VISION_PARALLEL_LANES_PER_RAY;
    let ray_idx = first_ray + local_ray;
    if step == 0u {
        atomicStore(&vision_first_events[local_ray], VISION_EVENT_NONE);
    }
    workgroupBarrier();

    let alive = !(physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5);
    if alive && ray_idx < VISION_RAYS && step < VISION_NUM_STEPS {
        let event = vision_parallel_sample(agent_id, ray_idx, step);
        if event != VISION_EVENT_NONE {
            atomicMin(&vision_first_events[local_ray], step * VISION_EVENT_STEP_STRIDE + event);
        }
    }
    workgroupBarrier();

    if alive && ray_idx < VISION_RAYS && step == 0u {
        let first_event = atomicLoad(&vision_first_events[local_ray]);
        let event = first_event % VISION_EVENT_STEP_STRIDE;
        var hit_color = vec4<f32>(0.53, 0.81, 0.92, 1.0);
        var hit_depth = VISION_MAX_DIST;
        if first_event != VISION_EVENT_NONE && event != VISION_EVENT_SKY {
            let first_step = first_event / VISION_EVENT_STEP_STRIDE;
            hit_depth = f32(first_step + 1u) * VISION_STEP_SIZE;
            if event == VISION_EVENT_FOOD {
                hit_color = vec4<f32>(0.7, 0.95, 0.2, 1.0);
            } else if event == VISION_EVENT_AGENT {
                hit_color = vec4<f32>(0.9, 0.2, 0.6, 1.0);
            } else if event == VISION_EVENT_TERRAIN_GREEN {
                hit_color = vec4<f32>(0.15, 0.5, 0.1, 1.0);
            } else if event == VISION_EVENT_TERRAIN_BROWN {
                hit_color = vec4<f32>(0.5, 0.4, 0.2, 1.0);
            } else {
                hit_color = vec4<f32>(0.6, 0.2, 0.1, 1.0);
            }
        }
        let s_base = agent_id * SENSORY_STRIDE;
        let ci = s_base + ray_idx * 4u;
        sensory_buffer[ci] = hit_color.x;
        sensory_buffer[ci + 1u] = hit_color.y;
        sensory_buffer[ci + 2u] = hit_color.z;
        sensory_buffer[ci + 3u] = hit_color.w;
        sensory_buffer[s_base + VISION_COLOR_COUNT + ray_idx] = hit_depth / VISION_MAX_DIST;
    }
}
