// Object queries retain the discrete ray samples and integer event ordering.
// Requires common.wgsl and phase_vision_parallel.wgsl. The caller supplies a
// 256-thread workgroup and enables this path only for populations of at most
// 32 agents and at most 256 food items.

override VISION_OBJECT_QUERIES: bool = false;
const VISION_OBJECT_FOOD_CAPACITY: u32 = 256u;
const VISION_OBJECT_AGENT_CAPACITY: u32 = 32u;
// A bounded binary search halves the remaining discrete sample interval.
const VISION_OBJECT_SEARCH_DIVISOR: u32 = 2u;
// The radius and its square are exactly representable for both object kinds.
const VISION_OBJECT_FOOD_RADIUS: f32 = 1.0;
const VISION_OBJECT_AGENT_RADIUS: f32 = 1.5;

const VISION_OBJECT_AGENT_BASE: u32 = VISION_OBJECT_FOOD_CAPACITY;
const VISION_OBJECT_DIRECTION_BASE: u32 = VISION_OBJECT_AGENT_BASE + VISION_OBJECT_AGENT_CAPACITY;
const VISION_OBJECT_ORIGIN_BASE: u32 = VISION_OBJECT_DIRECTION_BASE + VISION_PARALLEL_MAX_RAYS_PER_WORKGROUP;
const VISION_OBJECT_CACHE_SIZE: u32 = VISION_OBJECT_ORIGIN_BASE + 1u;
// One shared resource limits threadgroup-resource use when composed with the
// brain. Food stores position xyz and retained/live w;
// agents store position xyz and alive w; the final slots hold directions and
// the observer origin.
var<workgroup> vision_object_cache: array<vec4<f32>, VISION_OBJECT_CACHE_SIZE>;

fn vision_object_load_food(food_id: u32, grid_width: u32, grid_offset: i32) -> vec4<f32> {
    if food_flags[food_id] != 0u { return vec4<f32>(0.0); }
    let food_base = food_id * FOOD_STATE_STRIDE;
    let position = vec3<f32>(
        food_state[food_base + FOOD_POSITION_X],
        food_state[food_base + FOOD_POSITION_Y],
        food_state[food_base + FOOD_POSITION_Z],
    );
    let cell_x = cell_coord(position.x) + grid_offset;
    let cell_z = cell_coord(position.z) + grid_offset;
    if cell_x < 0 || cell_z < 0 { return vec4<f32>(0.0); }
    let grid_x = u32(cell_x);
    let grid_z = u32(cell_z);
    if grid_x >= grid_width || grid_z >= grid_width { return vec4<f32>(0.0); }
    let cell_base = (grid_x * grid_width + grid_z) * FOOD_GRID_CELL_STRIDE;
    let count = min(u32(food_grid[cell_base]), FOOD_GRID_MAX_PER_CELL);
    for (var slot = 0u; slot < count; slot++) {
        if u32(food_grid[cell_base + 1u + slot]) == food_id {
            return vec4<f32>(position, 1.0);
        }
    }
    return vec4<f32>(0.0);
}

fn vision_object_direction(agent_id: u32, ray_idx: u32) -> vec3<f32> {
    let base = agent_id * PHYS_STRIDE;
    let facing = vec3<f32>(
        physics_state[base + P_FACING_X],
        physics_state[base + P_FACING_Y],
        physics_state[base + P_FACING_Z],
    );
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
    return normalize(
        facing + right * u * tan_half_horizontal + vec3<f32>(0.0, -v * tan_half_vertical, 0.0));
}

fn vision_object_sample_position(origin: vec3<f32>, direction: vec3<f32>, step: u32) -> vec3<f32> {
    let t = f32(step + 1u) * VISION_STEP_SIZE;
    return origin + direction * t;
}

// Registered coordinates can differ from current positions after collisions.
// Membership is checked for every geometric hit, since the first geometric
// hit need not be the first sample whose registered block contains the agent.
fn vision_object_agent_registered(other: u32, ray_pos: vec3<f32>) -> bool {
    let grid_width = wc_u32(WC_GRID_WIDTH);
    let grid_offset = i32(wc_u32(WC_GRID_OFFSET));
    let center_x = cell_coord(ray_pos.x) + grid_offset;
    let center_z = cell_coord(ray_pos.z) + grid_offset;
    if VISION_AGENT_MASKS && wc_u32(WC_AGENT_COUNT) <= AGENT_VISIBILITY_MASK_BITS
        && center_x >= 0 && center_z >= 0
        && u32(center_x) < grid_width && u32(center_z) < grid_width {
        let center_cell = u32(center_x) * grid_width + u32(center_z);
        let mask = agent_grid[agent_visibility_mask_offset(grid_width) + center_cell];
        return (mask & (1u << other)) != 0u;
    }
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
                if u32(agent_grid[cell_base + 1u + slot]) == other { return true; }
            }
        }
    }
    return false;
}

// The dominant coordinate bounds the shortest conservative sample interval.
fn vision_object_dominant_axis(direction: vec3<f32>) -> u32 {
    let magnitude = abs(direction);
    if magnitude.x >= magnitude.y && magnitude.x >= magnitude.z { return 0u; }
    if magnitude.y >= magnitude.z { return 1u; }
    return 2u;
}

fn vision_object_signed_delta(
    origin: vec3<f32>, direction: vec3<f32>, position: vec3<f32>, axis: u32, step: u32,
) -> f32 {
    let ray_pos = vision_object_sample_position(origin, direction, step);
    let delta = ray_pos[axis] - position[axis];
    return select(-delta, delta, direction[axis] >= 0.0);
}

// Food uses VISION_EVENT_NONE as registered_agent; agents supply their ID.
// A true spherical hit requires every coordinate difference to lie strictly
// between -radius and radius. Ordered f32 sample-coordinate arithmetic is
// monotone, so endpoint rejection and binary search discard only misses.
fn vision_object_first_hit(
    origin: vec3<f32>, direction: vec3<f32>, position: vec3<f32>,
    radius: f32, radius_squared: f32, registered_agent: u32,
) -> u32 {
    let first_delta = vision_object_sample_position(origin, direction, 0u) - position;
    let last_delta = vision_object_sample_position(origin, direction, VISION_NUM_STEPS - 1u) - position;
    if any(min(first_delta, last_delta) >= vec3<f32>(radius))
        || any(max(first_delta, last_delta) <= vec3<f32>(-radius)) {
        return VISION_EVENT_NONE;
    }

    let axis = vision_object_dominant_axis(direction);
    var lower = 0u;
    var upper = VISION_NUM_STEPS;
    while lower < upper {
        let middle = (lower + upper) / VISION_OBJECT_SEARCH_DIVISOR;
        let delta = vision_object_signed_delta(origin, direction, position, axis, middle);
        if delta <= -radius {
            lower = middle + 1u;
        } else {
            upper = middle;
        }
    }

    for (var step = lower; step < VISION_NUM_STEPS; step++) {
        let ray_pos = vision_object_sample_position(origin, direction, step);
        let axis_delta = ray_pos[axis] - position[axis];
        let signed_delta = select(-axis_delta, axis_delta, direction[axis] >= 0.0);
        if signed_delta >= radius { break; }
        let dx = ray_pos.x - position.x;
        let dy = ray_pos.y - position.y;
        let dz = ray_pos.z - position.z;
        let dist_sq = dx * dx + dy * dy + dz * dz;
        if dist_sq < radius_squared {
            if registered_agent == VISION_EVENT_NONE {
                return step;
            }
            if vision_object_agent_registered(registered_agent, ray_pos) {
                return step;
            }
        }
    }
    return VISION_EVENT_NONE;
}

fn vision_object_terrain_event(origin: vec3<f32>, direction: vec3<f32>, step: u32) -> u32 {
    let ray_pos = vision_object_sample_position(origin, direction, step);
    let ground_h = sample_height(ray_pos.x, ray_pos.z);
    if ray_pos.y <= ground_h {
        let biome_type = sample_biome(ray_pos.x, ray_pos.z);
        if biome_type == 0u { return VISION_EVENT_TERRAIN_GREEN; }
        if biome_type == 1u { return VISION_EVENT_TERRAIN_BROWN; }
        return VISION_EVENT_TERRAIN_RED;
    }
    if direction.y > 0.3 && ray_pos.y > ground_h + 5.0 { return VISION_EVENT_SKY; }
    return VISION_EVENT_NONE;
}

fn vision_object_write_ray(agent_id: u32, ray_idx: u32, first_event: u32) {
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

// All 256 invocations participate in cache setup and both barriers, including
// padding rays and dead observers. Each 32-lane ray group queries object IDs
// in strides of 32 and reduces its earliest event independently.
fn vision_object_rays(agent_id: u32, first_ray: u32, tid: u32) {
    let local_ray = tid / VISION_PARALLEL_LANES_PER_RAY;
    let lane = tid % VISION_PARALLEL_LANES_PER_RAY;
    let ray_idx = first_ray + local_ray;
    let food_count = wc_u32(WC_FOOD_COUNT);
    let agent_count = wc_u32(WC_AGENT_COUNT);
    let base = agent_id * PHYS_STRIDE;
    let alive = !(physics_state[base + P_ALIVE] < 0.5);

    if tid == 0u {
        vision_object_cache[VISION_OBJECT_ORIGIN_BASE] = vec4<f32>(
            physics_state[base + P_POS_X],
            physics_state[base + P_POS_Y],
            physics_state[base + P_POS_Z],
            0.0,
        );
    }
    if tid < food_count {
        vision_object_cache[tid] = vision_object_load_food(
            tid, wc_u32(WC_GRID_WIDTH), i32(wc_u32(WC_GRID_OFFSET)));
    }
    if tid < agent_count {
        let other_base = tid * PHYS_STRIDE;
        vision_object_cache[VISION_OBJECT_AGENT_BASE + tid] = vec4<f32>(
            physics_state[other_base + P_POS_X],
            physics_state[other_base + P_POS_Y],
            physics_state[other_base + P_POS_Z],
            physics_state[other_base + P_ALIVE],
        );
    }
    if lane == 0u {
        atomicStore(&vision_first_events[local_ray], VISION_EVENT_NONE);
        if alive && ray_idx < VISION_RAYS {
            vision_object_cache[VISION_OBJECT_DIRECTION_BASE + local_ray] =
                vec4<f32>(vision_object_direction(agent_id, ray_idx), 0.0);
        }
    }
    workgroupBarrier();

    if alive && ray_idx < VISION_RAYS {
        let origin = vision_object_cache[VISION_OBJECT_ORIGIN_BASE].xyz;
        let direction = vision_object_cache[VISION_OBJECT_DIRECTION_BASE + local_ray].xyz;
        var first_event = VISION_EVENT_NONE;
        if lane < VISION_NUM_STEPS {
            let terrain_event = vision_object_terrain_event(origin, direction, lane);
            if terrain_event != VISION_EVENT_NONE {
                first_event = lane * VISION_EVENT_STEP_STRIDE + terrain_event;
            }
        }
        for (var food_id = lane; food_id < food_count; food_id += VISION_PARALLEL_LANES_PER_RAY) {
            let food = vision_object_cache[food_id];
            if food.w == 0.0 { continue; }
            let step = vision_object_first_hit(origin, direction, food.xyz,
                VISION_OBJECT_FOOD_RADIUS, FOOD_RAY_RADIUS_SQ, VISION_EVENT_NONE);
            if step != VISION_EVENT_NONE {
                first_event = min(first_event, step * VISION_EVENT_STEP_STRIDE + VISION_EVENT_FOOD);
            }
        }
        if lane < agent_count && lane != agent_id {
            let other = vision_object_cache[VISION_OBJECT_AGENT_BASE + lane];
            if !(other.w < 0.5) {
                let step = vision_object_first_hit(origin, direction, other.xyz,
                    VISION_OBJECT_AGENT_RADIUS, AGENT_RAY_RADIUS_SQ, lane);
                if step != VISION_EVENT_NONE {
                    first_event = min(first_event, step * VISION_EVENT_STEP_STRIDE + VISION_EVENT_AGENT);
                }
            }
        }
        if first_event != VISION_EVENT_NONE {
            atomicMin(&vision_first_events[local_ray], first_event);
        }
    }
    workgroupBarrier();

    if alive && ray_idx < VISION_RAYS && lane == 0u {
        vision_object_write_ray(agent_id, ray_idx, atomicLoad(&vision_first_events[local_ray]));
    }
}
