// ── Phase: Vision ──────────────────────────────────────────────────────────
// Two sub-phases:
//   vision_single_ray  — one ray, used by multi-workgroup vision_tick kernel
//   phase_vision_senses — per-agent proprioception / interoception / touch

// ── Single ray march ──────────────────────────────────────────────────────
// Called by vision_tick entry point: one thread handles one ray.
// agent_id and ray_idx are derived from global_invocation_id.

fn vision_single_ray(agent_id: u32, ray_idx: u32) {
    let base = agent_id * PHYS_STRIDE;
    if (physics_state[base + P_ALIVE] < 0.5) { return; }

    let s_base = agent_id * SENSORY_STRIDE;

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

    let grid_width  = wc_u32(WC_GRID_WIDTH);
    let grid_offset = i32(wc_u32(WC_GRID_OFFSET));

    // ── Ray direction ─────────────────────────────────────────────────
    let col = ray_idx % VISION_W;
    let row = ray_idx / VISION_W;
    let u = (f32(col) / f32(VISION_W - 1u)) * 2.0 - 1.0;
    let v = (f32(row) / f32(VISION_H - 1u)) * 2.0 - 1.0;
    // Heritable angles of view (degrees), clamped as the genes are bounded.
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

    // ── Ray march ─────────────────────────────────────────────────────
    var hit_color = vec4<f32>(0.53, 0.81, 0.92, 1.0);
    var hit_depth = VISION_MAX_DIST;
    var hit = false;
    // The agent-grid block the ray last checked, and whether it held another
    // live agent. Consecutive samples (1.2 apart) stay in one 8-unit cell for
    // several steps; while the block holds no other live agent no sample can
    // hit one, so the block is only rescanned when the sample enters a new
    // centre cell, and the hit tests run only while it is occupied.
    var agent_block_cx = AGENT_BLOCK_NONE;
    var agent_block_cz = AGENT_BLOCK_NONE;
    var agent_block_occupied = false;

    for (var step: u32 = 0u; step < VISION_NUM_STEPS; step++) {
        if hit { break; }

        let t = f32(step + 1u) * VISION_STEP_SIZE;
        let ray_pos = pos + ray_dir * t;

        // ── Check food grid ───────────────────────────────────────
        // Only the cells overlapping the box around the sample point can
        // hold a food item within reach (see FOOD_PROBE_HALF_WIDTH): one to
        // four cells instead of the 3x3 block.
        let food_cx_low = cell_coord(ray_pos.x - FOOD_PROBE_HALF_WIDTH) + grid_offset;
        let food_cx_high = cell_coord(ray_pos.x + FOOD_PROBE_HALF_WIDTH) + grid_offset;
        let food_cz_low = cell_coord(ray_pos.z - FOOD_PROBE_HALF_WIDTH) + grid_offset;
        let food_cz_high = cell_coord(ray_pos.z + FOOD_PROBE_HALF_WIDTH) + grid_offset;

        for (var ncx = food_cx_low; ncx <= food_cx_high; ncx++) {
            if hit { break; }
            for (var ncz = food_cz_low; ncz <= food_cz_high; ncz++) {
                if hit { break; }
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
                        hit_color = vec4<f32>(0.7, 0.95, 0.2, 1.0);
                        hit_depth = t;
                        hit = true;
                        break;
                    }
                }
            }
        }
        if hit { break; }

        // ── Check agent grid ──────────────────────────────────────
        let ag_cx = cell_coord(ray_pos.x) + grid_offset;
        let ag_cz = cell_coord(ray_pos.z) + grid_offset;
        if (ag_cx != agent_block_cx || ag_cz != agent_block_cz) {
            agent_block_cx = ag_cx;
            agent_block_cz = ag_cz;
            agent_block_occupied = agent_block_has_other_live_agent(
                agent_id, ag_cx, ag_cz, grid_width);
        }

        for (var di: i32 = -1; di <= 1 && agent_block_occupied; di++) {
            if hit { break; }
            for (var dj: i32 = -1; dj <= 1; dj++) {
                if hit { break; }
                let ncx = ag_cx + di;
                let ncz = ag_cz + dj;
                if ncx < 0 || ncz < 0 { continue; }
                let uncx = u32(ncx);
                let uncz = u32(ncz);
                if uncx >= grid_width || uncz >= grid_width { continue; }

                let cell_idx = uncx * grid_width + uncz;
                let cell_base = cell_idx * AGENT_GRID_CELL_STRIDE;
                let count = min(u32(agent_grid[cell_base]), AGENT_GRID_MAX_PER_CELL);

                for (var s: u32 = 0u; s < count; s++) {
                    let other = u32(agent_grid[cell_base + 1u + s]);
                    if other == agent_id { continue; }

                    let ob = other * PHYS_STRIDE;
                    if physics_state[ob + P_ALIVE] < 0.5 { continue; }

                    let ox = physics_state[ob + P_POS_X];
                    let oy = physics_state[ob + P_POS_Y];
                    let oz = physics_state[ob + P_POS_Z];

                    let dx = ray_pos.x - ox;
                    let dy = ray_pos.y - oy;
                    let dz = ray_pos.z - oz;
                    let dist_sq = dx * dx + dy * dy + dz * dz;

                    if dist_sq < AGENT_RAY_RADIUS_SQ {
                        hit_color = vec4<f32>(0.9, 0.2, 0.6, 1.0);
                        hit_depth = t;
                        hit = true;
                        break;
                    }
                }
            }
        }
        if hit { break; }

        // ── Check terrain ─────────────────────────────────────────
        let ground_h = sample_height(ray_pos.x, ray_pos.z);

        if ray_pos.y <= ground_h {
            let biome_type = sample_biome(ray_pos.x, ray_pos.z);
            if biome_type == 0u {
                hit_color = vec4<f32>(0.15, 0.5, 0.1, 1.0);
            } else if biome_type == 1u {
                hit_color = vec4<f32>(0.5, 0.4, 0.2, 1.0);
            } else {
                hit_color = vec4<f32>(0.6, 0.2, 0.1, 1.0);
            }
            hit_depth = t;
            hit = true;
            break;
        }

        if ray_dir.y > 0.3 && ray_pos.y > ground_h + 5.0 {
            break;
        }
    }

    // ── Write vision results ──────────────────────────────────────────
    let ci = s_base + ray_idx * 4u;
    sensory_buffer[ci]      = hit_color.x;
    sensory_buffer[ci + 1u] = hit_color.y;
    sensory_buffer[ci + 2u] = hit_color.z;
    sensory_buffer[ci + 3u] = hit_color.w;
    sensory_buffer[s_base + VISION_COLOR_COUNT + ray_idx] = hit_depth / VISION_MAX_DIST;
}

// Whether the 3x3 agent-grid block centred on cell (cx, cz) holds a live
// agent other than `agent_id`: the only agents a ray sample in that centre
// cell can hit.
fn agent_block_has_other_live_agent(agent_id: u32, cx: i32, cz: i32, grid_width: u32) -> bool {
    for (var di: i32 = -1; di <= 1; di++) {
        for (var dj: i32 = -1; dj <= 1; dj++) {
            let ncx = cx + di;
            let ncz = cz + dj;
            if ncx < 0 || ncz < 0 { continue; }
            let uncx = u32(ncx);
            let uncz = u32(ncz);
            if uncx >= grid_width || uncz >= grid_width { continue; }
            let cell_base = (uncx * grid_width + uncz) * AGENT_GRID_CELL_STRIDE;
            let count = min(u32(agent_grid[cell_base]), AGENT_GRID_MAX_PER_CELL);
            for (var s: u32 = 0u; s < count; s++) {
                let other = u32(agent_grid[cell_base + 1u + s]);
                if other != agent_id && physics_state[other * PHYS_STRIDE + P_ALIVE] >= 0.5 {
                    return true;
                }
            }
        }
    }
    return false;
}

// ── Per-agent non-visual senses ───────────────────────────────────────────

fn phase_vision_senses(tid: u32) {
    let base = tid * PHYS_STRIDE;
    if (physics_state[base + P_ALIVE] < 0.5) { return; }

    let agent_count = wc_u32(WC_AGENT_COUNT);
    if tid >= agent_count { return; }

    let s_base = tid * SENSORY_STRIDE;
    let grid_width  = wc_u32(WC_GRID_WIDTH);
    let grid_offset = i32(wc_u32(WC_GRID_OFFSET));

    let pos = vec3<f32>(
        physics_state[base + P_POS_X],
        physics_state[base + P_POS_Y],
        physics_state[base + P_POS_Z],
    );

    let nv_base = s_base + VISION_COLOR_COUNT + VISION_DEPTH_COUNT;
    var off = nv_base;

    sensory_buffer[off]      = physics_state[base + P_VEL_X];
    sensory_buffer[off + 1u] = physics_state[base + P_VEL_Y];
    sensory_buffer[off + 2u] = physics_state[base + P_VEL_Z];
    off += 3u;

    sensory_buffer[off]      = physics_state[base + P_FACING_X];
    sensory_buffer[off + 1u] = physics_state[base + P_FACING_Y];
    sensory_buffer[off + 2u] = physics_state[base + P_FACING_Z];
    off += 3u;

    sensory_buffer[off] = physics_state[base + P_ANGULAR_VEL];
    off += 1u;

    let energy = physics_state[base + P_ENERGY];
    let max_energy = physics_state[base + P_MAX_ENERGY];
    sensory_buffer[off] = energy / max(max_energy, 1e-6);
    off += 1u;

    let integrity = physics_state[base + P_INTEGRITY];
    let max_integrity = physics_state[base + P_MAX_INTEGRITY];
    sensory_buffer[off] = integrity / max(max_integrity, 1e-6);
    off += 1u;

    let prev_energy = physics_state[base + P_PREV_ENERGY];
    sensory_buffer[off] = energy - prev_energy;
    off += 1u;

    let prev_integrity = physics_state[base + P_PREV_INTEGRITY];
    sensory_buffer[off] = integrity - prev_integrity;
    off += 1u;

    // ── Touch contacts ────────────────────────────────────────────────
    var touch_count: u32 = 0u;
    let touch_base = off;

    for (var i: u32 = 0u; i < MAX_TOUCH_CONTACTS * 4u; i++) {
        sensory_buffer[touch_base + i] = 0.0;
    }

    // Hazard contact first: present-moment damage must never be evicted by
    // lower-stakes contacts when the four slots fill. Zero planar direction
    // (the hazard is the ground underfoot), fixed intensity — mirrors the
    // CPU reference in agent/senses.rs.
    if (sample_biome(pos.x, pos.z) == BIOME_DANGER) {
        sensory_buffer[touch_base]      = 0.0;
        sensory_buffer[touch_base + 1u] = 0.0;
        sensory_buffer[touch_base + 2u] = TOUCH_HAZARD_INTENSITY;
        sensory_buffer[touch_base + 3u] = f32(TOUCH_HAZARD) / 4.0;
        touch_count = 1u;
    }

    let self_cx = cell_coord(pos.x) + grid_offset;
    let self_cz = cell_coord(pos.z) + grid_offset;

    for (var di: i32 = -1; di <= 1; di++) {
        for (var dj: i32 = -1; dj <= 1; dj++) {
            if touch_count >= MAX_TOUCH_CONTACTS { break; }
            let ncx = self_cx + di;
            let ncz = self_cz + dj;
            if ncx < 0 || ncz < 0 { continue; }
            let uncx = u32(ncx);
            let uncz = u32(ncz);
            if uncx >= grid_width || uncz >= grid_width { continue; }

            let cell_idx = uncx * grid_width + uncz;
            let cell_base = cell_idx * FOOD_GRID_CELL_STRIDE;
            let count = min(u32(food_grid[cell_base]), FOOD_GRID_MAX_PER_CELL);

            for (var s: u32 = 0u; s < count; s++) {
                if touch_count >= MAX_TOUCH_CONTACTS { break; }
                let fidx = u32(food_grid[cell_base + 1u + s]);
                if food_flags[fidx] != 0u { continue; }

                let fbase = fidx * FOOD_STATE_STRIDE;
                let fdx = food_state[fbase + FOOD_POSITION_X] - pos.x;
                let fdz = food_state[fbase + FOOD_POSITION_Z] - pos.z;
                let dist = sqrt(fdx * fdx + fdz * fdz);

                if dist < TOUCH_FOOD_RANGE {
                    let slot = touch_base + touch_count * 4u;
                    let inv_dist = 1.0 / max(dist, 1e-6);
                    sensory_buffer[slot]      = fdx * inv_dist;
                    sensory_buffer[slot + 1u] = fdz * inv_dist;
                    sensory_buffer[slot + 2u] = 1.0 - dist / TOUCH_FOOD_RANGE;
                    sensory_buffer[slot + 3u] = f32(TOUCH_FOOD) / 4.0;
                    touch_count += 1u;
                }
            }
        }
        if touch_count >= MAX_TOUCH_CONTACTS { break; }
    }

    for (var di: i32 = -1; di <= 1; di++) {
        for (var dj: i32 = -1; dj <= 1; dj++) {
            if touch_count >= MAX_TOUCH_CONTACTS { break; }
            let ncx = self_cx + di;
            let ncz = self_cz + dj;
            if ncx < 0 || ncz < 0 { continue; }
            let uncx = u32(ncx);
            let uncz = u32(ncz);
            if uncx >= grid_width || uncz >= grid_width { continue; }

            let cell_idx = uncx * grid_width + uncz;
            let cell_base = cell_idx * AGENT_GRID_CELL_STRIDE;
            let count = min(u32(agent_grid[cell_base]), AGENT_GRID_MAX_PER_CELL);

            for (var s: u32 = 0u; s < count; s++) {
                if touch_count >= MAX_TOUCH_CONTACTS { break; }
                let other = u32(agent_grid[cell_base + 1u + s]);
                if other == tid { continue; }

                let ob = other * PHYS_STRIDE;
                if physics_state[ob + P_ALIVE] < 0.5 { continue; }

                let adx = physics_state[ob + P_POS_X] - pos.x;
                let adz = physics_state[ob + P_POS_Z] - pos.z;
                let dist = sqrt(adx * adx + adz * adz);

                if dist < TOUCH_AGENT_RANGE {
                    let slot = touch_base + touch_count * 4u;
                    let inv_dist = 1.0 / max(dist, 1e-6);
                    sensory_buffer[slot]      = adx * inv_dist;
                    sensory_buffer[slot + 1u] = adz * inv_dist;
                    sensory_buffer[slot + 2u] = 1.0 - dist / TOUCH_AGENT_RANGE;
                    sensory_buffer[slot + 3u] = f32(TOUCH_AGENT) / 4.0;
                    touch_count += 1u;
                }
            }
        }
        if touch_count >= MAX_TOUCH_CONTACTS { break; }
    }

    // Terrain-edge contacts: the world boundary pushes back. Direction
    // points inward (away from the wall), intensity rises as the wall
    // nears — mirrors the CPU reference in agent/senses.rs.
    let world_half_for_touch = wc_f32(WC_WORLD_HALF_BOUND);
    // `var` (not `let`): naga requires a mutable binding for dynamic
    // indexing.
    var wall_distances = array<f32, 4>(
        pos.x + world_half_for_touch,   // distance to the −X wall
        world_half_for_touch - pos.x,   // distance to the +X wall
        pos.z + world_half_for_touch,   // distance to the −Z wall
        world_half_for_touch - pos.z,   // distance to the +Z wall
    );
    var inward_x = array<f32, 4>(1.0, -1.0, 0.0, 0.0);
    var inward_z = array<f32, 4>(0.0, 0.0, 1.0, -1.0);
    for (var wall: u32 = 0u; wall < 4u; wall++) {
        if (touch_count >= MAX_TOUCH_CONTACTS) { break; }
        let wall_distance = wall_distances[wall];
        if (wall_distance < TOUCH_EDGE_RANGE) {
            let slot = touch_base + touch_count * 4u;
            sensory_buffer[slot]      = inward_x[wall];
            sensory_buffer[slot + 1u] = inward_z[wall];
            sensory_buffer[slot + 2u] = 1.0 - max(wall_distance, 0.0) / TOUCH_EDGE_RANGE;
            sensory_buffer[slot + 3u] = f32(TOUCH_TERRAIN_EDGE) / 4.0;
            touch_count += 1u;
        }
    }

    // ── Smell ─────────────────────────────────────────────────────────
    let scent_base = touch_base + MAX_TOUCH_CONTACTS * 4u;
    let facing = vec3<f32>(
        physics_state[base + P_FACING_X],
        physics_state[base + P_FACING_Y],
        physics_state[base + P_FACING_Z],
    );
    let smell_strength = clamp(
        brain_state[tid * BRAIN_STRIDE + O_SMELL_STRENGTH], SMELL_STRENGTH_MIN, SMELL_STRENGTH_MAX);
    let scent = sense_scent(pos, facing, smell_strength);
    sensory_buffer[scent_base] = scent.x;
    sensory_buffer[scent_base + 1u] = scent.y;
}

// Perceived food odour at the left and right nostrils. The nostrils sit
// NOSTRIL_FORWARD_OFFSET ahead of the body and NOSTRIL_SIDE_OFFSET to either
// side (left = −right, matching the vision columns). Every uneaten food item
// within SCENT_RANGE of a nostril adds exp(−d / SCENT_DECAY_LENGTH) to that
// nostril's concentration C; the nostril perceives 1 − exp(−strength · C),
// so 0 strength smells nothing and a strong nose saturates up close.
// Mirrors `sense_scent` in the sandbox's agent/senses.rs.
fn sense_scent(pos: vec3<f32>, facing: vec3<f32>, strength: f32) -> vec2<f32> {
    let right = vec3<f32>(facing.z, 0.0, -facing.x);
    let nose = pos + facing * NOSTRIL_FORWARD_OFFSET;
    let left_nostril = nose - right * NOSTRIL_SIDE_OFFSET;
    let right_nostril = nose + right * NOSTRIL_SIDE_OFFSET;
    let range_sq = SCENT_RANGE * SCENT_RANGE;
    var concentration = vec2<f32>(0.0, 0.0);
    let food_count = wc_u32(WC_FOOD_COUNT);
    for (var f: u32 = 0u; f < food_count; f++) {
        if (food_flags[f] != 0u) { continue; }
        let fbase = f * FOOD_STATE_STRIDE;
        let fx = food_state[fbase + FOOD_POSITION_X];
        let fz = food_state[fbase + FOOD_POSITION_Z];
        let left_dx = fx - left_nostril.x;
        let left_dz = fz - left_nostril.z;
        let left_sq = left_dx * left_dx + left_dz * left_dz;
        if (left_sq < range_sq) {
            concentration.x += exp(-sqrt(left_sq) / SCENT_DECAY_LENGTH);
        }
        let right_dx = fx - right_nostril.x;
        let right_dz = fz - right_nostril.z;
        let right_sq = right_dx * right_dx + right_dz * right_dz;
        if (right_sq < range_sq) {
            concentration.y += exp(-sqrt(right_sq) / SCENT_DECAY_LENGTH);
        }
    }
    return vec2<f32>(1.0, 1.0) - exp(-strength * concentration);
}
