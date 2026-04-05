// ── Phase: Vision ──────────────────────────────────────────────────────────
// Two sub-phases:
//   vision_single_ray  — one ray, used by multi-workgroup vision_tick kernel
//   phase_vision_senses — per-agent proprioception / interoception / touch

// ── Single ray march ──────────────────────────────────────────────────────
// Called by vision_tick entry point: one thread handles one ray.
// agent_id and ray_idx are derived from global_invocation_id.

fn vision_single_ray(agent_id: u32, ray_idx: u32) {
    let base = agent_id * PHYS_STRIDE;
    if (agent_phys[base + P_ALIVE] < 0.5) { return; }

    let s_base = agent_id * SENSORY_STRIDE;

    let pos = vec3<f32>(
        agent_phys[base + P_POS_X],
        agent_phys[base + P_POS_Y],
        agent_phys[base + P_POS_Z],
    );
    let facing = vec3<f32>(
        agent_phys[base + P_FACING_X],
        agent_phys[base + P_FACING_Y],
        agent_phys[base + P_FACING_Z],
    );

    let grid_width  = wc_u32(WC_GRID_WIDTH);
    let grid_offset = i32(wc_u32(WC_GRID_OFFSET));

    // ── Ray direction ─────────────────────────────────────────────────
    let col = ray_idx % VISION_W;
    let row = ray_idx / VISION_W;
    let u = (f32(col) / f32(VISION_W - 1u)) * 2.0 - 1.0;
    let v = (f32(row) / f32(VISION_H - 1u)) * 2.0 - 1.0;
    let tan_hf = tan(VISION_FOV_HALF);
    let right = vec3<f32>(facing.z, 0.0, -facing.x);
    let ray_dir = normalize(facing + right * u * tan_hf + vec3<f32>(0.0, -v * tan_hf, 0.0));

    // ── Ray march ─────────────────────────────────────────────────────
    var hit_color = vec4<f32>(0.53, 0.81, 0.92, 1.0);
    var hit_depth = VISION_MAX_DIST;
    var hit = false;

    for (var step: u32 = 0u; step < VISION_NUM_STEPS; step++) {
        if hit { break; }

        let t = f32(step + 1u) * VISION_STEP_SIZE;
        let ray_pos = pos + ray_dir * t;

        // ── Check food grid ───────────────────────────────────────
        let food_cx = cell_coord(ray_pos.x) + grid_offset;
        let food_cz = cell_coord(ray_pos.z) + grid_offset;

        for (var di: i32 = -1; di <= 1; di++) {
            if hit { break; }
            for (var dj: i32 = -1; dj <= 1; dj++) {
                if hit { break; }
                let ncx = food_cx + di;
                let ncz = food_cz + dj;
                if ncx < 0 || ncz < 0 { continue; }
                let uncx = u32(ncx);
                let uncz = u32(ncz);
                if uncx >= grid_width || uncz >= grid_width { continue; }

                let cell_idx = uncx * grid_width + uncz;
                let cell_base = cell_idx * FOOD_GRID_CELL_STRIDE;
                let count = min(u32(atomicLoad(&food_grid[cell_base])), FOOD_GRID_MAX_PER_CELL);

                for (var s: u32 = 0u; s < count; s++) {
                    let fidx = u32(atomicLoad(&food_grid[cell_base + 1u + s]));
                    if atomicLoad(&food_flags[fidx]) != 0u { continue; }

                    let fbase = fidx * FOOD_STATE_STRIDE;
                    let fx = food_state[fbase + F_POS_X];
                    let fy = food_state[fbase + F_POS_Y];
                    let fz = food_state[fbase + F_POS_Z];

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

        for (var di: i32 = -1; di <= 1; di++) {
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
                let count = min(u32(atomicLoad(&agent_grid[cell_base])), AGENT_GRID_MAX_PER_CELL);

                for (var s: u32 = 0u; s < count; s++) {
                    let other = u32(atomicLoad(&agent_grid[cell_base + 1u + s]));
                    if other == agent_id { continue; }

                    let ob = other * PHYS_STRIDE;
                    if agent_phys[ob + P_ALIVE] < 0.5 { continue; }

                    let ox = agent_phys[ob + P_POS_X];
                    let oy = agent_phys[ob + P_POS_Y];
                    let oz = agent_phys[ob + P_POS_Z];

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
    sensory_buf[ci]      = hit_color.x;
    sensory_buf[ci + 1u] = hit_color.y;
    sensory_buf[ci + 2u] = hit_color.z;
    sensory_buf[ci + 3u] = hit_color.w;
    sensory_buf[s_base + VISION_COLOR_COUNT + ray_idx] = hit_depth / VISION_MAX_DIST;
}

// ── Per-agent non-visual senses ───────────────────────────────────────────

fn phase_vision_senses(tid: u32) {
    let base = tid * PHYS_STRIDE;
    if (agent_phys[base + P_ALIVE] < 0.5) { return; }

    let agent_count = wc_u32(WC_AGENT_COUNT);
    if tid >= agent_count { return; }

    let s_base = tid * SENSORY_STRIDE;
    let grid_width  = wc_u32(WC_GRID_WIDTH);
    let grid_offset = i32(wc_u32(WC_GRID_OFFSET));

    let pos = vec3<f32>(
        agent_phys[base + P_POS_X],
        agent_phys[base + P_POS_Y],
        agent_phys[base + P_POS_Z],
    );

    let nv_base = s_base + VISION_COLOR_COUNT + VISION_DEPTH_COUNT;
    var off = nv_base;

    sensory_buf[off]      = agent_phys[base + P_VEL_X];
    sensory_buf[off + 1u] = agent_phys[base + P_VEL_Y];
    sensory_buf[off + 2u] = agent_phys[base + P_VEL_Z];
    off += 3u;

    sensory_buf[off]      = agent_phys[base + P_FACING_X];
    sensory_buf[off + 1u] = agent_phys[base + P_FACING_Y];
    sensory_buf[off + 2u] = agent_phys[base + P_FACING_Z];
    off += 3u;

    sensory_buf[off] = agent_phys[base + P_ANGULAR_VEL];
    off += 1u;

    let energy = agent_phys[base + P_ENERGY];
    let max_energy = agent_phys[base + P_MAX_ENERGY];
    sensory_buf[off] = energy / max(max_energy, 1e-6);
    off += 1u;

    let integrity = agent_phys[base + P_INTEGRITY];
    let max_integrity = agent_phys[base + P_MAX_INTEGRITY];
    sensory_buf[off] = integrity / max(max_integrity, 1e-6);
    off += 1u;

    let prev_energy = agent_phys[base + P_PREV_ENERGY];
    sensory_buf[off] = energy - prev_energy;
    off += 1u;

    let prev_integrity = agent_phys[base + P_PREV_INTEGRITY];
    sensory_buf[off] = integrity - prev_integrity;
    off += 1u;

    // ── Touch contacts ────────────────────────────────────────────────
    var touch_count: u32 = 0u;
    let touch_base = off;

    for (var i: u32 = 0u; i < MAX_TOUCH_CONTACTS * 4u; i++) {
        sensory_buf[touch_base + i] = 0.0;
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
            let count = min(u32(atomicLoad(&food_grid[cell_base])), FOOD_GRID_MAX_PER_CELL);

            for (var s: u32 = 0u; s < count; s++) {
                if touch_count >= MAX_TOUCH_CONTACTS { break; }
                let fidx = u32(atomicLoad(&food_grid[cell_base + 1u + s]));
                if atomicLoad(&food_flags[fidx]) != 0u { continue; }

                let fbase = fidx * FOOD_STATE_STRIDE;
                let fdx = food_state[fbase + F_POS_X] - pos.x;
                let fdz = food_state[fbase + F_POS_Z] - pos.z;
                let dist = sqrt(fdx * fdx + fdz * fdz);

                if dist < TOUCH_FOOD_RANGE {
                    let slot = touch_base + touch_count * 4u;
                    let inv_dist = 1.0 / max(dist, 1e-6);
                    sensory_buf[slot]      = fdx * inv_dist;
                    sensory_buf[slot + 1u] = fdz * inv_dist;
                    sensory_buf[slot + 2u] = 1.0 - dist / TOUCH_FOOD_RANGE;
                    sensory_buf[slot + 3u] = f32(TOUCH_FOOD) / 4.0;
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
            let count = min(u32(atomicLoad(&agent_grid[cell_base])), AGENT_GRID_MAX_PER_CELL);

            for (var s: u32 = 0u; s < count; s++) {
                if touch_count >= MAX_TOUCH_CONTACTS { break; }
                let other = u32(atomicLoad(&agent_grid[cell_base + 1u + s]));
                if other == tid { continue; }

                let ob = other * PHYS_STRIDE;
                if agent_phys[ob + P_ALIVE] < 0.5 { continue; }

                let adx = agent_phys[ob + P_POS_X] - pos.x;
                let adz = agent_phys[ob + P_POS_Z] - pos.z;
                let dist = sqrt(adx * adx + adz * adz);

                if dist < TOUCH_AGENT_RANGE {
                    let slot = touch_base + touch_count * 4u;
                    let inv_dist = 1.0 / max(dist, 1e-6);
                    sensory_buf[slot]      = adx * inv_dist;
                    sensory_buf[slot + 1u] = adz * inv_dist;
                    sensory_buf[slot + 2u] = 1.0 - dist / TOUCH_AGENT_RANGE;
                    sensory_buf[slot + 3u] = f32(TOUCH_AGENT) / 4.0;
                    touch_count += 1u;
                }
            }
        }
        if touch_count >= MAX_TOUCH_CONTACTS { break; }
    }
}
