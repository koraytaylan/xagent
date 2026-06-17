// Phase: physics — movement, gravity, terrain collision, energy, biome damage, death.
// Ported from physics.wgsl. 1 thread per agent, tid = agent index.

fn phase_physics(tid: u32, tick: u32) {
    let agent = tid;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    if agent >= agent_count { return; }
    let b = agent * PHYS_STRIDE;

    let alive = physics_state[b + P_ALIVE];
    if alive < 0.5 { return; }

    let dt = wc_f32(WC_DT);
    let world_half = wc_f32(WC_WORLD_HALF_BOUND);

    // Snapshot prev energy/integrity
    physics_state[b + P_PREV_ENERGY] = physics_state[b + P_ENERGY];
    physics_state[b + P_PREV_INTEGRITY] = physics_state[b + P_INTEGRITY];

    // Save last-good position/velocity for NaN recovery
    let last_pos = vec3<f32>(physics_state[b + P_POS_X], physics_state[b + P_POS_Y], physics_state[b + P_POS_Z]);
    let last_vel = vec3<f32>(physics_state[b + P_VEL_X], physics_state[b + P_VEL_Y], physics_state[b + P_VEL_Z]);

    // Read motor commands from decision_buffer
    let decision_base = agent * DECISION_STRIDE;
    let motor_offset = decision_base + DECISION_MOTOR;
    var motor_forward = decision_buffer[motor_offset];
    var motor_turn = decision_buffer[motor_offset + 1u];
    var motor_strafe = decision_buffer[motor_offset + 2u];

    // Sanitize motor: clamp [-1,1], NaN -> 0
    if !is_finite(motor_forward) { motor_forward = 0.0; }
    if !is_finite(motor_turn) { motor_turn = 0.0; }
    if !is_finite(motor_strafe) { motor_strafe = 0.0; }
    motor_forward = clamp(motor_forward, -1.0, 1.0);
    motor_turn = clamp(motor_turn, -1.0, 1.0);
    motor_strafe = clamp(motor_strafe, -1.0, 1.0);

    // Turning
    var yaw = physics_state[b + P_YAW];
    let prev_yaw = yaw;
    yaw += motor_turn * TURN_SPEED * dt;
    physics_state[b + P_YAW] = yaw;
    physics_state[b + P_ANGULAR_VEL] = (yaw - prev_yaw) / max(dt, 1e-6);
    let facing = normalize(vec3<f32>(sin(yaw), 0.0, cos(yaw)));
    physics_state[b + P_FACING_X] = facing.x;
    physics_state[b + P_FACING_Y] = 0.0;
    physics_state[b + P_FACING_Z] = facing.z;

    // Locomotion
    let right = vec3<f32>(facing.z, 0.0, -facing.x);
    var desired = facing * motor_forward + right * motor_strafe;
    let desired_sq = dot(desired, desired);
    if desired_sq > 1.0 {
        desired = desired / sqrt(desired_sq);
    }
    let move_speed = brain_state[agent * BRAIN_STRIDE + O_MOVEMENT_SPEED];
    physics_state[b + P_VEL_X] = desired.x * move_speed;
    physics_state[b + P_VEL_Z] = desired.z * move_speed;

    // Gravity
    physics_state[b + P_VEL_Y] = physics_state[b + P_VEL_Y] - GRAVITY * dt;

    // Integrate position
    var pos = vec3<f32>(
        physics_state[b + P_POS_X] + physics_state[b + P_VEL_X] * dt,
        physics_state[b + P_POS_Y] + physics_state[b + P_VEL_Y] * dt,
        physics_state[b + P_POS_Z] + physics_state[b + P_VEL_Z] * dt,
    );

    // Bounce off world bounds: clamp position and reflect velocity/facing
    let pre_clamp_x = pos.x;
    let pre_clamp_z = pos.z;
    pos.x = clamp(pos.x, -world_half, world_half);
    pos.z = clamp(pos.z, -world_half, world_half);
    var bounced = false;
    if (pos.x != pre_clamp_x) {
        physics_state[b + P_VEL_X] *= -1.0;
        yaw = -yaw;
        bounced = true;
    }
    if (pos.z != pre_clamp_z) {
        physics_state[b + P_VEL_Z] *= -1.0;
        yaw = PI - yaw;
        bounced = true;
    }
    if (bounced) {
        physics_state[b + P_YAW] = yaw;
        physics_state[b + P_FACING_X] = sin(yaw);
        physics_state[b + P_FACING_Z] = cos(yaw);
        var yaw_delta = yaw - prev_yaw;
        if (yaw_delta > PI) { yaw_delta -= TWO_PI; }
        if (yaw_delta < -PI) { yaw_delta += TWO_PI; }
        physics_state[b + P_ANGULAR_VEL] = yaw_delta / max(dt, 1e-6);
    }

    // Ground collision (use sample_height from common.wgsl)
    let ground = sample_height(pos.x, pos.z);
    if pos.y < ground + AGENT_HALF_HEIGHT {
        pos.y = ground + AGENT_HALF_HEIGHT;
        physics_state[b + P_VEL_Y] = 0.0;
    }

    // NaN recovery
    if !is_finite(pos.x) || !is_finite(pos.y) || !is_finite(pos.z) {
        pos = last_pos;
        physics_state[b + P_VEL_X] = last_vel.x;
        physics_state[b + P_VEL_Y] = last_vel.y;
        physics_state[b + P_VEL_Z] = last_vel.z;
    }

    physics_state[b + P_POS_X] = pos.x;
    physics_state[b + P_POS_Y] = pos.y;
    physics_state[b + P_POS_Z] = pos.z;

    // Accumulate distance traveled (planar displacement this tick)
    let step_len = length(vec2<f32>(pos.x - last_pos.x, pos.z - last_pos.z));
    physics_state[b + P_DISTANCE_TRAVELED] += step_len;

    // Energy depletion (scaled by metabolic_rate from brain config)
    let metabolic_rate = bc_f32(CFG_METABOLIC_RATE);
    // Super-linear drag (plan 0009, Layer A): above-baseline-only cost exponent.
    // Mirror of the identical block in kernel_tick.wgsl — both paths MUST match.
    // The exponent only activates when speed_ratio >= 1.0 (above baseline). Below
    // baseline the drag is exactly speed_ratio (same as k=1.0), so sub-baseline
    // drain is unchanged between k=1.0 and k>1.0: no torpor gradient.
    //
    //   k=1.0:                  drag = speed_ratio          (exact old expression, always)
    //   k>1.0, speed_ratio < 1: drag = speed_ratio          (unchanged from k=1.0)
    //   k>1.0, speed_ratio >= 1: drag = pow(speed_ratio, k) (super-linear above baseline)
    let speed_ratio = move_speed / 20.0;
    let speed_cost_exponent = wc_f32(WC_SPEED_COST_EXPONENT);
    let above_baseline = speed_ratio >= 1.0;
    let super_linear_drag = select(speed_ratio, pow(speed_ratio, speed_cost_exponent), above_baseline);
    let drag = select(super_linear_drag, speed_ratio, speed_cost_exponent == 1.0);
    let movement_mag = min(abs(motor_forward) + abs(motor_strafe), 1.414) * drag;
    var energy = physics_state[b + P_ENERGY];
    let depletion_drain = wc_f32(WC_ENERGY_DEPLETION) * metabolic_rate;
    let movement_drain = movement_mag * wc_f32(WC_MOVEMENT_COST) * metabolic_rate;
    energy -= depletion_drain;
    energy -= movement_drain;
    // Accumulate total energy spent this tick
    physics_state[b + P_ENERGY_SPENT] += depletion_drain + movement_drain;

    // Biome damage (scaled by integrity_scale from brain config)
    let integrity_scale = bc_f32(CFG_INTEGRITY_SCALE);
    let biome_type = sample_biome(pos.x, pos.z);
    let in_danger = (biome_type == BIOME_DANGER);
    if in_danger {
        // Path-length hazard dose (plan 0009, Layer B): integrity loss is proportional
        // to the distance traveled through danger this tick, not to the number of ticks
        // spent in it. reference_step is a default-speed agent's per-tick displacement
        // (default_speed * dt = 20.0 * WC_DT), so a default-speed agent (step_len ≈
        // reference_step) takes byte-identical per-tick damage to the old per-tick model,
        // a 2× agent pays 2× per tick over half the ticks (the same dose per crossing),
        // and a stationary agent (step_len = 0) takes zero dose. NO floor on step_len:
        // dose is strictly proportional to path length.
        let reference_step = 20.0 * wc_f32(WC_DT);
        physics_state[b + P_INTEGRITY] = physics_state[b + P_INTEGRITY]
            - wc_f32(WC_HAZARD_DAMAGE) * integrity_scale * (step_len / max(reference_step, EPSILON));
        physics_state[b + P_IN_DANGER_BIOME] = 1.0;
        physics_state[b + P_DANGER_PATH_LENGTH] += step_len;
    } else {
        physics_state[b + P_IN_DANGER_BIOME] = 0.0;
    }

    // Integrity regen when energy > 50%
    let max_e = physics_state[b + P_MAX_ENERGY];
    var integrity = physics_state[b + P_INTEGRITY];
    let max_i = physics_state[b + P_MAX_INTEGRITY];
    if energy / max_e > 0.5 && integrity < max_i {
        integrity = min(integrity + wc_f32(WC_INTEGRITY_REGEN) * integrity_scale, max_i);
    }

    // Metabolic brain drain
    let mem_cap = physics_state[b + P_MEMORY_CAP];
    let proc_slots = physics_state[b + P_PROCESSING_SLOTS];
    energy -= (METABOLIC_BASE_COST + mem_cap * METABOLIC_MEMORY_COST + proc_slots * METABOLIC_PROCESSING_COST) * metabolic_rate;

    // Clamp and death check
    energy = max(energy, 0.0);
    integrity = max(integrity, 0.0);
    physics_state[b + P_ENERGY] = energy;
    physics_state[b + P_INTEGRITY] = integrity;

    if energy <= 0.0 || integrity <= 0.0 {
        physics_state[b + P_ALIVE] = 0.0;
        physics_state[b + P_DIED_FLAG] = 1.0;
        // Record the exact tick of death for CPU-side longest_life accounting.
        // Stored as f32 (exact for integer ticks up to 2^24 — matches P_TICKS_ALIVE).
        physics_state[b + P_LAST_DEATH_TICK] = f32(tick);
    } else {
        // Increment ticks alive (stored as f32, safe for integers up to 2^24)
        physics_state[b + P_TICKS_ALIVE] = physics_state[b + P_TICKS_ALIVE] + 1.0;
    }

    // Danger detection: scan biome grid for nearest danger cell
    let biome_inv = wc_f32(WC_BIOME_INV_CELL);
    let biome_half = wc_f32(WC_TERRAIN_HALF);

    let sense_radius = DANGER_SENSE_RADIUS;
    let cell_size = 1.0 / biome_inv;
    let max_cell_delta = u32(ceil(sense_radius / cell_size)) + 1u;

    var best_distance = sense_radius;
    var best_bearing = 0.0;
    var found_danger = false;

    // Scan square region of cells around agent
    let agent_col_i = i32((pos.x + biome_half) * biome_inv);
    let agent_row_i = i32((pos.z + biome_half) * biome_inv);

    for (var dr = -i32(max_cell_delta); dr <= i32(max_cell_delta); dr++) {
        for (var dc = -i32(max_cell_delta); dc <= i32(max_cell_delta); dc++) {
            let row = u32(clamp(agent_row_i + dr, 0, 255));
            let col = u32(clamp(agent_col_i + dc, 0, 255));

            if (sample_biome(f32(col) / biome_inv - biome_half + 0.5 / biome_inv,
                             f32(row) / biome_inv - biome_half + 0.5 / biome_inv) == BIOME_DANGER) {
                // Compute world position of cell center
                let cell_x = (f32(col) + 0.5) / biome_inv - biome_half;
                let cell_z = (f32(row) + 0.5) / biome_inv - biome_half;
                let to_danger = vec3f(cell_x - pos.x, 0.0, cell_z - pos.z);
                let dist = length(to_danger);

                if (dist < best_distance && dist < sense_radius) {
                    best_distance = dist;
                    found_danger = true;

                    // Compute signed bearing from facing direction
                    let facing_x = physics_state[b + P_FACING_X];
                    let facing_z = physics_state[b + P_FACING_Z];
                    let cross_y = facing_x * to_danger.z - facing_z * to_danger.x;
                    let dot_val = facing_x * to_danger.x + facing_z * to_danger.z;
                    best_bearing = atan2(cross_y, dot_val);
                }
            }
        }
    }

    if (found_danger) {
        physics_state[b + P_NEAREST_DANGER_DISTANCE] = best_distance;
        physics_state[b + P_NEAREST_DANGER_BEARING] = best_bearing;
    } else {
        physics_state[b + P_NEAREST_DANGER_DISTANCE] = DANGER_SENSE_RADIUS;
        physics_state[b + P_NEAREST_DANGER_BEARING] = 0.0;
    }

    // Avoidance intent: accumulate fraction of ticks where danger was in sense range
    // and motor turn opposed the danger bearing (deliberate turn-away).
    let danger_distance = physics_state[b + P_NEAREST_DANGER_DISTANCE];
    let danger_bearing = physics_state[b + P_NEAREST_DANGER_BEARING];
    if danger_distance < DANGER_SENSE_RADIUS {
        // Danger is in sense range; count this tick
        physics_state[b + P_AVOIDANCE_SENSE_RANGE_TICKS] += 1.0;

        // Check if motor turn opposes the danger bearing (turn away = negative product)
        // motor_turn is in [-1, 1], positive = turn right, negative = turn left
        // danger_bearing is signed: positive = danger to the right, negative = danger to the left
        // Turn-away means motor_turn and danger_bearing have opposite signs
        let turn_opposes_bearing = (motor_turn * danger_bearing) < 0.0;
        if turn_opposes_bearing {
            physics_state[b + P_AVOIDANCE_TURNS_OPPOSING] += 1.0;
        }
    }
}
