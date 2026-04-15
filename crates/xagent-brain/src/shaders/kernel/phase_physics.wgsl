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
    if (pos.x != pre_clamp_x) {
        physics_state[b + P_VEL_X] *= -1.0;
        yaw = -yaw;
        physics_state[b + P_YAW] = yaw;
        physics_state[b + P_FACING_X] = sin(yaw);
        physics_state[b + P_FACING_Z] = cos(yaw);
    }
    if (pos.z != pre_clamp_z) {
        physics_state[b + P_VEL_Z] *= -1.0;
        yaw = 3.14159265 - yaw;
        physics_state[b + P_YAW] = yaw;
        physics_state[b + P_FACING_X] = sin(yaw);
        physics_state[b + P_FACING_Z] = cos(yaw);
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

    // Energy depletion (scaled by metabolic_rate from brain config)
    let metabolic_rate = bc_f32(CFG_METABOLIC_RATE);
    // Normalize by default speed (20.0) so baseline energy drain is unchanged;
    // faster agents pay proportionally more, slower agents pay less.
    let movement_mag = min(abs(motor_forward) + abs(motor_strafe), 1.414) * (move_speed / 20.0);
    var energy = physics_state[b + P_ENERGY];
    energy -= wc_f32(WC_ENERGY_DEPLETION) * metabolic_rate;
    energy -= movement_mag * wc_f32(WC_MOVEMENT_COST) * metabolic_rate;

    // Biome damage (scaled by integrity_scale from brain config)
    let integrity_scale = bc_f32(CFG_INTEGRITY_SCALE);
    let biome_type = sample_biome(pos.x, pos.z);
    if biome_type == BIOME_DANGER {
        physics_state[b + P_INTEGRITY] = physics_state[b + P_INTEGRITY] - wc_f32(WC_HAZARD_DAMAGE) * integrity_scale;
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
    } else {
        // Increment ticks alive (stored as f32, safe for integers up to 2^24)
        physics_state[b + P_TICKS_ALIVE] = physics_state[b + P_TICKS_ALIVE] + 1.0;
    }
}
