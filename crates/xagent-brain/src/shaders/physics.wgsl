// Physics: movement, gravity, terrain collision, energy, biome damage, death.
// 1 thread per agent. Reads motor commands from decision_buf (previous brain tick output).

@group(0) @binding(0) var<storage, read_write> agent_phys: array<f32>;
@group(0) @binding(1) var<storage, read> decision: array<f32>;
@group(0) @binding(2) var<storage, read> heightmap: array<f32>;
@group(0) @binding(3) var<storage, read> biome: array<u32>;
@group(0) @binding(4) var<uniform> wconfig: array<vec4<f32>, 6>;

fn wc(idx: u32) -> f32 {
    return wconfig[idx / 4u][idx % 4u];
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent = gid.x;
    if agent >= u32(wc(13u)) { return; } // agent_count
    let b = agent * PHYS_STRIDE;

    let alive = agent_phys[b + P_ALIVE];
    if alive < 0.5 { return; }

    let dt = wc(1u); // dt
    let world_half = wc(16u); // world_half_bound

    // Snapshot prev energy/integrity
    agent_phys[b + P_PREV_ENERGY] = agent_phys[b + P_ENERGY];
    agent_phys[b + P_PREV_INTEGRITY] = agent_phys[b + P_INTEGRITY];

    // Save last-good position/velocity for NaN recovery
    let last_pos = vec3<f32>(agent_phys[b + P_POS_X], agent_phys[b + P_POS_Y], agent_phys[b + P_POS_Z]);
    let last_vel = vec3<f32>(agent_phys[b + P_VEL_X], agent_phys[b + P_VEL_Y], agent_phys[b + P_VEL_Z]);

    // Read motor commands from decision_buf
    // Motor layout in decision_buf: offset DIM+DIM within DECISION_STRIDE per agent
    let dec_base = agent * DECISION_STRIDE;
    let motor_offset = dec_base + DIM + DIM;
    var motor_fwd = decision[motor_offset];
    var motor_turn = decision[motor_offset + 1u];
    var motor_strafe = decision[motor_offset + 2u];

    // Sanitize motor: clamp [-1,1], NaN -> 0
    if !is_finite(motor_fwd) { motor_fwd = 0.0; }
    if !is_finite(motor_turn) { motor_turn = 0.0; }
    if !is_finite(motor_strafe) { motor_strafe = 0.0; }
    motor_fwd = clamp(motor_fwd, -1.0, 1.0);
    motor_turn = clamp(motor_turn, -1.0, 1.0);
    motor_strafe = clamp(motor_strafe, -1.0, 1.0);

    // Turning
    var yaw = agent_phys[b + P_YAW];
    let prev_yaw = yaw;
    yaw += motor_turn * TURN_SPEED * dt;
    agent_phys[b + P_YAW] = yaw;
    agent_phys[b + P_ANGULAR_VEL] = (yaw - prev_yaw) / max(dt, 1e-6);
    let facing = normalize(vec3<f32>(sin(yaw), 0.0, cos(yaw)));
    agent_phys[b + P_FACING_X] = facing.x;
    agent_phys[b + P_FACING_Y] = 0.0;
    agent_phys[b + P_FACING_Z] = facing.z;

    // Locomotion
    let right = vec3<f32>(facing.z, 0.0, -facing.x);
    var desired = facing * motor_fwd + right * motor_strafe;
    let desired_sq = dot(desired, desired);
    if desired_sq > 1.0 {
        desired = desired / sqrt(desired_sq);
    }
    agent_phys[b + P_VEL_X] = desired.x * MOVE_SPEED;
    agent_phys[b + P_VEL_Z] = desired.z * MOVE_SPEED;

    // Gravity
    agent_phys[b + P_VEL_Y] = agent_phys[b + P_VEL_Y] - GRAVITY * dt;

    // Integrate position
    var pos = vec3<f32>(
        agent_phys[b + P_POS_X] + agent_phys[b + P_VEL_X] * dt,
        agent_phys[b + P_POS_Y] + agent_phys[b + P_VEL_Y] * dt,
        agent_phys[b + P_POS_Z] + agent_phys[b + P_VEL_Z] * dt,
    );

    // Clamp to world bounds
    pos.x = clamp(pos.x, -world_half, world_half);
    pos.z = clamp(pos.z, -world_half, world_half);

    // Ground collision (inlined height_at to avoid storage pointer passing restriction)
    let gx = clamp((pos.x + TERRAIN_HALF) * TERRAIN_INV_STEP, 0.0, TERRAIN_MAX_COORD);
    let gz = clamp((pos.z + TERRAIN_HALF) * TERRAIN_INV_STEP, 0.0, TERRAIN_MAX_COORD);
    let hix = min(u32(gx), TERRAIN_MAX_IDX);
    let hiz = min(u32(gz), TERRAIN_MAX_IDX);
    let hfx = gx - f32(hix);
    let hfz = gz - f32(hiz);
    let h00 = heightmap[hiz * TERRAIN_VPS + hix];
    let h10 = heightmap[hiz * TERRAIN_VPS + hix + 1u];
    let h01 = heightmap[(hiz + 1u) * TERRAIN_VPS + hix];
    let h11 = heightmap[(hiz + 1u) * TERRAIN_VPS + hix + 1u];
    let ground = mix(mix(h00, h10, hfx), mix(h01, h11, hfx), hfz);
    if pos.y < ground + AGENT_HALF_HEIGHT {
        pos.y = ground + AGENT_HALF_HEIGHT;
        agent_phys[b + P_VEL_Y] = 0.0;
    }

    // NaN recovery
    if !is_finite(pos.x) || !is_finite(pos.y) || !is_finite(pos.z) {
        pos = last_pos;
        agent_phys[b + P_VEL_X] = last_vel.x;
        agent_phys[b + P_VEL_Y] = last_vel.y;
        agent_phys[b + P_VEL_Z] = last_vel.z;
    }

    agent_phys[b + P_POS_X] = pos.x;
    agent_phys[b + P_POS_Y] = pos.y;
    agent_phys[b + P_POS_Z] = pos.z;

    // Energy depletion
    let movement_mag = min(abs(motor_fwd) + abs(motor_strafe), 1.414);
    var energy = agent_phys[b + P_ENERGY];
    energy -= wc(2u); // energy_depletion_rate
    energy -= movement_mag * wc(3u); // movement_energy_cost

    // Biome damage
    let biome_half = wc(10u); // terrain_half
    let biome_inv = wc(11u); // biome_inv_cell
    let bcol = min(u32((pos.x + biome_half) * biome_inv), 255u);
    let brow = min(u32((pos.z + biome_half) * biome_inv), 255u);
    let biome_type = biome[brow * 256u + bcol];
    if biome_type == 2u { // Danger
        agent_phys[b + P_INTEGRITY] = agent_phys[b + P_INTEGRITY] - wc(4u); // hazard_damage
    }

    // Integrity regen when energy > 50%
    let max_e = agent_phys[b + P_MAX_ENERGY];
    var integrity = agent_phys[b + P_INTEGRITY];
    let max_i = agent_phys[b + P_MAX_INTEGRITY];
    if energy / max_e > 0.5 && integrity < max_i {
        integrity = min(integrity + wc(5u), max_i); // integrity_regen_rate
    }

    // Metabolic brain drain
    let mem_cap = agent_phys[b + P_MEMORY_CAP];
    let proc_slots = agent_phys[b + P_PROCESSING_SLOTS];
    energy -= METABOLIC_BASE_COST + mem_cap * METABOLIC_MEMORY_COST + proc_slots * METABOLIC_PROCESSING_COST;

    // Clamp and death check
    energy = max(energy, 0.0);
    integrity = max(integrity, 0.0);
    agent_phys[b + P_ENERGY] = energy;
    agent_phys[b + P_INTEGRITY] = integrity;

    if energy <= 0.0 || integrity <= 0.0 {
        agent_phys[b + P_ALIVE] = 0.0;
        agent_phys[b + P_DIED_FLAG] = 1.0;
    } else {
        // Increment ticks alive (stored as f32, safe for integers up to 2^24)
        agent_phys[b + P_TICKS_ALIVE] = agent_phys[b + P_TICKS_ALIVE] + 1.0;
    }
}

fn is_finite(v: f32) -> bool {
    return v == v && abs(v) < 3.4e38;
}
