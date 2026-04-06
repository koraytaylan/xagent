// ── Fused mega-kernel: per-agent physics + food + death + brain ─────────────
// dispatch(agent_count, 1, 1) — one workgroup per agent, 256 threads each.
// Loops over vision_stride brain cycles internally.
// Requires: common.wgsl, brain_tick.wgsl functions (concatenated by Rust).

// ── Additional shared memory for food detection ───────────────────────────
var<workgroup> s_best_food_idx: u32;
var<workgroup> s_best_food_dist: f32;

// ── Eat radius constant (sqrt of FOOD_CONSUME_RADIUS_SQ) ─────────────────
const EAT_RADIUS: f32 = 2.5;  // sqrt(6.25)

// ══════════════════════════════════════════════════════════════════════════
// Per-agent physics (extracted from phase_physics.wgsl, single-agent)
// ══════════════════════════════════════════════════════════════════════════

fn agent_physics(agent_id: u32, tick: u32) {
    let b = agent_id * PHYS_STRIDE;

    let alive = agent_phys[b + P_ALIVE];
    if alive < 0.5 { return; }

    let dt = wc_f32(WC_DT);
    let world_half = wc_f32(WC_WORLD_HALF_BOUND);

    // Snapshot prev energy/integrity
    agent_phys[b + P_PREV_ENERGY] = agent_phys[b + P_ENERGY];
    agent_phys[b + P_PREV_INTEGRITY] = agent_phys[b + P_INTEGRITY];

    // Save last-good position/velocity for NaN recovery
    let last_pos = vec3<f32>(agent_phys[b + P_POS_X], agent_phys[b + P_POS_Y], agent_phys[b + P_POS_Z]);
    let last_vel = vec3<f32>(agent_phys[b + P_VEL_X], agent_phys[b + P_VEL_Y], agent_phys[b + P_VEL_Z]);

    // Read motor commands from decision_buf
    let dec_base = agent_id * DECISION_STRIDE;
    let motor_offset = dec_base + DIM + DIM;
    var motor_fwd = decision_buf[motor_offset];
    var motor_turn = decision_buf[motor_offset + 1u];
    var motor_strafe = decision_buf[motor_offset + 2u];

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

    // Ground collision
    let ground = sample_height(pos.x, pos.z);
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
    let metabolic_rate = bc_f32(CFG_METABOLIC_RATE);
    let movement_mag = min(abs(motor_fwd) + abs(motor_strafe), 1.414);
    var energy = agent_phys[b + P_ENERGY];
    energy -= wc_f32(WC_ENERGY_DEPLETION) * metabolic_rate;
    energy -= movement_mag * wc_f32(WC_MOVEMENT_COST) * metabolic_rate;

    // Biome damage
    let integrity_scale = bc_f32(CFG_INTEGRITY_SCALE);
    let biome_type = sample_biome(pos.x, pos.z);
    if biome_type == BIOME_DANGER {
        agent_phys[b + P_INTEGRITY] = agent_phys[b + P_INTEGRITY] - wc_f32(WC_HAZARD_DAMAGE) * integrity_scale;
    }

    // Integrity regen when energy > 50%
    let max_e = agent_phys[b + P_MAX_ENERGY];
    var integrity = agent_phys[b + P_INTEGRITY];
    let max_i = agent_phys[b + P_MAX_INTEGRITY];
    if energy / max_e > 0.5 && integrity < max_i {
        integrity = min(integrity + wc_f32(WC_INTEGRITY_REGEN) * integrity_scale, max_i);
    }

    // Metabolic brain drain
    let mem_cap = agent_phys[b + P_MEMORY_CAP];
    let proc_slots = agent_phys[b + P_PROCESSING_SLOTS];
    energy -= (METABOLIC_BASE_COST + mem_cap * METABOLIC_MEMORY_COST + proc_slots * METABOLIC_PROCESSING_COST) * metabolic_rate;

    // Clamp and death check
    energy = max(energy, 0.0);
    integrity = max(integrity, 0.0);
    agent_phys[b + P_ENERGY] = energy;
    agent_phys[b + P_INTEGRITY] = integrity;

    if energy <= 0.0 || integrity <= 0.0 {
        agent_phys[b + P_ALIVE] = 0.0;
        agent_phys[b + P_DIED_FLAG] = 1.0;
    } else {
        agent_phys[b + P_TICKS_ALIVE] = agent_phys[b + P_TICKS_ALIVE] + 1.0;
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Brute-force food detection (replaces grid-based phase_food_detect)
// ══════════════════════════════════════════════════════════════════════════

fn agent_food_detect(agent_id: u32, tid: u32) {
    let b = agent_id * PHYS_STRIDE;

    // Skip dead agents — all threads must agree (uniform control flow)
    if agent_phys[b + P_ALIVE] < 0.5 { return; }

    let pos = vec3f(
        agent_phys[b + P_POS_X],
        agent_phys[b + P_POS_Y],
        agent_phys[b + P_POS_Z]);
    let food_count = wc_u32(WC_FOOD_COUNT);

    // Each thread scans a slice of food_state
    var local_best_idx = 0xFFFFFFFFu;
    var local_best_dist = 1e12;
    for (var f = tid; f < food_count; f += 256u) {
        if (atomicLoad(&food_flags[f]) != 0u) { continue; } // already consumed
        let fbase = f * FOOD_STATE_STRIDE;
        let fp = vec3f(
            food_state[fbase + F_POS_X],
            food_state[fbase + F_POS_Y],
            food_state[fbase + F_POS_Z]);
        let d = distance(pos, fp);
        if (d < EAT_RADIUS && d < local_best_dist) {
            local_best_dist = d;
            local_best_idx = f;
        }
    }

    // Two-phase shared-memory reduction (s_similarities/s_sort_idx are 128 elements)
    // Phase 1: first 128 threads write directly
    if (tid < 128u) {
        s_similarities[tid] = local_best_dist;
        s_sort_idx[tid] = local_best_idx;
    }
    workgroupBarrier();

    // Phase 2: second 128 threads merge into first 128 slots
    if (tid >= 128u) {
        let slot = tid - 128u;
        if (local_best_dist < s_similarities[slot]) {
            s_similarities[slot] = local_best_dist;
            s_sort_idx[slot] = local_best_idx;
        }
    }
    workgroupBarrier();

    if (tid == 0u) {
        var best_idx = 0xFFFFFFFFu;
        var best_dist = 1e12;
        for (var i = 0u; i < 128u; i++) {
            if (s_similarities[i] < best_dist) {
                best_dist = s_similarities[i];
                best_idx = s_sort_idx[i];
            }
        }
        if (best_idx != 0xFFFFFFFFu) {
            // Atomic: claim food (prevents double-eating across workgroups)
            let result = atomicCompareExchangeWeak(&food_flags[best_idx], 0u, 1u);
            if (result.exchanged) {
                let food_energy = wc_f32(WC_FOOD_ENERGY);
                agent_phys[b + P_ENERGY] += food_energy;
                agent_phys[b + P_FOOD_COUNT] += 1.0;
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Death/respawn (extracted from phase_death.wgsl, single-agent)
// ══════════════════════════════════════════════════════════════════════════

fn agent_death_respawn(agent_id: u32, tick: u32) {
    let base = agent_id * PHYS_STRIDE;
    if (agent_phys[base + P_DIED_FLAG] < 0.5) { return; }

    // 1. Pick a safe spawn position
    let world_half = wc_f32(WC_WORLD_HALF_BOUND);
    var spawn_x = 0.0;
    var spawn_z = 0.0;
    var found = false;
    for (var attempt = 0u; attempt < 50u; attempt++) {
        let h = pcg_hash(tick * 256u + agent_id + attempt);
        let h2 = pcg_hash(h);
        let rx = hash_to_float(h) * 2.0 - 1.0;
        let rz = hash_to_float(h2) * 2.0 - 1.0;
        let cx = rx * world_half;
        let cz = rz * world_half;
        if (sample_biome(cx, cz) != BIOME_DANGER) {
            spawn_x = cx;
            spawn_z = cz;
            found = true;
            break;
        }
    }
    if (!found) {
        let h = pcg_hash(tick * 256u + agent_id);
        let h2 = pcg_hash(h);
        spawn_x = (hash_to_float(h) * 2.0 - 1.0) * world_half;
        spawn_z = (hash_to_float(h2) * 2.0 - 1.0) * world_half;
    }
    let spawn_y = sample_height(spawn_x, spawn_z) + AGENT_HALF_HEIGHT;

    // 2. Preserve fitness fields
    let saved_food_count   = agent_phys[base + P_FOOD_COUNT];
    let saved_ticks_alive  = agent_phys[base + P_TICKS_ALIVE];
    let saved_death_count  = agent_phys[base + P_DEATH_COUNT] + 1.0;
    let max_energy         = agent_phys[base + P_MAX_ENERGY];
    let max_integrity      = agent_phys[base + P_MAX_INTEGRITY];
    let memory_cap         = agent_phys[base + P_MEMORY_CAP];
    let processing_slots   = agent_phys[base + P_PROCESSING_SLOTS];

    // 3. Reset physics state
    for (var i = 0u; i < PHYS_STRIDE; i++) {
        agent_phys[base + i] = 0.0;
    }
    agent_phys[base + P_POS_X]           = spawn_x;
    agent_phys[base + P_POS_Y]           = spawn_y;
    agent_phys[base + P_POS_Z]           = spawn_z;
    agent_phys[base + P_FACING_Z]        = 1.0;
    agent_phys[base + P_ENERGY]          = max_energy;
    agent_phys[base + P_MAX_ENERGY]      = max_energy;
    agent_phys[base + P_INTEGRITY]       = max_integrity;
    agent_phys[base + P_MAX_INTEGRITY]   = max_integrity;
    agent_phys[base + P_PREV_ENERGY]     = max_energy;
    agent_phys[base + P_PREV_INTEGRITY]  = max_integrity;
    agent_phys[base + P_ALIVE]           = 1.0;
    agent_phys[base + P_MEMORY_CAP]      = memory_cap;
    agent_phys[base + P_PROCESSING_SLOTS] = processing_slots;
    agent_phys[base + P_FOOD_COUNT]      = saved_food_count;
    agent_phys[base + P_TICKS_ALIVE]     = saved_ticks_alive;
    agent_phys[base + P_DEATH_COUNT]     = saved_death_count;

    // 4. Reset brain state
    let brain_base = agent_id * BRAIN_STRIDE;

    let pat_base = agent_id * PATTERN_STRIDE;
    for (var i = 0u; i < MEMORY_CAP; i++) {
        pattern_buf[pat_base + O_PAT_REINF + i] *= 0.5;
    }

    for (var i = 0u; i < 6u; i++) {
        brain_state[brain_base + O_HOMEO + i] = 0.0;
    }

    brain_state[brain_base + O_EXPLORATION_RATE] = 0.5;

    for (var i = 0u; i < ACTION_HISTORY_LEN; i++) {
        brain_state[brain_base + O_FATIGUE_FWD_RING + i] = 0.0;
        brain_state[brain_base + O_FATIGUE_TURN_RING + i] = 0.0;
    }
    brain_state[brain_base + O_FATIGUE_CURSOR] = 0.0;
    brain_state[brain_base + O_FATIGUE_LEN] = 0.0;
    brain_state[brain_base + O_FATIGUE_FACTOR] = 1.0;

    for (var i = 0u; i < DIM; i++) {
        brain_state[brain_base + O_HAB_EMA + i] = 0.0;
        brain_state[brain_base + O_HAB_ATTEN + i] = 1.0;
        brain_state[brain_base + O_PREV_ENCODED + i] = 0.0;
    }

    let hist_base = agent_id * HISTORY_STRIDE;
    for (var i = 0u; i < HISTORY_STRIDE; i++) {
        history_buf[hist_base + i] = 0.0;
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Brain tick inner — delegates to the 7 cooperative passes from brain_tick.wgsl
// ══════════════════════════════════════════════════════════════════════════

fn brain_tick_inner(agent_id: u32, tid: u32 /* MEGA_SUBGROUP_TOPK_PARAMS */) {
    // Skip dead agents (uniform control flow — all threads agree)
    if (agent_phys[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }

    coop_feature_extract(agent_id, tid);
    workgroupBarrier();

    coop_encode(agent_id, tid);
    workgroupBarrier();

    coop_habituate_homeo(agent_id, tid);
    storageBarrier(); workgroupBarrier();

    coop_recall_score(agent_id, tid);
    workgroupBarrier();

    coop_recall_topk(agent_id, tid /* MEGA_SUBGROUP_TOPK_ARGS */);
    storageBarrier(); workgroupBarrier();

    coop_predict_and_act(agent_id, tid);
    storageBarrier(); workgroupBarrier();

    coop_learn_and_store(agent_id, tid);
}

// ══════════════════════════════════════════════════════════════════════════
// Entry point
// ══════════════════════════════════════════════════════════════════════════

@compute @workgroup_size(256)
fn mega_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
    // MEGA_SUBGROUP_ENTRY_PARAMS
) {
    let agent_id = wgid.x;
    let tid = lid.x;
    let vision_stride = wc_u32(WC_VISION_STRIDE);
    let stride = wc_u32(WC_BRAIN_TICK_STRIDE);
    let start_tick = wc_u32(WC_TICK);

    for (var cycle = 0u; cycle < vision_stride; cycle++) {
        let base_tick = start_tick + cycle * stride;

        // Per-agent physics: thread 0 loops over brain_tick_stride sub-ticks
        if (tid == 0u) {
            for (var t = 0u; t < stride; t++) {
                agent_physics(agent_id, base_tick + t);
            }
        }
        workgroupBarrier();

        // Brute-force food detection: all 256 threads cooperate
        agent_food_detect(agent_id, tid);
        workgroupBarrier();

        // Death/respawn: thread 0
        if (tid == 0u) {
            agent_death_respawn(agent_id, base_tick);
        }
        workgroupBarrier();

        // Brain: all 256 threads, 7 cooperative passes
        brain_tick_inner(agent_id, tid /* MEGA_SUBGROUP_TOPK_INNER_ARGS */);
        workgroupBarrier();
    }
}
