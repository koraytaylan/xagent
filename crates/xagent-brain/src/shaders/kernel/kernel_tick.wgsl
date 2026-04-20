// ── Fused kernel: per-agent physics + food + death + brain ─────────────
// dispatch(agent_count, 1, 1) — one workgroup per agent, 256 threads each.
// Loops over vision_stride brain cycles internally.
// Requires: common.wgsl, brain_tick.wgsl functions (concatenated by Rust).
//
// SAFETY INVARIANT — barrier uniformity:
// Multi-thread functions (`agent_food_detect`, `brain_tick_inner`) must reach
// every `workgroupBarrier()` / `storageBarrier()` from all 256 threads. To
// avoid relying on "all threads in this workgroup happen to evaluate the alive
// flag identically", they:
//   1. Read `P_ALIVE` once into a local `alive` variable.
//   2. Wrap per-agent work in `if (alive) { ... }` blocks rather than
//      early-returning.
//   3. Keep barriers outside those blocks so they execute regardless of
//      logical state.
// This is robust against mid-kernel death transitions and hypothetical memory
// corruption that could desynchronize the flag across threads.
//
// Single-thread helpers (`agent_physics`, `agent_death_respawn`) are invoked
// from inside `if (tid == 0u) { ... }` blocks in the entry point and therefore
// contain no barriers themselves; their internal early-returns only exit thread
// 0's call, and all threads still reach the outer barrier after the `if` block.

// EAT_RADIUS removed — read from wconfig via wc_f32(WC_FOOD_RADIUS)

// ══════════════════════════════════════════════════════════════════════════
// Per-agent physics (extracted from phase_physics.wgsl, single-agent)
// ══════════════════════════════════════════════════════════════════════════

fn agent_physics(agent_id: u32, tick: u32) {
    let b = agent_id * PHYS_STRIDE;

    // Called only from `if (tid == 0u) { ... }` in the entry point; contains
    // no barriers, so an early return here affects only thread 0's progression
    // through the caller and does not perturb workgroup barrier uniformity.
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
    let decision_base = agent_id * DECISION_STRIDE;
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
    let move_speed = brain_state[agent_id * BRAIN_STRIDE + O_MOVEMENT_SPEED];
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

    // Ground collision
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

    // Energy depletion
    let metabolic_rate = bc_f32(CFG_METABOLIC_RATE);
    // Normalize by default speed (20.0) so baseline energy drain is unchanged.
    let movement_mag = min(abs(motor_forward) + abs(motor_strafe), 1.414) * (move_speed / 20.0);
    var energy = physics_state[b + P_ENERGY];
    energy -= wc_f32(WC_ENERGY_DEPLETION) * metabolic_rate;
    energy -= movement_mag * wc_f32(WC_MOVEMENT_COST) * metabolic_rate;

    // Biome damage
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
        physics_state[b + P_TICKS_ALIVE] = physics_state[b + P_TICKS_ALIVE] + 1.0;
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Brute-force food detection (replaces grid-based phase_food_detect)
// ══════════════════════════════════════════════════════════════════════════

fn agent_food_detect(agent_id: u32, tid: u32) {
    let b = agent_id * PHYS_STRIDE;

    // Read the alive flag once into a uniform local. All subsequent barriers
    // must execute regardless of logical state — see top-of-file SAFETY
    // INVARIANT. The scan and the thread-0 eat step are gated on `alive`; the
    // reduction barriers below are not.
    let alive = physics_state[b + P_ALIVE] >= 0.5;

    // Default "no candidate" sentinel values so the reduction runs safely even
    // when the agent is dead (or a hypothetical divergence prevented the scan).
    var local_best_idx = 0xFFFFFFFFu;
    var local_best_dist_sq = 1e12;

    if (alive) {
        let pos = vec3f(
            physics_state[b + P_POS_X],
            physics_state[b + P_POS_Y],
            physics_state[b + P_POS_Z]);
        let food_count = wc_u32(WC_FOOD_COUNT);
        let eat_radius = wc_f32(WC_FOOD_RADIUS);
        let eat_radius_sq = eat_radius * eat_radius;

        // Each thread scans a slice of food_state
        for (var f = tid; f < food_count; f += 256u) {
            if (atomicLoad(&food_flags[f]) != 0u) { continue; } // already consumed
            let fbase = f * FOOD_STATE_STRIDE;
            let dx = pos.x - food_state[fbase + F_POS_X];
            let dz = pos.z - food_state[fbase + F_POS_Z];
            let d_sq = dx * dx + dz * dz;
            if (d_sq < eat_radius_sq && d_sq < local_best_dist_sq) {
                local_best_dist_sq = d_sq;
                local_best_idx = f;
            }
        }
    }

    // Two-phase shared-memory reduction (s_similarities/shared_sort_indices are 128 elements).
    // Runs unconditionally so both barriers are reached by every thread.
    // Phase 1: first 128 threads write directly
    if (tid < 128u) {
        s_similarities[tid] = local_best_dist_sq;
        shared_sort_indices[tid] = local_best_idx;
    }
    workgroupBarrier();

    // Phase 2: second 128 threads merge into first 128 slots
    if (tid >= 128u) {
        let slot = tid - 128u;
        if (local_best_dist_sq < s_similarities[slot]) {
            s_similarities[slot] = local_best_dist_sq;
            shared_sort_indices[slot] = local_best_idx;
        }
    }
    workgroupBarrier();

    if (tid == 0u && alive) {
        var best_idx = 0xFFFFFFFFu;
        var best_dist_sq = 1e12;
        for (var i = 0u; i < 128u; i++) {
            if (s_similarities[i] < best_dist_sq) {
                best_dist_sq = s_similarities[i];
                best_idx = shared_sort_indices[i];
            }
        }
        if (best_idx != 0xFFFFFFFFu) {
            // Atomic: claim food (prevents double-eating across workgroups)
            let result = atomicCompareExchangeWeak(&food_flags[best_idx], 0u, 1u);
            if (result.exchanged) {
                let food_energy = wc_f32(WC_FOOD_ENERGY);
                physics_state[b + P_ENERGY] += food_energy;
                physics_state[b + P_FOOD_COUNT] += 1.0;
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Death/respawn (extracted from phase_death.wgsl, single-agent)
// ══════════════════════════════════════════════════════════════════════════

fn agent_death_respawn(agent_id: u32, tick: u32) {
    let base = agent_id * PHYS_STRIDE;
    // Called only from `if (tid == 0u) { ... }` in the entry point; contains
    // no barriers, so an early return here affects only thread 0's progression
    // through the caller and does not perturb workgroup barrier uniformity.
    if (physics_state[base + P_DIED_FLAG] < 0.5) { return; }

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
    let saved_food_count   = physics_state[base + P_FOOD_COUNT];
    let saved_ticks_alive  = physics_state[base + P_TICKS_ALIVE];
    let saved_death_count  = physics_state[base + P_DEATH_COUNT] + 1.0;
    let max_energy         = physics_state[base + P_MAX_ENERGY];
    let max_integrity      = physics_state[base + P_MAX_INTEGRITY];
    let memory_cap         = physics_state[base + P_MEMORY_CAP];
    let processing_slots   = physics_state[base + P_PROCESSING_SLOTS];

    // 3. Reset physics state
    for (var i = 0u; i < PHYS_STRIDE; i++) {
        physics_state[base + i] = 0.0;
    }
    physics_state[base + P_POS_X]           = spawn_x;
    physics_state[base + P_POS_Y]           = spawn_y;
    physics_state[base + P_POS_Z]           = spawn_z;
    physics_state[base + P_FACING_Z]        = 1.0;
    physics_state[base + P_ENERGY]          = max_energy;
    physics_state[base + P_MAX_ENERGY]      = max_energy;
    physics_state[base + P_INTEGRITY]       = max_integrity;
    physics_state[base + P_MAX_INTEGRITY]   = max_integrity;
    physics_state[base + P_PREV_ENERGY]     = max_energy;
    physics_state[base + P_PREV_INTEGRITY]  = max_integrity;
    physics_state[base + P_ALIVE]           = 1.0;
    physics_state[base + P_MEMORY_CAP]      = memory_cap;
    physics_state[base + P_PROCESSING_SLOTS] = processing_slots;
    physics_state[base + P_FOOD_COUNT]      = saved_food_count;
    physics_state[base + P_TICKS_ALIVE]     = saved_ticks_alive;
    physics_state[base + P_DEATH_COUNT]     = saved_death_count;

    // 4. Reset brain state
    let brain_base = agent_id * BRAIN_STRIDE;

    let pattern_base = agent_id * PATTERN_STRIDE;
    for (var i = 0u; i < MEMORY_CAP; i++) {
        pattern_buffer[pattern_base + O_PAT_REINF + i] *= 0.5;
    }

    for (var i = 0u; i < 6u; i++) {
        brain_state[brain_base + O_HOMEO + i] = 0.0;
    }

    brain_state[brain_base + O_EXPLORATION_RATE] = 0.5;

    for (var i = 0u; i < POS_RING_LEN; i++) {
        brain_state[brain_base + O_POS_RING_X + i] = 0.0;
        brain_state[brain_base + O_POS_RING_Z + i] = 0.0;
    }
    brain_state[brain_base + O_POS_RING_CURSOR] = 0.0;
    brain_state[brain_base + O_POS_RING_LEN] = 0.0;
    brain_state[brain_base + O_ACCUM_FWD] = 0.0;
    brain_state[brain_base + O_FATIGUE_FACTOR] = 1.0;

    for (var i = 0u; i < ENCODED_DIMENSION; i++) {
        brain_state[brain_base + O_HAB_EMA + i] = 0.0;
        brain_state[brain_base + O_HAB_ATTEN + i] = 1.0;
        brain_state[brain_base + O_PREV_ENCODED + i] = 0.0;
    }

    let history_base = agent_id * HISTORY_STRIDE;
    for (var i = 0u; i < HISTORY_STRIDE; i++) {
        history_buffer[history_base + i] = 0.0;
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Brain tick inner — delegates to the 7 cooperative passes from brain_tick.wgsl
// ══════════════════════════════════════════════════════════════════════════

fn brain_tick_inner(agent_id: u32, tid: u32 /* KERNEL_SUBGROUP_TOPK_PARAMS */) {
    // Read the alive flag once into a uniform local, then wrap every
    // cooperative pass in `if (alive) { ... }`. Inter-pass barriers live
    // outside the guards so they are reached by all 256 threads regardless of
    // logical state — see top-of-file SAFETY INVARIANT. Passes with their own
    // internal barriers (`coop_recall_topk`, `coop_predict_and_act`,
    // `coop_learn_and_store`) stay consistent because the guard is uniform: all
    // threads either enter together (hitting every internal barrier) or skip
    // together (hitting none).
    let alive = physics_state[agent_id * PHYS_STRIDE + P_ALIVE] >= 0.5;

    if (alive) { coop_feature_extract(agent_id, tid); }
    workgroupBarrier();

    if (alive) { coop_encode(agent_id, tid); }
    workgroupBarrier();

    if (alive) { coop_habituate_homeo(agent_id, tid); }
    storageBarrier(); workgroupBarrier();

    if (alive) { coop_recall_score(agent_id, tid); }
    workgroupBarrier();

    if (alive) { coop_recall_topk(agent_id, tid /* KERNEL_SUBGROUP_TOPK_ARGS */); }
    storageBarrier(); workgroupBarrier();

    if (alive) { coop_predict_and_act(agent_id, tid); }
    storageBarrier(); workgroupBarrier();

    if (alive) { coop_learn_and_store(agent_id, tid); }
}

// ══════════════════════════════════════════════════════════════════════════
// Entry point
//
// Ordering guarantee within each inner cycle:
//   physics → food_detect → death_respawn → brain
//
// This guarantees same-dispatch ordering/visibility for data written in the
// earlier kernel phases, but it does NOT mean all brain inputs are from the
// same cycle.
//
// Vision data (sensory_buffer) is written by the external vision pass, which
// runs in the same GPU command encoder AFTER this kernel dispatch.  The
// brain therefore reads sensory_buffer written by the *previous* batch's vision
// pass — a one-batch lag that is consistent regardless of stride values.
// Non-visual proprioception (velocity, energy, etc.) follows the same lag
// because it is also packed into sensory_buffer by that vision pass.  In this
// function, the only physics state read directly from physics_state before
// feature extraction is the same-cycle P_ALIVE flag used as a uniform guard
// for the cooperative passes (see `brain_tick_inner`).
//
// When brain_tick_stride == vision_stride the batch covers exactly
// (vision_stride * brain_tick_stride) physics ticks and vision runs once
// at the end of the batch, ready for the next batch's kernel.
// ══════════════════════════════════════════════════════════════════════════

@compute @workgroup_size(256)
fn kernel_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
    // KERNEL_SUBGROUP_ENTRY_PARAMS
) {
    let agent_id = wgid.x;
    let tid = lid.x;
    let vision_stride = wc_u32(WC_VISION_STRIDE);
    let stride = wc_u32(WC_BRAIN_TICK_STRIDE);
    let start_tick = wc_u32(WC_TICK);

    for (var cycle = 0u; cycle < vision_stride; cycle++) {
        let base_tick = start_tick + cycle * stride;

        // Per-agent physics: thread 0 loops over brain_tick_stride sub-ticks.
        // Physics always precedes brain within the same cycle (barrier below).
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

        // Brain: all 256 threads, 7 cooperative passes.
        // Reads sensory_buffer (vision + proprioception) from the previous batch's
        // vision pass.  Physics state updated in this cycle is NOT yet in
        // sensory_buffer — that update happens in the vision pass at the end of
        // this batch, making it available for the following batch.
        brain_tick_inner(agent_id, tid /* KERNEL_SUBGROUP_TOPK_INNER_ARGS */);
        workgroupBarrier();
    }
}
