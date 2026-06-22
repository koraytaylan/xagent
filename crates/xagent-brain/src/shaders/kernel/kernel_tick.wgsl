// ── Fused kernel: per-agent physics + food + death + brain ─────────────
// dispatch(agent_count, 1, 1) — one workgroup per agent, 256 threads each.
// Loops over vision_stride brain cycles internally.
// Requires: common.wgsl, brain_tick.wgsl functions (concatenated by Rust).
//
// SAFETY INVARIANT — barrier uniformity:
// Multi-thread functions (`agent_food_detect`, `brain_tick_inner`) must reach
// every `workgroupBarrier()` / `storageBarrier()` from all 256 threads,
// including the internal barriers inside guarded cooperative passes
// (`coop_recall_topk`, `coop_predict_and_act`, `coop_learn_and_store`). To
// guarantee this, the alive flag is made workgroup-uniform by construction:
//   1. The sole writer to `P_ALIVE` (thread 0, via `agent_physics` /
//      `agent_death_respawn`) then broadcasts the post-write value into the
//      workgroup variable `s_alive` immediately before the next
//      `workgroupBarrier()`.
//   2. All threads read `s_alive` — never `physics_state[P_ALIVE]` directly —
//      after that barrier, so every thread observes the same value.
//   3. Per-agent work is wrapped in `if (alive) { ... }`; inter-pass barriers
//      live outside the guard so dead agents still execute them.
// `workgroupBarrier()` alone does not synchronize storage memory, so without
// this broadcast per-thread reads of `P_ALIVE` could disagree and deadlock the
// internal barriers inside guarded passes.
//
// Single-thread helpers (`agent_physics`, `agent_death_respawn`) are invoked
// from inside `if (tid == 0u) { ... }` blocks in the entry point and therefore
// contain no barriers themselves; their internal early-returns only exit thread
// 0's call, and all threads still reach the outer barrier after the `if` block.

// EAT_RADIUS removed — read from wconfig via wc_f32(WC_FOOD_RADIUS)

// Workgroup-uniform alive broadcast. Written by thread 0 immediately before a
// `workgroupBarrier()`, read by all threads after that barrier. Encoded as u32
// (1 = alive, 0 = dead) so no atomics are needed. See SAFETY INVARIANT above.
var<workgroup> s_alive: u32;

// Squared-distance reduction scratch for the nearest food within
// `FOOD_SENSE_RADIUS`. Reduced in parallel with the eat candidate in
// `agent_food_detect`, reusing the same two barriers. Sized to the reduction
// width (= MEMORY_CAP, half the 256-thread workgroup), matching `s_similarities`.
// Distance only; no food index is needed because the nearest in-range food is
// measured for sensory steering feedback.
var<workgroup> s_food_dist_sq: array<f32, MEMORY_CAP>;

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

    // Accumulate distance traveled (planar displacement this tick)
    let step_len = length(vec2<f32>(pos.x - last_pos.x, pos.z - last_pos.z));
    physics_state[b + P_DISTANCE_TRAVELED] += step_len;

    // Energy depletion
    let metabolic_rate = bc_f32(CFG_METABOLIC_RATE);
    // Super-linear drag (plan 0009, Layer A): above-baseline-only cost exponent.
    // Normalize by default speed (20.0); the exponent only activates when the agent
    // is above baseline speed (speed_ratio >= 1.0). Below baseline the drag is
    // exactly `speed_ratio` (the same as k=1.0), so sub-baseline drain is unchanged
    // between k=1.0 and k>1.0 — no torpor gradient, no new incentive to slow down.
    //
    //   k=1.0:                  drag = speed_ratio          (exact old expression, always)
    //   k>1.0, speed_ratio < 1: drag = speed_ratio          (unchanged from k=1.0)
    //   k>1.0, speed_ratio >= 1: drag = pow(speed_ratio, k) (super-linear above baseline)
    let speed_ratio = move_speed / DEFAULT_MOVE_SPEED;
    let speed_cost_exponent = wc_f32(WC_SPEED_COST_EXPONENT);
    let above_baseline = speed_ratio >= 1.0;
    // For k>1.0: apply pow only above baseline; below baseline keep speed_ratio.
    let super_linear_drag = select(speed_ratio, pow(speed_ratio, speed_cost_exponent), above_baseline);
    // For k=1.0: use speed_ratio exactly (bit-identical to pre-task expression).
    let drag = select(super_linear_drag, speed_ratio, speed_cost_exponent == 1.0);
    let movement_mag = min(abs(motor_forward) + abs(motor_strafe), SQRT_2) * drag;
    var energy = physics_state[b + P_ENERGY];
    let depletion_drain = wc_f32(WC_ENERGY_DEPLETION) * metabolic_rate;
    let movement_drain = movement_mag * wc_f32(WC_MOVEMENT_COST) * metabolic_rate;
    energy -= depletion_drain;
    energy -= movement_drain;
    // Accumulate total energy spent this tick
    physics_state[b + P_ENERGY_SPENT] += depletion_drain + movement_drain;

    // Biome damage
    let integrity_scale = bc_f32(CFG_INTEGRITY_SCALE);
    let biome_type = sample_biome(pos.x, pos.z);
    // P_IN_DANGER_BIOME is navigational telemetry consumed by danger_exit_probe
    // and behavior_metric danger-dwell, which read >0.5 as "in danger". Publish
    // 1.0 while the agent is in a danger biome (and taking hazard damage), 0.0
    // otherwise — the flag must match the agent's actual current biome.
    let in_danger = (biome_type == BIOME_DANGER);
    if in_danger {
        // Path-length hazard dose (plan 0009, Layer B): integrity loss is proportional
        // to the distance traveled through danger this tick, not to the number of ticks
        // spent in it. reference_step is a default-speed agent's per-tick displacement
        // (default_speed * dt = DEFAULT_MOVE_SPEED * WC_DT), so a default-speed agent (step_len ≈
        // reference_step) takes byte-identical per-tick damage to the old per-tick model,
        // a 2× agent pays 2× per tick over half the ticks (the same dose per crossing),
        // and a stationary agent (step_len = 0) takes zero dose. NO floor on step_len:
        // dose is strictly proportional to path length.
        let reference_step = DEFAULT_MOVE_SPEED * wc_f32(WC_DT);
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
    // Per-tick energy burned == per-tick energy delta. The brain metabolic drain is
    // part of the cost an effort-denominated foraging score must see; omitting it
    // makes the score scale with brain size instead of skill.
    let brain_drain = (METABOLIC_BASE_COST + mem_cap * METABOLIC_MEMORY_COST + proc_slots * METABOLIC_PROCESSING_COST) * metabolic_rate;
    energy -= brain_drain;
    physics_state[b + P_ENERGY_SPENT] += brain_drain;

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
        physics_state[b + P_TICKS_ALIVE] = physics_state[b + P_TICKS_ALIVE] + 1.0;
    }

}

// ══════════════════════════════════════════════════════════════════════════
// Danger detection: nearest danger cell via biome grid scan
// ══════════════════════════════════════════════════════════════════════════

fn agent_avoidance_accumulate(agent_id: u32, motor_turn: f32) {
    let b = agent_id * PHYS_STRIDE;
    let danger_distance = physics_state[b + P_NEAREST_DANGER_DISTANCE];
    let danger_bearing = physics_state[b + P_NEAREST_DANGER_BEARING];
    if danger_distance < DANGER_SENSE_RADIUS {
        // Danger is in sense range; count this tick
        physics_state[b + P_AVOIDANCE_SENSE_RANGE_TICKS] += 1.0;

        // danger_bearing is the signed facing-relative angle to the nearest danger:
        // NEGATIVE = danger to the right (positive motor_turn turns right), POSITIVE =
        // danger to the left. A genuine turn-AWAY rotates against the bearing, so the
        // product (motor_turn * danger_bearing) is POSITIVE for an avoidance turn.
        let turn_away = (motor_turn * danger_bearing) > 0.0;
        if turn_away {
            physics_state[b + P_AVOIDANCE_TURNS_OPPOSING] += 1.0;
        }
    }
}

fn agent_approach_accumulate(agent_id: u32, motor_turn: f32) {
    let b = agent_id * PHYS_STRIDE;
    let food_distance = physics_state[b + P_NEAREST_FOOD_DISTANCE];
    let food_bearing = physics_state[b + P_NEAREST_FOOD_BEARING];
    if food_distance < FOOD_SENSE_RADIUS {
        // Food is in sense range; count this tick
        physics_state[b + P_APPROACH_SENSE_RANGE_TICKS] += 1.0;

        // food_bearing is the signed facing-relative angle to the nearest food:
        // NEGATIVE = food to the right (positive motor_turn turns right), POSITIVE =
        // food to the left. A genuine turn-TOWARD rotates with the bearing (opposite sign to avoidance),
        // so the product (motor_turn * food_bearing) is NEGATIVE for an approach turn.
        let turn_toward = (motor_turn * food_bearing) < 0.0;
        if turn_toward {
            physics_state[b + P_APPROACH_TURNS_TOWARD] += 1.0;
        }
    }
}

fn agent_danger_detect(agent_id: u32) {
    let b = agent_id * PHYS_STRIDE;
    let alive = physics_state[b + P_ALIVE];
    if alive < 0.5 {
        physics_state[b + P_NEAREST_DANGER_DISTANCE] = DANGER_SENSE_RADIUS;
        physics_state[b + P_NEAREST_DANGER_BEARING] = 0.0;
        return;
    }

    let agent_pos = vec3f(
        physics_state[b + P_POS_X],
        physics_state[b + P_POS_Y],
        physics_state[b + P_POS_Z]);

    let biome_inv = wc_f32(WC_BIOME_INV_CELL);
    let biome_half = wc_f32(WC_TERRAIN_HALF);

    // Scan biome cells within DANGER_SENSE_RADIUS
    let sense_radius = DANGER_SENSE_RADIUS;
    let cell_size = 1.0 / biome_inv;
    let max_cell_delta = u32(ceil(sense_radius / cell_size)) + 1u;

    var best_distance = sense_radius;
    var best_bearing = 0.0;
    var found_danger = false;

    // Scan square region of cells around agent
    let agent_col_i = i32((agent_pos.x + biome_half) * biome_inv);
    let agent_row_i = i32((agent_pos.z + biome_half) * biome_inv);

    for (var dr = -i32(max_cell_delta); dr <= i32(max_cell_delta); dr++) {
        for (var dc = -i32(max_cell_delta); dc <= i32(max_cell_delta); dc++) {
            let row = u32(clamp(agent_row_i + dr, 0, i32(BIOME_GRID_MAX_INDEX)));
            let col = u32(clamp(agent_col_i + dc, 0, i32(BIOME_GRID_MAX_INDEX)));

            if (sample_biome(f32(col) / biome_inv - biome_half + 0.5 / biome_inv,
                             f32(row) / biome_inv - biome_half + 0.5 / biome_inv) == BIOME_DANGER) {
                // Compute world position of cell center
                let cell_x = (f32(col) + 0.5) / biome_inv - biome_half;
                let cell_z = (f32(row) + 0.5) / biome_inv - biome_half;
                let to_danger = vec3f(cell_x - agent_pos.x, 0.0, cell_z - agent_pos.z);
                let dist = length(to_danger);

                if (dist < best_distance && dist < sense_radius && dist > EPSILON) {
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
}

// ══════════════════════════════════════════════════════════════════════════
// Brute-force food detection (replaces grid-based phase_food_detect)
// ══════════════════════════════════════════════════════════════════════════

fn agent_food_detect(agent_id: u32, tid: u32) {
    let b = agent_id * PHYS_STRIDE;

    // Read the workgroup-uniform alive flag (broadcast by thread 0 before the
    // preceding workgroupBarrier()). Never read `physics_state[P_ALIVE]`
    // directly here — see top-of-file SAFETY INVARIANT. The scan and the
    // thread-0 eat step are gated on `alive`; the reduction barriers below are
    // not.
    let alive = s_alive != 0u;

    // Default "no candidate" sentinel values so the reduction runs safely even
    // when the agent is dead (or a hypothetical divergence prevented the scan).
    var local_best_idx = 0xFFFFFFFFu;
    var local_best_dist_sq = 1e12;
    // Nearest food within FOOD_SENSE_RADIUS, independent of the eat gate, for the
    // food-sense navigation feature. Same sentinel so the reduction runs safely when dead.
    var local_best_food_dist_sq = 1e12;

    if (alive) {
        let pos = vec3f(
            physics_state[b + P_POS_X],
            physics_state[b + P_POS_Y],
            physics_state[b + P_POS_Z]);
        let food_count = wc_u32(WC_FOOD_COUNT);
        let eat_radius = wc_f32(WC_FOOD_RADIUS);
        let eat_radius_sq = eat_radius * eat_radius;
        let food_sense_radius_sq = FOOD_SENSE_RADIUS * FOOD_SENSE_RADIUS;

        // Each thread scans a slice of food_state
        for (var f = tid; f < food_count; f += 256u) {
            if (atomicLoad(&food_flags[f]) != 0u) { continue; } // already consumed
            let fbase = f * FOOD_STATE_STRIDE;
            let dx = pos.x - food_state[fbase + FOOD_POSITION_X];
            let dz = pos.z - food_state[fbase + FOOD_POSITION_Z];
            let d_sq = dx * dx + dz * dz;
            if (d_sq < eat_radius_sq && d_sq < local_best_dist_sq) {
                local_best_dist_sq = d_sq;
                local_best_idx = f;
            }
            // Wider navigational reduction: nearest food in sense range, no
            // eat gate. FOOD_SENSE_RADIUS ≥ eat_radius, so this is a superset.
            if (d_sq < food_sense_radius_sq && d_sq < local_best_food_dist_sq) {
                local_best_food_dist_sq = d_sq;
            }
        }
    }

    // Two-phase shared-memory reduction (s_similarities/shared_sort_indices are 128 elements).
    // Runs unconditionally so both barriers are reached by every thread.
    // Phase 1: first 128 threads write directly
    if (tid < 128u) {
        s_similarities[tid] = local_best_dist_sq;
        shared_sort_indices[tid] = local_best_idx;
        s_food_dist_sq[tid] = local_best_food_dist_sq;
    }
    workgroupBarrier();

    // Phase 2: second 128 threads merge into first 128 slots
    if (tid >= 128u) {
        let slot = tid - 128u;
        if (local_best_dist_sq < s_similarities[slot]) {
            s_similarities[slot] = local_best_dist_sq;
            shared_sort_indices[slot] = local_best_idx;
        }
        s_food_dist_sq[slot] = min(s_food_dist_sq[slot], local_best_food_dist_sq);
    }
    workgroupBarrier();

    if (tid == 0u && alive) {
        var best_idx = 0xFFFFFFFFu;
        var best_dist_sq = 1e12;
        var best_food_dist_sq = 1e12;
        for (var i = 0u; i < 128u; i++) {
            if (s_similarities[i] < best_dist_sq) {
                best_dist_sq = s_similarities[i];
                best_idx = shared_sort_indices[i];
            }
            best_food_dist_sq = min(best_food_dist_sq, s_food_dist_sq[i]);
        }
        // Publish the nearest in-range food distance (food-sense navigation feature);
        // FOOD_SENSE_RADIUS sentinel when none is within range.
        let food_sense_radius_sq = FOOD_SENSE_RADIUS * FOOD_SENSE_RADIUS;
        physics_state[b + P_NEAREST_FOOD_DISTANCE] = select(
            FOOD_SENSE_RADIUS,
            sqrt(best_food_dist_sq),
            best_food_dist_sq < food_sense_radius_sq);

        // Compute signed bearing from facing direction to nearest food.
        // bearing = atan2(cross(facing, to_food).y, dot(facing, to_food))
        // In XZ plane: facing is normalized, to_food is displacement to food
        // Find the food with the minimum distance and compute bearing from it.
        if (best_food_dist_sq < food_sense_radius_sq) {
            let agent_pos = vec3f(
                physics_state[b + P_POS_X],
                physics_state[b + P_POS_Y],
                physics_state[b + P_POS_Z]);
            let food_count = wc_u32(WC_FOOD_COUNT);
            let food_sense_radius = FOOD_SENSE_RADIUS;

            // Find the food item with the minimum distance in food-sense range
            var min_dist_sq = food_sense_radius_sq;
            var best_food_idx = 0xFFFFFFFFu;
            for (var f = 0u; f < food_count; f++) {
                if (atomicLoad(&food_flags[f]) != 0u) { continue; } // already consumed
                let fbase = f * FOOD_STATE_STRIDE;
                let food_pos = vec3f(
                    food_state[fbase + FOOD_POSITION_X],
                    food_state[fbase + FOOD_POSITION_Y],
                    food_state[fbase + FOOD_POSITION_Z]);
                let to_food = food_pos - agent_pos;
                let d_sq = dot(to_food, to_food);
                if (d_sq < min_dist_sq) {
                    min_dist_sq = d_sq;
                    best_food_idx = f;
                }
            }

            if (best_food_idx != 0xFFFFFFFFu) {
                let food_base = best_food_idx * FOOD_STATE_STRIDE;
                let food_pos = vec3f(
                    food_state[food_base + FOOD_POSITION_X],
                    food_state[food_base + FOOD_POSITION_Y],
                    food_state[food_base + FOOD_POSITION_Z]);
                let to_food = food_pos - agent_pos;

                let facing_x = physics_state[b + P_FACING_X];
                let facing_z = physics_state[b + P_FACING_Z];
                // Cross product in XZ plane: (facing_x, facing_z) × (to_food.x, to_food.z)
                // gives y-component = facing_x * to_food.z - facing_z * to_food.x
                let cross_y = facing_x * to_food.z - facing_z * to_food.x;
                // Dot product: facing · to_food (for atan2 argument order)
                let dot_val = facing_x * to_food.x + facing_z * to_food.z;
                physics_state[b + P_NEAREST_FOOD_BEARING] = atan2(cross_y, dot_val);
            } else {
                // Should not happen if best_food_dist_sq < food_sense_radius_sq, but be defensive
                physics_state[b + P_NEAREST_FOOD_BEARING] = 0.0;
            }
        } else {
            // No food in range; bearing is undefined, sentinel to 0.0
            physics_state[b + P_NEAREST_FOOD_BEARING] = 0.0;
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
    // Preserve the physics-recorded death tick through the reset so CPU
    // readback can attribute this death to its exact tick.
    let saved_last_death_tick = physics_state[base + P_LAST_DEATH_TICK];
    // Preserve cumulative effort telemetry
    let saved_distance     = physics_state[base + P_DISTANCE_TRAVELED];
    let saved_energy_spent = physics_state[base + P_ENERGY_SPENT];
    let saved_danger_path  = physics_state[base + P_DANGER_PATH_LENGTH];
    // Preserve cumulative avoidance intent counters (generation-cumulative)
    let saved_avoidance_sense_range = physics_state[base + P_AVOIDANCE_SENSE_RANGE_TICKS];
    let saved_avoidance_turns_opposing = physics_state[base + P_AVOIDANCE_TURNS_OPPOSING];
    // Preserve cumulative approach intent counters (generation-cumulative)
    let saved_approach_sense_range = physics_state[base + P_APPROACH_SENSE_RANGE_TICKS];
    let saved_approach_turns_toward = physics_state[base + P_APPROACH_TURNS_TOWARD];

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
    physics_state[base + P_LAST_DEATH_TICK] = saved_last_death_tick;
    // Restore cumulative effort telemetry (generation-cumulative, never reset)
    physics_state[base + P_DISTANCE_TRAVELED]  = saved_distance;
    physics_state[base + P_ENERGY_SPENT]       = saved_energy_spent;
    physics_state[base + P_DANGER_PATH_LENGTH] = saved_danger_path;
    // Restore cumulative avoidance intent (generation-cumulative, never reset)
    physics_state[base + P_AVOIDANCE_SENSE_RANGE_TICKS] = saved_avoidance_sense_range;
    physics_state[base + P_AVOIDANCE_TURNS_OPPOSING] = saved_avoidance_turns_opposing;
    // Restore cumulative approach intent (generation-cumulative, never reset)
    physics_state[base + P_APPROACH_SENSE_RANGE_TICKS] = saved_approach_sense_range;
    physics_state[base + P_APPROACH_TURNS_TOWARD] = saved_approach_turns_toward;
    // Reset the food-sense distance to its no-food sentinel until the next
    // food-detect pass. P_PREV_POTENTIAL is a reserved slot (shaping removed);
    // zero it on respawn so no stale value carries across death.
    physics_state[base + P_NEAREST_FOOD_DISTANCE] = FOOD_SENSE_RADIUS;
    physics_state[base + P_PREV_POTENTIAL]        = 0.0;
    // Navigation telemetry: bearing and danger will be recomputed on next ticks
    physics_state[base + P_NEAREST_FOOD_BEARING]  = 0.0;
    physics_state[base + P_IN_DANGER_BIOME]       = 0.0;
    physics_state[base + P_NEAREST_DANGER_DISTANCE] = DANGER_SENSE_RADIUS;
    physics_state[base + P_NEAREST_DANGER_BEARING]  = 0.0;
    // P_PREV_DANGER_POTENTIAL is a reserved slot (avoidance shaping removed);
    // zero it on respawn so no stale value carries across death.
    physics_state[base + P_PREV_DANGER_POTENTIAL]   = 0.0;

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

    // Terminal lesson: the transition into death is the one experience the
    // within-lifetime learner must never miss. Apply one final TD update
    // with the maximum negative error through the eligibility traces the
    // dying life accumulated — then clear them below so no credit leaks
    // into the next life. Without this, dying carries zero learning signal
    // and the full-energy respawn makes death read as a free heal.
    let terminal_value_bias_trace = brain_state[brain_base + O_TRACE_BIASES];
    let terminal_forward_bias_trace = brain_state[brain_base + O_TRACE_BIASES + 1u];
    let terminal_turn_bias_trace = brain_state[brain_base + O_TRACE_BIASES + 2u];
    brain_state[brain_base + O_VALUE_BIAS] += CRITIC_LEARNING_RATE * TERMINAL_DEATH_TD_ERROR * terminal_value_bias_trace;
    brain_state[brain_base + O_ACT_BIASES] += ACTION_WEIGHT_LEARNING_RATE * TERMINAL_DEATH_TD_ERROR * terminal_forward_bias_trace;
    brain_state[brain_base + O_ACT_BIASES + 1u] += ACTION_WEIGHT_LEARNING_RATE * TERMINAL_DEATH_TD_ERROR * terminal_turn_bias_trace;
    for (var i = 0u; i < ENCODED_DIMENSION; i++) {
        brain_state[brain_base + O_VALUE_WEIGHTS + i] += CRITIC_LEARNING_RATE * TD_VECTOR_SCALE * TERMINAL_DEATH_TD_ERROR * brain_state[brain_base + O_TRACE_CRITIC + i];
        brain_state[brain_base + O_ACTION_FORWARD_WEIGHTS + i] += ACTION_WEIGHT_LEARNING_RATE * ACTOR_VECTOR_SCALE * TERMINAL_DEATH_TD_ERROR * brain_state[brain_base + O_TRACE_FWD + i];
        brain_state[brain_base + O_ACTION_TURN_WEIGHTS + i] += ACTION_WEIGHT_LEARNING_RATE * ACTOR_VECTOR_SCALE * TERMINAL_DEATH_TD_ERROR * brain_state[brain_base + O_TRACE_TURN + i];
    }

    // Reset TD transients: eligibility traces and the previous-state value
    // are episodic — credit must never leak across the death boundary.
    // The value weights themselves are learned knowledge and survive.
    for (var i = 0u; i < ENCODED_DIMENSION; i++) {
        brain_state[brain_base + O_TRACE_CRITIC + i] = 0.0;
        brain_state[brain_base + O_TRACE_FWD + i] = 0.0;
        brain_state[brain_base + O_TRACE_TURN + i] = 0.0;
    }
    brain_state[brain_base + O_TRACE_BIASES] = 0.0;
    brain_state[brain_base + O_TRACE_BIASES + 1u] = 0.0;
    brain_state[brain_base + O_TRACE_BIASES + 2u] = 0.0;
    brain_state[brain_base + O_PREV_VALUE] = 0.0;
}

// ══════════════════════════════════════════════════════════════════════════
// Brain tick inner — delegates to the 7 cooperative passes from brain_tick.wgsl
// ══════════════════════════════════════════════════════════════════════════

fn brain_tick_inner(agent_id: u32, tid: u32 /* KERNEL_SUBGROUP_TOPK_PARAMS */) {
    // Read the workgroup-uniform alive flag (broadcast by thread 0 before the
    // preceding workgroupBarrier()). Because `s_alive` is identical across the
    // workgroup by construction, cooperative passes with their own internal
    // barriers (`coop_recall_topk`, `coop_predict_and_act`,
    // `coop_learn_and_store`) are safe: all 256 threads either enter together
    // (hitting every internal barrier) or skip together. Inter-pass barriers
    // live outside the guards so they execute regardless of logical state.
    // See top-of-file SAFETY INVARIANT.
    let alive = s_alive != 0u;

    // Measurement-only per-pass cap: run only the first
    // `limit` cooperative passes so their cumulative GPU cost can be profiled
    // pass-by-pass (sweep `XAGENT_KERNEL_PASS_LIMIT = 0..7`; consecutive deltas
    // are the per-pass costs). `limit` is the kernel push constant, so it is
    // uniform across the whole dispatch; `alive` is the broadcast `s_alive`, so
    // `alive && (idx < limit)` is workgroup-uniform and every gated pass is
    // reached together by all 256 threads — exactly like the bare `alive` guard.
    // The barriers below stay UNCONDITIONAL, so barrier uniformity (the
    // top-of-file SAFETY INVARIANT) holds whether a pass runs or is skipped: a
    // skipped pass is skipped *with* all threads, never some. Default 7 runs all
    // passes ⇒ byte-identical to a build without this knob (the determinism
    // tests gate that). Setting it < 7 deliberately produces wrong results and
    // is never on in tests or release.
    let limit = kpc.pass_limit;

    if (alive && 0u < limit) { coop_feature_extract(agent_id, tid); }
    workgroupBarrier();

    // Visual cortex (plan 0008): inserted between feature extraction and encode.
    // It belongs to the early-visual stage, so it shares feature-extract's
    // profiling slot (`0u < limit`) rather than consuming a new `pass_limit`
    // index — keeping the seven counted passes (0..6) and the default limit of 7
    // unchanged. With `CFG_VISUAL_CORTEX_ENABLED` off (or in the passthrough
    // skeleton) it is a no-op, so the encoded state stays byte-identical to the
    // pre-task build. The guard is workgroup-uniform and precedes the barrier.
    if (alive && 0u < limit) { coop_visual_cortex(agent_id, tid); }
    workgroupBarrier();

    if (alive && 1u < limit) { coop_encode(agent_id, tid); }
    workgroupBarrier();

    if (alive && 2u < limit) { coop_habituate_homeo(agent_id, tid); }
    storageBarrier(); workgroupBarrier();

    if (alive && 3u < limit) { coop_recall_score(agent_id, tid); }
    workgroupBarrier();

    if (alive && 4u < limit) { coop_recall_topk(agent_id, tid /* KERNEL_SUBGROUP_TOPK_ARGS */); }
    storageBarrier(); workgroupBarrier();

    if (alive && 5u < limit) { coop_predict_and_act(agent_id, tid, false); }
    storageBarrier(); workgroupBarrier();

    if (alive && 6u < limit) { coop_learn_and_store(agent_id, tid, true); }
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
// Brain inputs have two visibility regimes:
//
//   * Lagged via `sensory_buffer` (one-batch lag, consistent across strides):
//     vision (color + depth), and the proprioceptive signals that the vision
//     pass packs alongside vision — velocity, facing, angular state, touch.
//     The external vision pass runs in the same GPU command encoder AFTER
//     this kernel dispatch, so the brain reads sensory_buffer written by the
//     *previous* batch's vision pass.
//
//   * Same-cycle direct reads from `physics_state`: `P_ALIVE` (broadcast
//     through `s_alive`, see SAFETY INVARIANT), and the homeostasis /
//     staleness inputs consumed by `coop_habituate_homeo`
//     (`P_ENERGY` / `P_INTEGRITY` / `P_MAX_*`) and `coop_predict_and_act`
//     (`P_POS_X` / `P_POS_Z`). These are made visible to all 256 threads by
//     the `storageBarrier(); workgroupBarrier();` pair that follows each
//     thread-0-only phase.
//
// When brain_tick_stride == vision_stride the batch covers exactly
// (vision_stride * brain_tick_stride) physics ticks and vision runs once
// at the end of the batch, ready for the next batch's kernel.
// ══════════════════════════════════════════════════════════════════════════

// `start_tick` arrives per-batch via a push constant so that multiple
// kernel-batches can share ONE `world_config` uniform write and ONE submit
// (`vision_stride` / `brain_tick_stride` are constant across full batches, so
// the uniform no longer needs to be rewritten per batch just to carry the
// tick). The exact `u32` is strictly more precise than the former
// `WC_TICK = (tick as f32)` round-trip and matches it for every tick ≤ 2^24.
// `pass_limit` is the measurement-only per-cooperative-pass cap:
// `brain_tick_inner` runs only the first `pass_limit` of its seven
// cooperative passes so their cumulative GPU cost can be profiled pass-by-pass.
// It reuses the formerly-unused second push-constant word, so no uniform-slot
// or `WORLD_CONFIG_SIZE` change is needed. The host sets it from
// `XAGENT_KERNEL_PASS_LIMIT` (default 7 = all passes ⇒ byte-identical results;
// the determinism tests gate this). It is a push constant, hence uniform across
// the whole dispatch — see the gating in `brain_tick_inner`.
struct KernelPushConstants {
    start_tick: u32,
    pass_limit: u32,
}
var<push_constant> kpc: KernelPushConstants;

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
    let start_tick = kpc.start_tick;

    for (var cycle = 0u; cycle < vision_stride; cycle++) {
        let base_tick = start_tick + cycle * stride;

        // Per-agent physics: thread 0 loops over brain_tick_stride sub-ticks.
        // Physics always precedes brain within the same cycle (barrier below).
        // Thread 0 is the sole writer of `P_ALIVE`, so it broadcasts the
        // post-physics value into `s_alive` for the workgroup to read after
        // the barrier. `storageBarrier()` is required because thread 0's
        // writes to `physics_state` (position, velocity, energy, integrity,
        // P_ALIVE) are read by other threads in `agent_food_detect` and by the
        // cooperative brain passes that read `physics_state` directly —
        // `workgroupBarrier()` alone would not publish storage writes.
        if (tid == 0u) {
            for (var t = 0u; t < stride; t++) {
                agent_physics(agent_id, base_tick + t);
            }
            s_alive = select(0u, 1u, physics_state[agent_id * PHYS_STRIDE + P_ALIVE] >= 0.5);
        }
        storageBarrier(); workgroupBarrier();

        // Brute-force food detection: all 256 threads cooperate
        agent_food_detect(agent_id, tid);
        workgroupBarrier();

        // Danger detection: thread 0 scans biome grid for nearest danger
        if (tid == 0u && wc_u32(WC_DANGER_PERCEPT_ENABLED) != 0u) {
            agent_danger_detect(agent_id);
        }
        workgroupBarrier();

        // Avoidance and approach accumulation: thread 0 increments the counters based on
        // motor turn and the danger/food bearings just computed. This runs after
        // danger_detect and food_detect so the counters read same-cycle sensory values.
        if (tid == 0u) {
            let decision_base = agent_id * DECISION_STRIDE;
            let motor_turn = decision_buffer[decision_base + DECISION_MOTOR + 1u];
            agent_avoidance_accumulate(agent_id, motor_turn);
            agent_approach_accumulate(agent_id, motor_turn);
        }
        workgroupBarrier();

        // Death/respawn: thread 0. Re-broadcasts `s_alive` because respawn may
        // flip `P_ALIVE` back to 1. `storageBarrier()` is required because
        // respawn rewrites `physics_state`, `brain_state`, and
        // `pattern_buffer`, all of which are read by the cooperative
        // passes in `brain_tick_inner` below.
        if (tid == 0u) {
            agent_death_respawn(agent_id, base_tick);
            s_alive = select(0u, 1u, physics_state[agent_id * PHYS_STRIDE + P_ALIVE] >= 0.5);
        }
        storageBarrier(); workgroupBarrier();

        // Brain: all 256 threads, 7 cooperative passes.
        // Reads sensory_buffer (vision + proprioception) from the previous batch's
        // vision pass.  Physics state updated in this cycle is NOT yet in
        // sensory_buffer — that update happens in the vision pass at the end of
        // this batch, making it available for the following batch.
        //
        // SENSORY-LAG RISK (issue #115): this read is stale by exactly one batch =
        // vision_stride * brain_tick_stride physics ticks. The lag is intentional
        // and constant, but credit assignment pairs a motor command with the
        // gradient it produced — the larger the lag, the more the visual evidence
        // at decision time desynchronizes from the action's outcome, the failure
        // mode behind the circling investigation. The product is bounded on the
        // Rust side by `BrainConfig::MAX_SENSORY_LAG_TICKS` (asserted in
        // `GpuKernel::new`); do NOT grow the strides or make them dynamic past that
        // bound without revalidating credit assignment.
        brain_tick_inner(agent_id, tid /* KERNEL_SUBGROUP_TOPK_INNER_ARGS */);
        workgroupBarrier();
    }
}
