// ── Phase: Death detection, respawn, and brain reset ───────────────────────
// Runs after physics. If an agent died (died_flag=1), pick a safe spawn
// position via GPU RNG, reset physics state, and halve/zero brain state.
// Mirrors flush_death_signals() + respawn_agent() from the CPU path.

fn phase_death_respawn(tid: u32, tick: u32) {
    let base = tid * PHYS_STRIDE;
    if (physics_state[base + P_DIED_FLAG] < 0.5) { return; }

    // ── 1. Pick a safe spawn position ──────────────────────────────────────
    let world_half = wc_f32(WC_WORLD_HALF_BOUND);
    var spawn_x = 0.0;
    var spawn_z = 0.0;
    var found = false;
    for (var attempt = 0u; attempt < 50u; attempt++) {
        let h = pcg_hash(tick * 256u + tid + attempt);
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
    // Fallback: if all 50 attempts hit danger, use last candidate anyway
    if (!found) {
        let h = pcg_hash(tick * 256u + tid);
        let h2 = pcg_hash(h);
        spawn_x = (hash_to_float(h) * 2.0 - 1.0) * world_half;
        spawn_z = (hash_to_float(h2) * 2.0 - 1.0) * world_half;
    }
    let spawn_y = sample_height(spawn_x, spawn_z) + AGENT_HALF_HEIGHT;

    // ── 2. Preserve fitness fields before reset ────────────────────────────
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

    // ── 3. Reset physics state ─────────────────────────────────────────────
    // Zero the full stride first, then write specific values.
    for (var i = 0u; i < PHYS_STRIDE; i++) {
        physics_state[base + i] = 0.0;
    }
    physics_state[base + P_POS_X]           = spawn_x;
    physics_state[base + P_POS_Y]           = spawn_y;
    physics_state[base + P_POS_Z]           = spawn_z;
    // velocity: already zero from the loop
    // facing: (0, 0, 1)
    physics_state[base + P_FACING_Z]        = 1.0;
    // yaw, angular_vel: already zero
    physics_state[base + P_ENERGY]          = max_energy;
    physics_state[base + P_MAX_ENERGY]      = max_energy;
    physics_state[base + P_INTEGRITY]       = max_integrity;
    physics_state[base + P_MAX_INTEGRITY]   = max_integrity;
    physics_state[base + P_PREV_ENERGY]     = max_energy;
    physics_state[base + P_PREV_INTEGRITY]  = max_integrity;
    physics_state[base + P_ALIVE]           = 1.0;
    // died_flag: already zero
    physics_state[base + P_MEMORY_CAP]      = memory_cap;
    physics_state[base + P_PROCESSING_SLOTS] = processing_slots;
    // Restore fitness fields
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
    // Danger percept telemetry: reset to sentinel values, will be recomputed
    physics_state[base + P_NEAREST_DANGER_DISTANCE] = DANGER_SENSE_RADIUS;
    physics_state[base + P_NEAREST_DANGER_BEARING]  = 0.0;
    // Reserved potential slots (shaping removed); zeroed on respawn so no stale
    // value carries across death. Food-approach and danger-avoidance both reset.
    physics_state[base + P_PREV_POTENTIAL]          = 0.0;
    physics_state[base + P_PREV_DANGER_POTENTIAL]   = 0.0;

    // ── 4. Reset brain state ───────────────────────────────────────────────
    let brain_base = tid * BRAIN_STRIDE;

    // Halve reinforcement values
    let pat_base = tid * PATTERN_STRIDE;
    for (var i = 0u; i < MEMORY_CAP; i++) {
        pattern_buffer[pat_base + O_PAT_REINF + i] *= 0.5;
    }

    // Zero homeostasis EMAs (6 values)
    for (var i = 0u; i < 6u; i++) {
        brain_state[brain_base + O_HOMEO + i] = 0.0;
    }

    // Reset exploration rate
    brain_state[brain_base + O_EXPLORATION_RATE] = 0.5;

    // Reset position-based staleness state (position ring, cursor, length, accum, factor)
    for (var i = 0u; i < POS_RING_LEN; i++) {
        brain_state[brain_base + O_POS_RING_X + i] = 0.0;
        brain_state[brain_base + O_POS_RING_Z + i] = 0.0;
    }
    brain_state[brain_base + O_POS_RING_CURSOR] = 0.0;
    brain_state[brain_base + O_POS_RING_LEN] = 0.0;
    brain_state[brain_base + O_ACCUM_FWD] = 0.0;
    brain_state[brain_base + O_FATIGUE_FACTOR] = 1.0;

    // Reset habituation (fresh start for new life's perceptual context)
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
