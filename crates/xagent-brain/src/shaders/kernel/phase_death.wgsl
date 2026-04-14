// ── Phase: Death detection, respawn, and brain reset ───────────────────────
// Runs after physics. If an agent died (died_flag=1), pick a safe spawn
// position via GPU RNG, reset physics state, and halve/zero brain state.
// Mirrors flush_death_signals() + respawn_agent() from the CPU path.

fn phase_death_respawn(tid: u32, tick: u32) {
    let base = tid * PHYS_STRIDE;
    if (agent_phys[base + P_DIED_FLAG] < 0.5) { return; }

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
    let saved_food_count   = agent_phys[base + P_FOOD_COUNT];
    let saved_ticks_alive  = agent_phys[base + P_TICKS_ALIVE];
    let saved_death_count  = agent_phys[base + P_DEATH_COUNT] + 1.0;
    let max_energy         = agent_phys[base + P_MAX_ENERGY];
    let max_integrity      = agent_phys[base + P_MAX_INTEGRITY];
    let memory_cap         = agent_phys[base + P_MEMORY_CAP];
    let processing_slots   = agent_phys[base + P_PROCESSING_SLOTS];

    // ── 3. Reset physics state ─────────────────────────────────────────────
    // Zero the full stride first, then write specific values.
    for (var i = 0u; i < PHYS_STRIDE; i++) {
        agent_phys[base + i] = 0.0;
    }
    agent_phys[base + P_POS_X]           = spawn_x;
    agent_phys[base + P_POS_Y]           = spawn_y;
    agent_phys[base + P_POS_Z]           = spawn_z;
    // velocity: already zero from the loop
    // facing: (0, 0, 1)
    agent_phys[base + P_FACING_Z]        = 1.0;
    // yaw, angular_vel: already zero
    agent_phys[base + P_ENERGY]          = max_energy;
    agent_phys[base + P_MAX_ENERGY]      = max_energy;
    agent_phys[base + P_INTEGRITY]       = max_integrity;
    agent_phys[base + P_MAX_INTEGRITY]   = max_integrity;
    agent_phys[base + P_PREV_ENERGY]     = max_energy;
    agent_phys[base + P_PREV_INTEGRITY]  = max_integrity;
    agent_phys[base + P_ALIVE]           = 1.0;
    // died_flag: already zero
    agent_phys[base + P_MEMORY_CAP]      = memory_cap;
    agent_phys[base + P_PROCESSING_SLOTS] = processing_slots;
    // Restore fitness fields
    agent_phys[base + P_FOOD_COUNT]      = saved_food_count;
    agent_phys[base + P_TICKS_ALIVE]     = saved_ticks_alive;
    agent_phys[base + P_DEATH_COUNT]     = saved_death_count;

    // ── 4. Reset brain state ───────────────────────────────────────────────
    let brain_base = tid * BRAIN_STRIDE;

    // Halve reinforcement values
    let pat_base = tid * PATTERN_STRIDE;
    for (var i = 0u; i < MEMORY_CAP; i++) {
        pattern_buf[pat_base + O_PAT_REINF + i] *= 0.5;
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
    for (var i = 0u; i < DIM; i++) {
        brain_state[brain_base + O_HAB_EMA + i] = 0.0;
        brain_state[brain_base + O_HAB_ATTEN + i] = 1.0;
        brain_state[brain_base + O_PREV_ENCODED + i] = 0.0;
    }

    // Zero action history
    let hist_base = tid * HISTORY_STRIDE;
    for (var i = 0u; i < HISTORY_STRIDE; i++) {
        history_buf[hist_base + i] = 0.0;
    }
}
