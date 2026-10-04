// One workgroup owns one agent's pattern maintenance. The preceding main
// stored its current key and norm, overwriting both reinforcement and valence
// of that slot. Other global groups write world state or encoder weights.
var<workgroup> memory_offload_key: array<f32, ENCODED_DIMENSION>;
var<workgroup> memory_offload_dot: array<f32, 256>;
var<workgroup> memory_offload_score: array<f32, MEMORY_CAP>;
var<workgroup> memory_offload_index: array<u32, MEMORY_CAP>;
var<workgroup> memory_offload_alive: u32;

fn memory_offload_reduce(tid: u32) {
    var stride: u32 = ENCODED_DIMENSION / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            memory_offload_dot[tid] = memory_offload_dot[tid] + memory_offload_dot[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
}

fn maintain_stored_memory(agent_id: u32, tid: u32) {
    if (tid == 0u) {
        memory_offload_alive = select(0u, 1u, physics_state[agent_id * PHYS_STRIDE + P_ALIVE] >= 0.5);
    }
    let alive = workgroupUniformLoad(&memory_offload_alive) != 0u;
    if (alive) {
        let brain_base = agent_id * BRAIN_STRIDE;
        let pattern_base = agent_id * PATTERN_STRIDE;
        let stored_idx = u32(pattern_buffer[pattern_base + O_LAST_STORED_IDX]);
        let learning_rate = brain_config[1].x;
        let decay_rate = brain_config[1].y;
        let tick = brain_state[brain_base + O_TICK_COUNT];
        let salience_label = brain_state[brain_base + O_SALIENCE_LABEL];
        let prediction_error = physics_state[agent_id * PHYS_STRIDE + P_PREDICTION_ERROR];
        let encoded_norm = pattern_buffer[pattern_base + O_PAT_NORMS + stored_idx];
        if (tid < ENCODED_DIMENSION) {
            memory_offload_key[tid] = pattern_buffer[pattern_base + tid * MEMORY_CAP + stored_idx];
        }
        storageBarrier();
        workgroupBarrier();

        // MEMORY_OFFLOAD_REINFORCEMENT

        // MEMORY_OFFLOAD_DECAY_AND_MINIMUM
    }
}
