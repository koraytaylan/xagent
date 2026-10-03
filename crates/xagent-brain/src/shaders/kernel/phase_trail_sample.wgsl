// Phase: record every agent's position into the trail ring.
// 1 thread per agent, tid = agent index. Runs at the end of the global pass,
// only when the pass's tick lies on a trail-sample boundary, so the recorded
// path depends on simulated ticks alone — never on how many ticks the host
// dispatches per submit or how often it reads state back.

fn phase_trail_sample(tid: u32, sample_number: u32) {
    let agent_count = wc_u32(WC_AGENT_COUNT);
    let slot_base = (sample_number % TRAIL_RING_SLOTS) * (agent_count + 1u) * TRAIL_RECORD_STRIDE;

    if tid < agent_count {
        let b = tid * PHYS_STRIDE;
        let record = slot_base + tid * TRAIL_RECORD_STRIDE;
        trail_ring[record]      = physics_state[b + P_POS_X];
        trail_ring[record + 1u] = physics_state[b + P_POS_Y];
        trail_ring[record + 2u] = physics_state[b + P_POS_Z];
        trail_ring[record + 3u] = physics_state[b + P_DEATH_COUNT];
    }

    if tid == 0u {
        let header = slot_base + agent_count * TRAIL_RECORD_STRIDE;
        trail_ring[header] = bitcast<f32>(sample_number + 1u);
    }
}
