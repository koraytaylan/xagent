// ParallelTiled tiled encoder credit: per-agent encoder-weight gradient updates.
//
// Dispatched as workgroups(agent_count, ENCODED_DIMENSION/16, 1), workgroup_size
// 256. Mirrors the fused coop_learn_and_store "7b" encoder-credit update but
// reads features from SCRATCH_FEATURES and the per-dim credit from
// decision_buffer[DECISION_CREDIT] (written by the tail). Each (dim, lane) writes
// a disjoint set of O_ENC_WEIGHTS columns (f stride-16), so the update is
// write-only with no reduction and no barriers — weight values are identical to
// the fused path regardless of lane count. Dead agents are skipped (the fused
// encoder-credit runs only under `alive`). Concatenated with common.wgsl only.

const CREDIT_DIMS_PER_WG: u32 = 16u;
const CREDIT_LANES_PER_DIM: u32 = 16u;

@compute @workgroup_size(256)
fn phase_brain_encoder_credit_tiled(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let agent_id = wgid.x;
    let dim_tile = wgid.y;
    let tid = lid.x;

    // Skip dead agents — no barriers in this shader, so a per-thread early
    // return is uniformity-safe.
    if (physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) {
        return;
    }

    let dim_in_tile = tid / CREDIT_LANES_PER_DIM;
    let lane = tid % CREDIT_LANES_PER_DIM;
    let dim = dim_tile * CREDIT_DIMS_PER_WG + dim_in_tile;

    let brain_base = agent_id * BRAIN_STRIDE;
    let decision_base = agent_id * DECISION_STRIDE;
    let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;

    let action_credit = decision_buffer[decision_base + DECISION_CREDIT + dim];
    if (abs(action_credit) >= CREDIT_EPSILON) {
        let learning_rate = brain_config[1].x;
        let scale = learning_rate * action_credit * ENCODER_CREDIT_SCALE;
        for (var f = lane; f < FEATURE_COUNT; f += CREDIT_LANES_PER_DIM) {
            var w = brain_state[brain_base + O_ENC_WEIGHTS + f * ENCODED_DIMENSION + dim]
                + scale * brain_scratch[agent_scratch + SCRATCH_FEATURES + f];
            w = clamp(w, -2.0, 2.0);
            brain_state[brain_base + O_ENC_WEIGHTS + f * ENCODED_DIMENSION + dim] = w;
        }
    }
}
