// ParallelTiled tiled predictor: per-agent TD(λ) value-function prediction.
//
// Dispatched as workgroups(agent_count, PREDICTOR_DIMENSION/16, 1),
// workgroup_size 256. Each workgroup (wgid.y in 0..7) owns 16 predictor rows;
// the 256 lanes split as 16 rows x 16 inner lanes. Mirrors the fused
// coop_predict_and_act predictor block but reads the encoded vector from
// SCRATCH_ENCODED, trains O_PREDICTOR_WEIGHTS (each lane owns disjoint weight
// columns), then computes the row prediction into SCRATCH_PREDICTION for the
// single-workgroup tail to consume. Dead agents are skipped before any barrier
// (the skip is per-agent, hence workgroup-uniform). Concatenated with
// common.wgsl only.

var<workgroup> s_pred_tile_partials: array<f32, 256>;

const PRED_ROWS_PER_WG: u32 = 16u;
const PRED_LANES_PER_ROW: u32 = 16u;

@compute @workgroup_size(256)
fn phase_brain_predictor_tiled(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let agent_id = wgid.x;
    let row_tile = wgid.y; // 0..7
    let tid = lid.x;

    // Per-agent (workgroup-uniform) liveness gate, taken before any barrier.
    if (physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) {
        return;
    }

    let row_in_tile = tid / PRED_LANES_PER_ROW; // 0..15
    let lane = tid % PRED_LANES_PER_ROW;        // 0..15
    let dim = row_tile * PRED_ROWS_PER_WG + row_in_tile; // 0..127

    let brain_base = agent_id * BRAIN_STRIDE;
    let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;
    let predictor_learning_rate = bc_f32(CFG_LEARNING_RATE);

    // TRAIN: each lane updates disjoint weight columns (j stride-16) of this row.
    let previous_prediction = brain_state[brain_base + O_PREV_PREDICTION + dim];
    let transition_error = previous_prediction - brain_scratch[agent_scratch + SCRATCH_ENCODED + dim];
    let tanh_derivative = 1.0 - previous_prediction * previous_prediction;
    for (var j = lane; j < ENCODED_DIMENSION; j += PRED_LANES_PER_ROW) {
        let previous_input = brain_state[brain_base + O_PREV_ENCODED + j];
        let grad = clamp(transition_error * tanh_derivative * previous_input, -1.0, 1.0);
        var w = brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j]
            - predictor_learning_rate * grad;
        w = clamp(w, -3.0, 3.0);
        brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j] = w;
    }
    workgroupBarrier(); // all weight writes visible before the predict reduction

    // PREDICT: each lane accumulates a stride-16 partial, lane 0 reduces in
    // fixed ascending order into SCRATCH_PREDICTION.
    var partial: f32 = 0.0;
    for (var j = lane; j < ENCODED_DIMENSION; j += PRED_LANES_PER_ROW) {
        partial += brain_scratch[agent_scratch + SCRATCH_ENCODED + j]
            * brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j];
    }
    s_pred_tile_partials[tid] = partial;
    workgroupBarrier();

    if (lane == 0u) {
        var sum: f32 = 0.0;
        let base = row_in_tile * PRED_LANES_PER_ROW;
        for (var l = 0u; l < PRED_LANES_PER_ROW; l = l + 1u) {
            sum += s_pred_tile_partials[base + l]; // fixed ascending lane order
        }
        brain_scratch[agent_scratch + SCRATCH_PREDICTION + dim] = sum;
    }
}
