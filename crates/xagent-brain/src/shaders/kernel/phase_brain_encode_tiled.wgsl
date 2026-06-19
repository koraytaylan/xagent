// ParallelTiled tiled encode: per-agent input matrix multiply across dispatches.
//
// Dispatched as workgroups(agent_count, ENCODED_DIMENSION/16, 1) with
// workgroup_size 256. Each workgroup (wgid.y in 0..7) owns 16 encoded output
// dimensions; the 256 lanes split as 16 dims x 16 inner lanes. Each lane
// reduces a stride-16 slice of FEATURE_COUNT inputs from SCRATCH_FEATURES, then
// lane 0 sums the 16 partials in fixed ascending order, adds the bias, and
// applies fast_tanh — matching coop_encode's value within bounded float drift.
// Output goes to SCRATCH_ENCODED (storage) so the single-workgroup tail can read
// it after a dispatch boundary. Concatenated with common.wgsl only.

var<workgroup> s_enc_tile_partials: array<f32, 256>;

const ENCODE_DIMS_PER_WG: u32 = 16u;
const ENCODE_LANES_PER_DIM: u32 = 16u;

@compute @workgroup_size(256)
fn phase_brain_encode_tiled(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let agent_id = wgid.x;
    let dim_tile = wgid.y; // 0..7
    let tid = lid.x;
    let dim_in_tile = tid / ENCODE_LANES_PER_DIM; // 0..15
    let lane = tid % ENCODE_LANES_PER_DIM;        // 0..15
    let dim = dim_tile * ENCODE_DIMS_PER_WG + dim_in_tile; // 0..127

    let brain_base = agent_id * BRAIN_STRIDE;
    let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;

    var partial: f32 = 0.0;
    for (var f = lane; f < FEATURE_COUNT; f += ENCODE_LANES_PER_DIM) {
        partial += brain_scratch[agent_scratch + SCRATCH_FEATURES + f]
            * brain_state[brain_base + O_ENC_WEIGHTS + f * ENCODED_DIMENSION + dim];
    }
    s_enc_tile_partials[tid] = partial;
    workgroupBarrier();

    if (lane == 0u) {
        var sum: f32 = brain_state[brain_base + O_ENC_BIASES + dim];
        let base = dim_in_tile * ENCODE_LANES_PER_DIM;
        for (var l = 0u; l < ENCODE_LANES_PER_DIM; l = l + 1u) {
            sum += s_enc_tile_partials[base + l]; // fixed ascending lane order
        }
        brain_scratch[agent_scratch + SCRATCH_ENCODED + dim] = fast_tanh(sum);
    }
}
