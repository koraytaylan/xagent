// Shared output layout: similarity, cached similarity, dot, query norm,
// and query squared norm, each with MEMORY_CAP entries.
const RAW_CACHE: u32 = MEMORY_CAP;
const RAW_DOT: u32 = RAW_CACHE + MEMORY_CAP;
const RAW_NORM: u32 = RAW_DOT + MEMORY_CAP;
const RAW_SQUARE: u32 = RAW_NORM + MEMORY_CAP;
override RAW_COOPERATIVE: bool = false;

@compute @workgroup_size(BRAIN_WORKGROUP_SIZE)
fn recall_cosine_probe(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let agent = group.x;
    if (tid < ENCODED_DIMENSION) {
        s_memory_key[tid] = brain_state[agent * BRAIN_STRIDE + O_PREV_ENCODED + tid];
    }
    workgroupBarrier();
    coop_recall_score(agent, tid);
    workgroupBarrier();
    if (tid < MEMORY_CAP) {
        let output = agent * BRAIN_SCRATCH_STRIDE;
        brain_scratch[output + tid] = s_similarities[tid];
        brain_scratch[output + RAW_CACHE + tid] = s_argmin_val[tid];
        if (RAW_COOPERATIVE) {
            brain_scratch[output + RAW_DOT + tid] = s_reinf_dot[tid] + s_reinf_dot[tid + MEMORY_CAP];
            brain_scratch[output + RAW_NORM + tid] = s_enc_norm;
            brain_scratch[output + RAW_SQUARE + tid] = s_dense_partials[0];
        }
    }
}
