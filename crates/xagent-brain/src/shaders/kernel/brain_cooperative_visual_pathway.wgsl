// Scalar hemifield pooling preserves the ordered per-ray divisions and additions.
// Released whitening scratch holds its eight centred components and then
// eight independently whitened outputs; covariance cells have unique writers.
fn cooperative_prepare_vision_pathway(brain_base: u32, tid: u32) {
    let cortex_on = bc_f32(CFG_VISUAL_CORTEX_ENABLED) != 0.0;
    if (!cortex_on && tid == 0u) {
        let raw = vision_hemifields();
        for (var k = 0u; k < VISION_PATHWAY_INPUTS; k++) {
            s_reinf_dot[k] = raw[k] - brain_state[brain_base + O_VISION_PATHWAY_MEAN + k];
        }
    }
    // Publish every old-mean-centred component before any mean/covariance write.
    workgroupBarrier();
    if (tid < VISION_PATHWAY_INPUTS) {
        if (cortex_on) {
            brain_state[brain_base + O_VISION_PATHWAY_INPUT + tid] = 0.0;
        } else {
            var whitened = 0.0;
            for (var j = 0u; j < VISION_PATHWAY_INPUTS; j++) {
                whitened += brain_state[brain_base + O_VISION_PATHWAY_WHITENING + tid * VISION_PATHWAY_INPUTS + j]
                    * s_reinf_dot[j];
            }
            brain_state[brain_base + O_VISION_PATHWAY_INPUT + tid] = whitened;
            s_reinf_dot[VISION_PATHWAY_INPUTS + tid] = whitened;
            brain_state[brain_base + O_VISION_PATHWAY_MEAN + tid] += VISION_PATHWAY_RATE * s_reinf_dot[tid];
        }
    }
    if (!cortex_on && tid < VISION_PATHWAY_INPUTS * VISION_PATHWAY_INPUTS) {
        let i = tid / VISION_PATHWAY_INPUTS;
        let j = tid % VISION_PATHWAY_INPUTS;
        let slot = brain_base + O_VISION_PATHWAY_COVARIANCE + tid;
        brain_state[slot] += VISION_PATHWAY_RATE * (s_reinf_dot[i] * s_reinf_dot[j] - brain_state[slot]);
    }
    storageBarrier();
    workgroupBarrier();
}

// Only invocation zero calls this after the cooperative preparation. Preserve
// the original ascending eight-term policy sum, including the cortex zero.
fn prepared_vision_pathway_turn(brain_base: u32, cortex_on: bool) -> f32 {
    if (cortex_on) { return 0.0; }
    var turn = 0.0;
    for (var i = 0u; i < VISION_PATHWAY_INPUTS; i++) {
        turn += brain_state[brain_base + O_VISION_TURN_WEIGHTS + i]
            * s_reinf_dot[VISION_PATHWAY_INPUTS + i];
    }
    return turn;
}
