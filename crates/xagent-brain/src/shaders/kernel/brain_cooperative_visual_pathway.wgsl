// A hemifield sense whitened by a Jacobi matrix is prepared cooperatively.
fn sense_prepared_cooperatively(sense: u32) -> bool {
    return sense_whitening_kind(sense) == SENSE_WHITENING_KIND_JACOBI
        && sense_input_kind(sense) == SENSE_INPUT_KIND_VISUAL_HEMIFIELDS;
}

// Scalar hemifield pooling preserves the ordered per-ray divisions and additions.
// Released whitening scratch holds a sense's centred components and then its
// independently whitened outputs; covariance cells have unique writers. Senses
// run one after another, each closed by its barriers.
fn cooperative_prepare_jacobi_senses(brain_base: u32, tid: u32) {
    let cortex_on = bc_f32(CFG_VISUAL_CORTEX_ENABLED) != 0.0;
    for (var sense = 0u; sense < SENSE_COUNT; sense++) {
        let n = sense_channels(sense);
        let prepared = sense_prepared_cooperatively(sense);
        let rate = sense_adaptation_rate(sense);
        if (prepared && !cortex_on && tid == 0u) {
            let raw = vision_hemifields();
            for (var k = 0u; k < n; k++) {
                s_reinf_dot[k] = raw[k] - brain_state[brain_base + sense_mean(sense) + k];
            }
        }
        // Publish every old-mean-centred component before any mean/covariance write.
        workgroupBarrier();
        if (prepared && tid < n) {
            if (cortex_on) {
                brain_state[brain_base + sense_input(sense) + tid] = 0.0;
            } else {
                var whitened = 0.0;
                for (var j = 0u; j < n; j++) {
                    whitened += brain_state[brain_base + sense_whitening(sense) + tid * n + j]
                        * s_reinf_dot[j];
                }
                brain_state[brain_base + sense_input(sense) + tid] = whitened;
                s_reinf_dot[n + tid] = whitened;
                brain_state[brain_base + sense_mean(sense) + tid] += rate * s_reinf_dot[tid];
            }
        }
        if (prepared && !cortex_on && tid < n * n) {
            let i = tid / n;
            let j = tid % n;
            let slot = brain_base + sense_covariance(sense) + tid;
            brain_state[slot] += rate * (s_reinf_dot[i] * s_reinf_dot[j] - brain_state[slot]);
        }
        storageBarrier();
        workgroupBarrier();
    }
}

// Only invocation zero calls this after the cooperative preparation. Preserve
// the original ascending policy sum, including the cortex zero.
fn prepared_sense_turn(brain_base: u32, sense: u32, cortex_on: bool) -> f32 {
    if (cortex_on) { return 0.0; }
    var turn = 0.0;
    for (var i = 0u; i < sense_channels(sense); i++) {
        turn += brain_state[brain_base + sense_turn_weights(sense) + i]
            * brain_state[brain_base + sense_input(sense) + i];
    }
    return turn;
}
