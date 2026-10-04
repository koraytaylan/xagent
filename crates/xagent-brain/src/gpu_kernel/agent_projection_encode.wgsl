// The cached values are the four original visual-prefix sums, including the
// lane-zero bias. Only identical visual inputs may continue those recurrences.
// s_reinf_dot is released before this entry and is overwritten by later phases.
fn coop_encode(agent_id: u32, tid: u32) {
    let private_base = PROJECTION_STORAGE_BASE + agent_id * PROJECTION_AGENT_STRIDE;
    if (tid < PROJECTION_GATE_GROUP_COUNT) {
        var same = true;
        let first = tid * PROJECTION_FEATURES_PER_GROUP;
        let last = min(first + PROJECTION_FEATURES_PER_GROUP, PROJECTION_VISUAL_COUNT);
        for (var feature = first; feature < last; feature++) {
            let predicted = packed_encoder.scratch[agent_id * BRAIN_SCRATCH_STRIDE + SCRATCH_FEATURES + feature];
            same = same && bitcast<u32>(predicted) == bitcast<u32>(s_features[feature]);
        }
        s_reinf_dot[tid] = select(0.0, 1.0, same);
    }
    workgroupBarrier();
    if (tid == 0u) {
        var reuse = packed_encoder.scratch[private_base + PROJECTION_VALID_OFFSET] == 1.0
            && packed_encoder.scratch[private_base + PROJECTION_DEATH_OFFSET] == physics_state[agent_id * PHYS_STRIDE + P_DEATH_COUNT];
        for (var group = 0u; group < PROJECTION_GATE_GROUP_COUNT; group++) {
            reuse = reuse && s_reinf_dot[group] == 1.0;
        }
        s_reinf_dot[PROJECTION_GATE_GROUP_COUNT] = select(0.0, 1.0, reuse);
        packed_encoder.scratch[private_base + PROJECTION_USED_OFFSET] = select(0.0, 1.0, reuse);
        let count_offset = select(PROJECTION_MISSES_OFFSET, PROJECTION_HITS_OFFSET, reuse);
        packed_encoder.scratch[private_base + count_offset] += 1.0;
    }
    let reuse = workgroupUniformLoad(&s_reinf_dot[PROJECTION_GATE_GROUP_COUNT]) != 0.0;
    if (!reuse) {
        fresh_projection_original_encode(agent_id, tid);
    } else {
        let output_vector = tid % PACKED_ENCODER_OUTPUT_VECTORS;
        let lane = tid / PACKED_ENCODER_OUTPUT_VECTORS;
        let dimension = output_vector * PACKED_ENCODER_WIDTH;
        if (lane < PACKED_ENCODER_INNER_LANES) {
            let base = private_base + PROJECTION_PARTIAL_OFFSET
                + lane * ENCODED_DIMENSION + dimension;
            var partial = vec4<f32>(packed_encoder.scratch[base], packed_encoder.scratch[base + 1u], packed_encoder.scratch[base + 2u], packed_encoder.scratch[base + 3u]);
            let matrix_base = agent_id * FEATURE_COUNT * PACKED_ENCODER_OUTPUT_VECTORS;
            // Resume the original residue class, including when the odd field
            // puts the visual/nonvisual boundary in the middle of a lane set.
            let first_nonvisual = PROJECTION_VISUAL_COUNT
                + (lane + PACKED_ENCODER_INNER_LANES - PROJECTION_VISUAL_COUNT % PACKED_ENCODER_INNER_LANES) % PACKED_ENCODER_INNER_LANES;
            for (var feature = first_nonvisual; feature < FEATURE_COUNT; feature += PACKED_ENCODER_INNER_LANES) {
                partial += s_features[feature]
                    * packed_encoder.weights[matrix_base + feature * PACKED_ENCODER_OUTPUT_VECTORS + output_vector];
            }
            let slot = tid * PACKED_ENCODER_WIDTH;
            s_dense_partials[slot] = partial.x;
            s_dense_partials[slot + 1u] = partial.y;
            s_dense_partials[slot + 2u] = partial.z;
            s_dense_partials[slot + 3u] = partial.w;
        }
        workgroupBarrier();
        if (lane == 0u) {
            for (var component = 0u; component < PACKED_ENCODER_WIDTH; component++) {
                let base = dimension + component;
                let reduced = s_dense_partials[base]
                    + s_dense_partials[base + ENCODED_DIMENSION]
                    + s_dense_partials[base + ENCODED_DIMENSION * 2u]
                    + s_dense_partials[base + ENCODED_DIMENSION * 3u];
                packed_encoder.scratch[private_base + PROJECTION_RAW_OFFSET + base] = reduced;
                s_encoded[base] = fast_tanh(reduced);
            }
        }
        workgroupBarrier();
    }
}
