// Cached visual partials are fresh products of the actual stored matrix and
// the predicted adapted inputs. They are used only if every visual input has
// the identical FP32 bits now. Otherwise the original packed encoder runs.
// s_reinf_dot is unused until later context/whitening/learning phases.
fn coop_encode(agent_id: u32, tid: u32) {
    let private_base = PROJECTION_STORAGE_BASE + agent_id * PROJECTION_AGENT_STRIDE;
    if (tid < PROJECTION_GROUP_COUNT) {
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
        for (var group = 0u; group < PROJECTION_GROUP_COUNT; group++) {
            reuse = reuse && s_reinf_dot[group] == 1.0;
        }
        s_reinf_dot[PROJECTION_GROUP_COUNT] = select(0.0, 1.0, reuse);
        packed_encoder.scratch[private_base + PROJECTION_USED_OFFSET] = select(0.0, 1.0, reuse);
        let count_offset = select(PROJECTION_MISSES_OFFSET, PROJECTION_HITS_OFFSET, reuse);
        packed_encoder.scratch[private_base + count_offset] += 1.0;
    }
    let reuse = workgroupUniformLoad(&s_reinf_dot[PROJECTION_GROUP_COUNT]) != 0.0;
    if (!reuse) {
        fresh_projection_original_encode(agent_id, tid);
    } else {
        let output_vector = tid % PACKED_ENCODER_OUTPUT_VECTORS;
        let lane = tid / PACKED_ENCODER_OUTPUT_VECTORS;
        let dimension = output_vector * PACKED_ENCODER_WIDTH;
        if (lane < DENSE_INNER_LANES) {
            var partial = vec4<f32>(0.0);
            if (lane == 0u) {
                let bias = agent_id * BRAIN_STRIDE + O_ENC_BIASES + dimension;
                partial = vec4<f32>(brain_state[bias], brain_state[bias + 1u], brain_state[bias + 2u], brain_state[bias + 3u]);
            }
            for (var group = lane; group < PROJECTION_GROUP_COUNT; group += DENSE_INNER_LANES) {
                let base = private_base + PROJECTION_PARTIAL_OFFSET + group * ENCODED_DIMENSION + dimension;
                partial += vec4<f32>(packed_encoder.scratch[base], packed_encoder.scratch[base + 1u], packed_encoder.scratch[base + 2u], packed_encoder.scratch[base + 3u]);
            }
            let matrix_base = agent_id * FEATURE_COUNT * PACKED_ENCODER_OUTPUT_VECTORS;
            for (var feature = PROJECTION_VISUAL_COUNT + lane; feature < FEATURE_COUNT; feature += DENSE_INNER_LANES) {
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
