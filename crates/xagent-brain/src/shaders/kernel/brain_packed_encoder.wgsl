// Four adjacent output weights share one typed storage load. Each component
// retains its original ascending stride-four accumulation and reduction.
// Only the first 128 threads perform arithmetic; all 256 reach both barriers.
fn coop_encode(agent_id: u32, tid: u32) {
    let brain_base = agent_id * BRAIN_STRIDE;
    let output_vector = tid % PACKED_ENCODER_OUTPUT_VECTORS;
    let lane = tid / PACKED_ENCODER_OUTPUT_VECTORS;
    if (lane < DENSE_INNER_LANES) {
        let dimension = output_vector * PACKED_ENCODER_WIDTH;
        var partial = vec4<f32>(0.0);
        if (lane == 0u) {
            partial = vec4<f32>(
                brain_state[brain_base + O_ENC_BIASES + dimension],
                brain_state[brain_base + O_ENC_BIASES + dimension + 1u],
                brain_state[brain_base + O_ENC_BIASES + dimension + 2u],
                brain_state[brain_base + O_ENC_BIASES + dimension + 3u],
            );
        }
        let matrix_base = agent_id * FEATURE_COUNT * PACKED_ENCODER_OUTPUT_VECTORS;
        // Two vector loads expose eight independent weights without the
        // thirty-two live weights that an eight-vector prefetch would need.
        for (var feature = lane; feature < FEATURE_COUNT; feature += DENSE_INNER_LANES * PACKED_ENCODER_PREFETCH) {
            let input_first = s_features[feature];
            let weight_first = packed_encoder.weights[matrix_base + feature * PACKED_ENCODER_OUTPUT_VECTORS + output_vector];
            let feature_second = feature + DENSE_INNER_LANES;
            var input_second = 0.0;
            var weight_second = vec4<f32>(0.0);
            if (feature_second < FEATURE_COUNT) {
                input_second = s_features[feature_second];
                weight_second = packed_encoder.weights[matrix_base + feature_second * PACKED_ENCODER_OUTPUT_VECTORS + output_vector];
            }
            partial += input_first * weight_first;
            if (feature_second < FEATURE_COUNT) {
                partial += input_second * weight_second;
            }
        }
        let slot = tid * PACKED_ENCODER_WIDTH;
        s_dense_partials[slot] = partial.x;
        s_dense_partials[slot + 1u] = partial.y;
        s_dense_partials[slot + 2u] = partial.z;
        s_dense_partials[slot + 3u] = partial.w;
    }
    workgroupBarrier();
    if (lane == 0u) {
        let base = output_vector * PACKED_ENCODER_WIDTH;
        let second_lane = base + ENCODED_DIMENSION;
        let third_lane = second_lane + ENCODED_DIMENSION;
        let fourth_lane = third_lane + ENCODED_DIMENSION;
        for (var component = 0u; component < PACKED_ENCODER_WIDTH; component++) {
            let reduced = s_dense_partials[base + component]
                + s_dense_partials[second_lane + component]
                + s_dense_partials[third_lane + component]
                + s_dense_partials[fourth_lane + component];
            s_encoded[base + component] = fast_tanh(reduced);
        }
    }
    workgroupBarrier();
}
