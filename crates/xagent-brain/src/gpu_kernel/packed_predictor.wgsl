// Four adjacent logical feature lanes share one row-major storage vector.
// Each component keeps its width-sixteen sequence of eight ordered terms.
// Four physical invocations own one output; every vector has one writer.
// In the 128-active variant the other 128 invocations still reach both barriers.
fn packed_predictor_rows(agent_id: u32, tid: u32) {
    let brain_base = agent_id * BRAIN_STRIDE;
    let output_in_tile = tid / PACKED_PREDICTOR_GROUPS;
    let vector_group = tid % PACKED_PREDICTOR_GROUPS;
    let first_lane = vector_group * PACKED_PREDICTOR_VECTOR_WIDTH;
    let predictor_learning_rate = bc_f32(CFG_LEARNING_RATE);

    for (var tile = 0u; tile < PREDICTOR_DIMENSION; tile += PACKED_PREDICTOR_OUTPUT_TILE) {
        let dimension = tile + output_in_tile;
        let slot = tid * PACKED_PREDICTOR_VECTOR_WIDTH;
        if (tid < PACKED_PREDICTOR_ACTIVE_THREADS) {
            let previous_prediction = brain_state[brain_base + O_PREV_PREDICTION + dimension];
            let transition_error = previous_prediction - s_encoded[dimension];
            let tanh_derivative = 1.0 - previous_prediction * previous_prediction;
            let matrix_base = (agent_id * PREDICTOR_DIMENSION + dimension) * PACKED_PREDICTOR_INPUT_VECTORS;
            var partial = vec4<f32>(0.0);
            for (var input_first = first_lane; input_first < ENCODED_DIMENSION; input_first += PACKED_PREDICTOR_LANES * PACKED_PREDICTOR_PREFETCH) {
                let input_second = input_first + PACKED_PREDICTOR_LANES;
                let previous_first = vec4<f32>(
                    brain_state[brain_base + O_PREV_ENCODED + input_first],
                    brain_state[brain_base + O_PREV_ENCODED + input_first + 1u],
                    brain_state[brain_base + O_PREV_ENCODED + input_first + 2u],
                    brain_state[brain_base + O_PREV_ENCODED + input_first + 3u],
                );
                let previous_second = vec4<f32>(
                    brain_state[brain_base + O_PREV_ENCODED + input_second],
                    brain_state[brain_base + O_PREV_ENCODED + input_second + 1u],
                    brain_state[brain_base + O_PREV_ENCODED + input_second + 2u],
                    brain_state[brain_base + O_PREV_ENCODED + input_second + 3u],
                );
                let encoded_first = vec4<f32>(s_encoded[input_first], s_encoded[input_first + 1u],
                    s_encoded[input_first + 2u], s_encoded[input_first + 3u]);
                let encoded_second = vec4<f32>(s_encoded[input_second], s_encoded[input_second + 1u],
                    s_encoded[input_second + 2u], s_encoded[input_second + 3u]);
                let first_address = matrix_base + input_first / PACKED_PREDICTOR_VECTOR_WIDTH;
                let second_address = matrix_base + input_second / PACKED_PREDICTOR_VECTOR_WIDTH;
                let old_first = packed_predictor.weights[first_address];
                let old_second = packed_predictor.weights[second_address];

                let gradient_first = clamp((transition_error * tanh_derivative) * previous_first,
                    vec4<f32>(-1.0), vec4<f32>(1.0));
                let gradient_second = clamp((transition_error * tanh_derivative) * previous_second,
                    vec4<f32>(-1.0), vec4<f32>(1.0));
                let updated_first = clamp(old_first - predictor_learning_rate * gradient_first,
                    vec4<f32>(-3.0), vec4<f32>(3.0));
                let updated_second = clamp(old_second - predictor_learning_rate * gradient_second,
                    vec4<f32>(-3.0), vec4<f32>(3.0));
                packed_predictor.weights[first_address] = updated_first;
                packed_predictor.weights[second_address] = updated_second;
                partial += encoded_first * updated_first;
                partial += encoded_second * updated_second;
            }
            s_dense_partials[slot] = partial.x;
            s_dense_partials[slot + 1u] = partial.y;
            s_dense_partials[slot + 2u] = partial.z;
            s_dense_partials[slot + 3u] = partial.w;
        }
        workgroupBarrier();
        if (tid < PACKED_PREDICTOR_ACTIVE_THREADS && vector_group == 0u) {
            let group_zero = vec4<f32>(s_dense_partials[slot], s_dense_partials[slot + 1u],
                s_dense_partials[slot + 2u], s_dense_partials[slot + 3u]);
            let group_one = vec4<f32>(s_dense_partials[slot + 4u], s_dense_partials[slot + 5u],
                s_dense_partials[slot + 6u], s_dense_partials[slot + 7u]);
            let group_two = vec4<f32>(s_dense_partials[slot + 8u], s_dense_partials[slot + 9u],
                s_dense_partials[slot + 10u], s_dense_partials[slot + 11u]);
            let group_three = vec4<f32>(s_dense_partials[slot + 12u], s_dense_partials[slot + 13u],
                s_dense_partials[slot + 14u], s_dense_partials[slot + 15u]);
            // Exactly the scalar reduction's strides eight, four, two, one.
            let first_pair = group_zero + group_two;
            let second_pair = group_one + group_three;
            let joined = first_pair + second_pair;
            s_prediction[dimension] = (joined.x + joined.z) + (joined.y + joined.w);
        }
        workgroupBarrier();
    }
}
