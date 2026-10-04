// One launch invocation retains its output vector across several feature rows.
// Each iteration still owns one complete vector and both authoritative stores.
override CREDIT_ROWS_PER_INVOCATION: u32 = 2u;

fn phase_encoder_credit(agent_id: u32, launch_vector: u32) {
    let tile = launch_vector / ENCODER_CREDIT_THREADS;
    let lane = launch_vector % ENCODER_CREDIT_THREADS;
    let weight_vector = tile * ENCODER_CREDIT_THREADS * CREDIT_ROWS_PER_INVOCATION + lane;
    // CREDIT_BATCH_GATE
    let learning_rate = brain_config[1].x;
    var scales = vec4<f32>(0.0);
    // CREDIT_BATCH_SCALES
    for (var row = 0u; row < CREDIT_ROWS_PER_INVOCATION; row++) {
        let row_vector = weight_vector + row * ENCODER_CREDIT_THREADS;
        if row_vector >= FEATURE_COUNT * PACKED_ENCODER_OUTPUT_VECTORS { break; }
        let row_feature = row_vector / PACKED_ENCODER_OUTPUT_VECTORS;
        packed_credit_apply_row(agent_id, row_vector, row_feature, credit_enabled, scales);
    }
}
