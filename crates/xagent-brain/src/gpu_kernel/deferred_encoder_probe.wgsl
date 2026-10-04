// Isolated FP32 primitive. Rust supplies dimensions and packed-buffer offsets.
// Each workgroup owns one complete matrix and executes all steps on the GPU.
@group(0) @binding(0) var<storage, read> probe_inputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> probe_weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> probe_outputs: array<f32>;

var<workgroup> probe_partials: array<f32, WORKGROUP_SIZE>;
var<workgroup> probe_maxima: array<vec2<f32>, WORKGROUP_SIZE>;
var<workgroup> probe_overlaps: array<f32, MAX_WINDOW>;
var<workgroup> probe_pending_start: u32;
var<workgroup> probe_pending_count: u32;
var<workgroup> probe_accept: u32;
var<workgroup> probe_fallbacks: u32;
var<workgroup> probe_flushes: u32;
var<workgroup> probe_bound: f32;

fn probe_feature(case_id: u32, step: u32, feature: u32) -> f32 {
    return probe_inputs[case_id * INPUT_STRIDE + FEATURES_OFFSET + step * FEATURE_COUNT + feature];
}

fn probe_scale(case_id: u32, step: u32, dimension: u32) -> f32 {
    return probe_inputs[case_id * INPUT_STRIDE + SCALES_OFFSET + step * OUTPUT_DIMENSION + dimension];
}

fn probe_active(case_id: u32, step: u32, dimension: u32) -> bool {
    return probe_inputs[case_id * INPUT_STRIDE + ACTIVE_OFFSET + step * OUTPUT_DIMENSION + dimension] != 0.0;
}

fn probe_query(case_id: u32, step: u32, feature: u32) -> f32 {
    return probe_inputs[case_id * INPUT_STRIDE + QUERIES_OFFSET + step * FEATURE_COUNT + feature];
}

fn probe_update(weight: f32, case_id: u32, step: u32, index: u32) -> f32 {
    let dimension = index % OUTPUT_DIMENSION;
    var result = weight;
    if (probe_active(case_id, step, dimension)) {
        let scale = probe_scale(case_id, step, dimension);
        let feature = probe_feature(case_id, step, index / OUTPUT_DIMENSION);
        result = result + scale * feature;
        result = clamp(result, -WEIGHT_LIMIT, WEIGHT_LIMIT);
    }
    return result;
}

fn probe_replay(weight: f32, case_id: u32, index: u32, start: u32, count: u32) -> f32 {
    var result = weight;
    for (var pending = 0u; pending < count; pending++) {
        result = probe_update(result, case_id, start + pending, index);
    }
    return result;
}

fn probe_sync() {
    storageBarrier();
    workgroupBarrier();
}

// Positive next-representable inflation covers correctly rounded arithmetic;
// the normal floor additionally admits implementations that flush subnormals.
// The host fixtures are finite. Overflow makes the certificate fail closed.
fn probe_upper(value: f32) -> f32 {
    if (!(value < MAX_FINITE)) {
        return MAX_FINITE;
    }
    return max(MIN_NORMAL, bitcast<f32>(bitcast<u32>(max(value, 0.0)) + 1u));
}

fn probe_reduce_max(tid: u32) {
    workgroupBarrier();
    for (var width = WORKGROUP_SIZE / 2u; width > 0u; width /= 2u) {
        if (tid < width) {
            probe_maxima[tid] = max(probe_maxima[tid], probe_maxima[tid + width]);
        }
        workgroupBarrier();
    }
}

fn probe_finish_bound(tid: u32, magnitude: f32) {
    probe_maxima[tid] = vec2<f32>(magnitude, 0.0);
    probe_reduce_max(tid);
    if (tid == 0u) {
        probe_bound = probe_upper(probe_maxima[0].x);
    }
    workgroupBarrier();
}

fn probe_flush(case_id: u32, tid: u32) {
    let start = probe_pending_start;
    let count = probe_pending_count;
    var magnitude = 0.0;
    for (var index = tid; index < MATRIX_WORDS; index += WORKGROUP_SIZE) {
        let address = case_id * MATRIX_WORDS + index;
        let weight = probe_replay(probe_weights[address], case_id, index, start, count);
        probe_weights[address] = weight;
        magnitude = max(magnitude, abs(weight));
    }
    probe_sync();
    if (tid == 0u) {
        probe_pending_count = 0u;
        probe_flushes += 1u;
    }
    probe_finish_bound(tid, magnitude);
}

fn probe_factor_overlaps(case_id: u32, step: u32, tid: u32) {
    let pending = tid / INNER_LANES;
    let lane = tid % INNER_LANES;
    var partial = 0.0;
    if (pending < probe_pending_count) {
        for (var feature = lane; feature < FEATURE_COUNT; feature += INNER_LANES) {
            partial += probe_feature(case_id, probe_pending_start + pending, feature)
                * probe_query(case_id, step, feature);
        }
    }
    probe_partials[tid] = partial;
    workgroupBarrier();
    if (tid < probe_pending_count) {
        let offset = tid * INNER_LANES;
        probe_overlaps[tid] = probe_partials[offset] + probe_partials[offset + 1u]
            + probe_partials[offset + 2u] + probe_partials[offset + 3u];
    }
    workgroupBarrier();
}

fn probe_dot(case_id: u32, step: u32, tid: u32, deferred: bool) {
    let lane = tid / OUTPUT_TILE;
    let local_output = tid % OUTPUT_TILE;
    for (var tile = 0u; tile < OUTPUT_DIMENSION; tile += OUTPUT_TILE) {
        let dimension = tile + local_output;
        var partial = 0.0;
        for (var feature = lane; feature < FEATURE_COUNT; feature += INNER_LANES) {
            partial += probe_query(case_id, step, feature)
                * probe_weights[case_id * MATRIX_WORDS + feature * OUTPUT_DIMENSION + dimension];
        }
        probe_partials[tid] = partial;
        workgroupBarrier();
        if (lane == 0u) {
            var result = probe_partials[local_output]
                + probe_partials[OUTPUT_TILE + local_output]
                + probe_partials[2u * OUTPUT_TILE + local_output]
                + probe_partials[3u * OUTPUT_TILE + local_output];
            if (deferred) {
                for (var pending = 0u; pending < probe_pending_count; pending++) {
                    let factor = probe_pending_start + pending;
                    if (probe_active(case_id, factor, dimension)) {
                        result += probe_scale(case_id, factor, dimension) * probe_overlaps[pending];
                    }
                }
            }
            let output = (case_id * STEP_COUNT + step) * OUTPUT_STRIDE + SNAPSHOT_WORDS;
            probe_outputs[output + dimension] = result;
        }
        workgroupBarrier();
    }
}

fn probe_capture(case_id: u32, step: u32, tid: u32, deferred: bool) {
    let output = (case_id * STEP_COUNT + step) * OUTPUT_STRIDE;
    if (CAPTURE_SNAPSHOTS) {
        for (var index = tid; index < MATRIX_WORDS; index += WORKGROUP_SIZE) {
            let weight = probe_weights[case_id * MATRIX_WORDS + index];
            probe_outputs[output + index] = weight;
            var materialized = weight;
            if (deferred) {
                materialized = probe_replay(weight, case_id, index, probe_pending_start, probe_pending_count);
            }
            probe_outputs[output + MATRIX_WORDS + index] = materialized;
        }
    }
    if (tid == 0u) {
        let metadata = output + SNAPSHOT_WORDS + OUTPUT_DIMENSION;
        probe_outputs[metadata + META_PENDING_START] = bitcast<f32>(probe_pending_start);
        probe_outputs[metadata + META_PENDING_COUNT] = bitcast<f32>(probe_pending_count);
        probe_outputs[metadata + META_ACCEPTED] = bitcast<f32>(probe_accept);
        probe_outputs[metadata + META_FALLBACKS] = bitcast<f32>(probe_fallbacks);
        probe_outputs[metadata + META_FLUSHES] = bitcast<f32>(probe_flushes);
        probe_outputs[metadata + META_BOUND] = probe_bound;
    }
    probe_sync();
}

fn probe_initialize(case_id: u32, tid: u32) {
    for (var index = tid; index < MATRIX_WORDS; index += WORKGROUP_SIZE) {
        probe_weights[case_id * MATRIX_WORDS + index] = probe_inputs[case_id * INPUT_STRIDE + index];
    }
    if (tid == 0u) {
        probe_pending_start = 0u;
        probe_pending_count = 0u;
        probe_accept = 0u;
        probe_fallbacks = 0u;
        probe_flushes = 0u;
        probe_bound = 0.0;
    }
    probe_sync();
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn sequential_encoder_probe(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let case_id = group.x;
    probe_initialize(case_id, tid);
    for (var step = 0u; step < STEP_COUNT; step++) {
        for (var index = tid; index < MATRIX_WORDS; index += WORKGROUP_SIZE) {
            let address = case_id * MATRIX_WORDS + index;
            probe_weights[address] = probe_update(probe_weights[address], case_id, step, index);
        }
        probe_sync();
        probe_dot(case_id, step, tid, false);
        probe_capture(case_id, step, tid, false);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn deferred_encoder_probe(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let case_id = group.x;
    probe_initialize(case_id, tid);
    var initial_magnitude = 0.0;
    for (var index = tid; index < MATRIX_WORDS; index += WORKGROUP_SIZE) {
        initial_magnitude = max(initial_magnitude, abs(probe_weights[case_id * MATRIX_WORDS + index]));
    }
    probe_finish_bound(tid, initial_magnitude);
    for (var step = 0u; step < STEP_COUNT; step++) {
        if (workgroupUniformLoad(&probe_pending_count) == WINDOW_SIZE) {
            probe_flush(case_id, tid);
        }
        var feature_max = 0.0;
        var scale_max = 0.0;
        for (var feature = tid; feature < FEATURE_COUNT; feature += WORKGROUP_SIZE) {
            feature_max = max(feature_max, abs(probe_feature(case_id, step, feature)));
        }
        if (tid < OUTPUT_DIMENSION && probe_active(case_id, step, tid)) {
            scale_max = abs(probe_scale(case_id, step, tid));
        }
        probe_maxima[tid] = vec2<f32>(feature_max, scale_max);
        probe_reduce_max(tid);
        if (tid == 0u) {
            let maxima = max(probe_maxima[0], vec2<f32>(MIN_NORMAL));
            let increment = probe_upper(maxima.x * maxima.y);
            let proposed = probe_upper(probe_bound + increment);
            probe_accept = select(0u, 1u, proposed <= WEIGHT_LIMIT);
            if (probe_accept != 0u) {
                if (probe_pending_count == 0u) {
                    probe_pending_start = step;
                }
                probe_pending_count += 1u;
                probe_bound = proposed;
            }
        }
        workgroupBarrier();
        if (workgroupUniformLoad(&probe_accept) == 0u) {
            if (workgroupUniformLoad(&probe_pending_count) != 0u) {
                probe_flush(case_id, tid);
            }
            // No uncertified update enters the deferred representation.
            var magnitude = 0.0;
            for (var index = tid; index < MATRIX_WORDS; index += WORKGROUP_SIZE) {
                let address = case_id * MATRIX_WORDS + index;
                let weight = probe_update(probe_weights[address], case_id, step, index);
                probe_weights[address] = weight;
                magnitude = max(magnitude, abs(weight));
            }
            probe_sync();
            if (tid == 0u) {
                probe_pending_start = step + 1u;
                probe_fallbacks += 1u;
            }
            probe_finish_bound(tid, magnitude);
        }
        probe_factor_overlaps(case_id, step, tid);
        probe_dot(case_id, step, tid, true);
        probe_capture(case_id, step, tid, true);
    }
    if (workgroupUniformLoad(&probe_pending_count) != 0u) {
        probe_flush(case_id, tid);
    }
}
