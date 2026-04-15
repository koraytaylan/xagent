// ── Cooperative brain tick: multi-workgroup, 256 threads per agent ──────────
// dispatch(agent_count, 1, 1) — one workgroup per agent.
// 256 threads cooperate on the 7 brain passes via shared memory.

const BRAIN_WORKGROUP_SIZE: u32 = 256u;

// ── Shared memory (~2.5 KB at default 8×6; scales with FEATURE_COUNT) ──────

var<workgroup> s_features: array<f32, FEATURE_COUNT>;
var<workgroup> s_encoded: array<f32, ENCODED_DIMENSION>;
var<workgroup> s_habituated: array<f32, ENCODED_DIMENSION>;
var<workgroup> s_homeo: array<f32, 6>;
var<workgroup> s_similarities: array<f32, MEMORY_CAP>;
var<workgroup> shared_sort_indices: array<u32, MEMORY_CAP>;
var<workgroup> s_recall: array<f32, 17>;
var<workgroup> s_prediction: array<f32, PREDICTOR_DIMENSION>;
var<workgroup> s_credit: array<f32, ENCODED_DIMENSION>;
var<workgroup> s_pred_error: f32;

// ── Helpers ────────────────────────────────────────────────────────────────

fn rand_f32_brain(seed: u32) -> f32 {
    return hash_to_float(pcg_hash(seed));
}

// Cosine similarity using shared encoded state (pre-habituation)
// for memory operations — avoids attenuation silencing recall.
fn cosine_sim_pat_s(agent_id: u32, idx: u32) -> f32 {
    let pattern_base = agent_id * PATTERN_STRIDE;
    var dot_val: f32 = 0.0;
    var e_norm_sq: f32 = 0.0;
    for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
        let e = s_encoded[d];
        let p = pattern_buffer[pattern_base + d * MEMORY_CAP + idx];
        dot_val += e * p;
        e_norm_sq += e * e;
    }
    let e_norm = sqrt(e_norm_sq);
    let p_norm = pattern_buffer[pattern_base + O_PAT_NORMS + idx];
    if (e_norm < 1e-8 || p_norm < 1e-8) { return 0.0; }
    return clamp(dot_val / (e_norm * p_norm), -1.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 1: Feature extract — all threads cooperatively load vision data,
// thread 0 handles the 25 non-visual features (velocity, facing, touch).
// Vision copy scales with VISION_COLOR_COUNT + VISION_DEPTH_COUNT;
// parallelizing across BRAIN_WORKGROUP_SIZE threads keeps it fast at any
// resolution.
// ═══════════════════════════════════════════════════════════════════════════

fn coop_feature_extract(agent_id: u32, tid: u32) {
    let s_base = agent_id * SENSORY_STRIDE;

    // All threads cooperatively copy vision color + depth into s_features.
    // sensory_buffer layout: [color(VISION_COLOR_COUNT) | depth(VISION_DEPTH_COUNT) | non-visual]
    // s_features layout:  [color(VISION_COLOR_COUNT) | depth(VISION_DEPTH_COUNT) | non-visual(25)]
    // The vision portion maps 1:1 between the two buffers.
    let vision_count = VISION_COLOR_COUNT + VISION_DEPTH_COUNT;
    for (var i = tid; i < vision_count; i += BRAIN_WORKGROUP_SIZE) {
        s_features[i] = sensory_buffer[s_base + i];
    }

    // Non-visual features (25 values) — thread 0 only.
    // Velocity magnitude requires a sqrt, so this can't be a bulk copy.
    if (tid == 0u) {
        var fi = vision_count;
        let vel_offset = vision_count;
        let vx = sensory_buffer[s_base + vel_offset];
        let vy = sensory_buffer[s_base + vel_offset + 1u];
        let vz = sensory_buffer[s_base + vel_offset + 2u];
        s_features[fi] = sqrt(vx * vx + vy * vy + vz * vz); fi = fi + 1u;
        let fac_offset = vel_offset + 3u;
        s_features[fi] = sensory_buffer[s_base + fac_offset]; fi = fi + 1u;
        s_features[fi] = sensory_buffer[s_base + fac_offset + 1u]; fi = fi + 1u;
        s_features[fi] = sensory_buffer[s_base + fac_offset + 2u]; fi = fi + 1u;
        let ang_offset = fac_offset + 3u;
        s_features[fi] = sensory_buffer[s_base + ang_offset]; fi = fi + 1u;
        s_features[fi] = sensory_buffer[s_base + ang_offset + 1u]; fi = fi + 1u;
        s_features[fi] = sensory_buffer[s_base + ang_offset + 2u]; fi = fi + 1u;
        s_features[fi] = sensory_buffer[s_base + ang_offset + 3u]; fi = fi + 1u;
        s_features[fi] = sensory_buffer[s_base + ang_offset + 4u]; fi = fi + 1u;
        let touch_offset = ang_offset + 5u;
        for (var t: u32 = 0u; t < 4u; t = t + 1u) {
            let to = touch_offset + t * 4u;
            s_features[fi] = sensory_buffer[s_base + to]; fi = fi + 1u;
            s_features[fi] = sensory_buffer[s_base + to + 1u]; fi = fi + 1u;
            s_features[fi] = sensory_buffer[s_base + to + 2u]; fi = fi + 1u;
            s_features[fi] = sensory_buffer[s_base + to + 3u]; fi = fi + 1u;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 2: Encode (threads 0..31 — FEATURE_COUNT MADs each, coalesced access)
// ═══════════════════════════════════════════════════════════════════════════

fn coop_encode(agent_id: u32, tid: u32) {
    if (tid < ENCODED_DIMENSION) {
        let brain_base = agent_id * BRAIN_STRIDE;
        var sum: f32 = brain_state[brain_base + O_ENC_BIASES + tid];
        for (var f: u32 = 0u; f < FEATURE_COUNT; f = f + 1u) {
            sum += s_features[f] * brain_state[brain_base + O_ENC_WEIGHTS + f * ENCODED_DIMENSION + tid];
        }
        s_encoded[tid] = fast_tanh(sum);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 3: Habituate + Homeostasis (threads 0..31 + thread 0)
// ═══════════════════════════════════════════════════════════════════════════

fn coop_habituate_homeo(agent_id: u32, tid: u32) {
    let brain_base = agent_id * BRAIN_STRIDE;

    if (tid < ENCODED_DIMENSION) {
        let enc = s_encoded[tid];
        let prev = brain_state[brain_base + O_PREV_ENCODED + tid];
        let delta = abs(enc - prev);
        let sensitivity = brain_state[brain_base + O_HAB_SENSITIVITY];
        let old_ema = brain_state[brain_base + O_HAB_EMA + tid];
        let new_ema = (1.0 - HAB_EMA_ALPHA) * old_ema + HAB_EMA_ALPHA * delta;
        brain_state[brain_base + O_HAB_EMA + tid] = new_ema;
        let atten = clamp(new_ema * sensitivity, ATTEN_FLOOR, 1.0);
        brain_state[brain_base + O_HAB_ATTEN + tid] = atten;
        s_habituated[tid] = enc * atten;
        brain_state[brain_base + O_PREV_ENCODED + tid] = enc;
    }

    if (tid == 0u) {
        let phys_base_homeo = agent_id * PHYS_STRIDE;
        let max_energy = physics_state[phys_base_homeo + P_MAX_ENERGY];
        let max_integrity = physics_state[phys_base_homeo + P_MAX_INTEGRITY];
        let energy = physics_state[phys_base_homeo + P_ENERGY] / max(max_energy, 1e-6);
        let integrity = physics_state[phys_base_homeo + P_INTEGRITY] / max(max_integrity, 1e-6);
        let prev_energy = brain_state[brain_base + O_HOMEO + 4u];
        let prev_integrity = brain_state[brain_base + O_HOMEO + 5u];
        let energy_delta = clamp(energy - prev_energy, -MAX_HOMEOSTATIC_DELTA, MAX_HOMEOSTATIC_DELTA);
        let integrity_delta = clamp(integrity - prev_integrity, -MAX_HOMEOSTATIC_DELTA, MAX_HOMEOSTATIC_DELTA);
        let raw_gradient = energy_delta * ENERGY_WEIGHT + integrity_delta * INTEGRITY_WEIGHT;
        let gradient_fast = brain_state[brain_base + O_HOMEO + 0u] * (1.0 - GRADIENT_FAST_BLEND) + raw_gradient * GRADIENT_FAST_BLEND;
        let gradient_medium = brain_state[brain_base + O_HOMEO + 1u] * (1.0 - GRADIENT_MEDIUM_BLEND) + raw_gradient * GRADIENT_MEDIUM_BLEND;
        let gradient_slow = brain_state[brain_base + O_HOMEO + 2u] * (1.0 - GRADIENT_SLOW_BLEND) + raw_gradient * GRADIENT_SLOW_BLEND;
        brain_state[brain_base + O_HOMEO + 0u] = gradient_fast;
        brain_state[brain_base + O_HOMEO + 1u] = gradient_medium;
        brain_state[brain_base + O_HOMEO + 2u] = gradient_slow;
        let distress_exp = brain_config[CFG_DISTRESS_EXP / 4u][CFG_DISTRESS_EXP % 4u];
        let e_clamped = clamp(energy, 0.01, 1.0);
        let i_clamped = clamp(integrity, 0.01, 1.0);
        let e_distress = min(pow(1.0 - e_clamped, distress_exp) * DISTRESS_SCALE, MAX_DISTRESS);
        let i_distress = min(pow(1.0 - i_clamped, distress_exp) * DISTRESS_SCALE, MAX_DISTRESS);
        let urgency = (e_distress + i_distress) * 0.5;
        brain_state[brain_base + O_HOMEO + 3u] = urgency;
        brain_state[brain_base + O_HOMEO + 4u] = energy;
        brain_state[brain_base + O_HOMEO + 5u] = integrity;
        let blended_gradient = gradient_fast * GRADIENT_WEIGHT_FAST + gradient_medium * GRADIENT_WEIGHT_MEDIUM + gradient_slow * GRADIENT_WEIGHT_SLOW;
        let gradient = blended_gradient * (1.0 + urgency);
        let raw_gradient_amplified = raw_gradient * (1.0 + urgency);
        s_homeo[0u] = gradient;
        s_homeo[1u] = raw_gradient_amplified;
        s_homeo[2u] = urgency;
        s_homeo[3u] = gradient_fast;
        s_homeo[4u] = gradient_medium;
        s_homeo[5u] = gradient_slow;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 4: Recall score (threads 0..127 — one pattern each)
// ═══════════════════════════════════════════════════════════════════════════

fn coop_recall_score(agent_id: u32, tid: u32) {
    if (tid < MEMORY_CAP) {
        let pattern_base = agent_id * PATTERN_STRIDE;

        // Each thread computes query norm independently (32 shared reads — fast)
        // Uses encoded (pre-habituation) state for memory queries.
        var q_norm_sq: f32 = 0.0;
        for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
            let v = s_encoded[d];
            q_norm_sq += v * v;
        }
        let q_norm = sqrt(q_norm_sq);

        let is_active = pattern_buffer[pattern_base + O_PAT_ACTIVE + tid];
        if (is_active < 0.5) {
            s_similarities[tid] = -2.0;
        } else {
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
                dot += s_encoded[d] * pattern_buffer[pattern_base + d * MEMORY_CAP + tid];
            }
            let p_norm = pattern_buffer[pattern_base + O_PAT_NORMS + tid];
            if (q_norm < 1e-8 || p_norm < 1e-8) {
                s_similarities[tid] = 0.0;
            } else {
                s_similarities[tid] = clamp(dot / (q_norm * p_norm), -1.0, 1.0);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 5: Top-K selection (64 threads — parallel bitonic sort of 128 shared values)
// ═══════════════════════════════════════════════════════════════════════════

fn coop_recall_topk(agent_id: u32, tid: u32 /* SUBGROUP_TOPK_PARAMS */) {
    let pattern_base = agent_id * PATTERN_STRIDE;
    let brain_base = agent_id * BRAIN_STRIDE;
    let tick = brain_state[brain_base + O_TICK_COUNT];

    // Initialize sort index: threads 0..127
    if (tid < MEMORY_CAP) {
        shared_sort_indices[tid] = tid;
    }
    workgroupBarrier();

    // Bitonic sort: 7 stages, 28 total barrier passes
    // Sort s_similarities descending (largest at index 0)
    // BEGIN_BITONIC_SORT
    for (var stage: u32 = 0u; stage < 7u; stage = stage + 1u) {
        for (var step: u32 = 0u; step <= stage; step = step + 1u) {
            if (tid < 64u) {
                let block_size = 1u << (stage + 1u - step);
                let half = block_size >> 1u;
                let group = tid / half;
                let local_id = tid % half;
                let i = group * block_size + local_id;
                let j = i + half;
                let descending = ((i >> (stage + 1u)) & 1u) == 0u;

                let val_i = s_similarities[i];
                let val_j = s_similarities[j];
                let idx_i = shared_sort_indices[i];
                let idx_j = shared_sort_indices[j];

                let should_swap = (descending && val_i < val_j) || (!descending && val_i > val_j);
                if (should_swap) {
                    s_similarities[i] = val_j;
                    s_similarities[j] = val_i;
                    shared_sort_indices[i] = idx_j;
                    shared_sort_indices[j] = idx_i;
                }
            }
            workgroupBarrier();
        }
    }
    // END_BITONIC_SORT

    // Thread 0: extract top-K from sorted array (index 0 = largest)
    if (tid == 0u) {
        var count: u32 = 0u;
        for (var k: u32 = 0u; k < RECALL_K; k = k + 1u) {
            if (s_similarities[k] <= -1.5) { break; }
            let idx = shared_sort_indices[k];
            s_recall[k] = f32(idx);
            count = count + 1u;
            pattern_buffer[pattern_base + O_PAT_META + idx * 3u + 1u] = tick;
            pattern_buffer[pattern_base + O_PAT_META + idx * 3u + 2u] += 1.0;
        }
        for (var k: u32 = count; k < RECALL_K; k = k + 1u) { s_recall[k] = 0.0; }
        s_recall[RECALL_K] = f32(count);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 6: Predict and act
// Predictor matmul: threads 0..31; everything else: thread 0
// ═══════════════════════════════════════════════════════════════════════════

fn coop_predict_and_act(agent_id: u32, tid: u32) {
    let brain_base = agent_id * BRAIN_STRIDE;
    let pattern_base = agent_id * PATTERN_STRIDE;
    let history_base = agent_id * HISTORY_STRIDE;
    let decision_base = agent_id * DECISION_STRIDE;
    let tick_count = brain_state[brain_base + O_TICK_COUNT];
    let recall_count = u32(s_recall[RECALL_K]);

    // ── Predictor matmul: threads 0..PREDICTOR_DIMENSION ────────────────
    if (tid < PREDICTOR_DIMENSION) {
        var s: f32 = 0.0;
        for (var j: u32 = 0u; j < ENCODED_DIMENSION; j = j + 1u) {
            s += s_habituated[j] * brain_state[brain_base + O_PREDICTOR_WEIGHTS + tid * ENCODED_DIMENSION + j];
        }
        s_prediction[tid] = s;
    }
    workgroupBarrier();

    // ── Thread 0: rest of predict + act ────────────────────────────────
    if (tid == 0u) {
        let gradient = s_homeo[0u];
        let urgency = s_homeo[2u];

        // Prediction error (computed in predictor space)
        var err_sum: f32 = 0.0;
        for (var d: u32 = 0u; d < PREDICTOR_DIMENSION; d = d + 1u) {
            let previous_prediction = brain_state[brain_base + O_PREV_PREDICTION + d];
            let e = previous_prediction - s_habituated[d];
            err_sum += e * e;
        }
        let prediction_error = sqrt(err_sum / f32(PREDICTOR_DIMENSION));

        // Error ring
        let err_cursor = u32(brain_state[brain_base + O_PREDICTION_ERROR_CURSOR]);
        brain_state[brain_base + O_PREDICTION_ERROR_RING + err_cursor] = prediction_error;
        brain_state[brain_base + O_PREDICTION_ERROR_CURSOR] = f32((err_cursor + 1u) % ERROR_HISTORY_LEN);
        let err_count = brain_state[brain_base + O_PREDICTION_ERROR_COUNT];
        if (err_count < f32(ERROR_HISTORY_LEN)) {
            brain_state[brain_base + O_PREDICTION_ERROR_COUNT] = err_count + 1.0;
        }

        // Recalled context blend
        if (recall_count > 0u) {
            let context_weight = brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT];
            var total_sim: f32 = 0.0;
            for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                let idx = u32(s_recall[k]);
                total_sim += max(cosine_sim_pat_s(agent_id, idx), 0.0);
            }
            if (total_sim > 1e-8) {
                for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                    let idx = u32(s_recall[k]);
                    let sim = cosine_sim_pat_s(agent_id, idx);
                    let w = context_weight * max(sim, 0.0) / total_sim;
                    for (var d: u32 = 0u; d < PREDICTOR_DIMENSION; d = d + 1u) {
                        s_prediction[d] += pattern_buffer[pattern_base + d * MEMORY_CAP + idx] * w;
                    }
                }
            }
        }

        for (var d: u32 = 0u; d < PREDICTOR_DIMENSION; d = d + 1u) {
            s_prediction[d] = fast_tanh(s_prediction[d]);
        }
        // Pass prediction_error to the post-credit block via shared memory.
        // s_pred_error is later overwritten for pass 7.
        s_pred_error = prediction_error;
    }
    workgroupBarrier();

    // ── Credit assignment: cooperative across threads 0..31 ────────────
    // Thread 0 computes per-entry credit scalars → s_similarities[0..N].
    // Threads 0..31 then apply all credits to their dimension in
    // parallel, eliminating the serial ENCODED_DIMENSION-loop bottleneck that was
    // blocking 255 threads at the barrier while thread 0 looped over
    // 64 entries × 32 dimensions.
    {
        let gradient = s_homeo[0u];
        let urgency = s_homeo[2u];
        let hist_len = u32(history_buffer[history_base + O_HIST_LEN]);

        // Phase 1: thread 0 computes credit scalar for each history entry
        if (tid == 0u) {
            for (var i: u32 = 0u; i < hist_len; i = i + 1u) {
                let h_off = i * 5u;
                let recorded_forward = history_buffer[history_base + O_MOTOR_RING + h_off];
                let recorded_turn = history_buffer[history_base + O_MOTOR_RING + h_off + 1u];
                let rec_tick = history_buffer[history_base + O_MOTOR_RING + h_off + 2u];
                let rec_grad = history_buffer[history_base + O_MOTOR_RING + h_off + 3u];
                let age = tick_count - rec_tick;
                let temporal = exp(-age * CREDIT_DECAY);
                var credit: f32 = 0.0;
                if (temporal >= 0.01) {
                    let improvement = gradient - rec_grad;
                    var credit_input = improvement;
                    if (abs(improvement) < DEADZONE) {
                        credit_input = gradient * urgency * TONIC_CREDIT_SCALE;
                    }
                    if (abs(credit_input) >= CREDIT_EPSILON) {
                        var effective: f32;
                        if (credit_input < 0.0) { effective = credit_input * PAIN_AMP; }
                        else { effective = credit_input; }
                        credit = effective * temporal;
                        brain_state[brain_base + O_ACT_BIASES] += ACTION_WEIGHT_LEARNING_RATE * credit * recorded_forward * 0.1;
                        brain_state[brain_base + O_ACT_BIASES + 1u] += ACTION_WEIGHT_LEARNING_RATE * credit * recorded_turn * 0.1;
                    }
                }
                s_similarities[i] = credit;
                // Cache motor values in shared memory so phase 2 threads
                // read from workgroup memory instead of storage.
                shared_sort_indices[i * 2u] = bitcast<u32>(recorded_forward);
                shared_sort_indices[i * 2u + 1u] = bitcast<u32>(recorded_turn);
            }
        }
        workgroupBarrier();

        // Phase 2: threads 0..31 apply all credits to their dimension
        if (tid < ENCODED_DIMENSION) {
            s_credit[tid] = 0.0;
            for (var i: u32 = 0u; i < hist_len; i = i + 1u) {
                let credit = s_similarities[i];
                if (abs(credit) > 0.0) {
                    let recorded_forward = bitcast<f32>(shared_sort_indices[i * 2u]);
                    let recorded_turn = bitcast<f32>(shared_sort_indices[i * 2u + 1u]);
                    let feat = history_buffer[history_base + O_STATE_RING + i * ENCODED_DIMENSION + tid];
                    let forward_weight_update = ACTION_WEIGHT_LEARNING_RATE * credit * recorded_forward * feat;
                    let turn_weight_update = ACTION_WEIGHT_LEARNING_RATE * credit * recorded_turn * feat;
                    brain_state[brain_base + O_ACTION_FORWARD_WEIGHTS + tid] += forward_weight_update;
                    brain_state[brain_base + O_ACTION_TURN_WEIGHTS + tid] += turn_weight_update;
                    s_credit[tid] += credit * recorded_forward * feat + credit * recorded_turn * feat;
                }
            }
        }
    }
    storageBarrier(); workgroupBarrier();

    // ── Thread 0: weight normalization, policy, exploration, motor ─────
    if (tid == 0u) {
        let gradient = s_homeo[0u];
        let urgency = s_homeo[2u];
        let prediction_error = s_pred_error;

        // Weight decay + normalization
        for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
            brain_state[brain_base + O_ACTION_FORWARD_WEIGHTS + d] *= (1.0 - ACTION_WEIGHT_DECAY);
            brain_state[brain_base + O_ACTION_TURN_WEIGHTS + d] *= (1.0 - ACTION_WEIGHT_DECAY);
        }
        brain_state[brain_base + O_ACT_BIASES] *= (1.0 - ACTION_WEIGHT_DECAY);
        brain_state[brain_base + O_ACT_BIASES + 1u] *= (1.0 - ACTION_WEIGHT_DECAY);
        brain_state[brain_base + O_ACT_BIASES] = clamp(brain_state[brain_base + O_ACT_BIASES], -MAX_WEIGHT_NORM, MAX_WEIGHT_NORM);
        brain_state[brain_base + O_ACT_BIASES + 1u] = clamp(brain_state[brain_base + O_ACT_BIASES + 1u], -MAX_WEIGHT_NORM, MAX_WEIGHT_NORM);
        var fwd_norm_sq: f32 = 0.0;
        var trn_norm_sq: f32 = 0.0;
        for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
            let fw = brain_state[brain_base + O_ACTION_FORWARD_WEIGHTS + d];
            let tw = brain_state[brain_base + O_ACTION_TURN_WEIGHTS + d];
            fwd_norm_sq += fw * fw;
            trn_norm_sq += tw * tw;
        }
        let fwd_norm = sqrt(fwd_norm_sq);
        if (fwd_norm > MAX_WEIGHT_NORM) {
            let scale = MAX_WEIGHT_NORM / fwd_norm;
            for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
                brain_state[brain_base + O_ACTION_FORWARD_WEIGHTS + d] *= scale;
            }
        }
        let trn_norm = sqrt(trn_norm_sq);
        if (trn_norm > MAX_WEIGHT_NORM) {
            let scale = MAX_WEIGHT_NORM / trn_norm;
            for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
                brain_state[brain_base + O_ACTION_TURN_WEIGHTS + d] *= scale;
            }
        }

        // Policy evaluation
        var forward: f32 = brain_state[brain_base + O_ACT_BIASES];
        var turn: f32 = brain_state[brain_base + O_ACT_BIASES + 1u];
        for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
            forward += brain_state[brain_base + O_ACTION_FORWARD_WEIGHTS + d] * s_encoded[d];
            turn += brain_state[brain_base + O_ACTION_TURN_WEIGHTS + d] * s_encoded[d];
        }

        // Memory blend: recalled experiences influence motor output via valence.
        // Positive valence (food memory) + similar state → reproduce approach action.
        // Negative valence (danger memory) + similar state → negate approach → escape.
        if (recall_count > 0u) {
            var mem_forward: f32 = 0.0;
            var mem_turn: f32 = 0.0;
            var total_weight: f32 = 0.0;
            for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                let idx = u32(s_recall[k]);
                let sim = cosine_sim_pat_s(agent_id, idx);
                let motor_base = pattern_base + O_PAT_MOTOR + idx * 3u;
                let valence = pattern_buffer[motor_base + 2u];
                let weight = sim * valence;
                mem_forward += weight * pattern_buffer[motor_base];
                mem_turn += weight * pattern_buffer[motor_base + 1u];
                total_weight += abs(weight);
            }
            if (total_weight > CREDIT_EPSILON) {
                mem_forward /= total_weight;
                mem_turn /= total_weight;
                let strength = clamp(total_weight / max(f32(recall_count), 1.0), 0.0, 1.0);
                let mix = strength * MEMORY_BLEND_STRENGTH;
                forward = forward * (1.0 - mix) + mem_forward * mix;
                turn = turn * (1.0 - mix) + mem_turn * mix;
            }
        }

        // Exploration
        let max_curiosity = brain_state[brain_base + O_HAB_MAX_CURIOSITY];
        var atten_sum: f32 = 0.0;
        for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
            atten_sum += brain_state[brain_base + O_HAB_ATTEN + d];
        }
        let mean_atten = atten_sum / f32(ENCODED_DIMENSION);
        let curiosity = (1.0 - mean_atten) * max_curiosity;
        let novelty_bonus = min(prediction_error * 2.0, 0.4);
        let urgency_penalty = min(urgency * 0.4, 0.5);
        let raw_signal = abs(forward) + abs(turn);
        let policy_confidence = clamp(raw_signal / 2.0, 0.0, 1.0);

        let exploration_rate = clamp(
            0.5 - policy_confidence * 0.25 + novelty_bonus + curiosity - urgency_penalty,
            0.10, 0.85
        );
        brain_state[brain_base + O_EXPLORATION_RATE] = exploration_rate;

        forward = fast_tanh(forward);
        turn = fast_tanh(turn);

        // Position-based staleness: record current XZ position and
        // accumulated forward output, then compare displacement against
        // expected travel to detect agents that aren't making progress.
        let phys_base_fat = agent_id * PHYS_STRIDE;
        let cur_x = physics_state[phys_base_fat + P_POS_X];
        let cur_z = physics_state[phys_base_fat + P_POS_Z];
        let pos_cursor = u32(brain_state[brain_base + O_POS_RING_CURSOR]);
        brain_state[brain_base + O_POS_RING_X + pos_cursor] = cur_x;
        brain_state[brain_base + O_POS_RING_Z + pos_cursor] = cur_z;
        let pos_len_val = brain_state[brain_base + O_POS_RING_LEN];
        let new_pos_len = min(pos_len_val + 1.0, f32(POS_RING_LEN));
        brain_state[brain_base + O_POS_RING_LEN] = new_pos_len;
        brain_state[brain_base + O_POS_RING_CURSOR] = f32((pos_cursor + 1u) % POS_RING_LEN);

        // Accumulate forward motor output (pre-noise) for expected displacement.
        // This is an approximate running total: when the position ring is full,
        // we do not subtract the overwritten slot's exact forward contribution.
        let old_accum = brain_state[brain_base + O_ACCUM_FWD];
        var new_accum = old_accum + max(forward, 0.0);
        if (new_pos_len >= f32(POS_RING_LEN)) {
            // We do not track per-slot forward values, so approximate a bounded
            // window by decaying the accumulator proportionally each overwrite.
            new_accum *= (f32(POS_RING_LEN) - 1.0) / f32(POS_RING_LEN);
        }
        brain_state[brain_base + O_ACCUM_FWD] = new_accum;

        // Compute staleness: compare actual displacement to expected
        let p_len = u32(new_pos_len);
        let floor_val = brain_state[brain_base + O_FATIGUE_FLOOR];
        var fatigue_factor: f32 = 1.0;
        if (p_len >= 4u) {
            // Oldest valid entry in the ring. When the ring is not yet full,
            // (pos_cursor + 1) points at the next write slot rather than the
            // oldest sample, so compute from the current valid length instead.
            let cursor_new = (pos_cursor + 1u) % POS_RING_LEN;
            let oldest_idx = (cursor_new + POS_RING_LEN - p_len) % POS_RING_LEN;
            let old_x = brain_state[brain_base + O_POS_RING_X + oldest_idx];
            let old_z = brain_state[brain_base + O_POS_RING_Z + oldest_idx];
            let dx = cur_x - old_x;
            let dz = cur_z - old_z;
            let displacement = sqrt(dx * dx + dz * dz);

            // Expected displacement: accumulated forward * move_speed * DT * stride
            // Each brain tick, the agent moves forward * move_speed * DT * stride units
            let move_speed = brain_state[brain_base + O_MOVEMENT_SPEED];
            let expected = new_accum * move_speed * wc_f32(WC_DT) * f32(wc_u32(WC_BRAIN_TICK_STRIDE));
            // Only penalize when the agent is actually trying to move;
            // idle agents (expected ≈ 0) keep fatigue_factor = 1.0.
            let expected_epsilon = 0.001;
            if (expected > expected_epsilon) {
                let ratio = displacement / expected;
                let staleness = 1.0 - clamp(ratio, 0.0, 1.0);
                let max_penalty = 1.0 - floor_val;
                fatigue_factor = 1.0 - staleness * max_penalty;
                fatigue_factor = clamp(fatigue_factor, floor_val, 1.0);
            }
        }
        brain_state[brain_base + O_FATIGUE_FACTOR] = fatigue_factor;

        // Exploration noise — constant amplitude, independent of habituation.
        // Scaling noise by mean_atten (as done previously) created a death
        // spiral: agents barely moved → input static → atten→0.1 → noise 10×
        // smaller → agents moved even less.  Exploration must stay vigorous
        // regardless of habituation state to drive REINFORCE-style learning.
        let tick_u = u32(tick_count);
        let exploration_seed = pcg_hash(agent_id ^ (tick_u * 747796405u));
        let noise_forward = (hash_to_float(exploration_seed) * 2.0 - 1.0) * 0.5;
        let noise_turn = (hash_to_float(pcg_hash(exploration_seed)) * 2.0 - 1.0) * 0.5;
        forward = clamp(forward + noise_forward * exploration_rate, -1.0, 1.0);
        turn = clamp(turn + noise_turn * exploration_rate, -1.0, 1.0);

        forward *= fatigue_factor;
        turn *= fatigue_factor;

        // Klinotaxis: use fast-vs-medium gradient deviation.
        // Fast responds in ~2 ticks and recovers in ~5. Medium responds in ~25.
        //   Entry (ticks 0-2): fast spikes, medium hasn't moved → amplify turns → change direction
        //   Sustained (ticks 4+): fast recovered, medium still shifted → suppress turns → go straight → escape
        // This produces the biological escape sequence: brief reorientation then straight-line flight.
        let gradient_deviation = s_homeo[3u] - s_homeo[4u];
        let klinotaxis_factor = clamp(1.0 - gradient_deviation * KLINOTAXIS_SENSITIVITY, 0.3, 3.0);
        turn *= klinotaxis_factor;

        // History records the exploration noise, not the full motor.
        // Credit × noise is the proper REINFORCE gradient: noise is zero-mean,
        // so only noise directions that correlate with outcomes get reinforced.
        // Using the full motor (policy + noise) creates a feedback loop where
        // any turn bias gets reinforced by every positive credit event.
        let exploration_forward = noise_forward * exploration_rate;
        let exploration_turn = noise_turn * exploration_rate;
        let hist_cursor = u32(history_buffer[history_base + O_HIST_CURSOR]);
        let hist_off = hist_cursor * 5u;
        history_buffer[history_base + O_MOTOR_RING + hist_off] = exploration_forward;
        history_buffer[history_base + O_MOTOR_RING + hist_off + 1u] = exploration_turn;
        history_buffer[history_base + O_MOTOR_RING + hist_off + 2u] = tick_count;
        history_buffer[history_base + O_MOTOR_RING + hist_off + 3u] = gradient;
        history_buffer[history_base + O_MOTOR_RING + hist_off + 4u] = 0.0;
        for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
            history_buffer[history_base + O_STATE_RING + hist_cursor * ENCODED_DIMENSION + d] = s_encoded[d];
        }
        history_buffer[history_base + O_HIST_CURSOR] = f32((hist_cursor + 1u) % ACTION_HISTORY_LEN);
        let hist_len_val = history_buffer[history_base + O_HIST_LEN];
        history_buffer[history_base + O_HIST_LEN] = min(hist_len_val + 1.0, f32(ACTION_HISTORY_LEN));

        // Save prediction + tick + decision buffer
        for (var d: u32 = 0u; d < PREDICTOR_DIMENSION; d = d + 1u) {
            brain_state[brain_base + O_PREV_PREDICTION + d] = s_prediction[d];
        }
        brain_state[brain_base + O_TICK_COUNT] = tick_count + 1.0;

        // Store prediction_error for pass 7 (shared scalar)
        var error_squared_sum: f32 = 0.0;
        for (var d: u32 = 0u; d < PREDICTOR_DIMENSION; d = d + 1u) {
            let e = s_prediction[d] - s_habituated[d];
            error_squared_sum += e * e;
        }
        s_pred_error = clamp(sqrt(error_squared_sum), 0.0, 1.0);

        for (var d: u32 = 0u; d < PREDICTOR_DIMENSION; d = d + 1u) {
            decision_buffer[decision_base + DECISION_PREDICTION + d] = s_prediction[d];
        }
        for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
            decision_buffer[decision_base + DECISION_CREDIT + d] = s_credit[d];
        }
        decision_buffer[decision_base + DECISION_MOTOR] = forward;
        decision_buffer[decision_base + DECISION_MOTOR + 1u] = turn;
        decision_buffer[decision_base + DECISION_MOTOR + 2u] = 0.0;
        decision_buffer[decision_base + DECISION_MOTOR + 3u] = 0.0;

        // Write telemetry to physics buffer for CPU readback
        let phys_base = agent_id * PHYS_STRIDE;
        physics_state[phys_base + P_PREDICTION_ERROR] = s_pred_error;
        physics_state[phys_base + P_EXPLORATION_RATE_OUT] = exploration_rate;
        physics_state[phys_base + P_FATIGUE_FACTOR_OUT] = fatigue_factor;
        physics_state[phys_base + P_MOTOR_FWD_OUT] = forward;
        physics_state[phys_base + P_MOTOR_TURN_OUT] = turn;
        physics_state[phys_base + P_GRADIENT_OUT] = gradient;
        physics_state[phys_base + P_URGENCY_OUT] = urgency;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 7: Learn and store
// Predictor: threads 0..31; Encoder: threads 0..31;
// Memory reinforcement: threads 0..127; Decay: threads 0..127
// ═══════════════════════════════════════════════════════════════════════════

fn coop_learn_and_store(agent_id: u32, tid: u32) {
    let brain_base = agent_id * BRAIN_STRIDE;
    let pattern_base = agent_id * PATTERN_STRIDE;
    let decision_base = agent_id * DECISION_STRIDE;

    let learning_rate = brain_config[1].x;
    let decay_rate = brain_config[1].y;
    let tick = brain_state[brain_base + O_TICK_COUNT];
    let raw_gradient = s_homeo[1u];

    // ── 7a. Predictor learning: threads 0..PREDICTOR_DIMENSION ──────────
    if (tid < PREDICTOR_DIMENSION) {
        let predicted_dimension = decision_buffer[decision_base + DECISION_PREDICTION + tid];
        let error_dimension = predicted_dimension - s_habituated[tid];
        let tanh_deriv = 1.0 - predicted_dimension * predicted_dimension;
        for (var j: u32 = 0u; j < ENCODED_DIMENSION; j = j + 1u) {
            let grad = clamp(error_dimension * tanh_deriv * s_habituated[j], -1.0, 1.0);
            var w = brain_state[brain_base + O_PREDICTOR_WEIGHTS + tid * ENCODED_DIMENSION + j] - learning_rate * grad;
            w = clamp(w, -3.0, 3.0);
            brain_state[brain_base + O_PREDICTOR_WEIGHTS + tid * ENCODED_DIMENSION + j] = w;
        }
    }

    // Thread 0: context weight adaptation
    if (tid == 0u) {
        var error_squared_sum: f32 = 0.0;
        for (var d: u32 = 0u; d < PREDICTOR_DIMENSION; d = d + 1u) {
            let e = decision_buffer[decision_base + DECISION_PREDICTION + d] - s_habituated[d];
            error_squared_sum += e * e;
        }
        let error_mag = sqrt(error_squared_sum);
        brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT] += learning_rate * 0.01 * (error_mag - 0.5);
        brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT] = clamp(brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT], 0.05, 0.5);
    }

    // ── 7b. Encoder credit: threads 0..31 ──────────────────────────────
    if (tid < ENCODED_DIMENSION) {
        let action_credit = decision_buffer[decision_base + DECISION_CREDIT + tid];
        if (abs(action_credit) >= CREDIT_EPSILON) {
            let scale = learning_rate * action_credit * ENCODER_CREDIT_SCALE;
            for (var j: u32 = 0u; j < FEATURE_COUNT; j = j + 1u) {
                var w = brain_state[brain_base + O_ENC_WEIGHTS + j * ENCODED_DIMENSION + tid] + scale * s_features[j];
                w = clamp(w, -2.0, 2.0);
                brain_state[brain_base + O_ENC_WEIGHTS + j * ENCODED_DIMENSION + tid] = w;
            }
        }
    }

    // ── 7c. Memory reinforcement: threads 0..127 ──────────────────────
    // Uses encoded (pre-habituation) state for memory similarity.
    if (tid < MEMORY_CAP) {
        if (pattern_buffer[pattern_base + O_PAT_ACTIVE + tid] >= 0.5) {
            var e_norm_sq: f32 = 0.0;
            var dot_val: f32 = 0.0;
            for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
                let e = s_encoded[d];
                e_norm_sq += e * e;
                dot_val += e * pattern_buffer[pattern_base + d * MEMORY_CAP + tid];
            }
            let e_norm = sqrt(e_norm_sq);
            let p_norm = pattern_buffer[pattern_base + O_PAT_NORMS + tid];
            if (e_norm >= 1e-8 && p_norm >= 1e-8) {
                let sim = clamp(dot_val / (e_norm * p_norm), -1.0, 1.0);
                if (sim > 0.3) {
                    pattern_buffer[pattern_base + O_PAT_REINF + tid] += sim * learning_rate * (1.0 - s_pred_error);
                    pattern_buffer[pattern_base + O_PAT_REINF + tid] = clamp(
                        pattern_buffer[pattern_base + O_PAT_REINF + tid], 0.0, 20.0);
                    let valence_lr = learning_rate * 0.3;
                    let old_valence = pattern_buffer[pattern_base + O_PAT_MOTOR + tid * 3u + 2u];
                    pattern_buffer[pattern_base + O_PAT_MOTOR + tid * 3u + 2u] +=
                        sim * valence_lr * (raw_gradient - old_valence);
                }
            }
        }
    }
    storageBarrier(); workgroupBarrier();

    // ── 7d. Memory store: thread 0 ─────────────────────────────────────
    // Stores encoded (pre-habituation) state for consistent memory keys.
    if (tid == 0u) {
        let min_idx = u32(pattern_buffer[pattern_base + O_MIN_REINF_IDX]);

        // Store the actual motor command from the decision buffer.
        // Negative-valence recall negates this → directed escape.
        let motor_forward = decision_buffer[decision_base + DECISION_MOTOR];
        let motor_turn = decision_buffer[decision_base + DECISION_MOTOR + 1u];
        var e_norm_sq: f32 = 0.0;
        for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u) {
            let e = s_encoded[d];
            e_norm_sq += e * e;
            pattern_buffer[pattern_base + d * MEMORY_CAP + min_idx] = e;
        }
        pattern_buffer[pattern_base + O_PAT_NORMS + min_idx] = sqrt(e_norm_sq);
        pattern_buffer[pattern_base + O_PAT_REINF + min_idx] = 1.0;
        pattern_buffer[pattern_base + O_PAT_MOTOR + min_idx * 3u] = motor_forward;
        pattern_buffer[pattern_base + O_PAT_MOTOR + min_idx * 3u + 1u] = motor_turn;
        pattern_buffer[pattern_base + O_PAT_MOTOR + min_idx * 3u + 2u] = raw_gradient;
        pattern_buffer[pattern_base + O_PAT_META + min_idx * 3u] = tick;
        pattern_buffer[pattern_base + O_PAT_META + min_idx * 3u + 1u] = tick;
        pattern_buffer[pattern_base + O_PAT_META + min_idx * 3u + 2u] = 1.0;
        pattern_buffer[pattern_base + O_PAT_ACTIVE + min_idx] = 1.0;
        pattern_buffer[pattern_base + O_LAST_STORED_IDX] = f32(min_idx);
    }
    storageBarrier(); workgroupBarrier();

    // ── 7e. Memory decay: threads 0..127 ───────────────────────────────
    // Reuse s_similarities for per-thread reinforcement tracking
    if (tid < MEMORY_CAP) {
        if (pattern_buffer[pattern_base + O_PAT_ACTIVE + tid] >= 0.5) {
            let recency = tick - pattern_buffer[pattern_base + O_PAT_META + tid * 3u + 1u];
            let act_count = pattern_buffer[pattern_base + O_PAT_META + tid * 3u + 2u];
            let freq_factor = 1.0 / (1.0 + act_count * 0.2);
            let recency_factor = min(recency / 100.0, 3.0);
            let effective_rate = decay_rate * freq_factor * (0.2 + recency_factor);
            pattern_buffer[pattern_base + O_PAT_REINF + tid] -= effective_rate;
            if (pattern_buffer[pattern_base + O_PAT_REINF + tid] <= 0.0) {
                pattern_buffer[pattern_base + O_PAT_ACTIVE + tid] = 0.0;
                s_similarities[tid] = 999.0;
            } else {
                s_similarities[tid] = pattern_buffer[pattern_base + O_PAT_REINF + tid];
            }
        } else {
            s_similarities[tid] = 999.0;
        }
    }
    workgroupBarrier();

    // ── 7f. Min tracking + active count: thread 0 ──────────────────────
    if (tid == 0u) {
        var min_reinf: f32 = 999.0;
        var min_reinf_idx: u32 = 0u;
        var active_count: u32 = 0u;
        for (var j: u32 = 0u; j < MEMORY_CAP; j = j + 1u) {
            if (s_similarities[j] < 999.0) {
                active_count += 1u;
                if (s_similarities[j] < min_reinf) {
                    min_reinf = s_similarities[j];
                    min_reinf_idx = j;
                }
            }
        }
        pattern_buffer[pattern_base + O_MIN_REINF_IDX] = f32(min_reinf_idx);
        pattern_buffer[pattern_base + O_ACTIVE_COUNT] = f32(active_count);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry point
// ═══════════════════════════════════════════════════════════════════════════

@compute @workgroup_size(256)
fn brain_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
    // SUBGROUP_ENTRY_PARAMS
) {
    let agent_id = wgid.x;
    let tid = lid.x;

    // All threads check same agent — uniform control flow, safe for barriers
    if (physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }

    coop_feature_extract(agent_id, tid);
    workgroupBarrier();

    coop_encode(agent_id, tid);
    workgroupBarrier();

    coop_habituate_homeo(agent_id, tid);
    storageBarrier(); workgroupBarrier();

    coop_recall_score(agent_id, tid);
    workgroupBarrier();

    coop_recall_topk(agent_id, tid /* SUBGROUP_TOPK_ARGS */);
    storageBarrier(); workgroupBarrier();

    coop_predict_and_act(agent_id, tid);
    storageBarrier(); workgroupBarrier();

    coop_learn_and_store(agent_id, tid);
}
