// ── Cooperative brain tick: multi-workgroup, 256 threads per agent ──────────
// dispatch(agent_count, 1, 1) — one workgroup per agent.
// 256 threads cooperate on the 7 brain passes via shared memory.

const BRAIN_WORKGROUP_SIZE: u32 = 256u;

// ── Shared memory (~2.5 KB at default 8×6; scales with FEATURE_COUNT) ──────

var<workgroup> s_features: array<f32, FEATURE_COUNT>;
var<workgroup> s_encoded: array<f32, 32>;
var<workgroup> s_habituated: array<f32, 32>;
var<workgroup> s_homeo: array<f32, 6>;
var<workgroup> s_similarities: array<f32, 128>;  // reused for decay tracking in pass 7
var<workgroup> shared_sort_indices: array<u32, 128>;   // tracks pattern index through bitonic sort
var<workgroup> s_recall: array<f32, 17>;
var<workgroup> s_prediction: array<f32, 32>;
var<workgroup> s_credit: array<f32, 32>;
var<workgroup> s_pred_error: f32;

// ── Helpers ────────────────────────────────────────────────────────────────

fn rand_f32_brain(seed: u32) -> f32 {
    return hash_to_float(pcg_hash(seed));
}

// Cosine similarity using shared encoded state (pre-habituation)
// for memory operations — avoids attenuation silencing recall.
fn cosine_sim_pat_s(agent_id: u32, idx: u32) -> f32 {
    let p_base = agent_id * PATTERN_STRIDE;
    var dot_val: f32 = 0.0;
    var e_norm_sq: f32 = 0.0;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let e = s_encoded[d];
        let p = pattern_buf[p_base + d * MEMORY_CAP + idx];
        dot_val += e * p;
        e_norm_sq += e * e;
    }
    let e_norm = sqrt(e_norm_sq);
    let p_norm = pattern_buf[p_base + O_PAT_NORMS + idx];
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
    // sensory_buf layout: [color(VISION_COLOR_COUNT) | depth(VISION_DEPTH_COUNT) | non-visual]
    // s_features layout:  [color(VISION_COLOR_COUNT) | depth(VISION_DEPTH_COUNT) | non-visual(25)]
    // The vision portion maps 1:1 between the two buffers.
    let vision_count = VISION_COLOR_COUNT + VISION_DEPTH_COUNT;
    for (var i = tid; i < vision_count; i += BRAIN_WORKGROUP_SIZE) {
        s_features[i] = sensory_buf[s_base + i];
    }

    // Non-visual features (25 values) — thread 0 only.
    // Velocity magnitude requires a sqrt, so this can't be a bulk copy.
    if (tid == 0u) {
        var fi = vision_count;
        let vel_offset = vision_count;
        let vx = sensory_buf[s_base + vel_offset];
        let vy = sensory_buf[s_base + vel_offset + 1u];
        let vz = sensory_buf[s_base + vel_offset + 2u];
        s_features[fi] = sqrt(vx * vx + vy * vy + vz * vz); fi = fi + 1u;
        let fac_offset = vel_offset + 3u;
        s_features[fi] = sensory_buf[s_base + fac_offset]; fi = fi + 1u;
        s_features[fi] = sensory_buf[s_base + fac_offset + 1u]; fi = fi + 1u;
        s_features[fi] = sensory_buf[s_base + fac_offset + 2u]; fi = fi + 1u;
        let ang_offset = fac_offset + 3u;
        s_features[fi] = sensory_buf[s_base + ang_offset]; fi = fi + 1u;
        s_features[fi] = sensory_buf[s_base + ang_offset + 1u]; fi = fi + 1u;
        s_features[fi] = sensory_buf[s_base + ang_offset + 2u]; fi = fi + 1u;
        s_features[fi] = sensory_buf[s_base + ang_offset + 3u]; fi = fi + 1u;
        s_features[fi] = sensory_buf[s_base + ang_offset + 4u]; fi = fi + 1u;
        let touch_offset = ang_offset + 5u;
        for (var t: u32 = 0u; t < 4u; t = t + 1u) {
            let to = touch_offset + t * 4u;
            s_features[fi] = sensory_buf[s_base + to]; fi = fi + 1u;
            s_features[fi] = sensory_buf[s_base + to + 1u]; fi = fi + 1u;
            s_features[fi] = sensory_buf[s_base + to + 2u]; fi = fi + 1u;
            s_features[fi] = sensory_buf[s_base + to + 3u]; fi = fi + 1u;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 2: Encode (threads 0..31 — FEATURE_COUNT MADs each, coalesced access)
// ═══════════════════════════════════════════════════════════════════════════

fn coop_encode(agent_id: u32, tid: u32) {
    if (tid < DIM) {
        let b_base = agent_id * BRAIN_STRIDE;
        var sum: f32 = brain_state[b_base + O_ENC_BIASES + tid];
        for (var f: u32 = 0u; f < FEATURE_COUNT; f = f + 1u) {
            sum += s_features[f] * brain_state[b_base + O_ENC_WEIGHTS + f * DIM + tid];
        }
        s_encoded[tid] = fast_tanh(sum);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 3: Habituate + Homeostasis (threads 0..31 + thread 0)
// ═══════════════════════════════════════════════════════════════════════════

fn coop_habituate_homeo(agent_id: u32, tid: u32) {
    let b = agent_id * BRAIN_STRIDE;

    if (tid < DIM) {
        let enc = s_encoded[tid];
        let prev = brain_state[b + O_PREV_ENCODED + tid];
        let delta = abs(enc - prev);
        let sensitivity = brain_state[b + O_HAB_SENSITIVITY];
        let old_ema = brain_state[b + O_HAB_EMA + tid];
        let new_ema = (1.0 - HAB_EMA_ALPHA) * old_ema + HAB_EMA_ALPHA * delta;
        brain_state[b + O_HAB_EMA + tid] = new_ema;
        let atten = clamp(new_ema * sensitivity, ATTEN_FLOOR, 1.0);
        brain_state[b + O_HAB_ATTEN + tid] = atten;
        s_habituated[tid] = enc * atten;
        brain_state[b + O_PREV_ENCODED + tid] = enc;
    }

    if (tid == 0u) {
        let s_base = agent_id * SENSORY_STRIDE;
        let vel_offset = VISION_COLOR_COUNT + VISION_DEPTH_COUNT;
        let energy = sensory_buf[s_base + vel_offset + 7u];
        let integrity = sensory_buf[s_base + vel_offset + 8u];
        let prev_energy = brain_state[b + O_HOMEO + 4u];
        let prev_integrity = brain_state[b + O_HOMEO + 5u];
        let energy_delta = energy - prev_energy;
        let integrity_delta = integrity - prev_integrity;
        let raw_grad = energy_delta * ENERGY_WEIGHT + integrity_delta * INTEGRITY_WEIGHT;
        let grad_fast = brain_state[b + O_HOMEO + 0u] * (1.0 - FAST_ALPHA) + raw_grad * FAST_ALPHA;
        let grad_med = brain_state[b + O_HOMEO + 1u] * (1.0 - MED_ALPHA) + raw_grad * MED_ALPHA;
        let grad_slow = brain_state[b + O_HOMEO + 2u] * (1.0 - SLOW_ALPHA) + raw_grad * SLOW_ALPHA;
        brain_state[b + O_HOMEO + 0u] = grad_fast;
        brain_state[b + O_HOMEO + 1u] = grad_med;
        brain_state[b + O_HOMEO + 2u] = grad_slow;
        let distress_exp = brain_config[CFG_DISTRESS_EXP / 4u][CFG_DISTRESS_EXP % 4u];
        let e_clamped = clamp(energy, 0.01, 1.0);
        let i_clamped = clamp(integrity, 0.01, 1.0);
        let e_distress = min(pow(1.0 - e_clamped, distress_exp) * DISTRESS_SCALE, MAX_DISTRESS);
        let i_distress = min(pow(1.0 - i_clamped, distress_exp) * DISTRESS_SCALE, MAX_DISTRESS);
        let urgency = (e_distress + i_distress) * 0.5;
        brain_state[b + O_HOMEO + 3u] = urgency;
        brain_state[b + O_HOMEO + 4u] = energy;
        brain_state[b + O_HOMEO + 5u] = integrity;
        let base_grad = grad_fast * GRAD_BLEND_FAST + grad_med * GRAD_BLEND_MED + grad_slow * GRAD_BLEND_SLOW;
        let gradient = base_grad * (1.0 + urgency);
        let raw_gradient_amplified = raw_grad * (1.0 + urgency);
        s_homeo[0u] = gradient;
        s_homeo[1u] = raw_gradient_amplified;
        s_homeo[2u] = urgency;
        s_homeo[3u] = grad_fast;
        s_homeo[4u] = grad_med;
        s_homeo[5u] = grad_slow;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 4: Recall score (threads 0..127 — one pattern each)
// ═══════════════════════════════════════════════════════════════════════════

fn coop_recall_score(agent_id: u32, tid: u32) {
    if (tid < MEMORY_CAP) {
        let p_base = agent_id * PATTERN_STRIDE;

        // Each thread computes query norm independently (32 shared reads — fast)
        // Uses encoded (pre-habituation) state for memory queries.
        var q_norm_sq: f32 = 0.0;
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            let v = s_encoded[d];
            q_norm_sq += v * v;
        }
        let q_norm = sqrt(q_norm_sq);

        let is_active = pattern_buf[p_base + O_PAT_ACTIVE + tid];
        if (is_active < 0.5) {
            s_similarities[tid] = -2.0;
        } else {
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < DIM; d = d + 1u) {
                dot += s_encoded[d] * pattern_buf[p_base + d * MEMORY_CAP + tid];
            }
            let p_norm = pattern_buf[p_base + O_PAT_NORMS + tid];
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
    let p_base = agent_id * PATTERN_STRIDE;
    let b_base = agent_id * BRAIN_STRIDE;
    let tick = brain_state[b_base + O_TICK_COUNT];

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
            pattern_buf[p_base + O_PAT_META + idx * 3u + 1u] = tick;
            pattern_buf[p_base + O_PAT_META + idx * 3u + 2u] += 1.0;
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
    let b = agent_id * BRAIN_STRIDE;
    let p_base = agent_id * PATTERN_STRIDE;
    let hi_base = agent_id * HISTORY_STRIDE;
    let d_base = agent_id * DECISION_STRIDE;
    let tick_count = brain_state[b + O_TICK_COUNT];
    let recall_count = u32(s_recall[RECALL_K]);

    // ── Predictor matmul: threads 0..31 ────────────────────────────────
    if (tid < DIM) {
        var s: f32 = 0.0;
        for (var j: u32 = 0u; j < DIM; j = j + 1u) {
            s += s_habituated[j] * brain_state[b + O_PRED_WEIGHTS + tid * DIM + j];
        }
        s_prediction[tid] = s;
    }
    workgroupBarrier();

    // ── Thread 0: rest of predict + act ────────────────────────────────
    if (tid == 0u) {
        let gradient = s_homeo[0u];
        let urgency = s_homeo[2u];

        // Prediction error
        var err_sum: f32 = 0.0;
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            let prev_pred = brain_state[b + O_PREV_PREDICTION + d];
            let e = prev_pred - s_habituated[d];
            err_sum += e * e;
        }
        let pred_error = sqrt(err_sum / f32(DIM));

        // Error ring
        let err_cursor = u32(brain_state[b + O_PRED_ERR_CURSOR]);
        brain_state[b + O_PRED_ERR_RING + err_cursor] = pred_error;
        brain_state[b + O_PRED_ERR_CURSOR] = f32((err_cursor + 1u) % ERROR_HISTORY_LEN);
        let err_count = brain_state[b + O_PRED_ERR_COUNT];
        if (err_count < f32(ERROR_HISTORY_LEN)) {
            brain_state[b + O_PRED_ERR_COUNT] = err_count + 1.0;
        }

        // Recalled context blend
        if (recall_count > 0u) {
            let context_weight = brain_state[b + O_PRED_CTX_WT];
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
                    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
                        s_prediction[d] += pattern_buf[p_base + d * MEMORY_CAP + idx] * w;
                    }
                }
            }
        }

        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            s_prediction[d] = fast_tanh(s_prediction[d]);
        }

        // Credit assignment
        let hist_len = u32(history_buf[hi_base + O_HIST_LEN]);
        for (var d: u32 = 0u; d < DIM; d = d + 1u) { s_credit[d] = 0.0; }
        var credit_mag: f32 = 0.0;

        for (var i: u32 = 0u; i < hist_len; i = i + 1u) {
            let h_off = i * 5u;
            let rec_fwd = history_buf[hi_base + O_MOTOR_RING + h_off];
            let rec_turn = history_buf[hi_base + O_MOTOR_RING + h_off + 1u];
            let rec_tick = history_buf[hi_base + O_MOTOR_RING + h_off + 2u];
            let rec_grad = history_buf[hi_base + O_MOTOR_RING + h_off + 3u];
            let age = tick_count - rec_tick;
            let temporal = exp(-age * CREDIT_DECAY);
            if (temporal < 0.01) { continue; }
            let improvement = gradient - rec_grad;
            var credit_input = improvement;
            var is_tonic = false;
            if (abs(improvement) < DEADZONE) {
                credit_input = gradient * urgency * TONIC_CREDIT_SCALE;
                is_tonic = true;
            }
            if (!is_tonic && abs(credit_input) < DEADZONE) { continue; }
            if (abs(credit_input) < 1e-6) { continue; }
            var effective: f32;
            if (credit_input < 0.0) { effective = credit_input * PAIN_AMP; }
            else { effective = credit_input; }
            let credit = effective * temporal;
            credit_mag += abs(credit);
            for (var d: u32 = 0u; d < DIM; d = d + 1u) {
                let feat = history_buf[hi_base + O_STATE_RING + i * DIM + d];
                let fwd_wt_update = WEIGHT_LR * credit * rec_fwd * feat;
                let trn_wt_update = WEIGHT_LR * credit * rec_turn * feat;
                brain_state[b + O_ACT_FWD_WTS + d] += fwd_wt_update;
                brain_state[b + O_ACT_TURN_WTS + d] += trn_wt_update;
                s_credit[d] += credit * rec_fwd * feat + credit * rec_turn * feat;
            }
            brain_state[b + O_ACT_BIASES] += WEIGHT_LR * credit * rec_fwd * 0.1;
            brain_state[b + O_ACT_BIASES + 1u] += WEIGHT_LR * credit * rec_turn * 0.1;
        }

        // Weight normalization
        var fwd_norm_sq: f32 = 0.0;
        var trn_norm_sq: f32 = 0.0;
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            let fw = brain_state[b + O_ACT_FWD_WTS + d];
            let tw = brain_state[b + O_ACT_TURN_WTS + d];
            fwd_norm_sq += fw * fw;
            trn_norm_sq += tw * tw;
        }
        let fwd_norm = sqrt(fwd_norm_sq);
        if (fwd_norm > MAX_WEIGHT_NORM) {
            let scale = MAX_WEIGHT_NORM / fwd_norm;
            for (var d: u32 = 0u; d < DIM; d = d + 1u) {
                brain_state[b + O_ACT_FWD_WTS + d] *= scale;
            }
        }
        let trn_norm = sqrt(trn_norm_sq);
        if (trn_norm > MAX_WEIGHT_NORM) {
            let scale = MAX_WEIGHT_NORM / trn_norm;
            for (var d: u32 = 0u; d < DIM; d = d + 1u) {
                brain_state[b + O_ACT_TURN_WTS + d] *= scale;
            }
        }

        // Policy evaluation
        var fwd: f32 = brain_state[b + O_ACT_BIASES];
        var trn: f32 = brain_state[b + O_ACT_BIASES + 1u];
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            fwd += brain_state[b + O_ACT_FWD_WTS + d] * s_habituated[d];
            trn += brain_state[b + O_ACT_TURN_WTS + d] * s_habituated[d];
        }

        // Prospective
        let confidence = 1.0 - clamp(pred_error, 0.0, 1.0);
        if (confidence > 0.1) {
            var fwd_future: f32 = brain_state[b + O_ACT_BIASES];
            var trn_future: f32 = brain_state[b + O_ACT_BIASES + 1u];
            for (var d: u32 = 0u; d < DIM; d = d + 1u) {
                fwd_future += brain_state[b + O_ACT_FWD_WTS + d] * s_prediction[d];
                trn_future += brain_state[b + O_ACT_TURN_WTS + d] * s_prediction[d];
            }
            fwd += confidence * ANTICIPATION_WEIGHT * (fwd_future - fwd);
            trn += confidence * ANTICIPATION_WEIGHT * (trn_future - trn);
        }

        // Memory blend
        if (recall_count > 0u) {
            var mem_fwd: f32 = 0.0;
            var mem_trn: f32 = 0.0;
            var total_w: f32 = 0.0;
            for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                let idx = u32(s_recall[k]);
                let sim = cosine_sim_pat_s(agent_id, idx);
                let motor_base = p_base + O_PAT_MOTOR + idx * 3u;
                let valence = pattern_buf[motor_base + 2u];
                let w = sim * valence;
                mem_fwd += w * pattern_buf[motor_base];
                mem_trn += w * pattern_buf[motor_base + 1u];
                total_w += abs(w);
            }
            if (total_w > 1e-6) {
                mem_fwd /= total_w;
                mem_trn /= total_w;
                let strength = clamp(total_w / max(f32(recall_count), 1.0), 0.0, 1.0);
                let mix_val = strength * 0.4;
                fwd = fwd * (1.0 - mix_val) + mem_fwd * mix_val;
                trn = trn * (1.0 - mix_val) + mem_trn * mix_val;
            }
        }

        // Exploration
        let max_curiosity = brain_state[b + O_HAB_MAX_CURIOSITY];
        var atten_sum: f32 = 0.0;
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            atten_sum += brain_state[b + O_HAB_ATTEN + d];
        }
        let mean_atten = atten_sum / f32(DIM);
        let curiosity = (1.0 - mean_atten) * max_curiosity;
        let novelty_bonus = min(pred_error * 2.0, 0.4);
        let urgency_penalty = min(urgency * 0.4, 0.5);
        let raw_signal = abs(fwd) + abs(trn);
        let policy_confidence = clamp(raw_signal / 2.0, 0.0, 1.0);
        let exploration_rate = clamp(
            0.5 - policy_confidence * 0.25 + novelty_bonus + curiosity - urgency_penalty,
            0.10, 0.85
        );
        brain_state[b + O_EXPLORATION_RATE] = exploration_rate;

        fwd = fast_tanh(fwd);
        trn = fast_tanh(trn);

        // Position-based staleness: record current XZ position and
        // accumulated forward output, then compare displacement against
        // expected travel to detect agents that aren't making progress.
        let phys_base_fat = agent_id * PHYS_STRIDE;
        let cur_x = agent_phys[phys_base_fat + P_POS_X];
        let cur_z = agent_phys[phys_base_fat + P_POS_Z];
        let pos_cursor = u32(brain_state[b + O_POS_RING_CURSOR]);
        brain_state[b + O_POS_RING_X + pos_cursor] = cur_x;
        brain_state[b + O_POS_RING_Z + pos_cursor] = cur_z;
        let pos_len_val = brain_state[b + O_POS_RING_LEN];
        let new_pos_len = min(pos_len_val + 1.0, f32(POS_RING_LEN));
        brain_state[b + O_POS_RING_LEN] = new_pos_len;
        brain_state[b + O_POS_RING_CURSOR] = f32((pos_cursor + 1u) % POS_RING_LEN);

        // Accumulate forward motor output (pre-noise) for expected displacement.
        // This is an approximate running total: when the position ring is full,
        // we do not subtract the overwritten slot's exact forward contribution.
        let old_accum = brain_state[b + O_ACCUM_FWD];
        var new_accum = old_accum + max(fwd, 0.0);
        if (new_pos_len >= f32(POS_RING_LEN)) {
            // We do not track per-slot forward values, so approximate a bounded
            // window by decaying the accumulator proportionally each overwrite.
            new_accum *= (f32(POS_RING_LEN) - 1.0) / f32(POS_RING_LEN);
        }
        brain_state[b + O_ACCUM_FWD] = new_accum;

        // Compute staleness: compare actual displacement to expected
        let p_len = u32(new_pos_len);
        let floor_val = brain_state[b + O_FATIGUE_FLOOR];
        var fatigue_factor: f32 = 1.0;
        if (p_len >= 4u) {
            // Oldest valid entry in the ring. When the ring is not yet full,
            // (pos_cursor + 1) points at the next write slot rather than the
            // oldest sample, so compute from the current valid length instead.
            let cursor_new = (pos_cursor + 1u) % POS_RING_LEN;
            let oldest_idx = (cursor_new + POS_RING_LEN - p_len) % POS_RING_LEN;
            let old_x = brain_state[b + O_POS_RING_X + oldest_idx];
            let old_z = brain_state[b + O_POS_RING_Z + oldest_idx];
            let dx = cur_x - old_x;
            let dz = cur_z - old_z;
            let displacement = sqrt(dx * dx + dz * dz);

            // Expected displacement: accumulated forward * move_speed * DT * stride
            // Each brain tick, the agent moves fwd * move_speed * DT * stride units
            let move_speed = brain_state[b + O_MOVEMENT_SPEED];
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
        brain_state[b + O_FATIGUE_FACTOR] = fatigue_factor;

        // Exploration noise
        let tick_u = u32(tick_count);
        let seed_base = agent_id * 1000u + tick_u;
        let noise_fwd = (rand_f32_brain(seed_base) * 2.0 - 1.0) * 0.5;
        let noise_trn = (rand_f32_brain(seed_base + 1u) * 2.0 - 1.0) * 0.5;
        fwd = clamp(fwd + noise_fwd * exploration_rate, -1.0, 1.0);
        trn = clamp(trn + noise_trn * exploration_rate, -1.0, 1.0);

        // Apply staleness-based fatigue to final output
        fwd *= fatigue_factor;
        trn *= fatigue_factor;

        // History recording
        let hist_cursor = u32(history_buf[hi_base + O_HIST_CURSOR]);
        let hist_off = hist_cursor * 5u;
        history_buf[hi_base + O_MOTOR_RING + hist_off] = fwd;
        history_buf[hi_base + O_MOTOR_RING + hist_off + 1u] = trn;
        history_buf[hi_base + O_MOTOR_RING + hist_off + 2u] = tick_count;
        history_buf[hi_base + O_MOTOR_RING + hist_off + 3u] = gradient;
        history_buf[hi_base + O_MOTOR_RING + hist_off + 4u] = 0.0;
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            history_buf[hi_base + O_STATE_RING + hist_cursor * DIM + d] = s_habituated[d];
        }
        history_buf[hi_base + O_HIST_CURSOR] = f32((hist_cursor + 1u) % ACTION_HISTORY_LEN);
        let hist_len_val = history_buf[hi_base + O_HIST_LEN];
        history_buf[hi_base + O_HIST_LEN] = min(hist_len_val + 1.0, f32(ACTION_HISTORY_LEN));

        // Save prediction + tick + decision buffer
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            brain_state[b + O_PREV_PREDICTION + d] = s_prediction[d];
        }
        brain_state[b + O_TICK_COUNT] = tick_count + 1.0;

        // Store pred_error for pass 7 (shared scalar)
        var err_sq_sum: f32 = 0.0;
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            let e = s_prediction[d] - s_habituated[d];
            err_sq_sum += e * e;
        }
        s_pred_error = clamp(sqrt(err_sq_sum), 0.0, 1.0);

        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            decision_buf[d_base + d] = s_prediction[d];
            decision_buf[d_base + DIM + d] = s_credit[d];
        }
        decision_buf[d_base + DIM + DIM] = fwd;
        decision_buf[d_base + DIM + DIM + 1u] = trn;
        decision_buf[d_base + DIM + DIM + 2u] = 0.0;
        decision_buf[d_base + DIM + DIM + 3u] = 0.0;

        // Write telemetry to physics buffer for CPU readback
        let phys_base = agent_id * PHYS_STRIDE;
        agent_phys[phys_base + P_PREDICTION_ERROR] = s_pred_error;
        agent_phys[phys_base + P_EXPLORATION_RATE_OUT] = exploration_rate;
        agent_phys[phys_base + P_FATIGUE_FACTOR_OUT] = fatigue_factor;
        agent_phys[phys_base + P_MOTOR_FWD_OUT] = fwd;
        agent_phys[phys_base + P_MOTOR_TURN_OUT] = trn;
        agent_phys[phys_base + P_GRADIENT_OUT] = gradient;
        agent_phys[phys_base + P_URGENCY_OUT] = urgency;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 7: Learn and store
// Predictor: threads 0..31; Encoder: threads 0..31;
// Memory reinforcement: threads 0..127; Decay: threads 0..127
// ═══════════════════════════════════════════════════════════════════════════

fn coop_learn_and_store(agent_id: u32, tid: u32) {
    let b = agent_id * BRAIN_STRIDE;
    let p_base = agent_id * PATTERN_STRIDE;
    let d_base = agent_id * DECISION_STRIDE;

    let learning_rate = brain_config[1].x;
    let decay_rate = brain_config[1].y;
    let tick = brain_state[b + O_TICK_COUNT];
    let raw_gradient = s_homeo[1u];

    // ── 7a. Predictor learning: threads 0..31 ──────────────────────────
    if (tid < DIM) {
        let pred_d = decision_buf[d_base + tid];
        let error_d = pred_d - s_habituated[tid];
        let tanh_deriv = 1.0 - pred_d * pred_d;
        for (var j: u32 = 0u; j < DIM; j = j + 1u) {
            let grad = clamp(error_d * tanh_deriv * s_habituated[j], -1.0, 1.0);
            var w = brain_state[b + O_PRED_WEIGHTS + tid * DIM + j] - learning_rate * grad;
            w = clamp(w, -3.0, 3.0);
            brain_state[b + O_PRED_WEIGHTS + tid * DIM + j] = w;
        }
    }

    // Thread 0: context weight adaptation
    if (tid == 0u) {
        var error_sq_sum: f32 = 0.0;
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            let e = decision_buf[d_base + d] - s_habituated[d];
            error_sq_sum += e * e;
        }
        let error_mag = sqrt(error_sq_sum);
        brain_state[b + O_PRED_CTX_WT] += learning_rate * 0.01 * (error_mag - 0.5);
        brain_state[b + O_PRED_CTX_WT] = clamp(brain_state[b + O_PRED_CTX_WT], 0.05, 0.5);
    }

    // ── 7b. Encoder credit: threads 0..31 ──────────────────────────────
    if (tid < DIM) {
        let credit_d = decision_buf[d_base + DIM + tid];
        if (abs(credit_d) >= 1e-6) {
            let scale = learning_rate * credit_d * 0.001;
            for (var j: u32 = 0u; j < FEATURE_COUNT; j = j + 1u) {
                var w = brain_state[b + O_ENC_WEIGHTS + j * DIM + tid] + scale * s_features[j];
                w = clamp(w, -2.0, 2.0);
                brain_state[b + O_ENC_WEIGHTS + j * DIM + tid] = w;
            }
        }
    }

    // ── 7c. Memory reinforcement: threads 0..127 ──────────────────────
    // Uses encoded (pre-habituation) state for memory similarity.
    if (tid < MEMORY_CAP) {
        if (pattern_buf[p_base + O_PAT_ACTIVE + tid] >= 0.5) {
            var e_norm_sq: f32 = 0.0;
            var dot_val: f32 = 0.0;
            for (var d: u32 = 0u; d < DIM; d = d + 1u) {
                let e = s_encoded[d];
                e_norm_sq += e * e;
                dot_val += e * pattern_buf[p_base + d * MEMORY_CAP + tid];
            }
            let e_norm = sqrt(e_norm_sq);
            let p_norm = pattern_buf[p_base + O_PAT_NORMS + tid];
            if (e_norm >= 1e-8 && p_norm >= 1e-8) {
                let sim = clamp(dot_val / (e_norm * p_norm), -1.0, 1.0);
                if (sim > 0.3) {
                    pattern_buf[p_base + O_PAT_REINF + tid] += sim * learning_rate * (1.0 - s_pred_error);
                    pattern_buf[p_base + O_PAT_REINF + tid] = clamp(
                        pattern_buf[p_base + O_PAT_REINF + tid], 0.0, 20.0);
                    let valence_lr = learning_rate * 0.3;
                    let old_valence = pattern_buf[p_base + O_PAT_MOTOR + tid * 3u + 2u];
                    pattern_buf[p_base + O_PAT_MOTOR + tid * 3u + 2u] +=
                        sim * valence_lr * (raw_gradient - old_valence);
                }
            }
        }
    }
    storageBarrier(); workgroupBarrier();

    // ── 7d. Memory store: thread 0 ─────────────────────────────────────
    // Stores encoded (pre-habituation) state for consistent memory keys.
    if (tid == 0u) {
        let min_idx = u32(pattern_buf[p_base + O_MIN_REINF_IDX]);

        // Store the *approach action* (motor from ~2 ticks ago) instead of the
        // current motor.  Negative-valence recall then negates the approach
        // action → directed escape, not freeze.  Uses the history ring buffer
        // which records post-noise commands; acceptable since staleness-based
        // fatigue doesn't distort the motor signal like variance-based did.
        let hi_base_mem = agent_id * HISTORY_STRIDE;
        let hist_len_u = u32(history_buf[hi_base_mem + O_HIST_LEN]);
        let hist_cur_u = u32(history_buf[hi_base_mem + O_HIST_CURSOR]);
        var motor_fwd: f32;
        var motor_trn: f32;
        if (hist_len_u >= 3u) {
            // cursor already advanced past current tick's write, so:
            //   cursor-1 = this tick, cursor-2 = 1 ago, cursor-3 = 2 ago.
            let ring_idx = (hist_cur_u + ACTION_HISTORY_LEN - 3u) % ACTION_HISTORY_LEN;
            motor_fwd = history_buf[hi_base_mem + O_MOTOR_RING + ring_idx * 5u];
            motor_trn = history_buf[hi_base_mem + O_MOTOR_RING + ring_idx * 5u + 1u];
        } else if (hist_len_u >= 1u) {
            // Use oldest available entry.
            let lookback = min(hist_len_u, 2u);
            let ring_idx = (hist_cur_u + ACTION_HISTORY_LEN - lookback) % ACTION_HISTORY_LEN;
            motor_fwd = history_buf[hi_base_mem + O_MOTOR_RING + ring_idx * 5u];
            motor_trn = history_buf[hi_base_mem + O_MOTOR_RING + ring_idx * 5u + 1u];
        } else {
            // History empty; fall back to decision buffer.
            motor_fwd = decision_buf[d_base + DIM + DIM];
            motor_trn = decision_buf[d_base + DIM + DIM + 1u];
        }
        var e_norm_sq: f32 = 0.0;
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            let e = s_encoded[d];
            e_norm_sq += e * e;
            pattern_buf[p_base + d * MEMORY_CAP + min_idx] = e;
        }
        pattern_buf[p_base + O_PAT_NORMS + min_idx] = sqrt(e_norm_sq);
        pattern_buf[p_base + O_PAT_REINF + min_idx] = 1.0;
        pattern_buf[p_base + O_PAT_MOTOR + min_idx * 3u] = motor_fwd;
        pattern_buf[p_base + O_PAT_MOTOR + min_idx * 3u + 1u] = motor_trn;
        pattern_buf[p_base + O_PAT_MOTOR + min_idx * 3u + 2u] = raw_gradient;
        pattern_buf[p_base + O_PAT_META + min_idx * 3u] = tick;
        pattern_buf[p_base + O_PAT_META + min_idx * 3u + 1u] = tick;
        pattern_buf[p_base + O_PAT_META + min_idx * 3u + 2u] = 1.0;
        pattern_buf[p_base + O_PAT_ACTIVE + min_idx] = 1.0;
        pattern_buf[p_base + O_LAST_STORED_IDX] = f32(min_idx);
    }
    storageBarrier(); workgroupBarrier();

    // ── 7e. Memory decay: threads 0..127 ───────────────────────────────
    // Reuse s_similarities for per-thread reinforcement tracking
    if (tid < MEMORY_CAP) {
        if (pattern_buf[p_base + O_PAT_ACTIVE + tid] >= 0.5) {
            let recency = tick - pattern_buf[p_base + O_PAT_META + tid * 3u + 1u];
            let act_count = pattern_buf[p_base + O_PAT_META + tid * 3u + 2u];
            let freq_factor = 1.0 / (1.0 + act_count * 0.2);
            let recency_factor = min(recency / 100.0, 3.0);
            let effective_rate = decay_rate * freq_factor * (0.2 + recency_factor);
            pattern_buf[p_base + O_PAT_REINF + tid] -= effective_rate;
            if (pattern_buf[p_base + O_PAT_REINF + tid] <= 0.0) {
                pattern_buf[p_base + O_PAT_ACTIVE + tid] = 0.0;
                s_similarities[tid] = 999.0;
            } else {
                s_similarities[tid] = pattern_buf[p_base + O_PAT_REINF + tid];
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
        pattern_buf[p_base + O_MIN_REINF_IDX] = f32(min_reinf_idx);
        pattern_buf[p_base + O_ACTIVE_COUNT] = f32(active_count);
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
    if (agent_phys[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }

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
