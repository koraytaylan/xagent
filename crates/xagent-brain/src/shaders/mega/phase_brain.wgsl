// ── Mega-kernel brain phase ────────────────────────────────────────────────
// Merges all 7 brain compute passes into a single function.
// Intermediate results use local arrays instead of global transient buffers.
// Persistent buffers (brain_state, pattern_buf, history_buf, decision_buf,
// sensory_buf) and brain_config are accessed globally from common.wgsl.

// ── Local helpers ──────────────────────────────────────────────────────────

fn rand_f32(seed: u32) -> f32 {
    return hash_to_float(pcg_hash(seed));
}

// Recompute cosine similarity between habituated state and a stored pattern.
// Needed because the similarities array is clobbered by topk selection.
fn cosine_sim_pat(
    tid: u32,
    hab: ptr<function, array<f32, 32>>,
    idx: u32,
) -> f32 {
    let p_base = tid * PATTERN_STRIDE;
    var dot_val: f32 = 0.0;
    var h_norm_sq: f32 = 0.0;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let h = (*hab)[d];
        let p = pattern_buf[p_base + O_PAT_STATES + idx * DIM + d];
        dot_val += h * p;
        h_norm_sq += h * h;
    }
    let h_norm = sqrt(h_norm_sq);
    let p_norm = pattern_buf[p_base + O_PAT_NORMS + idx];
    if (h_norm < 1e-8 || p_norm < 1e-8) {
        return 0.0;
    }
    return clamp(dot_val / (h_norm * p_norm), -1.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 1: Sensory input -> feature vector
// ═══════════════════════════════════════════════════════════════════════════

fn brain_feature_extract(tid: u32, out: ptr<function, array<f32, 217>>) {
    let s_base = tid * SENSORY_STRIDE;
    var fi: u32 = 0u;

    // Vision color: 192 RGBA values (direct copy)
    for (var i: u32 = 0u; i < 192u; i = i + 1u) {
        (*out)[fi] = sensory_buf[s_base + i];
        fi = fi + 1u;
    }

    // Skip vision depth (48 values at offset 192) -- not used as features

    // Proprioception: velocity magnitude (1)
    let vel_offset = 192u + 48u; // after vision_color + vision_depth
    let vx = sensory_buf[s_base + vel_offset];
    let vy = sensory_buf[s_base + vel_offset + 1u];
    let vz = sensory_buf[s_base + vel_offset + 2u];
    (*out)[fi] = sqrt(vx * vx + vy * vy + vz * vz);
    fi = fi + 1u;

    // Facing direction (3)
    let fac_offset = vel_offset + 3u;
    (*out)[fi] = sensory_buf[s_base + fac_offset];
    fi = fi + 1u;
    (*out)[fi] = sensory_buf[s_base + fac_offset + 1u];
    fi = fi + 1u;
    (*out)[fi] = sensory_buf[s_base + fac_offset + 2u];
    fi = fi + 1u;

    // Angular velocity (1)
    let ang_offset = fac_offset + 3u;
    (*out)[fi] = sensory_buf[s_base + ang_offset];
    fi = fi + 1u;

    // Energy signal (1)
    (*out)[fi] = sensory_buf[s_base + ang_offset + 1u];
    fi = fi + 1u;

    // Integrity signal (1)
    (*out)[fi] = sensory_buf[s_base + ang_offset + 2u];
    fi = fi + 1u;

    // Energy delta (1)
    (*out)[fi] = sensory_buf[s_base + ang_offset + 3u];
    fi = fi + 1u;

    // Integrity delta (1)
    (*out)[fi] = sensory_buf[s_base + ang_offset + 4u];
    fi = fi + 1u;

    // Touch contacts: 4 slots x 4 features = 16
    let touch_offset = ang_offset + 5u;
    for (var t: u32 = 0u; t < 4u; t = t + 1u) {
        let to = touch_offset + t * 4u;
        (*out)[fi] = sensory_buf[s_base + to];       // dir_x
        fi = fi + 1u;
        (*out)[fi] = sensory_buf[s_base + to + 1u];  // dir_z
        fi = fi + 1u;
        (*out)[fi] = sensory_buf[s_base + to + 2u];  // intensity
        fi = fi + 1u;
        (*out)[fi] = sensory_buf[s_base + to + 3u];  // surface_tag/4
        fi = fi + 1u;
    }
    // fi should now be 192 + 1 + 3 + 1 + 1 + 1 + 1 + 1 + 16 = 217 = FEATURE_COUNT
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 2: Features x encoder_weights + biases -> encoded state
// ═══════════════════════════════════════════════════════════════════════════

fn brain_encode(
    tid: u32,
    features: ptr<function, array<f32, 217>>,
    out: ptr<function, array<f32, 32>>,
) {
    let b_base = tid * BRAIN_STRIDE;

    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        var sum: f32 = brain_state[b_base + O_ENC_BIASES + d];
        for (var f: u32 = 0u; f < FEATURE_COUNT; f = f + 1u) {
            sum += (*features)[f] * brain_state[b_base + O_ENC_WEIGHTS + f * DIM + d];
        }
        (*out)[d] = fast_tanh(sum);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 3: Habituation + homeostasis
// ═══════════════════════════════════════════════════════════════════════════

fn brain_habituate_homeo(
    tid: u32,
    encoded: ptr<function, array<f32, 32>>,
    out_hab: ptr<function, array<f32, 32>>,
    out_homeo: ptr<function, array<f32, 6>>,
) {
    let b = tid * BRAIN_STRIDE;
    let s_base = tid * SENSORY_STRIDE;

    let sensitivity = brain_state[b + O_HAB_SENSITIVITY];

    // ── Habituation ──
    var atten_sum: f32 = 0.0;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let enc = (*encoded)[d];
        let prev = brain_state[b + O_PREV_ENCODED + d];
        let delta = abs(enc - prev);

        let old_ema = brain_state[b + O_HAB_EMA + d];
        let new_ema = (1.0 - HAB_EMA_ALPHA) * old_ema + HAB_EMA_ALPHA * delta;
        brain_state[b + O_HAB_EMA + d] = new_ema;

        let atten = clamp(new_ema * sensitivity, ATTEN_FLOOR, 1.0);
        brain_state[b + O_HAB_ATTEN + d] = atten;
        atten_sum += atten;

        (*out_hab)[d] = enc * atten;
        brain_state[b + O_PREV_ENCODED + d] = enc;
    }

    // ── Homeostasis ──
    // Read energy/integrity from sensory input (packed format)
    let vel_offset = 192u + 48u; // after vision_color + vision_depth
    let energy = sensory_buf[s_base + vel_offset + 7u];      // energy_signal
    let integrity = sensory_buf[s_base + vel_offset + 8u];    // integrity_signal

    let prev_energy = brain_state[b + O_HOMEO + 4u];
    let prev_integrity = brain_state[b + O_HOMEO + 5u];

    let energy_delta = energy - prev_energy;
    let integrity_delta = integrity - prev_integrity;
    let raw_grad = energy_delta * ENERGY_WEIGHT + integrity_delta * INTEGRITY_WEIGHT;

    // Update multi-timescale EMAs
    let grad_fast = brain_state[b + O_HOMEO + 0u] * (1.0 - FAST_ALPHA) + raw_grad * FAST_ALPHA;
    let grad_med = brain_state[b + O_HOMEO + 1u] * (1.0 - MED_ALPHA) + raw_grad * MED_ALPHA;
    let grad_slow = brain_state[b + O_HOMEO + 2u] * (1.0 - SLOW_ALPHA) + raw_grad * SLOW_ALPHA;

    brain_state[b + O_HOMEO + 0u] = grad_fast;
    brain_state[b + O_HOMEO + 1u] = grad_med;
    brain_state[b + O_HOMEO + 2u] = grad_slow;

    // Distress and urgency
    let distress_exp = brain_config[CFG_DISTRESS_EXP / 4u][CFG_DISTRESS_EXP % 4u];
    let e_clamped = clamp(energy, 0.01, 1.0);
    let i_clamped = clamp(integrity, 0.01, 1.0);
    let e_distress = min(pow(1.0 - e_clamped, distress_exp) * DISTRESS_SCALE, MAX_DISTRESS);
    let i_distress = min(pow(1.0 - i_clamped, distress_exp) * DISTRESS_SCALE, MAX_DISTRESS);
    let urgency = (e_distress + i_distress) * 0.5;

    brain_state[b + O_HOMEO + 3u] = urgency;
    brain_state[b + O_HOMEO + 4u] = energy;
    brain_state[b + O_HOMEO + 5u] = integrity;

    // Composite gradient (urgency-amplified)
    let base_grad = grad_fast * GRAD_BLEND_FAST + grad_med * GRAD_BLEND_MED + grad_slow * GRAD_BLEND_SLOW;
    let gradient = base_grad * (1.0 + urgency);
    let raw_gradient_amplified = raw_grad * (1.0 + urgency);

    // Output: [gradient, raw_gradient, urgency, grad_fast, grad_med, grad_slow]
    (*out_homeo)[0u] = gradient;
    (*out_homeo)[1u] = raw_gradient_amplified;
    (*out_homeo)[2u] = urgency;
    (*out_homeo)[3u] = grad_fast;
    (*out_homeo)[4u] = grad_med;
    (*out_homeo)[5u] = grad_slow;
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 4: Cosine similarity of habituated state vs all memory patterns
// ═══════════════════════════════════════════════════════════════════════════

fn brain_recall_score(
    tid: u32,
    habituated: ptr<function, array<f32, 32>>,
    out: ptr<function, array<f32, 128>>,
) {
    let p_base = tid * PATTERN_STRIDE;

    // Compute query norm
    var q_norm_sq: f32 = 0.0;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let v = (*habituated)[d];
        q_norm_sq += v * v;
    }
    let q_norm = sqrt(q_norm_sq);

    // Score each pattern slot
    for (var j: u32 = 0u; j < MEMORY_CAP; j = j + 1u) {
        let is_active = pattern_buf[p_base + O_PAT_ACTIVE + j];
        if (is_active < 0.5) {
            (*out)[j] = -2.0;
            continue;
        }

        // Dot product
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            dot += (*habituated)[d] * pattern_buf[p_base + O_PAT_STATES + j * DIM + d];
        }

        let p_norm = pattern_buf[p_base + O_PAT_NORMS + j];

        if (q_norm < 1e-8 || p_norm < 1e-8) {
            (*out)[j] = 0.0;
        } else {
            (*out)[j] = clamp(dot / (q_norm * p_norm), -1.0, 1.0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 5: Select top-K recalled patterns from similarity scores
// ═══════════════════════════════════════════════════════════════════════════

fn brain_recall_topk(
    tid: u32,
    sims: ptr<function, array<f32, 128>>,
    out: ptr<function, array<f32, 17>>,
) {
    let p_base = tid * PATTERN_STRIDE;
    let b_base = tid * BRAIN_STRIDE;
    let tick = brain_state[b_base + O_TICK_COUNT];

    var count: u32 = 0u;

    for (var k: u32 = 0u; k < RECALL_K; k = k + 1u) {
        var best_idx: u32 = 0u;
        var best_sim: f32 = -3.0;

        for (var j: u32 = 0u; j < MEMORY_CAP; j = j + 1u) {
            let sim = (*sims)[j];
            if (sim > best_sim) {
                best_sim = sim;
                best_idx = j;
            }
        }

        if (best_sim <= -1.5) {
            break;
        }

        (*out)[k] = f32(best_idx);
        count = count + 1u;

        // Update recalled pattern metadata
        pattern_buf[p_base + O_PAT_META + best_idx * 3u + 1u] = tick;  // last_accessed
        pattern_buf[p_base + O_PAT_META + best_idx * 3u + 2u] += 1.0;  // activation_count

        // Exclude from future iterations
        (*sims)[best_idx] = -3.0;
    }

    // Zero remaining slots
    for (var k: u32 = count; k < RECALL_K; k = k + 1u) {
        (*out)[k] = 0.0;
    }

    // Store count at the end
    (*out)[RECALL_K] = f32(count);
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 6: Prediction + credit assignment + action selection + motor fatigue
// ═══════════════════════════════════════════════════════════════════════════

fn brain_predict_and_act(
    tid: u32,
    hab: ptr<function, array<f32, 32>>,
    homeo: ptr<function, array<f32, 6>>,
    recall: ptr<function, array<f32, 17>>,
) {
    let b = tid * BRAIN_STRIDE;
    let p_base = tid * PATTERN_STRIDE;
    let hi_base = tid * HISTORY_STRIDE;
    let d_base = tid * DECISION_STRIDE;

    let tick_count = brain_state[b + O_TICK_COUNT];
    let gradient = (*homeo)[0u];
    let recall_count = u32((*recall)[RECALL_K]);

    // ────────────────────────────────────────────────────────────────────
    // 1. Prediction error (RMSE vs previous tick's prediction)
    // ────────────────────────────────────────────────────────────────────
    var err_sum: f32 = 0.0;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let prev_pred = brain_state[b + O_PREV_PREDICTION + d];
        let hab_d = (*hab)[d];
        let e = prev_pred - hab_d;
        err_sum += e * e;
    }
    let pred_error = sqrt(err_sum / f32(DIM));

    // Record into error ring
    let err_cursor = u32(brain_state[b + O_PRED_ERR_CURSOR]);
    brain_state[b + O_PRED_ERR_RING + err_cursor] = pred_error;
    brain_state[b + O_PRED_ERR_CURSOR] = f32((err_cursor + 1u) % ERROR_HISTORY_LEN);
    let err_count = brain_state[b + O_PRED_ERR_COUNT];
    if (err_count < f32(ERROR_HISTORY_LEN)) {
        brain_state[b + O_PRED_ERR_COUNT] = err_count + 1.0;
    }

    // ────────────────────────────────────────────────────────────────────
    // 2. Predictor: weighted blend of current + recalled context
    // ────────────────────────────────────────────────────────────────────
    var prediction: array<f32, 32>;

    // Matrix multiply: prediction = pred_weights x habituated
    for (var i: u32 = 0u; i < DIM; i = i + 1u) {
        var s: f32 = 0.0;
        for (var j: u32 = 0u; j < DIM; j = j + 1u) {
            s += (*hab)[j] * brain_state[b + O_PRED_WEIGHTS + i * DIM + j];
        }
        prediction[i] = s;
    }

    // Blend in recalled context
    if (recall_count > 0u) {
        let context_weight = brain_state[b + O_PRED_CTX_WT];

        // Compute total similarity
        var total_sim: f32 = 0.0;
        for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
            let idx = u32((*recall)[k]);
            let sim = cosine_sim_pat(tid, hab, idx);
            total_sim += max(sim, 0.0);
        }

        if (total_sim > 1e-8) {
            for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                let idx = u32((*recall)[k]);
                let sim = cosine_sim_pat(tid, hab, idx);
                let w = context_weight * max(sim, 0.0) / total_sim;
                for (var d: u32 = 0u; d < DIM; d = d + 1u) {
                    prediction[d] += pattern_buf[p_base + O_PAT_STATES + idx * DIM + d] * w;
                }
            }
        }
    }

    // Apply tanh
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        prediction[d] = fast_tanh(prediction[d]);
    }

    // ────────────────────────────────────────────────────────────────────
    // 3. Credit assignment over action history ring
    // ────────────────────────────────────────────────────────────────────
    var credit_signal: array<f32, 32>;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        credit_signal[d] = 0.0;
    }

    let hist_len = u32(history_buf[hi_base + O_HIST_LEN]);
    var credit_mag: f32 = 0.0;

    for (var i: u32 = 0u; i < hist_len; i = i + 1u) {
        let h_off = i * 5u;
        let rec_fwd = history_buf[hi_base + O_MOTOR_RING + h_off];
        let rec_turn = history_buf[hi_base + O_MOTOR_RING + h_off + 1u];
        let rec_tick = history_buf[hi_base + O_MOTOR_RING + h_off + 2u];
        let rec_grad = history_buf[hi_base + O_MOTOR_RING + h_off + 3u];

        let age = tick_count - rec_tick;
        let temporal = exp(-age * CREDIT_DECAY);
        if (temporal < 0.01) {
            continue;
        }

        let improvement = gradient - rec_grad;
        if (abs(improvement) < DEADZONE) {
            continue;
        }

        var effective: f32;
        if (improvement < 0.0) {
            effective = improvement * PAIN_AMP;
        } else {
            effective = improvement;
        }
        let credit = effective * temporal;
        credit_mag += abs(credit);

        // State snapshot at history[O_STATE_RING + i * DIM]
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            let feat = history_buf[hi_base + O_STATE_RING + i * DIM + d];
            let fwd_wt_update = WEIGHT_LR * credit * rec_fwd * feat;
            let trn_wt_update = WEIGHT_LR * credit * rec_turn * feat;
            brain_state[b + O_ACT_FWD_WTS + d] += fwd_wt_update;
            brain_state[b + O_ACT_TURN_WTS + d] += trn_wt_update;
            credit_signal[d] += credit * rec_fwd * feat + credit * rec_turn * feat;
        }

        // Bias updates
        brain_state[b + O_ACT_BIASES] += WEIGHT_LR * credit * rec_fwd * 0.1;
        brain_state[b + O_ACT_BIASES + 1u] += WEIGHT_LR * credit * rec_turn * 0.1;
    }

    // Normalize weights: cap L2 norm to MAX_WEIGHT_NORM
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

    // ────────────────────────────────────────────────────────────────────
    // 4. Policy evaluation (motor output)
    // ────────────────────────────────────────────────────────────────────

    // Raw policy: dot(weights, habituated) + bias
    var fwd: f32 = brain_state[b + O_ACT_BIASES];
    var trn: f32 = brain_state[b + O_ACT_BIASES + 1u];
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let hab_d = (*hab)[d];
        fwd += brain_state[b + O_ACT_FWD_WTS + d] * hab_d;
        trn += brain_state[b + O_ACT_TURN_WTS + d] * hab_d;
    }

    // Prospective: modulate toward predicted future
    let confidence = 1.0 - clamp(pred_error, 0.0, 1.0);
    if (confidence > 0.1) {
        var fwd_future: f32 = brain_state[b + O_ACT_BIASES];
        var trn_future: f32 = brain_state[b + O_ACT_BIASES + 1u];
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            fwd_future += brain_state[b + O_ACT_FWD_WTS + d] * prediction[d];
            trn_future += brain_state[b + O_ACT_TURN_WTS + d] * prediction[d];
        }
        fwd += confidence * ANTICIPATION_WEIGHT * (fwd_future - fwd);
        trn += confidence * ANTICIPATION_WEIGHT * (trn_future - trn);
    }

    // Memory blend: recalled patterns influence motor output
    if (recall_count > 0u) {
        var mem_fwd: f32 = 0.0;
        var mem_trn: f32 = 0.0;
        var total_w: f32 = 0.0;
        for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
            let idx = u32((*recall)[k]);
            let sim = cosine_sim_pat(tid, hab, idx);
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
            let mix = strength * 0.4;
            fwd = fwd * (1.0 - mix) + mem_fwd * mix;
            trn = trn * (1.0 - mix) + mem_trn * mix;
        }
    }

    // Exploration rate
    let urgency = (*homeo)[2u];
    let max_curiosity = brain_state[b + O_HAB_MAX_CURIOSITY];

    // Curiosity from habituation attenuation: (1 - mean_atten) * max_curiosity
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

    // Tanh squash + exploration noise (uniform [-1,1] scaled by exploration_rate)
    fwd = fast_tanh(fwd);
    trn = fast_tanh(trn);
    let tick_u = u32(tick_count);
    let seed_base = tid * 1000u + tick_u;
    let noise_fwd = (rand_f32(seed_base) * 2.0 - 1.0) * 0.5;
    let noise_trn = (rand_f32(seed_base + 1u) * 2.0 - 1.0) * 0.5;
    fwd = clamp(fwd + noise_fwd * exploration_rate, -1.0, 1.0);
    trn = clamp(trn + noise_trn * exploration_rate, -1.0, 1.0);

    // ────────────────────────────────────────────────────────────────────
    // 5. Motor fatigue
    // ────────────────────────────────────────────────────────────────────
    let fatigue_cursor = u32(brain_state[b + O_FATIGUE_CURSOR]);
    brain_state[b + O_FATIGUE_FWD_RING + fatigue_cursor] = fwd;
    brain_state[b + O_FATIGUE_TURN_RING + fatigue_cursor] = trn;
    let fatigue_len_val = brain_state[b + O_FATIGUE_LEN];
    let new_len = min(fatigue_len_val + 1.0, f32(ACTION_HISTORY_LEN));
    brain_state[b + O_FATIGUE_LEN] = new_len;
    brain_state[b + O_FATIGUE_CURSOR] = f32((fatigue_cursor + 1u) % ACTION_HISTORY_LEN);

    // Compute variance of fwd and turn over fatigue ring
    let f_len = u32(new_len);
    var mean_f: f32 = 0.0;
    var mean_t: f32 = 0.0;
    for (var i: u32 = 0u; i < f_len; i = i + 1u) {
        mean_f += brain_state[b + O_FATIGUE_FWD_RING + i];
        mean_t += brain_state[b + O_FATIGUE_TURN_RING + i];
    }
    mean_f /= f32(f_len);
    mean_t /= f32(f_len);

    var var_f: f32 = 0.0;
    var var_t: f32 = 0.0;
    for (var i: u32 = 0u; i < f_len; i = i + 1u) {
        let df = brain_state[b + O_FATIGUE_FWD_RING + i] - mean_f;
        let dt = brain_state[b + O_FATIGUE_TURN_RING + i] - mean_t;
        var_f += df * df;
        var_t += dt * dt;
    }
    var_f /= f32(f_len);
    var_t /= f32(f_len);

    let recovery = brain_state[b + O_FATIGUE_RECOVERY];
    let floor = brain_state[b + O_FATIGUE_FLOOR];
    let motor_variety = sqrt(var_f + var_t) * recovery;
    let fatigue_factor = clamp(
        floor + (1.0 - floor) * clamp(motor_variety, 0.0, 1.0),
        floor, 1.0
    );
    brain_state[b + O_FATIGUE_FACTOR] = fatigue_factor;

    fwd *= fatigue_factor;
    trn *= fatigue_factor;

    // ────────────────────────────────────────────────────────────────────
    // 6. Record motor output + update tick + write decision buffer
    // ────────────────────────────────────────────────────────────────────

    // Record to action history ring
    let hist_cursor = u32(history_buf[hi_base + O_HIST_CURSOR]);
    let hist_off = hist_cursor * 5u;
    history_buf[hi_base + O_MOTOR_RING + hist_off] = fwd;
    history_buf[hi_base + O_MOTOR_RING + hist_off + 1u] = trn;
    history_buf[hi_base + O_MOTOR_RING + hist_off + 2u] = tick_count;
    history_buf[hi_base + O_MOTOR_RING + hist_off + 3u] = gradient;
    history_buf[hi_base + O_MOTOR_RING + hist_off + 4u] = 0.0;  // pad

    // State snapshot
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        history_buf[hi_base + O_STATE_RING + hist_cursor * DIM + d] = (*hab)[d];
    }
    history_buf[hi_base + O_HIST_CURSOR] = f32((hist_cursor + 1u) % ACTION_HISTORY_LEN);
    let hist_len_val = history_buf[hi_base + O_HIST_LEN];
    history_buf[hi_base + O_HIST_LEN] = min(hist_len_val + 1.0, f32(ACTION_HISTORY_LEN));

    // Save prediction for next tick's error computation
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        brain_state[b + O_PREV_PREDICTION + d] = prediction[d];
    }

    // Increment tick
    brain_state[b + O_TICK_COUNT] = tick_count + 1.0;

    // Write decision buffer
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        decision_buf[d_base + d] = prediction[d];
        decision_buf[d_base + DIM + d] = credit_signal[d];
    }
    decision_buf[d_base + DIM + DIM] = fwd;
    decision_buf[d_base + DIM + DIM + 1u] = trn;
    decision_buf[d_base + DIM + DIM + 2u] = 0.0;  // strafe (unused)
    decision_buf[d_base + DIM + DIM + 3u] = 0.0;  // pad
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 7: Learning updates + memory storage + memory decay
// ═══════════════════════════════════════════════════════════════════════════

fn brain_learn_and_store(
    tid: u32,
    features: ptr<function, array<f32, 217>>,
    hab: ptr<function, array<f32, 32>>,
    homeo: ptr<function, array<f32, 6>>,
) {
    let b = tid * BRAIN_STRIDE;
    let p_base = tid * PATTERN_STRIDE;
    let d_base = tid * DECISION_STRIDE;

    let learning_rate = brain_config[1].x;
    let decay_rate = brain_config[1].y;
    let tick = brain_state[b + O_TICK_COUNT];

    // ────────────────────────────────────────────────────────────────────
    // Read prediction and credit_signal from decision buffer
    // ────────────────────────────────────────────────────────────────────
    var prediction: array<f32, 32>;
    var credit_signal: array<f32, 32>;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        prediction[d] = decision_buf[d_base + d];
        credit_signal[d] = decision_buf[d_base + DIM + d];
    }

    // ────────────────────────────────────────────────────────────────────
    // 1. Predictor learning (gradient descent on prediction error)
    // ────────────────────────────────────────────────────────────────────
    // Error: new prediction vs current habituated state
    var error_sq_sum: f32 = 0.0;
    var error_vec: array<f32, 32>;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let e = prediction[d] - (*hab)[d];
        error_vec[d] = e;
        error_sq_sum += e * e;
    }
    let error_mag = sqrt(error_sq_sum);

    // Weight update: grad = error_i * tanh_deriv * habituated_j
    for (var i: u32 = 0u; i < DIM; i = i + 1u) {
        let tanh_deriv = 1.0 - prediction[i] * prediction[i];
        for (var j: u32 = 0u; j < DIM; j = j + 1u) {
            let grad = clamp(error_vec[i] * tanh_deriv * (*hab)[j], -1.0, 1.0);
            var w = brain_state[b + O_PRED_WEIGHTS + i * DIM + j] - learning_rate * grad;
            w = clamp(w, -3.0, 3.0);
            brain_state[b + O_PRED_WEIGHTS + i * DIM + j] = w;
        }
    }

    // Context weight adaptation
    brain_state[b + O_PRED_CTX_WT] += learning_rate * 0.01 * (error_mag - 0.5);
    brain_state[b + O_PRED_CTX_WT] = clamp(brain_state[b + O_PRED_CTX_WT], 0.05, 0.5);

    // ────────────────────────────────────────────────────────────────────
    // 2. Encoder credit adaptation (Hebbian)
    // ────────────────────────────────────────────────────────────────────
    // GPU weight layout: [FEATURE_COUNT x DIM], index = j * DIM + i
    for (var i: u32 = 0u; i < DIM; i = i + 1u) {
        if (abs(credit_signal[i]) < 1e-6) {
            continue;
        }
        let scale = learning_rate * credit_signal[i] * 0.001;
        for (var j: u32 = 0u; j < FEATURE_COUNT; j = j + 1u) {
            var w = brain_state[b + O_ENC_WEIGHTS + j * DIM + i] + scale * (*features)[j];
            w = clamp(w, -2.0, 2.0);
            brain_state[b + O_ENC_WEIGHTS + j * DIM + i] = w;
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // 3. Memory reinforcement (reinforce similar active patterns)
    // ────────────────────────────────────────────────────────────────────
    let raw_gradient = (*homeo)[1u]; // urgency-amplified raw gradient
    let pred_error = clamp(error_mag, 0.0, 1.0);

    // Compute habituated norm
    var h_norm_sq: f32 = 0.0;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let h = (*hab)[d];
        h_norm_sq += h * h;
    }
    let h_norm = sqrt(h_norm_sq);

    for (var j: u32 = 0u; j < MEMORY_CAP; j = j + 1u) {
        if (pattern_buf[p_base + O_PAT_ACTIVE + j] < 0.5) {
            continue;
        }
        // Cosine similarity
        var dot_val: f32 = 0.0;
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            dot_val += (*hab)[d] * pattern_buf[p_base + O_PAT_STATES + j * DIM + d];
        }
        let p_norm = pattern_buf[p_base + O_PAT_NORMS + j];
        if (h_norm < 1e-8 || p_norm < 1e-8) {
            continue;
        }
        let sim = clamp(dot_val / (h_norm * p_norm), -1.0, 1.0);
        if (sim > 0.3) {
            pattern_buf[p_base + O_PAT_REINF + j] += sim * learning_rate * (1.0 - pred_error);
            pattern_buf[p_base + O_PAT_REINF + j] = clamp(pattern_buf[p_base + O_PAT_REINF + j], 0.0, 20.0);
            // Retroactive valence update
            let valence_lr = learning_rate * 0.3;
            let old_valence = pattern_buf[p_base + O_PAT_MOTOR + j * 3u + 2u];
            pattern_buf[p_base + O_PAT_MOTOR + j * 3u + 2u] += sim * valence_lr * (raw_gradient - old_valence);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // 4. Memory store (new pattern to weakest slot)
    // ────────────────────────────────────────────────────────────────────
    let min_idx = u32(pattern_buf[p_base + O_MIN_REINF_IDX]);
    let motor_fwd = decision_buf[d_base + DIM + DIM];
    let motor_trn = decision_buf[d_base + DIM + DIM + 1u];

    // Check if slot was previously empty before overwriting
    let was_active = pattern_buf[p_base + O_PAT_ACTIVE + min_idx];
    var active_count = u32(pattern_buf[p_base + O_ACTIVE_COUNT]);

    // Store habituated state to the slot
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        pattern_buf[p_base + O_PAT_STATES + min_idx * DIM + d] = (*hab)[d];
    }

    // Cache norm
    pattern_buf[p_base + O_PAT_NORMS + min_idx] = h_norm;

    // Set metadata
    pattern_buf[p_base + O_PAT_REINF + min_idx] = 1.0;
    pattern_buf[p_base + O_PAT_MOTOR + min_idx * 3u] = motor_fwd;
    pattern_buf[p_base + O_PAT_MOTOR + min_idx * 3u + 1u] = motor_trn;
    pattern_buf[p_base + O_PAT_MOTOR + min_idx * 3u + 2u] = raw_gradient; // outcome_valence
    pattern_buf[p_base + O_PAT_META + min_idx * 3u] = tick;       // created_at
    pattern_buf[p_base + O_PAT_META + min_idx * 3u + 1u] = tick;  // last_accessed
    pattern_buf[p_base + O_PAT_META + min_idx * 3u + 2u] = 1.0;   // activation_count
    pattern_buf[p_base + O_PAT_ACTIVE + min_idx] = 1.0;

    // Increment active_count if slot was empty
    if (was_active < 0.5) {
        active_count += 1u;
    }

    pattern_buf[p_base + O_LAST_STORED_IDX] = f32(min_idx);

    // ────────────────────────────────────────────────────────────────────
    // 5. Memory decay
    // ────────────────────────────────────────────────────────────────────
    var min_reinf: f32 = 999.0;
    var min_reinf_idx: u32 = 0u;

    for (var j: u32 = 0u; j < MEMORY_CAP; j = j + 1u) {
        if (pattern_buf[p_base + O_PAT_ACTIVE + j] < 0.5) {
            continue;
        }

        let recency = tick - pattern_buf[p_base + O_PAT_META + j * 3u + 1u]; // tick - last_accessed
        let act_count = pattern_buf[p_base + O_PAT_META + j * 3u + 2u];
        let freq_factor = 1.0 / (1.0 + act_count * 0.2);
        let recency_factor = min(recency / 100.0, 3.0);
        let effective_rate = decay_rate * freq_factor * (0.2 + recency_factor);

        pattern_buf[p_base + O_PAT_REINF + j] -= effective_rate;

        if (pattern_buf[p_base + O_PAT_REINF + j] <= 0.0) {
            pattern_buf[p_base + O_PAT_ACTIVE + j] = 0.0;
            active_count -= 1u;
        } else if (pattern_buf[p_base + O_PAT_REINF + j] < min_reinf) {
            min_reinf = pattern_buf[p_base + O_PAT_REINF + j];
            min_reinf_idx = j;
        }
    }

    pattern_buf[p_base + O_MIN_REINF_IDX] = f32(min_reinf_idx);
    pattern_buf[p_base + O_ACTIVE_COUNT] = f32(active_count);
}

// ═══════════════════════════════════════════════════════════════════════════
// Orchestrator: runs all 7 brain passes sequentially per agent
// ═══════════════════════════════════════════════════════════════════════════

fn phase_brain(tid: u32) {
    if (agent_phys[tid * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }

    // Local arrays replace global transient buffers
    var features: array<f32, 217>;     // was features_buf
    var encoded: array<f32, 32>;       // was encoded_buf
    var habituated: array<f32, 32>;    // was habituated_buf
    var homeo_out: array<f32, 6>;      // was homeo_out_buf
    var similarities: array<f32, 128>; // was similarities_buf
    var recall_indices: array<f32, 17>; // was recall_buf (16 indices + count)

    brain_feature_extract(tid, &features);
    brain_encode(tid, &features, &encoded);
    brain_habituate_homeo(tid, &encoded, &habituated, &homeo_out);
    brain_recall_score(tid, &habituated, &similarities);
    brain_recall_topk(tid, &similarities, &recall_indices);
    brain_predict_and_act(tid, &habituated, &homeo_out, &recall_indices);
    brain_learn_and_store(tid, &features, &habituated, &homeo_out);
}
