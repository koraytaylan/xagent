// Pass 6: Prediction + credit assignment + action selection + motor fatigue
// Combines predictor, credit, policy evaluation, exploration, and fatigue
// into a single pass. Writes motor_output, credit_signal, and prediction
// to the decision buffer.

@group(0) @binding(0) var<storage, read> habituated: array<f32>;
@group(0) @binding(1) var<storage, read_write> brain_state: array<f32>;
@group(0) @binding(2) var<storage, read> patterns: array<f32>;
@group(0) @binding(3) var<storage, read> recall_buf: array<f32>;
@group(0) @binding(4) var<storage, read> homeo_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> history: array<f32>;
@group(0) @binding(6) var<storage, read_write> decision: array<f32>;
@group(0) @binding(7) var<storage, read> encoded: array<f32>;

const CREDIT_DECAY: f32 = 0.04;
const WEIGHT_LR: f32 = 0.10;
const PAIN_AMP: f32 = 3.0;
const DEADZONE: f32 = 0.01;
const MAX_WEIGHT_NORM: f32 = 2.0;
const ANTICIPATION_WEIGHT: f32 = 0.5;
const TONIC_CREDIT_SCALE: f32 = 0.1;

// Recompute cosine similarity between encoded state and a stored pattern.
// Uses encoded (pre-habituation) state so memory recall is not silenced
// by habituation attenuation during sustained stimuli.
fn cosine_sim(e_base: u32, p_base: u32, idx: u32) -> f32 {
    var dot_val: f32 = 0.0;
    var e_norm_sq: f32 = 0.0;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let e = encoded[e_base + d];
        let p = patterns[p_base + d * MEMORY_CAP + idx];
        dot_val += e * p;
        e_norm_sq += e * e;
    }
    let e_norm = sqrt(e_norm_sq);
    let p_norm = patterns[p_base + O_PAT_NORMS + idx];
    if (e_norm < 1e-8 || p_norm < 1e-8) {
        return 0.0;
    }
    return clamp(dot_val / (e_norm * p_norm), -1.0, 1.0);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent = gid.x;
    let b = agent * BRAIN_STRIDE;
    let h_base = agent * DIM;
    let e_base = agent * DIM;
    let p_base = agent * PATTERN_STRIDE;
    let r_base = agent * RECALL_IDX_STRIDE;
    let ho_base = agent * HOMEO_OUT_STRIDE;
    let hi_base = agent * HISTORY_STRIDE;
    let d_base = agent * DECISION_STRIDE;

    let tick_count = brain_state[b + O_TICK_COUNT];
    let gradient = homeo_out[ho_base + 0u];
    let urgency = homeo_out[ho_base + 2u];
    let recall_count = u32(recall_buf[r_base + RECALL_K]);

    // ────────────────────────────────────────────────────────────────────
    // 1. Prediction error (RMSE vs previous tick's prediction)
    // ────────────────────────────────────────────────────────────────────
    var err_sum: f32 = 0.0;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let prev_pred = brain_state[b + O_PREV_PREDICTION + d];
        let hab = habituated[h_base + d];
        let e = prev_pred - hab;
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
    var prediction: array<f32, 32>;  // DIM = 32

    // Matrix multiply: prediction = pred_weights x habituated
    for (var i: u32 = 0u; i < DIM; i = i + 1u) {
        var s: f32 = 0.0;
        for (var j: u32 = 0u; j < DIM; j = j + 1u) {
            s += habituated[h_base + j] * brain_state[b + O_PRED_WEIGHTS + i * DIM + j];
        }
        prediction[i] = s;
    }

    // Blend in recalled context
    if (recall_count > 0u) {
        let context_weight = brain_state[b + O_PRED_CTX_WT];

        // Compute total similarity
        var total_sim: f32 = 0.0;
        for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
            let idx = u32(recall_buf[r_base + k]);
            let sim = cosine_sim(e_base, p_base, idx);
            total_sim += max(sim, 0.0);
        }

        if (total_sim > 1e-8) {
            for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                let idx = u32(recall_buf[r_base + k]);
                let sim = cosine_sim(e_base, p_base, idx);
                let w = context_weight * max(sim, 0.0) / total_sim;
                for (var d: u32 = 0u; d < DIM; d = d + 1u) {
                    prediction[d] += patterns[p_base + d * MEMORY_CAP + idx] * w;
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
    var credit_signal: array<f32, 32>;  // DIM = 32
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        credit_signal[d] = 0.0;
    }

    let hist_len = u32(history[hi_base + O_HIST_LEN]);
    var credit_mag: f32 = 0.0;

    for (var i: u32 = 0u; i < hist_len; i = i + 1u) {
        let h_off = i * 5u;
        let rec_fwd = history[hi_base + O_MOTOR_RING + h_off];
        let rec_turn = history[hi_base + O_MOTOR_RING + h_off + 1u];
        let rec_tick = history[hi_base + O_MOTOR_RING + h_off + 2u];
        let rec_grad = history[hi_base + O_MOTOR_RING + h_off + 3u];

        let age = tick_count - rec_tick;
        let temporal = exp(-age * CREDIT_DECAY);
        if (temporal < 0.01) {
            continue;
        }

        let improvement = gradient - rec_grad;

        // Tonic credit: when the gradient has stabilized (improvement ≈ 0)
        // but urgency is high, use the signed gradient scaled by urgency
        // and TONIC_CREDIT_SCALE as a persistent credit signal. This
        // preserves direction during sustained pain or sustained recovery
        // while in distress, preventing learning from freezing.
        var credit_input = improvement;
        if (abs(improvement) < DEADZONE) {
            credit_input = gradient * urgency * TONIC_CREDIT_SCALE;
        }
        if (abs(credit_input) < DEADZONE) {
            continue;
        }

        var effective: f32;
        if (credit_input < 0.0) {
            effective = credit_input * PAIN_AMP;
        } else {
            effective = credit_input;
        }
        let credit = effective * temporal;
        credit_mag += abs(credit);

        // State snapshot at history[O_STATE_RING + i * DIM]
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            let feat = history[hi_base + O_STATE_RING + i * DIM + d];
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
        let hab = habituated[h_base + d];
        fwd += brain_state[b + O_ACT_FWD_WTS + d] * hab;
        trn += brain_state[b + O_ACT_TURN_WTS + d] * hab;
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
            let idx = u32(recall_buf[r_base + k]);
            let sim = cosine_sim(e_base, p_base, idx);
            let motor_base = p_base + O_PAT_MOTOR + idx * 3u;
            let valence = patterns[motor_base + 2u];
            let w = sim * valence;
            mem_fwd += w * patterns[motor_base];
            mem_trn += w * patterns[motor_base + 1u];
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

    // Tanh squash
    fwd = fast_tanh(fwd);
    trn = fast_tanh(trn);

    // ────────────────────────────────────────────────────────────────────
    // 5. Motor fatigue — record PRE-NOISE commands so exploration noise
    //    doesn't mask repetitive brain decisions (matches brain_tick.wgsl).
    // ────────────────────────────────────────────────────────────────────
    let fatigue_cursor = u32(brain_state[b + O_FATIGUE_CURSOR]);
    brain_state[b + O_FATIGUE_FWD_RING + fatigue_cursor] = fwd;
    brain_state[b + O_FATIGUE_TURN_RING + fatigue_cursor] = trn;
    let fatigue_len_val = brain_state[b + O_FATIGUE_LEN];
    let new_len = min(fatigue_len_val + 1.0, f32(ACTION_HISTORY_LEN));
    brain_state[b + O_FATIGUE_LEN] = new_len;
    brain_state[b + O_FATIGUE_CURSOR] = f32((fatigue_cursor + 1u) % ACTION_HISTORY_LEN);

    // Compute variance of fwd and turn over fatigue ring,
    // accumulating turn-direction sums in the same pass.
    let f_len = u32(new_len);
    var mean_f: f32 = 0.0;
    var mean_t: f32 = 0.0;
    var sum_turn: f32 = 0.0;
    var sum_abs_turn: f32 = 0.0;
    for (var i: u32 = 0u; i < f_len; i = i + 1u) {
        let fi = brain_state[b + O_FATIGUE_FWD_RING + i];
        let ti = brain_state[b + O_FATIGUE_TURN_RING + i];
        mean_f += fi;
        mean_t += ti;
        sum_turn += ti;
        sum_abs_turn += abs(ti);
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

    // Turn-direction consistency: detect sustained same-direction turning
    // (circling signature). turn_bias → 1 when all turns share the same
    // sign, → 0 when turns are balanced left/right.
    // Require minimum 16 samples before engaging monotony detection to
    // prevent early-life trapping on noisy small-sample estimates.
    var monotony: f32 = 0.0;
    if (f_len >= 16u) {
        let turn_denom = max(sum_abs_turn, 0.01);
        let turn_bias = abs(sum_turn) / turn_denom;
        // Squared for gentle onset; suppresses variety when turning monotonically
        monotony = turn_bias * turn_bias;
    }

    let recovery = brain_state[b + O_FATIGUE_RECOVERY];
    let floor = brain_state[b + O_FATIGUE_FLOOR];
    let motor_variety = sqrt(var_f + var_t) * recovery;
    let effective_variety = motor_variety * (1.0 - monotony * 0.75);
    let fatigue_factor = clamp(
        floor + (1.0 - floor) * clamp(effective_variety, 0.0, 1.0),
        floor, 1.0
    );
    brain_state[b + O_FATIGUE_FACTOR] = fatigue_factor;

    fwd *= fatigue_factor;
    trn *= fatigue_factor;

    // Exploration noise (applied after fatigue ring to not pollute it)
    let tick_u = u32(tick_count);
    let seed_base = agent * 1000u + tick_u;
    let noise_fwd = (rand_f32(seed_base) * 2.0 - 1.0) * 0.5;
    let noise_trn = (rand_f32(seed_base + 1u) * 2.0 - 1.0) * 0.5;
    fwd = clamp(fwd + noise_fwd * exploration_rate, -1.0, 1.0);
    trn = clamp(trn + noise_trn * exploration_rate, -1.0, 1.0);

    // ────────────────────────────────────────────────────────────────────
    // 6. Record motor output + update tick + write decision buffer
    // ────────────────────────────────────────────────────────────────────

    // Record to action history ring
    let hist_cursor = u32(history[hi_base + O_HIST_CURSOR]);
    let hist_off = hist_cursor * 5u;
    history[hi_base + O_MOTOR_RING + hist_off] = fwd;
    history[hi_base + O_MOTOR_RING + hist_off + 1u] = trn;
    history[hi_base + O_MOTOR_RING + hist_off + 2u] = tick_count;
    history[hi_base + O_MOTOR_RING + hist_off + 3u] = gradient;
    history[hi_base + O_MOTOR_RING + hist_off + 4u] = 0.0;  // pad

    // State snapshot
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        history[hi_base + O_STATE_RING + hist_cursor * DIM + d] = habituated[h_base + d];
    }
    history[hi_base + O_HIST_CURSOR] = f32((hist_cursor + 1u) % ACTION_HISTORY_LEN);
    let hist_len_val = history[hi_base + O_HIST_LEN];
    history[hi_base + O_HIST_LEN] = min(hist_len_val + 1.0, f32(ACTION_HISTORY_LEN));

    // Save prediction for next tick's error computation
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        brain_state[b + O_PREV_PREDICTION + d] = prediction[d];
    }

    // Increment tick
    brain_state[b + O_TICK_COUNT] = tick_count + 1.0;

    // Write decision buffer
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        decision[d_base + d] = prediction[d];
        decision[d_base + DIM + d] = credit_signal[d];
    }
    decision[d_base + DIM + DIM] = fwd;
    decision[d_base + DIM + DIM + 1u] = trn;
    decision[d_base + DIM + DIM + 2u] = 0.0;  // strafe (unused)
    decision[d_base + DIM + DIM + 3u] = 0.0;  // pad
}
