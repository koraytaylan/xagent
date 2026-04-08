// Pass 7: Learning updates + memory storage + memory decay
// Performs predictor weight gradient descent, encoder credit adaptation,
// memory reinforcement, pattern storage to weakest slot, and pattern decay.

@group(0) @binding(0) var<storage, read> habituated: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read_write> brain_state: array<f32>;
@group(0) @binding(3) var<storage, read_write> patterns: array<f32>;
@group(0) @binding(4) var<storage, read> decision: array<f32>;
@group(0) @binding(5) var<storage, read> homeo_out: array<f32>;
@group(0) @binding(6) var<uniform> config: array<vec4<f32>, 2>;
@group(0) @binding(7) var<storage, read> encoded: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent = gid.x;
    let b = agent * BRAIN_STRIDE;
    let h_base = agent * DIM;
    let e_base = agent * DIM;
    let f_base = agent * FEATURES_STRIDE;
    let p_base = agent * PATTERN_STRIDE;
    let d_base = agent * DECISION_STRIDE;
    let ho_base = agent * HOMEO_OUT_STRIDE;

    let learning_rate = config[1].x;
    let decay_rate = config[1].y;
    let tick = brain_state[b + O_TICK_COUNT];

    // ────────────────────────────────────────────────────────────────────
    // Read prediction and credit_signal from decision buffer
    // ────────────────────────────────────────────────────────────────────
    var prediction: array<f32, 32>;
    var credit_signal: array<f32, 32>;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        prediction[d] = decision[d_base + d];
        credit_signal[d] = decision[d_base + DIM + d];
    }

    // ────────────────────────────────────────────────────────────────────
    // 1. Predictor learning (gradient descent on prediction error)
    // ────────────────────────────────────────────────────────────────────
    // Error: new prediction vs current habituated state
    var error_sq_sum: f32 = 0.0;
    var error_vec: array<f32, 32>;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let e = prediction[d] - habituated[h_base + d];
        error_vec[d] = e;
        error_sq_sum += e * e;
    }
    let error_mag = sqrt(error_sq_sum);

    // Weight update: grad = error_i * tanh_deriv * habituated_j
    for (var i: u32 = 0u; i < DIM; i = i + 1u) {
        let tanh_deriv = 1.0 - prediction[i] * prediction[i];
        for (var j: u32 = 0u; j < DIM; j = j + 1u) {
            let grad = clamp(error_vec[i] * tanh_deriv * habituated[h_base + j], -1.0, 1.0);
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
            var w = brain_state[b + O_ENC_WEIGHTS + j * DIM + i] + scale * features[f_base + j];
            w = clamp(w, -2.0, 2.0);
            brain_state[b + O_ENC_WEIGHTS + j * DIM + i] = w;
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // 3. Memory reinforcement (reinforce similar active patterns)
    //    Uses encoded (pre-habituation) state so attenuation does not
    //    degrade similarity for sustained stimuli.
    // ────────────────────────────────────────────────────────────────────
    let raw_gradient = homeo_out[ho_base + 1u]; // urgency-amplified raw gradient
    let pred_error = clamp(error_mag, 0.0, 1.0);

    // Compute encoded norm
    var enc_norm_sq: f32 = 0.0;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let e = encoded[e_base + d];
        enc_norm_sq += e * e;
    }
    let enc_norm = sqrt(enc_norm_sq);

    for (var j: u32 = 0u; j < MEMORY_CAP; j = j + 1u) {
        if (patterns[p_base + O_PAT_ACTIVE + j] < 0.5) {
            continue;
        }
        // Cosine similarity (encoded vs stored encoded pattern)
        var dot_val: f32 = 0.0;
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {
            dot_val += encoded[e_base + d] * patterns[p_base + d * MEMORY_CAP + j];
        }
        let p_norm = patterns[p_base + O_PAT_NORMS + j];
        if (enc_norm < 1e-8 || p_norm < 1e-8) {
            continue;
        }
        let sim = clamp(dot_val / (enc_norm * p_norm), -1.0, 1.0);
        if (sim > 0.3) {
            patterns[p_base + O_PAT_REINF + j] += sim * learning_rate * (1.0 - pred_error);
            patterns[p_base + O_PAT_REINF + j] = clamp(patterns[p_base + O_PAT_REINF + j], 0.0, 20.0);
            // Retroactive valence update
            let valence_lr = learning_rate * 0.3;
            let old_valence = patterns[p_base + O_PAT_MOTOR + j * 3u + 2u];
            patterns[p_base + O_PAT_MOTOR + j * 3u + 2u] += sim * valence_lr * (raw_gradient - old_valence);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // 4. Memory store (new pattern to weakest slot)
    //    Stores encoded (pre-habituation) state so that memory keys are
    //    consistent regardless of habituation level at storage time.
    // ────────────────────────────────────────────────────────────────────
    let min_idx = u32(patterns[p_base + O_MIN_REINF_IDX]);

    // Store the *approach action* (motor from ~2 ticks ago) instead of the
    // current motor.  When this memory is later recalled with negative valence,
    // negating the approach action produces "reverse the movement that led
    // into danger" — directed escape rather than freezing.
    let fatigue_len_u = u32(brain_state[b + O_FATIGUE_LEN]);
    let fatigue_cur  = u32(brain_state[b + O_FATIGUE_CURSOR]);
    var motor_fwd: f32;
    var motor_trn: f32;
    if (fatigue_len_u >= 3u) {
        // cursor already advanced past current tick's write, so:
        //   cursor-1 = this tick, cursor-2 = 1 ago, cursor-3 = 2 ago.
        let ring_idx = (fatigue_cur + ACTION_HISTORY_LEN - 3u) % ACTION_HISTORY_LEN;
        motor_fwd = brain_state[b + O_FATIGUE_FWD_RING + ring_idx];
        motor_trn = brain_state[b + O_FATIGUE_TURN_RING + ring_idx];
    } else if (fatigue_len_u == 2u) {
        // Use oldest available pre-noise command (1 tick ago).
        let ring_idx = (fatigue_cur + ACTION_HISTORY_LEN - 2u) % ACTION_HISTORY_LEN;
        motor_fwd = brain_state[b + O_FATIGUE_FWD_RING + ring_idx];
        motor_trn = brain_state[b + O_FATIGUE_TURN_RING + ring_idx];
    } else if (fatigue_len_u == 1u) {
        // Only current tick in ring; still pre-noise so prefer over decision.
        let ring_idx = (fatigue_cur + ACTION_HISTORY_LEN - 1u) % ACTION_HISTORY_LEN;
        motor_fwd = brain_state[b + O_FATIGUE_FWD_RING + ring_idx];
        motor_trn = brain_state[b + O_FATIGUE_TURN_RING + ring_idx];
    } else {
        // Ring truly empty; fall back to decision buffer.
        motor_fwd = decision[d_base + DIM + DIM];
        motor_trn = decision[d_base + DIM + DIM + 1u];
    }

    // Check if slot was previously empty before overwriting
    let was_active = patterns[p_base + O_PAT_ACTIVE + min_idx];
    var active_count = u32(patterns[p_base + O_ACTIVE_COUNT]);

    // Store encoded state to the slot
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        patterns[p_base + d * MEMORY_CAP + min_idx] = encoded[e_base + d];
    }

    // Cache norm
    patterns[p_base + O_PAT_NORMS + min_idx] = enc_norm;

    // Set metadata
    patterns[p_base + O_PAT_REINF + min_idx] = 1.0;
    patterns[p_base + O_PAT_MOTOR + min_idx * 3u] = motor_fwd;
    patterns[p_base + O_PAT_MOTOR + min_idx * 3u + 1u] = motor_trn;
    patterns[p_base + O_PAT_MOTOR + min_idx * 3u + 2u] = raw_gradient; // outcome_valence
    patterns[p_base + O_PAT_META + min_idx * 3u] = tick;       // created_at
    patterns[p_base + O_PAT_META + min_idx * 3u + 1u] = tick;  // last_accessed
    patterns[p_base + O_PAT_META + min_idx * 3u + 2u] = 1.0;   // activation_count
    patterns[p_base + O_PAT_ACTIVE + min_idx] = 1.0;

    // Increment active_count if slot was empty
    if (was_active < 0.5) {
        active_count += 1u;
    }

    patterns[p_base + O_LAST_STORED_IDX] = f32(min_idx);

    // ────────────────────────────────────────────────────────────────────
    // 5. Memory decay
    // ────────────────────────────────────────────────────────────────────
    var min_reinf: f32 = 999.0;
    var min_reinf_idx: u32 = 0u;

    for (var j: u32 = 0u; j < MEMORY_CAP; j = j + 1u) {
        if (patterns[p_base + O_PAT_ACTIVE + j] < 0.5) {
            continue;
        }

        let recency = tick - patterns[p_base + O_PAT_META + j * 3u + 1u]; // tick - last_accessed
        let act_count = patterns[p_base + O_PAT_META + j * 3u + 2u];
        let freq_factor = 1.0 / (1.0 + act_count * 0.2);
        let recency_factor = min(recency / 100.0, 3.0);
        let effective_rate = decay_rate * freq_factor * (0.2 + recency_factor);

        patterns[p_base + O_PAT_REINF + j] -= effective_rate;

        if (patterns[p_base + O_PAT_REINF + j] <= 0.0) {
            patterns[p_base + O_PAT_ACTIVE + j] = 0.0;
            active_count -= 1u;
        } else if (patterns[p_base + O_PAT_REINF + j] < min_reinf) {
            min_reinf = patterns[p_base + O_PAT_REINF + j];
            min_reinf_idx = j;
        }
    }

    patterns[p_base + O_MIN_REINF_IDX] = f32(min_reinf_idx);
    patterns[p_base + O_ACTIVE_COUNT] = f32(active_count);
}
