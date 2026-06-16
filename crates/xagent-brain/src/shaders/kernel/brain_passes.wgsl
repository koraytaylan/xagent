// ── Brain passes: the 7 cooperative functions shared by brain & kernel ──────
// dispatch(agent_count, 1, 1) — one workgroup per agent.
// 256 threads cooperate on the 7 brain passes via shared memory.
//
// This file contains workgroup storage + helpers + the 7 `coop_*` functions.
// Entry points live in:
//   - brain_tick.wgsl (standalone brain pass)
//   - kernel_tick.wgsl (fused kernel, delegates via brain_tick_inner)
//
// Composition contract (see the module-level "Shader composition contract"
// docs in `gpu_kernel.rs` and the inline composition in `GpuKernel::new`):
//   common.wgsl + brain_passes.wgsl + brain_tick.wgsl  (brain pipeline)
//   common.wgsl + brain_passes.wgsl + kernel_tick.wgsl (kernel pipeline)
//
// Subgroup markers below have stability guarantees — see the doc block at
// the top of `gpu_kernel.rs` for the full contract.

const BRAIN_WORKGROUP_SIZE: u32 = 256u;

// ── Dense tiling (plan 0006): same-dispatch cooperative parallelism ────────
// 64 output rows × 4 inner lanes = 256 invocations, using the whole workgroup
// while keeping one workgroup per agent.
const DENSE_OUTPUT_TILE: u32 = 64u;
const DENSE_INNER_LANES: u32 = 4u;

// ── Shared memory (~2.5 KB at default 8×6; scales with FEATURE_COUNT) ──────

var<workgroup> s_features: array<f32, FEATURE_COUNT>;
var<workgroup> s_encoded: array<f32, ENCODED_DIMENSION>;
var<workgroup> s_habituated: array<f32, ENCODED_DIMENSION>;
var<workgroup> s_homeo: array<f32, 6>;
var<workgroup> s_similarities: array<f32, MEMORY_CAP>;
var<workgroup> shared_sort_indices: array<u32, MEMORY_CAP>;
var<workgroup> s_recall: array<f32, 17>;
var<workgroup> s_recall_similarity: array<f32, RECALL_K>;
var<workgroup> s_prediction: array<f32, PREDICTOR_DIMENSION>;
var<workgroup> s_credit: array<f32, ENCODED_DIMENSION>;
var<workgroup> s_pred_error: f32;
var<workgroup> s_td_error: f32;
// Exploration noise terms [forward, turn] published by thread 0's motor
// block for the parallel eligibility-trace update.
var<workgroup> s_explore: array<f32, 2>;
// Reused cooperative dense-dot scratch, indexed by local invocation id; each
// group of DENSE_INNER_LANES entries reduces one output row.
var<workgroup> s_dense_partials: array<f32, BRAIN_WORKGROUP_SIZE>;

// ── Scalars for parallel action-tail reductions (plan 0006 parallelization) ──
// Each reduction writes its final scalar here so thread 0 can read without
// barrier deadlock. All 256 threads participate in the reductions.
var<workgroup> s_err_sum: f32;
var<workgroup> s_value: f32;
var<workgroup> s_fwd_norm_sq: f32;
var<workgroup> s_trn_norm_sq: f32;
var<workgroup> s_val_norm_sq: f32;
var<workgroup> s_forward_dot: f32;
var<workgroup> s_turn_dot: f32;
var<workgroup> s_atten_sum: f32;
var<workgroup> s_fwd_scale: f32;
var<workgroup> s_trn_scale: f32;
var<workgroup> s_val_scale: f32;

// ── Encoded-vector norm shared by memory reinforcement and store (plan 0006) ──
var<workgroup> s_enc_norm: f32;

// ── Memory reinforcement tiling (plan 0006): 256 threads × 2 lanes per pattern ──
var<workgroup> s_reinf_dot: array<f32, 256>;

// ── Argmin tracking for parallel min reduction (plan 0006) ────────────────────
var<workgroup> s_argmin_val: array<f32, MEMORY_CAP>;
var<workgroup> s_argmin_idx: array<u32, MEMORY_CAP>;

// ── Helpers ────────────────────────────────────────────────────────────────

fn rand_f32_brain(seed: u32) -> f32 {
    return hash_to_float(pcg_hash(seed));
}

// Fixed-order tree reduction of s_dense_partials[0..ENCODED_DIMENSION) into [0].
// Must be called by ALL workgroup invocations (barrier uniformity).
// This is the parallel action-tail reduction helper (plan 0006).
fn wg_reduce_dense(tid: u32) {
    var stride: u32 = ENCODED_DIMENSION / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            s_dense_partials[tid] = s_dense_partials[tid] + s_dense_partials[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
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
        // Interoception is read same-cycle from physics_state rather than
        // from the batch-lagged sensory_buffer: pain and satiety must be
        // felt at decision time, not one vision batch later. Same-cycle
        // physics reads from thread 0 are the established pattern in
        // coop_habituate_homeo. The packed sensory_buffer slots remain for
        // CPU readback; the brain just stops consuming them. The deltas
        // cover the last physics sub-tick, which is the one that contains
        // any eat event or hazard damage from this cycle.
        let interoception_base = agent_id * PHYS_STRIDE;
        let current_max_energy = max(physics_state[interoception_base + P_MAX_ENERGY], 1e-6);
        let current_max_integrity = max(physics_state[interoception_base + P_MAX_INTEGRITY], 1e-6);
        let current_energy = physics_state[interoception_base + P_ENERGY];
        let current_integrity = physics_state[interoception_base + P_INTEGRITY];
        s_features[fi] = current_energy / current_max_energy; fi = fi + 1u;
        s_features[fi] = current_integrity / current_max_integrity; fi = fi + 1u;
        s_features[fi] = current_energy - physics_state[interoception_base + P_PREV_ENERGY]; fi = fi + 1u;
        s_features[fi] = current_integrity - physics_state[interoception_base + P_PREV_INTEGRITY]; fi = fi + 1u;
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
// Pass 2: Encode (plan 0006 dense tiling: all 256 lanes, 64 output rows × 4 lanes)
// ═══════════════════════════════════════════════════════════════════════════

fn coop_encode(agent_id: u32, tid: u32) {
    let brain_base = agent_id * BRAIN_STRIDE;
    let output_in_tile = tid / DENSE_INNER_LANES;   // 0..63
    let lane = tid % DENSE_INNER_LANES;              // 0..3

    for (var tile = 0u; tile < ENCODED_DIMENSION; tile += DENSE_OUTPUT_TILE) {
        let dim = tile + output_in_tile;             // the output row this invocation serves

        // Bias seeding: lane 0 starts with bias, others with 0
        var partial: f32 = 0.0;
        if (lane == 0u) {
            partial = brain_state[brain_base + O_ENC_BIASES + dim];
        }

        // Each lane accumulates features with stride DENSE_INNER_LANES
        for (var f = lane; f < FEATURE_COUNT; f += DENSE_INNER_LANES) {
            partial += s_features[f] * brain_state[brain_base + O_ENC_WEIGHTS + f * ENCODED_DIMENSION + dim];
        }
        s_dense_partials[tid] = partial;
        workgroupBarrier();

        // Lane 0 reduces the 4 partials in ascending order and writes the result
        if (lane == 0u) {
            let base = tid; // When lane==0, tid = output_in_tile*4, which is the base
            let reduced = s_dense_partials[base] + s_dense_partials[base + 1u] + s_dense_partials[base + 2u] + s_dense_partials[base + 3u];
            s_encoded[dim] = fast_tanh(reduced);
        }
        workgroupBarrier();   // REQUIRED before the next tile overwrites s_dense_partials
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
        // Potential-based approach shaping (optimal-policy-invariant). Φ(s) =
        // −gain·d_norm rises toward 0 as the agent closes on in-range food;
        // F = γΦ(s′) − Φ(s) telescopes over an episode, so it only accelerates
        // credit toward the unchanged eat objective, never alters it. Folding F
        // into raw_gradient here (not just the TD reward) propagates the
        // approach signal to the reward, the homeostatic EMAs, and the memory
        // valence in one place. P_NEAREST_FOOD_DISTANCE was written same-cycle by
        // the food-detect pass; phys_base_homeo is already bound above.
        let d_norm = clamp(
            physics_state[phys_base_homeo + P_NEAREST_FOOD_DISTANCE] / SHAPING_RADIUS,
            0.0, 1.0);
        let potential = -APPROACH_SHAPING_GAIN * d_norm;
        let prev_potential = physics_state[phys_base_homeo + P_PREV_POTENTIAL];
        let shaping = TD_DISCOUNT * potential - prev_potential;
        physics_state[phys_base_homeo + P_PREV_POTENTIAL] = potential;
        let raw_gradient = energy_delta * ENERGY_WEIGHT
            + integrity_delta * INTEGRITY_WEIGHT
            + shaping;
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
// Predictor train-then-predict (forward model): threads 0..PREDICTOR_DIMENSION;
// rest (incl. novelty, TD credit, motor): thread 0
// ═══════════════════════════════════════════════════════════════════════════

fn coop_predict_and_act(agent_id: u32, tid: u32, use_scratch_prediction: bool) {
    let brain_base = agent_id * BRAIN_STRIDE;
    let pattern_base = agent_id * PATTERN_STRIDE;
    let decision_base = agent_id * DECISION_STRIDE;
    let tick_count = brain_state[brain_base + O_TICK_COUNT];
    let recall_count = u32(s_recall[RECALL_K]);

    // ── Predictor: train then predict — plan 0006 dense tiling ──
    // ParallelTiled (use_scratch_prediction = true): phase_brain_predictor_tiled
    // already trained O_PREDICTOR_WEIGHTS and wrote the row predictions into
    // SCRATCH_PREDICTION across more workgroups, so the tail just loads them.
    // Fused/SplitSerial (false): train+predict inline. `use_scratch_prediction`
    // is a uniform argument, so the branch and its barriers are uniform.
    if (use_scratch_prediction) {
        let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;
        for (var i = tid; i < PREDICTOR_DIMENSION; i += BRAIN_WORKGROUP_SIZE) {
            s_prediction[i] = brain_scratch[agent_scratch + SCRATCH_PREDICTION + i];
        }
        workgroupBarrier();
    } else {
        // All 256 threads participate: 64 rows × 4 lanes.
        // Each lane updates disjoint weights, then we reduce predictions.
        let output_in_tile = tid / DENSE_INNER_LANES;   // 0..63
        let lane = tid % DENSE_INNER_LANES;              // 0..3
        let predictor_learning_rate = bc_f32(CFG_LEARNING_RATE);

        for (var tile = 0u; tile < PREDICTOR_DIMENSION; tile += DENSE_OUTPUT_TILE) {
            let dim = tile + output_in_tile;

            // TRAIN sub-step: each lane updates disjoint weight columns
            let previous_prediction = brain_state[brain_base + O_PREV_PREDICTION + dim];
            let transition_error = previous_prediction - s_encoded[dim];
            let tanh_derivative = 1.0 - previous_prediction * previous_prediction;

            for (var j = lane; j < ENCODED_DIMENSION; j += DENSE_INNER_LANES) {
                let previous_input = brain_state[brain_base + O_PREV_ENCODED + j];
                let grad = clamp(transition_error * tanh_derivative * previous_input, -1.0, 1.0);
                var w = brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j] - predictor_learning_rate * grad;
                w = clamp(w, -3.0, 3.0);
                brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j] = w;
            }
            workgroupBarrier(); // Weight writes must be visible before predict step

            // PREDICT sub-step: each lane accumulates, lane 0 reduces
            var partial: f32 = 0.0;
            for (var j = lane; j < ENCODED_DIMENSION; j += DENSE_INNER_LANES) {
                partial += s_encoded[j] * brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j];
            }
            s_dense_partials[tid] = partial;
            workgroupBarrier();

            if (lane == 0u) {
                let base = tid; // When lane==0, tid = output_in_tile*4
                let reduced = s_dense_partials[base] + s_dense_partials[base + 1u] + s_dense_partials[base + 2u] + s_dense_partials[base + 3u];
                s_prediction[dim] = reduced;
            }
            workgroupBarrier(); // Required before next tile
        }
    }

    // ── Precompute recalled cosine similarities: threads 0..(RECALL_K-1) ──
    // Each thread computes one recall entry's cosine sim in parallel,
    // eliminating the serial 16×128 bottleneck in thread 0's memory
    // blend and context blend loops.
    if (tid < RECALL_K) {
        if (tid < recall_count) {
            s_recall_similarity[tid] = cosine_sim_pat_s(agent_id, u32(s_recall[tid]));
        } else {
            s_recall_similarity[tid] = 0.0;
        }
    }
    workgroupBarrier();

    // ── Per-dimension context blend and tanh: threads 0..PREDICTOR_DIMENSION ──
    if (tid < PREDICTOR_DIMENSION) {
        // Context blend contribution (per-dimension)
        if (recall_count > 0u) {
            let context_weight = brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT];
            var total_sim: f32 = 0.0;
            for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                total_sim += max(s_recall_similarity[k], 0.0);
            }
            if (total_sim > 1e-8) {
                for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                    let idx = u32(s_recall[k]);
                    let w = context_weight * max(s_recall_similarity[k], 0.0) / total_sim;
                    s_prediction[tid] += pattern_buffer[pattern_base + tid * MEMORY_CAP + idx] * w;
                }
            }
        }

        // Apply tanh to prediction
        s_prediction[tid] = fast_tanh(s_prediction[tid]);
    }
    workgroupBarrier();

    // ── Prediction error reduction (all threads, parallel tree reduce) ────────────────────────
    // All 256 threads cooperatively sum squared prediction errors into s_dense_partials,
    // then reduce into s_err_sum, which thread 0 uses for the error ring.
    {
        if (tid < PREDICTOR_DIMENSION) {
            let previous_prediction = brain_state[brain_base + O_PREV_PREDICTION + tid];
            let e = previous_prediction - s_encoded[tid];
            s_dense_partials[tid] = e * e;
        } else {
            s_dense_partials[tid] = 0.0;
        }
        workgroupBarrier();
        wg_reduce_dense(tid);
        if (tid == 0u) { s_err_sum = s_dense_partials[0]; }
        workgroupBarrier();
    }

    // ── Thread 0: prediction error and error ring ────────────────────────────────
    if (tid == 0u) {
        let gradient = s_homeo[0u];
        let urgency = s_homeo[2u];

        let prediction_error = sqrt(s_err_sum / f32(PREDICTOR_DIMENSION));

        // Error ring
        let err_cursor = u32(brain_state[brain_base + O_PREDICTION_ERROR_CURSOR]);
        brain_state[brain_base + O_PREDICTION_ERROR_RING + err_cursor] = prediction_error;
        brain_state[brain_base + O_PREDICTION_ERROR_CURSOR] = f32((err_cursor + 1u) % ERROR_HISTORY_LEN);
        let err_count = brain_state[brain_base + O_PREDICTION_ERROR_COUNT];
        if (err_count < f32(ERROR_HISTORY_LEN)) {
            brain_state[brain_base + O_PREDICTION_ERROR_COUNT] = err_count + 1.0;
        }

        // Pass prediction_error to the post-credit block via shared memory.
        // (Kept for pass 7 as the single forward-error value; no overwrite.)
        s_pred_error = prediction_error;
    }
    workgroupBarrier();

    // ── TD(λ) credit: value head + eligibility traces ───────────────────
    // The critic estimates the discounted homeostatic return from the
    // current encoded state. The TD error δ for the previous transition is
    // the single credit signal: it updates the critic, both policy
    // channels, and the encoder-credit vector through the per-dimension
    // eligibility traces — no deadzone, no tonic fallback, no history ring.
    {
        // Value partial products: threads 0..ENCODED_DIMENSION. Reuses
        // s_credit as ENCODED_DIMENSION-sized scratch — it is rewritten
        // with the encoder-credit values later in this block, after the
        // reduction below has consumed these partials.
        if (tid < ENCODED_DIMENSION) {
            s_credit[tid] =
                brain_state[brain_base + O_VALUE_WEIGHTS + tid] * s_encoded[tid];
        }
        workgroupBarrier();

        // Value dot reduction (all threads, parallel tree reduce) — MUST happen before
        // s_credit is overwritten with encoder-credit at the end of this block.
        {
            if (tid < ENCODED_DIMENSION) {
                s_dense_partials[tid] = s_credit[tid];
            } else {
                s_dense_partials[tid] = 0.0;
            }
            workgroupBarrier();
            wg_reduce_dense(tid);
            if (tid == 0u) { s_value = s_dense_partials[0]; }
            workgroupBarrier();
        }

        // Thread 0: form δ (using s_value) and update the scalar biases.
        if (tid == 0u) {
            var value: f32 = brain_state[brain_base + O_VALUE_BIAS] + s_value;
            // Reward is the immediate urgency-amplified homeostatic delta
            // accrued since the previous brain tick.
            let reward = s_homeo[1u];
            let prev_value = brain_state[brain_base + O_PREV_VALUE];
            let td_error = clamp(
                reward + TD_DISCOUNT * value - prev_value,
                -MAX_TD_ERROR, MAX_TD_ERROR,
            );
            brain_state[brain_base + O_PREV_VALUE] = value;
            s_td_error = td_error;

            let critic_bias_trace = brain_state[brain_base + O_TRACE_BIASES];
            let forward_bias_trace = brain_state[brain_base + O_TRACE_BIASES + 1u];
            let turn_bias_trace = brain_state[brain_base + O_TRACE_BIASES + 2u];
            brain_state[brain_base + O_VALUE_BIAS] +=
                CRITIC_LEARNING_RATE * td_error * critic_bias_trace;
            brain_state[brain_base + O_ACT_BIASES] +=
                ACTION_WEIGHT_LEARNING_RATE * td_error * forward_bias_trace;
            brain_state[brain_base + O_ACT_BIASES + 1u] +=
                ACTION_WEIGHT_LEARNING_RATE * td_error * turn_bias_trace;
        }
        workgroupBarrier();

        // Threads 0..ENCODED_DIMENSION: apply δ through the traces.
        if (tid < ENCODED_DIMENSION) {
            let td_error = s_td_error;
            let critic_trace = brain_state[brain_base + O_TRACE_CRITIC + tid];
            let forward_trace = brain_state[brain_base + O_TRACE_FWD + tid];
            let turn_trace = brain_state[brain_base + O_TRACE_TURN + tid];
            brain_state[brain_base + O_VALUE_WEIGHTS + tid] +=
                CRITIC_LEARNING_RATE * TD_VECTOR_SCALE * td_error * critic_trace;
            brain_state[brain_base + O_ACTION_FORWARD_WEIGHTS + tid] +=
                ACTION_WEIGHT_LEARNING_RATE * ACTOR_VECTOR_SCALE * td_error * forward_trace;
            brain_state[brain_base + O_ACTION_TURN_WEIGHTS + tid] +=
                ACTION_WEIGHT_LEARNING_RATE * ACTOR_VECTOR_SCALE * td_error * turn_trace;
            // Encoder credit: which encoded dimensions carried the policy's
            // eligibility when this outcome arrived.
            s_credit[tid] = td_error * (forward_trace + turn_trace);
        }
    }
    storageBarrier(); workgroupBarrier();

    // ── Weight normalization: L2 norm reductions (all threads, parallel tree reduce) ────────
    // Reduce fwd_norm_sq and trn_norm_sq in parallel to compute rescale factors.
    {
        // Forward norm squared reduction
        {
            if (tid < ENCODED_DIMENSION) {
                let fw = brain_state[brain_base + O_ACTION_FORWARD_WEIGHTS + tid];
                s_dense_partials[tid] = fw * fw;
            } else {
                s_dense_partials[tid] = 0.0;
            }
            workgroupBarrier();
            wg_reduce_dense(tid);
            if (tid == 0u) { s_fwd_norm_sq = s_dense_partials[0]; }
            workgroupBarrier();
        }

        // Turn norm squared reduction
        {
            if (tid < ENCODED_DIMENSION) {
                let tw = brain_state[brain_base + O_ACTION_TURN_WEIGHTS + tid];
                s_dense_partials[tid] = tw * tw;
            } else {
                s_dense_partials[tid] = 0.0;
            }
            workgroupBarrier();
            wg_reduce_dense(tid);
            if (tid == 0u) { s_trn_norm_sq = s_dense_partials[0]; }
            workgroupBarrier();
        }

        // Value norm squared reduction
        {
            if (tid < ENCODED_DIMENSION) {
                let vw = brain_state[brain_base + O_VALUE_WEIGHTS + tid];
                s_dense_partials[tid] = vw * vw;
            } else {
                s_dense_partials[tid] = 0.0;
            }
            workgroupBarrier();
            wg_reduce_dense(tid);
            if (tid == 0u) { s_val_norm_sq = s_dense_partials[0]; }
            workgroupBarrier();
        }

        // Thread 0: compute rescale factors and publish them
        if (tid == 0u) {
            // Weight normalization. No per-tick decay: TD updates are
            // surprise-driven (they stop when δ calibrates to zero), so decay
            // would only erase accumulated policy knowledge — including the
            // initial forward bias that provides exploration mobility. The
            // L2 balls below are the sole magnitude bound.
            brain_state[brain_base + O_ACT_BIASES] = clamp(brain_state[brain_base + O_ACT_BIASES], -MAX_WEIGHT_NORM, MAX_WEIGHT_NORM);
            brain_state[brain_base + O_ACT_BIASES + 1u] = clamp(brain_state[brain_base + O_ACT_BIASES + 1u], -MAX_WEIGHT_NORM, MAX_WEIGHT_NORM);

            var fwd_scale: f32 = 1.0;
            let fwd_norm = sqrt(s_fwd_norm_sq);
            if (fwd_norm > MAX_WEIGHT_NORM) {
                fwd_scale = MAX_WEIGHT_NORM / fwd_norm;
            }
            s_fwd_scale = fwd_scale;

            var trn_scale: f32 = 1.0;
            let trn_norm = sqrt(s_trn_norm_sq);
            if (trn_norm > MAX_WEIGHT_NORM) {
                trn_scale = MAX_WEIGHT_NORM / trn_norm;
            }
            s_trn_scale = trn_scale;

            // Value head: L2-ball clamp only — no per-tick decay. Decay would
            // continuously erase the learned value landscape, and the critic
            // must hold "states like this end well/badly" across episodes.
            var val_scale: f32 = 1.0;
            let val_norm = sqrt(s_val_norm_sq);
            if (val_norm > MAX_WEIGHT_NORM) {
                val_scale = MAX_WEIGHT_NORM / val_norm;
            }
            s_val_scale = val_scale;

            brain_state[brain_base + O_VALUE_BIAS] = clamp(
                brain_state[brain_base + O_VALUE_BIAS], -MAX_WEIGHT_NORM, MAX_WEIGHT_NORM);
        }
        workgroupBarrier();
    }

    // ── Weight rescaling: all threads apply the rescale factors in parallel ────────────────
    if (tid < ENCODED_DIMENSION) {
        brain_state[brain_base + O_ACTION_FORWARD_WEIGHTS + tid] *= s_fwd_scale;
        brain_state[brain_base + O_ACTION_TURN_WEIGHTS + tid] *= s_trn_scale;
        brain_state[brain_base + O_VALUE_WEIGHTS + tid] *= s_val_scale;
    }
    workgroupBarrier();

    // ── Policy dot product reductions (all threads, parallel tree reduce) ────────────────────
    // Reduce forward and turn policy dot products separately.
    {
        // Forward policy dot reduction
        {
            if (tid < ENCODED_DIMENSION) {
                s_dense_partials[tid] = brain_state[brain_base + O_ACTION_FORWARD_WEIGHTS + tid] * s_encoded[tid];
            } else {
                s_dense_partials[tid] = 0.0;
            }
            workgroupBarrier();
            wg_reduce_dense(tid);
            if (tid == 0u) { s_forward_dot = s_dense_partials[0]; }
            workgroupBarrier();
        }

        // Turn policy dot reduction
        {
            if (tid < ENCODED_DIMENSION) {
                s_dense_partials[tid] = brain_state[brain_base + O_ACTION_TURN_WEIGHTS + tid] * s_encoded[tid];
            } else {
                s_dense_partials[tid] = 0.0;
            }
            workgroupBarrier();
            wg_reduce_dense(tid);
            if (tid == 0u) { s_turn_dot = s_dense_partials[0]; }
            workgroupBarrier();
        }
    }


    // ── Attenuation sum reduction (all threads, parallel tree reduce) ───────────────────────
    // Must be done before thread 0 uses it in the exploration block.
    {
        if (tid < ENCODED_DIMENSION) {
            s_dense_partials[tid] = brain_state[brain_base + O_HAB_ATTEN + tid];
        } else {
            s_dense_partials[tid] = 0.0;
        }
        workgroupBarrier();
        wg_reduce_dense(tid);
        if (tid == 0u) { s_atten_sum = s_dense_partials[0]; }
        workgroupBarrier();
    }

    // ── Thread 0: exploration, noise, motor, telemetry ─────────────────────────────────────
    if (tid == 0u) {
        let gradient = s_homeo[0u];
        let urgency = s_homeo[2u];
        let prediction_error = s_pred_error;

        // Policy evaluation with bias and dot products
        var forward: f32 = brain_state[brain_base + O_ACT_BIASES] + s_forward_dot;
        var turn: f32 = brain_state[brain_base + O_ACT_BIASES + 1u] + s_turn_dot;

        // Memory blend: recalled experiences influence motor output via valence.
        // Positive valence (food memory) + similar state → reproduce approach action.
        // Negative valence (danger memory) + similar state → negate approach → escape.
        if (recall_count > 0u) {
            var mem_forward: f32 = 0.0;
            var mem_turn: f32 = 0.0;
            var total_weight: f32 = 0.0;
            for (var k: u32 = 0u; k < recall_count; k = k + 1u) {
                let idx = u32(s_recall[k]);
                let sim = s_recall_similarity[k];
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
        let mean_atten = s_atten_sum / f32(ENCODED_DIMENSION);
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

        // Publish the exploration noise terms for the eligibility-trace
        // update below. The traces carry noise, not the full motor: noise is
        // zero-mean, so only noise directions that correlate with TD errors
        // get reinforced. Using the full motor (policy + noise) creates a
        // feedback loop where any turn bias gets reinforced by every
        // positive credit event.
        s_explore[0u] = noise_forward * exploration_rate;
        s_explore[1u] = noise_turn * exploration_rate;

        // Save tick + decision buffer motor
        brain_state[brain_base + O_TICK_COUNT] = tick_count + 1.0;
        decision_buffer[decision_base + DECISION_MOTOR] = forward;
        decision_buffer[decision_base + DECISION_MOTOR + 1u] = turn;
        // Slot 2 is consumed by the physics phase as strafe — keep it zero.
        decision_buffer[decision_base + DECISION_MOTOR + 2u] = 0.0;
        // Slot 3 is TD-error telemetry for CPU readback.
        decision_buffer[decision_base + DECISION_MOTOR + 3u] = s_td_error;

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
    workgroupBarrier();

    // ── Per-dimension vector copies: threads 0..PREDICTOR_DIMENSION ──
    if (tid < PREDICTOR_DIMENSION) {
        brain_state[brain_base + O_PREV_PREDICTION + tid] = s_prediction[tid];
        decision_buffer[decision_base + DECISION_PREDICTION + tid] = s_prediction[tid];
    }

    // ── Per-dimension credit copy: threads 0..ENCODED_DIMENSION ──
    if (tid < ENCODED_DIMENSION) {
        decision_buffer[decision_base + DECISION_CREDIT + tid] = s_credit[tid];
    }
    workgroupBarrier();

    // ── Eligibility trace update (all dims in parallel) ─────────────────
    // Accumulating traces: z ← γλ·z + feature term. The critic trace
    // carries the state; the actor traces carry exploration-noise ×
    // state — the likelihood-ratio direction of the action actually
    // taken — so future TD errors credit exactly the noise kicks (and the
    // states they occurred in) that caused them.
    {
        let trace_decay = TD_DISCOUNT * TD_LAMBDA;
        if (tid < ENCODED_DIMENSION) {
            let enc = s_encoded[tid];
            brain_state[brain_base + O_TRACE_CRITIC + tid] =
                brain_state[brain_base + O_TRACE_CRITIC + tid] * trace_decay + enc;
            brain_state[brain_base + O_TRACE_FWD + tid] =
                brain_state[brain_base + O_TRACE_FWD + tid] * trace_decay + s_explore[0u] * enc;
            brain_state[brain_base + O_TRACE_TURN + tid] =
                brain_state[brain_base + O_TRACE_TURN + tid] * trace_decay + s_explore[1u] * enc;
        }
        if (tid == 0u) {
            brain_state[brain_base + O_TRACE_BIASES] =
                brain_state[brain_base + O_TRACE_BIASES] * trace_decay + 1.0;
            brain_state[brain_base + O_TRACE_BIASES + 1u] =
                brain_state[brain_base + O_TRACE_BIASES + 1u] * trace_decay + s_explore[0u];
            brain_state[brain_base + O_TRACE_BIASES + 2u] =
                brain_state[brain_base + O_TRACE_BIASES + 2u] * trace_decay + s_explore[1u];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pass 7: Learn and store
// Encoder credit: one encoded dim per thread; Memory reinforcement / decay:
// threads 0..MEMORY_CAP. (Predictor training moved to pass 6.)
// ═══════════════════════════════════════════════════════════════════════════

fn coop_learn_and_store(agent_id: u32, tid: u32, run_encoder_credit: bool) {
    let brain_base = agent_id * BRAIN_STRIDE;
    let pattern_base = agent_id * PATTERN_STRIDE;
    let decision_base = agent_id * DECISION_STRIDE;

    let learning_rate = brain_config[1].x;
    let decay_rate = brain_config[1].y;
    let tick = brain_state[brain_base + O_TICK_COUNT];
    let raw_gradient = s_homeo[1u];

    // Thread 0: context weight adaptation, driven by the same forward
    // prediction error that drives novelty.
    if (tid == 0u) {
        brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT] +=
            learning_rate * 0.01 * (s_pred_error - 0.5);
        brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT] = clamp(
            brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT], 0.05, 0.5);
    }

    // ── 7b. Encoder credit: plan 0006 dense tiling ──────────────────────
    // Task-driven nudge: features that co-occurred with TD-error eligibility
    // get their weights into this dimension strengthened.
    // Use the dense tiling: 64 output rows (encoded dims) × 4 inner lanes.
    // Each lane updates disjoint features (columns) for its dimension.
    // Skipped in ParallelTiled mode (run_encoder_credit=false): the tiled
    // phase_brain_encoder_credit_tiled dispatch performs this update instead,
    // reading SCRATCH_FEATURES rather than the workgroup s_features. No barrier
    // lives in this block, so gating it on the uniform `run_encoder_credit` flag
    // is barrier-uniformity-safe.
    if (run_encoder_credit) {
        let output_in_tile_enc = tid / DENSE_INNER_LANES;
        let lane_enc = tid % DENSE_INNER_LANES;

        for (var tile = 0u; tile < ENCODED_DIMENSION; tile += DENSE_OUTPUT_TILE) {
            let dim = tile + output_in_tile_enc;

            let action_credit = decision_buffer[decision_base + DECISION_CREDIT + dim];
            if (abs(action_credit) >= CREDIT_EPSILON) {
                let scale = learning_rate * action_credit * ENCODER_CREDIT_SCALE;
                for (var j = lane_enc; j < FEATURE_COUNT; j += DENSE_INNER_LANES) {
                    var w = brain_state[brain_base + O_ENC_WEIGHTS + j * ENCODED_DIMENSION + dim] + scale * s_features[j];
                    w = clamp(w, -2.0, 2.0);
                    brain_state[brain_base + O_ENC_WEIGHTS + j * ENCODED_DIMENSION + dim] = w;
                }
            }
        }
    }

    // ── Compute encoded-vector norm ONCE (plan 0006 memory reinforcement tiling) ──
    // All threads cooperate on the tree reduction; result is shared by 7c and 7d.
    {
        if (tid < ENCODED_DIMENSION) {
            let e = s_encoded[tid];
            s_dense_partials[tid] = e * e;
        }
        workgroupBarrier();
        wg_reduce_dense(tid);
        if (tid == 0u) { s_enc_norm = sqrt(s_dense_partials[0]); }
        workgroupBarrier();
    }

    // ── 7c. Memory reinforcement: tiled with all 256 threads (plan 0006) ───────
    // Uses encoded (pre-habituation) state for memory similarity.
    // All 256 threads compute partial dot products: pattern = tid % MEMORY_CAP,
    // lane = tid / MEMORY_CAP (0 or 1). Each lane reduces over its stride-2 half
    // of ENCODED_DIMENSION. Lane 0 combines and applies reinforcement logic;
    // lane 1 only contributes the partial.
    let pattern = tid % MEMORY_CAP;
    let lane = tid / MEMORY_CAP;

    // All 256 threads compute their partial dot product over their stride-2 dimension range
    {
        var dot: f32 = 0.0;
        for (var d = lane; d < ENCODED_DIMENSION; d += 2u) {
            dot += s_encoded[d] * pattern_buffer[pattern_base + d * MEMORY_CAP + pattern];
        }
        s_reinf_dot[tid] = dot;
    }
    workgroupBarrier();

    // Lane 0 (tid < MEMORY_CAP): combine the two partials and apply reinforcement logic
    if (tid < MEMORY_CAP) {
        let dot_val = s_reinf_dot[tid] + s_reinf_dot[tid + MEMORY_CAP];
        let e_norm = s_enc_norm;
        let p_norm = pattern_buffer[pattern_base + O_PAT_NORMS + pattern];
        if (e_norm >= 1e-8 && p_norm >= 1e-8) {
            let sim = clamp(dot_val / (e_norm * p_norm), -1.0, 1.0);
            if (sim > 0.3) {
                if (pattern_buffer[pattern_base + O_PAT_ACTIVE + pattern] >= 0.5) {
                    pattern_buffer[pattern_base + O_PAT_REINF + pattern] += sim * learning_rate * (1.0 - s_pred_error);
                    pattern_buffer[pattern_base + O_PAT_REINF + pattern] = clamp(
                        pattern_buffer[pattern_base + O_PAT_REINF + pattern], 0.0, 20.0);
                    let valence_lr = learning_rate * 0.3;
                    let old_valence = pattern_buffer[pattern_base + O_PAT_MOTOR + pattern * 3u + 2u];
                    pattern_buffer[pattern_base + O_PAT_MOTOR + pattern * 3u + 2u] +=
                        sim * valence_lr * (raw_gradient - old_valence);
                }
            }
        }
    }
    storageBarrier(); workgroupBarrier();

    // ── 7d. Memory store: parallelized per-dimension writes + thread-0 scalars ──
    // min_idx is read by all threads (same value everywhere); per-dimension writes
    // use tid < ENCODED_DIMENSION; scalar writes stay on thread 0.
    let min_idx = u32(pattern_buffer[pattern_base + O_MIN_REINF_IDX]);

    // Per-dimension encoded state write (threads 0..127)
    if (tid < ENCODED_DIMENSION) {
        pattern_buffer[pattern_base + tid * MEMORY_CAP + min_idx] = s_encoded[tid];
    }

    // Scalar writes (thread 0 only)
    if (tid == 0u) {
        let motor_forward = decision_buffer[decision_base + DECISION_MOTOR];
        let motor_turn = decision_buffer[decision_base + DECISION_MOTOR + 1u];
        pattern_buffer[pattern_base + O_PAT_NORMS + min_idx] = s_enc_norm;
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

    // ── 7f. Min tracking + active count: parallel reduction ──────────────────
    // Active count: parallel tree reduction counting entries < 999.0
    {
        if (tid < MEMORY_CAP) {
            s_dense_partials[tid] = select(0.0, 1.0, s_similarities[tid] < 999.0);
        }
        workgroupBarrier();
        wg_reduce_dense(tid);
        if (tid == 0u) { pattern_buffer[pattern_base + O_ACTIVE_COUNT] = s_dense_partials[0]; }
        workgroupBarrier();
    }

    // Argmin with first-minimum tie-break (all threads cooperate on binary-tree reduction)
    // Load values and indices; argmin reduction yields (value, index) tuple that matches
    // the serial scan (lower value wins; on tie, lower index wins).
    if (tid < MEMORY_CAP) {
        s_argmin_val[tid] = s_similarities[tid];
        s_argmin_idx[tid] = tid;
    }
    workgroupBarrier();

    // Binary-tree reduction (fixed-order loop, all threads participate uniformly)
    var stride: u32 = MEMORY_CAP / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            let other = tid + stride;
            let vo = s_argmin_val[other];
            let vt = s_argmin_val[tid];
            let io = s_argmin_idx[other];
            let it = s_argmin_idx[tid];
            // Lexicographic (value, index): lower value wins; on tie, lower index wins
            // — exactly matching the serial first-minimum scan.
            if (vo < vt || (vo == vt && io < it)) {
                s_argmin_val[tid] = vo;
                s_argmin_idx[tid] = io;
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) { pattern_buffer[pattern_base + O_MIN_REINF_IDX] = f32(s_argmin_idx[0]); }
    workgroupBarrier();

    // ── 7g. Publish this tick's encoded state for the next tick's
    // habituation delta and predictor training input ─────────────────────
    if (tid < ENCODED_DIMENSION) {
        brain_state[brain_base + O_PREV_ENCODED + tid] = s_encoded[tid];
    }
}
