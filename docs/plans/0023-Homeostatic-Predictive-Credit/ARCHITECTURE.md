# Architecture — Plan 0023 (deltas)

> Edits in `crates/xagent-brain/src/shaders/kernel/common.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`,
> `crates/xagent-brain/src/buffers.rs`,
> `crates/xagent-shared/src/config.rs`,
> `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-sandbox/tests/integration.rs`,
> `docs/plans/STATUS.md`, and
> `docs/plans/0023-Homeostatic-Predictive-Credit/0001-HOMEO-PREDICTOR-DECISION.md`.
> Line numbers are hints; locate by symbol (grep for `coop_predict_and_act`,
> `O_VALUE_WEIGHTS`, `O_ORIENTATION_OFFSET`, `raw_gradient`, `s_homeo`,
> `learning_probe_mirrored_steering_is_chance`).

## 0001 — Homeostatic Gradient Predictor

Today the forward model in `coop_predict_and_act()` (`brain_passes.wgsl:1068-1107`) predicts the next encoded state `s_prediction[dim]` from the current `s_encoded` via a 128→128 dense layer, trained online on transition error (`brain_passes.wgsl:1079-1088`). The predicted state is used for context blending (`brain_passes.wgsl:1122-1143`) and the TD critic's value estimate (`brain_passes.wgsl:1195-1197`). But the predictor never learns to anticipate the *homeostatic gradient* — the scalar `raw_gradient = energy_delta × 0.6 + integrity_delta × 0.4` (`brain_passes.wgsl:913-915`) that drives all credit assignment. The TD error `δ = reward + γ·V(s′) − V(s)` (`brain_passes.wgsl:1222-1224`) uses only the *actual* gradient as reward, which arrives after the outcome — creating the temporal gap that has resisted every mechanism tried (0017 trace decay, 0018 auxiliary loss/frame-sync/gradient-shaping, 0020 direct supervision).

### Homeostatic gradient prediction head

Add a 128→1 linear head on top of `s_prediction` that predicts the homeostatic gradient. The predicted gradient provides an *anticipatory* credit signal at decision time — before the actual outcome — bridging the temporal gap without introducing any external target.

Edits:

- **Add three new brain state slots** (`common.wgsl`, between `O_ORIENTATION_OFFSET` and `O_VALUE_WEIGHTS`, ≈248-255). The homeostatic predictor weights (128 floats), bias (1 float), and previous prediction (1 float, episodic — zeroed on death) are inserted into the brain state tail. All downstream offsets (`O_VALUE_WEIGHTS` through `BRAIN_STRIDE`) shift by `ENCODED_DIMENSION + 2`:

```wgsl
// ── Homeostatic gradient predictor (plan 0023) ─────────────────────────────
// Linear head (128→1) on top of the forward model's predicted state s_prediction.
// Trained online to predict raw_gradient; the previous tick's prediction provides
// an anticipatory credit signal that bridges the ~10-tick sensory latency.
// Weights are heritable (seeded at birth, inherited, mutated); the prev-prediction
// slot is episodic (zeroed on death), like O_PREV_VALUE.
override O_HOMEO_PREDICTOR_WEIGHTS: u32 = O_ORIENTATION_OFFSET + 1u;
override O_HOMEO_PREDICTOR_BIAS: u32 = O_HOMEO_PREDICTOR_WEIGHTS + ENCODED_DIMENSION;
override O_PREV_HOMEO_PREDICTION: u32 = O_HOMEO_PREDICTOR_BIAS + 1u;

override O_VALUE_WEIGHTS: u32 = O_PREV_HOMEO_PREDICTION + 1u;
```

- **Mirror the new offsets in Rust** (`buffers.rs`, ≈104-114). Insert the three new constants between `O_ORIENTATION_OFFSET` and `O_VALUE_WEIGHTS`, matching the WGSL layout exactly:

```rust
// ── Homeostatic gradient predictor (plan 0023) ─────────────────────────
pub const O_HOMEO_PREDICTOR_WEIGHTS: usize = O_ORIENTATION_OFFSET + 1;
pub const O_HOMEO_PREDICTOR_BIAS: usize = O_HOMEO_PREDICTOR_WEIGHTS + ENCODED_DIMENSION;
pub const O_PREV_HOMEO_PREDICTION: usize = O_HOMEO_PREDICTOR_BIAS + 1;

pub const O_VALUE_WEIGHTS: usize = O_PREV_HOMEO_PREDICTION + 1;
```

- **Update `BRAIN_STRIDE`** — it is derived from `O_TRACE_BIASES + 3`, and `O_TRACE_BIASES` is derived from `O_VALUE_WEIGHTS` which shifted. Since all offsets are relative, `BRAIN_STRIDE` automatically grows by `ENCODED_DIMENSION + 2`. The `FIXED_TAIL_SIZE` constant (`buffers.rs:126`) must be updated to reflect the new tail size. The `init_brain_state_for` function in `buffers.rs` must initialize the new slots (Xavier for weights, zero for bias and prev-prediction). The `agent_death_respawn` function in `kernel_tick.wgsl` must zero `O_PREV_HOMEO_PREDICTION` alongside the other episodic slots.

- **Add the `homeo_predictive_credit_enabled` flag to `BrainConfig`** (`config.rs`, the `BrainConfig` struct). Default-off, zero-cost when false:

```rust
/// Enable the homeostatic gradient predictor: a 128→1 linear head on the
/// forward model that learns to anticipate raw_gradient. The predicted
/// gradient provides an anticipatory credit signal (β-scaled term in the
/// TD reward), bridging sensory latency without external targets.
/// Weights are heritable. Zero-cost when false.
pub homeo_predictive_credit_enabled: bool,
```

- **Bind the flag into the wconfig uniform** (`gpu_kernel.rs`, `write_agent_heritable_config()`). Pack `homeo_predictive_credit_enabled` as a u32 (0 or 1) at a documented offset in the wconfig buffer. Add the matching `CFG_HOMEO_PREDICTIVE_CREDIT_ENABLED_OFFSET` constant in `common.wgsl`.

- **Initialize predictor weights at brain birth** (`buffers.rs`, `init_brain_state_for`). Xavier-uniform initialization: each weight drawn from `U(-sqrt(6/128), sqrt(6/128)) ≈ U(-0.216, 0.216)`. Bias initialized to zero. `O_PREV_HOMEO_PREDICTION` initialized to zero.

- **Zero `O_PREV_HOMEO_PREDICTION` on death** (`kernel_tick.wgsl`, `agent_death_respawn`, ≈638-651). Add a line zeroing `brain_state[brain_base + O_PREV_HOMEO_PREDICTION]` alongside the existing episodic resets (`O_PREV_VALUE`, traces, homeo state). The predictor *weights* survive death (they are learned knowledge); only the previous-prediction slot is episodic.

### Predicted gradient computation and training

In `coop_predict_and_act()` (`brain_passes.wgsl`), after the context-blended prediction is finalized (after the tanh at ≈1141 and the `workgroupBarrier()` at ≈1143) and before the TD credit block (≈1184):

- **Compute the predicted gradient via parallel dot-product reduction** (all 256 threads). Each thread computes a partial dot product of `s_prediction[tid]` with `homeo_predictor_weights[tid]` for its stride over `PREDICTOR_DIMENSION`, writes to `s_dense_partials`, then a tree reduction produces the dot product in `s_dense_partials[0]`. Thread 0 adds the bias:

```wgsl
// ── Homeostatic gradient prediction (plan 0023) ─────────────────────────
// Predict raw_gradient from the forward model's predicted state.
// Gated on the flag; no-op (zero prediction) when disabled.
var predicted_gradient: f32 = 0.0;
let homeo_pred_enabled = bc_f32(CFG_HOMEO_PREDICTIVE_CREDIT_ENABLED) != 0.0;
if (homeo_pred_enabled) {
    // Parallel dot product: s_prediction · homeo_predictor_weights
    if (tid < PREDICTOR_DIMENSION) {
        s_dense_partials[tid] = s_prediction[tid]
            * brain_state[brain_base + O_HOMEO_PREDICTOR_WEIGHTS + tid];
    } else {
        s_dense_partials[tid] = 0.0;
    }
    workgroupBarrier();
    wg_reduce_dense(tid);
    if (tid == 0u) {
        predicted_gradient = s_dense_partials[0]
            + brain_state[brain_base + O_HOMEO_PREDICTOR_BIAS];
    }
    workgroupBarrier();
}
```

- **Train the predictor** (thread 0, after the dot product). Use the *previous* tick's prediction (stored in `O_PREV_HOMEO_PREDICTION`) vs the *current* tick's actual `raw_gradient` (available as `s_homeo[6u]`, set in `coop_habituate_homeo` at `brain_passes.wgsl:937`). Online gradient descent with a small learning rate:

```wgsl
    // Train predictor: previous tick's prediction vs this tick's actual gradient.
    // The predictor learns to anticipate homeostatic outcomes from the predicted
    // state — a generative model of the agent's own homeostatic dynamics.
    if (tid == 0u) {
        let prev_pred = brain_state[brain_base + O_PREV_HOMEO_PREDICTION];
        let actual = s_homeo[6u];  // raw_gradient from coop_habituate_homeo
        let pred_error = prev_pred - actual;
        let pred_lr = bc_f32(CFG_HOMEO_PREDICTOR_LEARNING_RATE);
        // Update bias
        brain_state[brain_base + O_HOMEO_PREDICTOR_BIAS] -= pred_lr * pred_error;
        // Store current prediction for next tick's training
        brain_state[brain_base + O_PREV_HOMEO_PREDICTION] = predicted_gradient;
    }
    workgroupBarrier();

    // Update weights: each thread updates its stride over PREDICTOR_DIMENSION
    if (tid < PREDICTOR_DIMENSION) {
        let pred_error = brain_state[brain_base + O_PREV_HOMEO_PREDICTION]
            - s_homeo[6u];  // re-read after barrier
        let pred_lr = bc_f32(CFG_HOMEO_PREDICTOR_LEARNING_RATE);
        var w = brain_state[brain_base + O_HOMEO_PREDICTOR_WEIGHTS + tid]
            - pred_lr * pred_error * s_prediction[tid];
        w = clamp(w, -MAX_WEIGHT_NORM, MAX_WEIGHT_NORM);
        brain_state[brain_base + O_HOMEO_PREDICTOR_WEIGHTS + tid] = w;
    }
    workgroupBarrier();
```

Wait — there's a subtlety. The `pred_error` for the weight update needs to be the same value used for the bias update. But after the barrier, thread 0 has already updated `O_PREV_HOMEO_PREDICTION` to the *new* prediction. The weight update threads need the *old* prediction error. Let me restructure: thread 0 computes `pred_error` and stores it in shared memory (e.g., `s_pred_td[0]` is already used for prediction error, but we can use a new shared slot or repurpose one). Actually, `s_pred_td[0]` is `S_PRED_ERROR` (the forward model's prediction error). We can use `s_homeo` slot 6 for the raw gradient and add a new shared slot for the predictor error. Or simpler: thread 0 writes `pred_error` to a shared scalar before the barrier, and the weight-update threads read it after.

Let me use a simpler approach: thread 0 computes everything (prediction, error, bias update, prev-prediction store) and writes `pred_error` to a shared variable. Then the weight update threads read that shared variable.

Actually, looking at the existing code more carefully, `s_pred_td` is a 2-element array where `S_PRED_ERROR = 0` and `S_TD_ERROR = 1`. I can add a third element or use a separate shared variable. Let me add `s_homeo_pred_error` as a new workgroup variable. But that adds a threadgroup binding — the Metal ceiling issue. Let me just repurpose an existing slot. `s_explore` is 2 elements and is only used in the thread-0 motor block and the eligibility trace update. The predictor block runs before the motor block, so I can use `s_explore[0]` temporarily and overwrite it later. Or better: use `s_atten_sum` which is computed earlier and consumed by thread 0 in the exploration block — but the predictor block runs between the attenuation sum reduction and the thread 0 exploration block. Actually, `s_atten_sum` is set by thread 0 after the reduction at line 1394, and the exploration block reads it at line 1437. The predictor block would run between them. So I can't safely reuse `s_atten_sum`.

Simplest approach: add one more element to `s_pred_td` (making it 3 elements). The comment says it was packed to save a slot, but adding one more element to an existing array doesn't consume a new binding — it just grows the existing one.

Actually wait, let me re-read the comment at line 38-46:
```
// Prediction error (index 0) and TD error (index 1) share one threadgroup
// binding: macOS Metal caps the number of distinct threadgroup resource slots
// and the fused kernel is at that ceiling, so packing this scalar pair into one
// array frees a slot for the visual-cortex scratch (`s_visual`) without
// changing any value.
```

So `s_pred_td` was created to save a binding slot. Adding a third element to the same array doesn't consume a new slot — it's still one binding. So I can add `S_HOMEO_PRED_ERROR = 2u`:

```wgsl
const S_PRED_ERROR: u32 = 0u;
const S_TD_ERROR: u32 = 1u;
const S_HOMEO_PRED_ERROR: u32 = 2u;
var<workgroup> s_pred_td: array<f32, 3>;
```

Then thread 0 writes `pred_error` to `s_pred_td[S_HOMEO_PRED_ERROR]`, and the weight-update threads read it after the barrier.

OK let me rewrite the architecture more cleanly. Let me also think about the β term. The predicted gradient is added to the TD reward:

```
reward = raw_gradient_amplified + β * prev_predicted_gradient
```

Where `prev_predicted_gradient` is the prediction made at the *previous* tick (stored in `O_PREV_HOMEO_PREDICTION`). This is the agent's anticipation of the current tick's gradient, made before the outcome was known.

The β parameter should be a brain_config constant, defaulting to something like 0.3. It scales the anticipatory credit relative to the actual outcome.

Actually, I realize there's a cleaner formulation. Instead of modifying the TD reward, we can add the predicted gradient as a separate term in the action weight update. But that's getting complex. Let me keep it simple: add to the TD reward.

Let me also think about the predictor learning rate. It should be a brain_config constant. Default maybe 0.01 (same as the critic learning rate).

OK, let me finalize the architecture and write it up properly. I'll also need to handle the `s_pred_td` array size change and the new shared variable.

Let me also think about where exactly in `coop_predict_and_act` to insert the predictor block. The flow is:

1. Predictor train + predict (lines 1068-1107) → s_prediction populated
2. Recall similarity precompute (lines 1109-1120)
3. Context blend + tanh (lines 1122-1143) → s_prediction finalized
4. Prediction error reduction (lines 1145-1160)
5. Thread 0: prediction error + error ring (lines 1163-1181)
6. TD credit: value head + eligibility traces (lines 1184-1257)
7. Weight normalization (lines 1260-1351)
8. Policy dot products (lines 1353-1381)
9. Attenuation sum (lines 1383-1396)
10. Thread 0: exploration, noise, motor (lines 1398-1564)
11. Eligibility trace update (lines 1579-1604)

The predictor block should go between step 5 (prediction error ring) and step 6 (TD credit), because:
- s_prediction is finalized (after context blend + tanh)
- The predicted gradient needs to be available before the TD reward is computed
- The actual raw_gradient (s_homeo[6u]) is available from coop_habituate_homeo

Let me write the architecture now.</think>

<｜DSML｜tool_calls>
<｜DSML｜invoke name="write">
<｜DSML｜parameter name="content" string="true"># Architecture — Plan 0023 (deltas)

> Edits in `crates/xagent-brain/src/shaders/kernel/common.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`,
> `crates/xagent-brain/src/buffers.rs`,
> `crates/xagent-shared/src/config.rs`,
> `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-sandbox/tests/integration.rs`, and
> `docs/plans/STATUS.md`.
> Line numbers are hints; locate by symbol (grep for `coop_predict_and_act`,
> `O_VALUE_WEIGHTS`, `O_ORIENTATION_OFFSET`, `raw_gradient`, `s_homeo`,
> `learning_probe_mirrored_steering_is_chance`).

## 0001 — Homeostatic Gradient Predictor

Today the forward model in `coop_predict_and_act()` (`brain_passes.wgsl:1068-1107`) predicts the next encoded state `s_prediction[dim]` from the current `s_encoded` via a 128→128 dense layer, trained online on transition error (`brain_passes.wgsl:1079-1088`). The predicted state feeds context blending and the TD critic's value estimate. But the predictor never learns to anticipate the *homeostatic gradient* — the scalar `raw_gradient = energy_delta × 0.6 + integrity_delta × 0.4` (`brain_passes.wgsl:913-915`) that drives all credit assignment. The TD error `δ = reward + γ·V(s′) − V(s)` (`brain_passes.wgsl:1222-1224`) uses only the *actual* gradient as reward, which arrives after the outcome — creating the temporal gap that has resisted every mechanism tried (0017 trace decay, 0018 auxiliary loss/frame-sync/gradient-shaping, 0020 direct supervision).

### Homeostatic gradient prediction head

Add a 128→1 linear head on top of `s_prediction` that predicts the homeostatic gradient. The predicted gradient provides an *anticipatory* credit signal at decision time — before the actual outcome — bridging the temporal gap without introducing any external target. The prediction target is the same `raw_gradient` that already drives the TD path; the predictor learns from the agent's own experience what visual/action patterns precede homeostatic changes.

Edits:

- **Add three new brain state slots** (`common.wgsl`, between `O_ORIENTATION_OFFSET` and `O_VALUE_WEIGHTS`, ≈248-255). The homeostatic predictor weights (128 floats), bias (1 float), and previous prediction (1 float, episodic — zeroed on death) are inserted into the brain state tail. All downstream offsets (`O_VALUE_WEIGHTS` through `BRAIN_STRIDE`) shift by `ENCODED_DIMENSION + 2`:

```wgsl
// ── Homeostatic gradient predictor (plan 0023) ─────────────────────────────
// Linear head (128→1) on top of the forward model's predicted state s_prediction.
// Trained online to predict raw_gradient; the previous tick's prediction provides
// an anticipatory credit signal that bridges the ~10-tick sensory latency.
// Weights are heritable (seeded at birth, inherited, mutated); the prev-prediction
// slot is episodic (zeroed on death), like O_PREV_VALUE.
override O_HOMEO_PREDICTOR_WEIGHTS: u32 = O_ORIENTATION_OFFSET + 1u;
override O_HOMEO_PREDICTOR_BIAS: u32 = O_HOMEO_PREDICTOR_WEIGHTS + ENCODED_DIMENSION;
override O_PREV_HOMEO_PREDICTION: u32 = O_HOMEO_PREDICTOR_BIAS + 1u;

override O_VALUE_WEIGHTS: u32 = O_PREV_HOMEO_PREDICTION + 1u;
```

- **Mirror the new offsets in Rust** (`buffers.rs`, ≈104-114). Insert the three new constants between `O_ORIENTATION_OFFSET` and `O_VALUE_WEIGHTS`, matching the WGSL layout exactly:

```rust
// ── Homeostatic gradient predictor (plan 0023) ─────────────────────────
pub const O_HOMEO_PREDICTOR_WEIGHTS: usize = O_ORIENTATION_OFFSET + 1;
pub const O_HOMEO_PREDICTOR_BIAS: usize = O_HOMEO_PREDICTOR_WEIGHTS + ENCODED_DIMENSION;
pub const O_PREV_HOMEO_PREDICTION: usize = O_HOMEO_PREDICTOR_BIAS + 1;

pub const O_VALUE_WEIGHTS: usize = O_PREV_HOMEO_PREDICTION + 1;
```

- **Update `FIXED_TAIL_SIZE`** (`buffers.rs:126`). The tail grows by `ENCODED_DIMENSION + 2` (130 floats). Update the constant and verify `BRAIN_STRIDE` is correct (it derives from the shifted `O_TRACE_BIASES`).

- **Initialize predictor weights at brain birth** (`buffers.rs`, `init_brain_state_for`). Xavier-uniform initialization: each weight drawn from `U(-sqrt(6/128), sqrt(6/128)) ≈ U(-0.216, 0.216)`. Bias initialized to zero. `O_PREV_HOMEO_PREDICTION` initialized to zero.

- **Zero `O_PREV_HOMEO_PREDICTION` on death** (`kernel_tick.wgsl`, `agent_death_respawn`, ≈638-651). Add a line zeroing `brain_state[brain_base + O_PREV_HOMEO_PREDICTION]` alongside the existing episodic resets (`O_PREV_VALUE`, traces, homeo state). The predictor *weights* survive death (they are learned knowledge); only the previous-prediction slot is episodic.

- **Add the `homeo_predictive_credit_enabled` flag to `BrainConfig`** (`config.rs`, the `BrainConfig` struct). Default-off, zero-cost when false:

```rust
/// Enable the homeostatic gradient predictor: a 128→1 linear head on the
/// forward model that learns to anticipate raw_gradient. The predicted
/// gradient provides an anticipatory credit signal (β-scaled term in the
/// TD reward), bridging sensory latency without external targets.
/// Weights are heritable. Zero-cost when false.
pub homeo_predictive_credit_enabled: bool,
```

- **Add the predictor learning rate and β blend to `BrainConfig`** (`config.rs`):

```rust
/// Learning rate for the homeostatic gradient predictor's online gradient
/// descent. Trained on every brain tick against the actual raw_gradient.
pub homeo_predictor_learning_rate: f32,
/// Blend weight for the predicted gradient in the TD reward.
/// reward = raw_gradient_amplified + β * prev_predicted_gradient.
/// 0.0 = disabled; 0.3 = anticipatory credit at ~30% weight.
pub homeo_predictive_credit_beta: f32,
```

- **Bind the flags into the wconfig uniform** (`gpu_kernel.rs`, `write_agent_heritable_config()`). Pack `homeo_predictive_credit_enabled` as a u32 flag, `homeo_predictor_learning_rate` as f32, and `homeo_predictive_credit_beta` as f32 at documented offsets in the wconfig buffer. Add matching `CFG_HOMEO_PREDICTIVE_CREDIT_ENABLED_OFFSET`, `CFG_HOMEO_PREDICTOR_LEARNING_RATE_OFFSET`, and `CFG_HOMEO_PREDICTIVE_CREDIT_BETA_OFFSET` constants in `common.wgsl`.

### Predicted gradient computation and training

In `coop_predict_and_act()` (`brain_passes.wgsl`), after the context-blended prediction is finalized (after the tanh at ≈1141 and the `workgroupBarrier()` at ≈1143) and before the TD credit block (≈1184):

- **Grow `s_pred_td` to 3 elements** (`brain_passes.wgsl`, ≈44-46). Add a third slot for the predictor error, keeping the single-binding packing:

```wgsl
const S_PRED_ERROR: u32 = 0u;
const S_TD_ERROR: u32 = 1u;
const S_HOMEO_PRED_ERROR: u32 = 2u;
var<workgroup> s_pred_td: array<f32, 3>;
```

- **Insert the predictor block** between the prediction error ring (≈1181) and the TD credit block (≈1184). The block has three sub-steps separated by barriers:

```wgsl
    // ── Homeostatic gradient predictor (plan 0023) ─────────────────────────
    // Predict raw_gradient from the forward model's predicted state, train
    // the predictor online, and blend the previous tick's prediction into
    // the TD reward as an anticipatory credit signal. Gated on the flag;
    // when disabled the predicted gradient is zero and the block is a no-op.
    let homeo_pred_enabled = bc_f32(CFG_HOMEO_PREDICTIVE_CREDIT_ENABLED) != 0.0;

    // Sub-step 1: parallel dot product s_prediction · homeo_predictor_weights
    if (homeo_pred_enabled) {
        if (tid < PREDICTOR_DIMENSION) {
            s_dense_partials[tid] = s_prediction[tid]
                * brain_state[brain_base + O_HOMEO_PREDICTOR_WEIGHTS + tid];
        } else {
            s_dense_partials[tid] = 0.0;
        }
    } else {
        // When disabled, zero the partials so the reduction is a no-op
        // and predicted_gradient stays 0.0.
        s_dense_partials[tid] = 0.0;
    }
    workgroupBarrier();
    wg_reduce_dense(tid);
    // Sub-step 2: thread 0 computes prediction, trains bias, stores prev
    if (tid == 0u) {
        var predicted_gradient: f32 = 0.0;
        if (homeo_pred_enabled) {
            predicted_gradient = s_dense_partials[0]
                + brain_state[brain_base + O_HOMEO_PREDICTOR_BIAS];
            // Train: previous tick's prediction vs this tick's actual gradient
            let prev_pred = brain_state[brain_base + O_PREV_HOMEO_PREDICTION];
            let actual = s_homeo[6u];  // raw_gradient from coop_habituate_homeo
            let pred_error = prev_pred - actual;
            let pred_lr = bc_f32(CFG_HOMEO_PREDICTOR_LEARNING_RATE);
            brain_state[brain_base + O_HOMEO_PREDICTOR_BIAS] -= pred_lr * pred_error;
            // Store current prediction for next tick's training
            brain_state[brain_base + O_PREV_HOMEO_PREDICTION] = predicted_gradient;
            // Publish pred_error for the weight-update threads
            s_pred_td[S_HOMEO_PRED_ERROR] = pred_error;
        } else {
            s_pred_td[S_HOMEO_PRED_ERROR] = 0.0;
        }
        // Publish the predicted gradient for the TD reward blend below.
        // s_homeo[6u] is raw_gradient; we reuse s_homeo slot 6 to also
        // carry the predicted gradient by writing it to a new shared slot.
        // Use s_atten_sum (already consumed by thread 0 above, free here).
        s_atten_sum = predicted_gradient;
    }
    workgroupBarrier();
    // Sub-step 3: update predictor weights (all threads, stride over PREDICTOR_DIMENSION)
    if (homeo_pred_enabled && tid < PREDICTOR_DIMENSION) {
        let pred_error = s_pred_td[S_HOMEO_PRED_ERROR];
        let pred_lr = bc_f32(CFG_HOMEO_PREDICTOR_LEARNING_RATE);
        var w = brain_state[brain_base + O_HOMEO_PREDICTOR_WEIGHTS + tid]
            - pred_lr * pred_error * s_prediction[tid];
        w = clamp(w, -MAX_WEIGHT_NORM, MAX_WEIGHT_NORM);
        brain_state[brain_base + O_HOMEO_PREDICTOR_WEIGHTS + tid] = w;
    }
    workgroupBarrier();
```

- **Blend the predicted gradient into the TD reward** (`brain_passes.wgsl`, the TD error computation at ≈1220-1224). After the existing `let reward = s_homeo[1u]` (the urgency-amplified gradient), add the β-scaled predicted gradient:

```wgsl
            // ── Homeostatic predictive credit blend (plan 0023) ──────────
            // The previous tick's predicted gradient provides anticipatory
            // credit: the agent is rewarded for decisions its own learned
            // model predicts will improve homeostasis. β scales the blend;
            // 0.0 = disabled (pure homeostatic TD), 0.3 = moderate anticipation.
            let predictive_bonus = s_atten_sum * bc_f32(CFG_HOMEO_PREDICTIVE_CREDIT_BETA);
            let reward = s_homeo[1u] + predictive_bonus;
```

Note: `s_atten_sum` is repurposed here to carry the predicted gradient from thread 0 to the TD credit block. The attenuation sum reduction (≈1383-1396) runs *after* the TD credit block, so it overwrites `s_atten_sum` with its own value before the exploration block reads it. This is safe: the TD credit block consumes `s_atten_sum` (as predicted_gradient) before the attenuation sum reduction overwrites it.

Properties that make this safe:
- The predictor head is a linear 128→1 layer — the same pattern as the existing value head (`O_VALUE_WEIGHTS`). No new reduction primitive, no new buffer, no new dispatch.
- The prediction target is `raw_gradient` (`s_homeo[6u]`) — the same homeostatic signal that drives the TD path. No food bearing, no external target, no privileged geometry. The predictor learns from the agent's own experience.
- The β blend adds the predicted gradient to the TD reward, staying within the existing TD(λ) framework. The critic, actor, and encoder credit paths are unchanged — only the reward term is enriched with the agent's own learned anticipation.
- The flag is default-off, so production and every existing test run the untouched TD-only path. The predictor weights are initialized at brain birth and inherited/evolved — they are part of the agent's learned world model, not a separate mechanism.
- The `s_pred_td` array grows from 2 to 3 elements — still one threadgroup binding, no Metal ceiling impact. `s_atten_sum` is safely repurposed because the attenuation sum reduction runs after the TD credit block consumes it.
- Predictor weights are L2-clamped at `MAX_WEIGHT_NORM` (2.0), same as all other weight vectors. The learning rate is configurable and defaults to a small value (0.01).

## 0002 — Predictive-Credit Steering Probe

Today the mirrored-steering baseline (`learning_probe_mirrored_steering_is_chance()`, `integration.rs:2848-2929`) trains agents for 120 episodes on alternating left/right food with dense TD strides, then pins movement and scores turn-alignment. Baseline: 0.489 (chance band [0.38, 0.62]). The 0020 direct-supervision result (0.841) is the upper-bound reference.

Edits:

- **Add a GPU-gated probe** (`integration.rs`) named `homeo_predictive_credit_steering_probe`. Same protocol as the mirrored-steering baseline but with `homeo_predictive_credit_enabled: true`, `homeo_predictor_learning_rate: 0.01`, `homeo_predictive_credit_beta: 0.3`. Embeds the standard self-skip guard. Trains for 120 episodes, evaluates with pinned movement, scores turn-alignment with `score_turn_alignment()`, computes 95% Clopper–Pearson CI, and prints the result. The test does NOT assert the steering verdict — a chance-band outcome is a valid scientific result. Only control probes (encoder separability, food consumption) are asserted.

- **Add a predictor-accuracy diagnostic** within the probe: after training, run one additional episode and record the mean absolute prediction error `|prev_predicted_gradient − actual_raw_gradient|` across all brain ticks. This confirms the predictor is actually learning (error decreasing) vs just outputting zero. Print the diagnostic alongside the steering result.

Properties that make this safe:
- The probe is GPU-gated with the standard self-skip guard.
- The measurement is structurally identical to the 0018/0020 baseline — same strides, same evaluation geometry, same `score_turn_alignment()` — so the only changed variable is the predictive credit mechanism.
- The verdict is mechanical (95% Clopper–Pearson CI) and binary.
- The predictor-accuracy diagnostic provides a secondary signal: if steering stays at chance but prediction error is also high (predictor didn't learn), the mechanism wasn't fairly tested. If prediction error is low (predictor learned) but steering stays at chance, the mechanism is genuinely insufficient.

## 0003 — Decision Gate

Today the credit-path bottleneck carries forward from plan to plan with each mechanism recording REJECT. The decision gate for this plan is identical to 0018/0020: binary ACCEPT/REJECT based on 95% CI on the mirrored-steering probe.

Edits:

- **Author the decision doc** `0001-HOMEO-PREDICTOR-DECISION.md` in the plan folder. If ACCEPT (CI lower ≥ 0.70): record the measured alignment, confirm control probes hold, and trigger the integration task. If REJECT (CI upper ≤ 0.62): record the negative result, the predictor-accuracy diagnostic, and structural candidates for the next plan. If inconclusive (CI straddles [0.62, 0.70]): recommend re-run with larger sample.

- **If ACCEPT:** the mechanism stays behind its default-off flag. Update the steering baseline assertion to reflect the new band. No default flip — graduation is a separate decision.

- **If REJECT:** record the negative result. The structural candidates from 0020's decision doc (n-step returns, eligibility-decay rework, eligibility-reset at vision boundary, auxiliary-head through shared encoder) remain the fallback hypotheses. The predictor-accuracy diagnostic informs which candidate to try next: if the predictor learned (low error) but steering didn't improve, the bottleneck is in how the predicted gradient is used (β blend, TD integration). If the predictor didn't learn (high error), the bottleneck is in the prediction architecture itself (need recurrence, longer horizon, or different training signal).

Properties that make this safe:
- The decision is mechanical and binary — no interpretation gap.
- The predictor-accuracy diagnostic prevents a false REJECT where the mechanism was never fairly tested.
- The structural candidates are already enumerated in 0020's decision doc — this plan either accepts (unlocks steering) or sharpens the diagnosis for the next attempt.
