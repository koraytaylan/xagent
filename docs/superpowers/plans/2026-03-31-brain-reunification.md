# Brain Reunification: Trainable Encoder, Continuous Motor, Neuroevolution

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconnect the action selector to the brain's representational space, replace discrete actions with continuous motor output, and let evolution mutate action weights directly.

**Architecture:** The action selector moves back from raw 201-dim features to the encoded repr_dim space (default 32). The encoder becomes trainable again — credit from action selection backpropagates to encoder weights. Continuous motor output (forward + turn as tanh of dot products) replaces the 8 discrete actions. Evolution mutates inherited action weights alongside BrainConfig.

**Tech Stack:** Rust, xagent-brain crate (encoder.rs, action.rs, brain.rs), xagent-sandbox crate (governor.rs, main.rs, headless.rs, agent/mod.rs)

**Reference:** See `/EVOLUTION_JOURNEY.md` for full context on why each change is needed.

---

## File Structure

| File | Responsibility | Changes |
|------|---------------|---------|
| `crates/xagent-brain/src/action.rs` | Action selector → continuous motor | Rewrite: 2 continuous outputs (forward, turn) over encoded state; credit assignment in repr_dim space; remove discrete action machinery |
| `crates/xagent-brain/src/encoder.rs` | Sensory encoding | Trainable: adapt() implements gradient-like update from action credit; no longer frozen |
| `crates/xagent-brain/src/brain.rs` | Coordination | Remove raw_features plumbing; LearnedState includes encoder weights again; update tick_inner |
| `crates/xagent-brain/src/homeostasis.rs` | No changes | — |
| `crates/xagent-brain/src/predictor.rs` | No changes | — |
| `crates/xagent-sandbox/src/governor.rs` | Evolution | breed_next_generation returns weight perturbations alongside configs |
| `crates/xagent-sandbox/src/agent/mod.rs` | Mutation | New function: mutate_action_weights for neuroevolution |
| `crates/xagent-sandbox/src/main.rs` | Integration | Apply weight perturbations during spawn; update inheritance to include encoder |
| `crates/xagent-sandbox/src/headless.rs` | Integration | Same as main.rs |

---

## Task 1: Continuous Motor Output (Change C)

This goes first because it simplifies everything downstream — the action selector becomes 2 outputs instead of 8, reducing the weight space from 1,608 to 2 × repr_dim.

**Files:**
- Modify: `crates/xagent-brain/src/action.rs` (full rewrite of core)
- Modify: `crates/xagent-brain/src/brain.rs` (remove raw_features plumbing)
- Test: existing tests in `action.rs` updated

### Concept

Replace the 8 discrete actions with 2 continuous motor channels:

```
forward = tanh(dot(forward_weights, encoded_state) + forward_bias)
turn    = tanh(dot(turn_weights, encoded_state) + turn_bias)
```

- `forward_weights`: repr_dim floats (default 32)
- `turn_weights`: repr_dim floats (default 32)
- Total policy: 2 × repr_dim + 2 biases = 66 weights (was 1,608)
- Output range: [-1, 1] for both (already what MotorCommand expects)
- Exploration: additive Gaussian noise scaled by exploration_rate

Credit assignment records (encoded_state, motor_command, gradient) tuples. When the gradient improves, the motor command is decomposed back to update the forward/turn weight vectors.

- [ ] **Step 1: Define new ActionSelector struct**

Replace the 8-action machinery in `action.rs` with:

```rust
const MOTOR_CHANNELS: usize = 2; // forward, turn

pub struct ActionSelector {
    repr_dim: usize,
    // Policy weights: 2 channels × repr_dim
    forward_weights: Vec<f32>,
    turn_weights: Vec<f32>,
    forward_bias: f32,
    turn_bias: f32,
    // History ring for credit assignment
    action_ring: Vec<MotorRecord>,
    state_ring: Vec<f32>,  // ACTION_HISTORY_LEN × repr_dim
    history_len: usize,
    history_cursor: usize,
    // Exploration
    exploration_rate: f32,
    tick: u64,
    // Telemetry
    total_actions: u64,
    exploitative_actions: u64,
    current_state: Vec<f32>, // repr_dim
}

struct MotorRecord {
    forward_output: f32,  // what the policy produced (before noise)
    turn_output: f32,
    tick: u64,
    state_offset: usize,
    gradient_at_action: f32,
}
```

- [ ] **Step 2: Implement select() for continuous output**

```rust
pub fn select(
    &mut self,
    current: &EncodedState,
    prediction: &EncodedState,
    recalled: &[(EncodedState, f32)],
    homeostatic_gradient: f32,
    prediction_error: f32,
    urgency: f32,
) -> MotorCommand {
    self.tick += 1;
    let mut rng = rand::rng();
    let dim = self.repr_dim.min(current.len());

    // Snapshot state for credit
    self.current_state[..dim].copy_from_slice(&current.data()[..dim]);

    // Credit assignment
    self.assign_credit(homeostatic_gradient);

    // Exploration rate
    let stability = recalled.len() as f32 / 16.0;
    let novelty_bonus = (prediction_error * 2.0).min(0.4);
    let urgency_penalty = (urgency * 0.4).min(0.5);
    self.exploration_rate =
        (0.15 - stability * 0.1 + novelty_bonus - urgency_penalty).clamp(0.05, 0.85);

    // Compute continuous motor output
    let mut fwd = self.forward_bias;
    let mut trn = self.turn_bias;
    for i in 0..dim {
        fwd += self.forward_weights[i] * current.data()[i];
        trn += self.turn_weights[i] * current.data()[i];
    }

    // Prospective evaluation
    let confidence = 1.0 - prediction_error.clamp(0.0, 1.0);
    if confidence > 0.1 {
        let mut fwd_future = self.forward_bias;
        let mut trn_future = self.turn_bias;
        for i in 0..dim {
            fwd_future += self.forward_weights[i] * prediction.data()[i];
            trn_future += self.turn_weights[i] * prediction.data()[i];
        }
        fwd += confidence * 0.5 * (fwd_future - fwd);
        trn += confidence * 0.5 * (trn_future - trn);
    }

    let fwd_clean = fast_tanh(fwd);
    let trn_clean = fast_tanh(trn);

    // Additive noise exploration (not full replacement)
    let noise_scale = self.exploration_rate;
    let fwd_noisy = (fwd_clean + rng.random_range(-1.0..1.0) * noise_scale).clamp(-1.0, 1.0);
    let trn_noisy = (trn_clean + rng.random_range(-1.0..1.0) * noise_scale).clamp(-1.0, 1.0);

    let is_exploitative = noise_scale < 0.1;
    self.total_actions += 1;
    if is_exploitative { self.exploitative_actions += 1; }

    // Record clean output for credit assignment
    self.record_action(fwd_clean, trn_clean, current, homeostatic_gradient);

    MotorCommand {
        forward: fwd_noisy,
        turn: trn_noisy,
        strafe: 0.0,
        action: None,
    }
}
```

- [ ] **Step 3: Implement continuous credit assignment**

```rust
fn assign_credit(&mut self, gradient: f32) {
    let dim = self.repr_dim;
    let now = self.tick;

    let n = self.history_len.min(ACTION_HISTORY_LEN);
    for i in 0..n {
        let rec = &self.action_ring[i];
        let age = now.saturating_sub(rec.tick) as f32;
        let temporal = (-age * CREDIT_DECAY_RATE).exp();
        if temporal < 0.01 { continue; }

        let improvement = gradient - rec.gradient_at_action;
        if improvement.abs() < CREDIT_DEADZONE { continue; }

        let effective = if improvement < 0.0 {
            improvement * PAIN_AMPLIFIER
        } else {
            improvement
        };

        let credit = effective * temporal;
        let s_offset = rec.state_offset;

        // Update forward weights proportional to how much forward was commanded
        // and turn weights proportional to how much turn was commanded.
        // This naturally decomposes: "I went forward and things improved"
        // → strengthen forward weights for this state pattern.
        for d in 0..dim {
            let feat = self.state_ring[s_offset + d];
            self.forward_weights[d] += WEIGHT_LR * credit * rec.forward_output * feat;
            self.turn_weights[d] += WEIGHT_LR * credit * rec.turn_output * feat;
        }
        self.forward_bias += WEIGHT_LR * credit * rec.forward_output * 0.1;
        self.turn_bias += WEIGHT_LR * credit * rec.turn_output * 0.1;
    }

    self.normalize_weights();
}
```

- [ ] **Step 4: Implement export/import for continuous weights**

```rust
pub fn export_weights(&self) -> Vec<f32> {
    let mut w = self.forward_weights.clone();
    w.extend_from_slice(&self.turn_weights);
    w.push(self.forward_bias);
    w.push(self.turn_bias);
    w
}

pub fn import_weights(&mut self, weights: &[f32]) {
    let dim = self.repr_dim;
    let expected = dim * 2 + 2;
    if weights.len() != expected { return; }
    self.forward_weights.copy_from_slice(&weights[..dim]);
    self.turn_weights.copy_from_slice(&weights[dim..dim*2]);
    self.forward_bias = weights[dim*2];
    self.turn_bias = weights[dim*2 + 1];
}
```

- [ ] **Step 5: Remove raw_features plumbing from brain.rs**

In `tick()` and `tick_inner()`, remove `raw_features` extraction and passing. The action selector now receives only the `EncodedState`.

Update `select()` call in `tick_inner` to drop the `raw_features` parameter.

- [ ] **Step 6: Update all tests**

Rewrite action selector tests for continuous output. Key tests:
- `positive_gradient_increases_forward_weight`: gradient transition → forward weights increase
- `continuous_output_is_bounded`: output always in [-1, 1]
- `exploration_adds_noise`: with high exploration_rate, output varies from clean policy
- `credit_decomposes_to_motor_channels`: forward credit goes to forward weights, turn credit to turn weights
- `death_signal_clears_history`: history cleared on death

- [ ] **Step 7: Run tests, fix, commit**

```bash
cargo test
git add crates/xagent-brain/
git commit -m "feat: continuous motor output replaces 8 discrete actions"
```

---

## Task 2: Trainable Encoder (Change A)

With the action selector back in repr_dim space, the encoder needs to learn useful representations.

**Files:**
- Modify: `crates/xagent-brain/src/encoder.rs`
- Modify: `crates/xagent-brain/src/brain.rs` (LearnedState includes encoder again)

- [ ] **Step 1: Re-enable encoder adaptation**

In `encoder.rs`, replace the no-op `adapt()` with a credit-driven update. The brain should call this with the action selector's credit signal so the encoder learns to produce features the policy cares about.

```rust
/// Adapt encoder weights based on the action selector's recent credit.
/// When the action selector learns "dimension 5 in encoded state correlates
/// with positive outcomes for forward", the encoder should amplify the raw
/// features that contribute to dimension 5.
///
/// Simple Hebbian rule: Δw[i,j] += lr * credit_signal[i] * input_feature[j]
/// where credit_signal approximates which encoded dimensions were useful.
pub fn adapt(&mut self, credit_signal: &[f32], learning_rate: f32) {
    if !self.initialized || credit_signal.len() != self.representation_dim {
        return;
    }
    let fc = self.feature_count;
    for i in 0..self.representation_dim {
        if credit_signal[i].abs() < 1e-6 { continue; }
        let row_base = i * fc;
        let update = learning_rate * credit_signal[i] * 0.001; // small encoder LR
        for j in 0..fc {
            self.weights[row_base + j] += update * self.feature_scratch[j];
            self.weights[row_base + j] = self.weights[row_base + j].clamp(-2.0, 2.0);
        }
    }
}
```

- [ ] **Step 2: Generate credit signal in action selector**

Add a method to ActionSelector that returns which encoded dimensions received the most recent credit:

```rust
pub fn last_credit_signal(&self) -> Vec<f32> {
    // Return the sum of recent credit × state_feature per dimension
    // (computed during assign_credit and cached)
    self.cached_credit_signal.clone()
}
```

Populate `cached_credit_signal` during `assign_credit()`.

- [ ] **Step 3: Wire brain.rs to pass credit to encoder**

In `tick_inner()`, after action selection:
```rust
let credit_signal = self.action_selector.last_credit_signal();
self.encoder.adapt(&credit_signal, modulated_lr);
```

- [ ] **Step 4: Keep encoder deterministic for initialization but allow adaptation**

Keep the seeded RNG for initial weights (same config → same starting point). But allow `adapt()` to modify weights during lifetime. The inherited weights capture both the initial projection AND the learned adaptations.

- [ ] **Step 5: Re-add encoder to LearnedState**

Update `export_learned_state` and `import_learned_state` in `brain.rs`:
```rust
pub struct LearnedState {
    pub encoder_weights: Vec<f32>,
    pub encoder_biases: Vec<f32>,
    pub action_forward_weights: Vec<f32>,  // repr_dim + 1 (with bias)
    pub action_turn_weights: Vec<f32>,     // repr_dim + 1
    pub predictor_weights: Vec<f32>,
    pub predictor_context_weight: f32,
}
```

- [ ] **Step 6: Update tests, run, commit**

```bash
cargo test
git add crates/xagent-brain/
git commit -m "feat: trainable encoder driven by action credit signal"
```

---

## Task 3: Neuroevolution of Action Weights (Change B)

**Files:**
- Modify: `crates/xagent-sandbox/src/agent/mod.rs` (new weight mutation function)
- Modify: `crates/xagent-sandbox/src/main.rs` (apply weight perturbations)
- Modify: `crates/xagent-sandbox/src/headless.rs` (same)

- [ ] **Step 1: Add weight mutation function**

In `agent/mod.rs`:

```rust
/// Perturb inherited action weights for neuroevolution.
/// Mutates a random 10% of weights by ±(strength × weight_magnitude).
/// This lets evolution explore behavioral variations that within-lifetime
/// learning might miss.
pub fn mutate_learned_state(state: &LearnedState, strength: f32) -> LearnedState {
    let mut rng = rand::rng();
    let mut result = state.clone();

    // Perturb action weights (forward + turn)
    for w in result.action_forward_weights.iter_mut()
        .chain(result.action_turn_weights.iter_mut())
    {
        if rng.random::<f32>() < 0.1 { // 10% of weights
            let perturbation = rng.random_range(-strength..strength);
            *w += perturbation;
        }
    }

    // Perturb encoder weights more conservatively (1% of weights, smaller range)
    for w in result.encoder_weights.iter_mut() {
        if rng.random::<f32>() < 0.01 {
            let perturbation = rng.random_range(-strength * 0.1..strength * 0.1);
            *w += perturbation;
        }
    }

    result
}
```

- [ ] **Step 2: Apply weight perturbations during population spawn**

In `main.rs` and `headless.rs`, after inheriting weights:

```rust
if let Some(ref state) = inherited_state {
    let repeats = self.governor_config.eval_repeats.max(1);
    for (i, agent) in self.agents.iter_mut().enumerate() {
        if i < repeats {
            // Champion: exact inherited weights (no perturbation)
            agent.brain.import_learned_state(state);
        } else {
            // Mutant: inherited weights + small perturbation
            let mutated = mutate_learned_state(state, effective_mutation_strength);
            agent.brain.import_learned_state(&mutated);
        }
    }
}
```

- [ ] **Step 3: Update governor to expose mutation strength**

The governor already computes `effective_strength` in `breed_next_generation()`. Expose it via the `AdvanceResult`:

```rust
pub enum AdvanceResult {
    Continue {
        configs: Vec<BrainConfig>,
        messages: Vec<String>,
        mutation_strength: f32,  // for weight perturbation
    },
    Finished { messages: Vec<String> },
}
```

- [ ] **Step 4: Test and commit**

```bash
cargo test
git add crates/xagent-brain/ crates/xagent-sandbox/
git commit -m "feat: neuroevolution of action weights alongside BrainConfig"
```

---

## Task 4: Integration Testing

- [ ] **Step 1: Run headless 10 generations, verify fitness trend**

```bash
rm -f /tmp/test_unified.db
./target/release/xagent --no-render --generations 10 --db /tmp/test_unified.db
```

Verify:
- Avg fitness increases over generations
- Death count per agent decreases over generations
- Food consumed per agent increases over generations
- Different BrainConfig parameters now produce different fitness scores

- [ ] **Step 2: Compare min vs max brain configs**

Run two sessions: one with tiny config (memory=24, dim=16) and one with large (memory=512, dim=64). Fitness should now differ because the encoder/action selector share the representational space.

- [ ] **Step 3: Commit final integration**

```bash
git add .
git commit -m "test: verify brain reunification produces fitness improvement"
```
