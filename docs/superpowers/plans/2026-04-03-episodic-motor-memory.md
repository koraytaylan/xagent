# Episodic Motor Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix memory convergence to (1,1) by making memory directly useful for action selection, decoupling exploration from recall count, and adding metabolic cost for brain capacity.

**Architecture:** Four interdependent changes across the brain crate and sandbox: (1) `RecalledPattern` struct + `Pattern` motor fields in memory.rs, (2) exploration rate + memory-informed blending in action.rs, (3) wiring through brain.rs tick_inner, (4) metabolic energy drain in physics/mod.rs.

**Tech Stack:** Rust, xagent-brain crate, xagent-sandbox crate, xagent-shared crate

---

### Task 1: Add Motor Fields to Pattern and Create RecalledPattern

**Files:**
- Modify: `crates/xagent-brain/src/memory.rs:31-54` (Pattern struct)
- Modify: `crates/xagent-brain/src/memory.rs:92-110` (PatternMemory::new)
- Test: `crates/xagent-brain/src/memory.rs` (inline tests module at line 552)

- [ ] **Step 1: Write failing test for Pattern motor fields**

Add this test at the end of the `tests` module in `memory.rs` (before the closing `}`):

```rust
#[test]
fn pattern_stores_motor_context() {
    let mut mem = PatternMemory::new(10, 4);
    mem.store(
        make_state(&[1.0, 0.0, 0.0, 0.0]),
        0.8,   // motor_forward
        -0.3,  // motor_turn
        0.05,  // outcome_valence
    );
    let p = mem.get(0).unwrap();
    assert!((p.motor_forward - 0.8).abs() < 1e-6);
    assert!((p.motor_turn - (-0.3)).abs() < 1e-6);
    assert!((p.outcome_valence - 0.05).abs() < 1e-6);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xagent-brain pattern_stores_motor_context -- --nocapture 2>&1 | head -30`
Expected: Compilation error — `store()` doesn't accept motor args yet.

- [ ] **Step 3: Add motor fields to Pattern struct**

In `memory.rs`, add three fields to the `Pattern` struct (after `generation: u64` at line 53):

```rust
    /// Motor forward output active when this pattern was stored.
    pub motor_forward: f32,
    /// Motor turn output active when this pattern was stored.
    pub motor_turn: f32,
    /// Homeostatic gradient at storage time (positive = good outcome).
    pub outcome_valence: f32,
```

- [ ] **Step 4: Update store() signature and body**

Change `store()` at line 119 from:
```rust
pub fn store(&mut self, state: EncodedState) {
```
to:
```rust
pub fn store(&mut self, state: EncodedState, motor_forward: f32, motor_turn: f32, outcome_valence: f32) {
```

In the `Pattern` initialization block (lines 136-148), add the three new fields:
```rust
        self.patterns[slot] = Some(Pattern {
            state,
            norm,
            reinforcement: 1.0,
            created_at: self.current_tick,
            last_accessed: self.current_tick,
            activation_count: 1,
            associations: [empty_link; MAX_ASSOCIATIONS_PER_PATTERN],
            association_count: 0,
            predecessor: prev_idx,
            successor: None,
            generation: self.next_generation,
            motor_forward,
            motor_turn,
            outcome_valence,
        });
```

- [ ] **Step 5: Fix all existing store() call sites in tests**

Update every existing `mem.store(make_state(...))` call in the `tests` module to pass motor context. Use `0.0, 0.0, 0.0` as defaults for tests that don't care about motor fields. There are calls in these tests:
- `store_and_recall` (line 563)
- `decay_removes_weak_patterns` (line 575)
- `frequently_accessed_patterns_decay_slower` (lines 589-590)
- `temporal_sequence_tracking` (lines 611-613)
- `association_chain_retrieval` (lines 630-632)
- `capacity_limit_overwrites_weakest` (lines 641-644)
- `overwritten_associations_are_invalidated` (lines 656-658, 667)
- `co_occurrence_strengthens_associations` (lines 693-694)
- `trauma_reduces_reinforcement` (lines 726-727)
- `trauma_removes_weak_patterns` (line 759)

For each, change e.g.:
```rust
mem.store(make_state(&[1.0, 0.0, 0.0, 0.0]));
```
to:
```rust
mem.store(make_state(&[1.0, 0.0, 0.0, 0.0]), 0.0, 0.0, 0.0);
```

- [ ] **Step 6: Create RecalledPattern struct**

Add this struct above the `PatternMemory` struct (before line 56):

```rust
/// A pattern recalled from memory, including its motor context.
#[derive(Clone, Debug)]
pub struct RecalledPattern {
    pub state: EncodedState,
    pub similarity: f32,
    pub motor_forward: f32,
    pub motor_turn: f32,
    pub outcome_valence: f32,
}
```

- [ ] **Step 7: Update recall() to return Vec\<RecalledPattern\>**

Change `recall()` return type at line 184 from:
```rust
    ) -> Vec<(EncodedState, f32)> {
```
to:
```rust
    ) -> Vec<RecalledPattern> {
```

Update the result collection at lines 209-216 from:
```rust
        self.scored_scratch
            .iter()
            .filter_map(|&(idx, sim)| {
                self.patterns[idx]
                    .as_ref()
                    .map(|p| (p.state.clone(), sim))
            })
            .collect()
```
to:
```rust
        self.scored_scratch
            .iter()
            .filter_map(|&(idx, sim)| {
                self.patterns[idx].as_ref().map(|p| RecalledPattern {
                    state: p.state.clone(),
                    similarity: sim,
                    motor_forward: p.motor_forward,
                    motor_turn: p.motor_turn,
                    outcome_valence: p.outcome_valence,
                })
            })
            .collect()
```

- [ ] **Step 8: Update recall_with_gpu_similarities() to return Vec\<RecalledPattern\>**

Change return type at line 519 from:
```rust
    ) -> Vec<(EncodedState, f32)> {
```
to:
```rust
    ) -> Vec<RecalledPattern> {
```

Update the result collection at lines 541-548 from:
```rust
        self.scored_scratch
            .iter()
            .filter_map(|&(idx, sim)| {
                self.patterns[idx]
                    .as_ref()
                    .map(|p| (p.state.clone(), sim))
            })
            .collect()
```
to:
```rust
        self.scored_scratch
            .iter()
            .filter_map(|&(idx, sim)| {
                self.patterns[idx].as_ref().map(|p| RecalledPattern {
                    state: p.state.clone(),
                    similarity: sim,
                    motor_forward: p.motor_forward,
                    motor_turn: p.motor_turn,
                    outcome_valence: p.outcome_valence,
                })
            })
            .collect()
```

- [ ] **Step 9: Fix recall() test call sites**

In the test module, `store_and_recall` (line 566) accesses `results[0].0` and `results[0].1`. Update:
```rust
        let results = mem.recall(&make_state(&[0.9, 0.1, 0.0, 0.0]), 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].state.data()[0] > 0.5, "Should recall the [1,0,0,0] pattern");
```

In `frequently_accessed_patterns_decay_slower` (line 593), the `recall` call doesn't inspect the return value — no change needed to the assertion, just make sure the return type change compiles.

- [ ] **Step 10: Run all memory tests**

Run: `cargo test -p xagent-brain memory -- --nocapture 2>&1 | tail -20`
Expected: All 11 tests pass (10 existing + 1 new).

- [ ] **Step 11: Commit**

```bash
git add crates/xagent-brain/src/memory.rs
git commit -m "feat: add motor fields to Pattern and RecalledPattern return type"
```

---

### Task 2: Update learn() for Retroactive Valence Updates

**Files:**
- Modify: `crates/xagent-brain/src/memory.rs:285-317` (learn method)
- Test: `crates/xagent-brain/src/memory.rs` (inline tests module)

- [ ] **Step 1: Write failing test for valence update**

Add to the memory tests module:

```rust
#[test]
fn learn_updates_outcome_valence() {
    let mut mem = PatternMemory::new(10, 4);
    // Store a pattern with neutral valence
    mem.store(make_state(&[1.0, 0.0, 0.0, 0.0]), 0.5, 0.0, 0.0);

    // Learn with a positive gradient near the stored pattern
    let current = make_state(&[0.9, 0.1, 0.0, 0.0]);
    for _ in 0..20 {
        mem.learn(&current, 0.1, 0.1, 0.5);
    }

    let p = mem.get(0).unwrap();
    assert!(
        p.outcome_valence > 0.1,
        "Valence should become positive after positive gradient: got {}",
        p.outcome_valence
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xagent-brain learn_updates_outcome_valence -- --nocapture 2>&1 | head -20`
Expected: Compilation error — `learn()` doesn't accept `gradient` parameter.

- [ ] **Step 3: Update learn() signature and add valence update**

Change `learn()` signature at line 285 from:
```rust
    pub fn learn(&mut self, current: &EncodedState, error: f32, learning_rate: f32) {
```
to:
```rust
    pub fn learn(&mut self, current: &EncodedState, error: f32, learning_rate: f32, gradient: f32) {
```

In the reinforcement loop (lines 300-305), add valence update after the reinforcement line:
```rust
        for &(i, sim) in &sims {
            if let Some(ref mut pat) = self.patterns[i] {
                pat.reinforcement += sim * learning_rate * (1.0 - error);
                pat.reinforcement = pat.reinforcement.clamp(0.0, MAX_REINFORCEMENT);
                // Retroactive valence update: nudge toward current gradient
                let valence_lr = learning_rate * 0.3;
                pat.outcome_valence += sim * valence_lr * (gradient - pat.outcome_valence);
            }
        }
```

- [ ] **Step 4: Fix existing learn() call sites in tests**

In `co_occurrence_strengthens_associations` (line 706), update:
```rust
            mem.learn(&current, 0.1, 0.1);
```
to:
```rust
            mem.learn(&current, 0.1, 0.1, 0.0);
```

- [ ] **Step 5: Run all memory tests**

Run: `cargo test -p xagent-brain memory -- --nocapture 2>&1 | tail -20`
Expected: All 12 tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/xagent-brain/src/memory.rs
git commit -m "feat: add retroactive valence update to memory learn()"
```

---

### Task 3: Update ActionSelector for Memory-Informed Blending

**Files:**
- Modify: `crates/xagent-brain/src/action.rs:120-204` (select method)
- Test: `crates/xagent-brain/src/action.rs` (inline tests module at line 393)

- [ ] **Step 1: Write failing test for memory-informed blending**

Add to the action tests module:

```rust
#[test]
fn recalled_positive_valence_biases_motor_output() {
    use crate::memory::RecalledPattern;
    let dim = 4;
    let mut sel = ActionSelector::new(dim);

    // Create a recalled pattern with strong positive valence and forward thrust
    let recalled = vec![RecalledPattern {
        state: crate::encoder::EncodedState::from_slice(&[1.0, 0.0, 0.0, 0.0]),
        similarity: 0.9,
        motor_forward: 0.8,
        motor_turn: -0.5,
        outcome_valence: 0.5,
    }];

    let state = crate::encoder::EncodedState::from_slice(&[1.0, 0.0, 0.0, 0.0]);
    let prediction = crate::encoder::EncodedState::from_slice(&[1.0, 0.0, 0.0, 0.0]);

    // Run many times and average to cancel out noise
    let mut total_fwd = 0.0;
    let n = 200;
    for _ in 0..n {
        let cmd = sel.select(&state, &prediction, &recalled, 0.0, 0.0, 0.0, 0.0);
        total_fwd += cmd.forward;
    }
    let avg_fwd = total_fwd / n as f32;

    // With positive valence recalled pattern suggesting 0.8 forward,
    // average should be biased positive (untrained policy weights = 0,
    // so without memory the average would be ~0 from symmetric noise)
    assert!(
        avg_fwd > 0.05,
        "Memory with positive valence should bias forward output positive: avg={}",
        avg_fwd,
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xagent-brain recalled_positive_valence_biases_motor_output -- --nocapture 2>&1 | head -30`
Expected: Compilation error — `select()` still expects `&[(EncodedState, f32)]`.

- [ ] **Step 3: Update select() signature**

Change the `recalled` parameter type at line 124 from:
```rust
        recalled: &[(EncodedState, f32)],
```
to:
```rust
        recalled: &[crate::memory::RecalledPattern],
```

- [ ] **Step 4: Replace exploration rate formula with policy confidence**

Remove the old stability-based exploration rate block (lines 148-152). The new exploration rate computation goes *after* the raw motor output computation (after line 160). Replace the block:

```rust
        // 3. Adaptive exploration rate.
        // Base 0.5 for continuous motor: additive noise of ±0.5 produces
        // vigorous movement comparable to the old discrete system's random
        // full-strength actions. Both forward and turn channels get real
        // signal so credit assignment can learn both steering and thrust.
        // As the policy learns (weights grow), it dominates the noise and
        // behavior smoothly transitions from random to directed.
        let stability = recalled.len() as f32 / 16.0;
        let novelty_bonus = (prediction_error * 2.0).min(0.4);
        let urgency_penalty = (urgency * 0.4).min(0.5);
        self.exploration_rate =
            (0.5 - stability * 0.15 + novelty_bonus + curiosity_bonus - urgency_penalty).clamp(0.10, 0.85);
```

with a placeholder comment that will be filled in step 5:
```rust
        // 3. Exploration rate (computed after raw motor output, see below).
        let novelty_bonus = (prediction_error * 2.0).min(0.4);
        let urgency_penalty = (urgency * 0.4).min(0.5);
```

- [ ] **Step 5: Reorder select() — raw motor, then exploration, then memory blend**

After the raw motor output computation (after computing `fwd` and `trn` from policy weights), add the exploration rate and memory blending blocks. The full reordered flow after step 2 (credit assignment) becomes:

```rust
        // 3. Compute raw motor outputs from encoded state.
        let mut fwd = self.forward_bias;
        let mut trn = self.turn_bias;
        for d in 0..dim.min(current.data().len()) {
            fwd += self.forward_weights[d] * current.data()[d];
            trn += self.turn_weights[d] * current.data()[d];
        }

        // 4. Prospective evaluation: if confident, modulate output toward
        //    predicted future's policy response.
        let confidence = 1.0 - prediction_error.clamp(0.0, 1.0);
        if confidence > 0.1 {
            let mut fwd_future = self.forward_bias;
            let mut trn_future = self.turn_bias;
            let pred_len = dim.min(prediction.data().len());
            for d in 0..pred_len {
                fwd_future += self.forward_weights[d] * prediction.data()[d];
                trn_future += self.turn_weights[d] * prediction.data()[d];
            }
            fwd += confidence * ANTICIPATION_WEIGHT * (fwd_future - fwd);
            trn += confidence * ANTICIPATION_WEIGHT * (trn_future - trn);
        }

        // 5. Memory-informed motor blending: recalled patterns suggest
        //    actions based on past outcomes in similar states.
        let mut mem_fwd = 0.0_f32;
        let mut mem_trn = 0.0_f32;
        let mut total_weight = 0.0_f32;
        for rp in recalled {
            let w = rp.similarity * rp.outcome_valence;
            mem_fwd += w * rp.motor_forward;
            mem_trn += w * rp.motor_turn;
            total_weight += w.abs();
        }
        if total_weight > 1e-6 {
            mem_fwd /= total_weight;
            mem_trn /= total_weight;
            let memory_strength = (total_weight / recalled.len().max(1) as f32).clamp(0.0, 1.0);
            let mix = memory_strength * 0.4;
            fwd = fwd * (1.0 - mix) + mem_fwd * mix;
            trn = trn * (1.0 - mix) + mem_trn * mix;
        }

        // 6. Adaptive exploration rate: based on policy confidence, not recall count.
        let raw_signal = fwd.abs() + trn.abs();
        let policy_confidence = (raw_signal / 2.0).clamp(0.0, 1.0);
        self.exploration_rate =
            (0.5 - policy_confidence * 0.25 + novelty_bonus + curiosity_bonus - urgency_penalty)
                .clamp(0.10, 0.85);

        // 7. Clean output through tanh squashing.
        let fwd_clean = crate::fast_tanh(fwd);
        let trn_clean = crate::fast_tanh(trn);

        // 8. Add exploration noise.
        let fwd_noisy =
            (fwd_clean + rng.random_range(-1.0..1.0) * self.exploration_rate).clamp(-1.0, 1.0);
        let trn_noisy =
            (trn_clean + rng.random_range(-1.0..1.0) * self.exploration_rate).clamp(-1.0, 1.0);
```

- [ ] **Step 6: Fix existing test call sites in action.rs**

All existing tests that call `sel.select(...)` pass `recalled` as `&[(EncodedState, f32)]`. Update them to pass `&[]` (empty slice of `RecalledPattern`) or construct `RecalledPattern` values. The tests are:

- `continuous_output_is_bounded` — change `&[]` to `&[] as &[crate::memory::RecalledPattern]`
- `positive_gradient_transition_increases_forward` — same
- `negative_gradient_transition_penalizes_approach` — same
- `exploration_adds_noise` — same
- `death_signal_clears_history` — same (also update `assign_death_credit` if it calls select)
- `weight_export_import_roundtrip` — same
- `prospection_modulates_output` — same

For each call like:
```rust
let cmd = sel.select(&state, &pred, &[], 0.0, 0.0, 0.0, 0.0);
```
Change to:
```rust
let cmd = sel.select(&state, &pred, &[] as &[crate::memory::RecalledPattern], 0.0, 0.0, 0.0, 0.0);
```

Or add a type alias at the top of the test module:
```rust
use crate::memory::RecalledPattern;
```
and use `&[] as &[RecalledPattern]`.

- [ ] **Step 7: Run all action tests**

Run: `cargo test -p xagent-brain action -- --nocapture 2>&1 | tail -20`
Expected: All 8 tests pass (7 existing + 1 new).

- [ ] **Step 8: Commit**

```bash
git add crates/xagent-brain/src/action.rs
git commit -m "feat: memory-informed action blending and policy-confidence exploration rate"
```

---

### Task 4: Wire Changes Through brain.rs tick_inner

**Files:**
- Modify: `crates/xagent-brain/src/brain.rs:191-342` (tick_inner method)
- Test: `crates/xagent-brain/src/brain.rs` (inline tests module at line 433)

- [ ] **Step 1: Write failing test for motor context in store**

Add to the brain tests module:

```rust
#[test]
fn tick_stores_motor_context_in_memory() {
    let config = BrainConfig::default();
    let mut brain = Brain::new(config);
    let frame = SensoryFrame::new_blank(8, 6);

    // Tick a few times to build up state
    for _ in 0..10 {
        brain.tick(&frame);
    }

    // Memory should have patterns with motor fields set
    // (not all zeros, since exploration noise produces non-zero motors)
    let has_nonzero_motor = (0..brain.memory.active_count()).any(|i| {
        brain.memory.get(i).map_or(false, |p| {
            p.motor_forward.abs() > 1e-6 || p.motor_turn.abs() > 1e-6
        })
    });
    assert!(
        has_nonzero_motor,
        "At least one pattern should have non-zero motor context after ticking"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xagent-brain tick_stores_motor_context -- --nocapture 2>&1 | head -30`
Expected: Compilation error — `tick_inner` still calls `store()` with old signature.

- [ ] **Step 3: Update tick_inner to pass motor context to store()**

In `brain.rs:tick_inner`, the current flow is:
1. Lines 200-230: habituation, homeostasis, prediction error, learn
2. Lines 232-241: recall
3. Lines 243-253: predict
4. Line 256: `self.memory.store(habituated.clone());`
5. Line 259: `self.memory.decay(self.config.decay_rate);`
6. Lines 267-275: action selection

Reorder: move store to after action selection, passing motor context. Delete line 256 (`self.memory.store(habituated.clone());`).

After the action selector call (after line 275), before motor fatigue (line 283), insert:

```rust
        // 8b. Store current state as pattern with motor context.
        // Uses pre-fatigue motor output (the intended action) and
        // current homeostatic gradient as outcome signal.
        self.memory.store(
            habituated.clone(),
            command.forward,
            command.turn,
            homeo_state.raw_gradient,
        );

        // 8c. Decay old patterns.
        self.memory.decay(self.config.decay_rate);
```

And remove the old store and decay calls at lines 256 and 259.

- [ ] **Step 4: Update learn() call to pass gradient**

Change line 217 from:
```rust
            self.memory.learn(&habituated, scalar_error, modulated_lr);
```
to:
```rust
            self.memory.learn(&habituated, scalar_error, modulated_lr, homeo_state.raw_gradient);
```

Note: `homeo_state` is computed at line 205-206 *before* the learn call, so `raw_gradient` is available. However, `homeo_state` is declared inside `tick_inner` at line 205. The learn call at line 217 is inside an `if let Some(prev_prediction)` block. Verify that `homeo_state` is in scope — it is, since it's declared at the method level (line 205), not inside the if block.

- [ ] **Step 5: Update recalled pattern usage**

The recall result type changed from `Vec<(EncodedState, f32)>` to `Vec<RecalledPattern>`. Update the code that uses `recalled`:

At the action selector call (lines 267-275), `recalled` is passed as `&recalled`. The `select` method now expects `&[RecalledPattern]` — this matches.

The predictor call at line 244 (`self.predictor.predict_weighted(&habituated, &recalled)`) still expects `&[(EncodedState, f32)]`. We need to convert. Add a helper extraction before the predict call:

```rust
        // 5. Predict next state using recalled patterns
        let recalled_for_predictor: Vec<(EncodedState, f32)> = recalled
            .iter()
            .map(|rp| (rp.state.clone(), rp.similarity))
            .collect();
        let prediction = self.predictor.predict_weighted(&habituated, &recalled_for_predictor);
```

Also update the capacity report at line 241:
```rust
        self.capacity.report_usage(recalled.len());
```
This stays the same — `recalled.len()` works on `Vec<RecalledPattern>`.

- [ ] **Step 6: Add RecalledPattern import**

At the top of `brain.rs`, ensure `RecalledPattern` is imported:
```rust
use crate::memory::{PatternMemory, RecalledPattern};
```

If `PatternMemory` is not currently imported explicitly (it's used via `self.memory`), just add `RecalledPattern` to whatever existing import path covers memory types. Check the current imports at the top of brain.rs.

- [ ] **Step 7: Run all brain tests**

Run: `cargo test -p xagent-brain brain -- --nocapture 2>&1 | tail -30`
Expected: All 11 tests pass (10 existing + 1 new).

- [ ] **Step 8: Run the full xagent-brain test suite**

Run: `cargo test -p xagent-brain -- --nocapture 2>&1 | tail -30`
Expected: All tests pass across memory, action, brain, predictor modules.

- [ ] **Step 9: Commit**

```bash
git add crates/xagent-brain/src/brain.rs
git commit -m "feat: wire motor context through tick_inner — store after action, pass gradient to learn"
```

---

### Task 5: Add Metabolic Energy Drain

**Files:**
- Modify: `crates/xagent-sandbox/src/physics/mod.rs:105-108` (energy depletion section)
- Modify: `crates/xagent-sandbox/src/physics/mod.rs:22-23` (function signature of `step`)
- Test: `crates/xagent-sandbox/tests/integration.rs`

- [ ] **Step 1: Write failing integration test for metabolic cost**

Add to `crates/xagent-sandbox/tests/integration.rs`:

```rust
#[test]
fn metabolic_cost_drains_energy_proportional_to_capacity() {
    use xagent_shared::BrainConfig;

    // Two configs: tiny brain vs large brain
    let small = BrainConfig { memory_capacity: 1, processing_slots: 1, ..BrainConfig::default() };
    let large = BrainConfig { memory_capacity: 512, processing_slots: 32, ..BrainConfig::default() };

    let small_drain = xagent_sandbox::physics::metabolic_drain_per_tick(
        small.memory_capacity,
        small.processing_slots,
    );
    let large_drain = xagent_sandbox::physics::metabolic_drain_per_tick(
        large.memory_capacity,
        large.processing_slots,
    );

    assert!(small_drain > 0.0, "Even small brains have baseline cost");
    assert!(
        large_drain > small_drain * 10.0,
        "Large brain should cost significantly more: small={small_drain}, large={large_drain}",
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xagent-sandbox metabolic_cost_drains -- --nocapture 2>&1 | head -20`
Expected: Compilation error — `metabolic_drain_per_tick` doesn't exist.

- [ ] **Step 3: Add metabolic_drain_per_tick function**

In `crates/xagent-sandbox/src/physics/mod.rs`, add a public function after the constants (after line 20):

```rust
/// Metabolic cost per tick for maintaining brain capacity.
/// Larger memory and processing capacity drain more energy — the
/// biological cost of a bigger brain.
const METABOLIC_BASE_COST: f32 = 0.0001;
const METABOLIC_MEMORY_COST: f32 = 0.00003;
const METABOLIC_PROCESSING_COST: f32 = 0.0001;

pub fn metabolic_drain_per_tick(memory_capacity: usize, processing_slots: usize) -> f32 {
    METABOLIC_BASE_COST
        + memory_capacity as f32 * METABOLIC_MEMORY_COST
        + processing_slots as f32 * METABOLIC_PROCESSING_COST
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xagent-sandbox metabolic_cost_drains -- --nocapture 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 5: Apply metabolic drain in the simulation loop**

In `crates/xagent-sandbox/src/main.rs`, find the CPU agent tick loop (around line 1383) where `physics::step` is called. After the physics step, add the metabolic drain:

```rust
                                let consumed = xagent_sandbox::physics::step(
                                    &mut agent.body, &motor, world, SIM_DT,
                                );

                                // Metabolic cost: brain capacity drains energy
                                let brain_drain = xagent_sandbox::physics::metabolic_drain_per_tick(
                                    agent.brain.config.memory_capacity,
                                    agent.brain.config.processing_slots,
                                );
                                agent.body.body.internal.energy -= brain_drain;
```

Also find the GPU agent tick path (around line 1292-1303) and add the same drain there. Look for the equivalent `physics::step` call in the GPU path and add the drain after it.

- [ ] **Step 6: Run the full sandbox test suite**

Run: `cargo test -p xagent-sandbox -- --nocapture 2>&1 | tail -30`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/xagent-sandbox/src/physics/mod.rs crates/xagent-sandbox/src/main.rs crates/xagent-sandbox/tests/integration.rs
git commit -m "feat: add metabolic energy drain proportional to brain capacity"
```

---

### Task 6: Export RecalledPattern from xagent-brain crate

**Files:**
- Modify: `crates/xagent-brain/src/lib.rs` (public exports)

- [ ] **Step 1: Add RecalledPattern to public exports**

In `crates/xagent-brain/src/lib.rs`, find where `PatternMemory` or other memory types are exported. Add `RecalledPattern` to the public API:

```rust
pub use memory::RecalledPattern;
```

If there's no existing re-export of memory types, add the line to the `pub use` block or `pub mod memory` declaration.

- [ ] **Step 2: Verify full compilation**

Run: `cargo check -p xagent-sandbox`
Expected: Clean compilation with no errors.

- [ ] **Step 3: Run all tests across both crates**

Run: `cargo test -p xagent-brain && cargo test -p xagent-sandbox`
Expected: All tests pass in both crates.

- [ ] **Step 4: Commit**

```bash
git add crates/xagent-brain/src/lib.rs
git commit -m "feat: export RecalledPattern from xagent-brain public API"
```

---

### Task 7: Final Integration Verification

**Files:**
- No new files — verification only

- [ ] **Step 1: Run cargo check on the entire workspace**

Run: `cargo check`
Expected: Clean compilation across all crates.

- [ ] **Step 2: Run the full test suite**

Run: `cargo test -p xagent-sandbox`
Expected: All 87+ tests pass (existing + new tests).

- [ ] **Step 3: Run cargo clippy**

Run: `cargo clippy -p xagent-brain -p xagent-sandbox -- -D warnings 2>&1 | tail -20`
Expected: No warnings.

- [ ] **Step 4: Commit any clippy fixes if needed**

```bash
git add -A && git commit -m "fix: address clippy warnings from episodic motor memory changes"
```

(Skip if no warnings.)
