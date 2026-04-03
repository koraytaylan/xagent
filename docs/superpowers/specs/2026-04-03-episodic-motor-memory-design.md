# Episodic Motor Memory & Memory-Fitness Fix

**Date:** 2026-04-03
**Status:** Approved

## Problem

Evolution converges memory_capacity and processing_slots to 1 (minimum) across all top-performing agents. The best fitness score (~0.33) is achieved by agents with effectively no memory. This is paradoxical — agents with no memory shouldn't outperform agents that can learn from experience.

### Root Causes

1. **Memory suppresses exploration.** The exploration rate formula in `action.rs:148-152` uses `recalled.len() / 16.0` as a "stability" metric that reduces exploration. Agents with larger memory recall more patterns, explore less, and follow an untrained policy (weights ≈ 0) — effectively paralyzing themselves. Agents with memory_capacity=1 stay exploratory and stumble into food/exploration by random movement.

2. **Memory doesn't inform actions.** Recalled patterns are not used in motor output computation at all. The action selector computes `forward = dot(weights, current_state) + bias` — purely reactive. Recalled patterns only blend weakly into the predictor's next-state forecast (via `context_weight=0.2`). Memory has no direct pathway to help the agent choose better actions.

3. **No metabolic cost.** There is no penalty for having memory_capacity=1. Once evolution ratchets down to 1 (via the `.max(1)` floor in mutation), it can only escape via a lucky upward mutation — which is then selected against because of cause #1.

## Design

### 1. Exploration Rate Fix

**File:** `crates/xagent-brain/src/action.rs`

Replace the recall-count-based stability metric with policy confidence based on raw motor output magnitude.

**Current:**
```rust
let stability = recalled.len() as f32 / 16.0;
self.exploration_rate =
    (0.5 - stability * 0.15 + novelty_bonus + curiosity_bonus - urgency_penalty).clamp(0.10, 0.85);
```

**New:**
```rust
let raw_signal = fwd.abs() + trn.abs();  // from policy weights × current state
let policy_confidence = (raw_signal / 2.0).clamp(0.0, 1.0);
self.exploration_rate =
    (0.5 - policy_confidence * 0.25 + novelty_bonus + curiosity_bonus - urgency_penalty).clamp(0.10, 0.85);
```

Properties:
- Untrained agent (weights ≈ 0) → raw_signal ≈ 0 → high exploration
- Well-trained agent → strong signal → lower exploration
- Memory size has no effect on exploration rate

Note: `fwd` and `trn` are already computed at lines 155-160. The exploration rate computation (currently at lines 148-152) must move to after step 4 (raw motor output) since it now depends on those values.

### 2. Episodic Motor Memory

**File:** `crates/xagent-brain/src/memory.rs`

Extend `Pattern` struct with motor context:

```rust
pub motor_forward: f32,     // motor command active when pattern was stored
pub motor_turn: f32,        // motor command active when pattern was stored
pub outcome_valence: f32,   // homeostatic gradient at storage time
```

#### Store Signature

```rust
pub fn store(&mut self, state: EncodedState, motor_forward: f32, motor_turn: f32, outcome_valence: f32)
```

Called from `brain.rs:tick_inner` after action selection, passing:
- `habituated` — the encoded state
- `command.forward`, `command.turn` — motor output before fatigue damping (intended action)
- `homeo_state.raw_gradient` — immediate outcome signal

#### Recall Return Type

Replace `Vec<(EncodedState, f32)>` with a named struct:

```rust
pub struct RecalledPattern {
    pub state: EncodedState,
    pub similarity: f32,
    pub motor_forward: f32,
    pub motor_turn: f32,
    pub outcome_valence: f32,
}
```

Both `recall()` and `recall_with_gpu_similarities()` return `Vec<RecalledPattern>`.

#### Memory-Informed Action Selection

**File:** `crates/xagent-brain/src/action.rs`

In `ActionSelector::select`, after computing raw policy output (step 4) and prospective evaluation (step 5), blend in memory suggestions:

```
memory_fwd = Σ (similarity × valence × stored_forward) / Σ |similarity × valence|
memory_trn = Σ (similarity × valence × stored_turn)   / Σ |similarity × valence|
```

Properties:
- Positive valence → "repeat this action" (reinforce successful behavior)
- Negative valence → "do the opposite" (sign flip avoids past mistakes)
- Normalization prevents memory from dominating

Blending:
```
let memory_strength = (total_weight / recalled.len().max(1) as f32).clamp(0.0, 1.0);
let mix = memory_strength * 0.4;  // memory contributes up to 40% of motor signal
fwd = fwd * (1.0 - mix) + memory_fwd * mix;
trn = trn * (1.0 - mix) + memory_trn * mix;
```

### 3. Retroactive Valence Update

**File:** `crates/xagent-brain/src/memory.rs`

Extend `learn()` signature to accept the current gradient:

```rust
pub fn learn(&mut self, current: &EncodedState, error: f32, learning_rate: f32, gradient: f32)
```

In the existing similarity loop (where reinforcement is updated), also nudge outcome_valence:

```rust
let valence_lr = learning_rate * 0.3;
pat.outcome_valence += sim * valence_lr * (gradient - pat.outcome_valence);
```

This is an EMA toward the current gradient, weighted by similarity. Properties:
- Patterns near food accumulate positive valence over multiple eating ticks
- Patterns near danger accumulate negative valence
- 0.3 factor makes valence updates slower than reinforcement — a running assessment, not a snap judgment
- Distant patterns (low similarity) are unaffected

### 4. Metabolic Cost

**File:** Agent simulation loop (governor.rs or agent tick)

Per-tick energy drain proportional to brain capacity:

```
base_cost = 0.0001          // baseline brain metabolism
capacity_cost = memory_capacity * 0.00003 + processing_slots * 0.0001
energy_drain_per_tick = base_cost + capacity_cost
```

Cost regimes over 50,000 tick budget:
- (1, 1): 0.00013/tick → 6.5 total — negligible
- (128, 16): 0.00554/tick → 277 total — meaningful, must forage to offset
- (2048, 128): 0.07434/tick → 3,717 total — expensive, must demonstrably outperform

Applied by subtracting `energy_drain_per_tick` from the agent's energy before passing `energy_signal` to the brain. The brain perceives its own metabolic cost through the homeostatic system — increased hunger → increased urgency → drives foraging.

The cost coefficients are governor-level world constants, not evolvable. The agent evolves how much capacity to carry; the world dictates what that costs.

## Scope Boundaries

**Not changing:**
- Predictor's `predict_weighted` — continues blending recalled patterns via `context_weight`, benefits from the same recalled patterns
- Mutation/crossover logic — perturbation formula and bounds for memory_capacity/processing_slots stay the same
- Fitness function — survival/foraging/exploration weights stay at 0.4/0.3/0.3; metabolic cost feeds through survival
- Decay, trauma, association chains — untouched
- `representation_dim` — stays at 32
- GPU similarity path — same return type change to `RecalledPattern`, same logic

## Change Footprint

| File | Changes |
|------|---------|
| `memory.rs` | `Pattern` struct (3 fields), `RecalledPattern` struct, `store()` signature, `learn()` signature, `recall()` return type, `recall_with_gpu_similarities()` return type |
| `action.rs` | Exploration rate formula (move after raw motor computation, use policy confidence), memory-informed motor blending in `select()` |
| `brain.rs` | Wire new signatures through `tick_inner()` — store call moves after action selection, learn call passes gradient |
| `governor.rs` / agent sim | Metabolic energy drain per tick |
