# Anti-Loop Mechanisms Design

**Problem:** Agents optimize for prediction-error minimization, causing them to choose predictable actions (like spinning in circles). Once the predictor builds an accurate model, exploration drops to the 0.10 floor and stays there. The agent enters a stable attractor with no mechanism to escape.

**Solution:** Four biologically-inspired mechanisms that make monotony aversive through gradual pressure (~50-100 ticks), implemented as independent systems that plug into existing decision points.

**Approach:** Two new modules split by domain (sensory vs motor), plus a configurable parameter in the existing homeostasis module. Each mechanism produces its own observable signal for telemetry.

## Architecture

Two new files in `xagent-brain`, one modification to an existing file:

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `crates/xagent-brain/src/habituation.rs` | Sensory habituation + curiosity bonus |
| Create | `crates/xagent-brain/src/motor_fatigue.rs` | Motor output fatigue |
| Modify | `crates/xagent-brain/src/homeostasis.rs` | Configurable distress exponent |
| Modify | `crates/xagent-brain/src/brain.rs` | Integrate into tick loop |
| Modify | `crates/xagent-brain/src/action.rs` | Add curiosity_bonus to exploration formula |
| Modify | `crates/xagent-brain/src/lib.rs` | Export new modules |
| Modify | `crates/xagent-sandbox/src/replay.rs` | New telemetry fields in TickRecord |
| Modify | `crates/xagent-sandbox/src/ui.rs` | New telemetry in AgentSnapshot, vitals display, history chart |
| Modify | `crates/xagent-sandbox/src/main.rs` | Wire new telemetry fields |

### Pipeline

```
encoder.encode(frame) → encoded_state (raw, preserved for telemetry)
                              │
                    habituation.update(encoded_state)
                              │
                    ┌─────────┴──────────┐
                    │                    │
            habituated_state      curiosity_bonus
                    │                    │
         ┌──────────┤                    │
         │          │                    │
    predictor    memory          action_selector
    (predict,    (recall,        (exploration_rate
     compare)    learn,           += curiosity_bonus)
                 store)                  │
                                   raw motor_cmd
                                         │
                              motor_fatigue.update(motor_cmd)
                                         │
                              final_motor = raw * fatigue_factor
```

Urgency escalation operates independently via the existing homeostasis pipeline with a configurable distress exponent.

## Module 1: Sensory Habituation

**File:** `crates/xagent-brain/src/habituation.rs`

**Design principle:** The encoder is a pure function — same input, same output. Habituation is a post-encoder filter that attenuates the signal before downstream consumers see it. The raw encoding is preserved for telemetry and replay.

### Struct

```rust
pub struct SensoryHabituation {
    prev_encoded: Vec<f32>,     // Previous tick's encoded state
    variance_ema: Vec<f32>,     // Per-dimension EMA of change magnitude
    attenuation: Vec<f32>,      // Per-dimension dampening [ATTENUATION_FLOOR, 1.0]
    curiosity_bonus: f32,       // Scalar [0.0, MAX_CURIOSITY_BONUS]
    tick: u64,
}
```

### Algorithm

Each tick, given the raw `encoded_state`:

1. Compute per-dimension change: `delta[i] = |encoded[i] - prev_encoded[i]|`
2. Smooth with EMA: `variance_ema[i] = (1 - HABITUATION_EMA_ALPHA) * variance_ema[i] + HABITUATION_EMA_ALPHA * delta[i]`
3. Derive attenuation: `attenuation[i] = (variance_ema[i] * SENSITIVITY).clamp(ATTENUATION_FLOOR, 1.0)`
4. Produce habituated state: `habituated[i] = encoded[i] * attenuation[i]`
5. Curiosity bonus: `curiosity_bonus = (1.0 - mean(attenuation)) * MAX_CURIOSITY_BONUS`

**Recovery:** When input changes, `variance_ema` rises within ~50 ticks, attenuation recovers, habituation fades naturally.

### Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| HABITUATION_EMA_ALPHA | 0.02 | ~50-tick smoothing window |
| SENSITIVITY | 10.0 | Scales variance_ema into attenuation range |
| ATTENUATION_FLOOR | 0.1 | Never fully suppress — 90% max dampening |
| MAX_CURIOSITY_BONUS | 0.4 | Ceiling on exploration boost (same scale as existing novelty_bonus) |

### Outputs

- `habituated_state: Vec<f32>` — attenuated encoding, fed to predictor, memory, and action selector
- `curiosity_bonus: f32` — additive term in exploration rate formula
- `mean_attenuation: f32` — telemetry (average across dimensions)

## Module 2: Motor Fatigue

**File:** `crates/xagent-brain/src/motor_fatigue.rs`

**Design principle:** Fatigue is physical, not cognitive. The muscles fail, and the brain learns from the consequence through existing credit assignment. No explicit wiring into credit — the dampened output leads to worse outcomes, negative gradient flows, policy adapts.

### Struct

```rust
pub struct MotorFatigue {
    forward_ring: Vec<f32>,     // Ring buffer of recent forward outputs
    turn_ring: Vec<f32>,        // Ring buffer of recent turn outputs
    cursor: usize,
    len: usize,
    fatigue_factor: f32,        // [FATIGUE_FLOOR, 1.0]
}
```

### Algorithm

Each tick, given the raw `motor_command`:

1. Record `motor_command.forward` and `motor_command.turn` into ring buffers
2. Compute variance: `fwd_var = variance(forward_ring)`, `turn_var = variance(turn_ring)`
3. Combined: `total_var = fwd_var + turn_var`
4. Fatigue factor: `fatigue_factor = (total_var * RECOVERY_SENSITIVITY).clamp(FATIGUE_FLOOR, 1.0)`
5. Apply: `final_forward = raw_forward * fatigue_factor`, `final_turn = raw_turn * fatigue_factor`

**Recovery is immediate:** When output diversifies, variance rises, fatigue lifts within a few ticks. No cooldown — the metaphor is sustained load, not accumulated damage.

### Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| FATIGUE_WINDOW | 64 | Ring buffer size (matches ACTION_HISTORY_LEN) |
| RECOVERY_SENSITIVITY | 5.0 | Scales variance into [0,1] range |
| FATIGUE_FLOOR | 0.2 | Never fully paralyze — 80% max dampening |

### Outputs

- `fatigue_factor: f32` — multiplicative dampening on motor output
- `motor_variance: f32` — telemetry (total_var before scaling)

## Module 3: Urgency Escalation

**File:** `crates/xagent-brain/src/homeostasis.rs` (existing, modified)

**Design principle:** Simple, decoupled, evolvable. The distress curve exponent becomes a heritable parameter that natural selection tunes per lineage.

### Change

Add `distress_exponent: f32` to `HomeostaticMonitor`. Default `2.0` (preserves current behavior).

```rust
// Current:
distress = (1.0 - level).powi(2) * DISTRESS_SCALE

// New:
distress = (1.0 - level).powf(self.distress_exponent) * DISTRESS_SCALE
```

### Behavioral effect

| Energy | exp=2.0 (current) | exp=3.0 | exp=4.0 |
|--------|-------------------|---------|---------|
| 0.9 | 0.10 | 0.01 | 0.001 |
| 0.7 | 0.90 | 0.27 | 0.08 |
| 0.5 | 2.50 | 1.25 | 0.63 |
| 0.3 | 4.90 | 3.43 | 2.40 |
| 0.1 | 8.10 | 7.29 | 6.56 |

Higher exponents: calm longer, panic harder at critical levels. Lower exponents: earlier, gentler pressure.

### Evolution

- Exported/imported with heritable config
- Mutated during breeding (small perturbation)
- Clamped to `[1.5, 5.0]`

## Brain Tick Loop Integration

**File:** `crates/xagent-brain/src/brain.rs`

### New fields on Brain

```rust
pub struct Brain {
    // ...existing...
    habituation: SensoryHabituation,
    motor_fatigue: MotorFatigue,
}
```

### Modified pipeline

Steps marked with `>` are new or changed:

1. `encoded_state = encoder.encode(frame)`
2. `> habituation.update(encoded_state)`
3. `> habituated_state = habituation.habituated_state()`
4. `> curiosity_bonus = habituation.curiosity_bonus()`
5. `homeostasis.update(energy, integrity)` — uses configurable exponent
6. `prediction_error = predictor.compare(last_prediction, habituated_state)` — was `encoded_state`
7. `memory.learn(habituated_state, error, lr)` — was `encoded_state`
8. `capacity.allocate(prediction_error)`
9. `recalled = memory.recall(habituated_state, budget)` — was `encoded_state`
10. `prediction = predictor.predict(habituated_state, recalled)` — was `encoded_state`
11. `memory.store(habituated_state)` — was `encoded_state`
12. `memory.decay(...)`
13. `motor_cmd = action_selector.select(habituated_state, ..., curiosity_bonus)` — new param
14. `> motor_fatigue.update(motor_cmd)`
15. `> final_motor = motor_cmd * motor_fatigue.fatigue_factor()`
16. `encoder.adapt_from_credit(credit_signal, lr)`
17. Telemetry — includes new fields

### Exploration rate formula change

In `action_selector.select()`:

```rust
// Current:
exploration_rate = (0.5 - stability * 0.15 + novelty_bonus - urgency_penalty).clamp(0.10, 0.85)

// New:
exploration_rate = (0.5 - stability * 0.15 + novelty_bonus + curiosity_bonus - urgency_penalty).clamp(0.10, 0.85)
```

### Weight inheritance

`SensoryHabituation` and `MotorFatigue` have no learned weights — pure runtime state, reset each generation. Only `distress_exponent` is heritable.

## Telemetry & UI Integration

### BrainTelemetry additions

```rust
pub struct BrainTelemetry {
    // ...existing fields...
    pub mean_attenuation: f32,    // Sensory habituation level [0.1, 1.0]
    pub curiosity_bonus: f32,     // Exploration boost from monotony [0.0, 0.4]
    pub fatigue_factor: f32,      // Motor dampening [0.2, 1.0]
    pub motor_variance: f32,      // Recent motor output diversity
}
```

### TickRecord additions (replay.rs)

```rust
pub struct TickRecord {
    // ...existing fields...
    pub mean_attenuation: f32,
    pub curiosity_bonus: f32,
    pub fatigue_factor: f32,
    pub motor_variance: f32,
}
```

### AgentSnapshot additions (ui.rs)

```rust
pub struct AgentSnapshot {
    // ...existing fields...
    pub mean_attenuation: f32,
    pub curiosity_bonus: f32,
    pub fatigue_factor: f32,
    pub motor_variance: f32,
}
```

### Agent detail tab — Vitals section

Four new rows in the existing vitals grid:

| Label | Value | Color hint |
|-------|-------|-----------|
| Habituation | mean_attenuation | Fades toward red as attenuation drops |
| Curiosity | curiosity_bonus | Brightens toward green as curiosity rises |
| Fatigue | fatigue_factor | Fades toward red as fatigue increases |
| Motor Var. | motor_variance | Informational, gray |

### History chart

Add `fatigue_factor` as a fifth line (orange-red) to the existing chart alongside energy, integrity, prediction error, and exploration rate.

## Interaction Between Mechanisms

The four mechanisms are fully decoupled but interact through existing signals:

1. **Habituation dampens encoding** → predictor sees less change → prediction error drops → exploration rate drops (slightly). But simultaneously...
2. **Curiosity bonus rises** (from the same habituation signal) → exploration rate rises (more than the drop from #1). Net effect: more exploration.
3. **Motor fatigue dampens output** → agent moves less → energy still depletes → homeostatic gradient worsens → negative credit to recent actions → policy shifts away from repetitive pattern.
4. **Urgency escalation** (independent) → at critical health, urgency amplifies gradient → overwhelms any learned preference for the loop.

**Cascade scenario — agent stuck spinning:**
- Ticks 0-50: Habituation builds, sensory input dampens, curiosity bonus rises
- Ticks 20-80: Motor fatigue builds (low output variance), motor output weakens
- Tick ~60: Curiosity has pushed exploration rate up, fatigue has weakened the repetitive output, noise begins to dominate → agent breaks the loop
- If energy is critical: urgency escalation further amplifies pressure to act differently

## Non-Goals

- No changes to the memory system's storage or recall logic
- No changes to the predictor's learning algorithm
- No new evolutionary variables beyond `distress_exponent`
- No interaction between habituation and motor fatigue modules
- No cooldown mechanics for fatigue (immediate recovery on diversity)
