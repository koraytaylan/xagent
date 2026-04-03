# Anti-Loop Mechanisms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four biologically-inspired anti-loop mechanisms (sensory habituation, curiosity bonus, motor fatigue, urgency escalation) that make monotony aversive through gradual pressure, preventing agents from getting stuck in repetitive behavioral loops.

**Architecture:** Two new modules in `xagent-brain` (`habituation.rs`, `motor_fatigue.rs`) split by domain (sensory vs motor), plus a configurable distress exponent in the existing homeostasis module. The habituation module produces a post-encoder attenuation filter and a curiosity bonus for exploration. Motor fatigue dampens motor output when action variance is low. All signals flow through existing decision points.

**Tech Stack:** Rust, xagent-brain crate (cognitive architecture), xagent-sandbox crate (simulation/UI), xagent-shared crate (config), egui 0.31

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `crates/xagent-brain/src/habituation.rs` | SensoryHabituation struct: post-encoder attenuation + curiosity bonus |
| Create | `crates/xagent-brain/src/motor_fatigue.rs` | MotorFatigue struct: ring buffer variance → fatigue factor |
| Modify | `crates/xagent-brain/src/homeostasis.rs:26-86,166-170` | Add `distress_exponent` field, use in distress curve |
| Modify | `crates/xagent-brain/src/action.rs:120-151` | Add `curiosity_bonus` parameter to `select()` |
| Modify | `crates/xagent-brain/src/brain.rs:1-9,106-119,126-151,154-301` | Add habituation + motor_fatigue fields, rewire tick pipeline |
| Modify | `crates/xagent-brain/src/lib.rs:11-17,27` | Export new modules |
| Modify | `crates/xagent-shared/src/config.rs:11-24,141-177,252-273` | Add `distress_exponent` to BrainConfig, defaults, presets, mutation |
| Modify | `crates/xagent-sandbox/src/replay.rs:9-28` | Add 4 telemetry fields to TickRecord |
| Modify | `crates/xagent-sandbox/src/ui.rs:55-101,786-789,1009-1055` | Add 4 fields to AgentSnapshot, vitals display, history chart |
| Modify | `crates/xagent-sandbox/src/main.rs:171-198,1414-1438,1922-1973` | Wire telemetry to histories, TickRecord, AgentSnapshot |

---

### Task 1: Create the SensoryHabituation module

**Files:**
- Create: `crates/xagent-brain/src/habituation.rs`

- [ ] **Step 1: Write the tests**

Create `crates/xagent-brain/src/habituation.rs` with the full module including tests:

```rust
//! Sensory habituation: post-encoder attenuation filter + curiosity bonus.
//!
//! Tracks per-dimension variance of the encoded state over time. Dimensions
//! that stop changing get attenuated (habituated). A curiosity bonus derived
//! from the mean attenuation drives exploration when input is monotonous.

/// EMA smoothing factor for variance tracking (~50-tick window).
const HABITUATION_EMA_ALPHA: f32 = 0.02;
/// Scales variance_ema into the [0, 1] attenuation range.
const SENSITIVITY: f32 = 10.0;
/// Minimum attenuation — never fully suppress a dimension.
const ATTENUATION_FLOOR: f32 = 0.1;
/// Maximum curiosity bonus (same scale as novelty_bonus in action selector).
const MAX_CURIOSITY_BONUS: f32 = 0.4;

/// Post-encoder filter that attenuates repetitive sensory input and produces
/// a curiosity bonus that drives exploration during monotony.
pub struct SensoryHabituation {
    prev_encoded: Vec<f32>,
    variance_ema: Vec<f32>,
    attenuation: Vec<f32>,
    habituated: Vec<f32>,
    curiosity_bonus: f32,
    tick: u64,
}

impl SensoryHabituation {
    /// Create a new habituation filter for the given representation dimension.
    pub fn new(repr_dim: usize) -> Self {
        Self {
            prev_encoded: vec![0.0; repr_dim],
            variance_ema: vec![0.0; repr_dim],
            attenuation: vec![1.0; repr_dim],
            habituated: vec![0.0; repr_dim],
            curiosity_bonus: 0.0,
            tick: 0,
        }
    }

    /// Update with a new encoded state. Computes attenuation and curiosity bonus.
    pub fn update(&mut self, encoded: &[f32]) {
        self.tick += 1;
        let dim = self.prev_encoded.len().min(encoded.len());

        let mut attenuation_sum = 0.0_f32;
        for i in 0..dim {
            // Per-dimension change magnitude
            let delta = (encoded[i] - self.prev_encoded[i]).abs();

            // EMA smoothing
            self.variance_ema[i] =
                (1.0 - HABITUATION_EMA_ALPHA) * self.variance_ema[i] + HABITUATION_EMA_ALPHA * delta;

            // Derive attenuation from variance
            self.attenuation[i] = (self.variance_ema[i] * SENSITIVITY).clamp(ATTENUATION_FLOOR, 1.0);

            // Produce habituated state
            self.habituated[i] = encoded[i] * self.attenuation[i];

            attenuation_sum += self.attenuation[i];
            self.prev_encoded[i] = encoded[i];
        }

        // Curiosity bonus: high when attenuation is low (monotonous input)
        let mean_attenuation = if dim > 0 {
            attenuation_sum / dim as f32
        } else {
            1.0
        };
        self.curiosity_bonus = (1.0 - mean_attenuation) * MAX_CURIOSITY_BONUS;
    }

    /// The habituated (attenuated) encoded state.
    pub fn habituated_state(&self) -> &[f32] {
        &self.habituated
    }

    /// Current curiosity bonus [0.0, MAX_CURIOSITY_BONUS].
    pub fn curiosity_bonus(&self) -> f32 {
        self.curiosity_bonus
    }

    /// Mean attenuation across all dimensions [ATTENUATION_FLOOR, 1.0].
    pub fn mean_attenuation(&self) -> f32 {
        if self.attenuation.is_empty() {
            return 1.0;
        }
        self.attenuation.iter().sum::<f32>() / self.attenuation.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_input_increases_habituation() {
        let mut hab = SensoryHabituation::new(4);
        let state = vec![0.5, 0.3, -0.1, 0.2];

        // Feed the same input many times
        for _ in 0..200 {
            hab.update(&state);
        }

        // Attenuation should be near the floor
        assert!(
            hab.mean_attenuation() < 0.2,
            "Constant input should drive attenuation near floor: {}",
            hab.mean_attenuation()
        );
        // Curiosity should be high
        assert!(
            hab.curiosity_bonus() > 0.3,
            "Constant input should produce high curiosity: {}",
            hab.curiosity_bonus()
        );
    }

    #[test]
    fn varying_input_keeps_attenuation_high() {
        let mut hab = SensoryHabituation::new(4);

        for i in 0..200 {
            let v = (i as f32 * 0.1).sin();
            let state = vec![v, -v, v * 0.5, v * 0.3];
            hab.update(&state);
        }

        // Attenuation should stay high (input is changing)
        assert!(
            hab.mean_attenuation() > 0.5,
            "Varying input should keep attenuation high: {}",
            hab.mean_attenuation()
        );
        // Curiosity should be low
        assert!(
            hab.curiosity_bonus() < 0.2,
            "Varying input should produce low curiosity: {}",
            hab.curiosity_bonus()
        );
    }

    #[test]
    fn habituation_recovers_after_change() {
        let mut hab = SensoryHabituation::new(4);
        let constant = vec![0.5, 0.3, -0.1, 0.2];

        // Habituate
        for _ in 0..200 {
            hab.update(&constant);
        }
        assert!(hab.mean_attenuation() < 0.2);

        // Change input
        let different = vec![-0.5, 0.8, 0.4, -0.3];
        for _ in 0..100 {
            hab.update(&different);
        }

        // After some ticks with the new input, attenuation of the initial
        // change should have recovered then re-habituated. But the key test:
        // right after the change, attenuation jumped up.
        // We test by verifying the curiosity dropped after the change.
        // (After 100 ticks of the NEW constant, it re-habituates)
        assert!(
            hab.mean_attenuation() < 0.3,
            "Should re-habituate to new constant: {}",
            hab.mean_attenuation()
        );
    }

    #[test]
    fn habituated_state_is_attenuated() {
        let mut hab = SensoryHabituation::new(4);
        let state = vec![1.0, 1.0, 1.0, 1.0];

        // Habituate
        for _ in 0..200 {
            hab.update(&state);
        }

        // Habituated values should be significantly less than raw
        for &v in hab.habituated_state() {
            assert!(
                v < 0.3,
                "Habituated value should be attenuated: {}",
                v
            );
        }
    }

    #[test]
    fn attenuation_is_bounded() {
        let mut hab = SensoryHabituation::new(4);

        // Test with zero input (extreme habituation)
        for _ in 0..500 {
            hab.update(&[0.0, 0.0, 0.0, 0.0]);
        }

        for &a in &hab.attenuation {
            assert!(
                a >= ATTENUATION_FLOOR && a <= 1.0,
                "Attenuation {} out of bounds [{}, 1.0]",
                a, ATTENUATION_FLOOR
            );
        }

        // Curiosity should be bounded
        assert!(
            hab.curiosity_bonus() >= 0.0 && hab.curiosity_bonus() <= MAX_CURIOSITY_BONUS,
            "Curiosity {} out of bounds [0, {}]",
            hab.curiosity_bonus(), MAX_CURIOSITY_BONUS
        );
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p xagent-brain habituation 2>&1`

This will fail because the module isn't exported from `lib.rs` yet.

- [ ] **Step 3: Export the module from lib.rs**

In `crates/xagent-brain/src/lib.rs`, add the module declaration. Find:

```rust
pub mod action;
```

Add after it:

```rust
pub mod habituation;
```

Also update the re-export line. Find:

```rust
pub use brain::{Brain, BrainTelemetry, DecisionSnapshot, LearnedState};
```

Replace with:

```rust
pub use brain::{Brain, BrainTelemetry, DecisionSnapshot, LearnedState};
pub use habituation::SensoryHabituation;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p xagent-brain habituation 2>&1`
Expected: all 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/xagent-brain/src/habituation.rs crates/xagent-brain/src/lib.rs
git commit -m "feat: add SensoryHabituation module (post-encoder attenuation + curiosity)"
```

---

### Task 2: Create the MotorFatigue module

**Files:**
- Create: `crates/xagent-brain/src/motor_fatigue.rs`
- Modify: `crates/xagent-brain/src/lib.rs`

- [ ] **Step 1: Write the module with tests**

Create `crates/xagent-brain/src/motor_fatigue.rs`:

```rust
//! Motor fatigue: dampens motor output when action variance is low.
//!
//! Tracks recent forward and turn outputs in a ring buffer. When the variance
//! is low (repetitive motor commands), a fatigue factor reduces the effective
//! motor output. Recovery is immediate when output diversifies.

/// Ring buffer size for tracking recent motor outputs.
const FATIGUE_WINDOW: usize = 64;
/// Scales variance into the [0, 1] fatigue factor range.
const RECOVERY_SENSITIVITY: f32 = 5.0;
/// Minimum fatigue factor — never fully paralyze (80% max dampening).
const FATIGUE_FLOOR: f32 = 0.2;

/// Tracks motor output variance and produces a fatigue dampening factor.
pub struct MotorFatigue {
    forward_ring: Vec<f32>,
    turn_ring: Vec<f32>,
    cursor: usize,
    len: usize,
    fatigue_factor: f32,
    motor_variance: f32,
}

impl MotorFatigue {
    /// Create a new motor fatigue tracker.
    pub fn new() -> Self {
        Self {
            forward_ring: vec![0.0; FATIGUE_WINDOW],
            turn_ring: vec![0.0; FATIGUE_WINDOW],
            cursor: 0,
            len: 0,
            fatigue_factor: 1.0,
            motor_variance: 0.0,
        }
    }

    /// Record a motor command and update the fatigue factor.
    pub fn update(&mut self, forward: f32, turn: f32) {
        self.forward_ring[self.cursor] = forward;
        self.turn_ring[self.cursor] = turn;
        self.cursor = (self.cursor + 1) % FATIGUE_WINDOW;
        if self.len < FATIGUE_WINDOW {
            self.len += 1;
        }

        let fwd_var = Self::variance(&self.forward_ring[..self.len]);
        let turn_var = Self::variance(&self.turn_ring[..self.len]);
        self.motor_variance = fwd_var + turn_var;

        self.fatigue_factor = (self.motor_variance * RECOVERY_SENSITIVITY)
            .clamp(FATIGUE_FLOOR, 1.0);
    }

    /// Current fatigue factor [FATIGUE_FLOOR, 1.0]. Multiply motor output by this.
    pub fn fatigue_factor(&self) -> f32 {
        self.fatigue_factor
    }

    /// Current total motor variance (for telemetry).
    pub fn motor_variance(&self) -> f32 {
        self.motor_variance
    }

    /// Compute variance of a slice.
    fn variance(data: &[f32]) -> f32 {
        if data.len() < 2 {
            return 0.0;
        }
        let n = data.len() as f32;
        let mean = data.iter().sum::<f32>() / n;
        data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_output_causes_fatigue() {
        let mut mf = MotorFatigue::new();

        // Same motor output every tick
        for _ in 0..FATIGUE_WINDOW {
            mf.update(0.5, 0.3);
        }

        assert!(
            mf.fatigue_factor() < 0.3,
            "Constant motor output should cause fatigue: {}",
            mf.fatigue_factor()
        );
        assert!(
            mf.motor_variance() < 0.01,
            "Constant output should have near-zero variance: {}",
            mf.motor_variance()
        );
    }

    #[test]
    fn varied_output_prevents_fatigue() {
        let mut mf = MotorFatigue::new();

        for i in 0..FATIGUE_WINDOW {
            let v = (i as f32 * 0.2).sin();
            mf.update(v, -v);
        }

        assert!(
            mf.fatigue_factor() > 0.8,
            "Varied output should prevent fatigue: {}",
            mf.fatigue_factor()
        );
    }

    #[test]
    fn fatigue_recovers_when_output_diversifies() {
        let mut mf = MotorFatigue::new();

        // Build fatigue
        for _ in 0..FATIGUE_WINDOW {
            mf.update(0.5, 0.3);
        }
        assert!(mf.fatigue_factor() < 0.3);

        // Diversify output
        for i in 0..FATIGUE_WINDOW {
            let v = (i as f32 * 0.3).sin();
            mf.update(v, -v);
        }

        assert!(
            mf.fatigue_factor() > 0.8,
            "Fatigue should recover when output diversifies: {}",
            mf.fatigue_factor()
        );
    }

    #[test]
    fn fatigue_factor_is_bounded() {
        let mut mf = MotorFatigue::new();

        // Extreme case: zero output forever
        for _ in 0..200 {
            mf.update(0.0, 0.0);
        }
        assert!(
            mf.fatigue_factor() >= FATIGUE_FLOOR,
            "Fatigue factor should not go below floor: {}",
            mf.fatigue_factor()
        );

        // Extreme case: wildly varying output
        for i in 0..200 {
            let v = if i % 2 == 0 { 1.0 } else { -1.0 };
            mf.update(v, -v);
        }
        assert!(
            mf.fatigue_factor() <= 1.0,
            "Fatigue factor should not exceed 1.0: {}",
            mf.fatigue_factor()
        );
    }

    #[test]
    fn initial_state_has_no_fatigue() {
        let mf = MotorFatigue::new();
        assert_eq!(mf.fatigue_factor(), 1.0);
        assert_eq!(mf.motor_variance(), 0.0);
    }
}
```

- [ ] **Step 2: Export the module from lib.rs**

In `crates/xagent-brain/src/lib.rs`, find:

```rust
pub mod habituation;
```

Add after it:

```rust
pub mod motor_fatigue;
```

Update the re-exports. Find:

```rust
pub use habituation::SensoryHabituation;
```

Replace with:

```rust
pub use habituation::SensoryHabituation;
pub use motor_fatigue::MotorFatigue;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p xagent-brain motor_fatigue 2>&1`
Expected: all 5 tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/xagent-brain/src/motor_fatigue.rs crates/xagent-brain/src/lib.rs
git commit -m "feat: add MotorFatigue module (output dampening on low variance)"
```

---

### Task 3: Add configurable distress exponent to homeostasis

**Files:**
- Modify: `crates/xagent-brain/src/homeostasis.rs:26-48,73-86,166-170`
- Modify: `crates/xagent-shared/src/config.rs:11-24,141-151,154-177`
- Modify: `crates/xagent-sandbox/src/agent/mod.rs:252-273`

- [ ] **Step 1: Add `distress_exponent` to BrainConfig**

In `crates/xagent-shared/src/config.rs`, find:

```rust
    /// Decay rate for unreinforced patterns per tick.
    pub decay_rate: f32,
}
```

Replace with:

```rust
    /// Decay rate for unreinforced patterns per tick.
    pub decay_rate: f32,
    /// Exponent for the homeostatic distress curve. Higher = calm longer, panic harder.
    /// Heritable: mutated during breeding, clamped to [1.5, 5.0]. Default 2.0.
    pub distress_exponent: f32,
}
```

In the `Default` impl, find:

```rust
            decay_rate: 0.001,
        }
```

Replace with:

```rust
            decay_rate: 0.001,
            distress_exponent: 2.0,
        }
```

In the `tiny()` preset, find:

```rust
            decay_rate: 0.002,
        }
```

Replace with:

```rust
            decay_rate: 0.002,
            distress_exponent: 2.0,
        }
```

In the `large()` preset, find:

```rust
            decay_rate: 0.0005,
        }
```

Replace with:

```rust
            decay_rate: 0.0005,
            distress_exponent: 2.0,
        }
```

- [ ] **Step 2: Add `distress_exponent` to HomeostaticMonitor**

In `crates/xagent-brain/src/homeostasis.rs`, find the struct definition:

```rust
pub struct HomeostaticMonitor {
    /// Previous energy signal.
    prev_energy: f32,
```

Replace with:

```rust
pub struct HomeostaticMonitor {
    /// Exponent for distress curve (heritable, default 2.0).
    distress_exponent: f32,
    /// Previous energy signal.
    prev_energy: f32,
```

Update the constructor. Find:

```rust
    pub fn new() -> Self {
        Self {
            prev_energy: 1.0,
            prev_integrity: 1.0,
```

Replace with:

```rust
    pub fn new(distress_exponent: f32) -> Self {
        Self {
            distress_exponent,
            prev_energy: 1.0,
            prev_integrity: 1.0,
```

Update `reset()`. Find:

```rust
    pub fn reset(&mut self) {
        *self = Self::new();
    }
```

Replace with:

```rust
    pub fn reset(&mut self) {
        let exp = self.distress_exponent;
        *self = Self::new(exp);
    }
```

Change `distress_curve` from a static method to an instance method. Find:

```rust
    fn distress_curve(level: f32) -> f32 {
        let clamped = level.clamp(0.01, 1.0);
        let distress = (1.0 - clamped).powi(2) * DISTRESS_SCALE;
        distress.min(MAX_DISTRESS)
    }
```

Replace with:

```rust
    fn distress_curve(&self, level: f32) -> f32 {
        let clamped = level.clamp(0.01, 1.0);
        let distress = (1.0 - clamped).powf(self.distress_exponent) * DISTRESS_SCALE;
        distress.min(MAX_DISTRESS)
    }
```

Update the two call sites in `update()`. Find:

```rust
        let energy_distress = Self::distress_curve(energy_signal);
        let integrity_distress = Self::distress_curve(integrity_signal);
```

Replace with:

```rust
        let energy_distress = self.distress_curve(energy_signal);
        let integrity_distress = self.distress_curve(integrity_signal);
```

- [ ] **Step 3: Update Brain::new() to pass distress_exponent**

In `crates/xagent-brain/src/brain.rs`, find:

```rust
        let homeostasis = HomeostaticMonitor::new();
```

Replace with:

```rust
        let homeostasis = HomeostaticMonitor::new(config.distress_exponent);
```

- [ ] **Step 4: Add `distress_exponent` to mutation logic**

In `crates/xagent-sandbox/src/agent/mod.rs`, find the `mutate_config_with_strength` function's return block:

```rust
    BrainConfig {
        memory_capacity: perturb_u(&mut rng, parent.memory_capacity),
        processing_slots: perturb_u(&mut rng, parent.processing_slots),
        visual_encoding_size: parent.visual_encoding_size,
        representation_dim: perturb_u(&mut rng, parent.representation_dim).min(MAX_REPR_DIM),
        learning_rate: perturb_f(&mut rng, parent.learning_rate),
        decay_rate: perturb_f(&mut rng, parent.decay_rate),
    }
```

Replace with:

```rust
    BrainConfig {
        memory_capacity: perturb_u(&mut rng, parent.memory_capacity),
        processing_slots: perturb_u(&mut rng, parent.processing_slots),
        visual_encoding_size: parent.visual_encoding_size,
        representation_dim: perturb_u(&mut rng, parent.representation_dim).min(MAX_REPR_DIM),
        learning_rate: perturb_f(&mut rng, parent.learning_rate),
        decay_rate: perturb_f(&mut rng, parent.decay_rate),
        distress_exponent: perturb_f(&mut rng, parent.distress_exponent).clamp(1.5, 5.0),
    }
```

- [ ] **Step 5: Fix the homeostasis tests**

The tests call `HomeostaticMonitor::new()` without arguments. Update every test that constructs a monitor. In `crates/xagent-brain/src/homeostasis.rs`, replace all occurrences of `HomeostaticMonitor::new()` in the `#[cfg(test)]` module with `HomeostaticMonitor::new(2.0)`.

Also the `distress_curve_is_bounded` test calls `HomeostaticMonitor::distress_curve(level)` as a static method. Update it:

Find:

```rust
    fn distress_curve_is_bounded() {
        for i in 0..=100 {
            let level = i as f32 / 100.0;
            let d = HomeostaticMonitor::distress_curve(level);
```

Replace with:

```rust
    fn distress_curve_is_bounded() {
        let hm = HomeostaticMonitor::new(2.0);
        for i in 0..=100 {
            let level = i as f32 / 100.0;
            let d = hm.distress_curve(level);
```

Note: `distress_curve` is now `fn distress_curve(&self, ...)` so it needs `pub(crate)` or the test needs access. Since tests are in the same module, `fn` (private) is fine — tests in `mod tests` within the same file can access private methods.

- [ ] **Step 6: Run tests**

Run: `cargo test -p xagent-brain homeostasis 2>&1`
Expected: all 10 homeostasis tests pass

Run: `cargo test -p xagent-brain brain::tests 2>&1`
Expected: all brain tests pass (Brain::new now passes distress_exponent from config)

- [ ] **Step 7: Commit**

```bash
git add crates/xagent-shared/src/config.rs crates/xagent-brain/src/homeostasis.rs crates/xagent-brain/src/brain.rs crates/xagent-sandbox/src/agent/mod.rs
git commit -m "feat: configurable distress exponent in homeostasis (heritable)"
```

---

### Task 4: Add `curiosity_bonus` parameter to ActionSelector::select()

**Files:**
- Modify: `crates/xagent-brain/src/action.rs:120-151`

- [ ] **Step 1: Add the curiosity_bonus parameter**

In `crates/xagent-brain/src/action.rs`, find the `select()` signature:

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
```

Replace with:

```rust
    pub fn select(
        &mut self,
        current: &EncodedState,
        prediction: &EncodedState,
        recalled: &[(EncodedState, f32)],
        homeostatic_gradient: f32,
        prediction_error: f32,
        urgency: f32,
        curiosity_bonus: f32,
    ) -> MotorCommand {
```

Then update the exploration rate computation. Find:

```rust
        self.exploration_rate =
            (0.5 - stability * 0.15 + novelty_bonus - urgency_penalty).clamp(0.10, 0.85);
```

Replace with:

```rust
        self.exploration_rate =
            (0.5 - stability * 0.15 + novelty_bonus + curiosity_bonus - urgency_penalty).clamp(0.10, 0.85);
```

- [ ] **Step 2: Fix all call sites in action.rs tests**

Every test that calls `sel.select(...)` passes 6 arguments. They all need a 7th argument (`0.0` for curiosity_bonus). In the `tests` module of `action.rs`, update every `sel.select(` call by adding `, 0.0` before the closing `)`.

The calls to update (search for `sel.select(` in the tests):
- `continuous_output_is_bounded`: `sel.select(&state, &pred, &[], 0.0, 0.1, 0.0)` → `sel.select(&state, &pred, &[], 0.0, 0.1, 0.0, 0.0)`
- `positive_gradient_transition_increases_forward` (two loops): same pattern
- `negative_gradient_transition_penalizes_approach` (two loops): same pattern
- `exploration_adds_noise`: `sel.select(&state, &pred, &[], 0.0, 0.9, 0.0)` → add `, 0.0`
- `death_signal_clears_history`: same
- `weight_export_import_roundtrip` (two loops): same
- `prospection_modulates_output` (four calls): `sel.select(&state, &pred, &[], 0.0, 0.1, 0.0)` and the `sel_copy_n.select(...)` / `sel_copy_p.select(...)` calls — add `, 0.0` to each

- [ ] **Step 3: Fix the call site in brain.rs**

In `crates/xagent-brain/src/brain.rs`, find the `action_selector.select()` call:

```rust
        let command = self.action_selector.select(
            &encoded,
            &prospection_prediction,
            &recalled,
            homeo_state.raw_gradient,
            scalar_error,
            homeo_state.urgency,
        );
```

Replace with:

```rust
        let command = self.action_selector.select(
            &encoded,
            &prospection_prediction,
            &recalled,
            homeo_state.raw_gradient,
            scalar_error,
            homeo_state.urgency,
            0.0, // curiosity_bonus: wired in Task 5
        );
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p xagent-brain 2>&1`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/xagent-brain/src/action.rs crates/xagent-brain/src/brain.rs
git commit -m "feat: add curiosity_bonus parameter to ActionSelector::select()"
```

---

### Task 5: Integrate habituation + motor fatigue into the brain tick loop

**Files:**
- Modify: `crates/xagent-brain/src/brain.rs:1-9,17-44,106-119,126-151,154-301`

- [ ] **Step 1: Add imports and fields**

In `crates/xagent-brain/src/brain.rs`, find:

```rust
use crate::homeostasis::{HomeostaticMonitor, HomeostaticState};
use crate::memory::PatternMemory;
use crate::predictor::Predictor;
```

Replace with:

```rust
use crate::habituation::SensoryHabituation;
use crate::homeostasis::{HomeostaticMonitor, HomeostaticState};
use crate::memory::PatternMemory;
use crate::motor_fatigue::MotorFatigue;
use crate::predictor::Predictor;
```

Add telemetry fields to `BrainTelemetry`. Find:

```rust
    /// Composite decision quality score [0.0, 1.0].
    pub decision_quality: f32,
}
```

Replace with:

```rust
    /// Composite decision quality score [0.0, 1.0].
    pub decision_quality: f32,
    /// Sensory habituation: mean attenuation across encoded dimensions [0.1, 1.0].
    pub mean_attenuation: f32,
    /// Curiosity bonus from sensory monotony [0.0, 0.4].
    pub curiosity_bonus: f32,
    /// Motor fatigue factor [0.2, 1.0]. Low = fatigued.
    pub fatigue_factor: f32,
    /// Recent motor output variance (higher = more diverse).
    pub motor_variance: f32,
}
```

Add fields to `Brain` struct. Find:

```rust
    pub capacity: CapacityManager,
    tick_count: u64,
```

Replace with:

```rust
    pub capacity: CapacityManager,
    pub habituation: SensoryHabituation,
    pub motor_fatigue: MotorFatigue,
    tick_count: u64,
```

- [ ] **Step 2: Initialize new fields in Brain::new()**

In `Brain::new()`, find:

```rust
        Self {
            config,
            encoder,
            memory,
            predictor,
            action_selector,
            homeostasis,
            capacity,
            tick_count: 0,
```

Replace with:

```rust
        Self {
            config,
            encoder,
            memory,
            predictor,
            action_selector,
            homeostasis,
            capacity,
            habituation: SensoryHabituation::new(config.representation_dim),
            motor_fatigue: MotorFatigue::new(),
            tick_count: 0,
```

Note: `config` is moved into the struct, but `config.representation_dim` is a `usize` (Copy), so reading it before the move is fine. However, since the struct takes ownership of `config`, we need to read `representation_dim` before the struct construction. Update to:

```rust
        let repr_dim = config.representation_dim;

        Self {
            config,
            encoder,
            memory,
            predictor,
            action_selector,
            homeostasis,
            capacity,
            habituation: SensoryHabituation::new(repr_dim),
            motor_fatigue: MotorFatigue::new(),
            tick_count: 0,
```

- [ ] **Step 3: Rewire the tick pipeline**

In `tick_inner()`, the encoded state currently flows directly to all consumers. We insert habituation after encoding and feed `habituated_state` downstream instead. Also add motor fatigue after action selection.

Find the section after encoding (step 1 → step 2 in current code):

```rust
        // 1. Compute homeostatic gradient from interoceptive signals
        let homeo_state: HomeostaticState =
            self.homeostasis.update(frame.energy_signal, frame.integrity_signal);

        // 2. Compute prediction error from previous tick's prediction
        let mut scalar_error = 0.0_f32;
        let modulated_lr = self.config.learning_rate * (1.0 + homeo_state.gradient.abs());

        if let Some(prev_prediction) = self.predictor.last_prediction() {
            let prev_prediction = prev_prediction.clone();
            scalar_error = self.predictor.prediction_error(&prev_prediction, &encoded);
```

Replace with:

```rust
        // 1a. Sensory habituation: attenuate encoded state + compute curiosity
        self.habituation.update(encoded.data());
        let habituated = EncodedState::from_slice(self.habituation.habituated_state());

        // 1b. Compute homeostatic gradient from interoceptive signals
        let homeo_state: HomeostaticState =
            self.homeostasis.update(frame.energy_signal, frame.integrity_signal);

        // 2. Compute prediction error from previous tick's prediction
        let mut scalar_error = 0.0_f32;
        let modulated_lr = self.config.learning_rate * (1.0 + homeo_state.gradient.abs());

        if let Some(prev_prediction) = self.predictor.last_prediction() {
            let prev_prediction = prev_prediction.clone();
            scalar_error = self.predictor.prediction_error(&prev_prediction, &habituated);
```

Now replace all remaining references to `encoded` with `habituated` in the rest of `tick_inner`, EXCEPT:
- Keep `encoded` in the two existing places: `self.encoder.encode(frame)` at the top, and the initial prediction error vec computation.
- The `memory.learn`, `memory.recall`, `predictor.predict_weighted`, `memory.store`, and `action_selector.select` should all use `habituated`.

Find:

```rust
            self.memory.learn(&encoded, scalar_error, modulated_lr);

            // Compute error vector into scratch, then learn from it
            // We use the static version to avoid borrow conflict with learn()
            {
                let error_vec = Predictor::prediction_error_vec(&prev_prediction, &encoded);
```

Replace with:

```rust
            self.memory.learn(&habituated, scalar_error, modulated_lr);

            // Compute error vector into scratch, then learn from it
            {
                let error_vec = Predictor::prediction_error_vec(&prev_prediction, &habituated);
```

Find:

```rust
        let recalled = match gpu_similarities {
            Some(sims) => self.memory.recall_with_gpu_similarities(sims, recall_budget),
            None => self.memory.recall(&encoded, recall_budget),
        };
```

Replace with:

```rust
        let recalled = match gpu_similarities {
            Some(sims) => self.memory.recall_with_gpu_similarities(sims, recall_budget),
            None => self.memory.recall(&habituated, recall_budget),
        };
```

Find:

```rust
        let prediction = self.predictor.predict_weighted(&encoded, &recalled);
```

Replace with:

```rust
        let prediction = self.predictor.predict_weighted(&habituated, &recalled);
```

Find:

```rust
        self.memory.store(encoded.clone());
```

Replace with:

```rust
        self.memory.store(habituated.clone());
```

Find the action selection call (that we updated in Task 4):

```rust
        let command = self.action_selector.select(
            &encoded,
            &prospection_prediction,
            &recalled,
            homeo_state.raw_gradient,
            scalar_error,
            homeo_state.urgency,
            0.0, // curiosity_bonus: wired in Task 5
        );
```

Replace with:

```rust
        let curiosity_bonus = self.habituation.curiosity_bonus();
        let command = self.action_selector.select(
            &habituated,
            &prospection_prediction,
            &recalled,
            homeo_state.raw_gradient,
            scalar_error,
            homeo_state.urgency,
            curiosity_bonus,
        );
```

- [ ] **Step 4: Add motor fatigue after action selection**

Find the block after action selection and credit-driven encoder adaptation:

```rust
        // 8b. Credit-driven encoder adaptation: amplify raw features that
        //     contributed to behaviourally relevant encoded dimensions.
        let credit_signal = self.action_selector.last_credit_signal();
        self.encoder.adapt_from_credit(credit_signal, modulated_lr);

        // 9. Record prediction for next tick's error computation
        self.predictor.record_prediction(prediction);
```

Replace with:

```rust
        // 8b. Credit-driven encoder adaptation: amplify raw features that
        //     contributed to behaviourally relevant encoded dimensions.
        let credit_signal = self.action_selector.last_credit_signal();
        self.encoder.adapt_from_credit(credit_signal, modulated_lr);

        // 8c. Motor fatigue: dampen output when action variance is low.
        self.motor_fatigue.update(command.forward, command.turn);
        let fatigue = self.motor_fatigue.fatigue_factor();
        let command = MotorCommand {
            forward: command.forward * fatigue,
            turn: command.turn * fatigue,
            strafe: command.strafe,
            action: command.action,
        };

        // 9. Record prediction for next tick's error computation
        self.predictor.record_prediction(prediction);
```

- [ ] **Step 5: Update telemetry and decision snapshot**

Find the telemetry block:

```rust
        self.last_telemetry = BrainTelemetry {
            tick: self.tick_count,
            prediction_error: scalar_error,
            memory_utilization: self.memory.utilization(),
            memory_active_count: self.memory.active_count(),
            action_entropy: self.action_selector.action_entropy(),
            exploration_rate,
            homeostatic_gradient: homeo_state.gradient,
            homeostatic_urgency: homeo_state.urgency,
            recall_budget,
            avg_prediction_error: self.predictor.recent_avg_error(32),
            exploitation_ratio,
            decision_quality,
        };
```

Replace with:

```rust
        self.last_telemetry = BrainTelemetry {
            tick: self.tick_count,
            prediction_error: scalar_error,
            memory_utilization: self.memory.utilization(),
            memory_active_count: self.memory.active_count(),
            action_entropy: self.action_selector.action_entropy(),
            exploration_rate,
            homeostatic_gradient: homeo_state.gradient,
            homeostatic_urgency: homeo_state.urgency,
            recall_budget,
            avg_prediction_error: self.predictor.recent_avg_error(32),
            exploitation_ratio,
            decision_quality,
            mean_attenuation: self.habituation.mean_attenuation(),
            curiosity_bonus,
            fatigue_factor: self.motor_fatigue.fatigue_factor(),
            motor_variance: self.motor_fatigue.motor_variance(),
        };
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p xagent-brain 2>&1`
Expected: all tests pass (brain tests, action tests, homeostasis tests, habituation tests, motor_fatigue tests)

- [ ] **Step 7: Commit**

```bash
git add crates/xagent-brain/src/brain.rs
git commit -m "feat: integrate habituation + motor fatigue into brain tick loop"
```

---

### Task 6: Add telemetry fields to TickRecord and AgentSnapshot

**Files:**
- Modify: `crates/xagent-sandbox/src/replay.rs:9-28`
- Modify: `crates/xagent-sandbox/src/ui.rs:55-101`

- [ ] **Step 1: Add fields to TickRecord**

In `crates/xagent-sandbox/src/replay.rs`, find:

```rust
    /// Vision color data (8*6*4 = 192 f32 values). Only stored at keyframes
    /// (every VISION_KEYFRAME_INTERVAL ticks) to save memory.
    pub vision_color: Option<Vec<f32>>,
}
```

Replace with:

```rust
    /// Sensory habituation: mean attenuation [0.1, 1.0].
    pub mean_attenuation: f32,
    /// Curiosity bonus from sensory monotony [0.0, 0.4].
    pub curiosity_bonus: f32,
    /// Motor fatigue factor [0.2, 1.0].
    pub fatigue_factor: f32,
    /// Motor output variance.
    pub motor_variance: f32,
    /// Vision color data (8*6*4 = 192 f32 values). Only stored at keyframes
    /// (every VISION_KEYFRAME_INTERVAL ticks) to save memory.
    pub vision_color: Option<Vec<f32>>,
}
```

- [ ] **Step 2: Add fields to AgentSnapshot**

In `crates/xagent-sandbox/src/ui.rs`, find:

```rust
    /// Agent yaw (rotation around Y axis, radians).
    pub yaw: f32,
}
```

Replace with:

```rust
    /// Agent yaw (rotation around Y axis, radians).
    pub yaw: f32,
    /// Sensory habituation: mean attenuation [0.1, 1.0].
    pub mean_attenuation: f32,
    /// Curiosity bonus from sensory monotony [0.0, 0.4].
    pub curiosity_bonus: f32,
    /// Motor fatigue factor [0.2, 1.0].
    pub fatigue_factor: f32,
    /// Motor output variance.
    pub motor_variance: f32,
    /// Fatigue factor history for the chart.
    pub fatigue_history: Vec<f32>,
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p xagent-sandbox 2>&1`

This will fail because the struct construction sites in `main.rs` don't have the new fields yet. That's expected — Task 7 wires them.

- [ ] **Step 4: Commit**

```bash
git add crates/xagent-sandbox/src/replay.rs crates/xagent-sandbox/src/ui.rs
git commit -m "feat: add anti-loop telemetry fields to TickRecord and AgentSnapshot"
```

---

### Task 7: Wire telemetry through main.rs (histories, TickRecord, AgentSnapshot)

**Files:**
- Modify: `crates/xagent-sandbox/src/main.rs:171-198,1414-1438,1922-1973`
- Modify: `crates/xagent-sandbox/src/agent/mod.rs:102-136,138-159`

- [ ] **Step 1: Add fatigue_history to Agent struct**

In `crates/xagent-sandbox/src/agent/mod.rs`, find:

```rust
    pub integrity_history: std::collections::VecDeque<f32>,
```

Add after it:

```rust
    pub fatigue_history: std::collections::VecDeque<f32>,
```

In `Agent::new()`, find:

```rust
            integrity_history: std::collections::VecDeque::with_capacity(128),
```

Add after it (there may be lines between — add after the integrity_history initialization):

```rust
            fatigue_history: std::collections::VecDeque::with_capacity(128),
```

- [ ] **Step 2: Record fatigue history per tick**

In `crates/xagent-sandbox/src/main.rs`, find the `record_agent_histories` function. After the integrity history block:

```rust
    h.push_back(inf.clamp(0.0, 1.0));
```

Add:

```rust
    let h = &mut agent.fatigue_history;
    if h.len() >= cap {
        h.pop_front();
    }
    h.push_back(agent.brain.telemetry().fatigue_factor);
```

- [ ] **Step 3: Wire new fields into TickRecord construction**

In `crates/xagent-sandbox/src/main.rs`, find the TickRecord construction. After:

```rust
                                        phase: GenerationRecording::phase_to_u8(t.behavior_phase()),
```

Add:

```rust
                                        mean_attenuation: t.mean_attenuation,
                                        curiosity_bonus: t.curiosity_bonus,
                                        fatigue_factor: t.fatigue_factor,
                                        motor_variance: t.motor_variance,
```

- [ ] **Step 4: Wire new fields into AgentSnapshot construction**

In `crates/xagent-sandbox/src/main.rs`, find the AgentSnapshot construction. After:

```rust
                                        yaw: a.body.yaw,
```

Add:

```rust
                                        mean_attenuation: telemetry.mean_attenuation,
                                        curiosity_bonus: telemetry.curiosity_bonus,
                                        fatigue_factor: telemetry.fatigue_factor,
                                        motor_variance: telemetry.motor_variance,
                                        fatigue_history: tail(&a.fatigue_history),
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check -p xagent-sandbox 2>&1`
Expected: success (all struct fields now populated)

- [ ] **Step 6: Run full test suite**

Run: `cargo test 2>&1`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/xagent-sandbox/src/main.rs crates/xagent-sandbox/src/agent/mod.rs
git commit -m "feat: wire anti-loop telemetry through histories, TickRecord, AgentSnapshot"
```

---

### Task 8: Display anti-loop telemetry in the agent detail UI

**Files:**
- Modify: `crates/xagent-sandbox/src/ui.rs:786-789,1009-1055`

- [ ] **Step 1: Add anti-loop metrics to the Brain section in Vitals**

In `crates/xagent-sandbox/src/ui.rs`, find the Brain section in the right column of the Vitals/Motor layout:

```rust
                cols[1].label(format!("Gradient: {:+.4}", effective_snap.gradient));
                cols[1].label(format!("Urgency: {:.2}", effective_snap.urgency));
```

Add after the urgency line:

```rust
                cols[1].add_space(4.0);
                cols[1].label(
                    egui::RichText::new(format!(
                        "Habituation: {:.0}%",
                        effective_snap.mean_attenuation * 100.0
                    ))
                    .color(if effective_snap.mean_attenuation < 0.4 {
                        egui::Color32::from_rgb(220, 100, 60)
                    } else {
                        egui::Color32::GRAY
                    }),
                );
                cols[1].label(
                    egui::RichText::new(format!(
                        "Curiosity: {:.3}",
                        effective_snap.curiosity_bonus
                    ))
                    .color(if effective_snap.curiosity_bonus > 0.2 {
                        egui::Color32::from_rgb(80, 200, 80)
                    } else {
                        egui::Color32::GRAY
                    }),
                );
                cols[1].label(
                    egui::RichText::new(format!(
                        "Fatigue: {:.0}%",
                        effective_snap.fatigue_factor * 100.0
                    ))
                    .color(if effective_snap.fatigue_factor < 0.5 {
                        egui::Color32::from_rgb(220, 100, 60)
                    } else {
                        egui::Color32::GRAY
                    }),
                );
                cols[1].label(
                    egui::RichText::new(format!(
                        "Motor var: {:.4}",
                        effective_snap.motor_variance
                    ))
                    .small()
                    .color(egui::Color32::GRAY),
                );
```

- [ ] **Step 2: Add fatigue to the history chart**

Add a 5th series to the chart legend. Find:

```rust
            ui.horizontal(|ui| {
                for (label, color) in [
                    ("Energy", egui::Color32::from_rgb(80, 200, 80)),
                    ("Integrity", egui::Color32::from_rgb(100, 150, 255)),
                    ("Pred. Error", egui::Color32::from_rgb(200, 140, 60)),
                    ("Exploration", egui::Color32::from_rgb(180, 100, 220)),
                ] {
```

Replace with:

```rust
            ui.horizontal(|ui| {
                for (label, color) in [
                    ("Energy", egui::Color32::from_rgb(80, 200, 80)),
                    ("Integrity", egui::Color32::from_rgb(100, 150, 255)),
                    ("Pred. Error", egui::Color32::from_rgb(200, 140, 60)),
                    ("Exploration", egui::Color32::from_rgb(180, 100, 220)),
                    ("Fatigue", egui::Color32::from_rgb(220, 120, 60)),
                ] {
```

Add a 5th series to the data array. Find:

```rust
            let series_data: [(&[f32], egui::Color32); 4] = [
                (&effective_snap.energy_history, egui::Color32::from_rgb(80, 200, 80)),
                (&effective_snap.integrity_history, egui::Color32::from_rgb(100, 150, 255)),
                (&effective_snap.prediction_error_history, egui::Color32::from_rgb(200, 140, 60)),
                (&effective_snap.exploration_rate_history, egui::Color32::from_rgb(180, 100, 220)),
            ];
```

Replace with:

```rust
            let series_data: [(&[f32], egui::Color32); 5] = [
                (&effective_snap.energy_history, egui::Color32::from_rgb(80, 200, 80)),
                (&effective_snap.integrity_history, egui::Color32::from_rgb(100, 150, 255)),
                (&effective_snap.prediction_error_history, egui::Color32::from_rgb(200, 140, 60)),
                (&effective_snap.exploration_rate_history, egui::Color32::from_rgb(180, 100, 220)),
                (&effective_snap.fatigue_history, egui::Color32::from_rgb(220, 120, 60)),
            ];
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p xagent-sandbox 2>&1`
Expected: success

- [ ] **Step 4: Commit**

```bash
git add crates/xagent-sandbox/src/ui.rs
git commit -m "feat: display anti-loop telemetry in agent detail (habituation, curiosity, fatigue)"
```

---

### Task 9: Update headless mode and verify full build

**Files:**
- Modify: `crates/xagent-sandbox/src/headless.rs` (if needed)

- [ ] **Step 1: Check headless mode compiles**

Run: `cargo check -p xagent-sandbox 2>&1`

If headless mode references `BrainTelemetry` or constructs `TickRecord`/`AgentSnapshot`, those call sites need the new fields. Check for compilation errors and fix any missing field initializations.

- [ ] **Step 2: Run full test suite**

Run: `cargo test 2>&1`
Expected: all tests pass (brain: ~64 tests, sandbox: ~27 tests, integration: ~14 tests)

- [ ] **Step 3: Commit if any headless fixes were needed**

```bash
git add crates/xagent-sandbox/src/headless.rs
git commit -m "fix: update headless mode for anti-loop telemetry fields"
```

If no changes were needed, skip this commit.
