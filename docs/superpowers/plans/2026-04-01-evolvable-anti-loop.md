# Evolvable Anti-Loop Parameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote 4 anti-loop constants to heritable BrainConfig fields with stronger defaults so evolution can tune how aggressively each lineage responds to monotony.

**Architecture:** Four module-level constants (2 in habituation.rs, 2 in motor_fatigue.rs) become struct fields initialized from BrainConfig. The config fields are mutated during breeding and crossed over during reproduction, following the existing `distress_exponent` pattern. Defaults shift to steeper values.

**Tech Stack:** Rust, serde, xagent-brain crate, xagent-shared crate, xagent-sandbox crate

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `crates/xagent-shared/src/config.rs` | Add 4 fields to BrainConfig |
| Modify | `crates/xagent-brain/src/habituation.rs` | Replace 2 constants with struct fields |
| Modify | `crates/xagent-brain/src/motor_fatigue.rs` | Replace 2 constants with struct fields |
| Modify | `crates/xagent-brain/src/brain.rs` | Pass config values into constructors |
| Modify | `crates/xagent-sandbox/src/agent/mod.rs` | Mutate + crossover 4 new fields |
| Modify | `crates/xagent-sandbox/src/governor.rs` | Track mutations for 4 new fields |

---

### Task 1: Add 4 new fields to BrainConfig

**Files:**
- Modify: `crates/xagent-shared/src/config.rs`

- [ ] **Step 1: Add the 4 new fields to BrainConfig struct**

In `crates/xagent-shared/src/config.rs`, add these fields after `distress_exponent` (line 27):

```rust
    /// Scales per-dimension variance into attenuation range. Higher = faster boredom.
    /// Heritable: mutated during breeding, clamped to [5.0, 50.0]. Default 20.0.
    #[serde(default = "default_habituation_sensitivity")]
    pub habituation_sensitivity: f32,
    /// Maximum curiosity bonus from sensory monotony. Higher = stronger exploration drive.
    /// Heritable: mutated during breeding, clamped to [0.1, 1.0]. Default 0.6.
    #[serde(default = "default_max_curiosity_bonus")]
    pub max_curiosity_bonus: f32,
    /// Scales motor variance into fatigue relief. Higher = easier recovery from fatigue.
    /// Heritable: mutated during breeding, clamped to [2.0, 20.0]. Default 8.0.
    #[serde(default = "default_fatigue_recovery_sensitivity")]
    pub fatigue_recovery_sensitivity: f32,
    /// Minimum motor output under fatigue. Lower = harsher dampening.
    /// Heritable: mutated during breeding, clamped to [0.05, 0.4]. Default 0.1.
    #[serde(default = "default_fatigue_floor")]
    pub fatigue_floor: f32,
```

- [ ] **Step 2: Add the 4 default functions**

Add these after the existing `default_distress_exponent` function (after line 56):

```rust
fn default_habituation_sensitivity() -> f32 {
    20.0
}

fn default_max_curiosity_bonus() -> f32 {
    0.6
}

fn default_fatigue_recovery_sensitivity() -> f32 {
    8.0
}

fn default_fatigue_floor() -> f32 {
    0.1
}
```

- [ ] **Step 3: Add fields to Default impl**

In the `impl Default for BrainConfig` block (line 149), add after `distress_exponent: 2.0,`:

```rust
            habituation_sensitivity: 20.0,
            max_curiosity_bonus: 0.6,
            fatigue_recovery_sensitivity: 8.0,
            fatigue_floor: 0.1,
```

- [ ] **Step 4: Add fields to tiny() preset**

In the `tiny()` method (line 165), add after `distress_exponent: 2.0,`:

```rust
            habituation_sensitivity: 20.0,
            max_curiosity_bonus: 0.6,
            fatigue_recovery_sensitivity: 8.0,
            fatigue_floor: 0.1,
```

- [ ] **Step 5: Add fields to large() preset**

In the `large()` method (line 178), add after `distress_exponent: 2.0,`:

```rust
            habituation_sensitivity: 20.0,
            max_curiosity_bonus: 0.6,
            fatigue_recovery_sensitivity: 8.0,
            fatigue_floor: 0.1,
```

- [ ] **Step 6: Verify it compiles**

Run: `cargo check -p xagent-shared 2>&1`
Expected: success (warnings OK)

- [ ] **Step 7: Commit**

```bash
git add crates/xagent-shared/src/config.rs
git commit -m "feat: add evolvable anti-loop parameters to BrainConfig"
```

---

### Task 2: Make SensoryHabituation use config values instead of constants

**Files:**
- Modify: `crates/xagent-brain/src/habituation.rs`

- [ ] **Step 1: Update the struct and constructor**

Replace the `SENSITIVITY` and `MAX_CURIOSITY_BONUS` constants (lines 10, 14) with nothing (delete them). Keep `HABITUATION_EMA_ALPHA` (line 8) and `ATTENUATION_FLOOR` (line 12) as they are.

Add two fields to the `SensoryHabituation` struct after `tick: u64` (line 24):

```rust
    sensitivity: f32,
    max_curiosity_bonus: f32,
```

Change the constructor signature from `new(repr_dim: usize)` to:

```rust
    pub fn new(repr_dim: usize, sensitivity: f32, max_curiosity_bonus: f32) -> Self {
        Self {
            prev_encoded: vec![0.0; repr_dim],
            variance_ema: vec![0.0; repr_dim],
            attenuation: vec![1.0; repr_dim],
            habituated: vec![0.0; repr_dim],
            curiosity_bonus: 0.0,
            tick: 0,
            sensitivity,
            max_curiosity_bonus,
        }
    }
```

- [ ] **Step 2: Update the update() method to use struct fields**

In `update()`, change line 50 from:

```rust
            self.attenuation[i] = (self.variance_ema[i] * SENSITIVITY).clamp(ATTENUATION_FLOOR, 1.0);
```

to:

```rust
            self.attenuation[i] = (self.variance_ema[i] * self.sensitivity).clamp(ATTENUATION_FLOOR, 1.0);
```

Change line 61 from:

```rust
        self.curiosity_bonus = (1.0 - mean_attenuation) * MAX_CURIOSITY_BONUS;
```

to:

```rust
        self.curiosity_bonus = (1.0 - mean_attenuation) * self.max_curiosity_bonus;
```

- [ ] **Step 3: Update tests to use new constructor**

Replace every `SensoryHabituation::new(4)` in the test module with `SensoryHabituation::new(4, 20.0, 0.6)`.

There are 5 occurrences (lines 89, 108, 128, 147, 159).

Update the `attenuation_is_bounded` test (line 158): change the assertion `hab.curiosity_bonus() <= MAX_CURIOSITY_BONUS` to `hab.curiosity_bonus() <= 0.6` since the constant is gone.

- [ ] **Step 4: Verify tests pass**

Run: `cargo test -p xagent-brain -- habituation 2>&1`
Expected: 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/xagent-brain/src/habituation.rs
git commit -m "feat: make habituation sensitivity and max curiosity bonus configurable"
```

---

### Task 3: Make MotorFatigue use config values instead of constants

**Files:**
- Modify: `crates/xagent-brain/src/motor_fatigue.rs`

- [ ] **Step 1: Update the struct and constructor**

Delete the `RECOVERY_SENSITIVITY` (line 10) and `FATIGUE_FLOOR` (line 12) constants. Keep `FATIGUE_WINDOW` (line 8).

Add two fields to the `MotorFatigue` struct after `motor_variance: f32` (line 21):

```rust
    recovery_sensitivity: f32,
    fatigue_floor: f32,
```

Change the constructor from `new()` to:

```rust
    pub fn new(recovery_sensitivity: f32, fatigue_floor: f32) -> Self {
        Self {
            forward_ring: vec![0.0; FATIGUE_WINDOW],
            turn_ring: vec![0.0; FATIGUE_WINDOW],
            cursor: 0,
            len: 0,
            fatigue_factor: 1.0,
            motor_variance: 0.0,
            recovery_sensitivity,
            fatigue_floor,
        }
    }
```

- [ ] **Step 2: Update the update() method to use struct fields**

Change line 55-56 from:

```rust
        self.fatigue_factor = (self.motor_variance * RECOVERY_SENSITIVITY)
            .clamp(FATIGUE_FLOOR, 1.0);
```

to:

```rust
        self.fatigue_factor = (self.motor_variance * self.recovery_sensitivity)
            .clamp(self.fatigue_floor, 1.0);
```

- [ ] **Step 3: Update tests to use new constructor**

Replace every `MotorFatigue::new()` in the test module with `MotorFatigue::new(8.0, 0.1)`.

There are 5 occurrences (lines 86, 103, 117, 135, 157).

Update the `fatigue_factor_is_bounded` test (line 134): change the assertion `mf.fatigue_factor() >= FATIGUE_FLOOR` to `mf.fatigue_factor() >= 0.1` since the constant is gone.

Update the `constant_output_causes_fatigue` test (line 85): the fatigue floor changed from 0.2 to 0.1, so the assertion `mf.fatigue_factor() < 0.3` still holds. But verify the variance check: with constant output, variance is ~0, so `0.0 * 8.0 = 0.0`, clamped to `0.1`. The assertion `mf.fatigue_factor() < 0.3` passes since `0.1 < 0.3`.

- [ ] **Step 4: Verify tests pass**

Run: `cargo test -p xagent-brain -- motor_fatigue 2>&1`
Expected: 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/xagent-brain/src/motor_fatigue.rs
git commit -m "feat: make motor fatigue recovery sensitivity and floor configurable"
```

---

### Task 4: Wire config values into Brain constructors

**Files:**
- Modify: `crates/xagent-brain/src/brain.rs:151-162`

- [ ] **Step 1: Update Brain::new() constructor**

Change line 161 from:

```rust
            habituation: SensoryHabituation::new(repr_dim),
```

to:

```rust
            habituation: SensoryHabituation::new(
                repr_dim,
                config.habituation_sensitivity,
                config.max_curiosity_bonus,
            ),
```

Change line 162 from:

```rust
            motor_fatigue: MotorFatigue::new(),
```

to:

```rust
            motor_fatigue: MotorFatigue::new(
                config.fatigue_recovery_sensitivity,
                config.fatigue_floor,
            ),
```

- [ ] **Step 2: Verify full workspace compiles**

Run: `cargo check --workspace 2>&1`
Expected: success (warnings OK)

- [ ] **Step 3: Commit**

```bash
git add crates/xagent-brain/src/brain.rs
git commit -m "feat: wire evolvable anti-loop params from BrainConfig into brain"
```

---

### Task 5: Add mutation and crossover for the 4 new fields

**Files:**
- Modify: `crates/xagent-sandbox/src/agent/mod.rs:255-341`

- [ ] **Step 1: Add mutation for 4 new fields**

In `mutate_config_with_strength()` (line 255), add 4 lines after the `distress_exponent` line (line 276):

```rust
        habituation_sensitivity: perturb_f(&mut rng, parent.habituation_sensitivity).clamp(5.0, 50.0),
        max_curiosity_bonus: perturb_f(&mut rng, parent.max_curiosity_bonus).clamp(0.1, 1.0),
        fatigue_recovery_sensitivity: perturb_f(&mut rng, parent.fatigue_recovery_sensitivity).clamp(2.0, 20.0),
        fatigue_floor: perturb_f(&mut rng, parent.fatigue_floor).clamp(0.05, 0.4),
```

- [ ] **Step 2: Add crossover for 4 new fields**

In `crossover_config()` (line 306), add 4 blocks after the `distress_exponent` block (after line 339):

```rust
        habituation_sensitivity: if rng.random::<f32>() < 0.5 {
            a.habituation_sensitivity
        } else {
            b.habituation_sensitivity
        },
        max_curiosity_bonus: if rng.random::<f32>() < 0.5 {
            a.max_curiosity_bonus
        } else {
            b.max_curiosity_bonus
        },
        fatigue_recovery_sensitivity: if rng.random::<f32>() < 0.5 {
            a.fatigue_recovery_sensitivity
        } else {
            b.fatigue_recovery_sensitivity
        },
        fatigue_floor: if rng.random::<f32>() < 0.5 {
            a.fatigue_floor
        } else {
            b.fatigue_floor
        },
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p xagent-sandbox 2>&1`
Expected: success (warnings OK)

- [ ] **Step 4: Commit**

```bash
git add crates/xagent-sandbox/src/agent/mod.rs
git commit -m "feat: mutate and crossover evolvable anti-loop parameters"
```

---

### Task 6: Add mutation tracking in governor

**Files:**
- Modify: `crates/xagent-sandbox/src/governor.rs:1028-1054`

- [ ] **Step 1: Add 4 new entries to record_mutations**

In the `record_mutations` function, add 4 entries to the `params_to_check` vector after the `"decay_rate"` entry (after line 1053):

```rust
        (
            "distress_exponent",
            parent.distress_exponent as f64,
            child.distress_exponent as f64,
        ),
        (
            "habituation_sensitivity",
            parent.habituation_sensitivity as f64,
            child.habituation_sensitivity as f64,
        ),
        (
            "max_curiosity_bonus",
            parent.max_curiosity_bonus as f64,
            child.max_curiosity_bonus as f64,
        ),
        (
            "fatigue_recovery_sensitivity",
            parent.fatigue_recovery_sensitivity as f64,
            child.fatigue_recovery_sensitivity as f64,
        ),
        (
            "fatigue_floor",
            parent.fatigue_floor as f64,
            child.fatigue_floor as f64,
        ),
```

(Note: `distress_exponent` was also missing from this list — include it now.)

- [ ] **Step 2: Verify all tests pass**

Run: `cargo test --workspace 2>&1`
Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/xagent-sandbox/src/governor.rs
git commit -m "feat: track anti-loop parameter mutations in governor"
```
