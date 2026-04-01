# Directed Mutations via Per-Parameter Momentum — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace purely random mutations with momentum-biased mutations that learn which parameter directions improve fitness, per-island.

**Architecture:** A new `MutationMomentum` struct (one per island) accumulates directional signal from successful offspring. It biases future perturbations toward winning directions while preserving random noise for exploration. Momentum is serialized to the SQLite `run` table for persistence across restarts.

**Tech Stack:** Rust, serde (Serialize/Deserialize), serde_json, rusqlite, rand

---

## File Structure

| Action | File | Responsibility |
|---|---|---|
| Create | `crates/xagent-sandbox/src/momentum.rs` | `MutationMomentum` struct: momentum state, update, decay, biased perturbation |
| Modify | `crates/xagent-sandbox/src/lib.rs` | Register `momentum` module |
| Modify | `crates/xagent-shared/src/config.rs` | Add `momentum_decay` field to `GovernorConfig` |
| Modify | `crates/xagent-sandbox/src/agent/mod.rs` | `mutate_config_with_strength` takes `&MutationMomentum`, uses biased perturbation |
| Modify | `crates/xagent-sandbox/src/governor.rs` | Own `Vec<MutationMomentum>`, feed winners, persist/resume, pass to breed |
| Modify | `README.md` | Update evolution section |
| Modify | `crates/xagent-sandbox/README.md` | Update mutation description |
| Modify | `EVOLUTION_JOURNEY.md` | Add directed-mutations milestone |

---

### Task 1: `MutationMomentum` struct with tests

**Files:**
- Create: `crates/xagent-sandbox/src/momentum.rs`
- Modify: `crates/xagent-sandbox/src/lib.rs`

- [ ] **Step 1: Create `momentum.rs` with struct, constructor, `get`, `decay_step`, and serialization**

```rust
//! Per-parameter mutation momentum for directed evolution.
//!
//! Each island maintains its own momentum vector that biases future mutations
//! toward directions that previously improved fitness. Momentum decays each
//! generation so stale signals fade naturally.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Accumulates per-parameter directional signal from successful mutations.
///
/// After each generation, winning offspring (those that beat their parent's
/// fitness) contribute their mutation deltas to the momentum. Future
/// perturbations are biased in the momentum direction — parameters with
/// strong momentum get pushed toward winning values, while parameters with
/// weak momentum stay near random noise.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MutationMomentum {
    /// Per-parameter momentum. Positive = trending upward, negative = trending down.
    momentum: HashMap<String, f32>,
    /// Per-generation decay factor (e.g., 0.9). Applied multiplicatively.
    decay: f32,
}

impl MutationMomentum {
    /// Create a new momentum tracker with the given decay rate.
    pub fn new(decay: f32) -> Self {
        Self {
            momentum: HashMap::new(),
            decay,
        }
    }

    /// Get momentum for a parameter (0.0 if not tracked).
    pub fn get(&self, param: &str) -> f32 {
        self.momentum.get(param).copied().unwrap_or(0.0)
    }

    /// Decay all momentum values by the decay factor.
    pub fn decay_step(&mut self) {
        for v in self.momentum.values_mut() {
            *v *= self.decay;
        }
        // Remove near-zero entries to keep the map clean
        self.momentum.retain(|_, v| v.abs() > 1e-8);
    }
}
```

- [ ] **Step 2: Register the module in `lib.rs`**

Add after `pub mod governor;` in `crates/xagent-sandbox/src/lib.rs`:

```rust
pub mod momentum;
```

- [ ] **Step 3: Write unit tests for constructor, get, decay, and serialization**

Add at the bottom of `momentum.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_momentum_returns_zero() {
        let m = MutationMomentum::new(0.9);
        assert_eq!(m.get("learning_rate"), 0.0);
        assert_eq!(m.get("nonexistent"), 0.0);
    }

    #[test]
    fn decay_reduces_values() {
        let mut m = MutationMomentum::new(0.5);
        m.momentum.insert("learning_rate".into(), 1.0);
        m.momentum.insert("decay_rate".into(), -0.8);

        m.decay_step();

        assert!((m.get("learning_rate") - 0.5).abs() < 1e-6);
        assert!((m.get("decay_rate") - (-0.4)).abs() < 1e-6);
    }

    #[test]
    fn decay_cleans_near_zero_entries() {
        let mut m = MutationMomentum::new(0.1);
        m.momentum.insert("tiny".into(), 1e-7);
        m.momentum.insert("big".into(), 1.0);

        m.decay_step();

        assert_eq!(m.get("tiny"), 0.0); // removed
        assert!((m.get("big") - 0.1).abs() < 1e-6); // kept
    }

    #[test]
    fn serialization_round_trip() {
        let mut m = MutationMomentum::new(0.9);
        m.momentum.insert("learning_rate".into(), 0.05);
        m.momentum.insert("decay_rate".into(), -0.03);

        let json = serde_json::to_string(&m).unwrap();
        let restored: MutationMomentum = serde_json::from_str(&json).unwrap();

        assert!((restored.decay - 0.9).abs() < 1e-6);
        assert!((restored.get("learning_rate") - 0.05).abs() < 1e-6);
        assert!((restored.get("decay_rate") - (-0.03)).abs() < 1e-6);
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p xagent-sandbox momentum`
Expected: 4 tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/xagent-sandbox/src/momentum.rs crates/xagent-sandbox/src/lib.rs
git commit -m "feat: MutationMomentum struct with decay and serialization"
```

---

### Task 2: `update` method — accumulate directional signal from winners

**Files:**
- Modify: `crates/xagent-sandbox/src/momentum.rs`

This task adds the core learning logic: after each generation, compare winning offspring to their parent and blend directional deltas into momentum.

- [ ] **Step 1: Write failing tests for `update`**

Add the import to the `tests` module in `momentum.rs`:

```rust
    use xagent_shared::BrainConfig;
```

Add these tests:

```rust
    #[test]
    fn update_builds_positive_momentum() {
        let mut m = MutationMomentum::new(0.9);

        let parent = BrainConfig {
            learning_rate: 0.05,
            decay_rate: 0.001,
            ..BrainConfig::default()
        };

        // Winner has higher learning_rate, same decay_rate
        let winners = vec![BrainConfig {
            learning_rate: 0.06,
            decay_rate: 0.001,
            ..BrainConfig::default()
        }];

        m.update(&parent, &winners);

        // learning_rate delta = 0.06 - 0.05 = 0.01
        // momentum = 0.9 * 0.0 + 0.1 * 0.01 = 0.001
        assert!(m.get("learning_rate") > 0.0);
        // decay_rate unchanged — no momentum
        assert_eq!(m.get("decay_rate"), 0.0);
    }

    #[test]
    fn update_accumulates_across_calls() {
        let mut m = MutationMomentum::new(0.9);

        let parent = BrainConfig::default();
        let winners = vec![BrainConfig {
            learning_rate: parent.learning_rate + 0.01,
            ..BrainConfig::default()
        }];

        m.update(&parent, &winners);
        let after_one = m.get("learning_rate");

        m.update(&parent, &winners);
        let after_two = m.get("learning_rate");

        // Second update should strengthen momentum in same direction
        assert!(after_two > after_one);
    }

    #[test]
    fn update_opposing_signals_cancel() {
        let mut m = MutationMomentum::new(0.5); // fast decay for cleaner test

        let parent = BrainConfig::default();

        // First: winner increases learning_rate
        let up = vec![BrainConfig {
            learning_rate: parent.learning_rate + 0.02,
            ..BrainConfig::default()
        }];
        m.update(&parent, &up);
        let after_up = m.get("learning_rate");
        assert!(after_up > 0.0);

        // Second: winner decreases learning_rate by same amount
        let down = vec![BrainConfig {
            learning_rate: parent.learning_rate - 0.02,
            ..BrainConfig::default()
        }];
        m.update(&parent, &down);
        let after_down = m.get("learning_rate");

        // Should be smaller in magnitude than after_up (partially canceled)
        assert!(after_down.abs() < after_up.abs());
    }

    #[test]
    fn update_no_winners_is_noop() {
        let mut m = MutationMomentum::new(0.9);
        m.momentum.insert("learning_rate".into(), 0.05);

        let parent = BrainConfig::default();
        m.update(&parent, &[]); // no winners

        // Momentum unchanged
        assert!((m.get("learning_rate") - 0.05).abs() < 1e-6);
    }

    #[test]
    fn update_averages_across_multiple_winners() {
        let mut m = MutationMomentum::new(0.9);

        let parent = BrainConfig {
            learning_rate: 0.05,
            ..BrainConfig::default()
        };

        // Two winners: one went up +0.02, the other up +0.04
        // Average delta = +0.03
        let winners = vec![
            BrainConfig {
                learning_rate: 0.07,
                ..BrainConfig::default()
            },
            BrainConfig {
                learning_rate: 0.09,
                ..BrainConfig::default()
            },
        ];

        m.update(&parent, &winners);

        // momentum = 0.9 * 0.0 + 0.1 * 0.03 = 0.003
        let val = m.get("learning_rate");
        assert!((val - 0.003).abs() < 1e-5);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p xagent-sandbox momentum`
Expected: 5 new tests fail (method `update` not found)

- [ ] **Step 3: Implement `update`**

Add the import at the top of `momentum.rs`:

```rust
use xagent_shared::BrainConfig;
```

Add this method to the `impl MutationMomentum` block, after `decay_step`:

```rust
    /// Update momentum from a set of winning offspring.
    ///
    /// For each BrainConfig parameter, computes the average delta between parent
    /// and winners, then blends into momentum:
    ///   momentum[p] = decay * momentum[p] + (1 - decay) * avg_delta[p]
    ///
    /// Does nothing if `winners` is empty (no signal to learn from).
    pub fn update(&mut self, parent: &BrainConfig, winners: &[BrainConfig]) {
        if winners.is_empty() {
            return;
        }
        let n = winners.len() as f32;
        let blend = 1.0 - self.decay;

        let params: Vec<(&str, f32)> = vec![
            ("memory_capacity", parent.memory_capacity as f32),
            ("processing_slots", parent.processing_slots as f32),
            ("representation_dim", parent.representation_dim as f32),
            ("learning_rate", parent.learning_rate),
            ("decay_rate", parent.decay_rate),
            ("distress_exponent", parent.distress_exponent),
            ("habituation_sensitivity", parent.habituation_sensitivity),
            ("max_curiosity_bonus", parent.max_curiosity_bonus),
            ("fatigue_recovery_sensitivity", parent.fatigue_recovery_sensitivity),
            ("fatigue_floor", parent.fatigue_floor),
        ];

        for (name, parent_val) in &params {
            let avg_delta: f32 = winners
                .iter()
                .map(|w| {
                    let w_val = match *name {
                        "memory_capacity" => w.memory_capacity as f32,
                        "processing_slots" => w.processing_slots as f32,
                        "representation_dim" => w.representation_dim as f32,
                        "learning_rate" => w.learning_rate,
                        "decay_rate" => w.decay_rate,
                        "distress_exponent" => w.distress_exponent,
                        "habituation_sensitivity" => w.habituation_sensitivity,
                        "max_curiosity_bonus" => w.max_curiosity_bonus,
                        "fatigue_recovery_sensitivity" => w.fatigue_recovery_sensitivity,
                        "fatigue_floor" => w.fatigue_floor,
                        _ => *parent_val,
                    };
                    w_val - parent_val
                })
                .sum::<f32>()
                / n;

            if avg_delta.abs() > 1e-8 {
                let current = self.momentum.get(*name).copied().unwrap_or(0.0);
                let updated = self.decay * current + blend * avg_delta;
                if updated.abs() > 1e-8 {
                    self.momentum.insert((*name).to_string(), updated);
                } else {
                    self.momentum.remove(*name);
                }
            }
        }
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p xagent-sandbox momentum`
Expected: All 9 tests pass (4 from Task 1 + 5 new)

- [ ] **Step 5: Commit**

```bash
git add crates/xagent-sandbox/src/momentum.rs
git commit -m "feat: MutationMomentum::update accumulates directional signal from winners"
```

---

### Task 3: Biased perturbation methods

**Files:**
- Modify: `crates/xagent-sandbox/src/momentum.rs`

- [ ] **Step 1: Write failing tests for `biased_perturb_f` and `biased_perturb_u`**

Add to the `tests` module in `momentum.rs`:

```rust
    #[test]
    fn biased_perturb_f_no_momentum_stays_in_range() {
        let m = MutationMomentum::new(0.9); // empty momentum
        let mut rng = rand::rng();

        let value = 1.0;
        for _ in 0..100 {
            let result = m.biased_perturb_f(&mut rng, value, "learning_rate", 0.1);
            assert!(result >= 0.0001);
            // Without momentum, factor is in [0.9, 1.1], so result in [0.9, 1.1]
            assert!(result >= 0.89 && result <= 1.11,
                "result {} out of expected range", result);
        }
    }

    #[test]
    fn biased_perturb_f_with_momentum_shifts_distribution() {
        let mut m = MutationMomentum::new(0.9);
        m.momentum.insert("learning_rate".into(), 0.1);

        let mut rng = rand::rng();
        let value = 1.0;

        let mut sum = 0.0;
        let n = 1000;
        for _ in 0..n {
            sum += m.biased_perturb_f(&mut rng, value, "learning_rate", 0.1);
        }
        let avg = sum / n as f32;

        // Average should be above 1.0 (biased upward by momentum)
        assert!(avg > 1.0, "avg {} should be > 1.0 with positive momentum", avg);
    }

    #[test]
    fn biased_perturb_u_no_momentum_stays_reasonable() {
        let m = MutationMomentum::new(0.9);
        let mut rng = rand::rng();

        let value: usize = 100;
        for _ in 0..100 {
            let result = m.biased_perturb_u(&mut rng, value, "memory_capacity", 0.1);
            assert!(result >= 1);
            assert!(result >= 85 && result <= 115,
                "result {} out of expected range", result);
        }
    }

    #[test]
    fn biased_perturb_f_respects_min_clamp() {
        let mut m = MutationMomentum::new(0.9);
        m.momentum.insert("fatigue_floor".into(), -10.0);

        let mut rng = rand::rng();
        for _ in 0..100 {
            let result = m.biased_perturb_f(&mut rng, 0.001, "fatigue_floor", 0.5);
            assert!(result >= 0.0001, "result {} below min clamp", result);
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p xagent-sandbox momentum`
Expected: 4 new tests fail (methods not found)

- [ ] **Step 3: Implement `biased_perturb_f` and `biased_perturb_u`**

Add these methods to the `impl MutationMomentum` block:

```rust
    /// Biased perturbation for f32 parameters.
    ///
    /// Combines a random multiplicative factor (same as the old `perturb_f`)
    /// with an additive momentum nudge that shifts the center of perturbation.
    pub fn biased_perturb_f(
        &self,
        rng: &mut impl Rng,
        value: f32,
        param: &str,
        strength: f32,
    ) -> f32 {
        let lo = 1.0 - strength;
        let hi = 1.0 + strength;
        let random_factor: f32 = rng.random_range(lo..hi);
        let nudge = self.get(param);
        let biased_factor = random_factor + nudge;
        (value * biased_factor).max(0.0001)
    }

    /// Biased perturbation for usize parameters.
    pub fn biased_perturb_u(
        &self,
        rng: &mut impl Rng,
        value: usize,
        param: &str,
        strength: f32,
    ) -> usize {
        let lo = 1.0 - strength;
        let hi = 1.0 + strength;
        let random_factor: f32 = rng.random_range(lo..hi);
        let nudge = self.get(param);
        let biased_factor = random_factor + nudge;
        ((value as f32 * biased_factor).round() as usize).max(1)
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p xagent-sandbox momentum`
Expected: All 13 tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/xagent-sandbox/src/momentum.rs
git commit -m "feat: biased perturbation methods for momentum-directed mutations"
```

---

### Task 4: Add `momentum_decay` to `GovernorConfig`

**Files:**
- Modify: `crates/xagent-shared/src/config.rs`
- Modify: `crates/xagent-sandbox/src/governor.rs` (test fixtures)

- [ ] **Step 1: Add the field and default function to `GovernorConfig`**

In `crates/xagent-shared/src/config.rs`, add after the `migration_interval` field (line 146):

```rust
    /// Decay factor for per-island mutation momentum (0.0–1.0).
    /// Higher = longer memory of winning mutation directions.
    #[serde(default = "default_momentum_decay")]
    pub momentum_decay: f32,
```

Add the default function after `default_migration_interval` (around line 161):

```rust
fn default_momentum_decay() -> f32 {
    0.9
}
```

Update the `Default` impl for `GovernorConfig` (around line 176) to include:

```rust
            momentum_decay: 0.9,
```

- [ ] **Step 2: Fix all `GovernorConfig` struct literals in governor.rs tests**

Every `GovernorConfig` literal in `crates/xagent-sandbox/src/governor.rs` needs `momentum_decay: 0.9,` added. Search for `GovernorConfig {` in that file — there are fixtures at approximately lines 1099, 1416, 1464, 1577, 1646, 1699, 1728, 1901. Add the field to each.

- [ ] **Step 3: Run tests to verify everything compiles and passes**

Run: `cargo test -p xagent-sandbox`
Expected: All existing tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/xagent-shared/src/config.rs crates/xagent-sandbox/src/governor.rs
git commit -m "feat: add momentum_decay field to GovernorConfig"
```

---

### Task 5: Wire `mutate_config_with_strength` to use momentum

**Files:**
- Modify: `crates/xagent-sandbox/src/agent/mod.rs`

- [ ] **Step 1: Update `mutate_config_with_strength` signature and body**

In `crates/xagent-sandbox/src/agent/mod.rs`, add the import near the top:

```rust
use crate::momentum::MutationMomentum;
```

Change the function signature at line 255 from:

```rust
pub fn mutate_config_with_strength(parent: &BrainConfig, strength: f32) -> BrainConfig {
```

To:

```rust
pub fn mutate_config_with_strength(
    parent: &BrainConfig,
    strength: f32,
    momentum: &MutationMomentum,
) -> BrainConfig {
```

Replace the function body (lines 256–282). Remove the `perturb_f` and `perturb_u` closures and the `rng`/`lo`/`hi` setup. New body:

```rust
    let mut rng = rand::rng();

    BrainConfig {
        memory_capacity: momentum.biased_perturb_u(&mut rng, parent.memory_capacity, "memory_capacity", strength),
        processing_slots: momentum.biased_perturb_u(&mut rng, parent.processing_slots, "processing_slots", strength),
        visual_encoding_size: parent.visual_encoding_size,
        representation_dim: momentum.biased_perturb_u(&mut rng, parent.representation_dim, "representation_dim", strength).min(MAX_REPR_DIM),
        learning_rate: momentum.biased_perturb_f(&mut rng, parent.learning_rate, "learning_rate", strength),
        decay_rate: momentum.biased_perturb_f(&mut rng, parent.decay_rate, "decay_rate", strength),
        distress_exponent: momentum.biased_perturb_f(&mut rng, parent.distress_exponent, "distress_exponent", strength).clamp(1.5, 5.0),
        habituation_sensitivity: momentum.biased_perturb_f(&mut rng, parent.habituation_sensitivity, "habituation_sensitivity", strength).clamp(5.0, 50.0),
        max_curiosity_bonus: momentum.biased_perturb_f(&mut rng, parent.max_curiosity_bonus, "max_curiosity_bonus", strength).clamp(0.1, 1.0),
        fatigue_recovery_sensitivity: momentum.biased_perturb_f(&mut rng, parent.fatigue_recovery_sensitivity, "fatigue_recovery_sensitivity", strength).clamp(2.0, 20.0),
        fatigue_floor: momentum.biased_perturb_f(&mut rng, parent.fatigue_floor, "fatigue_floor", strength).clamp(0.05, 0.4),
    }
```

- [ ] **Step 2: Update `mutate_config` wrapper**

Update the `mutate_config` function at line 249 to pass empty momentum:

```rust
pub fn mutate_config(parent: &BrainConfig) -> BrainConfig {
    mutate_config_with_strength(parent, 0.1, &MutationMomentum::new(0.9))
}
```

- [ ] **Step 3: Update call sites in `governor.rs`**

In `crates/xagent-sandbox/src/governor.rs`, add the import:

```rust
use crate::momentum::MutationMomentum;
```

In `breed_next_generation()` (around line 609), after retrieving `parent_config` (line ~620) and computing `effective_strength` (line ~630), add:

```rust
        let momentum = &self.momentums[self.active_island];
```

Then update every `mutate_config_with_strength` call in that function to pass `momentum`:

- Line ~632: `let base_config = mutate_config_with_strength(&parent_config, effective_strength, momentum);`
- Line ~683: `unique_configs.push(mutate_config_with_strength(elite, effective_strength, momentum));`
- Line ~700: `unique_configs.push(mutate_config_with_strength(&child, effective_strength, momentum));`
- Line ~706: `unique_configs.push(mutate_config_with_strength(&parent_config, effective_strength, momentum));`

**Note:** This will not compile yet because `self.momentums` doesn't exist on `Governor`. Add a temporary field to make it compile — Task 6 will fill in the full lifecycle. Add to the `Governor` struct:

```rust
    /// Per-island mutation momentum vectors.
    pub momentums: Vec<MutationMomentum>,
```

And initialize in both `new()` and `resume()` with:

```rust
            momentums: (0..num_islands).map(|_| MutationMomentum::new(config.momentum_decay)).collect(),
```

Also add `use serde::{Deserialize, Serialize};` to governor.rs imports if only `Serialize` is imported.

- [ ] **Step 4: Run build to check compilation**

Run: `cargo build -p xagent-sandbox 2>&1 | head -30`
Expected: Compiles successfully

- [ ] **Step 5: Run tests**

Run: `cargo test -p xagent-sandbox`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/xagent-sandbox/src/agent/mod.rs crates/xagent-sandbox/src/governor.rs
git commit -m "feat: mutate_config_with_strength uses momentum-biased perturbation"
```

---

### Task 6: Governor momentum lifecycle — winners, persistence, resume

**Files:**
- Modify: `crates/xagent-sandbox/src/governor.rs`
- Modify: `crates/xagent-sandbox/src/momentum.rs` (test helper)

This task wires up the full momentum lifecycle: identifying winners after each generation, updating momentum, persisting to SQLite, and restoring on resume.

- [ ] **Step 1: Add `momentum_json` column to schema**

In the `init_schema` function (around line 968), after the main `db.execute_batch(...)` call, add a migration:

```rust
    // Backwards-compatible migration: add momentum_json if missing
    let _ = db.execute_batch(
        "ALTER TABLE run ADD COLUMN momentum_json TEXT DEFAULT '[]';"
    );
```

- [ ] **Step 2: Add `spawn_parent_config` helper method**

Add to `impl Governor`, near `spawn_parent_fitness()`:

```rust
    /// Get the spawn parent's BrainConfig for the active island.
    fn spawn_parent_config(&self) -> Option<BrainConfig> {
        let spawn_id = self.islands.get(self.active_island)?.spawn_parent_id?;
        let json: String = self
            .db
            .query_row(
                "SELECT config_json FROM node WHERE id = ?1",
                params![spawn_id],
                |row| row.get(0),
            )
            .ok()?;
        serde_json::from_str(&json).ok()
    }
```

- [ ] **Step 3: Update `advance()` to feed winners into momentum**

In `advance()`, after the backtracking check (after the `if should_backtrack` block, around line 466) and before the completion check (around line 470), insert:

```rust
        // ── Momentum update: learn from individual winners ──────────────
        // An individual "winner" is an offspring whose fitness ≥ spawn parent.
        // Even in a failed generation, strong individuals contribute directional data.
        if let Some(ref pc) = self.spawn_parent_config() {
            let winner_configs: Vec<BrainConfig> = reduced
                .iter()
                .filter(|f| f.composite_fitness >= parent_fitness)
                .map(|f| f.config.clone())
                .collect();
            self.momentums[self.active_island].update(pc, &winner_configs);
            self.momentums[self.active_island].decay_step();
        }
```

- [ ] **Step 4: Update `persist_state()` to serialize momentum**

Replace the `persist_state` method body:

```rust
    fn persist_state(&self) {
        let best_score = self.best_score();
        let spawn_parent = self.islands.get(self.active_island).and_then(|i| i.spawn_parent_id);
        let momentum_json = serde_json::to_string(&self.momentums).unwrap_or_else(|_| "[]".into());
        let _ = self.db.execute(
            "UPDATE run SET best_score = ?1, spawn_parent_id = ?2, momentum_json = ?3 WHERE id = ?4",
            params![best_score as f64, spawn_parent, momentum_json, self.run_id],
        );
    }
```

- [ ] **Step 5: Update `resume()` to deserialize momentum**

In `resume()`, update the query to also fetch `momentum_json`. Change the destructuring from:

```rust
        let (run_id, governor_json, spawn_parent_id): (i64, String, Option<i64>) =
            db.query_row(
                "SELECT id, governor_config, spawn_parent_id
                 FROM run ORDER BY id DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
```

To:

```rust
        let (run_id, governor_json, spawn_parent_id, momentum_json): (i64, String, Option<i64>, String) =
            db.query_row(
                "SELECT id, governor_config, spawn_parent_id,
                        COALESCE(momentum_json, '[]')
                 FROM run ORDER BY id DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )?;
```

After constructing `islands`, replace the temporary momentum initialization with:

```rust
        let mut momentums: Vec<MutationMomentum> =
            serde_json::from_str(&momentum_json).unwrap_or_default();
        // Ensure we have one momentum per island (handles old DBs or island count changes)
        while momentums.len() < num_islands {
            momentums.push(MutationMomentum::new(config.momentum_decay));
        }
        momentums.truncate(num_islands);
```

- [ ] **Step 6: Add test helper `momentum_mut` to `MutationMomentum`**

In `crates/xagent-sandbox/src/momentum.rs`, add to the `impl MutationMomentum` block:

```rust
    /// Mutable access to the momentum map (for testing).
    #[cfg(test)]
    pub fn momentum_mut(&mut self) -> &mut HashMap<String, f32> {
        &mut self.momentum
    }
```

- [ ] **Step 7: Write integration test for momentum persistence**

Add to the `tests` module in `governor.rs`:

```rust
    #[test]
    fn momentum_persists_across_resume() {
        use std::fs;
        use crate::momentum::MutationMomentum;

        let db_path = "/tmp/xagent_test_momentum_persist.db";
        let _ = fs::remove_file(db_path);

        let config = GovernorConfig {
            population_size: 10,
            tick_budget: 100,
            elitism_count: 3,
            patience: 5,
            max_generations: 0,
            mutation_strength: 0.1,
            eval_repeats: 1,
            num_islands: 2,
            migration_interval: 0,
            momentum_decay: 0.9,
        };
        let brain = BrainConfig::default();

        // Create governor, inject momentum, advance (which persists)
        {
            let mut gov = Governor::new(db_path, config.clone(), &brain, "{}").unwrap();
            gov.momentums[0].momentum_mut().insert("learning_rate".into(), 0.05);
            gov.momentums[1].momentum_mut().insert("decay_rate".into(), -0.03);
            gov.advance(&mock_fitness(0.1)); // triggers persist_state
        }

        // Resume and verify momentum survived
        {
            let gov = Governor::resume(db_path).unwrap();
            assert_eq!(gov.momentums.len(), 2);
            // Momentum was persisted after advance, which calls decay_step.
            // The injected values were also blended with winner data during advance.
            // Key assertion: momentum vectors exist and were deserialized (not empty defaults).
            // We can't predict exact values since advance() also updates momentum,
            // but at minimum the map shouldn't be entirely empty.
            let has_data = gov.momentums[0].get("learning_rate") != 0.0
                || gov.momentums[1].get("decay_rate") != 0.0;
            assert!(has_data, "momentum should have been persisted and restored");
        }

        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn governor_initializes_momentum_per_island() {
        let config = GovernorConfig {
            population_size: 10,
            tick_budget: 100,
            elitism_count: 3,
            patience: 5,
            max_generations: 0,
            mutation_strength: 0.1,
            eval_repeats: 1,
            num_islands: 3,
            migration_interval: 0,
            momentum_decay: 0.9,
        };
        let brain = BrainConfig::default();
        let gov = Governor::new(":memory:", config, &brain, "{}").unwrap();
        assert_eq!(gov.momentums.len(), 3);
    }
```

- [ ] **Step 8: Run all tests**

Run: `cargo test -p xagent-sandbox`
Expected: All tests pass

- [ ] **Step 9: Commit**

```bash
git add crates/xagent-sandbox/src/governor.rs crates/xagent-sandbox/src/momentum.rs
git commit -m "feat: integrate MutationMomentum into Governor lifecycle with persistence"
```

---

### Task 7: Fix remaining call sites and full workspace build

**Files:**
- Possibly modify: `crates/xagent-sandbox/src/main.rs`
- Possibly modify: `crates/xagent-sandbox/src/headless.rs`

- [ ] **Step 1: Search for remaining `mutate_config_with_strength` call sites**

Run: `grep -rn "mutate_config_with_strength\|mutate_config(" crates/xagent-sandbox/src/ --include="*.rs" | grep -v "^crates/xagent-sandbox/src/agent/mod.rs" | grep -v "^crates/xagent-sandbox/src/governor.rs"`

Any call sites outside `agent/mod.rs` and `governor.rs` that use `mutate_config_with_strength` directly need the new `momentum` parameter. Sites using `mutate_config()` (the wrapper) are fine — it passes empty momentum.

- [ ] **Step 2: Fix any remaining call sites**

For each call site found, either:
- Change to use `mutate_config()` (the wrapper that passes empty momentum), or
- Pass a `&MutationMomentum::new(0.9)` if the site should use momentum

- [ ] **Step 3: Build and test the full workspace**

Run: `cargo test --workspace`
Expected: All tests pass, clean build

- [ ] **Step 4: Commit (if any changes were needed)**

```bash
git add -A
git commit -m "fix: update remaining mutate_config call sites for momentum parameter"
```

---

### Task 8: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `crates/xagent-sandbox/README.md`
- Modify: `EVOLUTION_JOURNEY.md`

- [ ] **Step 1: Update root `README.md`**

Find the evolution-related content (search for "mutation" or "evolution" — around line 168 or wherever the reproduction/mutation strategy is described). After the existing mutation description, add:

```markdown
- **Directed mutations**: Per-parameter momentum vectors (one per island) bias mutations toward directions that previously improved fitness. Momentum decays each generation (configurable via `momentum_decay`) so stale signals fade. This provides directional bias, emergent correlated mutations, and selective mutation focus — all without hardcoded parameter relationships.
```

- [ ] **Step 2: Update sandbox `README.md`**

In `crates/xagent-sandbox/README.md`, find the "Reproduction" section (around line 913–921). Replace:

```markdown
- **Mutation**: Each `BrainConfig` parameter is independently perturbed ±10%.
  `visual_encoding_size` is preserved (must match the sensory pipeline).
```

With:

```markdown
- **Mutation**: Each `BrainConfig` parameter is perturbed using momentum-biased perturbation. Each island maintains a per-parameter momentum vector that learns which mutation directions improve fitness. The perturbation combines random noise (±strength%) with a directional nudge from momentum. Parameters with strong momentum are pushed toward winning values; parameters with weak momentum get mostly random exploration. `visual_encoding_size` is preserved (must match the sensory pipeline).
```

- [ ] **Step 3: Update `EVOLUTION_JOURNEY.md`**

Add a new entry at the end of the "Timeline of Issues Found" section (before "The Disconnect" section, around line 139). This is a capability milestone:

```markdown
### 14. Mutations Were Random Walks

**Symptom:** Evolution progressed slowly. Each mutation was a coin flip — no learning from what worked before.

**Root cause:** `mutate_config_with_strength` applied uniform random perturbation (±strength%) to every parameter independently. The mutation tracking data (`mutation_log` table) was collected but never used. Evolution was a random walk filtered by selection, with no directional intelligence.

**Fix:** Per-parameter momentum vectors (one set per island). After each generation, offspring that beat their parent contribute their mutation deltas to an exponentially-decaying momentum. Future perturbations are biased toward winning directions. This provides: (A) directional bias — parameters trend toward values that improve fitness, (B) emergent correlated mutations — parameters that consistently move together develop aligned momentum, (C) selective focus — parameters with strong signal get larger perturbations while stagnant ones stay near random noise.

**Lesson:** Evolution with memory is faster than evolution without. The same principle that makes gradient descent faster than random search applies to neuroevolution — you don't need exact gradients, just a noisy directional signal accumulated over time.
```

- [ ] **Step 4: Run a final build and test**

Run: `cargo test --workspace`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add README.md crates/xagent-sandbox/README.md EVOLUTION_JOURNEY.md
git commit -m "docs: document directed mutations in README and EVOLUTION_JOURNEY"
```
