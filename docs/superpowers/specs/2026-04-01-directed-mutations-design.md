# Directed Mutations via Per-Parameter Momentum

**Problem:** Mutations are purely random — each parameter is perturbed by a uniform random factor with no memory of what worked before. The system collects mutation tracking data (`mutation_log` table) but never uses it. Evolution is a random walk that happens to be filtered by selection, with no directional intelligence.

**Solution:** Add per-parameter momentum vectors (one set per island) that accumulate directional signal from successful mutations. Future mutations are biased toward directions that previously improved fitness, while retaining random noise for exploration. This provides three emergent properties:

- **A) Directional bias** — momentum IS the directional signal. Parameters trend toward values that improve fitness.
- **B) Correlated mutations** — if two parameters consistently move together in successful offspring, their momentum vectors align naturally without explicit covariance tracking.
- **C) Selective mutation** — parameters with strong momentum get larger effective perturbations; stagnant ones stay near baseline noise. Evolution focuses effort where it matters.

## Mechanism

After each generation, the governor compares successful offspring (those that beat their parent's fitness) to their parent config. For each parameter, it computes the delta (offspring − parent). These deltas are averaged across winners and blended into a decaying momentum:

```
momentum[param] = decay * momentum[param] + (1 - decay) * avg_delta
```

When mutating, each parameter's perturbation combines random noise with a momentum nudge:

```
perturbation = random_noise + momentum[param]
```

When momentum is strong, mutations are pushed in the winning direction. When momentum is near zero, mutations behave like the current random perturbation.

## Parameters

| Config field | Default | Serde default | Purpose |
|---|---|---|---|
| `momentum_decay` | 0.9 | Yes | Per-generation decay factor. Higher = longer memory. |

This is a `GovernorConfig` field (not `BrainConfig` — it governs the evolutionary process, not agent behavior).

## Architecture

One new file, modifications to three existing files.

| Action | File | Change |
|---|---|---|
| Create | `crates/xagent-sandbox/src/momentum.rs` | `MutationMomentum` struct with update, decay, biased perturbation |
| Modify | `crates/xagent-sandbox/src/governor.rs` | Own one `MutationMomentum` per island, feed winners after advance, pass to breed |
| Modify | `crates/xagent-sandbox/src/agent/mod.rs` | `mutate_config_with_strength` accepts `&MutationMomentum`, uses biased perturbation |
| Modify | `crates/xagent-shared/src/config.rs` | Add `momentum_decay` to `GovernorConfig` |

### `MutationMomentum` struct

```rust
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MutationMomentum {
    /// Per-parameter momentum values. Positive = trending upward, negative = trending downward.
    momentum: HashMap<String, f32>,
    /// Decay factor applied each generation (e.g., 0.9).
    decay: f32,
}
```

**Key methods:**

- `new(decay: f32)` — creates empty momentum (all zeros implicitly via empty HashMap)
- `update(parent: &BrainConfig, winners: &[BrainConfig])` — computes avg delta per parameter across winners, blends into momentum: `m = decay * m + (1 - decay) * avg_delta`
- `biased_perturb_f(rng, value, param_name, strength) -> f32` — replacement for `perturb_f` that adds momentum bias. When no momentum exists for a parameter, behaves identically to the current `perturb_f`.
- `biased_perturb_u(rng, value, param_name, strength) -> usize` — same for integer parameters
- `decay_step()` — multiplies all momentum values by `decay`. Called at generation end.
- `get(param_name) -> f32` — returns momentum for a parameter (0.0 if not tracked)

**Biased perturbation formula:**

```rust
fn biased_perturb_f(&self, rng: &mut impl Rng, value: f32, param: &str, strength: f32) -> f32 {
    let lo = 1.0 - strength;
    let hi = 1.0 + strength;
    let random_factor: f32 = rng.random_range(lo..hi);
    let momentum_nudge = self.momentum.get(param).copied().unwrap_or(0.0);
    // Momentum shifts the center of the perturbation range
    let biased_factor = random_factor + momentum_nudge;
    (value * biased_factor).max(0.0001)
}
```

The momentum nudge shifts the perturbation center away from 1.0. If momentum is +0.05, the effective range shifts from [0.9, 1.1] to [0.95, 1.15] — biased upward but still exploring both directions.

### Parameter extraction via name

`BrainConfig` fields are extracted as named `(String, f32/usize)` pairs for momentum tracking. This uses a helper method on `MutationMomentum` that takes the parent and offspring configs directly, computing deltas field-by-field. The field names are string constants matching the struct field names (same pattern as `record_mutations`).

### Integration with governor

**Governor struct changes:**

```rust
pub struct Governor {
    // ...existing fields...
    /// Per-island mutation momentum vectors.
    momentums: Vec<MutationMomentum>,
}
```

**Lifecycle in `advance()`:**

1. After scoring success/failure (existing logic), identify winners: offspring configs whose composite fitness ≥ parent fitness
2. Call `self.momentums[self.active_island].update(parent_config, &winner_configs)`
3. Call `self.momentums[self.active_island].decay_step()`
4. Before breeding: rotate to next island (existing logic)
5. In `breed_next_generation()`: pass `&self.momentums[self.active_island]` to mutation calls

**What counts as a "winner":** An offspring whose individual composite fitness ≥ the spawn parent's stored fitness. This is the same bar used for generation-level success, applied per-agent. This gives momentum a richer signal — even in a "failed" generation, individual high-performers contribute directional data.

**`breed_next_generation()` changes:**

```rust
fn breed_next_generation(&mut self, fitness: &[AgentFitness]) -> Vec<BrainConfig> {
    let momentum = &self.momentums[self.active_island];
    // ...existing logic, but mutate_config_with_strength now takes momentum...
    let base_config = mutate_config_with_strength(&parent_config, effective_strength, momentum);
    // Same for elite mutations and crossover offspring
}
```

**`mutate_config_with_strength()` signature:**

```rust
pub fn mutate_config_with_strength(
    parent: &BrainConfig,
    strength: f32,
    momentum: &MutationMomentum,
) -> BrainConfig {
    // Uses momentum.biased_perturb_f() instead of local perturb_f closure
    // Uses momentum.biased_perturb_u() instead of local perturb_u closure
}
```

### Persistence

Momentum is serialized as JSON and stored in the `run` table alongside governor config.

**Schema change:**

```sql
ALTER TABLE run ADD COLUMN momentum_json TEXT DEFAULT '[]';
```

**Persist:** In `persist_state()`, serialize `self.momentums` to JSON and store in `momentum_json`.

**Resume:** In `resume()`, deserialize `momentum_json` back into `Vec<MutationMomentum>`. If the column is missing or empty (old databases), initialize fresh empty momentum vectors.

**New runs:** `new()` initializes `momentums` as `Vec<MutationMomentum>` with one per island, all empty.

### Migration interaction

When migration spreads the best island's elite config to other islands, only the config migrates. Each island's momentum stays independent. If the migrant config performs well in the receiving island, that island's momentum will naturally update to reflect it. No special momentum transfer logic needed.

### Backtrack interaction

When an island backtracks (exhausts patience), its momentum is preserved. The momentum reflects what directions worked historically, and that knowledge remains relevant even when restarting from an ancestor node. If the directions stop working at the new spawn point, decay will naturally erode them.

## Non-Goals

- No covariance matrix tracking (can be added later if pure momentum proves insufficient for correlated mutations)
- No UI changes (momentum is internal to the mutation process)
- No new telemetry fields (momentum values are not exposed to AgentSnapshot or TickRecord)
- No heritable decay rate (starts as global GovernorConfig value; can be made per-island or heritable later)
- No momentum for learned-state weight mutations (`mutate_learned_state` stays random — momentum is for BrainConfig hyperparameters only)
