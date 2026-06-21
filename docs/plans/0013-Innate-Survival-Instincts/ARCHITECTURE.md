# Architecture — Plan 0013 (deltas)

> Edits in `crates/xagent-shared/src/config.rs`,
> `crates/xagent-brain/src/buffers.rs`,
> `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-sandbox/src/agent/mod.rs`,
> `crates/xagent-sandbox/src/headless.rs`, and the decision doc
> `docs/plans/0013-Innate-Survival-Instincts/0013-INNATE-INSTINCT-DECISION.md`.
> Line numbers are hints; locate by symbol.

## 0001 — Innate-Pattern-Seeding

Today `init_pattern_memory()` (`buffers.rs:803-804`) allocates a zero-filled
vector of `PATTERN_STRIDE` f32s, then `reset_agents_with_rng()`
(`gpu_kernel.rs:527-571`) writes it to the GPU with no priors. The memory layout
is fixed: `O_PAT_STATES` (128-D encoded state vector per slot × `MEMORY_CAP`),
`O_PAT_NORMS`, `O_PAT_REINF`, `O_PAT_MOTOR` (forward, turn, valence ×
`MEMORY_CAP`), `O_PAT_META` (created_at, last_accessed, activation_count),
`O_PAT_ACTIVE`, `O_ACTIVE_COUNT`, `O_MIN_REINF_IDX`, `O_LAST_STORED_IDX` (see
`buffers.rs:130-141`). The memory-blend kernel (`brain_passes.wgsl:1317-1341`)
already weights recalled memories by cosine similarity × valence and mixes the
recalled motor output into learned policy at `MEMORY_BLEND_STRENGTH=0.4`
(`common.wgsl:517`), so seeded patterns with pre-filled motor vectors and valence
signatures need no new shader logic — they are indistinguishable from learned
patterns once in the buffer.

Edits:

- **New heritable genes in `BrainConfig`** (`config.rs`, after `orientation_offset`,
  `config.rs:150-178`): two f32 multipliers controlling the seeded valence
  magnitudes, each with serde defaults and clamp bounds `[0.1, 1.0]`.

```rust
/// Heritable (innate instincts). Multiplier for the danger instinct pattern's
/// negative valence; seed 0.8, mutated during breeding, clamped to
/// [INSTINCT_DANGER_STRENGTH_MIN, INSTINCT_DANGER_STRENGTH_MAX].
#[serde(default = "default_instinct_danger_strength")]
pub instinct_danger_strength: f32,
/// Heritable (innate instincts). Multiplier for the food instinct pattern's
/// positive valence; seed 0.8, mutated during breeding, clamped to
/// [INSTINCT_FOOD_STRENGTH_MIN, INSTINCT_FOOD_STRENGTH_MAX].
#[serde(default = "default_instinct_food_strength")]
pub instinct_food_strength: f32,
```

- **New seeding function** in `buffers.rs` after `init_pattern_memory()`
  (`buffers.rs:803`): populates two instinct slots into an otherwise zero-filled
  `PATTERN_STRIDE`-length vector, scaling each valence by its strength gene.

```rust
/// Seed two innate instinct patterns into the pattern buffer for one agent.
/// Slot 0: danger context — high negative valence + avoidance motor priors.
/// Slot 1: energy-gain context — high positive valence + approach motor priors.
/// Both decay/reinforce/evict via normal shader logic, like learned patterns.
pub fn seed_instinct_patterns(danger_strength: f32, food_strength: f32) -> Vec<f32> {
    // Encoded state -0.5 (danger) / +0.5 (food) per dim; unit norm + reinforcement;
    // motor = [forward, turn, valence]: danger [-0.7, 0.5, -danger_strength],
    // food [0.7, 0.0, food_strength]; both slots pre-activated.
}
```

- **Integration into init path** (`buffers.rs:731-800`, `init_brain_state_for`):
  the function already accepts a `BrainConfig`; the gated reset path (workstream
  0003) passes `config.instinct_danger_strength` and `config.instinct_food_strength`
  to `seed_instinct_patterns()`.

Properties that make this safe:
- The new `BrainConfig` fields are CPU-only state read at reset time; they do not
  alter the GPU config uniform layout (populated by `build_config_for()` at a fixed
  `CONFIG_SIZE`), so `shader_*_constants_match_rust` and physics-equivalence tests
  are untouched.
- Seeded patterns use the same motor/valence encoding as learned patterns: once in
  the buffer they decay and reinforce via the existing memory-store and
  memory-blend logic (`brain_passes.wgsl:1317-1341`, `:1614-1641`) with no special
  case.
- Instinct slots are pre-activated (`O_PAT_ACTIVE`, `O_ACTIVE_COUNT`,
  `O_LAST_STORED_IDX` set nonzero); subsequent learning can overwrite or evict them
  through the normal min-reinforcement-index path (`brain_passes.wgsl:1639-1641`),
  so they hold no privileged status against the learner.

## 0002 — Heritable-Instinct-Config

Today `BrainConfig` carries four heritable visual-genome genes (`gabor_wavelength`,
`gabor_aspect_ratio`, `dog_surround_ratio`, `orientation_offset`) defined in
`config.rs:150-178` and mutated by the breeding loop in
`crates/xagent-sandbox/src/agent/mod.rs` via gaussian drift and clamp-to-range,
following the clamp constants at `config.rs:188-200` (`GABOR_WAVELENGTH_MIN`,
etc.). The instinct-strength genes must follow that precedent exactly.

Edits:

- **Default functions and clamp constants** in `config.rs` (after the existing
  visual-genome defaults and `ORIENTATION_OFFSET_PERIOD`, `config.rs:186+`):

```rust
fn default_instinct_danger_strength() -> f32 { 0.8 }
fn default_instinct_food_strength() -> f32 { 0.8 }

/// Inclusive clamp bounds for the heritable instinct-strength genes; re-imposed
/// during breeding and after seeding.
pub const INSTINCT_DANGER_STRENGTH_MIN: f32 = 0.1;
pub const INSTINCT_DANGER_STRENGTH_MAX: f32 = 1.0;
pub const INSTINCT_FOOD_STRENGTH_MIN: f32 = 0.1;
pub const INSTINCT_FOOD_STRENGTH_MAX: f32 = 1.0;
```

- **Mutation + crossover in `agent/mod.rs`.** The exhaustive `BrainConfig` literals
  that set heritable genes live in `mutate_config_with_strength_rng` (≈line 408;
  visual genes set ≈479 via `momentum.biased_perturb_f(rng, parent.<gene>, "<gene>",
  strength)` then clamp) and `crossover_config` (≈587; visual genes inherited ≈657
  via 50/50 `if rng.random::<f32>() < 0.5 { a.<gene> } else { b.<gene> }`). Add both
  instinct genes to BOTH, using those exact idioms — NOT a raw `rng.random()` drift,
  which would bypass the directed-mutation momentum. The wrappers `mutate_config`
  /`mutate_config_seeded` call `mutate_config_with_strength_rng`, so no direct edit
  there.

```rust
// In mutate_config_with_strength_rng, alongside the visual-genome drift block:
instinct_danger_strength: momentum
    .biased_perturb_f(rng, parent.instinct_danger_strength, "instinct_danger_strength", strength)
    .clamp(INSTINCT_DANGER_STRENGTH_MIN, INSTINCT_DANGER_STRENGTH_MAX),
instinct_food_strength: momentum
    .biased_perturb_f(rng, parent.instinct_food_strength, "instinct_food_strength", strength)
    .clamp(INSTINCT_FOOD_STRENGTH_MIN, INSTINCT_FOOD_STRENGTH_MAX),
// In crossover_config, alongside the visual-genome inheritance block:
instinct_danger_strength: if rng.random::<f32>() < 0.5 { a.instinct_danger_strength } else { b.instinct_danger_strength },
instinct_food_strength:   if rng.random::<f32>() < 0.5 { a.instinct_food_strength }   else { b.instinct_food_strength },
```

Properties that make this safe:
- The genes are f32 scalars clamped to `[0.1, 1.0]`, a subset of the existing
  `BrainConfig::max_curiosity_bonus` range `[0.1, 1.0]` (`config.rs:67`), so the
  mutation machinery reuses the same gaussian-drift and clamp pattern with no new
  numeric envelope.
- Respawn semantics are locked to re-seed (see SCOPE locked decisions): instinct
  patterns are re-seeded from the mutated genes at every `reset_agents_with_rng()`
  call, so evolved instinct strengths are heritable but learned instinct
  reinforcement is not carried across respawn, matching the brief's "one-time
  evolved prior" philosophy.

## 0003 — Default-Off-Gating

Today `BrainConfig` carries three gate flags (`visual_cortex_enabled`,
`danger_percept_enabled`, `effort_rebased_fitness`) defined in `config.rs:127-149`.
These are locked per batch (not heritable), control entire shader passes or fitness
recomputations, and default to false. The `innate_instincts_enabled` flag follows
the same pattern. Seeding is routed through a single gated helper called at **every**
fresh-agent init site — when off, all sites call `init_pattern_memory()`, so the path
is byte-identical to the pre-instinct codebase.

Edits:

- **New gate field in `BrainConfig`** (`config.rs`, after `effort_rebased_fitness`,
  `config.rs:150`):

```rust
/// Gate flag for seeded innate instinct priors. When `false`, pattern memory
/// initializes all-zero (blank slate, byte-identical to pre-instinct behavior).
/// When `true`, danger + food instinct patterns are seeded at initialization.
/// Locked per batch, not heritable. Default false until the prove-or-kill gate passes.
#[serde(default)]
pub innate_instincts_enabled: bool,
```

- **One gated helper, called at every fresh-agent site.** `init_pattern_memory()` is
  called for fresh agents at THREE sites in `gpu_kernel.rs` — `reset_agents_with_rng`
  (≈557), `GpuKernel::new` (≈855), and the agent-grow/add path (≈2820). Factor a
  single helper and call it at all three so seeding is consistent (gating only the
  reset path would leave generation-0 agents from `GpuKernel::new` unseeded, with no
  test catching it). Do NOT touch the saved-state restore path (≈2774) that uploads
  persisted `s.patterns`.

```rust
fn pattern_init_for(config: &BrainConfig) -> Vec<f32> {
    if config.innate_instincts_enabled {
        seed_instinct_patterns(config.instinct_danger_strength, config.instinct_food_strength)
    } else {
        init_pattern_memory()
    }
}
// at each fresh-agent fill site:
pattern_data.extend_from_slice(&pattern_init_for(brain_config));
```

Properties that make this safe:
- When `innate_instincts_enabled=false` (the default), `init_pattern_memory()` is
  called, returning all zeros — byte-identical to the pre-0013 behavior, so no
  serialized config or stored genome changes meaning on the default path.
- The gate is the single control point: seeding is disabled by default and enabled
  only after the A/B gate passes, leaving no flag state ambiguous and no second
  seeding entry point to keep in sync.
- The locked re-seed decision (SCOPE) prevents silent re-seeding from confusing
  learned vs innate contributions during training; the gate condition for that
  decision is the A/B measurement in workstream 0004.

## 0004 — Prove-Or-Kill-Gate

Today the headless validation framework (`headless.rs:508-580`,
`run_headless_with_flags`) runs paired A/B experiments on effort-rebased fitness and
danger-percept sensing, collecting `ValidationStats` (`headless.rs:458-474`:
`mean_fitness`, `mean_movement_speed`, `mean_ticks_alive`, `mean_death_count`,
`mean_danger_dwell_fraction`, `mean_avoidance_intent_fraction`) and checking explicit
gates (`BASELINE_CORR_MIN`, `DECOUPLE_CORR_MAX`, `DECOUPLE_MARGIN`,
`AVOIDANCE_FLOOR`). Plan 0013 adds a new harness following the same structure.

Edits:

- **New gate constants** in `headless.rs` (after `ON_SPEED_COST_EXPONENT`,
  `headless.rs:481`):

```rust
/// ON (instincts seeded) must improve survival over baseline by at least this
/// fraction (10%) to require meaningful benefit while tolerating natural variance.
const INSTINCT_SURVIVAL_MARGIN: f32 = 0.10;
/// Mean avoidance-intent fraction (turns opposing danger bearing) in the ON run
/// must exceed this floor to require substantial steering-alignment.
const INSTINCT_ALIGNMENT_FLOOR: f32 = 0.4;
/// ON run's food-per-death ratio (mean food / mean death) must exceed this to
/// require at least two food consumed per death.
const INSTINCT_FOOD_PER_DEATH_MIN: f32 = 2.0;
```

- **New A/B validator** in `headless.rs` (after `run_headless_with_flags`),
  mirroring the speed-decoupling validator: run baseline (flag off) and ON (flag on)
  on identical seed-deterministic worlds, then evaluate the three gates.

```rust
/// Run the innate-instinct prove-or-kill A/B benchmark.
/// Baseline: innate_instincts_enabled=false (blank slate). ON: =true (seeded).
/// Both arms share seeded populations + worlds (deterministic mutations).
/// Passed iff survival > baseline + margin AND avoidance-intent >= floor
/// AND food-per-death >= threshold.
fn run_innate_instinct_ab(
    mut config: FullConfig,
    num_generations: u64,
) -> (ValidationStats, ValidationStats, bool) {
    // baseline = run_headless_instinct_validation(.. false);
    // on_run   = run_headless_instinct_validation(.. true);
    // gate on mean_ticks_alive, mean_avoidance_intent_fraction, food-per-death.
}
```

- **New validation subcommand** in `main.rs` (alongside the speed-decoupling
  validation arm): wire `run_innate_instinct_ab()` into the headless CLI so the gate
  is runnable without the egui app.

- **Decision document** `0013-INNATE-INSTINCT-DECISION.md` in the plan folder
  (created when the gated task lands): restates the spike task and Done-when,
  leads with the PASS/FAIL verdict, tabulates the three paths (blank slate / seeded
  / seeded + default-on), records measured baseline and ON statistics, and gives
  explicit revisit conditions — matching the Plan 0004/0005/0006 decision-doc
  precedent.

Properties that make this safe:
- The harness enforces deterministic seeding (using `world.seed` / `pop_init_seed`
  with `mutate_config_seeded`), so both arms see identical initial genomes and
  worlds, isolating the instinct-seeding mechanism as the only variable.
- Pass/fail is binary and data-driven: if any of the three gates (survival,
  alignment, food-per-death) fails, the flag stays off and the decision doc records
  the negative result with the thresholds that would need to flip for reconsideration.
- Food-per-death divides by `mean_death_count` only when it exceeds `1e-4`, treating
  a near-zero death count as a thriving population (`f32::INFINITY`, pass), so the
  gate never reports a spurious NaN.
- The decision-doc pattern (gates, measured evidence, when-to-revisit) prevents the
  gate from being re-litigated without new data.

## Test strategy

The falsifiable gate for the code tasks (0001-0003) is the standing quality gate
plus the byte-identical-when-off invariant: with `innate_instincts_enabled=false`,
`reset_agents_with_rng()` must still call `init_pattern_memory()` and produce the
pre-0013 pattern buffer, verified by the existing reset/init unit tests in
`gpu_kernel.rs` and `buffers.rs`. Workstream 0004's harness is its own acceptance
gate: `run_innate_instinct_ab()` runs the paired experiment and prints each of the
three gate evaluations, and the gated decision task lands only when the decision doc
records the measured outcome.

CI gate (every task): `cargo fmt --all -- --check`,
`cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test -p xagent-sandbox`. GPU tests self-skip without an adapter
(`GpuKernel::is_available()`); CI runs Mesa lavapipe.

## Interaction with prior work

- **Replaces 0012's removed reward shaping with internal priors.** Plan 0012
  removed standing external reward shaping; this plan seeds one-time evolved priors
  into pattern memory instead. The two are orthogonal — no re-measurement of 0012's
  gates is required.
- **Follows the visual-genome heritability precedent.** The instinct-strength genes
  reuse the exact `BrainConfig` gene + clamp-constant + breeding-drift pattern
  already established for `gabor_wavelength` and its siblings, so they evolve under
  fitness selection without new evolution machinery.
- **Honors the default-off / prove-or-kill discipline.** Like Plans 0004, 0005, and
  0006, the feature ships behind a default-off flag validated by a headless A/B gate
  with explicit thresholds and a recorded decision doc; the flag flips only via a
  separate follow-up plan, deferred out of scope here.
