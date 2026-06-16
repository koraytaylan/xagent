# Plan 0004 — Approach Reward Shaping · status

Task-level execution status for this plan. Keep it current as tasks land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete — prove-or-kill terminal state reached (the `0001`
remeasure recorded a **negative**). Executed on `plan/0004-approach-reward-shaping`
off `develop` @ `069ff8a`. 9/14 tasks landed; the 5 gated tasks were correctly
**not opened** because the remeasure did not move the probe.
_Last updated: 2026-06-15, against `develop`._

- **Goal:** Give the within-lifetime reward a spatial approach gradient
  (potential-based shaping `F = γΦ(s′) − Φ(s)` into `raw_gradient`) so
  vision-conditional steering becomes learnable, then restore the selection
  signal and gate the reactive/heritable/perceptual follow-ups behind the one
  remeasure that proves the unlock.
- **Outcome:** The shaping + actor-scale machinery landed and is mechanism-tested,
  but the remeasure **falsified the unlock**: stationary turn/bearing alignment
  held at chance (`[0.36, 0.50]`) across a 1200-episode shaped mirrored-steering
  run, with no trend toward the `0.62` edge; free-run foraging was flat within
  single-seed noise. The baseline encoder separability margin (between `0.0036` ≫
  within `0.0002`) places the bottleneck on the **credit/temporal path** (the
  distance-closing reward credits forward motion; TD(λ) cannot extract the turn
  channel's second-order contribution into steering), **not** the encoder. Per
  the locked rule the gated workstreams stay closed. The PBRS shaping and
  actor-scale split are kept (optimal-policy invariant — they cannot corrupt the
  eat objective and are the substrate for the next credit-path attempt). Full
  verdict + trajectory in
  [`docs/superpowers/specs/2026-06-10-learning-baseline.md`](../../superpowers/specs/2026-06-10-learning-baseline.md)
  (2026-06-15 sections).

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Approach reward shaping | `approach-shaping-baseline`, `visible-food-potential-input`, `potential-based-reward-shaping`, `actor-vector-scale`, `approach-shaping-remeasure` | ✅ landed — remeasure **negative** (prove-or-kill gate) |
| 0002 | Reactive & valence layers on the external gradient | `klinotaxis-external-gradient` (GATED), `memory-valence-food-in-view` (GATED) | 🚫 not opened — gate closed by the `0001` negative |
| 0003 | Selection signal restoration | `foraging-primary-fitness`, `decoupled-experiment-resolution`, `selection-significance-guard`, `governor-within-life-metric` | ✅ landed (3 full + `num_islands→1`); the population-growing decoupling **superseded** by measurement (see note) |
| 0004 | Heritable learning dynamics | `policy-constants-to-genes` (GATED) | 🚫 not opened — gate closed by the `0001` negative |
| 0005 | Sensory lag & vision geometry | `heritable-stride-revisit` (GATED), `vision-row-geometry` (GATED) | 🚫 not opened — gate closed by the `0001` negative |

## Per-task notes

**0001 — all landed; the remeasure is the load-bearing result.**
- `approach-shaping-baseline` — recorded the pre-change probe numbers (alignment
  0.497, foraging 0.146, mirrored 0.524, separability between 0.0036 / within
  0.0002) and the `0.62` threshold.
- `visible-food-potential-input` — `P_NEAREST_FOOD_DISTANCE` / `P_PREV_POTENTIAL`
  physics slots (`PHYS_STRIDE` 32→34) + a parallel `SHAPING_RADIUS` reduction in
  the food-detect pass; mechanism test green.
- `potential-based-reward-shaping` — `F = γΦ(s′) − Φ(s)` folded into
  `raw_gradient`; `shaped_reward_rewards_approach` red→green verified.
- `actor-vector-scale` — `ACTOR_VECTOR_SCALE = 1/16` split from the critic's
  `1/128`; `actor_step_scales_with_actor_vector_scale` pins the split exactly.
- `approach-shaping-remeasure` — **negative** recorded; gated workstreams closed.

**0003 — selection-signal restoration (unconditional).**
- `foraging-primary-fitness` ✅ — `food_per_1k_alive_ticks` primary, survival a
  floored multiplier; restores dynamic range above the noise floor.
- `selection-significance-guard` ✅ — accept rule now `gen_avg − parent >
  K_SIGNIF·pooled_stderr`; falls back to the bare compare when repeats < 2.
- `governor-within-life-metric` ✅ — `WithinLifeTracker` + `q1_food_rate` /
  `q4_food_rate` persisted per generation (idempotent migrations); shared
  `quarter_food_rates` helper fed by both the headless and interactive drivers.
- `decoupled-experiment-resolution` ⚠️ **partial / superseded.** `num_islands → 1`
  landed (concentrates the thin foraging signal in one lineage). The
  `eval_unique_configs` knob (growing the population to 8–16 unique × 4–6
  repeats) is **not** implemented: it conflicts with the later 2026-06-15
  population-sweep measurement (larger shared-world populations compete for
  finite food and regress per-capita foraging; search breadth is not the
  bottleneck — learner strength is) and with the shared-world architecture (the
  GPU kernel is sized for one population in one world, so "independent per-repeat
  seeds" is not available without the deferred independent-arenas rework). Per
  the plan's own "measurement first, always" rule, the population-growing
  decoupling is recorded as superseded rather than implemented against the
  measurement.

## Verification

`cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D
warnings`, and `cargo test -p xagent-sandbox` (85 lib + 13 bin + 65 integration)
all green, plus `cargo test -p xagent-brain` (50). GPU tests ran on a Metal
adapter (not self-skipped), so the behavioral remeasure is a real measurement.
