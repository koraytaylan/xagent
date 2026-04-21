# Audit: Credit Assignment Parameterization

**Date:** 2026-04-21
**Issue:** #116
**Status:** Inventory + initial simplification. Adaptive-threshold work remains open.

## Purpose

Inventory every tunable constant that compounds into the credit signal fed to
the policy and encoder, flag the interactions that were brittle enough to
cause past regressions, and propose a path toward fewer hand-tuned knobs.

The prompt for this audit is the circling investigation on #13 (see PR #102),
which ultimately traced the bug to ~14 compounding causes spread across
deadzones, amplifiers, habituation floors, and multi-timescale EMAs. After
Brain Reunification (#107) the credit signal now drives both the action layer
and the encoder via shared scaffolding (`s_encoded`, `s_homeo`), so a clean
view of the full parameter surface is useful before the next round of tuning.

All references below are to `crates/xagent-brain/src/shaders/kernel/common.wgsl`
(constants) and `brain_passes.wgsl` (usage), on the commit this doc lands on.

## Inventory

### Homeostatic gradient — input to credit

The credit signal fundamentally rides on `gradient` / `urgency` produced in
`coop_habituate_homeo` (pass 3). Constants involved:

| Constant | Value | Where it lives | Role |
| --- | --- | --- | --- |
| `MAX_HOMEOSTATIC_DELTA` | `0.3` | per-tick clamp on energy/integrity delta | bounds raw gradient |
| `ENERGY_WEIGHT` | `0.6` | raw gradient mix | |
| `INTEGRITY_WEIGHT` | `0.4` | raw gradient mix | paired with above — sum = 1.0 |
| `GRADIENT_FAST_BLEND` | `0.6` | EMA α for `gradient_fast` | responds in ~2 ticks |
| `GRADIENT_MEDIUM_BLEND` | `0.04` | EMA α for `gradient_medium` | responds in ~25 ticks |
| `GRADIENT_SLOW_BLEND` | `0.004` | EMA α for `gradient_slow` | responds in ~250 ticks |
| `GRADIENT_WEIGHT_FAST` | `0.5` | blend over the three EMAs | |
| `GRADIENT_WEIGHT_MEDIUM` | `0.35` | | |
| `GRADIENT_WEIGHT_SLOW` | `0.15` | | sum = 1.0 |
| `DISTRESS_SCALE` | `10.0` | urgency amplifier | |
| `MAX_DISTRESS` | `10.0` | urgency cap | |
| `CFG_DISTRESS_EXP` | uniform | non-linear distress shaping | set from `BrainConfig`, not a WGSL const |

Effective `gradient` reaching the credit assignment (`s_homeo[0u]`):

```
gradient = blended_gradient * (1.0 + urgency)
```

with `blended_gradient` ∈ roughly ±0.3, `urgency` ∈ [0, 10]. Peak magnitude is
therefore ~3.3 in practice — three orders of magnitude larger than the
`DEADZONE`. That wide range is the fundamental source of brittleness: every
downstream threshold has to work across it.

### Credit assignment — action weights

`coop_predict_and_act` runs credit assignment over the last 64 history entries
(`ACTION_HISTORY_LEN = 64`):

| Constant | Value | Role |
| --- | --- | --- |
| `CREDIT_DECAY` | `0.3` | temporal exponential decay: `temporal = exp(-age * 0.3)` |
| `DEADZONE` | `0.005` | cutoff: improvements below this fall back to the tonic path |
| `TONIC_CREDIT_SCALE` | `0.5` | tonic fallback: `credit = gradient * urgency * 0.5` |
| `PAIN_AMP` | `3.0` | asymmetric amplifier: negative credit is scaled 3× |
| `CREDIT_EPSILON` | `1e-6` | skip updates below this magnitude |
| `ACTION_WEIGHT_LEARNING_RATE` | `0.10` | per-step update scale |
| `ACTION_WEIGHT_DECAY` | `0.01` | weight-decay each brain tick |
| `MAX_WEIGHT_NORM` | `2.0` | L2-ball clamp on forward/turn weight vectors |

Plus an undocumented inline `* 0.1` on both bias updates at
`brain_passes.wgsl:398-399` that effectively gives biases a second,
hand-tuned learning-rate multiplier — a classic example of the kind of
over-parameterization this audit is meant to surface.

### Encoder credit — backpropagation into representation

Encoder weights are updated from the same credit signal in
`coop_learn_and_store` (pass 7b):

| Constant | Value | Role |
| --- | --- | --- |
| `ENCODER_CREDIT_SCALE` | `0.1` | additional damping on the encoder path |
| `CREDIT_EPSILON` | `1e-6` | shared gate |

Encoder weights are clamped to `[-2.0, 2.0]` inline (`brain_passes.wgsl:715`)
rather than via `MAX_WEIGHT_NORM`. Values happen to match today, but the two
quantities logically differ and the inline constant will drift.

### Habituation — input scaling before prediction

| Constant | Value | Role |
| --- | --- | --- |
| `HAB_EMA_ALPHA` | `0.02` | per-dim EMA α for habituation |
| `ATTEN_FLOOR` | `0.1` | minimum attenuation applied to encoded state |

The post-reunification policy evaluates on `s_encoded` (not habituated), so
habituation no longer silently scales credit weight updates *directly*. But
it still scales the predictor input (`s_habituated`), which feeds
`prediction_error`, which feeds `novelty_bonus` → `exploration_rate`. The
coupling survived reunification; it is just one hop further from credit.

### Policy-side constants that affect how much of the credit survives

| Constant | Value | Where | Role |
| --- | --- | --- | --- |
| `MEMORY_BLEND_STRENGTH` | `0.4` | `brain_passes.wgsl:498` | motor signal attenuated by up to 40% when memory recall fires |
| `KLINOTAXIS_SENSITIVITY` | `500.0` | `brain_passes.wgsl:606` | turn multiplier ∈ [0.3, 3.0] based on fast-vs-medium deviation |
| `ANTICIPATION_WEIGHT` | `0.5` | `common.wgsl:273` | **dead** — declared but not referenced in any shader |

### Inline magic numbers in the credit/policy hot path

Values that escaped the named-constant gate and still shape credit math:

| Location | Literal | What it does |
| --- | --- | --- |
| `brain_passes.wgsl:398–399` | `* 0.1` | additional scale on bias update beyond `ACTION_WEIGHT_LEARNING_RATE` |
| `brain_passes.wgsl:512` | `* 2.0`, `0.4` | novelty-bonus cap |
| `brain_passes.wgsl:513` | `* 0.4`, `0.5` | urgency-penalty cap |
| `brain_passes.wgsl:515` | `/ 2.0` | policy-confidence normalization |
| `brain_passes.wgsl:517–520` | `0.5`, `0.25`, `0.10`, `0.85` | exploration-rate base, scaling, and clamp window |
| `brain_passes.wgsl:592–593` | `* 0.5` | exploration noise amplitude |
| `brain_passes.wgsl:704` | `* 0.01`, `- 0.5` | predictor context-weight adaptation rate and target |
| `brain_passes.wgsl:705` | `0.05`, `0.5` | predictor context-weight clamp |
| `brain_passes.wgsl:715` | `-2.0, 2.0` | encoder weight clamp (duplicates `MAX_WEIGHT_NORM` logic) |

Per CONTRIBUTING.md, each of these should be a named constant with a `why`
comment. None of them are safe to drop without understanding their role —
they are flagged here, not fixed here.

## Interaction Map — why this is brittle

The credit pipeline is a seven-stage product of these constants. The path
from a joules-gained event to a weight update:

1. **Raw gradient** is clamped by `MAX_HOMEOSTATIC_DELTA = 0.3` per tick.
2. **Three EMAs** blend the raw gradient at α = 0.6 / 0.04 / 0.004.
3. **Weighted blend** of those three (0.5 + 0.35 + 0.15) produces
   `blended_gradient`.
4. **Urgency amplification** (`1 + urgency` with urgency up to 10) scales it.
5. **Deadzone branch**: if `|improvement| < 0.005` the signal switches
   discontinuously to `gradient * urgency * 0.5`.
6. **Pain asymmetry** multiplies the negative half by `PAIN_AMP = 3.0`.
7. **Temporal decay** `exp(-age * 0.3)` is applied, gated at `0.01`.
8. **Scaling**: `ACTION_WEIGHT_LEARNING_RATE = 0.1`, then either `× 0.1`
   (bias path) or `× feat × ENCODER_CREDIT_SCALE = 0.1` (encoder path).
9. **Clamp** to `MAX_WEIGHT_NORM = 2.0` (policy) or `[-2, 2]` inline
   (encoder).
10. **Decay**: `ACTION_WEIGHT_DECAY = 0.01` applied every brain tick.

Known regressions this chain produced historically:

- **PR #20 / #92** — raising `TONIC_CREDIT_SCALE` and lowering `DEADZONE`
  together, because signals starved below the deadzone and the tonic
  fallback was too small to compensate.
- **PR #94 / #97** — oscillation between recording the pre-noise policy
  output and the post-noise motor in the history ring; exposed a
  zero-gradient trap when zero-init action weights caused
  `policy_fwd = tanh(0) = 0`.
- **PR #96** — changing the tonic fallback from "skip" to "tonic" turned a
  ~4-entry serial loop into a ~60-entry one (15× serial work on thread 0).
- **PR #100** — `s_habituated` vs `s_encoded` mismatch between the policy
  and the credit training features (SNR 0.25). Fixed by routing both through
  `s_encoded`, which is what reunification codified.

Each of these was fixed one-off. The pattern is the same: one constant
changes, another becomes misaligned, and a regression surfaces 1–5
generations later.

## What reunification already simplified

Worth calling out, because the issue predates the closure of #107:

- Policy and credit features are now both `s_encoded`. Prior versions mixed
  `s_habituated` / `s_encoded`, which required `ATTEN_FLOOR` compensation in
  the credit path. That compensation is gone.
- Weight space is bounded and small (two vectors × 128 dims = 256 weights,
  plus two biases), so `MAX_WEIGHT_NORM` and `ACTION_WEIGHT_DECAY` do their
  job directly.
- The noise path is now the REINFORCE gradient carrier (history records
  `noise_forward * exploration_rate`, not the full motor), so the deadzone /
  tonic branch no longer interacts with policy initialization the way it did
  before #97.

## What remains brittle

1. **The deadzone / tonic branch** is still a hard threshold on a signal
   whose magnitude varies by ~3 orders of magnitude over the lifetime of an
   agent. Small changes in `MAX_HOMEOSTATIC_DELTA`, urgency shape, or the
   EMA weights can cross the 0.005 threshold for common operating ranges.
2. **Bias vs weight update asymmetry**: biases are updated with an extra
   `× 0.1` inline, making the effective bias learning rate `0.01` and the
   weight learning rate `0.10`. Whether this is intentional is unclear; it
   is undocumented.
3. **Encoder scale (`ENCODER_CREDIT_SCALE = 0.1`)** is a second hand-tuned
   dampener on top of the already-small credit values. Its purpose is to
   prevent the encoder from destabilizing under the same credit signal that
   tunes the action layer, but the right value depends on
   `ACTION_WEIGHT_LEARNING_RATE`, `FEATURE_COUNT`, and the typical
   `s_features[j]` magnitude — none of which are currently tied together.
4. **Three time-scale EMAs** (`0.6 / 0.04 / 0.004`) plus a fixed weighted
   blend (`0.5 / 0.35 / 0.15`) is equivalent to a fixed low-pass filter
   applied at a single cutoff. The three-EMA scheme is intuitive but the
   parameter count (6 values) is higher than the expressive power warrants.
5. **Urgency amplification** (`× (1 + urgency)` with `urgency` up to 10)
   interacts multiplicatively with `PAIN_AMP = 3.0`, giving a negative
   signal under full distress a potential ~33× scale relative to a
   neutral-urgency positive signal. This is the single largest asymmetry in
   the pipeline and is not empirically justified anywhere.
6. **Policy-side magic numbers** (see the inline-literals table) all
   affect how credit updates translate into motor output; each is another
   lever the tuner has to hold constant.

## Proposals

Ordered by impact-to-risk:

1. **Remove `ANTICIPATION_WEIGHT`.** It is declared and never read. Trivial
   win; done as part of this PR.
2. **Name the inline magic numbers in the credit / policy hot path.** Per
   CONTRIBUTING.md §Magic Numbers. No behavior change, just naming. Each
   name should include the *why*. This makes the next tuning pass
   auditable. Out of scope for this PR — tracked as future work.
3. **Unify encoder and action weight clamps** onto a single constant, or
   rename the inline `[-2, 2]` in pass 7b to make the intent (encoder
   budget) distinct from `MAX_WEIGHT_NORM` (action budget). Either is a
   one-line change. Out of scope for this PR.
4. **Collapse the three-EMA blend into a single EMA** with an α chosen to
   match the current effective impulse response. Kills 5 constants, leaves
   one. Needs a follow-up with a before/after simulation; out of scope.
5. **Make `DEADZONE` / `TONIC_CREDIT_SCALE` adaptive** to the recent
   `|improvement|` distribution (for example, a running EMA of
   `|improvement|` with the deadzone set to some fraction of that). This
   removes the two most regression-prone constants at the cost of adding a
   single per-agent running statistic. Out of scope for this PR but the
   most promising line of work — matches the issue's "adaptive or learned
   alternatives" ask directly.
6. **Reconsider `PAIN_AMP` and the urgency-amplification interaction.**
   The ~33× asymmetry between distressed negative credit and neutral
   positive credit is very large. A sweep on `PAIN_AMP` ∈ {1, 2, 3} under
   evolution, with urgency amplification either removed or replaced by a
   distress-only floor, would tell us whether the amplification is doing
   useful work. Out of scope for this PR.

Items 4–6 each deserve their own issue and plan before any code change.

## What this PR actually does

- Files this audit under `docs/superpowers/specs/` so subsequent work has a
  canonical starting point.
- Removes the dead `ANTICIPATION_WEIGHT` constant (#116 proposal 1).

Everything else is tracked as future work and should land behind its own
before-and-after measurement.

## References

- Issue #116 (this audit)
- Issue #107 (Brain Reunification — enabling precondition, closed 2026-04-21)
- [Gemini 3.1 Pro — "Over-Parameterization of Credit Assignment"](https://github.com/koraytaylan/xagent/blob/develop/docs/reviews/2026-04-15-gemini-31-pro.md#over-parameterization-of-credit-assignment)
- PRs that tuned these constants historically: #20, #92, #94, #96, #97, #100, #102
