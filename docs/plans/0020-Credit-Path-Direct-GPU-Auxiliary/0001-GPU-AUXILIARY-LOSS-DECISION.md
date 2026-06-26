# Decision: GPU-Integrated Auxiliary Steering Loss

**Date:** 2026-06-26
**Status:** WORKS-BUT-NOT-INTEGRATED (mechanism clears the numeric gate; deliberately
not adopted on homeostasis-only grounds)
**Design:** GPU-side bearing-aligned auxiliary loss injecting gradients into
`O_ACTION_TURN_WEIGHTS` and `O_ACTION_FORWARD_WEIGHTS`.

## Summary

The GPU-integrated auxiliary loss **does** raise steering well above chance once a
sign-inversion bug in the turn target is corrected: measured alignment
**647 / 769 = 0.841** (95% CI [0.816, 0.867]), versus the mirrored-steering
baseline of 0.489 (chance band [0.38, 0.62]). The CI lower bound 0.816 clears the
ACCEPT gate (≥ 0.70).

The mechanism is therefore **not** rejected for failing — it works. It is
**deliberately not integrated** because it is explicit approach-shaping: it
supervises the turn action directly toward the food-direction signal, which the
project's homeostasis-only / "food-bearing-blind, approach incidental by
construction" contract forbids as a live learning term. This decision records the
corrected measurement as a diagnostic result and carries the sharpened problem to
Plan 0021. The auxiliary-loss code remains in the tree behind its default-off flag
(`auxiliary_steering_loss_enabled`), sign-corrected, as a measurement harness only.

### What the first run got wrong

The initial implementation set the turn target to `+food_bearing / PI`. But
`P_NEAREST_FOOD_BEARING = atan2(facing × to_food, facing · to_food) = yaw −
food_heading`, and physics applies `yaw += motor_turn · TURN_SPEED · dt` (positive
`motor_turn` increases yaw). The turn that rotates *toward* food and reduces the
bearing to zero is therefore `motor_turn = −bearing`. Supervising `turn_output →
+bearing` trains the agent to turn *away* from food, producing systematic
anti-alignment — exactly the observed 0.114 (≈ 1 − 0.886), not a near-chance
no-op. The probe's own scoring convention (`bearing_probe = atan2(Δx, Δz) −
prev_yaw = food_heading − yaw = −P_NEAREST_FOOD_BEARING`, correct when
`sign(motor_turn) == sign(bearing_probe)`) is physically correct and was never the
issue. Negating the target (`-food_bearing / PI`) flips the result from 0.114 to
0.841.

## Measurements

#### Steering Alignment After GPU-Auxiliary-Loss Training (Test: `gpu_auxiliary_steering_alignment_probe`)
- **Date/Adapter:** 2026-06-26 Metal (local GPU)
- **Training:** 100 dense-stride ticks with GPU-integrated auxiliary loss enabled
- **Evaluation:** pinned movement, auxiliary loss off, turn-alignment scored over a
  bearing window [0.05, 0.6] rad with non-zero `motor_turn`
- **Inverted-target (original) measurement:** 54 / 475 = 0.114 (95% CI [0.085, 0.142]) — anti-aligned
- **Sign-corrected measurement:** 647 / 769 = **0.841** (95% CI [0.816, 0.867])
- **Chance band:** [0.38, 0.62]
- **Numeric verdict:** ACCEPT — CI lower 0.816 ≥ 0.70

#### Supporting Probes
- **Encoder separability:** > 0.9 (intact, not regressed)
- **Food consumption:** > 0 (arena functional)

## Why Not Integrated (despite clearing the gate)

The auxiliary loss is the most explicit form of approach-shaping available: it uses
the privileged food-direction signal (`P_NEAREST_FOOD_BEARING`) as a direct
supervision target for the action channel. The project's central learning contract
(Plan 0012 homeostatic-only restoration; the keystone finding that approach-shaping
violates homeostasis-only) requires that food-seeking emerge *incidentally* from
homeostatic TD pressure, not from a term that tells the agent which way food is.
When the mechanism failed (0.114) this tension was moot; now that it succeeds
(0.841) the cost is real, and the project chooses to preserve the homeostasis-only
contract over adopting the working-but-out-of-philosophy shortcut. SCOPE.md
pre-authorized ACCEPT→integrate, but that authorization was written before the
mechanism was known to work; with the result in hand the homeostasis principle
takes precedence.

## Measured Cause (what this localizes)

This is a **positive diagnostic**, not a failure. Direct supervision routes the
encoder's already-separable food-bearing signal (cosine-diff 0.964, ~55×) to the
turn channel and lifts steering from chance (0.489) to 0.841. That isolates the
long-standing bottleneck precisely:

- **Not the encoder** — separability is intact (> 0.9) and was never the blocker.
- **Not signal magnitude** — Plan 0018-0003 amplified mean|δ| ~400× with no steering
  gain; magnitude is not the bottleneck.
- **The credit path itself** — TD(λ) bootstrap alone cannot route the separable
  signal across the ~10-tick sensory latency to the action weights, whereas a
  correctly-signed *direct* target can. The bottleneck is credit timing/alignment
  in the self-supervised path, and it is unblockable in principle — the open
  question is doing so without a privileged food-direction target.

## Next Step

Record this decision. Do **not** run the integration task; do **not** update the
steering baseline (mirrored-steering 0.489 stands as the homeostasis-only baseline).
Carry the sharpened problem to **Plan 0021**: achieve the same vision→turn routing
under homeostatic pressure alone.

## Structural Candidates for Plan 0021

Plan 0021 should evaluate candidates by whether they unblock vision→turn routing
**without** a privileged food-direction target — i.e., they must improve credit
assignment, not re-introduce approach-shaping. The 0.841 direct-supervision result
is the upper-bound reference these must approach under homeostasis-only.

| Candidate | Why it might work | Structural change |
|---|---|---|
| **n-step returns** | Explicitly bridges the ~10-tick sensory latency without relying on trace decay; credit accumulates in lookahead, not eligibility. | Modify `coop_predict_and_act()` in `brain_passes.wgsl` to compute n-step returns (n≈3–10) instead of 1-step TD-error scaling. |
| **Eligibility-decay rework** | Current traces decay as (0.873)^tick → ~6.7% at 10 ticks; the decay may be misaligned with the sensory rhythm. | Tune trace accumulation/decay in vision-tick and brain-tick phases; per-timestep vs per-vision-cycle decay. |
| **Eligibility reset at vision boundary** | Traces spanning multiple vision cycles may accumulate misaligned credit; vision latency is the dominant time constant. | Reset (or sharply reduce) traces each time vision updates. |
| **Auxiliary-head through shared encoder** | This experiment only supervised action weights; a head that predicts state (not bearing) and backprops through the shared encoder might improve routing without a food-direction action target. | Add a self-supervised prediction head in `phase_brain_tail_from_scratch.wgsl` or `brain_passes.wgsl`. |

**Authored by:** Plan 0020 Workstream 0002, corrected 2026-06-26 after sign-bug
diagnosis (`gpu_auxiliary_steering_alignment_probe`).

**Gate:** Numeric ACCEPT requires steering ≥ 0.70. Sign-corrected measurement:
0.841 (CI lower 0.816). Outcome: **mechanism works, deliberately not integrated**
(homeostasis-only contract).
