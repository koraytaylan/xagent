# Decision: Homeostatic Predictive Credit

**Date:** 2026-06-27
**Status:** REJECT
**Design:** 128→1 homeostatic gradient predictor on the forward model, with
β-scaled predicted gradient blended into the TD reward.

## Summary

The homeostatic gradient predictor head was implemented, numerically stabilized
(inverse-dimension step scaling, gradient and bias bounds, and proper temporal alignment
with the previous tick's prediction features), and evaluated on the mirrored-steering probe.
The predictor head learns online with low mean absolute prediction error (**MAE = 0.053108**
across 1,666 diagnostic samples), verifying that the forward model's predicted state reliably
anticipates the homeostatic gradient.

However, blending the predicted gradient as an anticipatory credit signal into the TD reward
fails to raise directional steering above chance: measured turn/bearing alignment is
**54 / 239 = 0.226 (95% CI [0.173, 0.279])**, versus the mirrored-steering baseline of 0.471
and chance band [0.38, 0.62]. Because the 95% CI upper bound 0.279 is ≤ 0.62, the mechanical
decision rule mandates **REJECT**.

## Measurements

#### Steering Alignment (Test: `homeo_predictive_credit_steering_probe`)
- **Training:** 120 mirrored episodes with predictor enabled (lr=0.01, β=0.3)
- **Evaluation:** pinned movement, predictor disabled, bearing window [0.05, 0.6] rad
- **Measured alignment:** 54 / 239 = **0.226** (95% CI [0.173, 0.279])
- **Chance band:** [0.38, 0.62]
- **Upper-bound reference (Plan 0020 direct supervision):** 0.841
- **Verdict:** REJECT — CI upper 0.279 ≤ 0.62

#### Predictor Diagnostic
- **Mean absolute prediction error:** 0.053108 across 1,666 samples (demonstrates solid
  convergence of the 128→1 linear prediction head on `raw_gradient`)

#### Supporting Probes
- **Foraging viability:** 816 food items consumed during training (arena and movement intact)
- **Baseline mirrored steering:** 171 / 363 = 0.471 (chance band intact)
- **Flag off-path:** inert, byte-identical liveness confirmed by `homeo_predictive_credit_flag_is_inert_when_disabled`

## Why Reject (Root Cause Analysis)

The predictor's low error (0.053) confirms that the mechanism was fairly and cleanly tested:
the network learned to anticipate its homeostatic gradient from the forward model's predicted
state. The failure to generate directional steering localizes to the credit routing:

1. **State-Level vs Action-Conditional Anticipation:** The forward model in `coop_predict_and_act`
   predicts `s_prediction = tanh(W · s_encoded)` without conditioning on the executed action.
   Consequently, `predicted_gradient` is a state-value expectation: it anticipates survival
   trends given current sensory evidence, but does not provide counterfactual action contrast
   (e.g., turning left vs turning right).
2. **Reward Blending Without Action Contrast:** Adding an action-agnostic expectation to the TD
   reward shifts the baseline reward but cannot guide the actor's turn trace toward food. In fact,
   blending non-contrastive anticipatory reward adds variance to the TD error δ, reducing turn
   selectivity below chance (0.226).

## Fallback Structural Candidates

With the homeostatic gradient predictor evaluated and rejected, the problem of routing
homeostatic credit across the ~10-tick sensory latency without approach-shaping remains
sharpened. As established in the Plan 0020 decision doc, the primary structural alternatives
for future credit-path research include:

1. **Action-Conditioned Forward Model:** Conditioning state/gradient prediction explicitly
   on candidate actions $(s_t, a_t) \to s_{t+1}$ so that anticipatory credit provides
   differential turning signals.
2. **n-Step Returns:** Propagating multi-step homeostatic returns directly across the 10-tick
   sensory lag to bridge the temporal gap without relying on intermediate state bootstrapping.
3. **Eligibility Decay Rework / Reset at Vision Boundary:** Pausing or reshaping eligibility
   decay between sensory frames to preserve turning credit until outcome evidence arrives.
4. **Auxiliary-Head Through Shared Visual Encoder:** Auxiliary self-supervised prediction tasks
   routing through the visual encoder without injecting approach-shaping into motor channels.

## Disposition

The mechanism remains gated behind the default-off `homeo_predictive_credit_enabled` flag
(zero-cost, byte-identical when disabled). The regression and steering probe tests are
retained on `develop` to guard against regressions.
