# Scope — Plan 0023

> Extend the existing predictive forward model with a homeostatic-gradient prediction head, using the learned anticipation as a dense intermediate credit signal that bridges the ~10-tick sensory latency — without introducing any external supervision target.

## Why this plan

The credit-path bottleneck is now precisely localized: the encoder separates food-bearing (cosine-diff 0.964, ~55×), direct supervision routes it to action (0.841 steering, Plan 0020), but the homeostatic TD(λ) path alone cannot bridge the ~10-tick gap between "seeing food" and "eating food." Plan 0018-0003 proved definitively that signal magnitude is not the bottleneck (400× amplification moved steering by zero). Plan 0020 proved the routing is learnable in principle but declined integration because direct turn→food-bearing supervision is approach-shaping. The open problem: achieve the same routing under pure homeostatic pressure.

1. **The TD(λ) reward arrives too late and carries no directional information.** The scalar homeostatic gradient `raw_gradient = energy_delta × 0.6 + integrity_delta × 0.4` (`brain_passes.wgsl:913-915`) fires *after* eating — it says "that was good" but carries zero information about which turn direction, which visual feature, or which action caused the outcome. The agent must discover through random exploration that turning toward visible food → eating → positive gradient, but the credit signal decays to 6.7% per vision cycle before the next frame's evidence arrives (`common.wgsl:560,564`; `(0.873)^10 ≈ 0.067`).

2. **The existing forward model already predicts the next encoded state but does not predict the homeostatic consequence.** `coop_predict_and_act()` (`brain_passes.wgsl:1049`) trains a 128→128 dense predictor via online gradient descent on transition error (`brain_passes.wgsl:1079-1088`), then uses the predicted state for context blending and the TD critic. But the predictor never learns to anticipate the *homeostatic gradient* — the very signal that drives all learning. Adding a small linear head (128→1) that predicts the gradient from the predicted state gives the agent an *anticipatory* credit signal: "given what I see and what I'm about to do, I expect my energy to rise/fall."

3. **This is the free energy principle in its purest form — not approach-shaping.** The prediction target is the same homeostatic gradient that already drives everything. The predictor learns from experience: "when I see this pattern and turn this way, my energy tends to rise." The predicted gradient is the agent's own learned expectation, not an external target. It provides dense, immediate feedback at decision time — before the actual outcome — bridging the temporal gap without introducing any new signal source. The agent acts to maximize expected homeostatic improvement, which is exactly minimizing surprise about its own continued existence.

4. **The mechanism is structurally minimal — one new linear head on the existing predictor.** The predicted encoded state `s_prediction` (`brain_passes.wgsl:1103`) already flows through the forward model. Adding a 128→1 dot product + bias to predict the homeostatic gradient requires ~130 new floats per agent in brain state (~5 KB at 10 agents), one new cooperative reduction in `coop_predict_and_act`, and a small β-scaled term in the TD error. No new buffers, no new dispatches, no new data dependencies. The predictor weights are heritable and evolve — they become part of the agent's learned model of the world.

**Provenance.** Every load-bearing claim verified against current source post-0020 reversion: the homeostatic `raw_gradient` at `brain_passes.wgsl:913-915`, the TD error at `brain_passes.wgsl:1222-1224`, the predictor at `brain_passes.wgsl:1068-1107`, the trace decay constants at `common.wgsl:560,564`, the 0018-0003 gradient-shaping result (`docs/plans/0018-Credit-Path-Mechanism-Attack/STATUS.md:32-33`), the 0020 direct-supervision result (`docs/plans/0020-Credit-Path-Direct-GPU-Auxiliary/0001-GPU-AUXILIARY-LOSS-DECISION.md:13`), and the brain state layout at `common.wgsl:211-265` / `buffers.rs:64-121`.

**Review claims rejected during verification:**

| Claim | Source | Why rejected |
|---|---|---|
| The predictor already exists and doesn't help steering — adding a head won't change anything. | Implicit in 0018/0020 results | The existing predictor predicts the next *encoded state*, not the homeostatic gradient. The encoded state is a 128-dim representation; the gradient is a scalar. The predictor has never been trained to anticipate the consequence that drives all learning. This is a new capability, not a retuning. |
| This is just approach-shaping by another name. | Anticipated objection | The prediction target is the homeostatic gradient — the same `energy_delta × 0.6 + integrity_delta × 0.4` that already drives the TD path. No food bearing, no external target, no privileged geometry. The predictor learns from the agent's own experience what visual/action patterns precede homeostatic changes. This is the free energy principle, not reward shaping. |

## In scope

- **0001 — Homeostatic Gradient Predictor.** Add a 128→1 linear prediction head on top of the existing forward model's predicted state `s_prediction`, trained via online gradient descent to predict the homeostatic gradient `raw_gradient`. Use the predicted gradient as an anticipatory credit signal (β-scaled term in the TD error), providing dense immediate feedback at decision time. Gated behind a default-off flag. See [TASKS.md](TASKS.md).
- **0002 — Predictive-Credit Steering Probe.** Measure steering alignment on the mirrored-steering probe with the homeostatic gradient predictor active. Same protocol as 0018/0020: 120 training episodes, dense strides, pinned-movement evaluation, 95% Clopper–Pearson CI. See [TASKS.md](TASKS.md).
- **0003 — Decision Gate.** Binary ACCEPT/REJECT based on 95% CI: ACCEPT if CI lower ≥ 0.70, REJECT if CI upper ≤ 0.62. If ACCEPT, integrate the mechanism (flag stays default-off, baseline updated). If REJECT, record the negative result with structural candidates for the next plan. See [TASKS.md](TASKS.md).

## Origin → workstream mapping

| Finding | Addressed by |
|---|---|
| TD(λ) reward arrives after eating with no directional information; credit decays to 6.7% per vision cycle before next frame (1) | `0001` |
| Existing forward model predicts encoded state but not homeostatic consequence — adding a gradient prediction head provides anticipatory credit (2) | `0001` |
| Free energy principle: agent should act to minimize surprise about its own homeostatic future; predicted gradient is learned expectation, not external target (3) | `0001` |
| Mechanism is structurally minimal — one linear head, no new buffers or dispatches (4) | `0001` |
| Steering alignment must clear 0.70 to accept; 0.841 direct-supervision upper bound is the reference (0020) | `0002`, `0003` |

## Locked decisions

- **Homeostasis-only contract holds — prediction target is the existing homeostatic gradient.** The predictor head learns to anticipate `raw_gradient = energy_delta × ENERGY_WEIGHT + integrity_delta × INTEGRITY_WEIGHT` (`brain_passes.wgsl:913-915`). No food bearing, no external target, no privileged geometry enters the prediction target. The predicted gradient is the agent's own learned expectation, derived entirely from the same homeostatic signal that drives the TD path. This is the free energy principle: the agent learns a generative model of its own homeostatic dynamics and acts to maximize expected stability. Gate: every landed change must pass a homeostatic-signal audit confirming the prediction target contains only `energy_delta` and `integrity_delta` terms.
- **Steering alignment must clear 0.70 (above chance band 0.38–0.62) for acceptance.** The measurement gate is the mirrored-steering probe (`learning_probe_mirrored_steering_is_chance()`, `integration.rs:2848-2929`): alignment must move from baseline 0.489 above the band to ≥ 0.70. The 0020 direct-supervision result (0.841) is the upper-bound reference. Gate condition: measured alignment on the mirrored-steering probe, after 120 training episodes on alternating left/right food, with movement pinned (`movement_speed=0`), must be ≥ 0.70 (CI lower bound). Secondary gate: steering improvement must not regress encoder separability or food-visibility.
- **Binary ACCEPT/REJECT based on 95% CI; no gradual gates.** The verdict is mechanical: if CI lower ≥ 0.70, ACCEPT; if CI upper ≤ 0.62, REJECT; if CI straddles [0.62, 0.70], inconclusive (re-run with larger sample). This avoids interpretation gaps and grounds the next plan's design.
- **Predictor weights are heritable and evolve — they are part of the agent's learned world model.** The homeostatic gradient predictor weights (`O_HOMEO_PREDICTOR_WEIGHTS`, `O_HOMEO_PREDICTOR_BIAS`) are seeded at brain birth (Xavier initialization), updated via online gradient descent during life, inherited by the champion across generations, and mutated in offspring. They are first-class brain state, not a separate mechanism. The `O_PREV_HOMEO_PREDICTION` slot is episodic (zeroed on death), like the existing `O_PREV_VALUE`.

## Out of scope

- **Direct turn→food-bearing supervision (Plan 0020 approach).** Plan 0020 proved this works (0.841) but rejected it on homeostasis-only grounds. This plan achieves the same routing under pure homeostatic pressure — the predicted gradient is learned from the homeostatic signal, not from food geometry.
- **N-step returns, trace-decay restructuring, or eligibility-reset at vision boundaries.** These are structural candidates from 0020's decision doc, deferred to a future plan if the predictive credit approach also fails. This plan tests one new hypothesis class.
- **Encoder architecture changes (visual cortex, emergent encoder, raycast tuning).** Encoder separability is already high (cosine-diff 0.964). The bottleneck is downstream credit routing, not feature extraction.
- **Multi-agent dynamics, environmental pressure, or fitness-lever graduation.** These are separate research directions. This plan addresses credit unlock within the existing single-agent, fixed-food probe regime.
- **Graduation decision for the `homeo_predictive_credit_enabled` flag (if ACCEPT).** If the probe accepts, the flag lands as default-off. A separate plan or decision doc will record the graduation decision.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
