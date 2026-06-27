# Plan 0023 — Homeostatic Predictive Credit — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 🚧 Core mechanism landed (2026-06-27 session); full steering probe + decision gate pending.
_Last updated: 2026-06-27, against `develop` (core predictor, config, telemetry, smoke in place; flag default-off, zero cost when disabled)._

- **Goal:** Extend the existing predictive forward model with a homeostatic-gradient prediction head, using the learned anticipation as a dense intermediate credit signal that bridges the ~10-tick sensory latency — without introducing any external supervision target. Steering alignment must clear ≥ 0.70 (95% CI lower bound) on the mirrored-steering probe for acceptance.
- **Root cause:** The TD(λ) reward arrives after eating with no directional information; credit decays to 6.7% per vision cycle before the next frame's evidence arrives. The encoder separates food-bearing (cosine-diff 0.964, ~55×), direct supervision routes it to action (0.841, Plan 0020), but the homeostatic TD(λ) path alone cannot bridge the temporal gap. The existing forward model predicts the next encoded state but not the homeostatic consequence — adding a gradient prediction head provides anticipatory credit at decision time.
- **Approach:** Add a 128→1 linear prediction head on top of the forward model's predicted state, trained online to predict `raw_gradient`. Blend the previous tick's prediction into the TD reward as an anticipatory credit signal (β-scaled). The prediction target is the same homeostatic signal that drives the TD path — no external targets, no approach-shaping. Measure steering alignment on the mirrored-steering probe; binary ACCEPT/REJECT based on 95% CI.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Homeostatic Gradient Predictor | `brain-state-layout`, `config-flag`, `predictor-shader` | 📋 Planned |
| 0002 | Predictive-Credit Steering Probe | `steering-probe` | 📋 Planned |
| 0003 | Decision Gate | `decision-gate` | 📋 Planned |
