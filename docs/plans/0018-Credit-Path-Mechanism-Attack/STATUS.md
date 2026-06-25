# Plan 0018 — Credit-Path Mechanism Attack: Auxiliary Loss, Trace Horizon, and Gradient Shaping — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 📋 Planned.
_Last updated: 2026-06-25, against `develop`._

- **Goal:** Steering alignment moves from chance (0.489 baseline, 0.38–0.62 band)
  to ≥0.70 via at least one of three mechanisms: auxiliary self-supervised
  steering objective, credit-horizon restructuring, or gradient-variance shaping.
  Encoder separability and food-visibility probes hold. Credit-path bottleneck is
  either unlocked (landing ≥0.70 alignment with all controls green) or falsified
  again (all three spikes reject, no fix clears gate), carrying the problem
  forward to plan 0019 with refined hypothesis.
- **Root cause:** Plan 0017 falsified simple TD-parameter tuning (trace decay,
  urgency isolation) and pinned the credit path as the bottleneck, not the
  encoder. The vision-to-action steering signal is degenerate (mean|δ| ~9e-5
  during foraging, 100× too small to move policy on single episode) and likely
  misaligned in timescale (traces decay (0.873)^10 ≈ 6.7% per vision cycle,
  faster than sensory latency) or structure (no direct supervision of turning
  behavior). Three independent mechanisms hypothesized: (1) auxiliary
  self-supervised steering objective bypassing slow TD bootstrap, (2) n-step
  returns or frame-synchronized trace decay bridging sensory latency, (3) gradient
  normalization/scaling restoring non-degenerate learning signal during foraging.
- **Approach:** Each of three parallel workstreams (0001, 0002, 0003) follows the
  spike-prototype-decide-integrate pattern: prototype a single mechanism on the
  GPU in a test harness or feature-flagged branch, measure steering alignment and
  control probes (encoder separability, food visibility) against the
  mirrored-steering baseline, record decision (ACCEPT ≥0.70 or REJECT ≤0.62) with
  measured evidence in a decision doc, and integrate the fix if accepted. All
  fixes must respect the homeostasis-only contract: no reward shaping, no goal
  signals, no new signal sources in the TD path. Landing consolidates any/all
  accepted fixes, updates the steering baseline assertion to reflect the new band
  (≥0.70), and documents the final outcome in the plan's STATUS.md. If all three
  reject, the negative result is recorded and the credit bottleneck carries to
  plan 0019.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Auxiliary Steering Objective | `baseline-steering-and-variance-probe`, `auxiliary-steering-spike`, `auxiliary-steering-integration`, `credit-path-fix-lands` | 📋 Planned |
| 0002 | Credit Horizon and Trace Restructuring | `trace-horizon-spike`, `trace-horizon-integration` | 📋 Planned |
| 0003 | Gradient Variance and Signal Shaping | `gradient-variance-diagnosis`, `gradient-shaping-integration` | 📋 Planned |
