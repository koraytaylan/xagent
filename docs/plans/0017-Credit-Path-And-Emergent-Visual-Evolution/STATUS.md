# Plan 0017 — Credit Path Fix and Emergent Visual Learning A/B — Four-Phase Measurement-Driven Roadmap — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 📋 Planned.
_Last updated: 2026-06-24, against `develop`._

- **Goal:** Credit path unlocked (steering alignment ≥0.70), emergent encoder
  self-organized and proven correct (orientation selectivity emergent, separability
  >0.5), optimized cortex at ≥50% throughput budget, and A/B verdict recorded for
  encoder promotion to default or keeping both gated.
- **Root cause:** TD actor-critic credit path (decay, traces, urgency scaling) fails
  to propagate vision-→-action gradients over the sensory-frame latency (~10 ticks
  between vision updates), leaving steering alignment at chance despite encoder
  separability 55×. The bottleneck is not the encoder or prior seeding (both tested
  in 0008, 0013), but the credit-assignment mechanism's timescale mismatch with the
  sensory lag.
- **Approach:** Fix credit path via decay-schedule hardening and/or urgency-scaling
  isolation, unlock steering learnability, then independently optimize cortex
  throughput and prototype self-organizing encoder. A/B with seeded-pair determinism
  and 95% CI to adjudicate learned vs imported, honoring the emergence principle by
  making the learned encoder co-equal and measuring the winner fairly.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Credit-Path Diagnosis and Learning Unlock | `baseline-mirrored-steering-probe`, `td-decay-schedule-audit`, `urgency-scaling-isolation`, `credit-path-fix-lands` | 📋 Planned |
| 0002 | Cortex Throughput Optimization | `cortex-throughput-profile-baseline`, `separable-dog-optimization`, `cortex-throughput-optimization-suite` | 📋 Planned |
| 0003 | Emergent Self-Organizing Encoder | `emergent-encoder-spike`, `emergent-encoder-integration` | 📋 Planned |
| 0004 | A/B Comparison and Winner Promotion | `ab-harness-design`, `ab-analysis-and-verdict` | 📋 Planned |
