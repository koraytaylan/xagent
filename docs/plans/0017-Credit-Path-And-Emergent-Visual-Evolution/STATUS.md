# Plan 0017 — Credit Path Fix and Emergent Visual Learning A/B — Four-Phase Measurement-Driven Roadmap — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 🚧 Partially landed — credit-path fix **FALSIFIED**, optimization landed.
_Last updated: 2026-06-26, against `develop`._

**Outcome:** WS0001's premise (decay-schedule hardening and/or urgency isolation
unlocks steering) was **falsified** — no candidate cleared the 0.62 gate (combined
`0.441` vs `0.489` baseline), so the prototype shader changes were **reverted from
`develop`**; only the chance-baseline probe and the falsification record landed
([`0001-CREDIT-PATH-DECISION.md`](0001-CREDIT-PATH-DECISION.md)). **WS0002 (cortex
throughput) is ⛔ blocked — the ≥50% budget assertion passes only on CI (lavapipe);
real-GPU hardware (Metal) measures ~83 tps vs ~7,625 tps baseline (~1.1%), a 45× miss
recorded in CORTEX-PROFILE-BASELINE.txt (18/256 active lanes, ~7% occupancy).
Workgroup restructuring required; carried to plan 0022.** WS0003's encoder spike landed with a **DEFER** verdict
([`0003-EMERGENT-ENCODER-DECISION.md`](0003-EMERGENT-ENCODER-DECISION.md)). WS0004 is
**superseded** — the A/B can't run until an encoder is integrated, which is gated on a
working credit path. The credit-path bottleneck is carried forward to **plan 0018**.

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
| 0001 | Credit-Path Diagnosis and Learning Unlock | `baseline-mirrored-steering-probe` ✅, `td-decay-schedule-audit` 🧪 (falsified, reverted), `urgency-scaling-isolation` 🧪 (falsified, reverted), `credit-path-fix-lands` ❌ (no fix cleared the gate) | ❌ Falsified |
| 0002 | Cortex Throughput Optimization | `cortex-throughput-profile-baseline` ✅, `separable-dog-optimization` ✅, `cortex-throughput-optimization-suite` ✅ | ⛔ Blocked (real-GPU 1.1% budget miss) |
| 0003 | Emergent Self-Organizing Encoder | `emergent-encoder-spike` ✅, `emergent-encoder-integration` ⛔ (gated → deferred) | 🚧 Spike landed, integration deferred |
| 0004 | A/B Comparison and Winner Promotion | `ab-harness-design` ⛔, `ab-analysis-and-verdict` ⛔ | 🗄️ Superseded (needs integrated encoder + working credit path) |

**Carried to plan 0018:** credit-path bottleneck (gradient structuring / auxiliary
loss / credit-horizon mismatch); emergent-encoder GPU integration; encoder A/B.
