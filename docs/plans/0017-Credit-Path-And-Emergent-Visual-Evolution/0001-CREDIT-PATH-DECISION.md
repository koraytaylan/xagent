# 0001 — Credit Path Fix Decision Record

> **OUTCOME (2026-06-25): FALSIFIED — prototypes reverted, not landed.**
> Every candidate fix (TD trace decay `0.99`, urgency-scaling isolation, and the
> two combined) stayed inside the chance band and never cleared the **0.62**
> steering-alignment gate (combined result `0.441` vs `0.489` baseline). The
> prototype shader changes (`TRACE_DECAY_PER_STEP`, `TRACE_MAX_MAGNITUDE`, the
> decay-first trace restructuring, and urgency removal from `raw_gradient_amplified`)
> were therefore **reverted from `develop`** — the live TD credit path is unchanged.
> This document, [`TD_DECAY_AUDIT.md`](TD_DECAY_AUDIT.md), and
> [`TRACE_DECAY_PROTOTYPE_SUMMARY.md`](TRACE_DECAY_PROTOTYPE_SUMMARY.md) are kept
> only as the falsification record. The credit-path bottleneck (gradient
> structuring / auxiliary loss / credit-horizon mismatch) is carried forward to
> **plan 0018**. What *did* land from 0017: the mirrored-steering chance-baseline
> probe, the full WS0002 cortex-throughput optimization, and the WS0003
> emergent-encoder spike.

## Workstream 0001: Credit-Path Diagnosis and Learning Unlock

This document records the measured results and hypothesis status for each spike
in workstream 0001. Each hypothesis is tested against the gate:
**steering alignment > 0.62** on `learning_probe_mirrored_steering_is_chance()`.

Baseline (from `baseline-mirrored-steering-probe`, 2026-06-24, Metal macOS aarch64):
- Alignment = 229/468 = 0.489 (chance band 0.38–0.62)
- Encoder separability (Gabor cortex): cosine-diff ≈ 0.964 (55× margin)
- Encoder separability (default raycast): cosine-diff ≈ 0.0036 (~18–24× above within-class noise)
- Diagnosis: encoder is working; credit path is the bottleneck

---

## Hypothesis 1: TD Trace Decay (task: td-decay-schedule-audit)

**Change:** Replaced combined `TD_DISCOUNT × TD_LAMBDA = 0.873` trace decay with
explicit `TRACE_DECAY_PER_STEP = 0.99`, plus `TRACE_MAX_MAGNITUDE = 10.0` clamp.

**Rationale:** Old decay: (0.873)^10 ≈ 6.7% per 10 ticks → traces decay before
credit from vision-stride latency arrives. New decay: (0.99)^10 ≈ 90.4% → 13.5×
longer trace longevity to bridge sensory lag.

**Measured alignment:** In chance band (did not clear 0.62 gate).

**Status: FAILED — spike continues to Hypothesis 2.**

---

## Hypothesis 2: Urgency Scaling Isolation (task: urgency-scaling-isolation)

**Change:** Removed `(1.0 + urgency)` factor from `raw_gradient_amplified`
(`brain_passes.wgsl:835`), so the TD learning signal uses the bare homeostatic
gradient. Urgency still scales the monitoring loop (`gradient`), but no longer
saturates δ at the [-1,1] clamp during foraging.

**Rationale:** Urgency up to ~5× near death spikes δ to the [-1,1] clamp, collapsing
credit variance during foraging when learning should occur; policy may learn only at
death/respawn events.

**Variance test:** `urgency_isolation_preserves_learning_signal()` — 20 δ samples
over 1000 ticks at mid-energy. Measured mean|δ| = 8.7×10⁻⁵, std|δ| = 6.5×10⁻⁵
(both traces decay + urgency isolation active). Signal is alive and has variance.

**Measured alignment (2026-06-24, Metal macOS aarch64):** 164/372 = 0.441
(still in chance band 0.38–0.62).

**Status: FAILED — urgency isolation alone does not clear the 0.62 gate.**

---

## Combined Fix (trace decay + urgency isolation)

Neither hypothesis individually cleared 0.62:
- Hypothesis 1 (trace decay alone): in chance band.
- Hypothesis 2 (urgency isolation alone): alignment 0.441, in chance band.

**Next step:** Test the combination (TRACE_DECAY_PER_STEP=0.99 + urgency isolation)
together. Both changes are present in this worktree branch simultaneously; the
steering probe result above already reflects both (since td-decay-schedule-audit
landed TRACE_DECAY_PER_STEP and urgency isolation was applied on top). Alignment
0.441 is the combined result, still in chance band.

**Best candidate identified so far:** Neither fix alone moves the needle. The
bottleneck likely lies elsewhere — credit path structuring (auxiliary loss, trace
clipping) or a fundamentally short credit horizon relative to the steering timescale.
Proceed to `credit-path-fix-lands` for a combined attempt or a gradient-structuring
probe.

---

## Gate

A fix is accepted when `learning_probe_mirrored_steering_is_chance()` produces
alignment **above 0.62** (ideally ≥ 0.70). The encoder-separability and food-visibility
probes must still pass. The accepted fix is then landed via `credit-path-fix-lands`.
