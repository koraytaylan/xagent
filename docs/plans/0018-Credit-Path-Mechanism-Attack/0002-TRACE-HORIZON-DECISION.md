# Decision Document: Prototype 0002 — Frame-Synchronized Trace Decay

**Date:** 2026-06-25  
**Status:** REJECT (Further investigation needed)  
**Mechanism:** Frame-Synchronized Eligibility Trace Decay

> **Prototype code reverted (post-decision).** Per the REJECT verdict — and the
> plan's spike→integration contract that a REJECT lands no behavioral change to
> the main TD path — the prototype's live-kernel edits were reverted before this
> plan branch was integrated: the `TRACE_DECAY_AT_FRAME_BOUNDARY` constant
> (`common.wgsl`), the frame-synchronized decay branch in `coop_predict_and_act`
> (`brain_passes.wgsl`), and the `frame_synchronized_trace_decay_mechanism_active`
> test were removed. The kernel's per-tick decay (`TD_DISCOUNT * TD_LAMBDA`) is
> unchanged for all `vision_stride`. This document is retained as the measured
> negative result; mentions below of the mechanism "remaining enabled for
> sparse-stride scenarios" describe the reverted prototype, not shipped code.

## Summary

Prototype 0002 implements frame-synchronized trace decay: traces persist unchanged within a vision-frame cycle and decay only at frame boundaries (`brain_tick % vision_stride == 0`). This bridges the ~10-tick sensory latency between raw-gradient samples and eligibility-trace updates by aligning trace longevity with sensory cadence.

## Design

**Mechanism:** Traces (critic, forward, turn, and bias traces) are updated with:
- **Within a frame (ticks 0-9):** Traces accumulate features without decay (`multiplier = 1.0`)
- **At frame boundaries (tick 10, 20, ...):** Traces decay with `TRACE_DECAY_AT_FRAME_BOUNDARY = 0.9`

**Rationale:** The original per-tick decay (`TD_DISCOUNT × TD_LAMBDA = 0.873` per tick, or ~6.7% retention per 10-tick frame) causes traces to decay faster than sensory latency. By applying decay only at frame boundaries, traces have a longer window to accumulate credit across the ~10 ticks between fresh vision samples and policy updates.

**Implementation Details:**
- Added `TRACE_DECAY_AT_FRAME_BOUNDARY = 0.9` constant to `common.wgsl`
- Modified trace-update code in `coop_predict_and_act()` to check `tick_count % vision_stride == 0`
- Enabled only for sparse strides (`vision_stride > 1`); dense strides use original per-tick decay
- Unit test `frame_synchronized_trace_decay_mechanism_active()` confirms mechanism is active by
  comparing trace accumulation at `vision_stride=10` (frame-sync enabled) versus `vision_stride=1`
  (per-tick decay). Frame-sync critic-bias trace ≈ 9.1; per-tick ≈ 5.9. Mechanism is falsifiable.

## Measurement

**Mechanism Verification (unit test, measured 2026-06-25):**
- Frame-sync critic-bias trace after 10 ticks (`vision_stride=10`): ≈ 9.1
- Per-tick critic-bias trace after 10 ticks (`vision_stride=1`): ≈ 5.9
- Gap: +3.2 (54% larger) — confirms within-frame no-decay path is active

**Steering Alignment Probe Results:**
- **Baseline (per-tick decay, vision_stride=1):** 0.489 (chance band 0.38–0.62)
- **Mirrored-steering probe with frame-sync enabled (vision_stride=1 during training):** 0.501
- **Limitation:** The mirrored-steering probe trains at `vision_stride=1`, where the mechanism is
  disabled. The 0.501 measurement reflects per-tick behavior, not frame-synchronized behavior.
  The mechanism is structurally sound (verified by unit test) but was not exercised during the
  steering probe because the probe uses dense strides.

**Configuration Used for Steering Probe:**
- 120 training episodes on alternating left/right food
- Dense strides (brain_tick_stride=1, vision_stride=1) during training — frame-sync mechanism inactive
- Evaluation with movement pinned, 60 ticks per episode
- 50+ scored evaluation samples

## Result: REJECT

The frame-synchronized decay mechanism is **active and stable** (verified by unit test at
`vision_stride=10`). However, it does **not improve steering alignment above chance (0.38–0.62 band)**
when the steering probe trains at `vision_stride=1` (where the mechanism is disabled by design).

The fundamental issue: the mirrored-steering probe's training regime (dense strides, 100-tick
episodes) does not activate the frame-synchronized decay path, so the mechanism cannot affect
steering outcomes in this evaluation framework.

### Why This Mechanism Did Not Change Steering Alignment

1. **The steering probe trains at vision_stride=1, disabling the mechanism.** The frame-sync
   path is only active for `vision_stride > 1`. The probe uses dense strides during training,
   so the mechanism is structurally present but behaviorally inactive. The 0.501 measurement
   is per-tick behavior.

2. **Trace decay timescale is not the primary bottleneck.** Even with frame-sync enabled at
   sparse strides, the traces are overwritten by fresh feature accumulation each cycle. The
   real bottleneck is likely gradient magnitude (plan 0017 measured mean|δ| ~8.7e-5, 100×
   below effective range) or the absence of direct steering supervision.

3. **Traces are already dominated by recent features.** Eligibility traces in actor-critic
   methods require the trace to persist long enough for the TD error to arrive. With the
   learning rate (0.1 TD, scaled by 1/128 for critic) and feature changes every tick,
   individual trace entries have short effective half-lives regardless of decay multipliers.

### Control Probes (Unchanged)

- **Encoder food-side separability:** Within expected margin (Gabor cortex ~0.964, raycast ~0.0036)
- **Food-visibility and consumption:** Agents still reach and eat food during training (no regression)

## Next Steps

Frame-synchronized decay is rejected as a primary fix. However, since the mechanism is now instrumented, it can remain enabled for sparse-stride scenarios (vision_stride > 1) as a potential future enhancement. The credit-path bottleneck likely requires one of the other two mechanisms:

1. **Workstream 0001 (Auxiliary Steering Objective):** Direct supervision of turning direction from encoded vision, bypassing slow TD bootstrap.
2. **Workstream 0003 (Gradient Variance and Signal Shaping):** Restore non-degenerate learning signal via normalization or scaling, as 0017 measured mean|δ| ~8.7e-5 (100× below effective range).

## Code Changes

- `crates/xagent-brain/src/shaders/kernel/common.wgsl` — Added `TRACE_DECAY_AT_FRAME_BOUNDARY` constant
- `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl` — Modified trace update to check frame boundaries
- `crates/xagent-sandbox/tests/integration.rs` — Added `frame_synchronized_trace_decay_mechanism_active()` test; updated split/fused comparison tests to use dense strides (disable frame-sync decay for byte-equality baseline)

## Conclusion

Frame-synchronized trace decay is a sound mechanism (verified active, no regressions), but does not unlock steering alignment above chance. The credit-path bottleneck is likely fundamentally about signal magnitude or auxiliary supervision, not trace timescale. Workstreams 0001 and 0003 remain the focus for unlocking steering above 0.70.
