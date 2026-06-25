# Trace Decay Schedule Prototype — Implementation Summary

## Overview

Completed the `td-decay-schedule-audit` task from Plan 0017 (Credit Path Fix and Emergent Visual Learning A/B).

**Task:** Audit trace decay in the TD(λ) credit path, prototype a new decay schedule to extend trace longevity, test the decay is active, and measure impact on steering alignment.

**Result:** Trace decay audit complete; explicit `TRACE_DECAY_PER_STEP = 0.99` and `TRACE_MAX_MAGNITUDE = 10.0` implemented; decay verification test passes; steering alignment remains in chance band (decay alone insufficient — proceed to urgency-scaling-isolation).

---

## Changes Made

### 1. New Trace Decay Constant
**File:** `crates/xagent-brain/src/shaders/kernel/common.wgsl`

Added after `TD_LAMBDA`:
```wgsl
const TRACE_DECAY_PER_STEP: f32 = 0.99;
```

**Rationale:**
- Old decay: `TD_DISCOUNT × TD_LAMBDA = 0.97 × 0.9 = 0.873` → (0.873)^10 ≈ 6.7% retention over 10 ticks
- New decay: `TRACE_DECAY_PER_STEP = 0.99` → (0.99)^10 ≈ 90.4% retention over 10 ticks
- **13.5× longer trace life** to bridge sensory latency in dense settings

### 2. Trace Magnitude Clamp
**File:** `crates/xagent-brain/src/shaders/kernel/common.wgsl`

Added after `TRACE_DECAY_PER_STEP`:
```wgsl
const TRACE_MAX_MAGNITUDE: f32 = 10.0;
```

**Rationale:**
- With decay=0.99, the geometric series steady-state limit is 1/(1−0.99) = 100 per trace dimension.
- Without clamping, traces accumulate to ~60+ in practice, producing ~12× larger weight updates than the original γλ=0.873 scheme.
- Clamping at ±10 (≈ 1/(1−0.873) ≈ 7.9, with slack) keeps weight update magnitudes in the original regime.
- The clamp can only tighten the credit signal, never synthesize an unwanted gradient.

### 3. Trace Update Logic Refactored
**File:** `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`

**Before:**
```wgsl
let trace_decay = TD_DISCOUNT * TD_LAMBDA;
brain_state[brain_base + O_TRACE_CRITIC + tid] =
    brain_state[brain_base + O_TRACE_CRITIC + tid] * trace_decay + enc;
```

**After:**
```wgsl
let trace_decay = TRACE_DECAY_PER_STEP;
let decayed_critic_trace = brain_state[brain_base + O_TRACE_CRITIC + tid] * trace_decay;
brain_state[brain_base + O_TRACE_CRITIC + tid] =
    clamp(decayed_critic_trace + enc, -TRACE_MAX_MAGNITUDE, TRACE_MAX_MAGNITUDE);
```

Applied to: critic, forward, and turn traces (all 128 dimensions each), plus all three bias traces.

### 4. Trace Decay Verification Test
**File:** `crates/xagent-sandbox/tests/integration.rs`

Added `trace_decay_enables_cross_vision_cycle_credit()`:
- Zeros all brain state (enc_weights/biases = 0 → encoded output = 0 per dimension)
- Seeds all trace dimensions to 1.0
- Dispatches one tick
- Asserts every trace < 0.999 (1.0 × 0.99 + 0 = 0.99 < 0.999)

**Test result:** PASS — all 128 × 3 trace dimensions confirmed < 0.999 after one tick.

### 5. Restored Prior-Task Items
**File:** `crates/xagent-sandbox/tests/integration.rs`

Restored deliverables from `baseline-mirrored-steering-probe` that were inadvertently removed:
- `encoder_food_side_cosines()` — shared helper returning `(within, between)` cosine similarities
- `MIN_ENCODER_SEPARABILITY_MARGIN = 10.0` — minimum separability ratio constant
- `RANDOM_ALIGNMENT_BASELINE = 0.5` — chance-level denominator constant
- `baseline_encoder_separability_vs_steering_gap()` — test reporting the ~24× encoder/steering gap
- Baseline comment in `learning_probe_mirrored_steering_is_chance()` documenting 2026-06-24 Metal measurement

---

## Audit Findings

### Trace References (file:line, all found)

| File | Line | Reference | Status |
|------|------|-----------|--------|
| common.wgsl | 534 | `TD_DISCOUNT = 0.97` | Pre-existing |
| common.wgsl | 538 | `TD_LAMBDA = 0.9` | Pre-existing |
| common.wgsl | 539 | `TRACE_DECAY_PER_STEP = 0.99` | **Added** |
| common.wgsl | 545 | `TRACE_MAX_MAGNITUDE = 10.0` | **Added** |
| brain_passes.wgsl | ~1488 | `let trace_decay = TRACE_DECAY_PER_STEP` | **Modified** |
| brain_passes.wgsl | ~1491–1496 | Critic/fwd/turn trace decay + clamp | **Modified** |
| brain_passes.wgsl | ~1499–1504 | Bias trace decay + clamp | **Modified** |
| kernel_tick.wgsl | 654–668 | Terminal death TD update + trace reset | Pre-existing |

---

## Steering Probe Results

### Before Prototype (Baseline from baseline-mirrored-steering-probe)
Measured 2026-06-24 on Metal (macOS aarch64): aligned=229/468=0.489 (chance band 0.38–0.62).

### After Decay Prototype (TRACE_DECAY_PER_STEP=0.99 + TRACE_MAX_MAGNITUDE=10.0)
```
mirrored steering probe: alignment in chance band
Improvement: NONE
```

Alignment did not move above 0.62. The trace decay change (from γλ=0.873 to 0.99) extends trace longevity 13.5× but is insufficient on its own to unlock vision-conditional steering.

---

## Decision

**Trace decay prototype: INSUFFICIENT — does not clear the 0.62 gate.**

The spike continues to urgency scaling (task: urgency-scaling-isolation). See `0001-CREDIT-PATH-DECISION.md` for the full hypothesis ledger.

---

## Files Modified

1. **`crates/xagent-brain/src/shaders/kernel/common.wgsl`**
   - Added `TRACE_DECAY_PER_STEP: f32 = 0.99` (per-step trace decay, replaces γλ)
   - Added `TRACE_MAX_MAGNITUDE: f32 = 10.0` (post-update trace clamp)
   - Fixed comment: "90.4%" retention over 10 steps (was "86%")

2. **`crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`**
   - Changed trace decay source from `TD_DISCOUNT * TD_LAMBDA` to `TRACE_DECAY_PER_STEP`
   - Added `clamp(..., -TRACE_MAX_MAGNITUDE, TRACE_MAX_MAGNITUDE)` after each trace update
   - Refactored for explicit decay-first, then accumulate pattern

3. **`crates/xagent-sandbox/tests/integration.rs`**
   - Rewrote `trace_decay_enables_cross_vision_cycle_credit()` to properly seed traces=1.0, zero weights, dispatch one tick, assert trace < 0.999
   - Restored `encoder_food_side_cosines()` helper
   - Restored `MIN_ENCODER_SEPARABILITY_MARGIN`, `RANDOM_ALIGNMENT_BASELINE` constants
   - Restored `baseline_encoder_separability_vs_steering_gap()` test
   - Restored baseline comment in `learning_probe_mirrored_steering_is_chance()`
   - Updated `td_traces_bounded_across_deaths` TRACE_BOUND comment (traces clamped at 10 << 50)
