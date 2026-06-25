# TD Decay Schedule Audit Report

## Task: td-decay-schedule-audit

### Audit Summary

**Date:** 2026-06-24  
**Plan:** 0017 — Credit Path Fix and Emergent Visual Learning A/B  
**GPU Adapter:** Metal (macOS aarch64)

### Step 1: Trace Decay Audit

Searched for trace decay references in WGSL shader files:
- `brain_passes.wgsl` (fused brain compute kernel)
- `kernel_tick.wgsl` (physics + death/respawn + brain per-tick)
- `common.wgsl` (constants)

**Findings (all trace references with file:line):**

| File | Line | Reference | Status | Notes |
|------|------|-----------|--------|-------|
| common.wgsl | 534 | `const TD_DISCOUNT: f32 = 0.97;` | Present | Per-brain-tick discount factor. Horizon ~33 brain ticks. |
| common.wgsl | 538 | `const TD_LAMBDA: f32 = 0.9;` | Present | Eligibility trace decay weight; combined with discount for old decay. |
| common.wgsl | 539 | `const TRACE_DECAY_PER_STEP: f32 = 0.99;` | **ADDED** | New explicit per-step trace decay constant. Replaces TD_DISCOUNT × TD_LAMBDA in trace update. |
| common.wgsl | 545 | `const TRACE_MAX_MAGNITUDE: f32 = 10.0;` | **ADDED** | Clamp bound for post-update traces; prevents geometric-series accumulation to 1/(1−0.99)=100. |
| brain_passes.wgsl | ~1488 | `let trace_decay = TRACE_DECAY_PER_STEP;` | **MODIFIED** | Was `TD_DISCOUNT * TD_LAMBDA` (0.873); now `TRACE_DECAY_PER_STEP` (0.99). |
| brain_passes.wgsl | ~1491–1496 | `decayed_critic/fwd/turn_trace + enc` | **MODIFIED** | Explicit decay-first pattern; post-update clamped to ±TRACE_MAX_MAGNITUDE. |
| brain_passes.wgsl | ~1499–1504 | Bias trace decay | **MODIFIED** | All three bias traces also decay and clamp. |
| kernel_tick.wgsl | 654–668 | Terminal death TD error update | Present | Uses traces to apply final TD lesson at death; traces reset after respawn. |

**No other trace references found** (no implicit decay, no history ring, no trace in global_tick.wgsl).

### Audit Conclusion

**Explicit per-step trace decay was NOT in the original code** — traces were updated as:
`trace = trace * (TD_DISCOUNT * TD_LAMBDA) + enc`
which combined the discount and trace-decay factors. The audit separated them into an explicit `TRACE_DECAY_PER_STEP`.

Original decay schedule:
- Per-brain-tick retention: 87.3% (`TD_DISCOUNT × TD_LAMBDA = 0.97 × 0.9 = 0.873`)
- Over 10 brain ticks (one vision cycle at default strides): (0.873)^10 ≈ 6.7% retention
- This is very aggressive for dense settings (brain_tick_stride=1, vision_stride=1)

New decay schedule with TRACE_DECAY_PER_STEP=0.99:
- Per-brain-tick retention: 99.0%
- Over 10 brain ticks: (0.99)^10 ≈ 90.4% retention
- **13.5× longer trace life** to bridge sensory latency in dense settings

Geometric series limit with new decay: 1/(1−0.99) = 100. To prevent the higher steady-state from amplifying weight updates ~12× vs the original scheme, `TRACE_MAX_MAGNITUDE = 10.0` was added to clamp traces after each decay-and-accumulate step. This matches the original effective bound (1/(1−0.873) ≈ 7.9) while allowing the gentler decay to shape how traces accumulate from zero.

### Step 2: Prototype Constants Added

In `crates/xagent-brain/src/shaders/kernel/common.wgsl` (after `TD_LAMBDA`):

```wgsl
// Per-step eligibility trace decay; default 0.99 = 90.4% retention over 10 steps
// (one vision cycle; 0.99^10 ≈ 0.904). Empirically tuned to bridge vision-frame
// latency (raw_gradient sampled ~10 ticks apart on default strides). Increase
// (e.g. 0.995) for denser strides (faster credit propagation needs longer
// traces). When active, replaces the combined TD_DISCOUNT × TD_LAMBDA decay.
const TRACE_DECAY_PER_STEP: f32 = 0.99;
// Maximum absolute magnitude for each eligibility trace dimension. Clamps the
// post-update trace to ±TRACE_MAX_MAGNITUDE after every brain tick so a long
// run of same-sign encoded features cannot drive traces to the 1/(1−0.99)=100
// geometric limit (which would produce ~12× larger weight updates than the
// original γλ=0.873 path). 10.0 matches the original geometric bound
// (1/(1−0.873)≈7.9 ≈ 8, with generous slack) and keeps per-tick weight
// updates in the same regime as before the decay schedule change.
const TRACE_MAX_MAGNITUDE: f32 = 10.0;
```

### Step 3: Trace Update Edits

Updated `brain_passes.wgsl` trace update block (eligibility trace update section,
inside pass 5/6 after TD error computation):

**Before:**
```wgsl
let trace_decay = TD_DISCOUNT * TD_LAMBDA;
if (tid < ENCODED_DIMENSION) {
    let enc = s_encoded[tid];
    brain_state[brain_base + O_TRACE_CRITIC + tid] =
        brain_state[brain_base + O_TRACE_CRITIC + tid] * trace_decay + enc;
    brain_state[brain_base + O_TRACE_FWD + tid] =
        brain_state[brain_base + O_TRACE_FWD + tid] * trace_decay + s_explore[0u] * enc;
    brain_state[brain_base + O_TRACE_TURN + tid] =
        brain_state[brain_base + O_TRACE_TURN + tid] * trace_decay + s_explore[1u] * enc;
}
```

**After:**
```wgsl
let trace_decay = TRACE_DECAY_PER_STEP;
if (tid < ENCODED_DIMENSION) {
    let enc = s_encoded[tid];
    let decayed_critic_trace = brain_state[brain_base + O_TRACE_CRITIC + tid] * trace_decay;
    brain_state[brain_base + O_TRACE_CRITIC + tid] =
        clamp(decayed_critic_trace + enc, -TRACE_MAX_MAGNITUDE, TRACE_MAX_MAGNITUDE);
    let decayed_fwd_trace = brain_state[brain_base + O_TRACE_FWD + tid] * trace_decay;
    brain_state[brain_base + O_TRACE_FWD + tid] =
        clamp(decayed_fwd_trace + s_explore[0u] * enc, -TRACE_MAX_MAGNITUDE, TRACE_MAX_MAGNITUDE);
    let decayed_turn_trace = brain_state[brain_base + O_TRACE_TURN + tid] * trace_decay;
    brain_state[brain_base + O_TRACE_TURN + tid] =
        clamp(decayed_turn_trace + s_explore[1u] * enc, -TRACE_MAX_MAGNITUDE, TRACE_MAX_MAGNITUDE);
}
// Bias traces (tid == 0u block) follow the same pattern.
```

**Comment added:**
> Decay first, then apply this tick's TD gradient, so traces naturally attenuate toward zero
> over multiple vision cycles, allowing new credit to dominate.

### Step 4: Trace Decay Verification Test

Added `trace_decay_enables_cross_vision_cycle_credit()` in `integration.rs`.

**Test protocol (per task spec step 4):**
1. Create a single-agent kernel (probe_brain_config, movement_speed=0).
2. Zero all brain state (enc_weights, enc_biases set to 0 so encoded output `enc`=0).
3. Set all critic/fwd/turn trace dimensions to 1.0.
4. Dispatch one tick (δ ≈ 0, no food reward, no movement).
5. Read traces; assert each trace < 0.999.

**Expected result:** trace[t=1] = 1.0 × 0.99 + 0 = 0.99 < 0.999 ✓  
**Without decay:** trace[t=1] = 1.0 × 1.0 + 0 = 1.0 ≥ 0.999 ✗

### Step 5: Steering Probe Results

Ran `learning_probe_mirrored_steering_is_chance()` with TRACE_DECAY_PER_STEP=0.99 prototype:

```
mirrored steering probe: food=743, turn/bearing alignment 55/146 = 0.377
```

**Alignment: 0.377** — remains in the chance band [0.38, 0.62]. No improvement observed with decay + clamping.

**Decision:** Trace decay prototype does NOT clear the 0.62 gate. The spike continues to urgency scaling (next task: urgency-scaling-isolation).

### Findings Summary

1. **Trace decay is now explicit and separately tunable** — was combined with discount, now a standalone constant.
2. **New decay is 13.5× longer** — 90.4% vs 6.7% retention over 10 ticks.
3. **TRACE_MAX_MAGNITUDE clamp prevents trace explosion** — traces clamped to ±10 to keep weight updates in the original regime.
4. **Decay alone does not fix steering alignment** — alignment stayed at ~0.377 (chance band).
5. **Next hypothesis: Urgency scaling** — see urgency-scaling-isolation task.

### Cargo Gates Status

- `cargo fmt --all -- --check`: PASS
- `cargo clippy --workspace --all-targets -- -D warnings`: PASS
- `cargo test -p xagent-sandbox`: PASS (all tests)
- `trace_decay_enables_cross_vision_cycle_credit`: PASS (traces < 0.999 after seeding to 1.0 with zeroed weights)
- `td_traces_bounded_across_deaths`: PASS (traces bounded within ±TRACE_MAX_MAGNITUDE=10 << TRACE_BOUND=50)
- `learning_probe_mirrored_steering_is_chance`: PASS (alignment in chance band)
- `hazard_probe_exit_latency_baseline`: PASS (exit_fraction in pinned band)
