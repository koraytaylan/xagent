# Decision: GPU-Integrated Auxiliary Steering Loss

**Date:** 2026-06-26
**Status:** REJECT
**Design:** GPU-side bearing-aligned auxiliary loss injecting gradients into `O_ACTION_TURN_WEIGHTS` and `O_ACTION_FORWARD_WEIGHTS`.

## Summary

Measured alignment rate: 54 / 475 = 0.114 (95% CI [0.085, 0.142]). Verdict: CI upper bound 0.142 << 0.62 (chance band). GPU-integrated auxiliary loss does not raise steering above chance; the mechanism is rejected at this configuration.

## Measurements

#### Steering Alignment After GPU-Auxiliary-Loss Training (Test: `gpu_auxiliary_steering_alignment_probe`)
- **Date/Adapter:** 2026-06-26 Metal (local GPU)
- **Training:** 100 dense-stride ticks with GPU-integrated auxiliary loss enabled
- **Evaluation:** pinned movement, 475 eval samples (16 agents × 60 ticks = 960 possible; ~475 passing bearing-window filter [0.05, 0.6] and non-zero motor_turn, scored by turn correctness against bearing target)
- **Measured alignment:** 54 correct / 475 total = 0.114 (95% CI [0.085, 0.142])
- **Chance band:** [0.38, 0.62]
- **Result:** REJECT — CI upper 0.142 << 0.62 (clear rejection)

#### Supporting Probes
- **Encoder separability:** > 0.9 (unchanged, not regressed)
- **Food consumption:** > 0 (arena functional)

## Why Reject

The GPU-integrated auxiliary loss with bearing targets and direct weight updates still leaves steering **far below** the chance band (0.114 vs 0.489 baseline, 77% worse). This is a decisive REJECT: the auxiliary mechanism with GPU injection of weight updates into `O_ACTION_TURN_WEIGHTS` is **insufficient to unblock credit alignment**. 

The measured degradation (0.114 vs baseline 0.489) is counterintuitive and suggests that the direct supervision signal is **misaligned with the actual credit path** or that the loss injection itself is destabilizing the learned policy. The encoder separates food-bearing adequately (cosine-diff 0.964 ~55×), but the auxiliary loss does not translate bearing targets into steering action — it may actively interfere.

## Measured Cause

Primary cause: GPU-integrated auxiliary loss with bearing supervision is **not sufficient** to raise steering above chance; worse, it degraded from baseline. This is a **structural failure**, not a tuning issue.

Possible root causes:
1. **Bearing target misalignment:** The bearing target computed on GPU (atan2 between agent and food) may not correspond to the actual TD credit signal the agent learns through exploration. The auxiliary loss supervises toward the instantaneous bearing, but credit may accumulate over delayed, exploratory trajectories where the instantaneous bearing is uncorrelated with eventual reward.
2. **Loss injection destabilizes learned policy:** The auxiliary gradient on `O_ACTION_TURN_WEIGHTS` may conflict with TD learning, causing oscillation or catastrophic forgetting. The weight updates are not gated by the TD signal or confidence, so spurious or noisy auxiliary signals could corrupt the policy.
3. **Encoder cannot route action-labeled features:** The encoder learned under homeostasis-only (no shaping, no direct supervision) may have routed features to encode food *detection* (for survival decisions) but not *direction* (for steering). Forcing a bearing-supervision signal onto action weights does not change the feature routing; it just amplifies confusion.
4. **Action-weight channel remains isolated from sensory latency:** The 10-tick sensory latency means the turn action is taken before bearing information settles. Direct supervision of turn output toward bearing does not solve the credit assignment over latency; it just supplies a noisy target. The TD path still fails to bridge the gap.

## Next Step

Record this decision doc. The credit-path problem carries forward to **Plan 0021** with the structural-rethink hypothesis. No integration task runs. The auxiliary-loss mechanism is rejected as a salvageable approach at this scope.

## Structural Candidates for Plan 0021

The next plan should prototype one of the following with the same measurement discipline (GPU-gated, steering probe, decision doc):

| Candidate | Why it might work | Structural change |
|---|---|---|
| **n-step returns** | Explicitly bridges the 10-tick sensory latency without relying on trace decay. Credit accumulates in lookahead, not in eligibility. | Modify `coop_predict_and_act()` in `brain_passes.wgsl` to compute n-step returns (e.g., n=3–5 or n=10) instead of 1-step TD-error scaling. |
| **Eligibility-decay-per-timestep rework** | Current traces decay as (0.873)^tick; at 10 ticks, they are ~6.7% of initial. Perhaps the decay is too aggressive or misaligned with the sensory rhythm. | Modify trace accumulation and decay in vision-tick and brain-tick phases. Experiment with per-timestep vs per-vision-cycle decay. |
| **Eligibility-reset at vision boundary** | Vision latency is the dominant time constant; traces spanning multiple vision cycles may accumulate misaligned credit. | Add trace reset logic at vision update boundary. Reset (or sharply reduce) traces each time vision updates. |
| **Auxiliary-head supervision with shared encoder** | The current auxiliary loss only updates action weights; a shared encoder supervision might unblock feature routing. | Add an auxiliary prediction head (e.g., predicting bearing directly) in `phase_brain_tail_from_scratch.wgsl` or `brain_passes.wgsl` that backprops through the shared encoder. |

**Authored by:** Plan 0020 Workstream 0002 (auxiliary-loss-decision-and-integration-gate task)

**Gate:** Steering alignment must clear >= 0.70. Measured: 0.114. Verdict: **REJECT** (CI upper 0.142 << 0.62).
