# Plan 0018 — Credit-Path Mechanism Attack: Auxiliary Loss, Trace Horizon, and Gradient Shaping — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete — **negative result.** All three spikes ran; WS0001 CPU-side overlay rejected (mechanism not tested at GPU-integrated level — deferred); WS0002 and WS0003 rejected under their tested conditions. No fix cleared the ≥0.70 gate, so no mechanism was integrated and the mirrored-steering baseline assertion stays at `[0.38, 0.62]`. The credit-path bottleneck carries forward to plan 0019.
_Last updated: 2026-06-26, against `develop`._

## Outcome

The three measurement gates held throughout: steering alignment stayed in the
chance band, encoder food-side separability was unchanged, and food visibility
held — so the encoder is confirmed not to be the bottleneck.

- **0001 Auxiliary steering loss — REJECT (CPU-side overlay; mechanism not tested at GPU-integrated level).** A direct-supervision auxiliary loss
  converges in a CPU-side test harness (−68.5%), but the harness only *measures*
  the existing GPU TD path; it injects no GPU weight updates. Steering stayed at
  **0.509** (chance). The CPU-side measurement overlay is rejected; the mechanism
  itself (GPU-integrated auxiliary loss with actual weight updates in
  `brain_passes.wgsl`) was never tested at the level required for falsification and
  is deferred to a future plan (e.g. 0020) for GPU-integrated evaluation.
- **0002 Frame-synchronized trace decay — REJECT.** Prototype only acts at
  `vision_stride>1`, but the mirrored-steering probe trains at `vision_stride=1`
  where it is disabled by design → **0.501**, mechanism behaviorally untested for
  steering. The prototype's live-kernel code was **reverted** post-decision (it
  had landed unflagged under a REJECT); the decision doc is retained as the
  measured negative result.
- **0003 Gradient-variance shaping — REJECT, and the sharpest finding.** TD-error
  normalization amplified mean|δ| from **4.5e-4 → 0.18 (~400×)** yet steering
  stayed at **0.498** (chance). Magnitude is *not* the bottleneck. The credit
  signal's **temporal alignment and lack of direct supervision** are.

**Carry-forward to plan 0019:** the credit path's problem is *alignment*, not
learning-rate scale — making the gradient larger moved steering by zero. Fixes
must deliver the (already-separable) directional encoder signal to action with
correct timing/supervision. The three gated integration tasks remain unrun on
branch `implement-plan/0018`; revisit them only with a mechanism that actually
writes GPU weights and is measured at the probe's training stride.

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
| 0001 | Auxiliary Steering Objective | `baseline-steering-and-variance-probe` ✅ landed, `auxiliary-steering-spike` ✅ landed (REJECT), `auxiliary-steering-integration` ⛔ gated (not run), `credit-path-fix-lands` ⛔ blocked (not run) | ❌ Rejected |
| 0002 | Credit Horizon and Trace Restructuring | `trace-horizon-spike` ✅ landed (REJECT, prototype reverted), `trace-horizon-integration` ⛔ gated (not run) | ❌ Rejected |
| 0003 | Gradient Variance and Signal Shaping | `gradient-variance-diagnosis` ✅ landed (REJECT), `gradient-shaping-integration` ⛔ gated (not run) | ❌ Rejected |

**What landed on `develop`:** the baseline + spike *measurement* tests and the
three decision docs (the negative-result record). No mechanism, flag, or
TD-path behavior change shipped. Gated integration tasks (`*-integration`) and
`credit-path-fix-lands` were never run and have no branches beyond the planned
worktrees; pursue them under plan 0019 if a viable mechanism emerges.
