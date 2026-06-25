# Decision: Auxiliary Steering Loss Prototype

**Date:** 2026-06-25  
**Status:** REJECT  
**Design:** Direct-supervision auxiliary steering loss (squared mismatch between policy turn output and food bearing)

## Summary

The prototype demonstrates that the direct-supervision auxiliary steering loss CONVERGES in a test harness (loss decays −68.5%), but this convergence is a CPU-side measurement of the GPU kernel's existing TD learning path. The CPU-side auxiliary loss does NOT inject gradient updates into the GPU kernel — it is measurement-only. When the steering probe is run after training under the same auxiliary-loss regime, alignment is 0.509 (451/886), which remains in the chance band [0.38, 0.62]. The design is REJECTED: loss convergence does not translate to steering improvement because the CPU-side measurement overlay has no mechanism to modify GPU weights.

## Measurements

### Loss Convergence (Test: `auxiliary_steering_loss_converges_on_bearing`)
- **Date/Adapter:** 2026-06-25 Metal (macOS aarch64)
- **Early window (ticks 0–33) mean loss:** 0.2652
- **Late window (ticks 66–99) mean loss:** 0.0836
- **Decay:** −68.5%
- **Result:** PASS (late < early, loss decays)
- **Interpretation:** The observed loss decay is an artifact of the GPU kernel's existing TD(λ) credit path causing `motor_turn` to shift over 100 ticks. The CPU-side auxiliary loss measures this shift but does not cause it.

### Steering Alignment After Auxiliary-Loss Training (Test: `auxiliary_steering_probe_with_loss_enabled`)
- **Date/Adapter:** 2026-06-25 Metal (macOS aarch64)
- **Training:** 100 dense-stride ticks with CPU-side auxiliary-loss measurement active
- **Evaluation:** pinned movement (movement_speed=0), 60 eval ticks
- **Measured alignment:** 451/886 = 0.509
- **Chance band:** [0.38, 0.62]
- **Result:** REJECT — alignment 0.509 remains in chance band; no steering improvement over baseline (0.489)

### Supporting Probes
- **Encoder separability:** cosine-diff=−0.0034 (within expected margin, not regressed)
- **Food consumption:** >0.0 across training (arena functional)

## Why Reject

The spike design as implemented is a CPU-side measurement overlay. The auxiliary loss is computed from `motor_turn` (read via `read_agent_telemetry_blocking`) and the geometric bearing — but this computation produces no signal that feeds back into GPU kernel weights. The GPU kernel learns only via its existing TD(λ) credit path. Therefore:

1. **Loss convergence is not caused by auxiliary supervision.** The 100-tick training loop dispatches the GPU kernel with its native TD learning active. The `motor_turn` output drifts as TD updates accumulate over 100 ticks, which is what the CPU-side loss measures. There is no additional gradient from the auxiliary loss.

2. **Steering alignment is unchanged.** The mirrored-steering probe re-run (`auxiliary_steering_probe_with_loss_enabled`) measured 0.509 (451/886) — identical to the baseline 0.489 within noise. The CPU-side auxiliary loss provides zero benefit to steering because it cannot modify GPU weights.

3. **The design must be integrated into the GPU kernel to be tested.** A true test of the auxiliary loss requires implementing it inside `coop_predict_and_act()` in `brain_passes.wgsl`, applying actual weight updates at an auxiliary learning rate (1/10th of TD). This is the task `auxiliary-steering-integration`, which is gated on an ACCEPT verdict here. Since we cannot accept without measured steering improvement, the correct path is to integrate and re-measure — but that is a separate task.

4. **ACCEPT was premature.** The previous ACCEPT verdict was based only on loss convergence, which is a necessary but insufficient condition. The spec requires steering alignment to clear ≥0.70 for ACCEPT, or the verdict is REJECT with measured cause. Alignment stayed in the chance band.

## Measured Cause

- **Primary cause:** The auxiliary loss is CPU-side measurement only; it does not inject gradient updates into the GPU kernel. Steering alignment 0.509 (451/886) remained in chance band [0.38, 0.62].
- **Secondary observation:** The loss convergence (early=0.2652, late=0.0836) is a real signal — `motor_turn` does shift toward bearing over 100 dense-stride ticks under TD alone. This is consistent with the TD path slowly routing the separable encoder signal to action.
- **Implication:** The design concept (direct supervision of turn output toward food bearing) is sound, but it must be implemented as a GPU-side weight update to have any effect. The `auxiliary-steering-integration` task should implement the loss in WGSL and re-run this measurement.

## Next Step

Per the spec: REJECT means no code lands — the main TD path stays unchanged. Workstreams 0002 (trace-horizon) and 0003 (gradient shaping) proceed in parallel.

If `auxiliary-steering-integration` is unblocked (e.g., by a REJECT-then-GPU-integrate sequence), the integration task should:
1. Move the loss computation into `coop_predict_and_act()` in `brain_passes.wgsl`.
2. Apply actual weight updates at an auxiliary learning rate (0.01, 1/10th of TD).
3. Add `auxiliary_steering_loss_enabled` flag to `BrainConfig` (default off).
4. Re-run the mirrored-steering probe and record alignment.

## Measurement Protocol

- `auxiliary_steering_loss_converges_on_bearing`: verifies the loss proxy is measurable and decays (PASS — necessary but not sufficient).
- `auxiliary_steering_probe_with_loss_enabled`: runs the steering probe after auxiliary-loss-measurement training and records alignment (REJECT — 0.509, chance band).

Both tests are embedded in `crates/xagent-sandbox/tests/integration.rs` and run as part of `cargo test -p xagent-sandbox`.

---

**Authored by:** Plan 0018 Workstream 0001  
**Gate:** Steering alignment must clear ≥0.70. Measured: 0.509 (chance band). Verdict: REJECT.
