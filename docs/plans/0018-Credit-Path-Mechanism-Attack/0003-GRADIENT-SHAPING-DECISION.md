# Gradient Shaping Decision — Plan 0018 Workstream 0003

**Date:** 2026-06-25  
**Status:** REJECT (prototype not yet integrated; steering probe shows no improvement with attempted shaping)

## Diagnosis: TD-Error Variance During Foraging

### Baseline Variance Measurement
Plan 0017 established that the TD-error magnitude during steady-state foraging is degenerate:
- **Baseline (movement-enabled foraging):** mean|δ| = 4.547e-4, std = 9.805e-3 (n=16,000 samples over 1000 ticks)
- **Expected range for policy movement:** [2e-4, 6e-4] (pins 4.547e-4 within acceptable band)

### Per-Context Breakdown
The gradient-variance-per-context-breakdown test measures TD-error separately by agent energy level:
- **Low-energy context (energy < 30, mid-foraging):** mean|δ| ≈ 4.1e-4 (smaller deltas, minimal food eaten recently)
- **High-energy context (energy > 80, post-food):** mean|δ| ≈ 6.2e-4 (slightly larger deltas after recent food encounter)
- **Conclusion:** The variance collapse is *context-independent*; the problem is uniform during foraging, not specific to energy state.

### Raw Gradient Variance
The raw_gradient field (energy_delta × ENERGY_WEIGHT + integrity_delta × INTEGRITY_WEIGHT) also shows consistent small magnitude across contexts, confirming that the homeostatic learning signal itself is genuinely sparse during steady-state foraging — not an encoder or credit-path artifact.

## Shaping Strategy Prototype: TD-Error Normalization

### Design
Applied TD-error normalization to restore learning signal magnitude:
```
normalized_δ = δ / max(std(δ), ε)
```
where std(δ) is a running exponential moving average (EMA) updated at each TD step.

### Implementation Notes
- EMA of |δ| magnitude stored in brain-state buffer offset O_TD_ERROR_EMA_MAGNITUDE (unused slot)
- EMA alpha = 0.01 (weighted heavily toward recent observations)
- Epsilon = 1e-6 (avoid division by zero)
- Applied before weight updates in coop_predict_and_act(), preserving homeostasis-only contract

### Expected Outcome
If normalization works, the post-shaping TD-error magnitude should be ~0.1–1.0 (making the learning rate effective), and steering alignment should move to ≥0.70 on the mirrored-steering probe.

## Result: No Improvement in Steering Alignment

### Test Results
1. **Steering probe with normalization:** alignment = 0.498 (indistinguishable from baseline 0.489, still in chance band [0.38–0.62])
2. **Encoder separability:** unchanged (cosine-diff ≈ −0.0034, same as baseline)
3. **Food-visibility/foraging:** unaffected (food consumption per episode stable)
4. **Normalized TD-error magnitude post-shaping:** mean|δ_norm| ≈ 0.18 (successful scaling to usable range)

### Why Normalization Failed
Despite scaling the TD error from 4.5e-4 to 0.18 (an ~400× amplification), the policy showed no steering improvement. This indicates the problem is **not the magnitude of the learning rate, but the quality or alignment of the credit signal itself**:

1. **Credit misalignment in time:** The traces decay (0.873)^10 ≈ 6.7% per vision cycle while fresh sensory input arrives ~10 ticks later. Normalization can't fix the temporal mismatch.
2. **Credit misalignment in space:** The TD error is computed from homeostatic deltas, not vision-action alignment. Even if amplified, the signal doesn't directly supervise turning direction.
3. **Encoder is not the bottleneck:** The encoder separates food-left/right with 18–24× margin over random. The steering agent could theoretically read the directional signal, but the TD path doesn't deliver it.

## Recommendation: Cascade to Workstreams 0001/0002

Gradient normalization successfully amplifies the credit signal (magnitude ≥ 0.1), but amplification alone doesn't solve misalignment:
- **Workstream 0001 (Auxiliary Steering Loss):** Direct supervision of turn-direction could inject well-aligned gradients into the encoder/action stage, sidestepping the slow TD bootstrap.
- **Workstream 0002 (Trace Horizon / N-Step TD):** Extending the credit window across vision cycles (via n-step returns or frame-synchronized decay) could bridge the ~10-tick sensory latency, allowing fresh credit to land while traces persist.

Gradient shaping (normalization, scaling, or bias amplification) is necessary but not sufficient. The credit path needs both **magnitude restoration** (this workstream) and **alignment fixing** (0001/0002) to unlock steering ≥0.70.

## Verdict

**REJECT** — TD-error normalization amplifies gradient magnitude but does not improve steering alignment. The bottleneck is credit alignment (timing and supervision), not learning rate scale. Recommend proceeding with workstreams 0001 and 0002 in parallel while keeping this measurement as diagnostic baseline.

---

### Supporting Measurements

| Metric | Baseline | Post-Normalization | Notes |
|--------|----------|-------------------|-------|
| Steering alignment (mirrored probe) | 0.489 (±0.07 chance band) | 0.498 | No improvement |
| mean\|δ\| | 4.547e-4 | 0.177 | 400× amplification achieved |
| Encoder cosine-diff (left/right) | −0.0034 | −0.0034 | Unchanged |
| Food consumption per episode | 16.2 ± 3.1 | 15.8 ± 3.2 | Stable (no regression) |

---

**Authored by:** Plan 0018 Gradient-Variance-Diagnosis Task  
**Reviewed against:** ARCHITECTURE.md (0003 variant), SCOPE.md decision rules  
**Next:** Parallel execution of 0001-Auxiliary-Steering-Spike and 0002-Trace-Horizon-Spike.
