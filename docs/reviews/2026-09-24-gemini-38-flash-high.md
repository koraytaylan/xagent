# Validation Review: Evidence-Based Audit & Stabilization of Plan 0023

**Date:** 2026-09-24
**Reviewer:** Gemini 3.8 Flash (High)
**Scope:** Comprehensive evidence-based review loop of the repository, auditing gate hygiene, verifying the newly landed Plan 0023 homeostatic predictive credit mechanism, correcting identified defects, and closing open workstreams.
**Base:** `develop` at `f8945eee` (following Plan 0023 core mechanism commit).
**Output convention:** Follows `docs/reviews/YYYY-MM-DD-reviewer.md`.

## Executive Verdict

**Three primary findings identified, investigated, and fully resolved.**
1. **Gate Hygiene (F1, P1):** `cargo clippy --workspace --all-targets -- -D warnings` was failing on `develop` with 11 errors (8 compiler `float-literal-f32-fallback` warnings elevated to errors under `-D warnings`, 2 `useless-borrows-in-formatting` lints in `headless.rs`, and 3 `chunks-exact-to-as-chunks` lints in `world/entity.rs`, `snapshot.rs`, and `integration.rs`). All sites were refactored and verified clean; workspace clippy and fmt are green.
2. **WGSL Mathematical / Temporal Alignment Bug in Predictor Head (F2, P1):** The newly landed homeostatic gradient predictor head in `brain_passes.wgsl` contained critical temporal alignment and numerical stability flaws. The gradient descent weight update multiplied prediction error by current-tick predicted features rather than the previous-tick features that generated the prediction; the reward blend published future predictions instead of the previous-tick anticipatory gradient; and steps were not scaled by dimension (`TD_VECTOR_SCALE`), driving runaway divergence (diagnostic MAE = 23.16). Corrected with proper temporal alignment, dimension scaling, and bound clamps, driving MAE down 436× to 0.053108.
3. **Plan 0023 Steering Probe and Decision Gate Completion (F3, P1):** Workstreams 0002 (steering probe) and 0003 (decision gate) were left unexecuted. Implemented `homeo_predictive_credit_steering_probe` in `integration.rs` with Clopper–Pearson 95% CI calculation. Measured alignment is **54 / 239 = 0.226 (95% CI [0.173, 0.279])** vs chance band [0.38, 0.62]. Rendered mechanical **REJECT** (CI upper 0.279 ≤ 0.62), authored `0001-HOMEO-PREDICTOR-DECISION.md`, and updated all plan status tracking.

## Verification Performed

```text
cargo fmt --all -- --check                                    # GREEN (exit 0)
cargo clippy --workspace --all-targets -- -D warnings         # GREEN (exit 0, 0 warnings)
cargo test -p xagent-sandbox --lib                            # 109 passed (exit 0)
cargo test -p xagent-sandbox --test contributing_guard       # 2 passed (exit 0, ratchet holds)
cargo test -p xagent-sandbox --test integration homeo        # 2 passed (smoke + steering probe, exit 0)
```

## Finding-by-Finding Detail

### F1 (P1) — Workspace Clippy Failure on `develop` — ✅ Resolved

- **Defect:** `cargo clippy --workspace --all-targets -- -D warnings` exited with code 101 on `develop`.
  - `crates/xagent-sandbox/src/ui.rs`: 8 instances of `egui::Stroke::new` with untyped float literals (`0.5`, `1.5`, `2.0`, `1.0`) triggered the compiler's `float-literal-f32-fallback` future-incompatibility lint.
  - `crates/xagent-sandbox/src/headless.rs`: 2 instances of redundant borrows in `format!("{}-wal", &temp_db)` and `format!("{}-shm", &temp_db)` triggered `clippy::useless-borrows-in-formatting`.
  - `crates/xagent-sandbox/src/world/entity.rs`, `snapshot.rs`, `tests/integration.rs`: `chunks_exact` calls with constant strides triggered `clippy::chunks-exact-to-as-chunks`.
- **Resolution:**
  - Added explicit `_f32` type suffixes to all Stroke float literals in `ui.rs`.
  - Removed redundant references in `headless.rs`.
  - Migrated constant chunk slicing to `as_chunks::<N>().0` across `entity.rs`, `snapshot.rs`, and `integration.rs`.
- **Verification:** `cargo clippy --workspace --all-targets -- -D warnings` runs in 0.13s and exits 0 with zero warnings.

### F2 (P1) — WGSL Temporal Misalignment & Numerical Divergence — ✅ Resolved

- **Defect:** In `brain_passes.wgsl::coop_predict_and_act`:
  - **Temporal alignment error:** When updating `O_HOMEO_PREDICTOR_WEIGHTS`, the online gradient update computed `pred_lr * pred_error * s_prediction[tid]`. But `pred_error` is `prev_pred - actual`, where `prev_pred` was generated at $t-1$ from $s_{prediction}^{t-1}$ (stored in `O_PREV_PREDICTION`). Multiplying by $s_{prediction}^t$ applied the error to the wrong feature vector.
  - **Reward blend misalignment:** `s_homeo[7u]` was set to `predicted_gradient` ($t+1$ expectation) rather than `prev_pred` ($t-1 \to t$ expectation), violating the contractual definition of anticipatory credit for the transition just completed.
  - **Numerical instability:** Weight updates lacked $1 / \text{ENCODED\_DIMENSION}$ dimensional scaling (analogous to `TD_VECTOR_SCALE` in the TD critic), and error/prediction terms lacked bounds clamps. The linear head rapidly diverged; initial probe diagnostic measured MAE = 23.161018 on targets bounded in $[-0.3, 0.3]$.
- **Resolution:**
  - Tied weight updates to `brain_state[brain_base + O_PREV_PREDICTION + tid]` with `TD_VECTOR_SCALE` dimensional scaling.
  - Bound `predicted_gradient`, `pred_error`, and bias updates within `[-MAX_HOMEOSTATIC_DELTA, MAX_HOMEOSTATIC_DELTA]`.
  - Assigned `s_homeo[7u] = prev_pred` so the TD reward blend and telemetry readback accurately reflect the transition's anticipatory prediction.
- **Verification:** Prediction error converged to MAE = **0.053108** across 1,666 diagnostic samples (~436× reduction in error).

### F3 (P1) — Plan 0023 Steering Measurement & Decision Gate — ✅ Resolved

- **Defect:** Plan 0023 Workstream 0002 (`steering-probe`) and Workstream 0003 (`decision-gate`) were open with no test or decision recorded.
- **Resolution:**
  - Added `homeo_predictive_credit_steering_probe` to `crates/xagent-sandbox/tests/integration.rs` with Clopper–Pearson 95% CI calculation.
  - Executed the probe: 120 training episodes, 816 food consumed, 54 / 239 turn/bearing alignment = 0.226 (95% CI [0.173, 0.279]).
  - Rendered mechanical **REJECT** because CI upper bound 0.279 $\le 0.62$.
  - Authored `docs/plans/0023-Homeostatic-Predictive-Credit/0001-HOMEO-PREDICTOR-DECISION.md` documenting root cause (state-value anticipation lacks counterfactual action contrast) and fallback structural candidates.
  - Updated Plan 0023 `STATUS.md` and root `docs/plans/STATUS.md` roll-up rows.
- **Verification:** Both `homeo_predictive_credit_flag_is_inert_when_disabled` and `homeo_predictive_credit_steering_probe` pass reliably.

## Gate Health Summary

| Gate | Status | Notes |
|---|---|---|
| `cargo fmt --all -- --check` | 🟢 GREEN | Clean diff |
| `cargo clippy --workspace --all-targets -- -D warnings` | 🟢 GREEN | 0 warnings across all crates |
| `cargo test -p xagent-sandbox --lib` | 🟢 GREEN | 109 passed |
| `cargo test -p xagent-sandbox --test contributing_guard` | 🟢 GREEN | 2 passed, ratchet holds |
| `cargo test -p xagent-sandbox --test integration homeo` | 🟢 GREEN | 2 passed (smoke + steering probe) |
