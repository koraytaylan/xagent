# Review: Plan 0009 Intent-Aware Fitness and Calibration

**Date:** 2026-06-18
**Reviewer:** Gemini 3.1 Pro (High)
**Report Path:** `docs/reviews/2026-06-18-gemini-31-pro-high.md`

## Naming Convention
The existing review files in `docs/reviews/` follow the format `YYYY-MM-DD-model-or-topic.md` (e.g., `2026-06-18-gpt-5-codex.md`, `2026-06-18-gemini-31-pro-high.md`, `2026-06-18-claude-opus-48.md`, and `2026-06-18-grok-43.md`). This report complies with this convention by adopting the name `2026-06-18-gemini-31-pro-high.md`.

## Scope
This review focuses on the recent changes up to HEAD, specifically evaluating the implementation and validation of the Intent-Aware Fitness plan:
- `916a25ad` docs(0009-Intent-Aware-Fitness): add fitness-calibration decision doc, fix nits
- `a77163c0` feat(0009-Intent-Aware-Fitness): implement plan 0009-Intent-Aware-Fitness
- `fae65c27` docs(plans): add intent-aware fitness plan (0009)

## Executive Summary
The implementation of the Intent-Aware Fitness mechanisms (effort-rebased fitness, speed-invariant hazard, super-linear drag, and danger percept) successfully executes the core design intent. The flag gating is robust and defaults accurately preserve byte-identical legacy behavior. 

However, evaluating the codebase against strict internal guidelines (`CONTRIBUTING.md`) and the robustness of its validation methods reveals several significant issues. Many of these observations align with findings raised concurrently by Grok 4.3 and GPT-5 Codex, pointing to shared areas of systemic technical debt.

## Findings

### 1. Falsifiability Violation in Calibration Test
**Impact:** High (Rule Violation)
**Location:** `crates/xagent-sandbox/src/governor.rs:2351` (`fitness_calibration_replay_profiles`)
**Description:** The test simply sets up synthetic agent profiles and prints out the composite fitness calculations. It contains no `assert!` statements to enforce the documented deltas or the relative ordering between the "Competent forager" and "Fast-aimless agent". This violates the `CONTRIBUTING.md` rule that "Tests must be falsifiable". If the math regressed, the test would still pass.
**Recommendation:** Introduce explicit `assert!` or `assert_relative_eq!` checks ensuring the competent forager outscores the aimless agent under effort mode by the required margin.

### 2. Temporal Misalignment in Avoidance Intent
**Impact:** Medium (Logic Defect)
**Location:** `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl`
**Description:** Inside the fused kernel, `agent_physics` increments the avoidance intent counters by reading `P_NEAREST_DANGER_DISTANCE` and `P_NEAREST_DANGER_BEARING`. However, `agent_danger_detect`, which computes and updates these values, is executed *after* `agent_physics` in the cycle loop. Consequently, the avoidance counters are driven by stale telemetry from the previous physics tick.
**Recommendation:** Reorder the update logic or extract the counter accumulation to run after `agent_danger_detect` within the cycle.

### 3. Divergent Telemetry Reductions
**Impact:** Low (Consistency)
**Location:** `crates/xagent-sandbox/src/governor.rs` vs `crates/xagent-sandbox/src/headless.rs`
**Description:** When persisting `danger_dwell_fraction` in `governor.rs`, it computes the global ratio: `sum(danger_path_length) / max(sum(distance_traveled), EPSILON)`. Conversely, the validation harness in `headless.rs` calculates a simple unweighted mean of per-agent ratios. This means the validation report might disagree with production persistence if distance-traveled variance is high.
**Recommendation:** Standardize the metric reduction strategy across both the validation harness and database persistence paths.

### 4. Excessive Plan References in Source
**Impact:** High (Rule Violation)
**Location:** Widespread (`config.rs`, `buffers.rs`, `kernel_tick.wgsl`, etc.)
**Description:** The source code contains pervasive references to the planning process (e.g., "plan 0009", "Layer A", "speed-decoupling gate"). `CONTRIBUTING.md` explicitly forbids this: "Source must be agnostic of the planning process... Never name an internal plan... anywhere in source".
**Recommendation:** Perform a global scrub on all touched files to replace plan-centric language with pure technical rationales.

### 5. Uncontrolled Validation Harness
**Impact:** Medium (Measurement Validity)
**Location:** `crates/xagent-sandbox/src/headless.rs`
**Description:** The `validate_speed_decoupling` harness runs baseline and ON trials using separate random seeds for genomes and brain mutations. The resulting deltas in metrics are thus vulnerable to stochastic noise. The pass criteria for the gate also lack a requirement for an actionable baseline correlation or strict improvement constraints.
**Recommendation:** Lock the RNG seed for the validation harness so A/B comparisons evaluate the same genome trajectories, and enforce strict, relative constraints on the correlation shift.

## Comparison with Previous Reports
These findings strongly align with the concurrent reviews from GPT-5 Codex and Grok 4.3 (both dated `2026-06-18`):
- **Grok 4.3** independently raised the plan-reference leakage (Finding 4) as its highest priority concern, confirming a systemic violation of `CONTRIBUTING.md`. It also pointed out the non-falsifiable nature of the calibration test (Finding 1).
- **GPT-5 Codex** extensively detailed the validation harness flaws, calling out the non-paired stochastic nature of the A/B test and the weak pass criteria (Finding 5). It also successfully identified the temporal misalignment in the fused danger bearing (Finding 2) and the divergent metric reductions (Finding 3).
- **Gemini 3.1 Pro (2026-06-12)** previously complained about similar doc/source drift and print-only "tests", demonstrating a historical pattern of technical debt regarding test rigor.

Overall, the architectural intent is solid, but the landing requires an immediate cleanup pass to enforce coding standards, synchronize telemetry paths, and harden the tests before the new flags are enabled by default.
