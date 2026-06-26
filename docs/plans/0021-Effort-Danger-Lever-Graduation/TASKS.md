# XAgent Plan 0021 — Effort-Danger-Lever-Graduation

Plan 0021 resolves the "gated lever accumulation" pattern flagged in the 2026-06-25 review: effort_rebased_fitness (math fixed by 0014) and danger_percept_enabled (0009/0010) have undergone correctness work but sit default-off with no graduation decision. This plan runs the hardened speed-decoupling A/B harness on production seeds to measure effort-fitness impact and record flip-or-retire verdict, runs danger-percept-enabled on production seeds with the same paired A/B to measure intent/survival/steering delta, and records a terminal decision for innate_instincts_enabled (0013's REJECT gate already documented). All three decisions are recorded as either flip-or-retire commitments with measured evidence, or documented unlock conditions if deferred.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md) for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/{plan_slug}--{id}/`).
- **Depends on** lists *direct* prerequisites only ("—" means none).
- **Done when** is the verifiable acceptance criterion; every task must keep
  `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo test -p xagent-sandbox` green (stated as "cargo fmt/clippy/test green").
- GPU tests self-skip without an adapter (`GpuKernel::is_available()`); CI runs Mesa lavapipe.
- Line numbers are hints; locate every site by the named symbol (grep).

---

## 0001 — Effort-Fitness-Production-A-B

### effort-fitness-production-a-b — Run Effort-Fitness Production A/B and Record Flip-Or-Retire Decision

The validate_speed_decoupling harness (`crates/xagent-sandbox/src/headless.rs:419-458`) runs a seeded-deterministic paired A/B: baseline (all flags OFF) vs ON (effort_rebased_fitness=true, speed_cost_exponent=2.0, danger_percept_enabled=false). It measures mean_ticks_alive, mean_composite_fitness, Pearson correlation between evolved movement_speed and fitness, and danger_dwell_fraction. The harness was used to validate Plan 0010 but has no recorded production-scale run and no regression test. Plan 0014 fixed the math (removed ticks factor, re-derived targets), and the calibration replay passes, so the lever is ready for a graduation decision. This task instruments the harness output with bootstrap 95% CI and records a machine-readable decision summary.

**Steps:**
1. Locate `crates/xagent-sandbox/src/headless.rs:419-458` (validate_speed_decoupling function). The function calls run_headless_with_flags twice (baseline and ON) and computes metrics via print_validation_metrics.
2. Extract the hardcoded num_generations and population size (current values ~50 gen, ~10 pop local test scale) into function parameters; default to 50 generations and population 100 for production runs. Confirm the function accepts config.world.seed and uses it for deterministic replay.
3. Wrap the two run_headless_with_flags calls and the metric computation in a bootstrap loop: collect N=100 replicate runs (each with a seeded but independent world, same generation/population count), compute each run's mean_ticks_alive, mean_fitness, speed_correlation, danger_dwell_fraction, store the 100-value vectors.
4. Compute for each metric: point estimate (mean of 100 replicates), lower 95% CI (2.5th percentile), upper 95% CI (97.5th percentile), and effect size (ON - baseline). Output as JSON: {metric_name: {point: X, lower_ci: Y, upper_ci: Z, effect: W}}.
5. Append a decision rule block that evaluates the metrics against thresholds: (a) 'speed_correlation BASELINE is strongly positive (>=0.5)' check the CI, (b) 'speed_correlation ON is near zero (in [-0.2, 0.2])' check the CI, (c) 'ticks_alive_mean ON within [-10%, +5%] of baseline' check the overlap of CIs, (d) 'danger_dwell_fraction ON >= baseline * 0.8' check the lower CI. Print a decision summary: 'FLIP if (a AND b AND c AND d)', 'RETIRE if (ON is negative on any axis OR ticks drop >10%)', 'DEFER if (thresholds do not align)'.
6. Run the instrumented harness locally on a CPU machine or the lab GPU with --validate-speed-decoupling --validation-generations 50 and capture the JSON output to a timestamped file (e.g., speed_decoupling_bootstrap_2026_06_25.json). Confirm no stderr warnings and the test completes without hang or panic.
7. Record the JSON output as an artifact (commit it as part of the plan's decision doc) and write the 0001-EFFORT-FITNESS-DECISION.md using the JSON as the measured evidence section.

- **Depends on:** —
- **Done when:** The validate_speed_decoupling harness accepts population and generation parameters, runs N=100 bootstrap replicates on production scale (100 pop, 50 gen), computes 95% CI for all four metrics, outputs JSON with point/CI/effect, and prints a human-readable decision rule and verdict. The decision is recorded in a 0001-EFFORT-FITNESS-DECISION.md artifact with measured evidence, thresholds, and outcome (flip/retire/defer). Cargo fmt/clippy/test green.

---

## 0002 — Danger-Percept-Production-A-B

### danger-percept-production-a-b — Run Danger-Percept Production A/B and Record Flip-Or-Retire Decision

The danger-percept-enabled flag gates (a) encoding of 2 danger-feature slots in the brain input, (b) avoidance-intent accumulation in kernel_tick.wgsl, and (c) avoidance potential-based reward shaping. The hardened tests (`crates/xagent-brain/tests/intent_baseline_measurement.rs`, `crates/xagent-brain/tests/danger_percept_ablation_ab.rs`) measure the intent distribution and a paired comparison with/without the percept signal. This task extends run_headless_with_flags (`crates/xagent-sandbox/src/headless.rs:628-664`) to run a full production A/B (danger_percept OFF baseline vs ON) with bootstrap 95% CI, measuring avoidance-intent fraction, survival (ticks_alive), and steering alignment. A new decision doc records the verdict and unlock conditions if deferred.

**Steps:**
1. Locate run_headless_with_flags (`crates/xagent-sandbox/src/headless.rs:628-664`); it currently takes effort_rebased_fitness, speed_cost_exponent, and innate_instincts_enabled flags and returns a ValidationStats struct with mean_ticks_alive, mean_fitness, mean_avoidance_intent_fraction, etc.
2. Confirm the function computes mean_approach_intent_fraction and mean_avoidance_intent_fraction from the telemetry (P_APPROACH_* and P_AVOIDANCE_* counters). Locate the steering_alignment computation (likely a per-agent approach+avoidance weighted decision or a separate telemetry column). If steering_alignment is not yet captured in ValidationStats, add it as a field and compute it in the readback loop (e.g., mean of per-agent (turn_action - expected_turn_from_gradient) correlation or a direct approach/avoidance intent sum).
3. Call run_headless_with_flags twice: (a) baseline with danger_percept_enabled=false, all other flags off, (b) ON with danger_percept_enabled=true, all other flags off. Repeat N=100 times (bootstrap loop), each with a seeded but independent world, 50 generations, population 100.
4. For each of the 100 replicates, store the four metrics: mean_ticks_alive, mean_avoidance_intent_fraction, mean_approach_intent_fraction, mean_steering_alignment. Compute 95% CI for each metric.
5. Append a decision rule: (a) 'avoidance-intent ON > baseline' (lower CI of ON > point of baseline), (b) 'survival ON within [-5%, +5%] of baseline' (CI overlap), (c) 'steering_alignment ON > 0.62 OR in same band as baseline' (separate outcomes: good steering = flip, steering at chance = defer, steering regressed = retire). Print the decision verdict and thresholds.
6. Run the harness on production seeds with --validate-danger-percept --validation-generations 50 and capture JSON to a timestamped artifact.
7. Write 0002-DANGER-PERCEPT-DECISION.md with the measured evidence, decision rule, and outcome (flip/retire/defer), explicitly noting if steering_alignment is in the chance band [0.38, 0.62] and the unlock condition (steering must exceed 0.62 on the default learning path before danger-percept-enabled contributes independently).

- **Depends on:** effort-fitness-production-a-b
- **Done when:** The run_headless_with_flags harness runs N=100 production-scale A/B replicates with danger_percept_enabled OFF vs ON, computing 95% CI for avoidance-intent, approach-intent, ticks_alive, and steering_alignment. Decision doc 0002-DANGER-PERCEPT-DECISION.md records measured evidence, decision rule (flip/retire/defer), and unlock conditions if steering remains in chance band. Cargo fmt/clippy/test green.

---

## 0003 — Innate-Instincts-Graduation-Status

### innate-instincts-terminal-or-gated-mark — Mark Innate-Instincts Decision as Terminal or Gated-on-Credit-Path

Plan 0013 ran the prove-or-kill gate (run_innate_instinct_ab, `crates/xagent-sandbox/src/headless.rs:492-560`) and recorded three gate failures in 0013-INNATE-INSTINCT-DECISION.md. The decision doc lists potential revisit conditions (danger_percept ON, weaker seeds, longer runs), but the decision's long-term status is unclear: is this a permanent REJECT (the mechanism does not work), or is it deferred pending credit-path improvements (a seeded prior cannot act through a chance-level credit path)? The 0018 gradient-shaping result (magnitude has no effect on steering, credit alignment is the bottleneck) clarifies that a static prior cannot overcome the credit bottleneck. This task marks the decision as either terminal (flag ships complete, research avenue closed) or gated-on-credit-path (unlock when steering > 0.62).

**Steps:**
1. Read 0013-INNATE-INSTINCT-DECISION.md in full. Confirm the three gate failures: survival +0.19% (threshold +10%), alignment 0.000 (threshold 0.4, caveat danger_percept OFF), food-per-death 0.37 (threshold 2.0).
2. Decide on the research direction (consult with the project lead or document the assumption): (a) **Terminal**: the blank-slate architecture is the committed path; seeding ships complete but will not be revisited unless the project fundamentally rethinks the learning structure. (b) **Gated on credit path**: innate instincts are a valid research direction but cannot be fairly evaluated until the credit path (steering alignment) is above chance; revisit this flag when a future plan demonstrates steering > 0.62.
3. If **Terminal**: append a closing section to 0013-INNATE-INSTINCT-DECISION.md: 'Terminal decision (recorded YYYY-MM-DD plan 0021): the innate-instincts mechanism is complete, proven not to contribute under the current credit path, and archived. The seeding code ships behind the default-off flag for future research reference. The blank-slate learning model is the committed baseline. Future credit-path improvements do not automatically re-open this gate; it remains closed unless the project explicitly decides to revisit seeded-prior research.'
4. If **Gated on credit path**: append a section to 0013-INNATE-INSTINCT-DECISION.md: 'Decision deferred (recorded YYYY-MM-DD plan 0021): the mechanism is sound but the credit bottleneck prevents any prior from being reliably acted upon. This gate re-opens when a subsequent plan achieves steering_alignment >= 0.62 on production seeds with the default learning path. At that point, re-run this A/B with danger_percept_enabled ON and innate_instincts_enabled ON/OFF to measure the seeded-prior contribution to the improved signal.' Also update the 0010 STATUS row to reference the deferred condition.
5. Update docs/plans/STATUS.md row 0013 to reflect the decision: if terminal, note 'Gate FAIL, flag stays default-off, decision TERMINAL. Seeding shipped complete.' If gated, note 'Gate FAIL, flag stays default-off, decision DEFERRED pending credit-path above 0.62.'
6. This is documentation-only; no code changes. Ensure the decision doc is clear enough that a future planner will understand the precondition and can execute the gate if it re-opens.

- **Depends on:** —
- **Done when:** 0013-INNATE-INSTINCT-DECISION.md is updated with an explicit terminal or gated-on-credit-path marking, with the decision date and reasoning. docs/plans/STATUS.md row 0013 is updated to reflect the marking. No code changes; purely clarifying the decision-doc status. Documentation-only task; no gate-command requirement.

---

**End of plan 0021 TASKS.** When every "Done when" bullet is green, the plan's end state is reached.
