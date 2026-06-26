# XAgent Plan 0019 — Gate-Hygiene-Credit-Path-Honesty

Plan 0019 fixes mechanical gate-integrity regressions: (1) run cargo fmt to fix ui.rs line-break defect on develop; (2) correct 0017 cortex STATUS to state real-GPU 1.1% budget miss not Done; (3) add regression-test harness invocation for 0013 A/B gate; (4) gate diagnostic eprintln tables behind XAGENT_VERBOSE_PROBES; (5) update test paths in STATUS docs to use full crate-relative paths. None flips a default or changes shipped signal.

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

## 0001 — Fmt-Gate-Restore

### fmt-ui-texture-key — Fix cargo fmt Failure on ui.rs:1142-1145

Commit 1da6e0f5 on develop contains ui.rs:1142-1145 with line breaks that cargo fmt wants to collapse. Commit message claims fmt green but gate is red. CONTRIBUTING.md lists cargo fmt as first CI gate.

**Steps:**
1. Run cargo fmt --all to format entire workspace.
2. Verify ui.rs:1142-1145 assignments collapse to single lines.
3. Run cargo fmt --all -- --check to confirm no issues remain.

- **Depends on:** —
- **Done when:** ui.rs:1142-1145 assignments are single-line per rustfmt style; cargo fmt/clippy/test green.

---

## 0002 — Cortex-Throughput-Honesty

### cortex-status-honesty — Correct 0017 STATUS to State Real-GPU Budget Miss

Plan 0017 STATUS marks cortex WS0002 Complete with 50% budget on CI; CORTEX-PROFILE-BASELINE.txt records real GPU at 1.1% (Metal ~83 tps vs ~7,625 tps baseline). Test skips on GPU. Readers conclude cortex met budget when real hardware is 45x below target.

**Steps:**
1. Open docs/plans/0017-Credit-Path-And-Emergent-Visual-Evolution/STATUS.md and update WS0002 row to state real-GPU 1.1% miss, mark as Blocked.
2. Update root docs/plans/STATUS.md Plan-0017 row with same information.
3. Bump Last updated line in both files.
4. In crates/xagent-sandbox/tests/integration.rs, rename the budget assertion `cortex_throughput_meets_budget` (≈line 703) to `cortex_throughput_meets_budget_on_software_adapter` so the name reflects its lavapipe-only scope (it self-skips on real GPU); update its doc comment to note real GPU sits at ~1.1% of baseline (18/256 lane occupancy), recorded in CORTEX-PROFILE-BASELINE.txt. This is a symbol-only rename — the assertion body and threshold are unchanged. (Note: this is a different symbol from `cortex_throughput_profile_baseline` at ≈658, which task `gate-diagnostic-tables` gates and which is NOT renamed here.)

- **Depends on:** —
- **Done when:** 0017 STATUS WS0002 and roll-up both state real-GPU 1.1% budget miss and lavapipe-only scope of 50% assertion; the budget test is renamed `cortex_throughput_meets_budget_on_software_adapter` (symbol-only, assertion and threshold unchanged) so its name no longer reads as a real-hardware claim. cargo fmt/clippy/test green.

---

## 0003 — Innate-Instinct-Harness-Regression-Test

### innate-instinct-regression-test — Add Regression Test for 0013 Innate-Instinct A/B Harness

Plan 0013 harness run_innate_instinct_ab ran once on Metal, produced FAILED verdict (decision doc). No integration test; harness is orphaned. If learning signal changes, gate cannot re-run mechanically to verify failure still holds.

**Steps:**
1. Open crates/xagent-sandbox/tests/integration.rs.
2. Add test run_innate_instinct_ab_produces_valid_stats that calls harness with tiny generation count (2 gens, N=5), asserts ValidationStats fields are finite and non-negative, includes GPU self-skip guard.
3. Confirm test passes with harness logic exercised.

- **Depends on:** cortex-status-honesty
- **Done when:** Test run_innate_instinct_ab_produces_valid_stats exists, invokes harness with tiny generation count, asserts well-formed ValidationStats (not gate pass), includes GPU self-skip guard. cargo fmt/clippy/test green.

---

## 0004 — Auxiliary-Loss-Mechanism-Honesty

### auxiliary-loss-decision-update — Re-Frame 0018-0001 Auxiliary Loss as Not-Tested-at-GPU-Level

0018-0001-AUXILIARY-LOSS-DECISION.md correctly notes CPU-side overlay (no GPU gradient injection); test shows 0.509 alignment (chance). Decision REJECT is correct for CPU-overlay scope. However, 3/3 REJECT framing can mislead readers into citing this as falsification of auxiliary loss mechanism, when truthful statement is CPU overlay (not GPU-integrated mechanism) was tested.

**Steps:**
1. Open docs/plans/0018-Credit-Path-Mechanism-Attack/0001-AUXILIARY-LOSS-DECISION.md.
2. Update Summary to explicitly state REJECT applies to CPU-side measurement overlay only.
3. Add Scope-of-This-Rejection section clarifying mechanism not tested at GPU-integrated level.
4. Update Next Step to reference GPU integration as future work.
5. If SCOPE.md lists this workstream, update language from REJECT to DEFERRED-pending-GPU-integration.

- **Depends on:** —
- **Done when:** Decision doc clearly states REJECT applies to CPU-side overlay, not GPU-integrated auxiliary loss. Mechanism noted as not-tested-at-relevant-level, not falsified. Documentation-only, no code changes.

---

## 0005 — Frame-Sync-Spike-Clarity

### frame-sync-decision-update — Re-Frame 0018-0002 Frame-Sync as Not-Tested-Under-Activation-Condition

0018-0002-TRACE-HORIZON-DECISION.md notes steering probe trained at vision_stride=1 (mechanism disabled); unit test confirms mechanism active at vision_stride=10. Yet 3/3 REJECT framing suggests mechanism was falsified when it was not tested under its activation condition.

**Steps:**
1. Open docs/plans/0018-Credit-Path-Mechanism-Attack/0002-TRACE-HORIZON-DECISION.md.
2. Update Result section to clarify REJECT applies to dense-stride probe, not frame-sync mechanism itself.
3. Add Scope-of-This-Result section stating mechanism not tested under activation condition.
4. Add When-to-Revisit section for sparse-stride evaluation.

- **Depends on:** —
- **Done when:** Decision doc clarifies REJECT applies to dense-stride probe only; mechanism not-tested-under-activation-condition, not falsified. Documentation-only, no code changes.

---

## 0006 — Diagnostic-Table-Silencing

### gate-diagnostic-tables — Gate Diagnostic Tables Behind XAGENT_VERBOSE_PROBES

Cortex and credit probes print large diagnostic tables. 2026-06-19 grok-43 recommended gating behind env var for silent, falsifying-only tests by default. Applied to fitness probes; not to 0017/0018 probes.

**Steps:**
1. Open crates/xagent-sandbox/tests/integration.rs.
2. Locate cortex_throughput_profile_baseline (line 658), baseline_td_error_variance_during_foraging (line 8705), gradient_variance_per_context_breakdown (line 9012), auxiliary_steering_loss_converges_on_bearing (line 8806).
3. Wrap diagnostic eprintln calls with: if std::env::var(XAGENT_VERBOSE_PROBES).is_ok() { eprintln!(...); }
4. Verify tests pass without output by default, print tables when XAGENT_VERBOSE_PROBES=1.

- **Depends on:** cortex-status-honesty, innate-instinct-regression-test
- **Done when:** Diagnostic eprintln tables gated behind XAGENT_VERBOSE_PROBES. Tests silent by default, tables available on demand. Assertions unchanged, still falsifiable. cargo fmt/clippy/test green.

---

## 0007 — Test-File-Path-Traceability

### test-path-traceability — Audit and Update Test File Paths in Plan STATUS Docs

Plan STATUS.md docs reference test deliverables (`crates/xagent-brain/tests/intent_baseline_measurement.rs`, `crates/xagent-brain/tests/danger_percept_ablation_ab.rs`) without crate prefix. They live in xagent-brain/tests but can be misread as xagent-sandbox/tests. Hampers traceability when project has two test crates.

**Steps:**
1. Open docs/plans/0015-Intent-Awareness-Measurement-Framework/STATUS.md and docs/plans/0016-Intent-As-Meta-Score-Evolved-Policy-Ablation/STATUS.md.
2. Update all test file refs to use full crate-relative paths (e.g., crates/xagent-brain/tests/intent_baseline_measurement.rs).
3. Check ARCHITECTURE.md files for same plans; update paths there too.
4. Review other plans for same pattern; apply consistently.

- **Depends on:** —
- **Done when:** All test file refs in plan STATUS and ARCHITECTURE docs use full crate-relative paths. No ambiguity on which crate owns which test. Documentation-only change.

---

**End of plan 0019 TASKS.** When every "Done when" bullet is green, the plan's end state is reached.
