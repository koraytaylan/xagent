# Plan 0019 — Gate-Hygiene-Credit-Path-Honesty — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete. All 7 tasks landed: fmt gate restored on `ui.rs`;
cortex-throughput STATUS corrected to state the real-GPU miss (~1.1% of
baseline) with the budget test renamed to its lavapipe-only scope; the 0013
innate-instinct A/B harness is now regression-tested; the 0018 auxiliary-loss
and frame-sync spikes are re-framed as rejected-only-at-the-tested-level (not
mechanism-falsified); diagnostic probe tables are gated behind
`XAGENT_VERBOSE_PROBES`; and brain-crate test paths are made crate-relative
across plans 0012/0015/0016/0019/0021. Gates green (fmt/clippy/check/test).
_Last updated: 2026-06-26, against `develop`._

- **Goal:** Develop branch has cargo fmt/clippy/test green; cortex throughput
  STATUS accurately reports real-GPU miss; 0013 A/B harness is regression-tested;
  spike mechanisms re-framed as not-tested-at-relevant-level where applicable.
- **Root cause:** Mechanical gate-integrity defects (fmt red on develop, cortex
  budget skip on GPU) and falsification re-framing concerns (orphaned harness,
  spike tested at wrong level) landed without process catching them.
- **Approach:** Fix fmt error, correct cortex STATUS to state real-GPU miss, add
  A/B harness regression test, gate diagnostic tables behind XAGENT_VERBOSE_PROBES,
  re-frame spikes honestly.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Fmt-Gate-Restore | `fmt-ui-texture-key` | ✅ Done |
| 0002 | Cortex-Throughput-Honesty | `cortex-status-honesty` | ✅ Done |
| 0003 | Innate-Instinct-Harness-Regression-Test | `innate-instinct-regression-test` | ✅ Done |
| 0004 | Auxiliary-Loss-Mechanism-Honesty | `auxiliary-loss-decision-update` | ✅ Done |
| 0005 | Frame-Sync-Spike-Clarity | `frame-sync-decision-update` | ✅ Done |
| 0006 | Diagnostic-Table-Silencing | `gate-diagnostic-tables` | ✅ Done |
| 0007 | Test-File-Path-Traceability | `test-path-traceability` | ✅ Done |
