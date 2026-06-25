# Plan 0019 — Gate-Hygiene-Credit-Path-Honesty — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 📋 Planned.
_Last updated: 2026-06-25, against `develop`._

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
| 0001 | Fmt-Gate-Restore | `fmt-ui-texture-key` | 📋 Planned |
| 0002 | Cortex-Throughput-Honesty | `cortex-status-honesty` | 📋 Planned |
| 0003 | Innate-Instinct-Harness-Regression-Test | `innate-instinct-regression-test` | 📋 Planned |
| 0004 | Auxiliary-Loss-Mechanism-Honesty | `auxiliary-loss-decision-update` | 📋 Planned |
| 0005 | Frame-Sync-Spike-Clarity | `frame-sync-decision-update` | 📋 Planned |
| 0006 | Diagnostic-Table-Silencing | `gate-diagnostic-tables` | 📋 Planned |
| 0007 | Test-File-Path-Traceability | `test-path-traceability` | 📋 Planned |
