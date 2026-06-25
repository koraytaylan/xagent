# Plan 0022 — Cortex-Workgroup-GPU-Occupancy — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 📋 Planned.
_Last updated: 2026-06-25, against `develop`._

- **Goal:** Cortex occupancy restructuring succeeds (≥50% on GPU hardware) and
  opens the path for emergence encoder A/B, OR occupancy ceiling is measured below
  50% and cortex is explicitly locked behind `visual_cortex_enabled=false` with
  emergence pivot recorded.
- **Root cause:** The cortex dispatch uses a 256-lane workgroup with only 18 active
  threads during the Gabor/pooling stages, producing ~7% GPU lane occupancy on
  hardware and ~1.1% throughput despite optimization. This is a structural mismatch
  (feature count vs workgroup size), not a tuning problem. The occupancy ceiling
  blocks any cortex-on default and forces an explicit decision: restructure to
  increase occupancy or accept permanent GPU-gating.
- **Approach:** Measure current occupancy as a baseline, explore
  occupancy-restructuring variants (feature reduction, multi-agent workgroup),
  select the viable path or escalate to fallback, implement the selected variant,
  re-validate on real GPU hardware, and record the decision (promotion to default,
  or permanence + emergence pivot).

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Workgroup-Occupancy-Restructuring | `occupancy-baseline-measurement`, `variant-a-feature-count-reduction`, `variant-b-multi-agent-workgroup`, `occupancy-restructuring-selection`, `variant-full-implementation` | 📋 Planned |
| 0002 | Occupancy-Improvement-Validation | `cortex-throughput-validation`, `cortex-encoder-path-decision` | 📋 Planned |
| 0003 | Cortex-Permanence-Fallback-Decision | `cortex-permanence-decision-fallback` | 📋 Planned |
