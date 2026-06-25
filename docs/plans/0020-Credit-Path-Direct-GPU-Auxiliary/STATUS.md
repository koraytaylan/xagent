# Plan 0020 — Credit-Path-Direct-GPU-Auxiliary — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 📋 Planned.
_Last updated: 2026-06-25, against `develop`._

- **Goal:** GPU-integrated auxiliary loss is either accepted (steering >= 0.70,
  integrated, baseline updated) or rejected (steering <= 0.62, structural rethink
  documented with candidates for Plan 0021).
- **Root cause:** The CPU overlay test (0018-0001) measured loss decay on GPU
  outputs without injecting GPU weight updates, so it falsified the measurement
  method not the mechanism. The true test requires GPU-side gradient injection into
  action weights under bearing-alignment supervision.
- **Approach:** Spike-prototype-decide-integrate: implement bearing-aligned
  auxiliary loss on GPU (0001), measure steering probe with loss active (0002),
  render binary decision by 95% CI (accept >= 0.70 or reject <= 0.62), integrate if
  accepted or record structural-rethink fallback if rejected (0003).

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | GPU-Auxiliary-Loss-Implementation | `implement-gpu-auxiliary-loss` | 📋 Planned |
| 0002 | GPU-Auxiliary-Steering-Probe | `gpu-auxiliary-steering-probe`, `auxiliary-loss-decision-and-integration-gate`, `auxiliary-loss-integrate-and-update-baseline` | 📋 Planned |
| 0003 | Structural-Rethink-Fallback-Decision | `structural-rethink-fallback` | 📋 Planned |
