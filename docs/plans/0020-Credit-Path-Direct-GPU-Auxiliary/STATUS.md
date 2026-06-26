# Plan 0020 — Credit-Path-Direct-GPU-Auxiliary — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete — mechanism WORKS, deliberately NOT integrated (homeostasis-only).
_Last updated: 2026-06-26, against `develop`._

- **Outcome:** Sign-corrected GPU auxiliary loss raises steering to **0.841**
  (95% CI [0.816, 0.867]) vs 0.489 chance baseline — clears the numeric ACCEPT gate
  (CI lower 0.816 ≥ 0.70). It is **not integrated**: direct turn→food-bearing
  supervision is approach-shaping that violates the homeostasis-only / food-bearing-blind
  contract. Following the 0017/0018 precedent, **the prototype code was reverted from
  `develop`** (flag, WC slot, WGSL loss block, and both probe tests removed); this
  decision doc + git history are the record. See
  [`0001-GPU-AUXILIARY-LOSS-DECISION.md`](0001-GPU-AUXILIARY-LOSS-DECISION.md).
- **Sign bug found post-merge:** the first implementation supervised `turn_output →
  +food_bearing/PI`, but `P_NEAREST_FOOD_BEARING = yaw − food_heading` and positive
  `motor_turn` increases yaw, so the correct target is `−food_bearing/PI`. The
  inverted target trained anti-steering (0.114). Negating it → 0.841. The original
  workflow recorded a REJECT against the buggy measurement; this corrects the record.
- **What this localizes:** not the encoder (separability > 0.9), not signal magnitude
  (0018-0003 amplified |δ| ~400× to no effect) — the bottleneck is **credit timing/
  alignment** in the self-supervised path across the ~10-tick sensory latency. Direct
  supervision proves the routing is learnable in principle.
- **Carry-forward (future credit-path plan):** achieve the same vision→turn routing
  under homeostatic pressure alone (no privileged food-direction target). 0.841 is the
  upper-bound reference candidates must approach without re-introducing approach-shaping.
  (0021/0022 are already allocated to other work — a new plan must be authored.)

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | GPU-Auxiliary-Loss-Implementation | `implement-gpu-auxiliary-loss` | ✅ Done (sign-corrected, then reverted) |
| 0002 | GPU-Auxiliary-Steering-Probe | `gpu-auxiliary-steering-probe`, `auxiliary-loss-decision-and-integration-gate` | ✅ Done (probe reverted, decision recorded) |
| 0002 | (gated, not run) | `auxiliary-loss-integrate-and-update-baseline` | ⛔ Not run — ACCEPT path declined on homeostasis grounds |
| 0003 | Structural-Rethink-Fallback-Decision | `structural-rethink-fallback` | ⛔ Not run — premise (mechanism failed) is false; superseded by corrected decision doc |
