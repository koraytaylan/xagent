# Plan 0012 — Homeostatic-Only Learning Restoration — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 📋 Planned.
_Last updated: 2026-06-19, against `claude/nice-liskov-fe972d`._

- **Goal:** Both PBRS terms removed; raw_gradient now purely homeostatic
  (energy_delta*ENERGY_WEIGHT + integrity_delta*INTEGRITY_WEIGHT); code-to-README
  alignment restored; falsifiable test added as regression guard; gates green. The
  removed approach term was already measured-ineffective (mirrored-steering held at
  chance), so the default learning signal is simplified without an expected
  observable-behavior change.
- **Root cause:** Plan 0004 (Approach PBRS) and Plan 0009 (Avoidance PBRS)
  introduced hand-engineered reward shaping into the learning signal, contradicting
  the README's stated homeostasis-only design. Plan 0004's own remeasure falsified
  the approach-shaping unlock (mirrored-steering at chance), yet the mechanism
  remained in code. Plan 0009 added avoidance shaping behind a flag (default false),
  but the code is mechanically identical to reward-injection. Both persist in
  production, making the codebase aspirational on homeostasis but implementational on
  reward-shaping — a philosophical incoherence the 2026-06-18 reviews (Grok 4.3,
  Gemini 3.1 Pro) flagged, and which the `2026-06-19-claude-opus-48.md` review
  independently confirmed as M1 ("the deepest purity breach on the floor" — the
  approach term is unconditionally folded into the brain's TD reward).
- **Approach:** Remove both PBRS terms from the raw_gradient assembly (line 823–826
  of brain_passes.wgsl), delete the supporting consts and buffer slots, add a
  falsifiable regression-guard test asserting raw_gradient is homeostatic-only at
  default config, and document the restored alignment in README. Single-phase change:
  no gated follow-up (design commits to homeostasis-only).

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Measurement-Baseline | `measure-baseline-pre-removal` | 📋 Planned |
| 0002 | Approach-Shaping-Removal | `remove-approach-shaping` | 📋 Planned |
| 0003 | Avoidance-Shaping-Removal | `remove-danger-shaping` | 📋 Planned |
| 0004 | Dead-Code-Cleanup | `remove-orphaned-consts` | 📋 Planned |
| 0005 | Parity-And-Tests | `verify-gradient-parity`, `add-homeostatic-only-gate` | 📋 Planned |
| 0006 | Documentation-Update | `update-readme-vision` | 📋 Planned |
