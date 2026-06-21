# Plan 0013 — Innate Survival Instincts — status

Task-level execution status for this plan lives here. Keep it current as tasks
land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 📋 Planned.
_Last updated: 2026-06-19, against `claude/nice-liskov-fe972d`._

- **Goal:** Seed instinct priors (danger→avoidance, food→approach) into pattern
  memory at brain birth, heritable via evolved strength genes, gated behind
  `innate_instincts_enabled` (default off), and validated via headless A/B
  benchmark (survive, steer, eat metrics). Pass/fail is recorded in a decision
  doc; the flag stays off until the gate passes.
- **Root cause:** Blank-slate pattern memory (`init_pattern_memory` all zeros)
  requires tight credit paths through recall and valence reinforcement to learn
  basic avoidance/foraging, making early survival fragile. Seeded one-time
  evolved priors (danger signature + strong negative valence, food signature +
  strong positive valence) bypass the initial exploration phase and subject
  themselves to normal decay/eviction, providing a biosimilar alternative to the
  standing external reward shaping removed in Plan 0012.
- **Approach:** Workstreams ordered by dependency — design the seeded patterns
  and add heritable genes (0001), thread genes through breeding (0002), gate
  behind the default-off flag (0003), implement A/B validation and the gated
  decision (0004). All pre-gate tasks are independent code changes; the final
  decision task is gated on prior completion. No behavioral change when the flag
  is off (byte-identical).

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Innate-Pattern-Seeding | `add-instinct-config-fields`, `implement-seed-instinct-patterns` | 📋 Planned |
| 0002 | Heritable-Instinct-Config | `add-instinct-mutation-to-breeding` | 📋 Planned |
| 0003 | Default-Off-Gating | `add-instinct-gate-flag`, `integrate-seeding-into-reset-path` | 📋 Planned |
| 0004 | Prove-Or-Kill-Gate | `implement-ab-validation-harness`, `author-ab-gate-decision-doc` | 📋 Planned |
