# Plan 0013 — Innate Survival Instincts — status

Task-level execution status for this plan lives here. Keep it current as tasks
land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete (prove-or-kill: **negative** — instincts rejected, flag stays default-off).
_Last updated: 2026-06-22, against `develop`._

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
| 0001 | Innate-Pattern-Seeding | `add-instinct-config-fields`, `implement-seed-instinct-patterns` | ✅ Done |
| 0002 | Heritable-Instinct-Config | `add-instinct-mutation-to-breeding` | ✅ Done |
| 0003 | Default-Off-Gating | `add-instinct-gate-flag`, `integrate-seeding-into-reset-path` | ✅ Done |
| 0004 | Prove-Or-Kill-Gate | `implement-ab-validation-harness`, `author-ab-gate-decision-doc` | ✅ Done (gate **FAILED**) |

## Prove-or-kill outcome

The A/B benchmark was run on the local Metal adapter (10 generations, population
10) and **failed all three gates** — see
[`0013-INNATE-INSTINCT-DECISION.md`](0013-INNATE-INSTINCT-DECISION.md):

- **Survival** +0.19% (baseline 995074 → ON 996975; needs +10%); ON trailed
  baseline in early generations.
- **Alignment** 0.000 (needs ≥0.4) — but the benchmark ran with
  `danger_percept=false`, so the danger instinct is never sensed; the primary
  revisit condition is to re-run with danger perception enabled.
- **Food-per-death** 0.37 (needs ≥2.0) — the food instinct (food perception is on,
  so this gate had a fair test) gave no foraging benefit.

Decision: **reject this iteration**, `innate_instincts_enabled` stays default-off.
The seeding code ships complete behind the flag. Revisit conditions (fair danger
test, early-learning interference, alternative signatures) are in the decision doc.
