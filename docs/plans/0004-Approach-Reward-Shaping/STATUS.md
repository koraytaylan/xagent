# Plan 0004 — Approach Reward Shaping · status

Task-level execution status for this plan. Keep it current as tasks land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 📋 Planned · 0/14 tasks · authored against `develop` @ `1f1b421`.
_Last updated: 2026-06-14, against `develop`._

- **Goal:** Give the within-lifetime reward a spatial approach gradient
  (potential-based shaping `F = γΦ(s′) − Φ(s)` into `raw_gradient`) so
  vision-conditional steering becomes learnable — the genuine delivery of Plan
  0001's stated goal — then restore the selection signal and gate the reactive,
  heritable, and perceptual follow-ups behind the one remeasure that proves the
  unlock. Compiles `docs/reviews/2026-06-14-claude-opus-48.md`.
- **Outcome:** _Pending._ Expected: turn/bearing alignment moves decisively above
  the `0.62` chance band and free-run foraging rises; selection regains dynamic
  range and an above-noise accept rule; or, if alignment stays at chance after
  shaping, the encoder/representation is recorded as the next suspect and the
  gated workstreams stay closed.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Approach reward shaping | `approach-shaping-baseline`, `visible-food-potential-input`, `potential-based-reward-shaping`, `actor-vector-scale`, `approach-shaping-remeasure` | 📋 |
| 0002 | Reactive & valence layers on the external gradient | `klinotaxis-external-gradient` (GATED), `memory-valence-food-in-view` (GATED) | 📋 gated on `0001` remeasure |
| 0003 | Selection signal restoration | `foraging-primary-fitness`, `decoupled-experiment-resolution`, `selection-significance-guard`, `governor-within-life-metric` | 📋 |
| 0004 | Heritable learning dynamics | `policy-constants-to-genes` (GATED) | 📋 gated on `0001` remeasure |
| 0005 | Sensory lag & vision geometry | `heritable-stride-revisit` (GATED), `vision-row-geometry` (GATED) | 📋 gated on `0001` remeasure |
