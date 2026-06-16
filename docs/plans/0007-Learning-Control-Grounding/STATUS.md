# Plan 0007 - Learning Control Grounding - status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete. All 12 tasks implemented and reviewed task-by-task via
the `implement-plan` workflow and squash-merged to `develop` (`c40dcb7`). A
post-merge full-workspace gate found one functional defect that the per-task
gates missed (the navigation danger flag was published inverted; see
**Post-merge correction** below); it is fixed and all gates are green.
_Last updated: 2026-06-16, against `develop`._

- **Goal:** Make the live GPU runtime evaluate the genes persisted in
  `xagent.db`, add behavior evidence that can prove or falsify food chasing and
  danger avoidance, slow the perception-action loop through a learning
  curriculum, repair multiplier-only klinotaxis, and require red-green probes
  before interpreting a long evolution run.
- **Measured baseline:** `xagent.db` has 1 run, 251 nodes, 2500 agent results,
  250 recordings; status distribution is 249 failed descendants, 1 exhausted
  root, 1 active node. Root score remains best (`0.0042712856`). Late
  generations 225-249 average `0.0014031432` fitness, 262.096 food, 324.42
  deaths, and 174.592 cells. Decoded recordings show mean straightness
  `0.007105`, mean absolute turn `0.375534`, turn-bias ratio `0.851702`, and
  turn-sign persistence `0.940761`; mean absolute turn correlates with deaths
  (`r=0.9107`) and against fitness (`r=-0.6596`).
- **Outcome:** Runtime genome authority, behavior-evidence telemetry (recording
  v2, nav slots, `behavior_metric` table), the slow learning curriculum, the
  lowered `[4.0, 30.0]` speed range, authoritative turn persistence, and
  sign-breaking klinotaxis all landed with their tests green. The food-closure,
  danger-exit, and anti-circle probes pass as red-green control/telemetry gates
  under the curriculum. The 20-generation evolution comparison is an offline run
  not executed in CI — see [`DECISION-short-evolution-gate.md`](DECISION-short-evolution-gate.md).
  Full workspace gate green: `cargo fmt`/`clippy` clean; `cargo test --workspace`
  232 tests pass (51 brain + 87 sandbox-lib + 13 bin + 75 integration + 6 shared).

## Post-merge correction

The independent per-task gates run only the changed crate's lib unit tests and
the reviewer's GPU tests self-skip without an adapter, so two issues survived to
`develop` and were caught by the repo-root full-workspace gate the README
mandates after a plan lands:

1. **Stale incremental build (not a source defect).** A first full-workspace run
   reported a compile error and three failing GPU telemetry tests, all traced to
   a stale `xagent-brain` rlib (its `include_str!`'d shaders were baked from an
   intermediate merge state). A clean rebuild (`cargo clean -p xagent-brain`)
   compiled the squashed source correctly.
2. **Danger flag published inverted (fixed).** `agent_physics`
   (`kernel_tick.wgsl`) wrote `P_IN_DANGER_BIOME = 0.0` while *in* a danger biome
   and `1.0` while safe — the opposite of what its consumers expect
   (`danger_exit_probe` reads `>0.5` as "in danger"; CPU `biome_at` and GPU
   `sample_biome` agree on `Danger==2`). The inverted write let `danger_exit_probe`
   pass vacuously (it saw the agent as never in danger). It almost certainly
   slipped in to make the two telemetry tests pass while they read the flag after
   the agent had wandered to an unknown position. Fix: publish `1.0` in danger /
   `0.0` otherwise, and rewrite `danger_biome_flag_marks_hazardous_locations` /
   `safe_biome_flag_marks_safe_locations` to assert the flag against the biome at
   the agent's *actual readback position* (deterministic, robust to drift, and
   red-green for the inversion). With the corrected flag `danger_exit_probe` now
   measures a real dwell fraction (0.7) instead of a vacuous ~0.3.

Known minor follow-ups (not blocking, logged): the legacy split-pass
(`phase_physics.wgsl`) does not publish the nav telemetry slots (fused is the
default/production path); `behavior_metric.danger_dwell_fraction`/
`food_distance_delta` column semantics are looser than their names imply (review
nits); `danger_exit_probe` thresholds are explicit placeholders pending the
offline evolution run.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Runtime genome authority | `effective-agent-config-upload`, `mutation-provenance-for-effective-genes` | ✅ Done |
| 0002 | Behavioral evidence telemetry | `recording-format-v2`, `navigation-telemetry-slots`, `behavior-metric-table` | ✅ Done (danger flag corrected post-merge) |
| 0003 | Control-rate curriculum | `learning-curriculum-preset`, `movement-speed-range-revisit` | ✅ Done |
| 0004 | Turn-attractor and klinotaxis repair | `authoritative-turn-persistence`, `sign-breaking-klinotaxis` | ✅ Done |
| 0005 | Food/danger emergence gates | `food-closure-probe`, `danger-exit-probe`, `short-evolution-gate` | ✅ Probes done; evolution gate deferred to offline run (decision note) |

## Verification

Full workspace gate, re-run from the repo root after the squash + post-merge fix:

```
cargo fmt --all -- --check                              # clean
cargo clippy --workspace --all-targets -- -D warnings   # clean
cargo test --workspace --no-fail-fast                   # 232 passed, 0 failed
```

GPU tests executed locally against a real adapter (did not self-skip).
