# Plan 0009 - Intent-Aware Fitness - status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete. 17/18 tasks landed; `default-flip-gate` deliberately held (GATED — awaits measured speed-decoupling confirmation). All landed tasks squash-merged into `claude/sweet-golick-78e294` as `b314da1`.
_Last updated: 2026-06-18, against `claude/sweet-golick-78e294`._

- **Goal:** Make evolutionary fitness reward *deliberate* foraging and
  *deliberate* danger avoidance instead of the accidental by-products of raw
  speed. Re-base every time-denominated score axis onto effort, make hazard
  exposure speed-invariant (graded, never lethal), make locomotion pay for itself
  above baseline, and give the agent a dedicated danger percept so avoidance is a
  learnable, measurable decision. All behind flags, default no-op, graduated on a
  measured speed-decoupling.
- **Measured baseline (current code):** `composite_fitness` (`governor.rs:84-102`)
  is `food_per_1k = food/(ticks_alive/1000)`, absolute `cells_explored`, and a
  `death_count` survival multiplier — every axis per-time. `movement_speed` is a
  heritable gene (`config.rs:120`, `[1.0,100.0]`, default 20). Movement energy is
  linear in speed (`kernel_tick.wgsl:163`) so cost-per-distance is flat; hazard is
  per-tick (`:176-178`) so a fast crossing dodges damage. Danger is visible only
  as an entangled dark-red terrain color (`phase_vision.wgsl:149-151`) plus the
  post-contact `TOUCH_HAZARD` flag. `behavior_metric.danger_dwell_fraction` is
  declared (`governor.rs:1725`) but unpopulated. Two physics paths
  (`kernel_tick.wgsl` fused + `phase_physics.wgsl`/`phase_death.wgsl` split) are
  both compiled, each with a respawn save/restore whitelist.
- **Root cause:** one dimensional flaw with three symptoms — every fitness axis is
  denominated in TIME and `movement_speed` is the only gene that buys time, so a
  faster mutant inflates foraging rate, cell coverage, and survival
  simultaneously, with no super-linear cost to push back.
- **Approach:** three reinforcing layers + a percept upgrade — (A) super-linear
  locomotor energetics (keystone), (B) dwell-invariant path-length hazard, (C)
  effort-rebased fitness (food/energy + cells/distance), (D) danger percept +
  symmetric avoidance potential + sensed-then-turned intent metric — measured and
  default-flipped by a headless speed-decoupling gate.
- **Outcome:** 17/18 tasks landed on `implement-plan/0009` and squash-merged into `claude/sweet-golick-78e294` (`b314da1`). All four layers implemented and behind flags: (A) super-linear locomotor energetics, (B) dwell-invariant path-length hazard, (C) effort-rebased fitness, (D) danger percept + avoidance potential + intent metric. Speed-decoupling validation run (`speed-decoupling-validation`) completed. `default-flip-gate` deliberately held — gated on the measured decoupling numbers before flipping defaults.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Effort & exposure telemetry | `phys-accumulator-slots`, `effort-telemetry-fused`, `effort-telemetry-split`, `effort-telemetry-readback`, `populate-danger-dwell-metric` | ✅ Done |
| 0002 | Dwell-invariant hazard (Layer B) | `path-length-hazard-fused`, `path-length-hazard-split` | ✅ Done |
| 0003 | Super-linear locomotor energetics (Layer A) | `speed-cost-exponent-config`, `super-linear-drag-fused`, `super-linear-drag-split` | ✅ Done |
| 0004 | Effort-rebased fitness (Layer C) | `composite-fitness-effort-rebase`, `fitness-calibration-replay` | ✅ Done |
| 0005 | Danger percept + avoidance + intent metric (Layer D) | `nearest-danger-telemetry`, `danger-percept-sense`, `danger-avoidance-potential`, `avoidance-intent-metric` | ✅ Done |
| 0006 | Validation & default-flip gate | `speed-decoupling-validation` ✅ Done, `default-flip-gate` ⛔ Gated (not run) | 🚧 In progress |

## Verification

_Pending._ The plan's success criterion is a measured speed-decoupling
(`0006`): the correlation between evolved `movement_speed` and `composite_fitness`
falls from strongly-positive to ≈0 / single-peaked (peak well below the clamp),
mean `ticks_alive` does not collapse, and `danger_dwell_fraction` /
`avoidance_intent_fraction` stay non-zero (danger-decision data still collected).
The default-flip is gated on those numbers.
