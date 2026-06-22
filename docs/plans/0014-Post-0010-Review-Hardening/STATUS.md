# Plan 0014 — Post-0010 Review Hardening — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete.
_Last updated: 2026-06-22, against `develop`._

- **Goal:** Fix the three confirmed correctness/validity defects the four
  `2026-06-19` due-diligence reviews surfaced that the reward-model re-think (0012,
  0013) does not cover — the effort-fitness camper inversion, CI silently skipping
  the GPU test core, and the 8× terminal-death TD divergence. None flips a default.
- **Root cause:** (1) `composite_fitness()` effort mode keeps `ticks_alive` in the
  numerator of both axes (`food·ticks/energy`, `cells·ticks/distance`), so a camper
  is the fittest archetype (Camper 0.8646 > Competent 0.3209) and no test asserts
  `competent > camper` — a measured kill of 0010's keystone, pinned as expected
  (opus H3/M5, gpt-5-codex F1). (2) `ci.yml`/`release.yml` install no Vulkan driver,
  so `GpuKernel::is_available()` is false on CI and ~63 GPU-gated tests self-skip
  green — "green on CI" is not evidence of GPU correctness (opus H4). (3)
  `phase_death.wgsl` scales the actor terminal lesson by `TD_VECTOR_SCALE` while the
  fused `kernel_tick.wgsl` uses `ACTOR_VECTOR_SCALE` (8× divergence), live via the
  physics-remainder dispatch and unguarded because parity tests never kill an agent
  (opus M4).
- **Approach:** Denominate both effort axes on cumulative totals (drop the `ticks`
  factor), re-derive the targets, and add `competent > camper` + duration-invariance
  guards (effort flag stays default-off). Install Mesa lavapipe in CI + a required-
  adapter gate so CI runs the GPU suite and cannot silently all-skip. Bring
  `phase_death.wgsl`'s actor scale into agreement with the fused path and add a
  death-crossing fused/split parity test. The deeper fitness-philosophy decision
  (fix-and-keep vs. retire for environmental pressure) is deferred to a future plan.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Effort-Fitness-Camper-Fix | `fix-foraging-duration-leak`, `add-anti-camper-assertion` | ✅ Done |
| 0002 | CI-GPU-Test-Execution | `install-lavapipe-in-ci`, `add-gpu-test-runcount-gate` | ✅ Done |
| 0003 | Death-Path-TD-Parity | `fix-phase-death-actor-scale`, `add-death-crossing-parity-test` | ✅ Done |

## Execution notes

The parallel run landed 3/6 tasks; the rest were recovered by hand (one stalled,
one blocked, one redundant). Resolutions:

- **`add-anti-camper-assertion`** stalled because its `competent > camper`
  assertion was **unsatisfiable as authored**: removing the `ticks` factor did not
  invert the ranking — the old camper fixture (`food=250/energy=60`) is a
  hyper-efficient exploiter with a *higher* food/energy ratio than the competent
  forager, so both cap the foraging axis and the camper wins on zero deaths. Per
  an explicit decision, the calibration camper was redefined as a true **idle**
  agent (near-zero food), matching the plan's own description; it now scores
  **0.1504 ≪ 0.7034**. Added the `competent > camper` guard, a foraging negative
  control, and `effort_fitness_is_duration_invariant` (with a legacy-mode control).
  See [`0014-CAMPER-FITNESS-DECISION.md`](0014-CAMPER-FITNESS-DECISION.md).
- **`add-gpu-test-runcount-gate`** was implemented by the `install-lavapipe-in-ci`
  developer (the `gpu_adapter_present_when_required` guard + `XAGENT_REQUIRE_GPU`
  env in both workflows already landed on the plan branch), so its standalone merge
  was a redundant no-op. Deliverables verified present.
- **`add-death-crossing-parity-test`** could not be the authored fused-vs-split
  byte comparison: both execution modes drive the physics-only remainder through
  the same `physics_pipeline` (`phase_physics` + `phase_death`) and full cycles
  through the same fused `kernel_tick.wgsl`, so they never diverge on death. The
  real M4 divergence is full-cycle-death vs remainder-death (shared by both modes).
  Replaced with a single-run scale-recovery test that forces a death through the
  physics-remainder path and recovers `lr·scale` from the terminal update; teeth
  verified (recovers `0.000781` on the TD-bug, `0.006250` with the fix).
