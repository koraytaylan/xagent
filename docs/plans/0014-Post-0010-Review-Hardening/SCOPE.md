# Scope — Plan 0014

> Fix the three confirmed correctness / scientific-validity defects the
> `2026-06-19` due-diligence reviews surfaced that the reward-model re-think
> (Plans 0012, 0013) does **not** cover: the effort-fitness camper inversion, CI
> silently skipping the GPU test core, and the 8× terminal-death TD divergence
> between the fused and split kernels. All three are correctness/validity fixes,
> not feature work; none flips a default.

## Why this plan

Four independent `2026-06-19` forward-looking reviews (claude-opus-48, grok-43,
gpt-5-codex, gemini-31-pro-high) audited `HEAD` after Plans 0010/0011. They
agreed the engineering is strong and the build green, but surfaced three concrete
defects that are **not** about the within-life reward model (Plan 0012) or innate
priors (Plan 0013). Each was re-verified against the code before inclusion here.

1. **The effort-rebased fitness keystone is camper-dominated — its headline
   "scale-invariant" claim is mathematically false.** `composite_fitness()` in
   effort mode computes `foraging = (food_consumed / (energy_spent / ticks_alive))
   / FORAGING_ENERGY_TARGET` (`governor.rs:135-151`), i.e. `food·ticks /
   (energy·250000)` — `ticks_alive` is in the **numerator**, so a long-lived,
   low-burn camper saturates the axis. The same `ticks` factor sits in the
   exploration axis (`cells_per_distance_rate = cells / (distance/ticks)`). The
   calibration replay pins the inversion as *expected*: Camper effort **0.8646** >
   Competent **0.3209** (run live by `2026-06-19-gpt-5-codex.md` F1 and
   `2026-06-19-claude-opus-48.md` H3/M5), and **no test asserts `competent >
   camper`** — `fitness_calibration_replay_profiles` asserts only `competent >
   aimless` and `camper < 1.0`, and even pins `camper ≈ 0.8646`. This is the exact
   camping inversion the effort rebasing was invented to defeat ("food-per-energy
   to defeat camping"). Default-off, so live blast radius is zero — but under the
   project's own *prove-or-kill* discipline it is a **measured kill of the headline
   deliverable**, encoded as a passing test.

2. **CI has been silently skipping the GPU test core.** `CLAUDE.md` claims
   "CI/dev installs Mesa lavapipe," but `.github/workflows/ci.yml` runs
   `cargo test -p xagent-sandbox` on bare `ubuntu-latest` with **no** lavapipe /
   mesa / vulkan install step, so `GpuKernel::is_available()` returns false and the
   ~63 GPU-gated integration tests `eprintln!("Skipping…"); return;` and pass green
   without executing the kernel — including the falsifiable steering probe, every
   `*_byte_identical_when_flag_off` no-op proof, and every fused/split determinism
   test. "Green on CI" is therefore not evidence of GPU-path correctness, and no
   run-count gate exists to catch an all-skip. `release.yml` has the same gap.
   (`2026-06-19-claude-opus-48.md` H4 — ranked "fix before anything else.")

3. **The terminal-death TD lesson diverges 8× between the fused and split paths.**
   The fused death path scales the actor (forward/turn) terminal weight update by
   `ACTOR_VECTOR_SCALE` (1/16) (`kernel_tick.wgsl:642-643`); the split
   `phase_death.wgsl:147-148` uses `TD_VECTOR_SCALE` (1/128) — an 8× weaker lesson
   (the critic uses 1/128 in both, so the divergence is isolated to the actor). The
   fused path was migrated to `ACTOR_VECTOR_SCALE` but `phase_death.wgsl` was never
   updated. It is **live in the default runtime**: `dispatch_ticks_fused_serial`
   runs the split `physics_pipeline` for the physics-only remainder whenever
   `ticks_to_run % brain_tick_stride != 0`, which the wall-clock interactive loop
   hits frequently — so an agent's terminal lesson depends on which dispatch
   processed its death. No parity test catches it because every parity test
   deliberately avoids killing an agent. (`2026-06-19-claude-opus-48.md` M4 —
   contradicts the "fused/split exact symmetry" claim.)

**Provenance.** Every finding re-verified against `develop` source: the foraging
formula at `governor.rs:135-151` (ticks in numerator, confirmed); `ci.yml` has no
GPU-driver step (confirmed); `phase_death.wgsl:147-148` uses `TD_VECTOR_SCALE`
while `kernel_tick.wgsl:642-643` uses `ACTOR_VECTOR_SCALE` (confirmed). The other
06-19 findings are addressed elsewhere or deferred (see Out of scope).

## In scope

- **0001 — Effort-Fitness-Camper-Fix.** Remove the `ticks_alive` factor from both
  effort axes so they are genuinely duration-independent (`foraging = food/energy`,
  `cells_per_distance_rate = cells/distance`), re-derive the two targets for the new
  units, update the calibration replay fixtures/decision doc, and add the missing
  `competent > camper` and duration-invariance assertions. Effort fitness stays
  default-off. See [TASKS.md](TASKS.md).
- **0002 — CI-GPU-Test-Execution.** Install Mesa lavapipe (+ `VK_ICD_FILENAMES` /
  `LIBGL_ALWAYS_SOFTWARE`) in `ci.yml` and `release.yml` so the GPU-gated tests
  actually run, and add a run-count / required-adapter gate that fails CI when zero
  GPU tests execute. Reconcile `CLAUDE.md`. See [TASKS.md](TASKS.md).
- **0003 — Death-Path-TD-Parity.** Change `phase_death.wgsl` to use
  `ACTOR_VECTOR_SCALE` for the actor terminal lesson (matching the fused path), and
  add a death-crossing fused/split parity test that actually kills an agent. See
  [TASKS.md](TASKS.md).

## Origin -> workstream mapping

| Finding (2026-06-19) | Addressed by |
|---|---|
| Effort fitness not scale-invariant; camper is fittest; no `competent > camper` guard (opus H3/M5, gpt-5-codex F1) | `0001` |
| CI silently skips ~63 GPU-gated tests; `CLAUDE.md` lavapipe claim false (opus H4) | `0002` |
| Terminal-death TD lesson diverges 8× (actor scale) fused vs split (opus M4) | `0003` |

## Locked decisions

- **Fix the math, do not graduate the lever.** `effort_rebased_fitness` stays
  default-off. This plan makes the axis correct (duration-independent,
  camper-resistant) and adds the guard test that the kill should have tripped; it
  does **not** flip the default. A separate decision (see Out of scope) governs
  whether to graduate the corrected lever or pivot to environmental pressure.
- **Default path is unaffected.** All three fixes touch either a default-off lever
  (effort fitness), CI configuration, or a parity-only correctness bug whose
  default-runtime effect is a stronger (correct) terminal lesson. None changes the
  default legacy fitness or the shipped brain's steady-state behavior, beyond
  making the split death path agree with the fused one.
- **CI required-GPU gate is opt-in via env var.** A new `XAGENT_REQUIRE_GPU` env
  var (set only in CI) turns "no adapter" into a hard failure, so local runs
  without a GPU still self-skip gracefully while CI cannot silently all-skip.

## Out of scope

- **The deeper fitness-philosophy decision: fix-and-keep vs. retire-for-environmental
  pressure.** All four 06-19 reviews (and Gemini explicitly) argue the "pure"
  alternative to hand-shaped fitness is to make the *environment* demand cognition
  (depleting food zones, seasons, multi-agent competition). That is a research
  direction, not a hardening fix; this plan only corrects the broken math and adds
  the guard. Whether to graduate the corrected effort lever or replace fitness
  shaping with environmental pressure is a future plan / decision doc.
- **The within-life reward model (Plan 0012) and innate priors (Plan 0013).**
  Those carry the "step back from the engineered reward" theme; this plan is the
  disjoint correctness/validity cleanup.
- **Lower-priority 06-19 findings:** partial multi-generation A/B pairing
  (gpt-5-codex F2), the `abs()` signed-correlation gate gap (F3), the unbounded
  recording table (opus M3), the untested `.claude/` workflow (F5), the governor
  monolith refactor, retiring the split kernel path, and the residual
  planning-reference scrub. Each is real but independent; defer to their own plans.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
