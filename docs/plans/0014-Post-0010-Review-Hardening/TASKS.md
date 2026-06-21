# XAgent Plan 0014 — Post-0010 Review Hardening

Fix the three confirmed correctness / scientific-validity defects the four
`2026-06-19` due-diligence reviews surfaced that the reward-model re-think (Plans
0012, 0013) does not cover: (1) the effort-rebased fitness camper inversion —
remove the `ticks_alive` factor from both effort axes so they are genuinely
duration-independent, re-derive the targets, and add the missing `competent >
camper` guard; (2) CI silently skipping the GPU test core — install Mesa lavapipe
and add a required-adapter gate; (3) the 8× terminal-death TD divergence — bring
`phase_death.wgsl`'s actor scale into agreement with the fused path and add a
death-crossing parity test. None flips a default.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md) for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/{plan_slug}--{id}/`).
- **Depends on** lists *direct* prerequisites only ("—" means none).
- **Done when** is the verifiable acceptance criterion; every task must keep
  `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo test -p xagent-sandbox` green (stated as "cargo fmt/clippy/test green").
- GPU tests self-skip without an adapter (`GpuKernel::is_available()`); CI runs Mesa lavapipe after `0002`.
- Line numbers are hints; locate every site by the named symbol (grep).

---

## 0001 — Effort-Fitness-Camper-Fix

### fix-foraging-duration-leak — Make Both Effort Axes Genuinely Duration-Independent

`composite_fitness()` effort mode (`governor.rs:135-151`) computes `foraging =
(food_consumed / (energy_spent/ticks_alive)) / FORAGING_ENERGY_TARGET` and
`cells_per_distance_rate = cells_explored / (distance_traveled/ticks_alive) /
EXPLORATION_RATE_TARGET`. Algebraically both carry `ticks_alive` in the numerator
(`food·ticks/(energy·TARGET)`, `cells·ticks/(distance·TARGET)`), so a long-lived
camper saturates both axes. Remove the per-tick rebasing and denominate on the
cumulative totals directly, then re-derive the two targets for the new units.

**Steps:**
1. Open `crates/xagent-sandbox/src/governor.rs`, locate the `if effort_rebased_fitness { … }` branch in `composite_fitness()` (≈135-151).
2. Replace the foraging computation with a direct cumulative ratio (no `ticks`):
   `let foraging = ((food_consumed as f32 / energy_spent.max(ENERGY_FLOOR)) / FORAGING_ENERGY_TARGET).min(1.0);`
3. Replace `cells_per_distance_rate` with the cumulative ratio (no `ticks`):
   `let cells_per_distance = (cells_explored as f32 / distance_traveled.max(DISTANCE_FLOOR) / EXPLORATION_RATE_TARGET).min(1.0);` and keep `exploration = coverage.min(cells_per_distance)` with `coverage = (cells_explored / total_grid_cells).min(1.0)`. Drop the now-unused `per_tick_energy`/`per_tick_distance`/`ticks` locals in this branch.
4. Re-derive `FORAGING_ENERGY_TARGET` and `EXPLORATION_RATE_TARGET` (`governor.rs` const block ≈48-77) for the new units: pick each so the competent-forager archetype scores its axis ≈ 1.0 and the camper ≈ 0, using the existing archetype fixtures (`fitness_calibration_replay_profiles` / `recorded_generation_production_scale_replay`) as the anchor. Update the const doc-comments with the *why* (no plan/process language).
5. Update the effort-mode doc-comment (`governor.rs:114-120`) to describe the axes as duration-independent ratios of cumulative totals — remove the "per-tick-rate scale-invariant" wording (it is the bug's source).
6. Update every calibration fixture/assertion that pins the OLD magnitudes (`fitness_calibration_replay_profiles` ≈3093-3143 pins `camper ≈ 0.8646`, `competent ≈ 0.3209`; the production-scale replay test pins prod-scale values) to the new formula's values, and fix the internally-stale explanatory comment (`governor.rs:3078`).
7. Update `docs/plans/0010-Intent-Aware-Fitness-Hardening/0010-FITNESS-RECALIBRATION-DECISION.md` (or add a short addendum / new `0014` decision note) recording the corrected formula and re-derived targets.

- **Depends on:** —
- **Done when:** Both effort axes are duration-independent (`food/energy`, `cells/distance`; no `ticks` factor); the two targets are re-derived and documented; all existing calibration fixtures/assertions are updated to the new magnitudes; the effort flag stays default-off; cargo fmt/clippy/test green.

---

### add-anti-camper-assertion — Add the `competent > camper` and Duration-Invariance Guards

The calibration replay currently pins the camper inversion as *expected*
(`fitness_calibration_replay_profiles` asserts only `competent > aimless` and
`camper < 1.0`). After the foraging fix, add the assertions that turn the
prove-or-kill discipline into a real guard.

**Steps:**
1. In `crates/xagent-sandbox/src/governor.rs` calibration tests, add to `fitness_calibration_replay_profiles` (and/or the production-scale replay) a hard `assert!(effort_competent > effort_camper, …)` with a descriptive message.
2. Add a **duration-invariance** test: construct two synthetic profiles with identical `food_consumed/energy_spent` and `cells_explored/distance_traveled` ratios but different `ticks_alive`, and assert their effort composites are equal within a tight tolerance (proving the `ticks` leak is gone).
3. Add a **negative control**: a camper profile (high `ticks_alive`, near-zero `food_consumed`, low `energy_spent`) must score `foraging` near zero and a total composite below the competent forager.
4. Fix the stale explanatory comment (`governor.rs:3078`) so the prose matches the printed composite.

- **Depends on:** fix-foraging-duration-leak
- **Done when:** The calibration tests assert `competent > camper`, duration-invariance (equal-rate / different-lifetime profiles score equal), and the camper negative control scores low; deleting the fix would make these fail; cargo fmt/clippy/test green.

---

## 0002 — CI-GPU-Test-Execution

### install-lavapipe-in-ci — Install Mesa lavapipe so CI Runs the GPU Test Core

`.github/workflows/ci.yml` runs `cargo test -p xagent-sandbox` on bare
`ubuntu-latest` with no GPU driver, so `GpuKernel::is_available()` is false and the
~63 GPU-gated tests self-skip green. `CLAUDE.md` claims CI installs lavapipe — it
does not.

**Steps:**
1. In `.github/workflows/ci.yml`, before the `cargo test` step, add a step that installs a software Vulkan adapter:
   `sudo apt-get update && sudo apt-get install -y mesa-vulkan-drivers vulkan-tools libvulkan1`.
2. Set the environment the wgpu backend needs on the test step (e.g. `LIBGL_ALWAYS_SOFTWARE: "1"`, and the lavapipe ICD via `VK_ICD_FILENAMES` if the distro package does not register it automatically). Add a one-line `vulkaninfo --summary || true` debug step to confirm lavapipe is the selected adapter.
3. Mirror the same install + env into `.github/workflows/release.yml` (it has the identical gap).
4. Reconcile `CLAUDE.md`: the lavapipe claim is now true; also fix the stale "156 tests" count to the actual total (217 at the time of the 06-19 reviews — re-count and state the current number).
5. Confirm locally (or via a CI dry run) that the GPU-gated tests now execute rather than skip (the run output should no longer print the "Skipping: no GPU/fallback adapter" lines for the integration suite).

- **Depends on:** —
- **Done when:** `ci.yml` and `release.yml` install lavapipe and run the GPU-gated tests (no mass "Skipping…" output); `CLAUDE.md` no longer overclaims (lavapipe now installed; test count corrected); cargo fmt/clippy/test green.

---

### add-gpu-test-runcount-gate — Fail CI When Zero GPU Tests Execute

Installing lavapipe is necessary but not self-protecting: a future driver/runner
change could silently return to all-skip. Add a gate so CI cannot pass without
the GPU core having run.

**Steps:**
1. Add an opt-in required-adapter check keyed on a new env var `XAGENT_REQUIRE_GPU`: when it is set (only in CI), a missing adapter is a hard failure, not a skip.
2. Implement as a single guard test (e.g. `gpu_adapter_present_when_required` in `crates/xagent-sandbox/tests/integration.rs`): if `std::env::var("XAGENT_REQUIRE_GPU").is_ok()` then `assert!(GpuKernel::is_available(), "XAGENT_REQUIRE_GPU set but no GPU/lavapipe adapter — CI would silently skip the GPU suite")`; otherwise return (local runs unaffected).
3. Set `XAGENT_REQUIRE_GPU: "1"` on the CI/release `cargo test` step (from `install-lavapipe-in-ci`).
4. Leave every existing per-test `GpuKernel::is_available()` self-skip intact — they still self-skip locally when the env var is unset.

- **Depends on:** install-lavapipe-in-ci
- **Done when:** With `XAGENT_REQUIRE_GPU` set and no adapter, the guard test fails the run; unset (local), it self-skips; CI sets the var so an all-skip can no longer pass green; cargo fmt/clippy/test green.

---

## 0003 — Death-Path-TD-Parity

### fix-phase-death-actor-scale — Match the Split Death Path's Actor Scale to the Fused Path

`phase_death.wgsl:147-148` scales the actor (forward/turn) terminal-death weight
update by `TD_VECTOR_SCALE` (1/128); the fused `kernel_tick.wgsl:642-643` uses
`ACTOR_VECTOR_SCALE` (1/16) — an 8× weaker lesson on the split path, live in the
default runtime via the physics-only remainder dispatch.

**Steps:**
1. Open `crates/xagent-brain/src/shaders/kernel/phase_death.wgsl`, locate the terminal-death weight updates (≈146-148).
2. Change the two **actor** updates (`O_ACTION_FORWARD_WEIGHTS`, `O_ACTION_TURN_WEIGHTS`, ≈147-148) from `TD_VECTOR_SCALE` to `ACTOR_VECTOR_SCALE`, exactly matching `kernel_tick.wgsl:642-643`. Leave the **critic** update (`O_VALUE_WEIGHTS`, ≈146) on `TD_VECTOR_SCALE` (both paths already agree there).
3. Confirm `ACTOR_VECTOR_SCALE` is in scope in `phase_death.wgsl` (it is a `common.wgsl` const shared via the include cascade); if not, ensure the include/const is available.
4. `cargo build -p xagent-brain` and run the existing death/respawn tests.

- **Depends on:** —
- **Done when:** `phase_death.wgsl` actor terminal-death updates use `ACTOR_VECTOR_SCALE` (critic unchanged), byte-matching the fused path; existing respawn tests pass; cargo fmt/clippy/test green.

---

### add-death-crossing-parity-test — Add a Fused/Split Parity Test That Actually Kills an Agent

Every existing fused/split parity test deliberately avoids death, which is why the
actor-scale divergence (M4) hid. Add a test that forces a death and asserts the
two paths agree on the post-death brain state.

**Steps:**
1. In `crates/xagent-sandbox/tests/integration.rs`, add a test (e.g. `death_crossing_brain_state_matches_fused_split`) that constructs a seed/config where an agent reliably dies inside the dispatched window (e.g. spawn with near-zero energy, or place it in a damaging Danger biome).
2. Run the scenario once via `FusedSerial` and once via the split path that exercises `phase_death.wgsl` (the physics-remainder dispatch), reading back `brain_state`.
3. Assert the actor and critic weight slots (`O_ACTION_FORWARD_WEIGHTS`, `O_ACTION_TURN_WEIGHTS`, `O_VALUE_WEIGHTS`) are byte-identical (or within exact-equality, matching the other parity tests) across the two paths after the death tick. Without the `fix-phase-death-actor-scale` change this assertion must fail.
4. Embed the standard GPU self-skip guard as the first lines:

```rust
if !xagent_brain::GpuKernel::is_available() {
    eprintln!("Skipping: no GPU/fallback adapter available");
    return;
}
```

- **Depends on:** fix-phase-death-actor-scale
- **Done when:** A death-crossing fused/split parity test exists, forces an agent death, and asserts byte-identical post-death actor/critic weights across paths; it would fail without the actor-scale fix; cargo fmt/clippy/test green.

---

**End of plan 0014 TASKS.** When every "Done when" bullet is green, the plan's end state is reached.
