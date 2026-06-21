# Architecture — Plan 0014 (deltas)

> Edits in `crates/xagent-sandbox/src/governor.rs`,
> `crates/xagent-brain/src/shaders/kernel/phase_death.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl` (read-only reference),
> `crates/xagent-sandbox/tests/integration.rs`,
> `.github/workflows/ci.yml`, `.github/workflows/release.yml`, and `CLAUDE.md`.
> Line numbers are hints; locate by symbol (grep for `FORAGING_ENERGY_TARGET`,
> `per_tick_energy`, `TD_VECTOR_SCALE`, `ACTOR_VECTOR_SCALE`, `is_available`).

## 0001 — Effort-Fitness-Camper-Fix

Today `composite_fitness()` (`governor.rs:122-167`) in effort mode computes, with
`ticks = ticks_alive.max(1)`:

```rust
let per_tick_energy = energy_spent.max(ENERGY_FLOOR) / ticks;
let foraging = ((food_consumed / per_tick_energy) / FORAGING_ENERGY_TARGET).min(1.0);
let per_tick_distance = distance_traveled.max(DISTANCE_FLOOR) / ticks;
let cells_per_distance_rate = (cells_explored / per_tick_distance / EXPLORATION_RATE_TARGET).min(1.0);
```

Algebraically `foraging = food·ticks / (energy·FORAGING_ENERGY_TARGET)` and
`cells_per_distance_rate = cells·ticks / (distance·EXPLORATION_RATE_TARGET)`. The
`ticks` numerator is the camper inflation: a long-lived low-burn agent saturates
both axes regardless of food found or ground covered. The doc-comment claims the
scores are "independent of simulation duration" — false.

Edits:

- **Drop the per-tick rebasing; denominate on the cumulative totals directly** so
  both axes are genuine duration-independent ratios:

```rust
// Food found per unit energy burned — both cumulative, so the ratio is
// duration-independent. A camper that does not eat scores ~0 (food ~ 0).
let foraging = ((food_consumed as f32 / energy_spent.max(ENERGY_FLOOR)) / FORAGING_ENERGY_TARGET).min(1.0);
// Cells covered per unit distance moved — duration-independent.
let coverage = (cells_explored as f32 / total_grid_cells).min(1.0);
let cells_per_distance = (cells_explored as f32 / distance_traveled.max(DISTANCE_FLOOR) / EXPLORATION_RATE_TARGET).min(1.0);
let exploration = coverage.min(cells_per_distance);
```

- **Re-derive the two targets for the new units.** `FORAGING_ENERGY_TARGET` and
  `EXPLORATION_RATE_TARGET` were sized for the old `·ticks` units (~250000, 440);
  the ratios are now ~`food/energy` (O(0.01–0.02) for a competent forager at
  production scale) and ~`cells/distance`. Pick each so a competent-forager profile
  scores foraging/exploration ≈ 1.0 and a camper ≈ 0, using the existing
  archetype fixtures as the calibration anchor; document the derivation in the
  decision doc.
- **Update the doc-comment** to state the axes are duration-independent ratios of
  cumulative totals (the previous "per-tick-rate scale-invariant" wording is the
  bug's source and must go), with no plan/process language.

Properties that make this safe:
- `effort_rebased_fitness` is **default-off** in all presets, so the legacy
  time-denominated path (the shipped default) is untouched; this only changes the
  gated lever's math.
- Both axes remain `.min(1.0)`-bounded and divide by `*.max(FLOOR)` guards, so no
  new divide-by-zero or unbounded path is introduced.
- A camper (food ≈ 0) now scores foraging ≈ 0; a competent forager's food/energy
  ratio is unchanged by lifetime — the inversion is removed by construction, and
  the new guard test (below) proves it.

## 0002 — CI-GPU-Test-Execution

Today `.github/workflows/ci.yml` runs `cargo test -p xagent-sandbox` on
`ubuntu-latest` with no GPU-driver install, so `GpuKernel::is_available()`
(`gpu_kernel.rs:466-484`) returns false and ~63 GPU-gated tests self-skip. There
is no signal that the GPU core ran at all.

Edits:

- **Install a software Vulkan adapter (Mesa lavapipe) before the test step** in
  `ci.yml` (and mirror in `release.yml`), e.g.:

```yaml
- name: Install Mesa lavapipe (software Vulkan)
  run: |
    sudo apt-get update
    sudo apt-get install -y mesa-vulkan-drivers vulkan-tools libvulkan1
- name: Cargo test
  env:
    XAGENT_REQUIRE_GPU: "1"
    LIBGL_ALWAYS_SOFTWARE: "1"
  run: cargo test -p xagent-sandbox
```

  (Use the exact ICD/env vars the project's wgpu backend needs; `vulkaninfo`
  in a debug step confirms lavapipe is the chosen adapter.)
- **Add a required-adapter gate** so CI cannot silently all-skip: when
  `XAGENT_REQUIRE_GPU` is set, `GpuKernel::is_available()` returning false must be
  a hard failure rather than a skip. Implement as a single guard test
  (`gpu_adapter_present_when_required`) that reads the env var and `panic!`s if no
  adapter is found; leave the existing per-test self-skip intact for local runs
  where the env var is unset.
- **Reconcile `CLAUDE.md`**: it currently claims "CI/dev installs Mesa lavapipe"
  (untrue until this lands) and "156 tests" (actual 217). Fix both.

Properties that make this safe:
- Local developer runs without the env var keep self-skipping gracefully — no
  hard GPU requirement is imposed off CI.
- The change is CI/config + one guard test; no runtime or shader code changes.
- lavapipe is the same software adapter the project already assumes for headless
  Linux; this makes the documented intent real.

## 0003 — Death-Path-TD-Parity

Today the terminal-death lesson scales differ between paths
(`TERMINAL_DEATH_TD_ERROR` times the eligibility trace):

```wgsl
// kernel_tick.wgsl:641-643 (fused) — actor uses ACTOR_VECTOR_SCALE (1/16)
brain_state[.. O_VALUE_WEIGHTS  + i] += CRITIC_LEARNING_RATE       * TD_VECTOR_SCALE    * TERMINAL_DEATH_TD_ERROR * trace_critic;
brain_state[.. O_ACTION_FORWARD + i] += ACTION_WEIGHT_LEARNING_RATE * ACTOR_VECTOR_SCALE * TERMINAL_DEATH_TD_ERROR * trace_fwd;
brain_state[.. O_ACTION_TURN    + i] += ACTION_WEIGHT_LEARNING_RATE * ACTOR_VECTOR_SCALE * TERMINAL_DEATH_TD_ERROR * trace_turn;

// phase_death.wgsl:146-148 (split) — actor uses TD_VECTOR_SCALE (1/128): 8x weaker
brain_state[.. O_ACTION_FORWARD + i] += ACTION_WEIGHT_LEARNING_RATE * TD_VECTOR_SCALE    * TERMINAL_DEATH_TD_ERROR * trace_fwd;
brain_state[.. O_ACTION_TURN    + i] += ACTION_WEIGHT_LEARNING_RATE * TD_VECTOR_SCALE    * TERMINAL_DEATH_TD_ERROR * trace_turn;
```

Edits:

- **Change `phase_death.wgsl:147-148`** (the two actor-weight terminal updates) from
  `TD_VECTOR_SCALE` to `ACTOR_VECTOR_SCALE`, matching the fused path. Leave the
  critic update (`:146`) on `TD_VECTOR_SCALE` (both paths already agree there).
- **Add a death-crossing fused/split parity test** in `integration.rs`: drive a
  seed where an agent dies inside the batch, run both `FusedSerial` and the split
  path, and assert byte-identical `brain_state` (or at least the actor/critic
  weight slots) across paths after the death tick. Existing parity tests
  deliberately avoid death, which is why this divergence hid — the new test must
  force a death (e.g. spawn with near-zero energy, or place in a damaging biome).

Properties that make this safe:
- The fused path is the reference (already correct and matching the within-life
  learner's `ACTOR_VECTOR_SCALE`); this brings the split path into agreement, so
  the terminal lesson no longer depends on which dispatch processed the death.
- The change is confined to two WGSL lines plus a new test; no buffer-layout or
  uniform change, and the critic scaling is untouched.
- The new test runs only under a GPU adapter — and after workstream 0002 lands, CI
  actually executes it (closing the loop that let M4 hide).

## Test strategy

- **0001:** `add-anti-camper-assertion` adds (a) `competent > camper` and (b) a
  duration-invariance assertion (two profiles with identical food/energy and
  cells/distance ratios but different `ticks_alive` score equal) to the calibration
  replay tests, and fixes the stale explanatory comment. The existing pinned
  magnitudes (`camper ≈ 0.8646`, `competent ≈ 0.3209`) are updated by
  `fix-foraging-duration-leak` to the new formula's values, so the suite stays
  green after the fix and red if the inversion ever returns.
- **0002:** the `gpu_adapter_present_when_required` guard fails CI when no adapter
  is present and `XAGENT_REQUIRE_GPU` is set; a green CI run now implies the GPU
  core executed.
- **0003:** the death-crossing parity test fails if the actor terminal scales ever
  diverge again.
- CI gate (every task): `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test -p xagent-sandbox` (and `-p xagent-brain`). GPU tests self-skip
  without an adapter locally; CI runs lavapipe after workstream 0002.

## Interaction with prior work

- **Completes the 0009/0010 fitness arc honestly.** 0010 attempted the
  scale-invariant recalibration but the result is camper-dominated; this fixes the
  math and adds the guard the prove-or-kill discipline requires, without graduating
  the lever.
- **Makes 0012/0013 (and every prior plan) actually verified on CI.** Until the
  lavapipe gap is closed, the GPU-gated regression guards those plans add — the
  homeostatic-only gate (0012), the byte-identical-when-off instinct test (0013) —
  do not run in CI. Workstream 0002 is the precondition for trusting any of them
  off the author's machine.
- **Hardens the dual-path parity discipline** the project relies on: the split
  death path was the one place parity was asserted-but-false (M4/L10), and the new
  death-crossing test closes it.
