---
id: supervisor-gpu-parity
title: "Supervisor fused/split GPU parity"
workstream: "0600"
kind: task
role: developer
depends_on:
  - inheritance-default
gated: false
status: done
merged_as: "dfeee201720d38c28e583d541e90a4d351de4cb4"
touches:
  - crates/xagent-sandbox/tests/supervisor_gpu_parity.rs
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_gpu_parity"
    timeout_ms: 3600000
  - name: fmt
    command: "cargo fmt --all -- --check"
    fix: "cargo fmt --all"
  - name: clippy
    command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
---

# Supervisor fused/split GPU parity

## Outcome

`crates/xagent-sandbox/tests/supervisor_gpu_parity.rs` is a new GPU integration
test file that proves the supervisor run is independent of how the kernel
batches its ticks, and that newborns keep the modulator flags the base config
turns off. It contains three tests:

- `gpu_guard_decision_fails_closed` (pure, no GPU): the guard rule returns
  `Fail` for an absent adapter with `XAGENT_REQUIRE_GPU=1`, `Skip` for an absent
  adapter when the variable is unset or not `1`, and `Run` when an adapter is
  present.
- `supervisor_fused_matches_split_dispatch` (GPU): for both
  `Inheritance::WeightCopy` and `Inheritance::Darwinian`, a seeded supervisor
  run dispatched one 100-tick `dispatch_batch` per sample (run A) and the same
  100 ticks as `dispatch_ticks` chunks of `kernel_batch_size()` (run B) gives
  bit-identical physics at every sample, equal `SampleReport`s and bit-identical
  birth brains at the end.
- `supervisor_newborns_keep_modulator_flags_off` (GPU): with
  `brain.motor_fatigue_enabled = false` and
  `brain.klinotaxis_turn_gain_enabled = false` in the base config, a run that
  performs at least one replacement leaves every newborn ledger row with both
  flags `false`.

The GPU tests fail closed: they return early only when no adapter exists and
`XAGENT_REQUIRE_GPU` is unset, and they panic when it is `1`.

## Decisions

- Guard: copy the `GuardDecision` enum, the pure `gpu_guard_decision(available,
  required)` rule and the `gpu_or_skip()` wrapper from
  `crates/xagent-sandbox/tests/supervisor_run.rs` exactly. Each GPU test begins
  with `if !gpu_or_skip() { return; }`. Helpers are copied, not shared; no new
  dependency.
- Configuration: copy `config(kinds, inheritance)` from
  `crates/xagent-sandbox/tests/supervisor_run.rs` with `kinds = false`. It sets
  `population_size = 10`, boundary interval 1,000 ticks, min age 2,000 ticks,
  score window 1,000 ticks, season period 3,000 ticks, archive capacity 8,
  `agent_tick_budget = 200_000` and world seed 5, and gives several replacements
  over the run. The inheritance is passed explicitly, so the test does not rely
  on the default.
- Sampling: a sample is 100 ticks (`SAMPLE`). Each run loops until
  `run.is_finished()`, as `drive` does in `supervisor_run.rs`, and each sample
  is passed to `run.on_sample(&mut kernel, tick, &physics, food.as_deref())`
  with the post-sample tick.
- Run A (split-free fused path): one `kernel.dispatch_batch(tick, 100)`, then
  `while !kernel.try_collect_state_snapshot() { std::hint::spin_loop(); }`, then
  `kernel.cached_state().to_vec()` and `kernel.cached_food_state()`.
- Run B (`dispatch_ticks` path): let `batch = kernel.kernel_batch_size()`
  (`u32`). For `done` from 0 while `done < 100`, call
  `kernel.dispatch_ticks(tick + done, min(batch, 100 - done))` and add that count
  to `done`. Then perform the same `try_collect_state_snapshot` loop and the same
  copies as run A. If `dispatch_ticks` does not leave a state that
  `try_collect_state_snapshot` collects, the test uses
  `kernel.read_full_state_blocking()` for both runs and the packet's
  physics comparison is unchanged.
- Comparison: at every sample, the physics vector is compared with
  `f32::to_bits` element by element (lengths equal). Food state, when present,
  is compared the same way. The `SampleReport` vectors are compared with `==`
  (`SampleReport` derives `PartialEq`). The final birth brains are compared with
  the `bits` helper copied from `supervisor_run.rs`.
- Newborn flags test: the base config is `config(false, Inheritance::WeightCopy)`
  with the two `brain` flags set to `false`. The store is
  `SupervisorStore::create(db_path, &config)` on a `tempfile::TempDir` path,
  passed as `Some(store)` to `SupervisorRun::start(&mut kernel, &config, Some(store),
  true)`. After `finish()`, the test opens the database with `rusqlite::Connection`
  and reads `SELECT config_json FROM supervisor_slot WHERE occupant_id >= 10`.
  Initial occupants have ids 0 through `population_size - 1` (`population_size =
  10`), so every row with `occupant_id >= 10` is a newborn; `born_tick` is not
  used to select newborns. The test asserts at least one row, deserializes each
  `config_json` as `BrainConfig` through `serde_json`, and asserts both flags are
  `false`.
- The test file carries no plan numbers, task slugs or workstream words; comments
  describe the supervisor GPU parity only.
- No new GPU dispatch path, no WGSL change, no change to any product file.

## Non-goals

- No product code: no change under `crates/xagent-sandbox/src/` or
  `crates/xagent-brain/src/`.
- No change to `crates/xagent-sandbox/tests/modulator_flags.rs`,
  `supervisor_run.rs`, or any other existing test or fixture.
- No golden, fixture file or captured digest.
- No sense-parity, digest or food-kind checks (other tasks).
- No change to the supervisor decision, slot choice, parent draw, mutation or
  any random stream.

**Steps:**

1. Create `crates/xagent-sandbox/tests/supervisor_gpu_parity.rs` with the module
   doc comment, the imports (`xagent_brain::buffers::AgentBrainState`,
   `xagent_brain::GpuKernel`, `xagent_sandbox::supervisor_run::{SampleReport,
   SupervisorRun}`, `xagent_sandbox::supervisor_store::SupervisorStore`,
   `xagent_sandbox::world::WorldState`, `xagent_shared::{BrainConfig, FullConfig,
   Inheritance, SupervisorConfig}`; use the module path where `BrainConfig` is
   exported if `xagent_shared` does not re-export it),
   the constants, `GuardDecision`, `gpu_guard_decision`, `gpu_or_skip`, `config`,
   `kernel_for`, `bits`, and the test `gpu_guard_decision_fails_closed` with the
   three cases in Outcome.
2. Add `fn advance_fused(kernel, tick)` (run A: one `dispatch_batch` of 100 ticks,
   collection loop, copies) and `fn advance_split(kernel, tick)` (run B:
   `dispatch_ticks` chunks of `kernel_batch_size()` totalling 100, then the same
   collection loop and copies), both returning `(Vec<f32>, Option<Vec<f32>>)`.
3. Add `fn drive(config, split: bool) -> (Vec<(Vec<u32>, Option<Vec<u32>>)>,
   Vec<SampleReport>, Vec<Vec<u32>>)` that builds a kernel, starts a
   `SupervisorRun` with `None` store and `measure = false`, drives samples until
   finished, and returns per-sample bit vectors, reports and birth brain bits.
   Add `#[test] fn supervisor_fused_matches_split_dispatch()` that, for each of
   `Inheritance::WeightCopy` and `Inheritance::Darwinian`, starts with the
   guard, runs `drive(.., false)` and `drive(.., true)`, and asserts equal
   per-sample physics bits, equal food bits, equal `SampleReport` vectors (with
   a count greater than zero), equal birth bits, and that at least one report
   records a replacement.
4. Add `#[test] fn supervisor_newborns_keep_modulator_flags_off()` as described
   in Decisions: guard first, base config with both `brain` flags `false`, a
   `tempfile::TempDir` store, one run to finish, then the `supervisor_slot`
   `config_json` assertions on the newborn rows (`occupant_id >= 10`). Use only `tempfile`, `rusqlite` and `serde_json`,
   which `xagent-sandbox` already depends on or dev-depends on.
5. Run the focused-proof check. Expected: `test result: ok` with 3 passed
   (`gpu_guard_decision_fails_closed`, `supervisor_fused_matches_split_dispatch`,
   `supervisor_newborns_keep_modulator_flags_off`). Then run the fmt and clippy
   checks and fix only the new test file, leaving the test bodies' assertions
   unchanged. Run the guard check and confirm it passes.

- **Done when:** the focused-proof check passes with XAGENT_REQUIRE_GPU=1 showing supervisor_gpu_parity tests gpu_guard_decision_fails_closed, supervisor_fused_matches_split_dispatch (both inheritances: bit-equal physics at every sample, equal reports, equal birth brains) and supervisor_newborns_keep_modulator_flags_off (every newborn supervisor_slot config with occupant_id >= 10 keeps both flags false), proving INV-12.
