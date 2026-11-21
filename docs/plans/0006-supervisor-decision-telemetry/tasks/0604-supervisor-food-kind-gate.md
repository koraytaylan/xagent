---
id: supervisor-food-kind-gate
title: "Supervisor food-kind default and season gate"
workstream: "0600"
kind: task
role: developer
depends_on:
  - inheritance-default
gated: false
status: todo
merged_as: ""
touches:
  - crates/xagent-sandbox/tests/supervisor_food_kind.rs
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_food_kind"
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

# Supervisor food-kind default and season gate

## Outcome

`crates/xagent-sandbox/tests/supervisor_food_kind.rs` is a new GPU integration
test file that pins the supervisor's food-kind behaviour (INV-13): kinds are off
by default, and with kinds on the seasons flip exactly on the season schedule,
alternate kinds, and keep the kernel's nourishing kind equal to the reported
kind. It contains four tests:

- `gpu_guard_fails_closed_when_required` (pure, no GPU): `gpu_guard_decision`
  returns `Fail` for an absent adapter with `XAGENT_REQUIRE_GPU=1`, `Skip` for an
  absent adapter when the variable is unset or not `1`, and `Run` when an adapter
  is present.
- `supervisor_food_kinds_off_by_default` (GPU): a `FullConfig::default()` world
  (`food_kinds_enabled` false) run in supervisor mode with a small budget has
  `kernel.nourishing_food_kind() == None` after `SupervisorRun::start` and after
  the run finishes; `food_kinds_slot_value(None) == 0.0`
  (`xagent_brain::buffers`); and no `SampleReport` has `season` `Some`.
- `supervisor_seasons_flip_on_schedule` (GPU, for both `Inheritance::WeightCopy`
  and `Inheritance::Darwinian`): food kinds on, second odour sense on as in
  `config(true, ..)` of `tests/supervisor_run.rs`, `season_period_ticks` 3,000 and
  the 200,000 agent-tick budget (20,000 ticks, at least three flips expected).
  The expected flip ticks are computed independently with
  `TickSchedule::new(stream_seed(config.world.seed, SEASON_STREAM), SEASON_TICKS)`
  from `xagent_sandbox::supervisor`, iterating `next_tick()` / `advance()` and
  keeping the ticks at most the run's final tick. The test asserts:
  - the reports with `season` `Some` occur exactly at those ticks, in order;
  - the kinds alternate, the first being the other kind of the kernel's initial
    `nourishing_food_kind()` (read right after `start`);
  - after each flip `kernel.nourishing_food_kind()` equals that report's kind,
    and it is unchanged between flips;
  - at least three flips happened.
- `supervisor_start_draws_world_slot_one` (GPU, both inheritances): after
  `SupervisorRun::start` the kernel's `nourishing_food_kind()` equals
  `Some(config.world.nourishing_food_kind(1))`.

The GPU tests fail closed: they return early only when no adapter exists and
`XAGENT_REQUIRE_GPU` is unset, and they panic when it is `1`.

## Draw index (decided from the code)

The brief suggested draw index 0 for the initial kind. The code gives index 1
after `start`, and this test asserts index 1:

- `GpuKernel::new` calls `draw_food_kind()` once
  (`crates/xagent-brain/src/gpu_kernel.rs`, constructor, draw index 0).
- `SupervisorRun::start` calls `upload_population`, which calls
  `kernel.reset_agents_seeded(..)` exactly once
  (`crates/xagent-sandbox/src/supervisor_run.rs`). `reset_agents_with_rng` calls
  `draw_food_kind()` (draw index 1). `replace_agent_slot` and
  `set_nourishing_food_kind` do not draw.
- `crates/xagent-brain/tests/food_kind_draw.rs` confirms the sequence: after
  construction the kind is draw 0, and each reset advances it by one.

So the kind `start` leaves in the kernel, and that `begin_run_season` records as
the first season, is `config.world.nourishing_food_kind(1)`. Do not edit
`food_kind_draw.rs`.

## Decisions

- Guard: copy the `GuardDecision` enum, `gpu_guard_decision(available,
  required)` and the `gpu_or_skip()` wrapper from
  `crates/xagent-sandbox/tests/supervisor_run.rs` exactly. Each GPU test begins
  with `if !gpu_or_skip() { return; }`. Helpers are copied, not shared.
- Configuration: copy `config(kinds, inheritance)` from
  `crates/xagent-sandbox/tests/supervisor_run.rs` (population 10, boundary 1,000
  ticks, min age 2,000, score window 1,000, season period 3,000, archive 8,
  world seed 5). The inheritance is passed explicitly in the parameterised tests.
- The off-by-default test builds `FullConfig::default()`, sets
  `governor.population_size = 10`, sets the `governor.supervisor` fields as in
  `config()` with `inheritance: Inheritance::WeightCopy` and
  `agent_tick_budget = 10 * 2_000` (a multiple of agent count times the 100-tick
  sample, as validation requires), and sets `world.seed = 5`. It must not set
  `food_kinds_enabled`; it asserts it is `false` first.
- Driving: copy `kernel_for`, `advance` (one 100-tick `dispatch_batch`, the
  `try_collect_state_snapshot` loop, copies of `cached_state` and
  `cached_food_state`) and the `drive()` loop from `supervisor_run.rs`: loop
  until `run.is_finished()`, call `run.on_sample(&mut kernel, tick, &physics,
  food.as_deref())` with the post-sample tick, and `finish()` at the end.
- The schedule test records each report's `(tick, season)` and compares the
  `Some` ticks with the independently computed list by `==`.
- The file carries no plan numbers, task slugs or workstream words; comments
  describe the supervisor food-kind behaviour only.
- No product change, no WGSL change, no change to any other file.

## Non-goals

- No product code under `crates/xagent-sandbox/src/` or `crates/xagent-brain/src/`.
- No change to `crates/xagent-brain/tests/food_kind_draw.rs`,
  `crates/xagent-sandbox/tests/supervisor_run.rs` or any other existing test.
- No golden, fixture file or captured digest.
- No digest, sense-parity or GPU-parity checks (other tasks).
- No change to the season schedule, the kind flip, or any random stream.

**Steps:**

1. Create `crates/xagent-sandbox/tests/supervisor_food_kind.rs` with the module
   doc comment, the imports (`xagent_brain::buffers::food_kinds_slot_value`,
   `xagent_brain::GpuKernel`, `xagent_sandbox::supervisor::{stream_seed,
   TickSchedule, SEASON_STREAM}`, `xagent_sandbox::supervisor_run::{SampleReport,
   SupervisorRun}`, `xagent_sandbox::world::WorldState`, `xagent_shared::{FoodKind,
   FullConfig, Inheritance, SupervisorConfig}`), the constants, `GuardDecision`,
   `gpu_guard_decision`, `gpu_or_skip`, and the pure test
   `gpu_guard_fails_closed_when_required` with the cases in Outcome.
2. Add the helpers copied from `supervisor_run.rs`: `config(kinds, inheritance)`,
   `kernel_for`, `advance`, and a local `other(kind: FoodKind) -> FoodKind`
   matching `First` and `Second`.
3. Add `supervisor_food_kinds_off_by_default` as described in Decisions: assert
   `food_kinds_enabled` is false, run to finish, and assert `None` kinds at start
   and at the end, `food_kinds_slot_value(None) == 0.0`, and no `Some` season in
   any report.
4. Add `supervisor_seasons_flip_on_schedule` looping over
   `Inheritance::WeightCopy` and `Inheritance::Darwinian`, with the schedule
   computed from `config.world.seed` as in Outcome, and the four assertions
   listed there.
5. Add `supervisor_start_draws_world_slot_one` looping over both inheritances:
   after `start`, assert `kernel.nourishing_food_kind() ==
   Some(config.world.nourishing_food_kind(1))`, with a comment on the two draws
   (construction and the single `start` reset).
6. Run the focused-proof, fmt and clippy checks and fix only the new test file.
   Run the guard check and confirm it passes. Expected: `test result: ok` for all
   four tests.

- **Done when:** the focused-proof check passes with `XAGENT_REQUIRE_GPU=1` showing `supervisor_food_kinds_off_by_default` (no kind and no season at any point with kinds off), `supervisor_seasons_flip_on_schedule` (season reports at exactly the independently computed schedule ticks, alternating kinds equal to `nourishing_food_kind()` after each flip, for both inheritances), and `supervisor_start_draws_world_slot_one` (initial kind equal to draw index 1), proving INV-13.
