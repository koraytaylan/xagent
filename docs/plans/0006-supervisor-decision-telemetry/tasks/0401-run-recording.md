---
id: run-recording
title: "Driver builds and records telemetry without changing decisions"
workstream: "0400"
kind: task
role: developer
depends_on:
  - telemetry-store
gated: false
status: done
merged_as: "86aa59defda9c8c0177dd18ef410e60a230b3364"
touches:
  - crates/xagent-sandbox/src/supervisor_run.rs
  - crates/xagent-sandbox/src/headless.rs
  - crates/xagent-sandbox/tests/supervisor_telemetry_run.rs
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_telemetry_run"
    timeout_ms: 3600000
  - name: driver-regression
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_run --test supervisor_headless"
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

# Driver builds and records telemetry without changing decisions

## Outcome

`SupervisorRun` builds a `TelemetryFrame` at every sample and persists it when
recording is on, while decisions, `SampleReport`s, birth brains and 0005 rows
stay identical with recording on and off. The headless supervisor loop feeds
ticks per second and gains `run_supervisor_with_recording`. The "Driver" section
of ARCHITECTURE.md is binding and is not repeated here.

## Decisions

- `supervisor_run.rs` private fields: `telemetry: TelemetryBuilder` (agent count
  at `start`), `recording: bool` (`store.is_some()`), `telemetry_stats:
  Option<RunSummaryAccumulator>` (only when `food_kinds_enabled`, same
  constructor arguments as the measure accumulator but separate),
  `last_ticks_per_second: Option<f64>`, `last_frame: Option<TelemetryFrame>`,
  `telemetry_warned: bool`.
- Public methods, exact signatures: `pub fn set_recording(&mut self, on: bool)`,
  `pub fn note_ticks_per_second(&mut self, rate: f64)` (stores finite rates
  only), `pub fn take_frame(&mut self) -> Option<TelemetryFrame>`
  (`Option::take` of `last_frame`).
- `on_sample`: keep the decision, season and boundary steps unchanged. Then keep
  the `replace()` outcome and build the frame with `TelemetryBuilder::on_sample`
  from `standings`, `last_boundary` (boundary ticks only), the outcome's
  `parent_config`, `newborn_config` and occupant, the archive outcomes, and the
  progress inputs (`total_ticks`, agent-tick budget, `next_boundary_tick - tick`,
  nourishing kind, `season.next_tick()` when food kinds are on,
  `last_ticks_per_second`).
- Physics: when `food` is `Some`, feed `record_physics_sample` on
  `telemetry_stats` with the measure accumulator's data. At a season flip, call
  `begin_season` and pass `closed_totals()` to `on_season_closed`.
- Persistence: when `recording` and `frame.persist`, call
  `store.record_telemetry(&frame)`. `last_frame` is set in every case.
- Failure policy (INV-8): any telemetry error sets `recording = false`, drops
  `telemetry_stats`, logs once through `log::warn!` (guarded by
  `telemetry_warned`), and never returns `Err` from `on_sample` or touches
  `replace`, `plan_parent` or any draw.
- `SampleReport`, `BoundaryReport` and the `start` signature are unchanged.
- `headless.rs`: the body of `run_supervisor` moves into `pub fn
  run_supervisor_with_recording(config: FullConfig, db_path: &str, options:
  &HeadlessOptions, record: bool) -> Result<(), HeadlessError>`, which calls
  `run.set_recording(record)` after `start`. `run_supervisor` calls it with
  `true`. Each `supervisor_step` is timed with `Instant`, and when the elapsed
  time is positive, `note_ticks_per_second(batch as f64 / elapsed.as_secs_f64())`
  is called. `HeadlessOptions` gets no new field.
- `tests/supervisor_telemetry_run.rs` uses the GPU guard pattern of
  `tests/supervisor_run.rs`: pure `gpu_guard_decision(available, required)` with a
  `#[test]` asserting `Run`, `Skip` and `Fail` for the four combinations, and
  `gpu_or_skip()`, which panics under `XAGENT_REQUIRE_GPU=1` with no adapter. Small
  config as in `tests/supervisor_headless.rs` (6 agents, boundary 1000, min age
  2000, window 1000, season 3000, a budget yielding several replacements).
  Databases are `tempfile::TempDir` paths read with `rusqlite::Connection`.

Named tests in that file:

- `recording_on_off_same_decisions`: two temp databases, `record` true and false
  (food kinds off). `supervisor_slot`, `supervisor_archive` and `supervisor_event`
  rows are compared column by column, excluding `id`, for exact equality. The
  record-off database has zero rows in the six telemetry tables.
- `recording_writes_consistent_rows`: each `replace` boundary row has a matching
  `supervisor_event` replace row (tick, slot, occupant); each `below_median`
  candidate has `score < median`; each idle boundary's `idle_*` counts sum to the
  agent count; `food_per_agent_tick` series rows exist for every
  `TELEMETRY_CADENCE_SAMPLES` cadence tick; every progress row has
  `ticks_to_boundary > 0`. With food kinds on and season flips, `late_*` rows
  appear exactly at season ticks.
- `take_frame_is_some_after_every_sample_and_repeats`: drives `SupervisorRun` as
  `drive()` in `tests/supervisor_run.rs` does. `take_frame()` is `Some` after each
  `on_sample`, with `ticks_per_second` `None` unless noted. Two seeded runs give
  equal frames field for field; a third run calling `note_ticks_per_second(1000.0)`
  differs only in `ticks_per_second`, which is `Some(1000.0)`.
- `telemetry_write_failure_keeps_decisions`: a recorded `SupervisorRun` started
  with `SupervisorStore::create` on a temp database. After `start`, a second
  `rusqlite::Connection` runs `DROP TABLE supervisor_series`, so the next
  `record_telemetry` fails. Driving to the end, every `on_sample` returns `Ok`,
  and `SampleReport`s and birth brains equal an unrecorded run (store `None`) of
  the same config. Frames after the failure still come from `take_frame`.

## Non-goals

- No change to slot choice, parent draw, mutation, random streams, the season
  schedule, the kernel or WGSL; no new GPU dispatch or readback.
- No change to 0005 tables and writers, `HeadlessOptions`, the generation path,
  or `BoundaryReport`.
- No GUI, worker or `SimEvent` change; no checkpoint or replay.
- No edit to `supervisor.rs`, `supervisor_telemetry.rs`, `supervisor_store.rs`,
  `run_summary.rs` or existing test files.
- Doc comments and test names are plain prose: no plan numbers, task slugs or
  workstream words.

**Steps:**

1. In `supervisor_run.rs`, add the six private fields, initialize them in `start`,
   and add `set_recording`, `note_ticks_per_second` and `take_frame` with the
   exact signatures above.
2. In `on_sample`, after the existing steps, keep the `replace()` outcome, build
   the frame, feed the physics and season hooks, persist when recording and
   `frame.persist`, store `last_frame`, and apply the INV-8 policy to every error.
3. In `headless.rs`, add `run_supervisor_with_recording`, make `run_supervisor`
   delegate with `true`, call `set_recording` after `start`, and feed
   `note_ticks_per_second` from `Instant` timing around each `supervisor_step`.
4. Create `tests/supervisor_telemetry_run.rs` with the GPU guard, its pure unit
   test, the config helper and the temp-database helper.
5. Add `recording_on_off_same_decisions` and `recording_writes_consistent_rows`,
   including the food-kind season variant.
6. Add `take_frame_is_some_after_every_sample_and_repeats` and
   `telemetry_write_failure_keeps_decisions`.

- **Done when:** `env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_telemetry_run` (focused-proof) passes with `gpu_guard_decision`, `recording_on_off_same_decisions`, `recording_writes_consistent_rows`, `take_frame_is_some_after_every_sample_and_repeats` and `telemetry_write_failure_keeps_decisions` green, proving INV-8.
