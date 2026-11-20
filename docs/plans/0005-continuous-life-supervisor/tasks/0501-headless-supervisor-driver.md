---
id: headless-supervisor-driver
title: "Headless driver selection and the supervisor loop"
workstream: "0500"
kind: task
role: developer
depends_on: [supervisor-run]
gated: false
touches: [crates/xagent-sandbox/src/headless.rs, crates/xagent-sandbox/tests/supervisor_headless.rs]
status: todo
merged_as: ""
checks:
  - {name: focused-proof, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_headless", timeout_ms: 3600000}
  - {name: seeded-digest-determinism, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test seeded_headless_digest", timeout_ms: 3600000}
  - {name: sandbox-clippy, command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings", timeout_ms: 3600000}
  - {name: sandbox-fmt, command: "cargo fmt -p xagent-sandbox -- --check", fix: "cargo fmt -p xagent-sandbox"}
  - {name: sandbox-guard, command: "cargo test -p xagent-sandbox --test contributing_guard"}
---

# Headless driver selection and the supervisor loop

## Outcome

`headless.rs` picks the headless driver from `governor.evolution_mode`. Generation
mode runs the existing loop unchanged; supervisor mode runs `SupervisorRun` on one
kernel and can write the run summary. `run_headless` keeps its signature and
generation behaviour.

## Scope

- `headless.rs`: `HeadlessOptions`, `HeadlessError`, `run_headless_with_options`,
  the private `run_generations` (the current `run_headless` body, moved verbatim),
  and the supervisor loop. `run_free_life`, `BrainSource` and the validation
  helpers stay untouched.
- `tests/supervisor_headless.rs` (new): the five tests below and small named consts
  (10 agents, boundary 1,000, min age 2,000, window 1,000, season 3,000, budget
  10 × 20,000 agent-ticks), the same values as 0401's tests.

## Non-goals

- No edit inside the generation loop body (generation-run-summary adds its
  measurement there). No free-life change (archive-brain-source). No `main.rs`
  change (evolution-cli). No GUI. No edit to `supervisor_run.rs`, `supervisor.rs`,
  `supervisor_store.rs` or `gpu_kernel.rs`.

## Prerequisites

- `supervisor-run` (0401): `SupervisorRun::start`, `tick`, `next_sample_tick`,
  `total_ticks`, `is_finished`, `on_sample`, `finish`, `SupervisorFinish`,
  `SampleReport`, `SupervisorRunError`; `SupervisorStore::create(db_path, &FullConfig)`
  and `StoreError`. Read the landed names in `supervisor_run.rs` and
  `supervisor_store.rs`; a mismatch is a contradiction to report with file and line.
- Baseline: `run_headless(config, db_path, resume, _has_gpu)` is called by the
  seeded digest test with this exact signature.

## Decisions

- `HeadlessError` (`Debug`, `Display`, `Error`) has variants
  `ResumeInSupervisorMode`, `Store(StoreError)`, `SupervisorRun(SupervisorRunError)`,
  `Summary(std::io::Error | serde_json::Error)` and the temporary
  `GenerationSummaryUnsupported`, with `From` impls for each wrapped error.
- Dispatch: `Generations` with `run_summary` set returns
  `GenerationSummaryUnsupported` before any write; otherwise it calls
  `run_generations(config, db_path, resume)`. Generation-mode panics stay as they
  are inside `run_generations`.
- `Supervisor`: `resume` returns `ResumeInSupervisorMode` before any database is
  touched. Then `SupervisorStore::create`, `u32::try_from(population_size)`,
  `GpuKernel::new(pop, food_count, &config.brain, &config.world)` with
  `food_count` from `WorldState::new(config.world)`, and
  `SupervisorRun::start(&mut kernel, &config, Some(store), options.run_summary.is_some())`.
- Loop: while `!run.is_finished()`: `dispatch_batch(run.tick(),
  u32::try_from(SUPERVISOR_SAMPLE_TICKS))`, spin `try_collect_state_snapshot` with
  `yield_now` as `run_generations` does, then `on_sample(&mut kernel,
  run.next_sample_tick(), kernel.cached_state(), kernel.cached_food_state())`.
- Progress: one line per replacement and per season flip, plus one every 10% of
  the budget when `total_ticks()` is `Some`. Budget 0 runs until interrupted.
- Summary: `finish()` returns `summary: Option<RunSummary>`; when
  `options.run_summary` is `Some`, `None` is an error, otherwise the summary is
  written with `serde_json::to_string_pretty` to that path.
- Store comparison: rows compared are all columns except the autoincrement `id`,
  ordered by `id`.
- `run_headless` calls `run_headless_with_options(config, db_path, resume,
  &HeadlessOptions::default())`; on `Err` it prints the error to stderr with
  `eprintln!`. `println!` progress lines are allowed (as the existing loop does).

## Steps

**Steps:**
1. Move the current `run_headless` body into private `run_generations` with no
   edits; add `HeadlessOptions`, `HeadlessError` and its `From` impls.
2. Add `run_headless_with_options` with the Generations and Supervisor dispatch
   and the resume and summary refusals; make `run_headless` delegate to it.
3. Implement the supervisor loop (kernel, start, dispatch, readback, `on_sample`)
   with the progress lines, in helpers each under about 50 lines.
4. Implement finish and the summary write using the Decisions above.
5. Create `tests/supervisor_headless.rs` with the fail-closed GPU helper pattern of
   `gpu_adapter_present_when_required` in `tests/integration.rs` and these tests:
   `gpu_guard_fails_closed_when_required`, `supervisor_headless_store_is_reproducible`,
   `supervisor_headless_writes_run_summary`, `supervisor_headless_refuses_resume`,
   `generation_mode_is_the_default_driver` (`FullConfig::default()`, a tiny
   `tick_budget`, `max_generations` 2, through `run_headless_with_options`; asserts
   generation node rows exist and no `supervisor_*` rows).

- **Done when:** the checks focused-proof (all five named tests report ok) and
  seeded-digest-determinism pass on the host, proving INV-12 (mode picks the driver,
  `run_headless` signature and generation path unchanged, identical supervisor
  event and archive rows across two same-seed runs), INV-01 (the seeded headless
  run through the unchanged `run_headless` gives the same digest twice, and the
  generation body is moved into `run_generations` verbatim; this task does not
  require the digest to equal the recorded fixture, whose value is
  adapter-specific and was recorded on a different adapter at 0976448) and INV-15 (the GPU guard panics when required and no
  adapter exists).

## Evidence and risks

- The worker has no shell; the host-run checks named above are the proof.
- Risk: the generation-mode byte path must not change. Any edit to the moved body
  must be a verbatim move; the seeded digest determinism check runs that path.
- Risk: `on_sample` must receive `run.next_sample_tick()`, not a locally counted
  tick, or it fails the `tick == next_sample_tick()` guard from 0401.
