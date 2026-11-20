---
id: worker-supervisor
title: "Simulation worker drives the supervisor on the sample grid"
workstream: "0600"
kind: task
role: developer
depends_on: [supervisor-run]
gated: false
touches: [crates/xagent-sandbox/src/sim_runtime.rs, crates/xagent-sandbox/src/gpu_orchestration.rs]
status: done
merged_as: "505b9b11a7364b29ab1ad91802fa1e7840b70116"
checks:
  - {name: focused-proof, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --bin xagent -- sim_runtime::tests::supervisor_", timeout_ms: 3600000}
  - {name: worker-tests, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --bin xagent -- sim_runtime::", timeout_ms: 3600000}
  - {name: sandbox-clippy, command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings", timeout_ms: 3600000}
  - {name: sandbox-fmt, command: "cargo fmt -p xagent-sandbox -- --check", fix: "cargo fmt -p xagent-sandbox"}
  - {name: sandbox-guard, command: "cargo test -p xagent-sandbox --test contributing_guard"}
---

# Simulation worker drives the supervisor on the sample grid

## Outcome

In supervisor mode the sim worker owns one `SupervisorRun` and the kernel, clamps dispatches to the sample grid, does every supervisor readback on the worker thread, and emits the same `SampleReport`s as the headless driver for the same seed at any speed. Generation mode is unchanged.

## Scope

- `sim_runtime.rs`: `pub struct SupervisorWorkerInit { pub config: FullConfig, pub db_path: Option<String> }`; `SimInit.supervisor: Option<SupervisorWorkerInit>`; `SimEvent::SupervisorSample(SampleReport)`; `fn clamp_ticks_to_supervisor_sample(ticks_to_run: u32, tick: u64, sample_ticks: u64) -> u32`; Worker fields `supervisor: Option<SupervisorRun>`, `supervisor_mode: bool`, `supervisor_events: Vec<SimEvent>`; imports; tests.
- `gpu_orchestration.rs`: `supervisor: None` in its `SimInit` literal and the arm `SimEvent::SupervisorSample(_) => {}`.

Non-goals: no UI, `evolution.rs` or `ui.rs` change; no generation-mode cadence, snapshot or `GenerationBudgetReached` change; no new kernel, store or driver API.

## Prerequisites

`supervisor-run` lands `SupervisorRun` (`start`, `tick`, `next_sample_tick`, `on_sample`, `birth_state`, `finish`, `is_finished`), `SampleReport`, `BoundaryReport` in `crates/xagent-sandbox/src/supervisor_run.rs`, and `SupervisorStore::create(db_path: &str, config: &FullConfig)` in `supervisor_store.rs`. A name or signature mismatch is a contradiction to report with file and line.

## Decisions

- **Construction.** With `supervisor` Some, `Worker::new` skips the `upload_world`, `upload_agents`, `patch_agent_configs` and `init.upload` brains, since `SupervisorRun::start` does the uploads and seeds slots. It sets `tick_budget = 0`, opens the store when `db_path` is Some, and calls `SupervisorRun::start(&mut kernel, &config, store, false)`. The test sets `init.agent_count` and `init.food_count` from `config.world` and `config.governor` population. On any error `supervisor` stays None, `paused = true`, and `SimEvent::Error` goes onto `supervisor_events`.
- **Birth states.** Supervisor mode fills `birth_states[slot]` from `run.birth_state(slot)` at construction, and refreshes a slot after a report whose `boundary.slot` is Some.
- **Clamp.** `clamp_ticks_to_supervisor_sample` returns `ticks_to_run` when `sample_ticks == 0`, else the minimum of `ticks_to_run` and `sample_ticks - tick % sample_ticks`, so the batch ends on the next multiple. It is pure, like `clamp_ticks_to_generation_budget`. `advance_compute` applies it after the alignment and generation clamp, when a run exists, with `SUPERVISOR_SAMPLE_TICKS`.
- **Sample path.** After `dispatch_ticks` and the tick update, when `self.tick == run.next_sample_tick()`: `physics = read_full_state_blocking().to_vec()`, then `run.on_sample(&mut self.kernel, self.tick, &physics, None)`, with disjoint field borrows. Push `SimEvent::SupervisorSample(report)` only when the report has a boundary, a season or `finished`.
- **Finish and errors.** If `report.finished`, send the sample, then `take()` the run and call `finish()`, logging `log::error!` on `Err`; set `paused = true`. If `on_sample` fails, push `SimEvent::Error`, drop the run and pause. Fail closed: no further dispatch.
- **Delivery.** `step` drains `supervisor_events` with blocking `send` before the paused check; control events are never dropped.
- **Commands.** `ResetPopulation` in supervisor mode logs `log::warn!` and changes nothing; `RequestBirthState` answers from `birth_states`.
- **Idle.** With `supervisor_mode` and no run, `advance_compute` returns false.

## Steps

**Steps:**
1. Add the Scope items and `supervisor: None` to every `SimInit` literal in the `sim_runtime` tests module. In `gpu_orchestration.rs`, add `supervisor: None` and the `SupervisorSample(_)` arm.
2. Implement the supervisor branch of `Worker::new` per Decisions.
3. Make `advance_compute` apply the clamp and sample path per Decisions.
4. In `step`, drain `supervisor_events` before the paused check. In `handle_command`, ignore `ResetPopulation` in supervisor mode and answer `RequestBirthState` from `birth_states`.
5. Add `supervisor_clamp_never_crosses_sample_grid` (pure) and `supervisor_worker_matches_headless_driver`. The reference is one `GpuKernel` with `SupervisorRun::start`, then `dispatch_batch(start, 100)`, a spin on `try_collect_state_snapshot`, and `on_sample(cached_state, None)`, collecting reports with the same filter. The worker runs at speed 1 (one-batch dispatches) and at a fast multiplier (fused). The test asserts `counters.dispatch_calls` differs between the runs and the report sequences equal the reference's.
6. Add the helpers and tests `supervisor_gpu_guard_fails_closed_when_required` (CPU-only), `supervisor_worker_uses_one_kernel_and_exact_batches`, `supervisor_worker_ignores_reset_population`, `supervisor_worker_pauses_when_finished` and `supervisor_worker_answers_birth_state_from_ledger`. Copy 0004's fail-closed helper (existing guards unchanged): `fn supervisor_gpu_guard_decision(available: bool, required: bool) -> Result<bool, &'static str>` returns `Ok(true)` when available, `Err` naming the missing adapter when unavailable and required, else `Ok(false)`. `fn supervisor_gpu_or_skip() -> bool` calls it with `GpuKernel::is_available()` and `std::env::var_os("XAGENT_REQUIRE_GPU").is_some()`, panics on `Err`, prints a skip line on `Ok(false)`, and returns the bool. Every new GPU test starts with `if !supervisor_gpu_or_skip() { return; }`, not the existing guards, which never read `XAGENT_REQUIRE_GPU`. The guard test asserts `(false, true)` is `Err`, `(false, false)` is `Ok(false)`, and `(true, true)` and `(true, false)` are `Ok(true)`. Shared helpers: `supervisor_test_config()` (10 agents, boundary 1,000, min age 2,000, window 1,000, season 3,000, `agent_tick_budget` 100,000) and `supervisor_init(config, db_path: None)`. `supervisor_worker_uses_one_kernel_and_exact_batches` builds one supervisor-mode Worker and steps it as `matches_headless_driver` does, at a fast multiplier so fused dispatches occur, until its tick reaches a multiple of `SUPERVISOR_SAMPLE_TICKS` of at least 3,000. It asserts `worker.kernel.probe_kernel_batches() == worker.tick / worker.kernel.kernel_batch_size() as u64`; that the kernel agent count equals the `config.governor` population (10), via an existing kernel accessor or else `worker.birth_states` length (the report says which); and that `ResetPopulation` sent through `handle_command` between batches neither rebuilds the kernel nor changes the batch accounting, the same assertion holding afterwards. Drop each kernel before creating the next.

- **Done when:** check focused-proof reports ok, none skipped, for all seven `supervisor_` tests (supervisor_gpu_guard_fails_closed_when_required, supervisor_clamp_never_crosses_sample_grid, supervisor_worker_matches_headless_driver, supervisor_worker_uses_one_kernel_and_exact_batches, supervisor_worker_ignores_reset_population, supervisor_worker_pauses_when_finished, supervisor_worker_answers_birth_state_from_ledger); check worker-tests reports every pre-existing `sim_runtime` test ok; `matches_headless_driver` and `clamp_never_crosses_sample_grid` prove INV-13; `uses_one_kernel_and_exact_batches` proves INV-08; `supervisor_gpu_guard_fails_closed_when_required`, with `XAGENT_REQUIRE_GPU=1` set in the GPU checks, proves INV-15.

## Evidence and risks

- INV-13 assumes a clamp-boundary chunked dispatch gives the same samples as a whole-batch one. If `matches_headless_driver` fails there, report the failing assertion; do not weaken the reference. `dispatch_ticks` already accepts partial batches.
