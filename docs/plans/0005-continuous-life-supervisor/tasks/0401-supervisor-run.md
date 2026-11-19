---
id: supervisor-run
title: "Shared kernel-facing supervisor driver"
workstream: "0400"
kind: task
role: developer
depends_on: [supervisor-store, kernel-supervisor-hooks]
gated: false
touches: [crates/xagent-sandbox/src/lib.rs, crates/xagent-sandbox/src/supervisor_run.rs, crates/xagent-sandbox/tests/supervisor_run.rs]
status: todo
merged_as: ""
checks:
  - {name: focused-proof, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_run", timeout_ms: 3600000}
  - {name: sandbox-clippy, command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings", timeout_ms: 3600000}
  - {name: sandbox-fmt, command: "cargo fmt -p xagent-sandbox -- --check", fix: "cargo fmt -p xagent-sandbox"}
  - {name: sandbox-guard, command: "cargo test -p xagent-sandbox --test contributing_guard"}
---

# Shared kernel-facing supervisor driver

## Outcome

`lib.rs` declares `pub mod supervisor_run;` after `supervisor_store`. `supervisor_run.rs`
provides `SupervisorRun`, `SampleReport`, `BoundaryReport`, `SupervisorFinish` and
`SupervisorRunError` exactly as ARCHITECTURE.md "Shared driver" binds them. The driver
borrows the caller's `GpuKernel`, never builds one, and never builds a second
`WorldState` for the kernel.

## Scope

- `lib.rs`: one module line.
- `supervisor_run.rs` (new): the types, `start`, `tick`, `next_sample_tick`,
  `total_ticks`, `is_finished`, `on_sample`, `birth_state`, `finish`, and
  `SupervisorRunError` (`Debug`, `Display`, `Error`, `From` for each wrapped source
  error). Documented consts for the reset seed tag and any driver constant.
- `tests/supervisor_run.rs` (new): `gpu_or_skip` and `gpu_guard_decision` copied from
  0004's test helper, and the nine tests below. Named consts: 10 agents, boundary 1,000,
  min age 2,000, window 1,000, season 3,000, budget 10 × 20,000 agent-ticks.

## Non-goals

- No edit to `headless.rs`, `main.rs`, `sim_runtime.rs`, `supervisor.rs`,
  `supervisor_store.rs`, `run_summary.rs` or `gpu_kernel.rs`.
- No CLI, no new kernel API, no resume of a supervisor run.

## Prerequisites

- `supervisor-store` (lands `supervisor-core`, `run-summary`, `run-mode-column`,
  `evolution-mode-config`): `Supervisor`, `SlotCounters`, `SampleDecision`,
  `ReplacementOutcome`, `ArchiveChange`, `ParentRef`, `ReplaceReason`,
  `RunSummaryAccumulator`, `SupervisorStore`, `StoreError`.
- `kernel-supervisor-hooks`: `SlotSpawn`, `slot_spawn_physics_row`,
  `replace_agent_slot`, `SlotReplaceError`, `set_nourishing_food_kind`,
  `FoodKindSetError`.
- Plan 0004: `FoodKind`, `food_kinds_enabled`, `nourishing_food_kind()`.
- Read landed names in `supervisor.rs`, `supervisor_store.rs` and `run_summary.rs`
  and use them; do not edit those files. A mismatch is a contradiction to report with
  file and line.

## Decisions

- **Validation first.** `start` runs `config.governor.supervisor.validate(population_size)`,
  requires `kernel.agent_count() == population_size` and
  `SUPERVISOR_SAMPLE_TICKS % kernel_batch_size() == 0`, and returns an error before
  any write. `config.brain.vision_stride` is unconstrained.
- **Initial population.** Brains and configs come from the core's slot ledgers
  (`birth_brain`, `config`). Upload order: `upload_world`, `upload_agents` with
  `Agent::new(id, world.safe_spawn_position(), slot, config.brain.clone(), 0)` rows,
  `reset_agents_seeded` with seed `config.world.seed ^ RESET_SEED_TAG` (documented
  const), then one `replace_agent_slot` per slot. This uses only the public
  single-slot writer, so no private kernel write is needed.
- **Season at start.** With food kinds on and `measure`, `begin_season(0,
  kernel.nourishing_food_kind())`; a store gets `record_season(0, kind)`.
- **Sample order (`on_sample`).** Check `tick == next_sample_tick()` and
  `physics.len()` equals `population_size * PHYS_STRIDE`, else error. Build
  `SlotCounters::from_physics` per slot, `core.record_sample`, then
  `accumulator.record_physics_sample`. On a season flip, call
  `kernel.set_nourishing_food_kind(other)` only with food kinds on, then
  `store.record_season` and `accumulator.begin_season(tick, Some(new))`.
- **Boundary.** For `PeerLearned` read `kernel.read_agent_state(parent_slot)`, then
  `core.complete_replacement`. Spawn with `Agent::new(occupant_id,
  world.safe_spawn_position(), slot, newborn_config, tick)` into `SlotSpawn`, call
  `kernel.replace_agent_slot`, `accumulator.reset_agent_baseline(slot)`,
  `store.record_boundary`. An idle boundary calls `store.record_idle_boundary`.
- **Finish.** `core.finish` → `store.record_archive_changes` → `store.record_ticks`;
  `accumulator.finish(ticks)` gives the summary. Done when `tick` reaches
  `total_ticks` (`agent_tick_budget / population_size`; 0 means never).

## Steps

**Steps:**
1. Add `pub mod supervisor_run;` to `lib.rs`. Create `supervisor_run.rs` with the
   types, error enum with `From` impls, and the documented consts.
2. Implement `start` and its helpers (validation, world and agent rows, upload,
   reset, slot replacement writes, store slot records, initial season) as Decisions
   binds, each function under about 50 lines.
3. Implement `on_sample` with the order in Decisions, using private helpers for
   counters, season flip and boundary.
4. Implement `tick`, `next_sample_tick`, `total_ticks`, `is_finished`, `birth_state`
   and `finish`.
5. Create `tests/supervisor_run.rs` with the helpers and the tests
   `gpu_guard_fails_closed_when_required`, `supervisor_run_is_reproducible`,
   `supervisor_run_uses_one_kernel_and_exact_batches`,
   `supervisor_run_boundary_writes_only_replaced_slot`,
   `supervisor_run_flips_kind_only_at_season_ticks` and
   `supervisor_run_rejects_invalid_setup`. The helper loop is
   `dispatch_batch(tick, 100)`, spin `try_collect_state_snapshot`, then
   `on_sample(cached_state, cached_food_state)`. Only one kernel is alive at a time.
6. Add `supervisor_run_weight_copy_reads_learned_parent` and
   `supervisor_run_summary_counts_agent_ticks` (`measure = true`, `mode` Supervisor,
   inheritance as configured, `agent_ticks == budget`).

- **Done when:** the checks focused-proof, sandbox-clippy, sandbox-fmt and sandbox-guard
  pass on the host, with focused-proof reporting ok for all eight named tests
  (`gpu_guard_fails_closed_when_required`, `supervisor_run_is_reproducible`,
  `supervisor_run_uses_one_kernel_and_exact_batches`,
  `supervisor_run_boundary_writes_only_replaced_slot`,
  `supervisor_run_flips_kind_only_at_season_ticks`,
  `supervisor_run_rejects_invalid_setup`,
  `supervisor_run_weight_copy_reads_learned_parent`,
  `supervisor_run_summary_counts_agent_ticks`); together they prove
  INV-02 (invalid config or agent count errors before any write), INV-05 (a replacement
  inherits only the genome, and WeightCopy reads the parent's learned state), INV-07 (at
  the first replacement only that slot's row and brain change, with no dispatch between
  the reads), INV-08 (one kernel, `ticks / kernel_batch_size()` batches), INV-10 (the
  kind changes only at season ticks), INV-15 (the GPU guard fails closed), and INV-16
  (the summary measures without writes).

## Evidence and risks

- The worker has no shell; the host-run checks are the proof.
- The boundary test compares `read_full_state_blocking` and `read_agent_state` taken
  immediately before and after `on_sample`, with no dispatch between them.
- Reproducibility compares `SampleReport` vectors and final birth brains per slot.
- Risk: if the landed `SampleDecision` or `SupervisorRunError` names differ, report
  the file and line rather than renaming landed items.
