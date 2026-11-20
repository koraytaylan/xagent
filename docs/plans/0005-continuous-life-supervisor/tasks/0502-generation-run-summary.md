---
id: generation-run-summary
title: "Generation-mode run summary and agent-tick budget conversion"
workstream: "0500"
kind: task
role: developer
depends_on: [headless-supervisor-driver]
gated: false
touches: [crates/xagent-sandbox/src/headless.rs, crates/xagent-sandbox/tests/generation_run_summary.rs]
status: todo
merged_as: ""
checks:
  - {name: focused-proof, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test generation_run_summary", timeout_ms: 3600000}
  - {name: seeded-digest-determinism, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test seeded_headless_digest", timeout_ms: 3600000}
  - {name: sandbox-clippy, command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings", timeout_ms: 3600000}
  - {name: sandbox-fmt, command: "cargo fmt -p xagent-sandbox -- --check", fix: "cargo fmt -p xagent-sandbox"}
  - {name: sandbox-guard, command: "cargo test -p xagent-sandbox --test contributing_guard"}
---

# Generation-mode run summary and agent-tick budget conversion

## Outcome

- The generation driver `run_generations` (from 0501) records a `RunSummary` as a
  read-only measurement when a summary is requested. With no summary requested it
  runs the same kernel, governor and DB calls in the same order as before.
- `max_generations_for_agent_ticks` converts an agent-tick budget into the
  `max_generations` value that makes the generation loop run exactly that budget.
- The 0501 `GenerationSummaryUnsupported` refusal is removed.

## Scope

- `headless.rs`: the measurement sites, the summary write, the converter, and the
  `HeadlessError` change below.
- `tests/generation_run_summary.rs` (new): a fail-closed GPU helper copied from
  `gpu_adapter_present_when_required` in `tests/integration.rs`, named consts (4
  agents, tick budget 2,000), and the five tests below.

## Non-goals

- No change to `governor.rs`, fitness, selection, inheritance or the supervisor
  path. No CLI (`main.rs`), no GUI, no new fields on `RunSummary`.

## Prerequisites

- headless-supervisor-driver: `HeadlessOptions { run_summary }`,
  `run_headless_with_options`, private `run_generations`, `HeadlessError` with
  `Summary` and `GenerationSummaryUnsupported`.
- run-summary (0201): `RunSummaryAccumulator::new`, `begin_season`,
  `reset_all_baselines`, `record_physics_sample`, `finish`; `RunSummary`.
- Plan 0004: `WorldConfig.food_kinds_enabled` and `GpuKernel::nourishing_food_kind`.
- Baseline: `GpuKernel::cached_food_state` (`gpu_kernel.rs`), `Governor::evolution_complete`
  (`governor.rs`), `WorldConfig.seed`. A mismatch is a contradiction to report.

## Decisions

- `run_generations` keeps 0501's parameters and gains
  `summary: Option<&mut RunSummaryAccumulator>`. It returns `u64`: the ticks of
  completed generations (`tick_budget` added where a generation completes, next to
  `simulation_ticks`).
- Generation start, after the existing per-generation reset: `reset_all_baselines()`
  and `begin_season(generation_start_tick, kernel.nourishing_food_kind())`, where
  `generation_start_tick` is the returned counter before this generation.
- Chunk site, after the existing snapshot spin: `record_physics_sample(
  generation_start_tick + ticks_done, kernel.cached_state(),
  kernel.cached_food_state())`, with `ticks_done` already counting this chunk.
- No extra end-of-generation sample: the last chunk covers it.
- Keep these lines textually identical (the 0502 preflight greps them):
  `governor.record_within_life_sample(cumulative_food, cumulative_alive);`,
  `let fitness = governor.evaluate(&agents);`,
  `agents[i].death_count = state[base + P_DEATH_COUNT] as u32;`.
- Accumulator: `RunSummaryAccumulator::new(EvolutionMode::Generations, None,
  config.world.seed, population, config.world.food_kinds_enabled)`.
- `run_headless_with_options` with a summary requested: create the accumulator,
  call `run_generations`, then `finish(ticks)` (a `Result`; its error is propagated as a
  `HeadlessError`, never unwrapped), then write
  `serde_json::to_string_pretty` to the path (I/O and JSON errors through `Summary`).
- `HeadlessError`: remove `GenerationSummaryUnsupported`; add
  `InvalidAgentTickBudget` (Display).
- `pub fn max_generations_for_agent_ticks(budget, population, tick_budget) ->
  Result<u64, HeadlessError>`: `population` converted with `u64::try_from`;
  `product = population * tick_budget` by `checked_mul`; error
  `InvalidAgentTickBudget` if budget, population or tick_budget is 0, if
  `budget % product != 0`, or on overflow. Let `q = budget / product` (at least 1).
  Return `q.checked_add(1)`, erroring on overflow.
- Derivation (doc comment): the loop checks `evolution_complete()` at the top of
  each generation (`headless.rs` loop start), which is `max > 0 && generation + 1 >= max`
  (`governor.rs`). A fresh governor starts at `generation = 0`, and `advance`
  breeds (increments `generation`) only when it does not finish. So generation
  indices `0..max-1` run, which is `max - 1` generations. Hence `max = q + 1`.
  The GPU test confirms this empirically. Resumed runs are out of scope.

## Steps

**Steps:**
1. Replace `GenerationSummaryUnsupported` with `InvalidAgentTickBudget`. In
   `run_headless_with_options`, when a summary is requested, create the
   accumulator, call `run_generations` with `Some(&mut acc)`, then `finish` and write
   the JSON. Without a request, pass `None`.
2. Change `run_generations` to take the summary option and return the completed
   ticks, with the counter and the `generation_start_tick` as in Decisions. Do not
   change any other statement in the loop body.
3. Add the generation-start calls (`reset_all_baselines`, `begin_season`) and the
   chunk-site `record_physics_sample` as in Decisions.
4. Implement `max_generations_for_agent_ticks` with the checks and derivation doc
   comment in Decisions.
5. Create `tests/generation_run_summary.rs` with the GPU helper, the consts, and
   these tests: `gpu_guard_fails_closed_when_required` (CPU);
   `max_generations_for_agent_ticks_rejects_bad_budgets` (CPU: zero, non-multiple,
   overflow); `generation_summary_matches_agent_results` (`nourishing_meals` equals
   the sum of `agent_result.food_consumed` and `deaths` the sum of
   `agent_result.death_count` over the run's nodes; `agent_ticks == generations *
   tick_budget * population`; `seasons == generations`);
   `generation_budget_runs_exact_agent_ticks` (budget = 3 generations' worth,
   `governor.max_generations` from the converter, `agent_ticks == budget`, exactly 3
   generations of `agent_result` rows); `generation_summary_kinds_follow_each_generation`
   (food kinds and second odour sense on: `kinds` is `Some`, `seasons == generations`,
   early and late meal counts at most the totals).

- **Done when:** the checks focused-proof (all five named tests report ok) and
  seeded-digest-determinism pass on the host, proving INV-16 (the summary equals the
  `agent_result` sums and has one season per generation), INV-12 (the converter
  yields the exact agent-tick budget) and INV-01 (with no summary requested the
  seeded headless run gives the same digest twice, and the no-summary path runs
  the same kernel, governor and DB calls in the same order as Outcome and Steps 2
  require; this task does not require the digest to equal the recorded fixture,
  whose value is adapter-specific and was recorded on a different adapter at 0976448).

## Evidence and risks

- The worker has no shell; the host-run checks named in the frontmatter are the
  proof. The derivation is checked by `generation_budget_runs_exact_agent_ticks`.
- Risk: an off-by-one in the `max = q + 1` derivation. If the GPU test shows a
  different count, report the measured count and the loop lines; do not change
  the formula silently.
- Risk: if `tests/supervisor_headless.rs` (0501, not in touches) names
  `GenerationSummaryUnsupported`, removing it breaks that file. Report it as a
  contradiction with file and line.
- Risk: the seeded digest depends on the moved loop body staying byte-identical
  apart from the measurement calls.
