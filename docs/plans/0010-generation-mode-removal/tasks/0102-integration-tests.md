---
id: integration-tests
title: "Strip generation-mode tests from the integration suite"
workstream: "0100"
kind: task
depends_on: []
gated: false
role: developer
status: done
merged_as: "5055b8f6debc29196d341d0615a68f3f5260ed58"
touches:
  - crates/xagent-sandbox/tests/integration.rs
checks:
  - name: integration-suite
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration
    timeout_ms: 5400000
  - name: integration-generation-free
    command: |-
      sh -c '! grep -nE "\bGovernor\b|EvolutionMode|evolution_mode|GenerationBudgetReached|xagent_sandbox::governor|max_generations|elitism_count|eval_repeats|num_islands|migration_interval|reset_database|check_existing_session|generation_recording" crates/xagent-sandbox/tests/integration.rs'
  - name: all-targets-compile
    command: cargo check --workspace --all-targets
    timeout_ms: 3600000
---

# 0102 Strip generation-mode tests from the integration suite

## Scope

Edit only `crates/xagent-sandbox/tests/integration.rs`. Remove every test and
helper that serves only generation mode, and keep every test that guards
surviving behaviour (kernel, physics, brain, bench, probes, GPU telemetry,
supervisor-independent config and the UI members that survive).

Baseline inventory, by category. The worker re-derives this list from the
current file before editing; names are a guide, not a contract.

- Governor and generation SQL (delete whole test):
  `simulation_ticks_survive_resume` (uses `Governor::new`/`resume`),
  `async_recording_persists_and_round_trips` (uses `Governor`,
  `replay::GenerationRecording`, `store_recording`, `load_recording`),
  `recorded_telemetry_persists_in_agent_fitness` (builds a `GovernorConfig`
  literal with `tick_budget`, `elitism_count`, `patience`, `max_generations`,
  `eval_repeats`, `num_islands`, `migration_interval`, then `Governor::new`),
  `avoidance_counters_round_trip_to_agent_result` (opens `Governor` and queries
  the `agent_result` table).
- Generation handoff and UI (delete whole test):
  `evolution_snapshot_default_tree_pane_fraction` (asserts the removed evolution
  tree pane member of `ui::EvolutionSnapshot`), together with its
  `// ── UI Snapshot Tests` section header.
- Generation driver and validation A/B (delete whole test):
  `run_innate_instinct_ab_produces_valid_stats` (calls
  `headless::run_innate_instinct_ab` and builds a `FullConfig` literal).

Surviving tests that stay unchanged: `gpu_adapter_present_when_required` (kept
exactly, PLAN preflight and routes use it), `effort_telemetry_populates_during_generation`
(kernel-level telemetry; its `tick_budget` is a local variable, not a
`GovernorConfig` field), and all kernel, physics, brain, bench, probe and
hazard tests. No surviving test builds a `GovernorConfig` literal at baseline,
so the `GovernorConfig { population_size: .., ..GovernorConfig::default() }`
rewrite applies only if a surviving literal turns up while editing; a rewrite
uses only `population_size`, `mutation_strength`, `momentum_decay` and
`supervisor`.

## Non-goals

- No change to any other file, including `crates/xagent-sandbox/tests/contributing_guard.rs`;
  its baseline is updated by a later task.
- No weakening, skipping or `#[ignore]` of any surviving test; no
  `#[allow(...)]` added to silence a warning.
- No planning references (plan, task or workstream names) in the source.
- No product code change; the removed tests exercise surface that later tasks
  delete anyway.

## Invariants

- INV-1: the integration suite no longer names the removed generation surface
  (`Governor`, `EvolutionMode`, generation tables, generation-only
  `GovernorConfig` fields, `reset_database`, `check_existing_session`).
- INV-7: no generation mode survives in test-only form in the integration suite.

**Steps:**

1. Search `crates/xagent-sandbox/tests/integration.rs` for the removed-surface
   tokens (`Governor`, `GovernorConfig`, `EvolutionMode`, `evolution_mode`,
   `tick_budget`, `elitism_count`, `max_generations`, `patience`,
   `eval_repeats`, `num_islands`, `migration_interval`, `reset_database`,
   `check_existing_session`, `load_champion_read_only`, `generation_recording`,
   `xagent_sandbox::replay`, `xagent_sandbox::headless`, `ui::`, `FullConfig`).
   For each hit, confirm it lies inside one of the tests in the Scope inventory.
   `tick_budget` in a kernel test is a local variable and stays.
2. Delete each whole test listed under the three generation categories in Scope,
   with its doc comment and any section header that becomes empty. Delete a
   helper only when `cargo check` reports it unused after the deletions.
3. Rewrite any surviving `GovernorConfig` literal as
   `GovernorConfig { population_size: .., ..GovernorConfig::default() }` using
   only `population_size`, `mutation_strength`, `momentum_decay` and
   `supervisor`. At baseline there is none outside the deleted tests, so this
   step is expected to change nothing.
4. Remove the imports that become unused: `xagent_sandbox::headless` and
   `FullConfig` from the top of the file, and any local `use` lines that only
   deleted tests needed. Keep `gpu_adapter_present_when_required` byte-identical.
5. Run the `integration-generation-free` check and confirm it passes; then run
   `integration-suite` and confirm all surviving tests pass with
   `XAGENT_REQUIRE_GPU=1`; then run `all-targets-compile` and confirm no warnings
   from `integration.rs`.
6. Do not add `#[allow(dead_code)]`, `#[ignore]` or any suppression, and do not
   touch any file other than `crates/xagent-sandbox/tests/integration.rs`.

- **Done when:** the `integration-suite`, `integration-generation-free` and `all-targets-compile` checks pass, proving no generation mode survives in test-only form in the integration suite (INV-7) and its generation surface is gone (INV-1).
