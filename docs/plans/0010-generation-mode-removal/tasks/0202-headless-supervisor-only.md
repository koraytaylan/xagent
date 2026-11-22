---
id: headless-supervisor-only
title: "Headless runs only the supervisor and refuses legacy databases"
workstream: "0200"
kind: task
depends_on:
  - cli-supervisor-only
  - generation-test-files
  - sense-parity-supervisor
  - integration-tests
  - store-legacy-guard
gated: false
role: developer
touches:
  - crates/xagent-sandbox/src/headless.rs
  - crates/xagent-sandbox/src/run_summary.rs
  - crates/xagent-sandbox/src/supervisor_run.rs
  - crates/xagent-sandbox/tests/supervisor_headless.rs
  - crates/xagent-sandbox/tests/supervisor_run.rs
status: done
merged_as: "fbf33da583b1fb9cbc275235cdccb35de7034ad0"
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_headless --test supervisor_run"
    timeout_ms: 5400000
  - name: headless-unit
    command: "cargo test -p xagent-sandbox --lib headless::"
    timeout_ms: 3600000
  - name: summary-unit
    command: "cargo test -p xagent-sandbox --lib run_summary::"
    timeout_ms: 3600000
  - name: headless-generation-free
    command: |-
      sh -c '! grep -nE "EvolutionMode|evolution_mode|run_generations|max_generations|dump_tree|validate_danger_percept|validate_innate_instincts|remove_file|reset_database|AdvanceResult" crates/xagent-sandbox/src/headless.rs crates/xagent-sandbox/src/run_summary.rs crates/xagent-sandbox/src/supervisor_run.rs crates/xagent-sandbox/tests/supervisor_headless.rs crates/xagent-sandbox/tests/supervisor_run.rs'
  - name: all-targets-compile
    command: "cargo check --workspace --all-targets"
    timeout_ms: 3600000
---

# 0202 Headless runs only the supervisor and refuses legacy databases

## Outcome

`run_headless_with_options` always runs the supervisor driver. Before any store
open it classifies the database read-only with `supervisor_store::inspect_database`
and refuses a legacy generations database with the store's exact message, so
the file's bytes and its directory are unchanged. The generation driver, tree
dump, validation A/B code, bootstrap statistics and generation run summary are
deleted, and `mode` leaves `RunSummary`. No headless path deletes a database.
The champion birth-brain source is removed afterwards by
`headless-champion-source`.

This task runs after `cli-supervisor-only` (which removed the `main.rs` callers
of `dump_tree`, `validate_*` and `max_generations_for_agent_ticks`), after
`generation-test-files` and after `sense-parity-supervisor`. It does not touch
`governor.rs`, `main.rs`, the config types or the free-life test files.

## Decisions

- **Dispatch.** `run_headless_with_options(config, db_path, resume, options)`
  keeps its signature. The `match config.governor.evolution_mode` goes; the body
  is the supervisor path (`run_supervisor`). `resume` handling stays exactly as
  plan 0007 left it. `RunSummaryAccumulator::new` loses the mode argument.
- **Deleted in `headless.rs`.** `run_generations`, `resolve_inherited_brain`,
  `log_learning_metrics`, `dump_tree`, `max_generations_for_agent_ticks`,
  `validate_danger_percept`, `validate_innate_instincts`,
  `run_innate_instinct_ab`, `run_headless_with_flags`, `BootstrapMetric`,
  `ValidationStats`, `compute_bootstrap_metric`,
  `print_danger_percept_decision_rule`, `print_validation_metrics`, the bootstrap
  and validation constants (`CI_LOWER_PERCENTILE`, `CI_UPPER_PERCENTILE`,
  `FOOD_PER_DEATH_ZERO_GUARD`, `INSTINCT_*`, `STEERING_*`,
  `SURVIVAL_STABILITY_BAND`, `VALIDATION_HEATMAP_SAMPLES`) and every helper left
  unused (`compute_correlation`, `compute_regression`, `repeated_config`,
  `write_steering_population`, `HEATMAP_INTERVAL` when unused). The
  `crate::governor` imports only those used go (`compute_danger_dwell_fraction`,
  `AdvanceResult`, `Governor`); the champion imports stay for
  `headless-champion-source`. `EvolutionMode` leaves the `xagent_shared` import.
- **Legacy database.** At the top of `run_headless_with_options`, before any
  store open, call `supervisor_store::inspect_database(std::path::Path::new(db_path))`.
  `Missing`, `Empty` and `Supervisor` proceed; `LegacyGenerations` and an
  inspection `Err` return `Err(HeadlessError::Store(e))`, so the message is the
  store's `Display` text. `run_headless` prints it with `eprintln!` and exits
  with status 1. Headless never calls `remove_file`, `truncate`,
  `reset_database` or a writable open of a file it did not create.
- **Summary.** `RunSummary` and `RunSummaryAccumulator` lose the `mode` field and
  parameter; the JSON key `mode` is no longer written; every other field stays.
- **Tests.** Delete `generation_mode_is_the_default_driver`, its `GENERATION_*`
  constants and every `summary.mode` assertion; drop the `evolution_mode`
  assignment in `supervisor_headless.rs` and the `EvolutionMode` import in
  `tests/supervisor_run.rs`. Add `headless_refuses_legacy_generation_db` to
  `supervisor_headless.rs`: it builds with rusqlite raw SQL a file with
  `CREATE TABLE node (id INTEGER PRIMARY KEY, generation INTEGER NOT NULL)`
  holding rows up to generation 3670, records its bytes and directory entries,
  calls `run_headless_with_options` with a supervisor config, and asserts an
  `Err` whose `Display` contains `generation 3670` and `cannot run it`, identical
  bytes afterwards and no sibling file.

**Steps:**

1. In `headless.rs` delete the generation driver, validation A/B code, bootstrap
   statistics and helpers listed in Decisions, and make
   `run_headless_with_options` call only the supervisor path.
2. In `headless.rs` add the read-only `inspect_database` refusal before any store
   open and make `run_headless` print the error and exit with status 1.
3. In `run_summary.rs` and `supervisor_run.rs` drop the `mode` field and
   parameter and the `EvolutionMode` imports.
4. In `tests/supervisor_headless.rs` and `tests/supervisor_run.rs` make the edits
   in Decisions, including `headless_refuses_legacy_generation_db`.
5. Remove imports and helpers left unused, with no `#[allow(dead_code)]` and no
   planning reference in source.

- **Done when:** `focused-proof` passes with `headless_refuses_legacy_generation_db` and the other `supervisor_headless` and `supervisor_run` tests reported ok, and `headless-unit`, `summary-unit`, `headless-generation-free` and `all-targets-compile` pass, proving headless carries no generation driver (INV-1, INV-7), refuses a legacy database without changing it (INV-2) and deletes no database (INV-3).
