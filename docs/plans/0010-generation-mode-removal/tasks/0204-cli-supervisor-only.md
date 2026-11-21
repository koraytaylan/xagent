---
id: cli-supervisor-only
title: "CLI refuses removed flags and logs dead config keys"
workstream: "0200"
kind: task
depends_on:
  - ui-replay-supervisor-only
gated: false
role: architect
touches:
  - crates/xagent-sandbox/src/main.rs
  - crates/xagent-sandbox/src/replay_coord.rs
  - crates/xagent-sandbox/tests/evolution_cli_exit.rs
status: todo
merged_as: ""
checks:
  - name: bin-unit
    command: "cargo test -p xagent-sandbox --bin xagent"
    timeout_ms: 3600000
  - name: cli-exit
    command: "cargo test -p xagent-sandbox --test evolution_cli_exit"
    timeout_ms: 3600000
  - name: cli-generation-free
    command: "sh -c 'test ! -e crates/xagent-sandbox/src/replay_coord.rs && ! grep -nE \"EvolutionMode|crate::governor|xagent_sandbox::governor|mod replay_coord|max_generations_for_agent_ticks|BrainSource::Champion|apply_evolution_cli\" crates/xagent-sandbox/src/main.rs'"
  - name: all-targets-compile
    command: cargo check --workspace --all-targets
    timeout_ms: 3600000
---

# 0204 CLI refuses removed flags and logs dead config keys

## Outcome

The command line carries no generation flag. Removed flags exit with status 2
before any database is created. Generation-era config keys are ignored and
logged once per load. The empty `replay_coord.rs` and its `mod` line go.

## Decisions

- **Removed flags.** Each becomes a hidden clap argument of the same long name
  that accepts any value (`String` for flags that took a value, `bool` for
  switches): `generations`, `dump_tree`, `evolution_mode`,
  `validate_innate_instincts`, `validate_danger_percept`,
  `validation_generations`, `validation_population`, `validation_replicates`,
  `validation_tick_budget`, `validation_seed_start`, `free_life_champion_db`,
  `free_life_champion_node`. Right after `Cli::parse()`, before the banner or any
  config work, `fn removed_flag(cli: &Cli) -> Option<&'static str>` names the
  first flag present in that order; main then prints
  `error: --<flag> was removed with generation mode; this version runs only the continuous supervisor`
  to stderr and exits with status 2.
- **Kept and deleted CLI logic.** `--resume` keeps the meaning plan 0007 gave
  it; delete only the generation-era rejection
  `--resume is not supported in supervisor mode`. Delete the
  `--supervisor-inheritance requires supervisor evolution mode` rejection, the
  generation branch and the `--generations` override (the remaining supervisor
  logic of `apply_evolution_cli` moves inline or into a function named
  `apply_supervisor_cli`), the `--agent-tick-budget conflicts with --generations`
  error, the `headless::dump_tree`, `validate_*` and
  `max_generations_for_agent_ticks` calls, `free_life_archive_rank`'s
  `conflicts_with = "free_life_champion_db"` and the `BrainSource::Champion` arm
  of `free_life_options_from_cli`. Keep the supervisor `agent_tick_budget` check
  so `--agent-tick-budget 7` still fails with exit 1 and an `agent_tick_budget`
  message. `main` keeps calling `headless::run_headless_with_options` with its
  current signature and does not set `evolution_mode`.
- **Config keys.** `resolve_config` parses the `--config` file once into
  `serde_json::Value` as well as `FullConfig`. Pure helper
  `fn ignored_governor_keys(json: &serde_json::Value) -> Vec<&'static str>` returns,
  in the order `tick_budget, elitism_count, max_generations, patience, eval_repeats, num_islands, migration_interval, evolution_mode`,
  the keys present under the `governor` object (empty for a non-object). When
  non-empty, `log::warn!` logs one line
  `ignored generation-era governor keys: <comma-separated list>` once per load.
- **Module.** Delete `mod replay_coord;` and, with shell `rm`, the file
  `crates/xagent-sandbox/src/replay_coord.rs` (one doc line after
  `ui-replay-supervisor-only`).
- **Tests file.** In `tests/evolution_cli_exit.rs` delete the `--resume` plus
  `--evolution-mode` test. Add `removed_generations_flag_exits_2`,
  `removed_dump_tree_flag_exits_2`, `removed_evolution_mode_flag_exits_2`,
  `removed_validation_flags_exit_2` and `removed_free_life_champion_flags_exit_2`,
  each running the binary with `--no-render --db <tempdir>/run.db` and the flag
  and asserting exit status 2, stderr naming the flag and containing
  `removed with generation mode`, and that the database does not exist. Keep
  `evolution_cli_exit_bad_budget` without `--evolution-mode` (exit 1, `Error:`,
  `agent_tick_budget`, no database).
- **Unit tests in `main.rs`:** `ignored_governor_keys_lists_dead_keys_once`
  (INV-4) and `removed_flag_names_the_first_removed_flag` (INV-5); rewrite the
  existing `main.rs` tests so none builds a generation config.

**Steps:**

1. In `main.rs` add the hidden removed-flag arguments and `removed_flag` with
   the early exit.
2. In `main.rs` delete the generation CLI logic listed in Decisions, keeping the
   supervisor budget check and the plan 0007 `--resume` path.
3. In `main.rs` add `ignored_governor_keys` and its one warning line in
   `resolve_config`, and the two named unit tests.
4. Delete `mod replay_coord;` from `main.rs` and remove
   `crates/xagent-sandbox/src/replay_coord.rs` with shell `rm`.
5. Rewrite `tests/evolution_cli_exit.rs` as in Decisions.

- **Done when:** `bin-unit` passes with `ignored_governor_keys_lists_dead_keys_once` and `removed_flag_names_the_first_removed_flag` reported ok, `cli-exit` passes with `removed_generations_flag_exits_2`, `removed_dump_tree_flag_exits_2`, `removed_evolution_mode_flag_exits_2`, `removed_validation_flags_exit_2`, `removed_free_life_champion_flags_exit_2` and `evolution_cli_exit_bad_budget` reported ok, and `cli-generation-free` and `all-targets-compile` pass, proving dead config keys are logged once (INV-4), removed flags exit 2 before any database exists (INV-5) and the CLI carries no generation surface (INV-1).
