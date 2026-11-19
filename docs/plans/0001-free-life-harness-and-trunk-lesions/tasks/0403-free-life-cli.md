---
id: free-life-cli
title: Expose the free-life harness on the xagent command line
workstream: "0400"
kind: task
role: developer
depends_on: [free-life-runner]
gated: false
touches: [crates/xagent-sandbox/src/main.rs]
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --bin xagent tests::free_life"
    timeout_ms: 3600000
  - name: bin-unit-tests
    command: "cargo test -p xagent-sandbox --bin xagent -- --skip sim_runtime::"
    timeout_ms: 3600000
  - name: cli-smoke
    command: "cargo run -p xagent-sandbox -- --free-life --free-life-agents 2 --free-life-ticks 1000 --seed 5"
    timeout_ms: 3600000
  - name: sandbox-guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
  - name: sandbox-clippy
    command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: "cargo fmt -p xagent-sandbox -- --check"
    fix: "cargo fmt -p xagent-sandbox"
---

# Expose the free-life harness on the xagent command line

## Outcome

`crates/xagent-sandbox/src/main.rs` gains the free-life flags, two helpers and
bin unit tests; no other file changes. The test that runs the built binary is
task `free-life-cli-exit-test`. The flag list, the pure option builder
signature and the consumed runner types are in ARCHITECTURE.md, sections
"CLI (free-life-cli)" and "Free-life run (free-life-runner)". `xagent
--free-life` runs `headless::run_free_life` on the seed already applied by
`resolve_config`, writes the pretty JSON report to stdout (or to
`--free-life-out` with a one-line stderr summary), and never prints the config
in this branch. Every bad input exits with status 1 after a stderr message;
the rules are INV-9 and the "CLI flags" row of "Untrusted inputs".

## Decisions

1. **Flags.** Add `use std::path::PathBuf;` and eight `Cli` fields, each with a
   `///` doc comment, after `generations`: `free_life` (mode switch),
   `free_life_agents` (default `headless::DEFAULT_FREE_LIFE_AGENTS`),
   `free_life_ticks` (default `headless::DEFAULT_FREE_LIFE_TICKS`),
   `free_life_champion_db: Option<String>`, `free_life_champion_node:
   Option<i64>`, `free_life_lesion: String` (default `none`),
   `free_life_log: Option<PathBuf>`, `free_life_out: Option<PathBuf>`. No new
   seed field; `--seed` is the free-life seed.
2. **Option builder.** `free_life_options_from_cli(cli, config)` is pure and
   checks, in order: agents outside `1..=MAX_AGENTS`, with the error text
   exactly `--free-life-agents must be in 1..={MAX_AGENTS}, got {N}` (the
   value formatted from `xagent_sandbox::agent::MAX_AGENTS`, not a literal);
   ticks `== 0`; a champion node without a database, with the text
   `--free-life-champion-node requires --free-life-champion-db`. It then calls
   `headless::parse_lesions`, mapping the error with `to_string()`. The source
   is `Champion { db_path, node_id }` when a database is given, else `Fresh`.
   The seed is `config.world.seed`. The batch-multiple check stays in the runner.
3. **Branch.** `run_free_life_cli(cli, config)` stays under about 50 lines and
   is called by `if cli.free_life { run_free_life_cli(&cli, &config); return; }`
   placed immediately before `print_config(&config);`. Option, runner,
   serialisation and I/O errors go to `eprintln!` and `std::process::exit(1)`.
   With `--free-life-out` the file is written and one stderr line gives the
   agent count (`report.agents.len()`) and the path; without it the JSON is
   `println!`ed. No `expect` or `unwrap` on I/O or serialisation.
4. **Bin tests.** Add these to the existing `tests` module, each named with the
   `free_life_` prefix, parsing with `Cli::try_parse_from(["xagent", ...])` and
   `FullConfig::default()` (the seed test uses `resolve_config`). Use `expect`
   and `assert!` only:
   - `free_life_defaults_are_ten_agents_300000_ticks_fresh_no_lesions`: only
     `--free-life` gives the defaults, `TrunkLesions::NONE`, `BrainSource::Fresh`
     and no log path.
   - `free_life_seed_flows_from_seed_flag`: `--seed 5` gives seed 5.
   - `free_life_lesion_list_is_parsed`: `turn-weights,critic` sets only those
     two; `all` sets all four.
   - `free_life_unknown_lesion_is_rejected`: `Critic` and `turn-weights,,critic`
     both fail.
   - `free_life_champion_node_without_db_is_rejected`: `--free-life-champion-node 3`
     alone fails.
   - `free_life_agent_count_outside_range_is_rejected`: `0` and `MAX_AGENTS + 1`
     fail, the `0` error equal to the exact Decision 2 text; `MAX_AGENTS` is
     accepted.
   - `free_life_zero_ticks_is_rejected`: `--free-life-ticks 0` fails.
   - `free_life_champion_db_and_node_map_to_champion_source`: database plus node
     3 gives `Champion` with `node_id: Some(3)`; database alone gives `None`.
5. **Style.** Existing tests and code stay unchanged. The file stays
   rustfmt-clean with trailing commas. No `TODO`, plan, task or workstream name,
   or `000N-` slug enters any source, comment or string; `main.rs` keeps its one
   tolerated planning reference and must not gain another.

## Steps

**Steps:**

1. In `crates/xagent-sandbox/src/main.rs`, add `use std::path::PathBuf;` and the
   eight `Cli` fields from Decision 1 after `generations`.
2. Add `free_life_options_from_cli` (Decision 2) and `run_free_life_cli`
   (Decision 3) beside the other helpers.
3. Insert the `if cli.free_life` branch immediately before `print_config(&config);`,
   leaving every other branch unchanged.
4. Add the eight `free_life_*` tests from Decision 4 to the existing `tests`
   module, importing `xagent_brain::buffers::TrunkLesions` inside it; keep the
   `speed_label` tests unchanged.
5. Make the file rustfmt-clean and free of forbidden names (Decision 5). Touch no
   other file.

- **Done when:** `cargo test -p xagent-sandbox --bin xagent tests::free_life` (the focused-proof check) reports all eight `free_life_*` tests `ok`, proving the input bounds of INV-9 (agents bounded to 1..=MAX_AGENTS with the exact range message, zero ticks rejected, unknown lesion names and a champion node without its database rejected); the binary's non-zero exit with a stderr message is proved by task `free-life-cli-exit-test`.

## Non-goals

- No change to `run_free_life`, `FreeLifeOptions`, the runner's validation order,
  `run_headless`, the governor, kernel, WGSL, metrics or tick-log modules, or any
  file other than `main.rs`.
- No new seed flag; no change to `--dump-config`, `--bench*`, `--validate-*`,
  the GUI paths or any default value.
- No setup checks, dependency installation or shell use; the host runs every
  check.
- No matrix, decision record or reference-run work.
