---
id: evolution-cli
title: "Evolution-mode, budget, inheritance, summary and archive CLI flags"
workstream: "0600"
kind: task
role: developer
depends_on: [archive-brain-source]
gated: false
touches: [crates/xagent-sandbox/src/main.rs, crates/xagent-sandbox/tests/evolution_cli_exit.rs]
status: done
merged_as: "5a495efd0787cb96e71ef331e5111305fb4889ea"
checks:
  - {name: focused-proof, command: "cargo test -p xagent-sandbox --bin xagent -- evolution_cli_"}
  - {name: exit-test, command: "cargo test -p xagent-sandbox --test evolution_cli_exit", timeout_ms: 3600000}
  - {name: sandbox-clippy, command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings", timeout_ms: 3600000}
  - {name: sandbox-fmt, command: "cargo fmt -p xagent-sandbox -- --check", fix: "cargo fmt -p xagent-sandbox"}
  - {name: sandbox-guard, command: "cargo test -p xagent-sandbox --test contributing_guard"}
---

# Evolution-mode, budget, inheritance, summary and archive CLI flags

## Outcome

`crates/xagent-sandbox/src/main.rs` gains the evolution flags and dispatches
headless runs through the mode-aware driver. With no new flag the configuration
and the default CLI path are unchanged. Invalid combinations exit 1 before any
kernel or database is opened.

## Scope

- `main.rs` only, plus the new integration test file.
- Flags, each `#[arg(long)]` with a `///` help line:
  - `--evolution-mode <generations|supervisor>`: local clap `ValueEnum`
    `EvolutionModeArg`, mapped to the config's `EvolutionMode`. Absent = keep config.
  - `--agent-tick-budget N`: `Option<u64>`.
  - `--supervisor-inheritance <darwinian|weight-copy>`: local `ValueEnum`
    `InheritanceArg`, mapped to the config's inheritance value. Absent = keep config.
  - `--run-summary PATH`: `Option<PathBuf>`.
  - `--free-life-archive-db PATH`: `Option<String>`.
  - `--free-life-archive-rank N`: `Option<usize>`, `requires` the archive db,
    `conflicts_with` the 0001 `--free-life-champion-db` argument. Missing = rank 0.
- Pure `fn apply_evolution_cli(cli: &Cli, config: &mut FullConfig) -> Result<(), String>`:
  - Sets mode and inheritance when the flags are given.
  - Budget in supervisor mode: sets `config.governor.supervisor.agent_tick_budget`,
    then `validate(population_size)`; the error text is the `SupervisorConfigError`
    text, which must name the agent-tick budget.
  - Budget in generation mode: error if `--generations` was also given; otherwise
    `config.governor.max_generations = headless::max_generations_for_agent_ticks(budget, population_size, tick_budget)?`
    (map its `HeadlessError` to `String` with `Display`).
  - Effective mode supervisor with `--resume`: error.
  - `--supervisor-inheritance` with the effective mode generations: error.
- Call `apply_evolution_cli` right after the `--generations` override in `main`,
  before the `--dump-config` branch.
- `--no-render` branch: `headless::run_headless_with_options(config, &cli.db, cli.resume, &HeadlessOptions { run_summary: cli.run_summary.clone() })`.
  On `Err`, print `Error: <message>` to stderr and exit 1.
- `free_life_options_from_cli` (0001) maps the archive flags to
  `BrainSource::Archive { db_path, rank }`.

## Non-goals

- No GUI change; the mode travels in `config.governor`.
- No change to `headless.rs`, `config.rs` or any shared crate.
- No new config defaults.

## Prerequisites

- `archive-brain-source`: provides `run_headless_with_options`, `HeadlessOptions`,
  `max_generations_for_agent_ticks` and `BrainSource::Archive`.
- Already in `main.rs`: 0001 `--free-life*` flags and `free_life_options_from_cli`;
  0004 `--food-kinds` flags.

## Decisions

- Clap enums are local to `main.rs` so no shared crate changes.
- The archive rank is `Option` so `requires` fires only on explicit use.
- Budget conversion lives in `headless::max_generations_for_agent_ticks`; this task
  only calls it.

## Steps

**Steps:**

1. Add the six flags and the two local `ValueEnum` types with their mappings to `Cli`.
2. Implement `apply_evolution_cli` and its helpers exactly as in Scope.
3. Wire `main`: the call after the generations override, the `--no-render` branch
   through `run_headless_with_options`, and the archive mapping in
   `free_life_options_from_cli`.
4. Add the ten `evolution_cli_*` unit tests in the existing `mod tests`, parsing
   with `Cli::try_parse_from`.
5. Create `crates/xagent-sandbox/tests/evolution_cli_exit.rs` with the two exit tests
   using `env!("CARGO_BIN_EXE_xagent")`, `std::process::Command` and a `tempfile` db
   path; each asserts the db file was not created.

## Done when

- **Done when:** The `focused-proof` check (ten `evolution_cli_*` unit tests) and the `exit-test`
  check (two integration tests) pass, proving INV-02 (CLI values validated before
  any run), INV-12 (mode selects the headless driver; budget conversion) and
  INV-01 (no new flag leaves config unchanged); `sandbox-clippy` and `sandbox-fmt`
  stay green.

## Evidence and risks

- Unit tests named: `evolution_cli_defaults_leave_config_unchanged`,
  `evolution_cli_mode_and_inheritance_set_config`,
  `evolution_cli_supervisor_budget_sets_and_validates`,
  `evolution_cli_generation_budget_sets_max_generations`,
  `evolution_cli_generation_budget_conflicts_with_generations`,
  `evolution_cli_resume_with_supervisor_is_error`,
  `evolution_cli_inheritance_with_generations_is_error`,
  `evolution_cli_archive_flags_build_archive_source`,
  `evolution_cli_archive_rank_requires_db`,
  `evolution_cli_archive_conflicts_with_champion`.
- Exit tests named: `evolution_cli_exit_resume_supervisor` (`--no-render --evolution-mode supervisor --resume --db <tmp>`; status 1, stderr contains `Error:` and `supervisor`) and `evolution_cli_exit_bad_budget` (`--no-render --evolution-mode supervisor --agent-tick-budget 7 --db <tmp>`; status 1, stderr names the budget rule).
- Both exit tests need no GPU: each fails before any kernel is created.
- Risk: the 0001 champion argument's field name must match the existing `Cli`
  struct when `conflicts_with` is added; the worker reads it from `main.rs`.
