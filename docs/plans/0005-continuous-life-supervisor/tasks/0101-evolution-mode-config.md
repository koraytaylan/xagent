---
id: evolution-mode-config
title: "Evolution mode and supervisor config"
workstream: "0100"
kind: task
role: developer
depends_on: []
gated: false
touches:
  - crates/xagent-shared/src/config.rs
  - crates/xagent-shared/src/lib.rs
  - crates/xagent-sandbox/src/governor.rs
  - crates/xagent-sandbox/tests/integration.rs
  - crates/xagent-sandbox/tests/pathway_route_state.rs
status: done
merged_as: "d3cd74aa775c5c25bc3c4a48eee04e207da37255"
checks:
  - name: focused-proof
    command: "cargo test -p xagent-shared --lib supervisor_config_"
  - name: governor-tests
    command: "cargo test -p xagent-sandbox --lib governor::tests::"
    timeout_ms: 3600000
  - name: crates-clippy
    command: "cargo clippy -p xagent-shared -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: crates-fmt
    command: "rustfmt --check --edition 2021 crates/xagent-shared/src/lib.rs crates/xagent-sandbox/src/governor.rs crates/xagent-sandbox/tests/integration.rs crates/xagent-sandbox/tests/pathway_route_state.rs"
    fix: "rustfmt --edition 2021 crates/xagent-shared/src/lib.rs crates/xagent-sandbox/src/governor.rs crates/xagent-sandbox/tests/integration.rs crates/xagent-sandbox/tests/pathway_route_state.rs"
  - name: sandbox-guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
---

# Evolution mode and supervisor config

## Outcome

`GovernorConfig` gains serde-defaulted `evolution_mode` and `supervisor` fields
with the types, defaults and validation in ARCHITECTURE.md "Interfaces
(binding)" → "Config". Omitted keys reproduce today's behaviour. No run table,
database, resume or supervisor logic changes.

## Scope

- `crates/xagent-shared/src/config.rs`:
  - `EvolutionMode` and `Inheritance` (derives and serde casing per
    ARCHITECTURE.md; `Generations` and `Darwinian` are `#[default]`).
  - `SUPERVISOR_SAMPLE_TICKS = 100` and `MAX_ARCHIVE_CAPACITY = 1024`, each with a
    `///` doc.
  - Named default consts, each with a `///` doc stating its reason:
    `DEFAULT_BOUNDARY_INTERVAL_TICKS` (4_000), `DEFAULT_MIN_AGE_TICKS` (20_000),
    `DEFAULT_SCORE_WINDOW_TICKS` (10_000), `DEFAULT_ARCHIVE_CAPACITY` (32),
    `DEFAULT_SEASON_PERIOD_TICKS` (40_000), `DEFAULT_AGENT_TICK_BUDGET` (0).
  - `SupervisorConfig` with the seven fields of the ARCHITECTURE.md table. Each
    field has `#[serde(default = "...")]` naming a function that returns its
    const. `impl Default` returns the same values.
  - `SupervisorConfigError` (`Debug`, `Display`, `std::error::Error`), one
    variant per rule: `AgentCountTooSmall { agent_count: usize }`,
    `BoundaryIntervalInvalid`, `SeasonPeriodInvalid`, `ScoreWindowInvalid`,
    `MinAgeBelowWindow`, `ArchiveCapacityOutOfRange { archive_capacity: usize }`,
    `AgentTickBudgetNotAligned`, `AgentTickBudgetOverflow`.
  - `SupervisorConfig::validate(&self, agent_count: usize) -> Result<(),
    SupervisorConfigError>` with a `///` doc, rules as in ARCHITECTURE.md.
  - `GovernorConfig` gains `#[serde(default)] pub evolution_mode` and
    `#[serde(default)] pub supervisor`; its existing `impl Default` fills both.
    No other field or default changes. `BrainConfig` is untouched.
- `crates/xagent-shared/src/lib.rs`: add `EvolutionMode`, `Inheritance`,
  `SupervisorConfig`, `SupervisorConfigError`, `SUPERVISOR_SAMPLE_TICKS` and
  `MAX_ARCHIVE_CAPACITY` to the existing `pub use config::{ ... };` list, in
  sorted position.
- `crates/xagent-sandbox/src/governor.rs`: in the tests module only, append
  `..GovernorConfig::default()` as the last field of every full
  `GovernorConfig {` literal (about 20; search the file). Literals already ending
  in `..Default::default()` stay. No other change.
- `crates/xagent-sandbox/tests/integration.rs`: the full literal
  `let gov_config = GovernorConfig {` (about line 7416) gets
  `..GovernorConfig::default()` as its last field. The literal near line 10011
  already ends in `..Default::default()` and stays.
- `crates/xagent-sandbox/tests/pathway_route_state.rs`: the full literal in
  `fn governor_config()` (about line 39) gets `..GovernorConfig::default()` as
  its last field. No other change.
- crates-fmt formats only the touched files (`lib.rs` pulls in `config.rs`).
  `src/bin/food_kind_verdict.rs` and `src/food_kind_metrics.rs` are already
  unformatted at the baseline; they are out of scope and stay unedited.

## Non-goals

- No run-table `mode` column, resume refusal, visibility change or grid-size
  change (task `run-mode-column`).
- No supervisor logic, CLI flags or preset.
- No change to existing `GovernorConfig` fields or defaults, or to `BrainConfig`.
- No edit to this packet's checks or touches.

## Prerequisites

None inside this plan. `GovernorConfig`, its `impl Default` and the re-export
list already exist at the baseline in `config.rs` and `lib.rs`.

## Decisions

- Shapes and serde casing follow ARCHITECTURE.md exactly.
- `validate` returns the first failing rule in ARCHITECTURE.md order.
  `agent_tick_budget == 0` passes for any agent count. The budget rule uses
  `checked_mul`; overflow returns `AgentTickBudgetOverflow`.
- Numeric literals other than 0 and 1 in non-test code use the named consts. No
  new lossy `as` cast.
- Comments, doc strings and test names carry no plan, task or workstream
  identifier.

## Steps

**Steps:**
1. In `config.rs`, add `EvolutionMode`, `Inheritance`, the two public consts, the
   six default consts with reason docs, the serde default functions,
   `SupervisorConfig` and its `impl Default`.
2. In `config.rs`, add `SupervisorConfigError` with `Display` and `Error`,
   `SupervisorConfig::validate`, and the two new `GovernorConfig` fields with the
   `impl Default` extension.
3. In `lib.rs`, add the six new names to the `pub use config::{ ... };` list.
4. In the `config.rs` tests module, add the five `supervisor_config_` tests listed
   under Tests.
5. In the `governor.rs` tests module and in `tests/integration.rs`, add
   `..GovernorConfig::default()` to every full literal listed under Scope,
   including the one in `tests/pathway_route_state.rs`.
6. Fix any failure the host reports, in the touched files only.

## Tests

Config tests module in `config.rs`, one line per test:

- `supervisor_config_defaults_when_omitted`: `FullConfig` from `{}` and a
  `GovernorConfig` JSON without the new keys give `EvolutionMode::Generations`
  and `SupervisorConfig::default()`.
- `supervisor_config_round_trip`: `EvolutionMode::Supervisor`,
  `Inheritance::WeightCopy` and other changed fields survive serde_json, and the
  output contains `"supervisor"` and `"weight_copy"`.
- `supervisor_config_validate_rejects_each_rule`: one failing case per error
  variant asserting the matching variant, plus one passing case.
- `supervisor_config_default_is_valid_for_default_population`: the default config
  validates for the existing `default_population_size()` (10).
- `supervisor_config_not_in_brain_config`: serialised `BrainConfig::default()`
  has no `supervisor`, `evolution_mode` or `inheritance` key and no
  `SupervisorConfig` field name.

## Done when

- **Done when:** the checks focused-proof (all five supervisor_config_ tests pass) and governor-tests (the whole governor test module passes with every edited literal compiling) pass, proving INV-01 (omitted keys give Generations and the default SupervisorConfig) and INV-02 (no supervisor key in BrainConfig, and validate rejects each infeasible rule).

## Evidence and risks

- The worker has no shell. Proof is the host-run checks named above; the worker
  only reads and edits.
- If a full `GovernorConfig {` literal exists outside the touched files, or
  `FullConfig` or `GovernorConfig` lacks a serde trait the tests need, report the
  file and line. Do not edit outside touches or change other fields. At planning,
  only `governor.rs`, `tests/integration.rs` and `tests/pathway_route_state.rs`
  hold full literals (`tests/sense_block_parity.rs` already ends in
  `..Default::default()`).
