---
id: docs-supervisor-only
title: "Docs describe the supervisor-only system"
workstream: "0400"
kind: task
depends_on:
  - study-binaries
  - integration-tests
  - generation-test-files
  - sense-parity-supervisor
  - store-legacy-guard
  - worker-session-supervisor-only
  - ui-replay-supervisor-only
  - cli-supervisor-only
  - headless-supervisor-only
  - headless-champion-source
  - governor-removal
  - config-generation-fields
  - guard-baseline
gated: false
role: developer
touches:
  - README.md
  - CLAUDE.md
  - CONTRIBUTING.md
  - EVOLUTION_JOURNEY.md
  - crates/xagent-sandbox/README.md
  - crates/xagent-shared/README.md
status: todo
merged_as: ""
checks:
  - name: contributing-guard
    command: cargo test -p xagent-sandbox --test contributing_guard
    timeout_ms: 1800000
  - name: sandbox-tests
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox
    timeout_ms: 7200000
  - name: docs-generation-free
    command: |-
      sh -c '! grep -nE "governor\.rs|evolution state machine|--evolution-mode|--generations|--dump-tree|Previous Session" CLAUDE.md README.md CONTRIBUTING.md crates/xagent-sandbox/README.md'
---

# 0402 Docs describe the supervisor-only system

## Outcome

README.md, CLAUDE.md, CONTRIBUTING.md, EVOLUTION_JOURNEY.md and
`crates/xagent-sandbox/README.md` describe the continuous supervisor as the only
mode. The contributing-guard baseline was already tightened by `guard-baseline`.
This task runs last. Every code task it depends on
has already landed, so the file contents and the guard's printed baseline are
read from the tree as it is at execution time.

## Decisions

- **Removed flags are named without leading dashes in prose.** The
  `docs-generation-free` check forbids the literal strings `--generations`,
  `--dump-tree` and `--evolution-mode`. Docs therefore list removed flags as
  bare names, for example `generations`, `dump-tree`, `evolution-mode`,
  `validate-innate-instincts`, `validate-danger-percept`, `validation-*`,
  `free-life-champion-db`, `free-life-champion-node`. The sentence says these
  exit with status 2 and a message that generation mode was removed. Current
  flags are written with their real leading dashes, but only for flags that
  still exist.
- **Current flags come from the code.** Read the clap argument struct in
  `crates/xagent-sandbox/src/main.rs` as it stands after the GUI/CLI task. List
  only the non-hidden flags it defines. Do not copy the flag list from the old
  docs.
- **Supervisor description comes from the code.** Read the module doc comments
  of `crates/xagent-sandbox/src/supervisor.rs` and
  `crates/xagent-sandbox/src/supervisor_run.rs`. Describe the supervisor in the
  same terms. Do not invent new behaviour.
- **Legacy database wording.** State that a generation-era database (one that
  `supervisor_store::inspect_database` classifies as `LegacyGenerations`) is
  refused by the GUI and the headless driver, and that the file is never
  modified and no sidecar file is created beside it. Do not describe any
  migration or conversion.
- **Journal chapter.** The chapter cites the user's decision record
  `docs/measurements/continuous-supervisor/INHERITANCE-DEFAULT.md`, which states
  that the supervisor becomes the only mode. It does not restate or change the
  older `DECISION.md` verdict.
- **Non-generation wording stays.** Words such as "world generation",
  "mesh generation", "seed-based generation" and the changelog note about
  `cliff.toml` are not about generation mode and are left unchanged.

**Steps:**

1. In `CLAUDE.md`, replace the `governor.rs` architecture line (the line that
   reads "evolution state machine, SQLite persistence") with one line for each of
   `crates/xagent-sandbox/src/supervisor.rs`, `crates/xagent-sandbox/src/supervisor_run.rs`
   and `crates/xagent-sandbox/src/supervisor_store.rs`. The store line says it
   refuses a generation-era database (via `inspect_database`) before any write.
   Remove the phrases "enforces the generation tick budget" from the
   `sim_runtime.rs` line and "drives the generation handoff" from the `main.rs`
   line. Reword the `ui.rs` line that names `governor↔UI` to name the surviving
   observatory and supervisor snapshot. Update the test-count line
   (`cargo test -p xagent-sandbox` — runs N total tests (...)) from the
   `sandbox-tests` output. Count, for each target kind, the sum of `passed` plus
   `ignored` over its `test result:` lines: lib unit (`Running unittests src/lib.rs`),
   bin unit (`Running unittests src/main.rs`), integration (every
   `Running tests/<name>.rs` target together), and doctests (`Doc-tests`).
   Write the total and the four parts exactly, and correct the line until it equals the `sandbox-tests` totals.
2. In `README.md` and `crates/xagent-sandbox/README.md`, remove or rewrite every
   passage that describes generation mode: the generation search and selection
   sections, the generation tree and detail panels, the Reset and "Replay Gen N"
   controls, per-generation replay recording and playback, generation tick
   budgets and handoffs, and generation-era CLI flags. Describe the continuous
   supervisor as the only mode. Give the current CLI flags as read in Decisions.
   State that the removed flags (named as in Decisions) exit with status 2 and a
   message that generation mode was removed. State that a generation-era database
   is refused and never modified. If `crates/xagent-shared/README.md` documents
   any removed `GovernorConfig` field or `EvolutionMode`, correct it. If it does
   not, leave it unchanged.
3. In `CONTRIBUTING.md`, delete the rule list item that starts with
   `governor.rs — extract magic numbers` and renumber the list that contains it.
   Rewrite or delete any other sentence that states a generation-mode rule. Leave
   the SQLite `busy_timeout` and serde alias rules unchanged, since they still
   apply to the supervisor store.
4. In `EVOLUTION_JOURNEY.md`, keep chapters 1 to 24 and the other sections
   unchanged. Insert one new H3 chapter directly after chapter `### 24.`, before
   `## Current Parameter Roles`, titled `### 25. Generation mode removed (2026-11-21)`.
   It records that the Governor generation state machine, the generation
   database, the generation handoff and per-generation replay were removed; that
   the continuous supervisor is the only mode; that generation-era databases are
   refused and never modified; and that the decision is the user's, recorded in
   `docs/measurements/continuous-supervisor/INHERITANCE-DEFAULT.md`.
- **Done when:** the checks `contributing-guard`, `sandbox-tests` and `docs-generation-free` pass, and the CLAUDE.md test-count line equals the totals of the `sandbox-tests` output, proving the docs describe the supervisor-only system (INV-10).

## Non-goals

- No change to any `.rs` source file.
- No edit of history records under `docs/decisions/`, `docs/measurements/` or
  other plans, and no rewrite of EVOLUTION_JOURNEY chapters 1 to 24.
- No new dependency, Cargo file, release identifier or check beyond the three
  listed.
- No Git operation. The host commits the packet.
