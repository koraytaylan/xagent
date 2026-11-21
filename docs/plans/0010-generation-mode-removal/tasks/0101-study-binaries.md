---
id: study-binaries
title: "Delete the generation-era study binaries"
workstream: "0100"
kind: task
depends_on: []
gated: false
role: architect
status: todo
merged_as: ""
touches:
  - crates/xagent-sandbox/src/bin/food_kind_verdict.rs
  - crates/xagent-sandbox/src/bin/pathway_route_gate.rs
  - crates/xagent-sandbox/src/bin/pathway_return_fit.rs
  - crates/xagent-sandbox/src/bin/supervisor_verdict.rs
checks:
  - name: bins-removed
    command: sh -c 'test ! -e crates/xagent-sandbox/src/bin/food_kind_verdict.rs && test ! -e crates/xagent-sandbox/src/bin/pathway_route_gate.rs && test ! -e crates/xagent-sandbox/src/bin/pathway_return_fit.rs && test ! -e crates/xagent-sandbox/src/bin/supervisor_verdict.rs'
  - name: all-targets-compile
    command: cargo check --workspace --all-targets
    timeout_ms: 3600000
---

# 0101 Delete the generation-era study binaries

## Scope

Delete the four study binaries under `crates/xagent-sandbox/src/bin/`:
`food_kind_verdict.rs`, `pathway_route_gate.rs`, `pathway_return_fit.rs` and
`supervisor_verdict.rs`. Cargo auto-discovers `src/bin` targets and
`crates/xagent-sandbox/Cargo.toml` declares only the `xagent` bin
(`default-run = "xagent"`), so no manifest edit is needed.

Preconditions checked at baseline: no test or source under `crates/` runs these
bins through `CARGO_BIN_EXE_*`. The only textual matches are the bins' own
files, and the `CARGO_BIN_EXE_xagent` uses in tests, which stay.

## Non-goals

- No edit of `docs/decisions/` or `docs/measurements/`; those records stay as
  history.
- No edit of `Cargo.toml` or any other file.
- Library code that these bins used and that becomes unused is removed by its
  owning later tasks (`headless-supervisor-only`, `governor-removal`), not here.
- No `#[allow(dead_code)]` anywhere.

## Invariants

- INV-1: no generation-era study binary remains under `crates/`.

**Steps:**

1. Search under `crates/` for `CARGO_BIN_EXE_` and the four bin names
   (`food_kind_verdict`, `pathway_route_gate`, `pathway_return_fit`,
   `supervisor_verdict`); the only matches must be inside the four bin files and
   the `CARGO_BIN_EXE_xagent` uses. Any other reference is a blocker to report.
2. With shell `rm`, delete the four files listed in `touches`, leaving
   `crates/xagent-sandbox/Cargo.toml` and every other file unchanged and adding
   no `#[allow(dead_code)]` or other suppression.

- **Done when:** the `bins-removed` and `all-targets-compile` checks pass, proving the study binaries that drove generation comparisons are gone (INV-1).
