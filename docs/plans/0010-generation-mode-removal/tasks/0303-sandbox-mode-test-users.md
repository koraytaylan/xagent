---
id: sandbox-mode-test-users
title: "Drop EvolutionMode from the supervisor sandbox tests"
workstream: "0300"
kind: task
depends_on:
  - governor-removal
gated: false
role: developer
touches:
  - crates/xagent-sandbox/tests/sense_block_parity.rs
  - crates/xagent-sandbox/tests/supervisor_digest.rs
  - crates/xagent-sandbox/tests/supervisor_resume_digest.rs
  - crates/xagent-sandbox/tests/supervisor_run_resume.rs
  - crates/xagent-sandbox/tests/supervisor_sense_parity.rs
  - crates/xagent-sandbox/tests/supervisor_tick_window_digest.rs
status: done
merged_as: "f9b9645c5650cfbec3be9d96c09c3c0271d0de79"
checks:
  - name: mode-test-users-free
    command: "sh -c 'cd crates/xagent-sandbox/tests && ! grep -nE \"EvolutionMode|evolution_mode\" sense_block_parity.rs supervisor_digest.rs supervisor_resume_digest.rs supervisor_run_resume.rs supervisor_sense_parity.rs supervisor_tick_window_digest.rs'"
  - name: all-targets-compile
    command: cargo check --workspace --all-targets
    timeout_ms: 3600000
---

# 0303 Drop EvolutionMode from the supervisor sandbox tests

## Outcome

The six supervisor tests under `crates/xagent-sandbox/tests/` no longer name
`EvolutionMode` or set `config.governor.evolution_mode`, so
`config-generation-fields` can delete both without breaking the workspace.
No earlier task owns these lines. After `governor-removal` the supervisor is
the only driver and nothing in the sandbox reads `evolution_mode`, so the
tests behave exactly as before.

## Decisions

- In each of `sense_block_parity.rs`, `supervisor_digest.rs`,
  `supervisor_resume_digest.rs`, `supervisor_run_resume.rs`,
  `supervisor_sense_parity.rs` and `supervisor_tick_window_digest.rs`, delete
  the line `config.governor.evolution_mode = EvolutionMode::Supervisor;` and
  remove `EvolutionMode` from the `use xagent_shared::{...}` import, reflowed as
  `cargo fmt` would (a single remaining name loses its braces).
- No other line changes. Goldens, fixtures, seeds and assertions stay.

**Steps:**

1. Make the two edits above in each of the six files.
2. Run `cargo fmt --all`, then the checks `mode-test-users-free` and
   `all-targets-compile` on the host.

- **Done when:** `mode-test-users-free` exits 0 and `all-targets-compile`
  finishes with no error, proving the tests no longer depend on `EvolutionMode`
  (INV-1) while still compiling against the current config.

## Non-goals

- No change to `crates/xagent-shared`, to test behaviour or to any other file.
