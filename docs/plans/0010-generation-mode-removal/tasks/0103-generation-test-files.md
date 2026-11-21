---
id: generation-test-files
title: "Delete the generation-only test files"
workstream: "0100"
kind: task
depends_on: []
gated: false
role: architect
status: todo
merged_as: ""
touches:
  - crates/xagent-sandbox/tests/seeded_headless_digest.rs
  - crates/xagent-sandbox/tests/fixtures/seeded_headless_digest.json
  - crates/xagent-sandbox/tests/generation_run_summary.rs
checks:
  - name: generation-tests-removed
    command: |-
      sh -c 'test ! -e crates/xagent-sandbox/tests/seeded_headless_digest.rs && test ! -e crates/xagent-sandbox/tests/fixtures/seeded_headless_digest.json && test ! -e crates/xagent-sandbox/tests/generation_run_summary.rs && test -f crates/xagent-sandbox/tests/supervisor_seeded_digest.rs'
  - name: all-targets-compile
    command: cargo check --workspace --all-targets
    timeout_ms: 3600000
---

# 0103 Delete the generation-only test files

## Scope

Delete exactly three files, and edit no other file:

- `crates/xagent-sandbox/tests/seeded_headless_digest.rs` (the generation digest
  test). Its replacement is the 0006 supervisor digest gate
  `crates/xagent-sandbox/tests/supervisor_seeded_digest.rs`, which is a PLAN
  check (`supervisor-seeded-digest`) and is not touched here.
- `crates/xagent-sandbox/tests/fixtures/seeded_headless_digest.json` (the
  generation digest fixture, read only by the file above).
- `crates/xagent-sandbox/tests/generation_run_summary.rs` (measures generation
  runs and `max_generations_for_agent_ticks`; the supervisor run summary stays
  covered by `supervisor_headless.rs` and `supervisor_run.rs`). The function
  itself is deleted later by `headless-supervisor-only`; this task only removes
  the test file that calls it.

Deletion is allowed only because the 0006 gate file exists. If
`crates/xagent-sandbox/tests/supervisor_seeded_digest.rs` is absent when this
task runs, or if the 0006 gate is named differently, stop and report a plan
correction for the `supervisor-seeded-digest` check. Do not delete anything in
that case.

Nothing under `crates/xagent-sandbox/src/` reads the digest fixture at baseline,
and Cargo auto-discovers test targets, so no manifest entry needs editing.

## Non-goals

- No edit of any file other than the three listed above, including
  `crates/xagent-sandbox/tests/integration.rs`, `contributing_guard.rs`, and
  `Cargo.toml`.
- No edit of `max_generations_for_agent_ticks` in `headless.rs` or of
  `main.rs`; those belong to `headless-supervisor-only` and `cli-supervisor-only`.
- No change to `docs/decisions/`, `docs/measurements/` or other plans.
- No `#[allow]`, `#[ignore]` or replacement test added to make a check pass.

## Invariants

- INV-6: the removal does not change a seeded supervisor run. The 0006 gate
  file must remain present and unmodified by this task.
- INV-7: no generation mode lives on in test-only form. The generation digest
  test, its fixture and the generation run-summary test are gone.

**Steps:**

1. Confirm `crates/xagent-sandbox/tests/supervisor_seeded_digest.rs` exists.
   Search `crates/xagent-sandbox/src/` for `seeded_headless_digest` and confirm
   there are no hits, so nothing in `src/` reads the fixture. If the gate
   is absent, stop and report a plan correction instead of deleting.
2. Delete the three touched files with `rm` on their exact repository-relative
   paths. Do not use a glob or delete the `fixtures/` directory.
3. Run the `generation-tests-removed` check and confirm it passes, proving the
   three files are gone and the 0006 gate file is still present.
4. Run the `all-targets-compile` check and confirm the workspace, including every
   test target, still compiles. If a removal causes a compile error elsewhere,
   stop and report it as a blocker; do not edit other files to fix it.
5. Confirm with `git status` that only the three touched paths changed in this
   task. Do not stage or commit; the host commits the packet.
6. Do not add any file, suppression or replacement test. Leave every other
   file byte-identical.

- **Done when:** the `generation-tests-removed` and `all-targets-compile` checks pass, proving the generation-only tests are gone (INV-7) while the supervisor digest gate that replaces them is present (INV-6).
