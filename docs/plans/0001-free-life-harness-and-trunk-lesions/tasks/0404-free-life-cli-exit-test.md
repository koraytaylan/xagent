---
id: free-life-cli-exit-test
title: Prove the built xagent binary exits non-zero with a stderr message on a bad free-life agent count
workstream: "0400"
kind: task
role: developer
depends_on: [free-life-cli]
gated: false
touches: [crates/xagent-sandbox/tests/free_life_cli.rs]
status: done
merged_as: "76cfbbb3837e829e197e4435fdfd0f6b3838d162"
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --test free_life_cli"
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

# Prove the built xagent binary exits non-zero on a bad free-life agent count

## Outcome

The new integration test `crates/xagent-sandbox/tests/free_life_cli.rs` runs the
built `xagent` binary with `--free-life --free-life-agents 0` and asserts a
non-zero exit (status 1) and the agent-range message on stderr. It closes the
exit-status half of INV-9 ("CLI (free-life-cli)" in ARCHITECTURE.md), which the
bin unit tests of `free-life-cli` cannot observe because they call the pure
option builder. No other file changes.

## Decisions

1. **Binary.** The package's `[[bin]]` target is named `xagent`, so Cargo sets
   `CARGO_BIN_EXE_xagent` for integration tests; run it with
   `std::process::Command::new(env!("CARGO_BIN_EXE_xagent"))`. Do not build or
   locate the binary any other way.
2. **Invocation.** Arguments exactly `--free-life`, `--free-life-agents`, `0`.
   Set the current directory to a `tempfile::TempDir` (already a
   dev-dependency) so no stray file lands in the checkout, and collect with
   `.output()`.
3. **Assertions.** `!output.status.success()`; `output.status.code() ==
   Some(1)`; and `String::from_utf8_lossy(&output.stderr)` contains
   `format!("--free-life-agents must be in 1..={}", xagent_sandbox::agent::MAX_AGENTS)`,
   the text `free-life-cli` fixes in its Decision 2. Every assertion message
   prints the captured stderr. Stdout is not asserted (the startup banner goes
   to stderr, and the option builder rejects before any report).
4. **No GPU.** The option builder rejects the count before the runner starts,
   so the test needs no adapter and never skips.
5. **Style.** One test, `free_life_zero_agents_exits_non_zero_with_message`,
   with a `///` doc comment saying the built binary rejects a zero agent count.
   No `unwrap` outside `expect` with a message, no plan, task or workstream name
   or `000N-` slug in the file, rustfmt-clean.

## Steps

**Steps:**

1. Create `crates/xagent-sandbox/tests/free_life_cli.rs` with the module doc
   comment and the one test of Decision 5.
2. Build the command per Decisions 1-2 and the assertions per Decision 3.
3. Keep the file free of forbidden names and lint-clean (Decision 5); touch no
   other file. If the binary's stderr lacks the exact message, report a defect
   in `free-life-cli` instead of loosening the assertion.

- **Done when:** the focused-proof check (`cargo test -p xagent-sandbox --test free_life_cli`) reports `free_life_zero_agents_exits_non_zero_with_message` `ok`: the built binary run with `--free-life --free-life-agents 0` exits with status 1 and prints `--free-life-agents must be in 1..=` followed by `MAX_AGENTS` on stderr, proving the non-zero-exit-with-message part of INV-9.

## Non-goals

- No change to `main.rs`, the runner or any other file.
- No GPU use; no other flag combination.
