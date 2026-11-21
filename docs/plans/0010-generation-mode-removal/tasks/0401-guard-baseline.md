---
id: guard-baseline
title: "Contributing-guard baseline tightened to the supervisor-only tree"
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
gated: false
role: developer
touches:
  - crates/xagent-sandbox/tests/contributing_guard.rs
status: todo
merged_as: ""
checks:
  - name: contributing-guard
    command: cargo test -p xagent-sandbox --test contributing_guard
    timeout_ms: 1800000
  - name: guard-governor-row-gone
    command: |-
      sh -c '! grep -qF "crates/xagent-sandbox/src/governor.rs" crates/xagent-sandbox/tests/contributing_guard.rs'
---

# 0401 Contributing-guard baseline tightened to the supervisor-only tree

## Outcome

`PLANNING_REFERENCE_BASELINE` in
`crates/xagent-sandbox/tests/contributing_guard.rs` matches the tree left by the
code tasks: the `governor.rs` row is gone, every row whose count fell is
lowered and every row whose file has no match any more is deleted.

## Decisions

- Never raise a number and never add a row; the test code, its comments and
  the scanned-file rules stay unchanged.
- Counts are read from the tree as it is at execution time with a content
  search over `crates/**/*.rs` and `crates/**/*.wgsl`, excluding
  `contributing_guard.rs`, for the pattern
  `(?i)workstream|status\.md|tasks\.md|docs/superpowers|plan 0[0-9]{3}|000[0-9]-[a-z]`.
  Markdown files are not scanned by the guard. When the `contributing-guard`
  check prints a replacement table, that table wins over the search.

**Steps:**

1. Search the tree as in Decisions and record the per-file counts.
2. In `PLANNING_REFERENCE_BASELINE` delete the `governor.rs` row and any row
   whose file is gone or has no match, and lower every row whose count fell,
   using the printed table when the check prints one.

- **Done when:** `contributing-guard` and `guard-governor-row-gone` pass, proving the contributing-guard baseline matches the supervisor-only tree (INV-10).
