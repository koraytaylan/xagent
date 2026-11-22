---
id: sandbox-governor-default-users
title: "Stop reading removed GovernorConfig defaults in ui.rs and agent docs"
workstream: "0300"
kind: task
depends_on:
  - governor-removal
gated: false
role: developer
touches:
  - crates/xagent-sandbox/src/ui.rs
  - crates/xagent-sandbox/src/agent/mod.rs
status: done
merged_as: "2a07d96c9f7223a37d0bb1896910299354be486e"
checks:
  - name: governor-default-users-free
    command: "sh -c '! grep -nF \"GovernorConfig::default().tick_budget\" crates/xagent-sandbox/src/ui.rs && grep -qF \"const DEFAULT_SNAPSHOT_TICK_BUDGET: u64 = 1_000_000;\" crates/xagent-sandbox/src/ui.rs && ! grep -nF \"eval_repeats\" crates/xagent-sandbox/src/agent/mod.rs'"
  - name: ui-unit
    command: "cargo test -p xagent-sandbox --lib ui::"
    timeout_ms: 3600000
  - name: all-targets-compile
    command: cargo check --workspace --all-targets
    timeout_ms: 3600000
---

# 0304 Stop reading removed GovernorConfig defaults in ui.rs and agent docs

## Outcome

`impl Default for EvolutionSnapshot` in `crates/xagent-sandbox/src/ui.rs` no
longer reads `xagent_shared::GovernorConfig::default().tick_budget`, a field
`config-generation-fields` deletes; it uses a private constant with the same
value, so the default UI is unchanged. The `steering_population` doc comment in
`crates/xagent-sandbox/src/agent/mod.rs` no longer names the removed
`eval_repeats`. No earlier task owns these lines.

## Decisions

- **ui.rs.** Add, directly above `impl Default for EvolutionSnapshot`:

  ```rust
  /// Default `EvolutionSnapshot::tick_budget`: the value the snapshot took from
  /// the removed `GovernorConfig::tick_budget` default, kept so the default UI
  /// is unchanged.
  const DEFAULT_SNAPSHOT_TICK_BUDGET: u64 = 1_000_000;
  ```

  and set `tick_budget: DEFAULT_SNAPSHOT_TICK_BUDGET,` in that `Default`.
  Nothing else in `ui.rs` changes.
- **agent/mod.rs.** In the `steering_population` doc comment, replace
  "shares one [`mutate_steering_weights`] draw, matching / the way fitness is
  averaged over `eval_repeats` slots." with "shares one
  [`mutate_steering_weights`] draw, / so every agent in a group starts from the
  same mutated weights." No code changes.

**Steps:**

1. Make the two edits above.
2. Run `cargo fmt --all`, then the checks `governor-default-users-free`,
   `ui-unit` and `all-targets-compile` on the host.

- **Done when:** all three checks pass, proving the sandbox no longer reads a
  generation-only `GovernorConfig` field (INV-1) with the UI default unchanged.

## Non-goals

- No change to `crates/xagent-shared`, to UI behaviour or to any other file.
