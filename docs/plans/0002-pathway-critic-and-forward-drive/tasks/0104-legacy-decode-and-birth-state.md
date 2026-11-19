---
id: legacy-decode-and-birth-state
title: "Legacy blob decode and zero-at-birth proofs"
workstream: "0100"
kind: task
role: developer
depends_on:
  - brain-slot-layout
gated: false
status: todo
merged_as: ""
touches:
  - crates/xagent-sandbox/src/governor.rs
  - crates/xagent-sandbox/tests/pathway_route_state.rs
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test pathway_route_state"
    timeout_ms: 3600000
  - name: governor-unit
    command: "cargo test -p xagent-sandbox --lib governor"
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: "cargo fmt -p xagent-sandbox --bin xagent -- --check"
    fix: "cargo fmt -p xagent-sandbox"
  - name: sandbox-clippy
    command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: contributing-guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
    timeout_ms: 3600000
---

# Legacy blob decode and zero-at-birth proofs

## Outcome

`decode_brain_state` in `governor.rs` upgrades legacy-length brain blobs to the
current layout, so every champion read path sees current-length brains. A new
integration test file proves the legacy decode, that the pathway-route block is
zero at birth, and that mutation, config mutation and crossover never write the
block or change either route flag.

## Scope

- In `governor.rs`, `decode_brain_state` applies `upgrade_legacy_brain_state`
  (from `xagent_brain::buffers`, added by brain-slot-layout) to the decoded
  `brain_state`. `None` from the upgrade makes decode return `None`, so
  `champion_brain` and the read-only champion loader already log and skip a
  malformed blob. Extend the doc comment with: blobs of the previous layout
  gain a zero pathway-route block.
- A `#[cfg(test)]` unit test in the existing `governor.rs` tests module:
  `decode_brain_state` of an encoded legacy-length brain yields the upgraded
  length, the block is zero, and a truncated blob returns `None`.
- `tests/pathway_route_state.rs` (new), with the tests below.

## Non-goals

- No other change to `governor.rs`: no change to encode, storage schema,
  `champion_brain`, `store_champion_brain` or the read-only loader bodies.
- No change to `agent/mod.rs`, `config.rs`, `buffers.rs` or any kernel file.
  If a mutation or crossover function does not preserve the flags or the block,
  report it as a contradiction; do not patch it here.
- No new GPU requirement: the tests run on the CPU. `XAGENT_REQUIRE_GPU=1` is
  set only so the focused check follows the repository convention.

## Design

**Decode.** `decode_brain_state` keeps its header and payload checks, then
passes the decoded `brain_state` through `upgrade_legacy_brain_state`.
`encode_brain_state` is unchanged, so a stored legacy blob is written with its
legacy length and read back upgraded.

**Legacy fixture.** A legacy-length brain is a current `fresh_brain_state`
(from `xagent_sandbox::agent`) with the `PATHWAY_ROUTE_SLOT_COUNT` values
starting at the current `O_SENSORY_MEAN` offset removed. Those offsets are the
`pub` constants of `xagent_brain::buffers`; do not hard-code numbers.

**Integration tests** (`tests/pathway_route_state.rs`), each a `#[test]`:

1. `legacy_champion_loads_upgraded`: build a legacy-length brain, store it as
   a champion in a `tempfile` database through the public `Governor::new` and
   `store_champion_brain` path, read it back with `champion_brain`. Assert the
   length is current, the block is zero, and every other value equals the
   legacy input in order.
2. `current_length_blob_round_trips`: a current fresh brain stored and read
   back is equal to the input.
3. `truncated_blob_is_rejected`: write a truncated `node_brain` blob directly
   with `rusqlite` into the same database; `champion_brain` returns `None`.
4. `fresh_brain_has_zero_pathway_block`: the block is zero for the default
   config.
5. `brain_mutation_keeps_block_zero`: for seeds `0..256` call
   `mutate_brain_state_seeded` on a fresh brain and assert the block is zero.
6. `config_inheritance_keeps_route_flags`: for seeds `0..256`, `mutate_config_seeded`
   and `crossover_config` of parents with every `CriticInput` value and both
   `forward_pathway_enabled` values assert the child's `critic_input` and
   `forward_pathway_enabled` equal the parent's (for crossover, the first
   parent's).

## Steps

**Steps:**

1. In `governor.rs`, make `decode_brain_state` call `upgrade_legacy_brain_state`
   and return `None` when it does, and extend its doc comment.
2. In the existing `governor.rs` tests module, add the legacy-length decode unit
   test and the truncated-blob unit test described above.
3. Create `tests/pathway_route_state.rs` with tests 1 to 3 (legacy champion
   load, current-length round trip, truncated blob rejection) using a `tempfile`
   database.
4. In the same file, add tests 4 to 6 (fresh block zero, mutation keeps the
   block zero, config inheritance keeps both flags).
5. Format the two files, and make every new doc comment free of plan, task and
   workstream names.
6. Run each check of this task; fix only failures in these two files.

- **Done when:** `env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test pathway_route_state` passes the legacy-decode, zero-at-birth and flag-inheritance tests, with the governor-unit and contributing-guard checks green (proves INV-3, INV-4, INV-12).
