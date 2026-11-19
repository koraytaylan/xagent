---
id: read-only-champion-loader
title: Load a stored champion's birth brain from a governor database read-only
workstream: "0300"
kind: task
role: developer
depends_on: []
gated: false
touches: [crates/xagent-sandbox/src/governor.rs]
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --lib governor::tests::read_only_champion"
    timeout_ms: 3600000
  - name: governor-tests
    command: "cargo test -p xagent-sandbox --lib governor::tests"
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

# Load a stored champion's birth brain from a governor database read-only

## Outcome

- `crates/xagent-sandbox/src/governor.rs` gains the read-only loader specified
  in ARCHITECTURE.md, section "Read-only champion loader
  (read-only-champion-loader)". That section is binding for `StoredChampion`,
  `ChampionLoadError` (Debug, Display, Error) and `load_champion_read_only`.
- Only `governor.rs` changes. `Governor`, `init_schema`,
  `encode_brain_state` and `decode_brain_state` keep their behaviour.

## Decisions

- Open through a private helper `open_read_only` with
  `OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX` and no
  URI flags. The write-rejection test reuses this helper.
- Set `busy_timeout` from the new documented constant
  `CHAMPION_LOAD_BUSY_TIMEOUT: Duration = Duration::from_millis(5_000)`, which
  matches the governor's own 5000 ms. Do not refactor the inline
  `PRAGMA busy_timeout=5000` strings.
- Issue only `SELECT` statements. Never run `init_schema`,
  `NODE_BRAIN_SCHEMA`, `PRAGMA journal_mode`, or any ALTER, CREATE, INSERT,
  UPDATE or DELETE.
- Check table presence first through `sqlite_master`. A missing `node_brain`
  table means `NoStoredChampion` in both selection modes.
- Default selection orders by `n.generation DESC, b.node_id DESC`, so the
  newest generation wins even when its node id is smaller.
- The blob goes through the existing private `decode_brain_state`; `None`
  means `MalformedBrain(node_id)`. Add no second decoder.
- `config_json` goes through `serde_json::from_str::<BrainConfig>`; an error
  means `MalformedConfig(node_id)`. No `BrainConfig::default()` fallback.
- No layout check against the config here; the runner does it.
- Open failures map to `Open`, other rusqlite errors to `Query`. No `unwrap`
  or `expect` on I/O in non-test code.
- WAL sidecars (owner decision): the guarantee is that the main database
  file's bytes are unchanged by a load. A read-only open of a WAL database may
  create `-wal` and `-shm` sidecar files, and they may remain afterwards; that
  is accepted, so the file-bytes test hashes and compares the main file only
  and does not compare the directory listing or the sidecars. Do not use the
  `immutable=1` or `nolock` URI parameters to avoid the sidecars:
  `immutable=1` misses rows still in the WAL. Open the plain path with
  `SQLITE_OPEN_READ_ONLY`.

**Steps:**

1. Read `governor.rs` near `NODE_BRAIN_SCHEMA`, the blob helpers and the
   `Governor` impl for import style. Add only the missing `use` items:
   `std::time::Duration`, `std::fmt`, `rusqlite::OpenFlags`.
2. In non-test code, after `decode_brain_state`, add the constant, the two
   public types and the enum with doc comments, the `Display` and `Error`
   impls, the private `open_read_only` and `table_exists` helpers, a private
   row-query helper returning `(node_id, generation, config_json, blob)`, and
   `load_champion_read_only` with a `# Errors` doc section. Keep each helper
   under about 40 lines.
3. In `mod tests`, add `read_only_champion_fixture(dir: &std::path::Path) ->
   String`. It creates the file database `dir.join("champion.db")` with
   `Governor::new` (the same `GovernorConfig` literal as the existing test near
   line 5726), inserts nodes 50 (generation 7), 100 (generation 3) and 200
   (generation 9) through `gov.db.execute`, stores brains for 50 (tag 2.0) and
   100 (tag 1.0) with `gov.store_champion_brain` and `tagged_brain`, leaves 200
   without a brain row, and sets each node's config to
   `serde_json::to_string(&BrainConfig::default())`. It drops the governor
   before returning. Compare `AgentBrainState` by its `brain_state` and
   `patterns` fields, since it derives only `Clone` and `Debug`.
4. Add these eight tests, named exactly, each with a doc comment and exact
   assertions:
   - `read_only_champion_default_selects_greatest_generation_then_id`: `None`
     returns node 50, generation 7, with its exact brain and config.
   - `read_only_champion_explicit_node_loads_that_node`: `Some(100)` returns
     node 100's exact brain and config, generation 3.
   - `read_only_champion_node_without_brain_is_reported`: `Some(200)` returns
     `NodeWithoutBrain(200)`.
   - `read_only_champion_corrupt_blob_is_malformed_brain`: a read-write
     connection sets node 100's blob to bytes `[1, 2]`; the loader returns
     `MalformedBrain(100)`.
   - `read_only_champion_malformed_config_is_rejected`: node 50's `config_json`
     becomes `{not json`; `None` returns `MalformedConfig(50)`.
   - `read_only_champion_database_without_table_has_no_champion`: a read-write
     `DROP TABLE node_brain` makes `None` and `Some(100)` return
     `NoStoredChampion`.
   - `read_only_champion_load_leaves_file_bytes_unchanged`: the main database
     file `champion.db` only is read with `std::fs::read` and hashed (an
     in-test FNV-1a 64-bit helper with named, documented constants) before
     and after loads with `None` and with `Some(100)`; the bytes and the hash
     are equal before and after, and both loads return their expected nodes
     (50 and 100). Sidecar files (`-wal`, `-shm`) are neither read nor
     compared, and the directory listing is not compared.
   - `read_only_champion_write_through_read_only_handle_fails`: through
     `open_read_only`, an `UPDATE node_brain` on node 100 and an `INSERT` of
     node 300 both return `Err`; a fresh `None` load still returns node 50 with
     its original brain.
5. Keep the source free of planning references: no "plan", "task" or
   "workstream" and no plan or task numbers in comments, doc comments or
   strings. Leave the one existing older comment in `governor.rs` untouched.
   Edit no other file. Do not run `cargo fmt` by hand; the `sandbox-fmt` check
   applies its own fix.

- **Done when:** the focused-proof check `cargo test -p xagent-sandbox --lib governor::tests::read_only_champion` passes reporting 8 passed, covering the eight `read_only_champion_*` tests in step 4, each asserting its exact values, which proves INV-5 (no write, no schema statement, the main database file's bytes unchanged with `-wal`/`-shm` sidecars allowed, and malformed or missing data reported as errors).

## Non-goals

- No migration, repair or write path. The loader never writes.
- No layout check against the config and no `BrainLayout` use.
- No change to `Governor`, `init_schema`, `NODE_BRAIN_SCHEMA`, the encoder, the
  decoder or any existing test.
- No new Cargo dependency. `rusqlite`, `serde_json` and `tempfile` are present.
- CPU-only tests; no GPU use.
