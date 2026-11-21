---
id: checkpoint-store
title: Checkpoint table, read-only load and resume rewind
workstream: "0100"
kind: task
role: developer
depends_on:
  - checkpoint-envelope
gated: false
touches:
  - crates/xagent-sandbox/src/supervisor_store.rs
status: done
merged_as: "21905b31b507b81b8f013117244b969bbaf8b0a7"
checks:
  - name: focused-proof
    command: cargo test -p xagent-sandbox --lib supervisor_store
    timeout_ms: 3600000
  - name: sandbox-clippy
    command: cargo clippy -p xagent-sandbox --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: cargo fmt -p xagent-sandbox -- --check
    fix: cargo fmt -p xagent-sandbox
---

# Checkpoint table, read-only load and resume rewind

## Outcome

`crates/xagent-sandbox/src/supervisor_store.rs` gains the single-row checkpoint
table, an atomic overwrite, a read-only loader and the resume-time rewind. It
stores opaque bytes only; encoding is `supervisor_checkpoint.rs` (prerequisite).
The storage contract is the ARCHITECTURE.md "Storage" bullet, extended here by
`simulation_ticks` and `format_version`. No other file changes.

## Decisions (explicit)

- **File scope.** Edit only `supervisor_store.rs`. Plan 0006's run-keyed tables
  (with a `tick` column) are found by reading `ensure_schema`, not from a list.
- **Prerequisite.** Import `FORMAT_VERSION` and `MAX_CHECKPOINT_BYTES` from
  `crate::supervisor_checkpoint`; do not redefine them.
- **Table.** Inside `ensure_schema`, after the existing supervisor tables, add
  exactly:
  `CREATE TABLE IF NOT EXISTS supervisor_checkpoint (run_id INTEGER PRIMARY KEY REFERENCES run(id), tick INTEGER NOT NULL, format_version INTEGER NOT NULL, simulation_ticks INTEGER, payload BLOB NOT NULL);`
  `simulation_ticks` is nullable so the table exists before any write.
- **`write_checkpoint(&mut self, tick: u64, payload: &[u8]) -> Result<(), StoreError>`.**
  One transaction on the store's connection (field near line 306), one statement:
  `INSERT OR REPLACE INTO supervisor_checkpoint (run_id, tick, format_version, simulation_ticks, payload) SELECT id, ?1, ?2, simulation_ticks, ?3 FROM run WHERE id = ?4`
  with `?1` = `db_int(tick)`, `?2` = `FORMAT_VERSION as i64`, `?3` = payload,
  `?4` = `self.run_id`. The `SELECT` copies the run's current `simulation_ticks`
  for the rewind. Zero rows changed returns `StoreError::Sqlite` wrapping
  `rusqlite::Error::QueryReturnedNoRows`. Commit.
- **`run_configs(&self) -> Result<(String, String, String), StoreError>`.**
  Returns `governor_config`, `brain_config`, `world_config` JSON text of the run
  row, in order. No caller in this task.
- **`StoredCheckpoint`.** `#[derive(Debug, Clone, PartialEq, Eq)] pub struct
  StoredCheckpoint { pub run_id: i64, pub tick: u64, pub format_version: u32,
  pub payload: Vec<u8> }`. No `simulation_ticks` field.
- **`read_checkpoint(db_path: &str) -> Result<Option<StoredCheckpoint>, StoreError>`.**
  Never creates a file: a missing path returns `Ok(None)`. Otherwise open
  `SQLITE_OPEN_READ_ONLY | SQLITE_OPEN_NO_MUTEX`. Newest run:
  `SELECT id FROM run WHERE mode = ?1 ORDER BY id DESC LIMIT 1` with
  `MODE_SUPERVISOR`. No run, no checkpoint row, or no table gives `Ok(None)`.
  Select `tick, format_version, length(payload)` first; if the length exceeds
  `MAX_CHECKPOINT_BYTES`, return `StoreError::CheckpointTooLarge` without
  selecting the blob. Otherwise select `payload`. Stored `i64` values use a
  checked cast; a negative tick or format version is `StoreError::Sqlite`
  wrapping `rusqlite::Error::IntegralValueOutOfRange`. No malformed-blob variant.
- **`reopen_for_resume(db_path: &str, run_id: i64, tick: u64) -> Result<Self, StoreError>`.**
  Open read-write with the same connection setup as `create` (WAL and shared
  pragmas), inserting no run row. Verify run `run_id` has `mode = MODE_SUPERVISOR`
  and its checkpoint row has `tick == tick`; otherwise return
  `StoreError::CheckpointTickMismatch { run_id, stored: Option<i64>, requested: u64 }`
  (`stored` is `None` when the row is missing). Then, in one transaction before
  any other write:
  1. Discover run-keyed tables, interpolating nothing:
     `SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name NOT IN ('supervisor_checkpoint', 'supervisor_archive') AND EXISTS (SELECT 1 FROM pragma_table_info(name) c WHERE c.name = 'run_id') AND EXISTS (SELECT 1 FROM pragma_table_info(name) c WHERE c.name = 'tick')`
  2. For each name, `DELETE FROM "<name>" WHERE run_id = ?1 AND tick >= ?2`
     (name double-quoted, inner `"` doubled).
  3. `DELETE FROM supervisor_archive WHERE run_id = ?1 AND retired_tick >= ?2`.
  4. `UPDATE run SET simulation_ticks = (SELECT simulation_ticks FROM supervisor_checkpoint WHERE run_id = ?1) WHERE id = ?1 AND (SELECT simulation_ticks FROM supervisor_checkpoint WHERE run_id = ?1) IS NOT NULL`.

  Commit, then return `Self` with `run_id` set. Other runs, rows with `tick < T`,
  and the checkpoint row are untouched.
- **Errors.** Add `CheckpointTooLarge { len: i64, max: usize }` and
  `CheckpointTickMismatch` (as above) to `StoreError`, each with a plain-English
  `Display` arm; add `use` lines for `Path`, `rusqlite::OpenFlags` and the two
  constants. Change no other error.
- **Not decided here.** Existing writes to run `simulation_ticks` (near line 503)
  are unchanged.

## Non-goals

- No envelope codec, `KernelCheckpoint` or supervisor types.
- No change to `supervisor_run.rs`, the drivers or `supervisor_checkpoint.rs`.
- No checkpoint history, no new dependency, no version change.
- Do not edit existing tests in `supervisor_store::tests`; add new tests only.

## Steps

**Steps:**

1. Add the imports, the two `StoreError` variants with `Display` arms, and the
   `supervisor_checkpoint` statement in `ensure_schema`. Keep existing tables.
2. Add `write_checkpoint` and `run_configs`. Add test
   `store_checkpoint_single_row_overwritten`: create a store with the existing
   test config helper, write 50 checkpoints with increasing ticks and distinct
   payloads, assert `COUNT(*)` is 1 and the row holds the last tick and payload.
3. Add `StoredCheckpoint` and `read_checkpoint`. Add tests
   `store_checkpoint_read_only_does_not_create_file` (missing path gives
   `Ok(None)` and the file still does not exist) and
   `store_checkpoint_missing_returns_none` (a run with no checkpoint gives
   `Ok(None)`; a written checkpoint reads back with the same tick, format
   version and payload).
4. Add `reopen_for_resume`. Add test `store_reopen_rewinds_rows_at_or_after_tick`:
   two supervisor runs; `test_extra (id INTEGER PRIMARY KEY, run_id INTEGER NOT
   NULL, tick INTEGER NOT NULL)`; rows below, at and above T in `supervisor_event`,
   `test_extra` and `supervisor_archive` (`retired_tick`) for both runs; checkpoint
   at T; more rows at or after T. Reopen at T: first run's rows with tick `>= T`
   are gone from every run-keyed table, earlier rows and the second run remain.
   Build a second database with the same history and no rewind and assert
   post-reopen inserted ids match it per table. Reopen at `T + 1` returns
   `CheckpointTickMismatch`.
5. Request the host-run `focused-proof`, `sandbox-clippy` and `sandbox-fmt`
   checks. If `sandbox-fmt` fails, run its `fix` and rerun all three. If the fix
   touches any file other than `supervisor_store.rs`, stop and report it.

- **Done when:** the `focused-proof` check passes `supervisor_store::tests::store_checkpoint_single_row_overwritten`, `store_checkpoint_read_only_does_not_create_file`, `store_checkpoint_missing_returns_none`, `store_reopen_rewinds_rows_at_or_after_tick`, and every existing `supervisor_store` test, and `sandbox-clippy` and `sandbox-fmt` pass, proving INV-3 (one `supervisor_checkpoint` row per run, written by one `INSERT OR REPLACE` in one transaction) and INV-8 (resume deletes exactly the run's rows with tick `>= T` and archive rows with `retired_tick >= T`, in one transaction before any write).
