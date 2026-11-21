---
id: telemetry-store
title: "Additive idempotent telemetry tables and writer"
workstream: "0300"
kind: task
role: developer
depends_on:
  - telemetry-records
gated: false
status: todo
merged_as: ""
touches:
  - crates/xagent-sandbox/src/supervisor_store.rs
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --lib supervisor_store::tests::"
  - name: fmt
    command: "cargo fmt --all -- --check"
    fix: "cargo fmt --all"
  - name: clippy
    command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
---

# Additive idempotent telemetry tables and writer

## Outcome

`crates/xagent-sandbox/src/supervisor_store.rs` gains six additive telemetry
tables created by `ensure_schema`, and `SupervisorStore::record_telemetry`
writes one `TelemetryFrame` in one transaction. The 0005 tables, their DDL and
their writers stay unchanged.

## Decisions

- `ensure_schema` keeps `init_schema(db)?` and the existing batch, and appends
  six `CREATE TABLE IF NOT EXISTS` statements after `supervisor_event` in the
  same `execute_batch`. No `ALTER`, no change to the three existing statements.
- Tables are exactly those in ARCHITECTURE.md "Telemetry tables" (columns,
  primary keys, `run_id INTEGER NOT NULL REFERENCES run(id)`). Columns marked
  NULL there are nullable; all others are `NOT NULL`. Series `value` is nullable.
- `pub fn record_telemetry(&mut self, frame: &TelemetryFrame) -> Result<(), StoreError>`
  returns `Ok(())` without touching the database unless `frame.persist`. Otherwise
  one transaction on `self.db` (using `self.run_id`) writes: slot snapshots, then
  the boundary row and candidate rows when `frame.boundary` is `Some`, then series,
  intake and the progress row; then commits. Plain `INSERT` only, so a duplicate
  key is an error.
- Integers via `db_int` / `db_index`; `u32` and `bool` via `i64::from`; `f32` via
  `f64::from`. One helper `finite_real(f64) -> Option<f64>` is applied to every
  REAL column, so NaN and infinity store as NULL.
- Ground text: `Death` death, `BelowMedian` below_median, `TooYoung` too_young,
  `NoFullWindow` no_full_window, `NoPeerMedian` no_peer_median,
  `AtOrAboveMedian` at_or_above_median. Parent kind and ref reuse the existing
  `PARENT_KIND_*` constants as `record_boundary` does: `ArchiveEntry` is
  `birth`, `PeerBirth` is `birth`, `PeerLearned` is `learned` for `parent_state`.
  A `None` parent writes NULLs.
- `Replaced` boundary row: `outcome` `replace`; slot, reason (`REASON_*`
  constants), tiebreak pool as a JSON array, tiebreak index, parent columns,
  `parent_pool_len` / `parent_index` from `ParentDraw` (NULL for peer parents),
  retired and newborn occupants and lifespan (NULL when `None`), and `config_diff`
  as a JSON array of `{"field","parent","newborn"}` objects built with `json!`.
  `idle_*` columns NULL.
- `Idle` boundary row: `outcome` `idle`, the four `idle_*` counts from
  `IdleReason`; every replace-only column NULL, including `config_diff`.
- Candidate rows only for a `Replaced` detail: each candidate whose ground is
  `Death` or `BelowMedian` is written (other grounds skipped). Score and median
  come from `BelowMedian` through `finite_real`, NULL for `Death`. `Idle` writes
  no candidates.
- Progress: `nourishing_kind` is `serde_json::to_string` of the `FoodKind`, as
  `record_season` does; `total_ticks` and `next_flip_tick` NULL when `None`;
  `ticks_per_second` through `finite_real`.

## Non-goals

- No change to the 0005 DDL, `init_schema`, `create`, `record_slot`,
  `record_boundary`, `record_idle_boundary`, `record_season`, `record_ticks` or
  the archive loader. No `OR REPLACE`, `ALTER`, `DROP` or `VACUUM`.
- No edit outside `supervisor_store.rs`; no new dependency (`tempfile` already
  exists as a dev-dependency).
- No worker, driver or headless call of `record_telemetry`; no read-side helpers.
- Doc comments are plain prose: no plan numbers, task slugs or workstream words.

**Steps:**

1. In `ensure_schema`, append the six table statements from Decisions to the
   existing `execute_batch` after `supervisor_event`, with exact columns, NULL
   rules, primary keys and `run_id INTEGER NOT NULL REFERENCES run(id)`. Leave
   the three existing statements byte-for-byte unchanged.
2. Add private helpers `finite_real`, the ground-text `match`, and the
   `ParentRef` to (kind, ref, state) mapping that reuses `PARENT_KIND_*`.
3. Implement `record_telemetry` with the persist early return, one transaction,
   the write order above, the `Replaced` and `Idle` boundary rules and
   candidate rules, and plain `INSERT` throughout.
4. Add test `telemetry_migration_keeps_old_rows_and_adds_tables_once` to the
   existing tests module: a `tempfile` database built with `init_schema` and the
   three 0005 `CREATE TABLE IF NOT EXISTS` statements copied verbatim from the
   baseline `ensure_schema`; insert one `run` row and one row in each 0005
   table; call `ensure_schema` twice; assert old rows unchanged, each new table
   appears once in `sqlite_master`, and each new table's `run_id` has
   `notnull = 1` in `PRAGMA table_info` and a foreign key to `run.id` in
   `PRAGMA foreign_key_list`.
5. Add tests `telemetry_writer_round_trips_frames` (a persisted frame with a
   `Replaced` boundary holding a `Death` and a `BelowMedian` candidate plus one
   `AtOrAboveMedian` standing that is skipped; a persisted `Idle` frame; a
   non-persisted frame adding zero rows; a repeated write of a persisted frame
   returning `Err`) and `telemetry_nan_score_is_stored_as_null` (a snapshot
   with `window_score: Some(f32::NAN)` reads back NULL). Use
   `SupervisorStore::create` on a tempfile path with the default `FullConfig`
   the existing tests use.

- **Done when:** the focused-proof check `cargo test -p xagent-sandbox --lib supervisor_store::tests::` passes with `telemetry_migration_keeps_old_rows_and_adds_tables_once`, `telemetry_writer_round_trips_frames` and `telemetry_nan_score_is_stored_as_null` green, proving INV-7.
