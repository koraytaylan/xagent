---
id: observatory-store-read
title: Read-only detection and load of a supervisor run into ObservatoryModel
workstream: "0200"
kind: task
role: developer
depends_on:
  - observatory-model
gated: false
touches:
  - crates/xagent-sandbox/src/observatory/store_read.rs
  - crates/xagent-sandbox/tests/observatory_browse.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: cargo test -p xagent-sandbox --test observatory_browse
    timeout_ms: 3600000
  - name: sandbox-clippy
    command: cargo clippy -p xagent-sandbox --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: cargo fmt -p xagent-sandbox -- --check
    fix: cargo fmt -p xagent-sandbox
  - name: sandbox-guard
    command: cargo test -p xagent-sandbox --test contributing_guard
---

# Read-only detection and load of a supervisor run into ObservatoryModel

## Outcome

`crates/xagent-sandbox/src/observatory/store_read.rs` provides the read-only
browse path: it detects the latest supervisor run in a database and loads its
stored records into an `ObservatoryModel` through `ObservatoryModel::fold`.
Nothing in this path writes to, creates or migrates the database. The new test
file `crates/xagent-sandbox/tests/observatory_browse.rs` proves the file is
byte-identical after browsing, a missing path is not created, an `INSERT`
through the browse connection is refused, the loaded model equals the live fold
of the same records, and a database without the tables yields `None`.

## Decisions (binding for this task)

- `store_read.rs` declares exactly these public items:
  - `pub enum BrowseError` with variants `Missing` (the database path is not an
    existing regular file), `Sqlite(rusqlite::Error)`, `NoRun { run_id: i64 }`
    and `Malformed { detail: String }`. It implements `Display` and
    `std::error::Error`. `Display` text must not contain the database path.
  - `#[derive(Clone, Debug, PartialEq, Eq)] pub struct SupervisorRunSummary {
    pub run_id: i64, pub last_tick: u64, pub budget_ticks: Option<u64> }`.
  - `pub fn open_read_only(db_path: &str) -> Result<rusqlite::Connection,
    BrowseError>`, `pub fn detect_supervisor_run(db_path: &str) ->
    Result<Option<SupervisorRunSummary>, BrowseError>`, and
    `pub fn load_run(db_path: &str, run_id: i64) -> Result<ObservatoryModel,
    BrowseError>`.
- `open_read_only` first checks `std::path::Path::new(db_path).is_file()` and
  returns `BrowseError::Missing` when it is false, so no file is created. It then
  calls `Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY |
  OpenFlags::SQLITE_OPEN_NO_MUTEX)`, sets `busy_timeout` to the existing
  `STORE_BUSY_TIMEOUT` from `supervisor_store.rs`, and runs
  `PRAGMA query_only = ON;`. It never calls `ensure_schema`, `init_schema`, or any
  writer method.
- Table presence: a table is present when `sqlite_master` lists it with
  `type = 'table'`. The run is `run` with a `mode` column and the three
  supervisor tables named in `supervisor_store.rs` (`supervisor_slot`,
  `supervisor_archive`, `supervisor_event`) plus the plan 0006 snapshot, series
  and progress tables that plan 0006 lands in `supervisor_store.rs`. Names are
  read from that landed code at implementation time. If a required table or
  record method is absent, stop and report the file and line as a blocker.
- `detect_supervisor_run` returns `Ok(None)` when the path is `Missing`, when a
  required table is absent, or when no `run` row has `mode = 'supervisor'`. It
  otherwise returns the latest such run (`ORDER BY id DESC LIMIT 1`), with
  `last_tick` from the run's stored simulation ticks (NULL reads as 0) and
  `budget_ticks` from the latest plan 0006 progress row (`None` when no row).
- `load_run` returns `BrowseError::NoRun { run_id }` when the run id is not a
  supervisor run. Otherwise it reads each plan 0006 table with
  `ORDER BY tick DESC LIMIT <cap>` for the matching model cap (`MAX_SERIES_POINTS`,
  `MAX_BOUNDARY_RECORDS`, `MAX_TIMELINE_EVENTS`, `MAX_ARCHIVE_ROWS`,
  `MAX_TRACKED_OCCUPANTS`), reverses each result to ascending tick order, converts
  each row to an `ObservatoryRecord` through the `From` conversions in `observatory/mod.rs`,
  and folds the records in ascending tick order. The model's `source` is set to
  `ModelSource::Browse { run_id }`. Text values longer than 256 bytes are
  truncated at a character boundary. A row whose integer does not fit `u64`,
  a non-finite float, or an undecodable JSON or brain blob returns
  `BrowseError::Malformed`; no row is skipped silently and nothing panics.
- Comments and strings name no plan, task or INV identifiers. No `as` integer
  casts (use `try_from`), no `TODO` or `FIXME`, no magic numbers outside named
  constants.
- Test fixture: `SupervisorStore::create(path, &FullConfig::default())` from
  `supervisor_store.rs`, then the plan 0006 record writers for slots, boundaries,
  series, progress and events, then `drop(store)`, then
  `Connection::open(path)` with `PRAGMA wal_checkpoint(TRUNCATE);` and the
  connection dropped. The bytes are read only after this checkpoint.
- Test equality for `observatory_browse_matches_live_fold`: the live model is
  `ObservatoryModel::default()` folded with the same `ObservatoryRecord` values
  in tick order. Both models have `source` and `progress.ticks_per_sec` set to
  `ModelSource::Live` and `None` before comparison. The comparison uses
  `assert_eq!` on the whole model, which requires `ObservatoryModel` and its
  field types to derive `PartialEq` and `Debug`. The `observatory-model` task
  owns those derives. If they are absent in `observatory/mod.rs`, stop and report the file
  and line as a blocker; do not add them here, since `observatory/mod.rs` is not in touches.

## Prerequisites and blocker rule

Plan 0006 (`supervisor-decision-telemetry`) must have landed its snapshot, series,
progress and event tables and record methods in `supervisor_store.rs`. At the
baseline, `supervisor_store.rs` lines 155 to 195 create only `supervisor_slot`,
`supervisor_archive` and `supervisor_event`, and `SupervisorStore` (lines 305 to
509) has no series or progress writer. This task depends on plan 0006 at plan
level. If those names are still absent at implementation time, stop and report
the file and line; do not invent table or method names.

**Steps:**

1. Read `crates/xagent-sandbox/src/observatory/mod.rs` for the cap constants,
   `ObservatoryRecord`, `ModelSource` and the `From` conversions, and read
   `crates/xagent-sandbox/src/supervisor_store.rs` lines 555 to 622 as the
   read-only open pattern. Confirm the plan 0006 table and writer names in
   `supervisor_store.rs`; if any is absent, stop and report the blocker.
2. In `crates/xagent-sandbox/src/observatory/store_read.rs`, add the module doc
   line, `BrowseError` with `Display` and `std::error::Error`, and
   `SupervisorRunSummary` as in Decisions.
3. In the same file, implement `open_read_only` and a private `has_tables`
   helper that checks `sqlite_master` for the required table names.
4. Implement `detect_supervisor_run` as in Decisions.
5. Implement `load_run` with the capped `ORDER BY tick DESC LIMIT` reads,
   ascending fold, `ModelSource::Browse`, truncation and `Malformed` errors.
6. Create `crates/xagent-sandbox/tests/observatory_browse.rs` with a fixture
   builder and exactly these five tests, each a top-level `#[test]` function:
   `observatory_browse_file_byte_identical` (bytes before and after
   `detect_supervisor_run` plus `load_run` are equal),
   `observatory_browse_missing_path_not_created` (a missing path stays missing
   after detect and load, and detect returns `None`),
   `observatory_browse_refuses_insert` (an `INSERT` through `open_read_only`
   returns an error), `observatory_browse_matches_live_fold` (the loaded model
   equals the live fold of the same records, per Decisions), and
   `observatory_browse_no_tables_is_none` (a database with only the governor
   schema, or an empty file, makes detect return `Ok(None)`).

- **Done when:** `focused-proof` (`cargo test -p xagent-sandbox --test observatory_browse`) passes with `observatory_browse_file_byte_identical`, `observatory_browse_missing_path_not_created`, `observatory_browse_refuses_insert`, `observatory_browse_matches_live_fold` and `observatory_browse_no_tables_is_none` reported as ok, which proves INV-2 (the loaded model equals the live fold) and INV-3 (the store part: byte-identical file, no creation of a missing path, and refused writes).
