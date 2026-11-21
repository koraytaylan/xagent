---
id: store-legacy-guard
title: "Refuse legacy generation databases in the supervisor store and cut its governor imports"
workstream: "0100"
kind: task
depends_on: []
gated: false
role: developer
touches:
  - crates/xagent-sandbox/src/supervisor_store.rs
  - crates/xagent-sandbox/src/supervisor.rs
status: todo
merged_as: ""
checks:
  - name: store-unit
    command: "cargo test -p xagent-sandbox --lib supervisor_store"
  - name: supervisor-unit
    command: "cargo test -p xagent-sandbox --lib supervisor::"
  - name: store-governor-free
    command: "sh -c '! grep -nE \"crate::governor|governor::\" crates/xagent-sandbox/src/supervisor_store.rs crates/xagent-sandbox/src/supervisor.rs'"
  - name: all-targets-compile
    command: cargo check --workspace --all-targets
    timeout_ms: 3600000
---

# 0105 Refuse legacy generation databases in the supervisor store and cut its governor imports

## Outcome

`supervisor_store.rs` classifies a database read-only before any writable open,
refuses a legacy generations database with `StoreError::LegacyGenerations`
without changing its bytes, and no longer imports `governor`. The brain-blob
codec and run schema move into the store. `supervisor.rs` gets local copies of
the fitness helpers and no longer imports `governor`. `governor.rs` and
`tests/pathway_route_state.rs` are not edited.

## Decisions

- **inspect_database.** `pub fn inspect_database(path: &Path) -> Result<DatabaseKind, StoreError>`:
  1. `std::fs::metadata` reports `NotFound`: return `Missing`, opening nothing.
  2. Open `file:<encoded path>?immutable=1` with `Connection::open_with_flags` and
     `SQLITE_OPEN_READ_ONLY | SQLITE_OPEN_URI | SQLITE_OPEN_NO_MUTEX`. Encode the
     path by replacing `%`, `?` and `#` with `%25`, `%3F` and `%23`. This creates no `-wal`/`-shm`.
  3. Table presence from `sqlite_master` (`type = 'table'`, names `run`, `node`).
  4. `node_rows` = `SELECT EXISTS(SELECT 1 FROM node)` when `node` exists. If true,
     return `LegacyGenerations` with `generation = SELECT MAX(generation) FROM node`
     (a failing MAX yields `None`).
  5. When `run` exists: column names from `SELECT * FROM run LIMIT 0`, and
     `run_rows = SELECT COUNT(*) FROM run`. If `run_rows > 0` and the `mode` column
     is absent or `SELECT COUNT(*) FROM run WHERE mode IS NULL OR mode <> 'supervisor'`
     is nonzero, return `LegacyGenerations` with the same generation rule as step 4.
  6. `run_rows` is zero (or `run` absent) and `node_rows` is false: `Empty`.
  7. Otherwise `Supervisor`.

  Any SQLite error in steps 2 to 6 is returned as `StoreError::Sqlite`. No
  `PRAGMA`, no `ALTER`, and no other statement runs on this connection.
- **Error text.** `StoreError::LegacyGenerations { path: String, generation: Option<u64> }`
  displays `<path> holds a generation-mode run (generation <n>). This version runs only the continuous supervisor and cannot run it. The file was not changed; choose another database with --db.`
  The ` (generation <n>)` part is omitted when `generation` is `None`.
- **Create guard.** `SupervisorStore::create` calls `inspect_database(Path::new(db_path))?`
  before `Connection::open(db_path)`. `LegacyGenerations` returns
  `Err(StoreError::LegacyGenerations { path: db_path.to_owned(), generation })`.
  Keep the exact line `let mut db = Connection::open(db_path)?;`.
- **Archive-loader guard.** Add `ArchiveLoadError::Store(StoreError)`, whose Display
  delegates to the inner error. `load_archive_entry_read_only` calls
  `inspect_database` before its open. `LegacyGenerations` and any inspection `Err`
  return `ArchiveLoadError::Store(...)`. Other kinds continue unchanged.
- **Codec move.** Move `decode_raw_brain_state` with its constants, `encode_brain_state`
  and `decode_brain_state` into `supervisor_store.rs`. The encode and decode functions
  are `pub(crate)`. Copy the bodies verbatim. `governor.rs` keeps its own copies, since it still calls them.
- **Schema.** `ensure_schema` stops calling `init_schema`. It creates the `run` table
  with every column of `governor::init_schema`'s run table, including `mode TEXT`,
  applies the same idempotent `ALTER TABLE run ADD COLUMN` statements, then runs
  the existing supervisor-table batch. It creates none of `node`, `agent_result`,
  `mutation`, `generation_recording`, `behavior_metric` or `node_brain`.
- **Fitness move.** Copy `composite_fitness`, `exploration_grid_cells` and the
  constants they read into `supervisor.rs` as `pub(crate)`, unchanged in arithmetic.
  `governor.rs` keeps its copies.
- **Tests.** Replace the Governor-based store test with raw-SQL fixtures, or a run
  made through `SupervisorStore::create` on a fresh path, using the module's existing
  temp-directory helper. Copy the three legacy codec cases from
  `tests/pathway_route_state.rs` with the same expected values.

**Steps:**

1. Read `supervisor_store.rs` and `supervisor.rs`, the codec, fitness, `init_schema`
   and constants in `governor.rs`, and the three legacy cases in `tests/pathway_route_state.rs`.
2. In `supervisor_store.rs`, add `DatabaseKind`, `inspect_database`,
   `StoreError::LegacyGenerations` and `ArchiveLoadError::Store`, then guard
   `create` and `load_archive_entry_read_only` as in Decisions.
3. In `supervisor_store.rs`, move the codec, rewrite `ensure_schema` as in
   Decisions, and remove the governor import line.
4. In `supervisor.rs`, copy the fitness helpers and constants as `pub(crate)`,
   and remove the governor import line.
5. Rewrite the `supervisor_store.rs` test module without Governor. Add the ten
   tests named in Done when. The `create_refuses_legacy_and_leaves_bytes_unchanged`
   test byte-compares the fixture and asserts the directory listing is unchanged.

- **Done when:** the checks store-unit (with the ten tests inspect_missing_file_is_missing, inspect_empty_file_is_empty, inspect_supervisor_run_is_supervisor, inspect_generation_db_is_legacy_with_generation, inspect_premode_run_row_is_legacy, create_refuses_legacy_and_leaves_bytes_unchanged, archive_loader_refuses_legacy, codec_legacy_blob_upgrades, codec_current_blob_roundtrips, codec_wrong_layout_rejected passing), supervisor-unit, store-governor-free and all-targets-compile all pass on the host, proving the store refuses a legacy generations database without changing a byte (INV-2) and the moved codec still upgrades legacy blobs (INV-8).
