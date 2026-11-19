---
id: supervisor-store
title: "Supervisor tables, run row and read-only archive loader"
workstream: "0200"
kind: task
role: developer
depends_on: [supervisor-core]
gated: false
touches: [crates/xagent-sandbox/src/lib.rs, crates/xagent-sandbox/src/supervisor_store.rs]
status: todo
merged_as: ""
checks:
  - {name: focused-proof, command: "cargo test -p xagent-sandbox --lib supervisor_store::tests::"}
  - {name: sandbox-clippy, command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings", timeout_ms: 3600000}
  - {name: sandbox-fmt, command: "cargo fmt -p xagent-sandbox -- --check", fix: "cargo fmt -p xagent-sandbox"}
  - {name: sandbox-guard, command: "cargo test -p xagent-sandbox --test contributing_guard"}
---

# Supervisor tables, run row and read-only archive loader

## Outcome

`lib.rs` declares `pub mod supervisor_store;` after `pub mod supervisor;`. `supervisor_store.rs` creates the three supervisor tables idempotently, creates a supervisor run row only in a fresh database, writes each boundary, season, idle and finish change in one transaction, and loads the archive entry at a rank of the latest supervisor run read-only. Loading leaves the database file bytes unchanged.

## Scope

- `lib.rs`: add the module line.
- `supervisor_store.rs` (new), with an inline `#[cfg(test)] mod tests`.
- Table consts, `ensure_schema`, `SupervisorStore`, `StoredArchiveEntry`, `load_archive_entry_read_only`, `StoreError`, `ArchiveLoadError`. Table columns, constraints and the `Store` layout are in ARCHITECTURE.md "Interfaces (binding)" → "Store"; the `run_id` foreign keys and `UNIQUE (run_id, entry_id)` are there too.
- Tests (names exact):
  - `store_schema_is_idempotent`: `ensure_schema` twice is Ok; three tables exist.
  - `store_create_records_supervisor_mode`: one run row, `mode = 'supervisor'`, seed via `cast_signed()`.
  - `store_refuses_non_fresh_database`: a generation-seeded file gives `NotFresh`; row count unchanged.
  - `store_schema_on_generation_database_keeps_it_resumable`: `Governor::resume` is Ok and `mode` is `'generations'`.
  - `store_archive_table_mirrors_bounded_archive`: capacity 2, several replacements; rows equal in-memory entries.
  - `archive_loader_returns_rank_order`: ranks 0 to 2 return outcome-descending entries.
  - `archive_loader_leaves_database_bytes_unchanged`: byte-hash proof (Decisions).
  - `archive_loader_rejects_rank_beyond_archive`: `RankOutOfRange`.
  - `archive_loader_rejects_generation_database`: `NoSupervisorRun`.
  - `archive_loader_rejects_malformed_blob`: 3-byte `birth_brain` gives `MalformedBlob`.

## Non-goals

- No change to `governor.rs`, `supervisor.rs` or any other file.
- No resume of a supervisor run; `create` is the only writer opener.
- No GPU, `SupervisorRun`, run summary, CLI flag or `ui.rs` change.

## Prerequisites

- supervisor-core names as in its Decisions. `ArchiveEntry` and `SlotLedger` fields are `pub`; `Archive::entries()` is rank order; `Archive::get(entry_id)` looks one up; `ReplacementOutcome` carries `archived` and `evicted`; `ArchiveChange` carries `entry` and `evicted`.
- evolution-mode-config: `init_schema`, `encode_brain_state` and `decode_brain_state` are `pub(crate)`; `run.mode` exists with default `'generations'`; resume refuses supervisor runs.

## Decisions

- **Signatures (binding; the store adapts to supervisor-core):**
  - `record_slot(&mut self, slot: usize, ledger: &SlotLedger)`.
  - `record_boundary(&mut self, tick: u64, plan: &Replacement, outcome: &ReplacementOutcome, archived: Option<&ArchiveEntry>)`: one transaction: `replace` event (slot, reason, new occupant, `parent_kind`/`parent_ref` from `plan.parent`), archive row when `archived` is `Some`, delete the `outcome.evicted` row and write an `evict` event, upsert the replaced slot. `outcome.archived` and `archived` must agree on `entry_id`, else `StoreError::Mismatch`.
  - `record_idle_boundary(&mut self, tick: u64)`: one `idle` event.
  - `record_season(&mut self, tick: u64, kind: FoodKind)`: `nourishing_kind` is `serde_json::to_string(&kind)`, the JSON string form of `FoodKind`. If that is not a string literal, report it; do not change `FoodKind`.
  - `record_archive_changes(&mut self, tick: u64, changes: &[ArchiveChange], archive: &Archive)`: one transaction; insert each `change.entry` row via `archive.get`, delete the `evicted` row, write `archive` and `evict` events.
  - `record_ticks(&mut self, ticks: u64)`: `i64::try_from`; overflow is `StoreError::Range`.
- **Blobs.** `encode_brain_state` output; `None` is `StoreError::Encode`. Config is `serde_json::to_string` of `BrainConfig`.
- **Create.** Opens with the `Governor::new_seeded` pragma string, runs `ensure_schema`, and refuses any `run` row as `NotFresh`. The run row inserts `mode = 'supervisor'` in one transaction.
- **Loader (0001 `load_champion_read_only` convention).** Read-only open, named 5000 ms busy timeout; no pragma, DDL, DML, `immutable` or `nolock`. Pre-checks `table_info(run)` and `sqlite_master` (missing gives `NoSupervisorRun`). Rank query orders `outcome DESC, entry_id ASC` with `OFFSET rank`; no row gives `RankOutOfRange { rank, len }`. Blob through `decode_brain_state` (`MalformedBlob`); config through strict `BrainConfig` parse (`MalformedConfig`); other errors `Sqlite`.
- **Errors.** `StoreError { Sqlite, Encode, NotFresh, Mismatch, Range, Json }`; `ArchiveLoadError { NoSupervisorRun, RankOutOfRange, MalformedBlob, MalformedConfig, Sqlite }` with `entry_id` on the malformed variants. Each derives `Debug` and implements `Display` and `Error`.
- **Consts.** Busy timeout, pragma string, event kinds (`EVENT_REPLACE` and so on), `MODE_SUPERVISOR`, each with `///` docs. No plan, task or workstream id in comments or test names.
- **Byte-hash proof.** Drop the writer before hashing, so the WAL is checkpointed; compare `DefaultHasher` over `std::fs::read` before and after the load.

## Steps

**Steps:**
1. Add the module line to `lib.rs`. Create `supervisor_store.rs` with the module doc, consts, error enums and `ensure_schema`.
2. Add `SupervisorStore`, `create`, `run_id` and the six write methods exactly as Decisions binds them, each in one transaction.
3. Add `StoredArchiveEntry` and `load_archive_entry_read_only` as Decisions binds them.
4. Add `store_schema_is_idempotent`, `store_create_records_supervisor_mode`, `store_refuses_non_fresh_database` and `store_schema_on_generation_database_keeps_it_resumable`.
5. Add `store_archive_table_mirrors_bounded_archive` and the four `archive_loader_*` tests, including the byte-hash proof and the test-only malformed-blob insert.

- **Done when:** the check focused-proof passes with all ten tests reported ok (store_schema_is_idempotent, store_create_records_supervisor_mode, store_refuses_non_fresh_database, store_schema_on_generation_database_keeps_it_resumable, store_archive_table_mirrors_bounded_archive, archive_loader_returns_rank_order, archive_loader_leaves_database_bytes_unchanged, archive_loader_rejects_rank_beyond_archive, archive_loader_rejects_generation_database, archive_loader_rejects_malformed_blob), and the checks sandbox-clippy, sandbox-fmt and sandbox-guard pass; together they prove INV-06 and INV-11.

## Evidence and risks

- The worker has no shell; the proof is the host-run checks above.
- If a named supervisor-core item is absent or differs at the base, report the file and line as a contradiction; do not edit `supervisor.rs`.
- If the main database hash changes after the load, report both values; do not weaken the test.
