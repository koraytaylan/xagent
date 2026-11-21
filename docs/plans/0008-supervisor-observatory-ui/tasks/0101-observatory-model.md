---
id: observatory-model
title: Bounded ObservatoryModel and fold from plan 0006 records
workstream: "0100"
kind: task
role: developer
depends_on: []
gated: false
touches:
  - crates/xagent-sandbox/src/lib.rs
  - crates/xagent-sandbox/src/observatory/mod.rs
  - crates/xagent-sandbox/src/observatory/store_read.rs
  - crates/xagent-sandbox/src/observatory/tables.rs
  - crates/xagent-sandbox/src/observatory/plots.rs
  - crates/xagent-sandbox/src/observatory/history.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: cargo test -p xagent-sandbox --lib observatory::tests
    timeout_ms: 3600000
  - name: sandbox-clippy
    command: cargo clippy -p xagent-sandbox --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: cargo fmt -p xagent-sandbox -- --check
    fix: cargo fmt -p xagent-sandbox
  - name: sandbox-guard
    command: cargo test -p xagent-sandbox --test contributing_guard
    timeout_ms: 3600000
---

# Bounded ObservatoryModel and fold from plan 0006 records

## Outcome

`crates/xagent-sandbox/src/observatory/mod.rs` holds a pure, bounded
`ObservatoryModel` that folds one `ObservatoryRecord` at a time and declares the
four submodules `store_read`, `tables`, `plots` and `history`, each created as a
one-line doc stub that exactly one later task owns. Every collection obeys its
named cap under any number of folded records.

## Decisions (binding for this task)

- Caps are `pub const` in `mod.rs` with these exact names and values:
  `MAX_SLOTS: usize = 256`, `MAX_SERIES_POINTS: usize = 4096`,
  `MAX_BOUNDARY_RECORDS: usize = 4096`, `MAX_TIMELINE_EVENTS: usize = 20_000`,
  `MAX_TRACKED_OCCUPANTS: usize = 4096`, `MAX_INTAKE_POINTS: usize = 512`,
  `MAX_ARCHIVE_ROWS: usize = 1024`, `MAX_DISPLAY_TEXT_BYTES: usize = 256` and
  `DECIMATION_STRIDE: usize = 2`, each with a doc comment saying why. Every other
  numeric literal other than `0` and `1` is a named, documented const.
- Types and fields follow ARCHITECTURE Interfaces exactly. `ModelSource` derives
  `Default` with `#[default] Live`; `ObservatoryModel` and `Progress` derive
  `Default` (`season` is `None` by default).
- `enum ObservatoryRecord` variants: `Slot(SlotRow)`, `Boundary(BoundaryRecord)`,
  `Series(SeriesPoint)`, `Occupant(OccupantRecord)` (upserts parent, birth,
  retirement and late window and keeps existing intake), `Intake { occupant_id:
  u64, age_ticks: u64, meals: u32 }`, `Archive(ArchiveRow)`,
  `Season(SeasonState)`, `Progress(Progress)`, `Event(TimelineEvent)`. `From`
  impls in `mod.rs` convert each plan 0006 record type the live fold receives,
  field to field, using `u32::try_from(..).unwrap_or(u32::MAX)` for narrowing and
  never an `as` integer cast.
- Fold rules:
  - `Slot`: ignored when `slot >= MAX_SLOTS`; otherwise replace the row with the
    same `slot` or push it.
  - `Boundary` and `Event`: push back; above the cap pop the front. Event `text`,
    `tie_break` and `idle_reason` are truncated to `MAX_DISPLAY_TEXT_BYTES` on a
    char boundary.
  - `Series` and `Intake`: push; above the cap keep every element whose index is a
    multiple of `DECIMATION_STRIDE` plus the last element.
  - `Occupant`: upsert by id. A new occupant at the cap evicts the retired
    occupant with the smallest `(retired_tick, occupant_id)`; a living occupant is
    never evicted; with no retired occupant the new record is dropped. `Intake`
    for an untracked occupant is ignored.
  - `Archive`: upsert by `entry_id`; at the cap a new entry with a higher outcome
    (`f32::total_cmp`) replaces the lowest-outcome row (ties: lowest `entry_id`),
    otherwise it is dropped.
  - `Season` and `Progress`: replace.
- The stubs `store_read.rs`, `tables.rs`, `plots.rs` and `history.rs` each hold
  exactly one `//!` line naming the file's purpose, with no plan, task or INV
  reference.

## Prerequisites and blocker rule

The `From` conversions read the plan 0006 record types and their `SimEvent`
variants from `supervisor_run.rs`, `supervisor_store.rs` and `sim_runtime.rs` at
implementation time. If a field ARCHITECTURE requires is absent, stop and report
the file and line as a blocker; do not invent it.

**Steps:**

1. In `crates/xagent-sandbox/src/lib.rs` add `pub mod observatory;` between
   `pub mod momentum;` and `pub mod overlay;`.
2. Create the four one-line stub files named in `touches`.
3. Create `crates/xagent-sandbox/src/observatory/mod.rs` with the submodule
   declarations, the caps, the model types, `ObservatoryRecord`, the `From`
   conversions and `ObservatoryModel::fold` implementing every rule above.
4. Add `#[cfg(test)] mod tests` in `mod.rs` with `observatory_model_caps_hold`
   (fold more than each cap of every kind, assert each length is at most its cap
   and a slot at `MAX_SLOTS` is ignored), `observatory_model_decimation_keeps_ends`
   (ticks 0..N-1 beyond the series and intake caps keep tick 0 and tick N-1),
   `observatory_model_living_never_evicted` (living occupants survive, the retired
   one is evicted) and `observatory_model_fold_basic` (one record of each kind
   lands in its field).
5. Keep the code free of planning references, `TODO`/`FIXME`, `as` integer casts
   and magic numbers, and make the four checks pass.

- **Done when:** `focused-proof` passes with `observatory::tests::observatory_model_caps_hold`, `observatory_model_decimation_keeps_ends`, `observatory_model_living_never_evicted` and `observatory_model_fold_basic` reported ok, proving INV-1.
