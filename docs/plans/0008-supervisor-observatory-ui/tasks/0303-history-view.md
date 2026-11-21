---
id: history-view
title: Lineage forest, archive browser and event timeline
workstream: "0300"
kind: task
role: developer
depends_on:
  - observatory-model
gated: false
touches:
  - crates/xagent-sandbox/src/observatory/history.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: cargo test -p xagent-sandbox --lib observatory::history::tests
    timeout_ms: 3600000
  - name: sandbox-clippy
    command: cargo clippy -p xagent-sandbox --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: rustfmt --check --edition 2021 crates/xagent-sandbox/src/observatory/history.rs
    fix: rustfmt --edition 2021 crates/xagent-sandbox/src/observatory/history.rs
---

# Lineage forest, archive browser and event timeline

## Outcome

`crates/xagent-sandbox/src/observatory/history.rs` (a stub created by
`observatory-model`) shows who descends from whom, the archive with outcome, age
and config, and the browsable event timeline.

## Decisions (binding)

- `pub struct LineageNode { pub occupant_id: u64, pub depth: usize, pub
  parent_evicted: bool }`; `pub fn lineage_forest(&ObservatoryModel) ->
  Vec<LineageNode>` in pre-order. Roots are occupants with no parent or whose
  parent is absent from `occupants` (then `parent_evicted = true`); children are
  sorted by `occupant_id`. It is iterative with an explicit stack and a visited
  set, so each occupant appears exactly once and a malformed cycle terminates
  (members of an unrooted cycle are emitted as roots, smallest id first).
- `pub fn archive_rows(&ObservatoryModel) -> Vec<&ArchiveRow>` sorted by outcome
  descending with NaN last, ties by `entry_id`.
- `pub struct TimelineFilter { pub kinds: Vec<TimelineKind>, pub from_tick:
  Option<u64>, pub to_tick: Option<u64> }` (empty `kinds` means all);
  `pub fn filtered<'a>(model: &'a ObservatoryModel, filter: &TimelineFilter) ->
  Vec<&'a TimelineEvent>`, newest first, inclusive tick range.
  `pub fn truncate_display(&str) -> &str` cuts to `MAX_DISPLAY_TEXT_BYTES` on a
  char boundary and is used for every displayed text.
- Renderers `show_lineage(ui, &ObservatoryModel, &mut LineageViewState)`
  (labels indented by depth, a `(parent evicted)` marker), `show_archive(ui,
  &ObservatoryModel)` (`egui::Grid`: entry, occupant, outcome, age, and a
  collapsing config summary) and `show_timeline(ui, &ObservatoryModel, &mut
  TimelineViewState)` (kind toggles, tick range fields, then
  `egui::ScrollArea::vertical().show_rows` over `filtered`). View states derive
  `Clone, Debug, Default`.
- No `GpuKernel`, `wgpu`, `rusqlite`, `Connection`, `store_read::` or
  `read_full_state_blocking` name, no planning references, `TODO`, `FIXME` or
  `as` integer casts.

**Steps:**

1. Read `observatory/mod.rs` for the model types; report a missing one as a blocker.
2. Write `lineage_forest` and `archive_rows`.
3. Write `TimelineFilter`, `filtered` and `truncate_display`.
4. Write the three renderers and their view states.
5. Add `#[cfg(test)] mod tests` with `history_lineage_each_occupant_once` (a
   three-generation family plus an evicted-parent root),
   `history_lineage_cycle_terminates`, `history_archive_sorted_by_outcome` (NaN
   last), `history_timeline_filter_kind_and_range` (newest first, inclusive
   bounds) and `history_truncate_on_char_boundary` (multi-byte text over the cap).
6. Make the three checks pass.

- **Done when:** `focused-proof` passes with `history_lineage_each_occupant_once`, `history_lineage_cycle_terminates`, `history_archive_sorted_by_outcome`, `history_timeline_filter_kind_and_range` and `history_truncate_on_char_boundary` reported ok, proving INV-9 and INV-11.
