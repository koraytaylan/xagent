---
id: tables-view
title: Slot table, boundary inspector, progress header and season strip
workstream: "0300"
kind: task
role: developer
depends_on:
  - observatory-model
gated: false
touches:
  - crates/xagent-sandbox/src/observatory/tables.rs
status: done
merged_as: "e23c9b96ab5cdcc98a9d05479b71527638ba1ae3"
checks:
  - name: focused-proof
    command: cargo test -p xagent-sandbox --lib observatory::tables::tests
    timeout_ms: 3600000
  - name: sandbox-clippy
    command: cargo clippy -p xagent-sandbox --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: rustfmt --check --edition 2021 crates/xagent-sandbox/src/observatory/tables.rs
    fix: rustfmt --edition 2021 crates/xagent-sandbox/src/observatory/tables.rs
---

# Slot table, boundary inspector, progress header and season strip

## Outcome

`crates/xagent-sandbox/src/observatory/tables.rs` (a stub created by
`observatory-model`) shows the slot table, the boundary inspector and the
run-progress header with the season strip from `&ObservatoryModel`. Every value
shown comes from a pure helper with unit tests.

## Decisions (binding)

- Pure helpers, all `pub`:
  - `slot_rows(&ObservatoryModel) -> Vec<[String; 7]>` with columns occupant,
    age, `alive`/`dead`, meals, deaths, score against median and parent. The
    score column is `"{score:.3} ({delta:+.3})"` with `delta = score - median`, or
    `n/a` when either is `None` or non-finite. Parent is `n/a`, `archive #<id>
    (<birth|learned>)` or `peer slot <n> (<birth|learned>)`.
  - `boundary_lines(&BoundaryRecord) -> Vec<String>`: for a replacement one line
    per eligible slot (`slot <n>: died` or `slot <n>: score <s> below median
    <m>`, three decimals), then `picked slot <n>`, `tie-break: <text>` when
    present, the parent line as above, and one `<field>: <parent> -> <child>`
    line per config delta; for an idle boundary exactly one line `idle: <reason>`.
  - `progress_text(&Progress) -> String`: `tick <t> / <budget|unlimited>`,
    `<x> ticks/s` (one decimal) only when `ticks_per_sec` is `Some` (live), and
    `next boundary in <n> ticks` using `saturating_sub` (omitted when `None` or
    `finished`).
  - `season_text(Option<&SeasonState>, tick: u64) -> String`: `nourishing: <kind>`
    and `next flip in <n> ticks` with `saturating_sub`; `season: n/a` for `None`.
- Renderers `show_slots(ui, &ObservatoryModel, &mut SlotsViewState)` (a striped
  `egui::Grid`; `SlotsViewState { selected: Option<usize> }`),
  `show_boundaries(ui, &ObservatoryModel, &mut BoundaryViewState)` (newest first,
  one `CollapsingHeader` per boundary showing `boundary_lines`;
  `BoundaryViewState { selected_tick: Option<u64> }`) and `show_header(ui,
  &ObservatoryModel)` (one wrapped row of `progress_text` and `season_text`).
  View states derive `Clone, Debug, Default`.
- No `ScrollArea` inside these renderers; the caller owns scrolling (CLAUDE.md
  egui notes). No `GpuKernel`, `wgpu`, `rusqlite`, `Connection`, `store_read::`
  or `read_full_state_blocking` name. No planning references, `TODO`, `FIXME` or
  `as` integer casts.

**Steps:**

1. Read `observatory/mod.rs` for the model types; report a missing one as a blocker.
2. Write the four pure helpers in `tables.rs`.
3. Write the three renderers and the two view state types.
4. Add `#[cfg(test)] mod tests` with `tables_slot_rows_show_every_column` (a
   living and a dead slot, signed delta, `n/a` for a missing median, both parent
   sources), `tables_boundary_lines_replacement` (died and below-median grounds
   with numbers, picked slot, tie-break, parent, every delta),
   `tables_boundary_lines_idle` (one idle line),
   `tables_progress_text_never_underflows` (next boundary before the tick gives 0,
   unlimited budget, no ticks/s when `None`) and `tables_season_text_counts_to_flip`.
5. Make the three checks pass.

- **Done when:** `focused-proof` passes with `tables_slot_rows_show_every_column`, `tables_boundary_lines_replacement`, `tables_boundary_lines_idle`, `tables_progress_text_never_underflows` and `tables_season_text_counts_to_flip` reported ok, proving INV-5, INV-6 and INV-10.
