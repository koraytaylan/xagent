---
id: observatory-panel
title: Observatory panel, session dialog states and redraw purity test
workstream: "0400"
kind: task
role: developer
depends_on:
  - tables-view
  - plots-view
  - history-view
  - observatory-store-read
gated: false
touches:
  - crates/xagent-sandbox/src/ui.rs
  - crates/xagent-sandbox/src/ui_chrome.rs
  - crates/xagent-sandbox/tests/observatory_purity.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: cargo test -p xagent-sandbox --test observatory_purity
    timeout_ms: 3600000
  - name: ui-tests
    command: cargo test -p xagent-sandbox --lib ui::tests
    timeout_ms: 3600000
  - name: snapshot-default
    command: cargo test -p xagent-sandbox --test integration -- --exact evolution_snapshot_default_tree_pane_fraction
    timeout_ms: 3600000
  - name: sandbox-clippy
    command: cargo clippy -p xagent-sandbox --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: rustfmt --check --edition 2021 crates/xagent-sandbox/src/ui.rs crates/xagent-sandbox/src/ui_chrome.rs crates/xagent-sandbox/tests/observatory_purity.rs
    fix: rustfmt --edition 2021 crates/xagent-sandbox/src/ui.rs crates/xagent-sandbox/src/ui_chrome.rs crates/xagent-sandbox/tests/observatory_purity.rs
  - name: sandbox-guard
    command: cargo test -p xagent-sandbox --test contributing_guard
    timeout_ms: 3600000
---

# Observatory panel, session dialog states and redraw purity test

## Outcome

`ui.rs` gains the observatory snapshot fields, `SessionChoice`, the
`SupervisorSession` and `BrowsingSupervisor` states, `render_observatory` with a
header and seven tabs, and the "Supervisor run found" dialog; the dashboard
renders the observatory instead of the six-label supervisor section.
`ui_chrome.rs` handles the two new states. `tests/observatory_purity.rs` scans
the redraw sources for GPU, readback and database names.

## Decisions (binding)

- Names come from the landed files, each `pub`: `ObservatoryModel` (`Default`),
  `tables::{show_slots, show_boundaries, show_header, SlotsViewState,
  BoundaryViewState}`, `plots::{show_charts, show_learning, LearningViewState}`,
  `history::{show_lineage, show_archive, show_timeline, LineageViewState,
  TimelineViewState}`, `store_read::SupervisorRunSummary`. A missing one is a
  blocker reported with file and line.
- In `ui.rs`: `#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)] pub enum
  ObservatoryTab { #[default] Slots, Boundaries, Charts, Learning, Lineage,
  Archive, Timeline }`; `#[derive(Clone, Debug, Default)] pub struct
  ObservatoryViewState` with `tab` and one field per view state;
  `#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)] pub enum SessionChoice {
  #[default] None, ResumeLive, Browse }`. `EvolutionSnapshot` gains `observatory:
  Option<ObservatoryModel>`, `observatory_view`, `supervisor_session:
  Option<SupervisorRunSummary>`, `supervisor_resume_error: Option<String>` and
  `session_choice: SessionChoice`, defaulting to `None`, default, `None`, `None`
  and `SessionChoice::None`. `EvolutionState` gains unit variants
  `SupervisorSession` and `BrowsingSupervisor`. No new `EvolutionAction` variant.
- `pub fn render_observatory(ui: &mut egui::Ui, model: &ObservatoryModel, view:
  &mut ObservatoryViewState)`; callers pass `evo.observatory.as_ref()` (or a
  local default model, no per-frame clone of a model) and `&mut
  evo.observatory_view`. It calls `show_header`, a separator,
  a `horizontal_wrapped` row of `selectable_value` tabs, a separator, then the
  selected view. `render_running_dashboard` replaces the whole `if let Some(sup) =
  &evo.supervisor` block with a call to `render_observatory` when
  `evo.supervisor.is_some()`. Generation charts, tree, node detail and
  `SupervisorSnapshot` are unchanged.
- `SupervisorSession` renders the dialog: run id, last tick, budget, buttons
  "Resume live" and "Browse read-only" that set `evo.session_choice`, the resume
  error (if any) truncated to `MAX_DISPLAY_TEXT_BYTES`, and no Reset.
  `BrowsingSupervisor` renders "Browsing a stored supervisor run (read-only)."
  and `render_observatory`, with no Pause or Reset.
- `ui_chrome.rs` `draw_top_bar` match: `SupervisorSession` shows "Supervisor run
  found"; `BrowsingSupervisor` shows "Browsing (read-only)". Pause and Reset keep
  their `Running | Paused` guards.
- Purity test (std only): `include_str!` of `src/observatory/tables.rs`,
  `plots.rs`, `history.rs` and `src/ui.rs`. Needles are built with `concat!` so
  the test file holds none literally: for the three view files `GpuKernel`,
  `wgpu`, `rusqlite`, `Connection`, `store_read::`, `read_full_state_blocking`;
  for `ui.rs` `store_read::load_run`, `store_read::detect_supervisor_run`,
  `store_read::open_read_only`, `GpuKernel`, `read_full_state_blocking`.
- Edited files hold no planning reference, `TODO`, `FIXME` or emoji.

**Steps:**

1. Read the observatory files and the relevant `ui.rs` and `ui_chrome.rs`
   regions; report an absent name as a blocker.
2. Add the types, snapshot fields and states to `ui.rs`.
3. Add `render_observatory`, the session dialog and the browse state rendering,
   and replace the six-label block.
4. Add the two `ui_chrome.rs` arms.
5. Add `ui::tests` `observatory_snapshot_default_has_no_observatory`,
   `observatory_dialog_buttons_set_session_choice` (clicking is simulated by
   calling the dialog's pure `choice_for(button)` mapping) and
   `observatory_dashboard_renders_without_six_labels` (render inside
   `egui::Context::default().run` with `supervisor: Some(..)`, collect galley text
   and assert no `"Replacements: "`, `"Last replacement: "`, `"Nourishing kind: "`
   or `"Archive: "` prefix; generation mode still renders the fitness chart
   heading), then create `tests/observatory_purity.rs` with
   `observatory_sources_name_no_gpu_readback_or_database_api`.
6. Make the six checks pass.

- **Done when:** `focused-proof` passes with `observatory_sources_name_no_gpu_readback_or_database_api` reported ok; `ui-tests` passes with `observatory_snapshot_default_has_no_observatory`, `observatory_dialog_buttons_set_session_choice` and `observatory_dashboard_renders_without_six_labels` reported ok; and `snapshot-default` passes with `evolution_snapshot_default_tree_pane_fraction` reported ok, proving INV-4 (redraw part), INV-12 (dialog part) and INV-13.
