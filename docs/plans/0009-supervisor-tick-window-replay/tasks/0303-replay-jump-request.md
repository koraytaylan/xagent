---
id: replay-jump-request
title: "Replay buttons on timeline and boundary rows"
workstream: "0300"
kind: task
role: developer
depends_on: []
gated: false
touches:
  - crates/xagent-sandbox/src/observatory/history.rs
  - crates/xagent-sandbox/src/observatory/tables.rs
  - crates/xagent-sandbox/src/ui.rs
status: done
merged_as: "1b01f9e7f0d874900c4fc66f8b26164a5e904e46"
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --lib observatory::"
    timeout_ms: 3600000
  - name: ui-proof
    command: "cargo test -p xagent-sandbox --lib ui::tests::observatory_"
    timeout_ms: 3600000
---

# Replay buttons on timeline and boundary rows

## Outcome

A "Replay" button on every rendered supervisor timeline event row and every
rendered boundary inspector row records that row's tick in the view state, and
`ui.rs` `render_observatory` moves that request into
`ObservatoryViewState.replay_jump_request` so the bin side can open the
playback view at that tick. Clicking nothing leaves the request `None`.

## Scope

- `crates/xagent-sandbox/src/observatory/history.rs`: `TimelineViewState`
  gains `pub replay_request: Option<u64>` (`None` through its `Default`
  derive). In `show_timeline(ui, model, state: &mut TimelineViewState)`, every
  rendered `TimelineEvent` row, of every `TimelineKind`, draws
  `ui.small_button("Replay")`; a click sets `state.replay_request =
  Some(event.tick)`. `filtered` and the row text are unchanged.
- `crates/xagent-sandbox/src/observatory/tables.rs`: `BoundaryViewState` gains
  the same `pub replay_request: Option<u64>`. In `show_boundaries(ui, model,
  state: &mut BoundaryViewState)`, every rendered `BoundaryRecord`, replacement
  and idle alike, draws `ui.small_button("Replay")`; a click sets
  `state.replay_request = Some(record.tick)`. `boundary_lines` is unchanged.
- `crates/xagent-sandbox/src/ui.rs`: `ObservatoryViewState` gains `pub
  replay_jump_request: Option<u64>` (`None` by default). In
  `render_observatory(ui, model, view: &mut ObservatoryViewState)`, after the
  selected view has drawn, `if let Some(t) = <timeline view
  state>.replay_request.take() { view.replay_jump_request = Some(t); }`, then
  the same for the boundary view state (a boundary move in the same frame wins).
  The per-view field names are those plan 0008 gave inside
  `ObservatoryViewState`; the new field names above are fixed.
- Tests, driven by headless egui (`egui::Context::run` with `egui::RawInput`,
  no GPU, no window):
  - `history.rs` `mod tests`: `timeline_replay_click_records_row_tick`. One
    timeline event at tick `4242`. Frame 1 renders with no input so the button
    has a rect; frame 2 sends `egui::Event::PointerMoved` to the button centre,
    frame 3 `egui::Event::PointerButton` with `pressed: true`, frame 4 the same
    with `pressed: false`. Assert `state.replay_request == Some(4242)`.
  - `tables.rs` `mod tests`: `boundary_replay_click_records_row_tick`, the same
    sequence with one boundary record at tick `5151`.
  - `ui.rs` `mod tests`: `observatory_render_moves_replay_request`. Pre-set the
    timeline view's `replay_request` to `Some(4242)`, run one
    `render_observatory` frame with no input, assert `replay_jump_request ==
    Some(4242)` and the timeline field `None`; a fresh state with no request
    keeps `replay_jump_request == None`.
- The test finds the button rect with an egui 0.31 accessor for a widget's
  last-frame rect, confirmed in the egui 0.31 source. No product-code hook,
  extra `pub` item or `cfg(test)` branch is added.

## Non-goals

- No edit to `app.rs`, `main.rs` or any bin module; the bin-side forwarding of
  `replay_jump_request` is `gui-tick-window-wiring`.
- No change to `filtered`, `boundary_lines`, the charts, lineage, archive,
  session dialog or header, and no new UI besides the two Replay buttons.
- No `GpuKernel`, `wgpu`, `rusqlite`, `Connection` or `store_read` name in the
  edited observatory files, and none of the `ui.rs` needles of plan 0008's
  purity test (0008 INV-4).
- No change to the existing 0008 tests' expectations; no comment or string
  naming a plan, task, workstream or packet.

**Steps:**

1. In `history.rs` add `replay_request` to `TimelineViewState` and the Replay
   button on every rendered timeline row in `show_timeline`.
2. In `tables.rs` add `replay_request` to `BoundaryViewState` and the Replay
   button on every rendered boundary record in `show_boundaries`.
3. In `ui.rs` add `replay_jump_request` to `ObservatoryViewState` and the move
   after the view draw in `render_observatory`.
4. Add `timeline_replay_click_records_row_tick` to `history.rs` and
   `boundary_replay_click_records_row_tick` to `tables.rs`.
5. Add `observatory_render_moves_replay_request` to `ui.rs` `mod tests`.

- **Done when:** `focused-proof` passes with `timeline_replay_click_records_row_tick` and `boundary_replay_click_records_row_tick` reported ok and `ui-proof` passes with `observatory_render_moves_replay_request` reported ok, none failing, which proves INV-12: a simulated click on a timeline or boundary row's Replay button records that row's tick in `ObservatoryViewState.replay_jump_request` after `render_observatory`, and no click leaves it `None`.
