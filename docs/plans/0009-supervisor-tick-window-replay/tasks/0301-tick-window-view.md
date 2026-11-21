---
id: tick-window-view
title: "Tick-window playback view"
workstream: "0300"
kind: task
role: developer
depends_on: [tick-window-core]
gated: false
touches: [crates/xagent-sandbox/src/main.rs, crates/xagent-sandbox/src/tick_window_view.rs]
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --bin xagent tick_window_view::"
    timeout_ms: 3600000
---

# Tick-window playback view

## Outcome

`main.rs` gains one line, `mod tick_window_view;`, between `mod snapshot;` and `mod ui_chrome;`. The new bin module `tick_window_view.rs` holds the playback state `TickWindowView`, its pure update rules and an egui `render`. It never touches the GPU or the database and is unit-tested without a GPU.

## Scope

- `main.rs`: only the `mod tick_window_view;` line.
- `tick_window_view.rs` (new, bin-side, `pub(crate)` or private items). Import `xagent_sandbox::tick_window::{record_from_physics_row, Pin, PinKind, TickWindow, WindowFrame}` from `tick-window-core` with its exact signatures. Add no methods to `TickWindow`.
- Constants with `///` docs: `DEFAULT_PLAYBACK_SPEED: u32 = 1` (frames stepped per redraw), `MAX_PLAYBACK_SPEED: u32 = 64` (speed control upper bound), `DISPLAY_DECIMALS: usize = 2` (value label precision).
- `pub(crate) struct TickWindowView { cursor_tick: Option<u64>, playing: bool, speed: u32, selected_agent: Option<usize>, pending_jump: Option<u64> }` with a hand-written `Default` (`None`, `false`, `DEFAULT_PLAYBACK_SPEED`, `None`, `None`).
- `request_jump(&mut self, tick: u64)` sets `pending_jump`.
- `advance(&mut self, window: &TickWindow, frames: u32)`, in order:
  1. Target: `pending_jump.take()`, else `cursor_tick`, else `0`.
  2. Cursor: first frame tick `>= target`, else the last frame tick; `None` for an empty window.
  3. When `playing` and the cursor is `Some`, move to the frame `frames` positions later, or to the last frame. Convert with `usize::try_from(frames).unwrap_or(usize::MAX)`, not `as`.
  4. When the cursor is `None` or the last frame tick, set `playing = false`.
  5. When `selected_agent` is not below the current frame's agent count (0 without a frame), clear it.
- `current_frame<'a>(&self, window: &'a TickWindow) -> Option<&'a WindowFrame>`: the frame whose tick equals `cursor_tick`.
- `render(ui: &mut egui::Ui, view: &mut TickWindowView, window: &TickWindow)`: call `view.advance(window, speed)` with `let speed = view.speed;`; with no frames show "No frames recorded" and return. Otherwise draw a play/pause button, a frame-index scrub slider (a change sets `cursor_tick` from `frames().nth(index)`), a speed slider over `1..=MAX_PLAYBACK_SPEED`, an agent `ComboBox` ("None" plus the current frame's agents), the selected agent's position, energy, integrity, motor forward/turn and alive formatted with `format!("{:.*}", DISPLAY_DECIMALS, v)`, fixed "Phase: not recorded" and "Vision: not recorded" labels, and one row per `window.pins()` entry labelled "Replacement slot N at tick T" or "Season flip at tick T" with `jump_button`. Call `ui.ctx().request_repaint()` while playing or when a jump is pending.
- `jump_button(ui: &mut egui::Ui, tick: u64, label: &str, jump: &mut Option<u64>)`: `ui.button(label)`; on click set `*jump = Some(tick)`.
- Items with no caller yet carry `#[allow(dead_code)]` and a one-line reason. No module-level allow.
- Tests in `#[cfg(test)] mod tests` in the same file. Frames are built as `WindowFrame { tick, agents: vec![record_from_physics_row(&[0.0; xagent_brain::buffers::PHYS_STRIDE])], food: vec![] }`. Use `TickWindow::new(TickWindowConfig::default()).expect(...)` in tests only.

## Non-goals

- No edit to any file other than the two listed. No timeline or boundary jump buttons (GUI wiring task).
- No `GpuKernel`, GPU, SQLite or database write.
- No comment or string naming a plan, task, workstream or packet.

## Steps

**Steps:**
1. In `crates/xagent-sandbox/src/main.rs`, add `mod tick_window_view;` between `mod snapshot;` and `mod ui_chrome;`, with no other change.
2. Create `crates/xagent-sandbox/src/tick_window_view.rs` with the imports, the three constants, `TickWindowView` and its `Default`, `request_jump`, the private helpers for the first frame at or after a tick and the last frame tick, `advance` and `current_frame`.
3. In the same file, add `render` and `jump_button` as described in Scope, with the `#[allow(dead_code)]` reason comments on items without a caller.
4. In the same file, add `#[cfg(test)] mod tests` with the frame helper and the pure tests `cursor_clamped_into_window_on_advance` (cursor `Some(5)` and `Some(100)` on frames 10, 20, 30 become `Some(10)` and `Some(30)`), `play_stops_at_last_frame` (playing from 10 with `frames = 1` reaches 20, then 30 with `playing == false`, and stays), `jump_selects_nearest_at_or_after_and_past_end_goes_last` (request 15 gives 20, request 1000 gives 30, with `playing == false`), `empty_window_yields_no_frame` and `selected_agent_cleared_when_out_of_range` (index 3 on a one-agent frame clears, index 0 stays).
5. In the same test module, add `render_smoke_empty_window` and `render_smoke_populated_window`, each running `egui::Context::default().run(egui::RawInput::default(), ...)` around `render` with `CentralPanel`. The populated window has frames at 10, 20, 30 and one pin of each `PinKind`; after the pass the cursor is `Some(10)`.

- **Done when:** the check `focused-proof` passes with `cursor_clamped_into_window_on_advance`, `play_stops_at_last_frame`, `jump_selects_nearest_at_or_after_and_past_end_goes_last`, `empty_window_yields_no_frame`, `selected_agent_cleared_when_out_of_range`, `render_smoke_empty_window` and `render_smoke_populated_window` reported ok with none failing (INV-9: the cursor is `None` or a recorded frame tick in the window, playback stops at the last frame, and the jump selects the nearest frame at or after the target or the last frame).
