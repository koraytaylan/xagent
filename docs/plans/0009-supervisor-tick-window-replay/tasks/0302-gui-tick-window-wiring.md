---
id: gui-tick-window-wiring
title: "Live and read-only browse playback wiring"
workstream: "0300"
kind: task
role: developer
depends_on: [tick-window-view, replay-jump-request]
gated: false
touches: [crates/xagent-sandbox/src/app.rs, crates/xagent-sandbox/src/gpu_orchestration.rs, crates/xagent-sandbox/src/render_pipeline.rs, crates/xagent-sandbox/src/tick_window_view.rs]
status: done
merged_as: "2b81bd6c75401e2f4f6936279315f50a35607d3f"
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --bin xagent gpu_orchestration::"
    timeout_ms: 3600000
---

# Live and read-only browse playback wiring

## Outcome

The bin-side GUI keeps an in-memory tick window for the live supervisor run,
fed from applied snapshots and supervisor sample pins on the main thread with no
kernel access. A browsed database's supervisor window is loaded once, read-only,
through `BrowseTickWindow::sync` while the 0008 `BrowsingSupervisor` state is
active, and cleared when that state is left. The playback view renders the
browse window when present, else the live window. Each frame, a Replay request
in `ObservatoryViewState.replay_jump_request` is taken by `forward_replay_request`
and passed to the view's `request_jump`.

## Scope

- `gpu_orchestration.rs`: `forward_replay_request`, `BrowseTickWindow` with
  `sync`, `browse_run`, the capture helper, the live window lifecycle, supervisor
  sample pins, and the tests in Steps.
- `app.rs`: fields `tick_window: Option<TickWindow>`, `tick_window_view:
  TickWindowView`, `browse_tick_window: BrowseTickWindow` on `App`, initialised
  to `None`, `TickWindowView::default()` and `BrowseTickWindow::default()` in
  `App::new`; imports; `drive_tick_window` (Step 4).
- `render_pipeline.rs`: one `drive_tick_window` call in the egui closure right
  after the `DockArea` `show_inside`, before `self.evo_snapshot = evo_snap;`.
- `tick_window_view.rs`: only the import move in Step 4.
- `ui.rs`, `main.rs`, `observatory/*.rs` and `tick_window*.rs` are not edited.

## Non-goals

- No `GpuKernel`, kernel, GPU dispatch or readback name in `app.rs` or
  `gpu_orchestration.rs`.
- No database write on browse. The only database read is the `load` closure in
  `BrowseTickWindow::sync`, which runs once per browse session.
- No change to any `EvolutionSnapshot` field, supervisor decision, schedule,
  config default or CLI flag.
- No comment or string naming a plan, task, workstream or packet.

## Steps

**Steps:**
1. In `gpu_orchestration.rs`, add `pub(crate) fn forward_replay_request(
   view_state: &mut ObservatoryViewState, view: &mut TickWindowView)`: `if let
   Some(tick) = view_state.replay_jump_request.take() { view.request_jump(tick); }`.
   Add `pub(crate) struct BrowseTickWindow { pub window: Option<TickWindow>, pub
   loaded_run: Option<i64>, pub loads: u32 }` (derive `Default`) with `fn
   sync(&mut self, db_path: &str, browse_run: Option<i64>, load: impl
   FnMut(&str, Option<i64>) -> Result<TickWindow, TickWindowStoreError>)`:
   `Some(run)` with `loaded_run != Some(run)` calls `load(db_path, Some(run))`
   once, increments `loads`, stores `Some(window)` on `Ok`, and on `Err` logs
   the error and stores an empty `TickWindowConfig::default()` window (`None` if
   its construction fails), then sets `loaded_run = Some(run)`. `Some(run)` with
   `loaded_run == Some(run)` does nothing. `None` sets `window` and `loaded_run`
   to `None`. Add `pub(crate) fn browse_run(evo: &EvolutionSnapshot) ->
   Option<i64>`: `Some(run_id)` only when `evo.state ==
   EvolutionState::BrowsingSupervisor` and `evo.observatory.as_ref().map(|m|
   &m.source)` is `Some(ModelSource::Browse { run_id })`, else `None`.
2. In `gpu_orchestration.rs`, add `pub(crate) fn capture_tick_window_snapshot(
   window: Option<&mut TickWindow>, tick: u64, physics: &[f32], food: &[f32]) ->
   Result<(), TickWindowError>`: `Ok(())` with no effect when `window` is `None`,
   else `frame_from_snapshot` then `TickWindow::capture`. Create the live window
   on supervisor worker start (on error, log and leave it `None`) and set it to
   `None` on supervisor stop; generation mode never creates it. In
   `apply_state_snapshot`, supervisor mode only, call the helper with
   `snapshot.tick, &snapshot.physics, &snapshot.food`; on error, log once and set
   `tick_window = None`.
3. In the `SimEvent::SupervisorSample(report)` arm, pin `PinKind::Replacement {
   slot }` at the sample tick for a non-idle replacement boundary and
   `PinKind::SeasonFlip` for a season flip; idle boundaries pin nothing.
4. In `app.rs`, add the three `App` fields, their `App::new` initialisers and
   imports (`tick_window_store`, `TickWindowConfig`, `BrowseTickWindow`). Add
   `pub(crate) fn drive_tick_window(ui: &mut egui::Ui, evo_snapshot: &mut
   EvolutionSnapshot, db_path: &str, tick_window_view: &mut TickWindowView,
   browse_tick_window: &mut BrowseTickWindow, live: Option<&TickWindow>)`
   which, in this order: (a) `crate::gpu_orchestration::forward_replay_request(
   &mut evo_snapshot.observatory_view, tick_window_view);` (b)
   `browse_tick_window.sync(db_path, crate::gpu_orchestration::browse_run(
   evo_snapshot), |p, r| tick_window_store::load_read_only(p, r,
   TickWindowConfig::default()));` (c) `tick_window_view::render` with
   `browse_tick_window.window` when `Some`, else `live`. It carries no
   `allow(dead_code)`. Each frame, after the `ui.rs` pass, call `crate::app::drive_tick_window(ui,
   &mut evo_snap, &self.db_path, &mut self.tick_window_view, &mut
   self.browse_tick_window, self.tick_window.as_ref())` from the Scope's
   `render_pipeline.rs` site (`evo_snap` is the taken-out `self.evo_snapshot`). In `tick_window_view.rs`, move the unused top-level
   `record_from_physics_row` import into its `mod tests`, its only user, so the
   `clippy` check passes. Change nothing else.
5. In the `#[cfg(test)] mod tests` of `gpu_orchestration.rs`: update
   `gui_jump_request_opens_view_at_requested_tick` to start from
   `ObservatoryViewState::default()` with `replay_jump_request = Some(20)`, a
   window with frames at 10, 20 and 30, and `TickWindowView::default()`; call
   `forward_replay_request`, then `advance(&window, 1)`; assert the current frame
   tick is `Some(20)` and the request is `None`. Keep
   `gui_supervisor_snapshot_is_captured_and_pins_replacement_and_season` (one
   frame at the snapshot tick, two pins) and `gui_generation_mode_captures_nothing`
   (`None` window gives `Ok(())` and creates no window). Add
   `gui_browse_sync_loads_once_and_clears_on_leave`: a counting closure returning
   `Ok` of an empty default-config window; `sync(Some(7))` three times gives
   `loads == 1`, `window` `Some`, `loaded_run == Some(7)`; `sync(None)` gives
   `window` and `loaded_run` `None`; `sync(Some(7))` again gives `loads == 2`.
   Add `gui_browse_run_only_in_browsing_supervisor`: an `EvolutionSnapshot` whose
   observatory source is `ModelSource::Browse { run_id: 7 }` yields `Some(7)`
   only in `EvolutionState::BrowsingSupervisor`, and `None` in other states and
   for a `Live` source. Keep `gui_browse_load_leaves_db_bytes_unchanged`, driven
   through `BrowseTickWindow::sync` with the real
   `tick_window_store::load_read_only` closure: tempfile database, `run` table as
   the 0006 supervisor fixture creates it, `ensure_schema`, one `mode =
   'supervisor'` row whose id is kept; bytes read; `sync(Some(run_id))`; bytes
   read again and equal; second `sync` with the same run leaves `loads == 1`.
6. Extend the existing `gui_supervisor_main_thread_never_touches_kernel` by
   adding the needle `concat!("TickWindow", "Store::open")` to its needle list.
   Keep its files, other needles and assertions; modify no other existing test.

- **Done when:** the check `focused-proof` passes with `gui_supervisor_snapshot_is_captured_and_pins_replacement_and_season`, `gui_generation_mode_captures_nothing`, `gui_jump_request_opens_view_at_requested_tick`, `gui_browse_sync_loads_once_and_clears_on_leave`, `gui_browse_run_only_in_browsing_supervisor`, `gui_browse_load_leaves_db_bytes_unchanged` and `gui_supervisor_main_thread_never_touches_kernel` reported ok and none failing, proving INV-7 (browse load leaves database bytes unchanged), INV-10 (supervisor snapshots and pins captured, generation mode captures nothing, forwarded Replay request opens the view at its tick) and INV-13 (browse window loads once and clears on leaving BrowsingSupervisor).
