# Architecture: supervisor tick-window replay

## Chosen interfaces

`crates/xagent-sandbox/src/tick_window.rs` (library, `pub mod tick_window;`):

- Named bounds, all `pub const`:
  `DEFAULT_TICK_WINDOW_TICKS: u64 = 20_000` (the rolling window),
  `MAX_WINDOW_FRAMES: usize = 4_096` (rolling frames held at once),
  `PIN_HALF_WINDOW_TICKS: u64 = 2_000` (a pin keeps `[t - H, t + H]`),
  `MAX_PINNED_WINDOWS: usize = 32`, `MAX_FRAMES_PER_PIN: usize = 512`,
  `MAX_FRAME_AGENTS: usize = 1_024`, `MAX_FRAME_FOOD: usize = 65_536`,
  `PHASE_NOT_RECORDED: u8 = 0`.
- `pub struct TickWindowConfig { pub window_ticks: u64, pub max_frames: usize,
  pub pin_half_window_ticks: u64, pub max_pins: usize, pub max_frames_per_pin:
  usize }` with `Default` built from the constants above and
  `validate(&self) -> Result<(), TickWindowError>` refusing zero bounds.
- `pub enum PinKind { Replacement { slot: usize }, SeasonFlip }` and
  `pub struct Pin { pub tick: u64, pub kind: PinKind }`.
- `pub struct WindowFrame { pub tick: u64, pub agents: Vec<replay::TickRecord>,
  pub food: Vec<[f32; 4]> }`.
- `pub fn record_from_physics_row(row: &[f32]) -> replay::TickRecord`: position
  (`P_POS_*`), yaw, alive (`P_ALIVE > 0.5`), energy, integrity, motor forward
  and turn, exploration rate, prediction error, gradient, raw gradient,
  urgency, fatigue factor from the `P_*` slots; every field the row does not
  carry is at its not-recorded value (`phase = PHASE_NOT_RECORDED`,
  `vision_color = None`, zeros elsewhere).
- `pub fn frame_from_snapshot(tick: u64, physics: &[f32], food: &[f32]) ->
  Result<WindowFrame, TickWindowError>` (rejects a physics length not a
  multiple of `PHYS_STRIDE`, a food length not a multiple of
  `FOOD_STATE_STRIDE`, or counts past `MAX_FRAME_AGENTS` / `MAX_FRAME_FOOD`).
- `pub struct TickWindow` with `new(TickWindowConfig)`, `capture(&mut self,
  frame: WindowFrame)` (ignores a frame whose tick is not greater than the
  newest), `pin(&mut self, pin: Pin)`, `frames(&self) -> impl Iterator<Item =
  &WindowFrame>` in tick order, `pins(&self) -> &[Pin]`, `frames_in(&self,
  start: u64, end: u64)`, `newest_tick`, `len`.
- `pub fn encode_frame(frame: &WindowFrame) -> Vec<u8>` and
  `pub fn decode_frame(bytes: &[u8]) -> Result<WindowFrame, TickWindowError>`:
  little-endian, leading `FRAME_CODEC_VERSION: u32 = 1`.

`crates/xagent-sandbox/src/tick_window_store.rs` (library):

- `pub fn ensure_schema(db: &Connection) -> SqlResult<()>`: `CREATE TABLE IF
  NOT EXISTS supervisor_replay_frame (run_id INTEGER NOT NULL, tick INTEGER NOT
  NULL, frame BLOB NOT NULL, PRIMARY KEY (run_id, tick))` and
  `supervisor_replay_pin (run_id INTEGER NOT NULL, tick INTEGER NOT NULL, kind
  TEXT NOT NULL, slot INTEGER, PRIMARY KEY (run_id, tick, kind))`.
- `pub struct TickWindowStore`: `open_for_latest_supervisor_run(db_path: &str,
  config: TickWindowConfig) -> Result<Self, TickWindowStoreError>` (latest
  `run` row with `mode = 'supervisor'`), `record_frame(&mut self, &WindowFrame)`,
  `record_pin(&mut self, Pin)`; every write prunes in the same transaction.
- `pub fn load_read_only(db_path: &str, run_id: Option<i64>, config:
  TickWindowConfig) -> Result<TickWindow, TickWindowStoreError>` opening with
  `SQLITE_OPEN_READ_ONLY`.

`crates/xagent-sandbox/src/headless.rs`: `pub fn
run_headless_supervisor_with_tick_window(config: FullConfig, db_path: &str,
options: &HeadlessOptions, tick_window: Option<TickWindowConfig>) -> Result<(),
HeadlessError>`; `run_supervisor` gains the same optional argument and
`run_headless_with_options` passes `None`. `HeadlessOptions` gains no field.

`crates/xagent-sandbox/src/tick_window_view.rs` (bin, `mod tick_window_view;`
in `main.rs`): `pub(crate) struct TickWindowView { cursor_tick, playing,
speed, selected_agent, pending_jump }`, `request_jump(tick)`,
`advance(&mut self, window: &TickWindow, frames: u32)`, and
`render(ui: &mut egui::Ui, view: &mut TickWindowView, window: &TickWindow)`;
`pub(crate) fn jump_button(ui, tick, label, jump: &mut Option<u64>)`.

Jump route (task `replay-jump-request`). The 0008 library row renderers own
the button, since only they draw the rows; `ui.rs` moves the request:

- `observatory/history.rs`: `TimelineViewState` gains `pub replay_request:
  Option<u64>`; `show_timeline` draws `ui.small_button("Replay")` on every
  rendered `TimelineEvent` row, and a click sets `state.replay_request =
  Some(event.tick)`. Every row kind gets the button.
- `observatory/tables.rs`: `BoundaryViewState` gains `pub replay_request:
  Option<u64>`; `show_boundaries` draws the same button on every rendered
  `BoundaryRecord` (replacement and idle), setting `Some(record.tick)`.
- `ui.rs`: `ObservatoryViewState` gains `pub
  replay_jump_request: Option<u64>` (`None` by default); after the timeline and
  boundary views draw, `render_observatory` moves each view's `replay_request.take()`
  into it (a later move in the same frame wins).
- The view state types are the ones 0008 passes as each `show`'s `state`
  argument (`<View>State`); if 0008 named them differently the task uses
  0008's names, the field names above stay fixed.
- Bin side (`gpu_orchestration.rs`): `pub(crate) fn
  forward_replay_request(view_state: &mut ObservatoryViewState, view: &mut
  TickWindowView)` takes `replay_jump_request` and calls
  `TickWindowView::request_jump`; `App` calls it on
  `self.evo_snapshot.observatory_view` each frame after the UI pass.

Browse route (task `gui-tick-window-wiring`): `pub(crate) struct
BrowseTickWindow { pub window: Option<TickWindow>, pub loaded_run:
Option<i64>, pub loads: u32 }` with `fn sync(&mut self, db_path: &str,
browse_run: Option<i64>, load: impl FnMut(&str, Option<i64>) ->
Result<TickWindow, TickWindowStoreError>)`: with `Some(run)` and
`loaded_run != Some(run)` it calls `load` once, increments `loads`, stores the
window (or an empty default window on error) and sets `loaded_run`; with
`None` it clears `window` and `loaded_run`. `pub(crate) fn browse_run(evo:
&EvolutionSnapshot) -> Option<i64>` returns `Some(run_id)` only when
`evo.state == EvolutionState::BrowsingSupervisor` and
`evo.observatory.as_ref().map(|m| &m.source)` is `ModelSource::Browse { run_id
}`. `App` holds `browse_tick_window: BrowseTickWindow` and calls
`sync(&self.db_path, browse_run(&self.evo_snapshot),
|p, r| tick_window_store::load_read_only(p, r, TickWindowConfig::default()))`
each frame; the database is read only inside `load`, so it runs once per
browse session.

## Invariants

| id | invariant | guards | enforced at |
|---|---|---|---|
| INV-1 | Capture never alters a seeded supervisor run: the 0006 supervisor digest, every non-replay table row and the run summary are identical with capture on and off. | determinism of decisions | `crates/xagent-sandbox/tests/supervisor_tick_window_digest.rs` (`headless-tick-window`) |
| INV-2 | Capture adds no GPU dispatch or readback: frames are built only from physics/food slices already copied (headless sample copies, worker published snapshot, main-thread applied snapshot); `tick_window*.rs` and `tick_window_view.rs` never name `GpuKernel`. | GPU budget, CLAUDE.md redraw rule | unit test in `tick_window.rs` scanning `include_str!` sources (`tick-window-core`); probe-count assertion in the digest test (`headless-tick-window`); worker test (`worker-tick-window`) |
| INV-3 | In-memory retention is bounded: unpinned frames older than `newest - window_ticks` are dropped and at most `max_frames` rolling frames, `max_pins` pins (oldest evicted) and `max_frames_per_pin` frames per pin are held. | memory | unit tests in `tick_window.rs` (`tick-window-core`) |
| INV-4 | On-disk retention has the same bounds: after every write, `supervisor_replay_frame` holds only frames inside the rolling window or a retained pin's range, and `supervisor_replay_pin` at most `max_pins` rows per run. | disk | unit tests in `tick_window_store.rs` (`tick-window-core`) |
| INV-5 | Frames inside `[t - H, t + H]` of every retained replacement-boundary or season-flip pin survive rolling eviction, both in memory and on disk, and pins are recorded only for replacements (not idle boundaries) and season flips. | replay of inspector/timeline events | `tick_window.rs` and `tick_window_store.rs` tests (`tick-window-core`); digest test pinned-frame assertion (`headless-tick-window`) |
| INV-6 | The schema change is additive and idempotent: `ensure_schema` twice succeeds, and a database written before it (generation or supervisor) loads unchanged. | serialization compatibility | `tick_window_store.rs` tests (`tick-window-core`) |
| INV-7 | Browsing never writes: `load_read_only` opens read-only and the database file bytes are unchanged after loading. | user data | `tick_window_store.rs` test (`tick-window-core`); `gpu_orchestration.rs` browse test (`gui-tick-window-wiring`) |
| INV-8 | Untrusted frame blobs and rows are bounded: a wrong version, truncated blob, oversized count or non-finite position is an error, never a panic or unbounded allocation. | robustness | codec tests in `tick_window.rs` (`tick-window-core`) |
| INV-9 | Playback is pure and clamped: the cursor stays within the window's first and last frame, play stops at the end, a jump selects the nearest frame at or after the target (or the last), and an out-of-range selected agent is cleared. | UI correctness | `tick_window_view.rs` tests (`tick-window-view`) |
| INV-10 | The live window is fed from applied snapshots and supervisor sample pins on the main thread with no kernel access, and a forwarded jump request opens the view at that tick. | live playback | `gpu_orchestration.rs` tests (`gui-tick-window-wiring`) |
| INV-12 | A simulated click on the Replay button of a timeline event row and of a boundary record row records that row's tick in `ObservatoryViewState.replay_jump_request` after `render_observatory`; no click leaves it `None`. | jump from timeline/boundary | egui click tests in `observatory/history.rs`, `observatory/tables.rs`, `ui.rs` (`replay-jump-request`) |
| INV-13 | Entering `BrowsingSupervisor` loads the browsed run's window exactly once however many frames pass, and leaving it clears the window. | browse playback | `gpu_orchestration.rs` `BrowseTickWindow` test (`gui-tick-window-wiring`) |
| INV-11 | The worker persists a frame for each snapshot it publishes and a pin for each replacement or season-flip sample when supervisor mode has a database; a store failure is logged once and capture stops, the run continues. | live persistence | `sim_runtime.rs` tests (`worker-tick-window`) |

## State

| state | owner | lifecycle | concurrency |
|---|---|---|---|
| headless `TickWindowStore` | `run_supervisor` | opened after `SupervisorRun::start` when capture is on; dropped at the end | single thread |
| worker `TickWindowStore` | simulation worker | opened after `start_supervisor` when `db_path` is set; dropped on shutdown | worker thread only; separate SQLite connection under WAL and `busy_timeout` |
| live `TickWindow` + `TickWindowView` | `App` (main thread) | created on supervisor start, cleared on stop | main thread only; fed by drained events |
| browse `TickWindow` | `App.browse_tick_window` | loaded once on entering `BrowsingSupervisor`; cleared on leaving | main thread; no writes |
| `replay_request` / `replay_jump_request` | `TimelineViewState`, `BoundaryViewState`, `ObservatoryViewState` inside `EvolutionSnapshot.observatory_view` | set by a click, moved by `render_observatory`, taken by the bin side the same frame | main thread only |

## Untrusted inputs

| input | bound |
|---|---|
| frame blobs read from a database | version must equal `FRAME_CODEC_VERSION`; agent count `<= MAX_FRAME_AGENTS`, food count `<= MAX_FRAME_FOOD`, exact remaining length; finite positions |
| frame and pin rows read from a database | at most `max_frames + max_pins * max_frames_per_pin` frames and `max_pins` pins loaded; unknown pin kind is an error |
| published physics and food slices | length a multiple of `PHYS_STRIDE` / `FOOD_STATE_STRIDE`, counts within the constants |

## Feasibility

| task | constraints and sources | compatibility witness | preflight check |
|---|---|---|---|
| tick-window-core | `TickRecord` fields and `vision_color: Option<Vec<f32>>` in `replay.rs`; `P_*`, `PHYS_STRIDE = 54`, `FOOD_STATE_STRIDE = 4` in `buffers.rs`; rusqlite and tempfile in `Cargo.toml`; module list in `lib.rs` | every field `record_from_physics_row` reads has a `P_*` slot below `PHYS_STRIDE`; new files absent | pf-tick-window-core |
| headless-tick-window | `run_supervisor` and `supervisor_step` copy `physics` and `food` per sample in `headless.rs`; `SampleReport.boundary/season` in `supervisor_run.rs` | capture reads those copies after `on_sample`; no `HeadlessOptions` literal changes; GPU adapter probed by `gpu-adapter-present` | pf-headless-tick-window |
| worker-tick-window | `publish_snapshot`, `start_supervisor`, `SupervisorWorkerInit.db_path`, `SupervisorSample` in `sim_runtime.rs` | the snapshot built for publication is the only capture input; GPU adapter probed by `gpu-adapter-present` | pf-worker-tick-window |
| tick-window-view | bin module list in `main.rs`; egui dependency | a new bin module needs only one `mod` line | pf-tick-window-view |
| gui-tick-window-wiring | `apply_state_snapshot` and the `SupervisorSample` arm in `gpu_orchestration.rs`; `App::new`, `App.db_path: String` and `App.evo_snapshot` in `app.rs`; 0008 ARCHITECTURE.md `Browse { run_id: i64 }` and `BrowsingSupervisor` | both feed sites exist on the main thread; the browse path and run id are reachable from `App` without new 0008 interfaces; `app.rs` and `gpu_orchestration.rs` contain no `GpuKernel` at baseline | pf-gui-tick-window-wiring |
| replay-jump-request | 0008 ARCHITECTURE.md binds `filtered(model,`, `boundary_lines(&BoundaryRecord)`, `show_boundaries(ui,`, `show_timeline`, `observatory_view: ObservatoryViewState` and `render_observatory(ui, model, view)`; egui 0.31 in `Cargo.toml` (headless `egui::Context::run` with `RawInput` pointer events available) | the row renderers and `render_observatory` are the 0008 render sites and receive mutable state, so a click can be recorded and moved | pf-replay-jump-request |

## Notes

- Source comments must not reference plans or tasks (`contributing_guard.rs`
  ratchet).
- generation-mode-removal deletes `replay.rs`; it must keep or move
  `TickRecord`, which this plan reuses because it carries no generation field.
