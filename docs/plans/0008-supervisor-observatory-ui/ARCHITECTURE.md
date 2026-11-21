# 0008 Supervisor observatory UI: architecture

## Design

```
worker (plan 0006 telemetry) --SimEvent--> gpu_orchestration.rs --fold--> ObservatoryModel
database (plan 0006 tables) --observatory::store_read (read-only, once)--> ObservatoryModel
ObservatoryModel --&--> ui.rs render_observatory --> tables | plots | history
startup: app.rs --store_read::detect_supervisor_run (read-only)--> SupervisorSession state
         --> session dialog (ui.rs) --> evo.session_choice --> evolution.rs
             ResumeLive (plan 0007 path) | Browse (read-only)
```

## Interfaces (binding)

All paths are under `crates/xagent-sandbox/src/`. `lib.rs` gains
`pub mod observatory;`.

- `observatory/mod.rs` (`observatory-model`) holds the model itself and
  declares `pub mod store_read; pub mod tables; pub mod plots; pub mod
  history;`. The model task creates those four files as one-line `//!` doc
  stubs; afterwards exactly one later task edits each (see State).
- Model types (all `Clone, Debug, PartialEq`; enums also `Copy, Eq` where
  field-free; `ObservatoryModel: Default`):
  - `slots: Vec<SlotRow>` (bounded by the run's slot count, never more than
    `MAX_SLOTS = 256`); `SlotRow { slot: usize, occupant_id: u64, age_ticks:
    u64, alive: bool, meals: u32, deaths: u32, window_score: Option<f32>,
    peer_median: Option<f32>, parent: Option<ParentRef> }`;
  - `boundaries: VecDeque<BoundaryRecord>`; `BoundaryRecord { tick: u64,
    eligible: Vec<EligibleSlot>, picked: Option<usize>, tie_break:
    Option<String>, parent: Option<ParentRef>, config_diff: Vec<ConfigDelta>,
    idle_reason: Option<String> }`; `EligibleSlot { slot: usize, ground:
    EligibilityGround }`; `enum EligibilityGround { Died, BelowMedian { score:
    f32, median: f32 } }`; `ParentRef { occupant_id: u64, source: ParentSource,
    state: InheritedState }`; `enum ParentSource { Archive { entry_id: u64 },
    Peer { slot: usize } }`; `enum InheritedState { Birth, Learned }`;
    `ConfigDelta { field: String, parent: String, child: String }`;
  - `series: Vec<SeriesPoint>`; `SeriesPoint { tick: u64, food_per_agent_tick:
    f32, deaths_per_agent_tick: f32, score_spread: Option<f32>,
    retired_lifespan: Option<u64>, archive_best: Option<f32>, archive_median:
    Option<f32> }`;
  - `occupants: BTreeMap<u64, OccupantRecord>`; `OccupantRecord { occupant_id:
    u64, parent: Option<ParentRef>, born_tick: u64, retired_tick: Option<u64>,
    intake: Vec<(u64, u32)>` (age ticks, cumulative meals) `, late:
    Option<LateWindow> }`; `LateWindow { meals_per_alive: f32,
    turn_toward_food: f32, sighting_meal_share: f32 }`;
  - `archive: Vec<ArchiveRow>`; `ArchiveRow { entry_id: u64, occupant_id: u64,
    outcome: f32, age_ticks: u64, config: xagent_shared::BrainConfig }`; when
    full, the lowest-outcome row is evicted;
  - `season: Option<SeasonState>`, `SeasonState { nourishing: FoodKind,
    next_flip_tick: Option<u64> }`; `progress: Progress { tick: u64,
    budget_ticks: Option<u64>, ticks_per_sec: Option<f64>, next_boundary_tick:
    Option<u64>, finished: bool }`;
  - `timeline: VecDeque<TimelineEvent>`; `TimelineEvent { tick: u64, kind:
    TimelineKind, text: String }`, `enum TimelineKind { Replacement, Idle,
    Season, Archive, Death, Run }`;
  - `source: ModelSource` (`Live` default, or `Browse { run_id: i64 }`).
  - `pub fn fold(&mut self, record: ObservatoryRecord)`, where `enum
    ObservatoryRecord` wraps one plan 0006 record (slot snapshot, boundary,
    series point, occupant intake, archive change, season, progress, event);
    the `From` conversions from the plan 0006 types live in `mod.rs`.
- Caps (named `pub const` in `mod.rs`): `MAX_SERIES_POINTS = 4096` (on overflow
  keep every second point, always keeping first and last), `MAX_BOUNDARY_RECORDS
  = 4096` and `MAX_TIMELINE_EVENTS = 20_000` (drop oldest), `MAX_TRACKED_OCCUPANTS
  = 4096` (evict the oldest retired occupant; living occupants never evicted),
  `MAX_INTAKE_POINTS = 512` per occupant (decimate like the series),
  `MAX_ARCHIVE_ROWS = 1024`, `MAX_DISPLAY_TEXT_BYTES = 256`.
- `observatory/store_read.rs` (`observatory-store-read`): `pub struct
  SupervisorRunSummary { pub run_id: i64, pub last_tick: u64, pub budget_ticks:
  Option<u64> }`; `pub fn open_read_only(db_path: &str) -> Result<Connection,
  BrowseError>` (`SQLITE_OPEN_READ_ONLY` with `SQLITE_OPEN_NO_MUTEX`, never creates a
  file, `PRAGMA query_only = ON`); `pub fn detect_supervisor_run(db_path: &str)
  -> Result<Option<SupervisorRunSummary>, BrowseError>` (`Ok(None)` for a missing
  file or no plan 0006 tables); `pub fn load_run(db_path: &str, run_id: i64) ->
  Result<ObservatoryModel, BrowseError>` reading only the plan 0006 snapshot,
  series and event tables, bounded by the caps via `LIMIT`, folded oldest first
  through `ObservatoryModel::fold`. Never calls `ensure_schema` or a writer.
- `observatory/tables.rs` (`tables-view`): slot table, boundary inspector, and
  the run-progress header with the season strip. Pure helpers `slot_rows`,
  `boundary_lines(&BoundaryRecord)`, `progress_text`, `season_text`; renderers
  `show_slots(ui, &ObservatoryModel, &mut SlotsViewState)`, `show_boundaries(ui,
  &ObservatoryModel, &mut BoundaryViewState)`, `show_header(ui,
  &ObservatoryModel)`.
- `observatory/plots.rs` (`plots-view`): painter-based `line_chart(ui, title,
  &[ChartSeries], &[AxisMarker])` like `ui.rs` `render_fitness_chart`;
  `axis_markers(&ObservatoryModel)` (one per replacement boundary, one per
  season flip, none for idle boundaries); the four population charts
  (`show_charts`), and the learning view (`intake_curves`, x = occupant age,
  `late_window_rows`, `show_learning`).
- `observatory/history.rs` (`history-view`): `lineage_forest` (each occupant
  once; a parent absent from `occupants` gives a root marked parent-evicted;
  iterative with a visited set so a malformed cycle terminates),
  `archive_rows` (outcome descending, NaN last), timeline `filtered(model,
  &TimelineFilter)` (kind set and inclusive tick range, newest first); renderers
  `show_lineage`, `show_archive`, `show_timeline` (`ScrollArea::show_rows`).
- `ui.rs` (`observatory-panel`): `EvolutionSnapshot` gains `observatory:
  Option<ObservatoryModel>`, `observatory_view: ObservatoryViewState` (per-view
  states, selected tab), `supervisor_session: Option<SupervisorRunSummary>`,
  `supervisor_resume_error: Option<String>`, `session_choice: SessionChoice`
  (`enum SessionChoice { None, ResumeLive, Browse }`, default `None`);
  `EvolutionState` gains `SupervisorSession` and `BrowsingSupervisor`.
  `render_observatory(ui, model, view)` shows the header on top and tabs Slots |
  Boundaries | Charts | Learning | Lineage | Archive | Timeline.
  `render_running_dashboard` calls it when `evo.supervisor` is `Some` instead of
  the six labels; `SupervisorSession` renders the "Supervisor run found" dialog
  (summary, Resume live, Browse read-only, the resume error if any, no Reset)
  and sets `evo.session_choice`; `BrowsingSupervisor` renders a read-only banner
  and `render_observatory` without Pause or Reset. No new `EvolutionAction`
  variant. `ui_chrome.rs` handles the two new states in its match.
- `app.rs` / `evolution.rs` / `gpu_orchestration.rs` (`session-wiring`):
  `App::new` checks for a nonexistent path first (no open), then calls
  `store_read::detect_supervisor_run`; on `Ok(Some(summary))` it sets
  `SupervisorSession` and skips `check_existing_session` and `Governor::resume`
  (both open read-write). `handle_evolution_action` (called every frame from
  `main.rs`) takes `evo_snapshot.session_choice`: Browse loads
  `store_read::load_run` into `observatory`, sets `BrowsingSupervisor`, starts no
  worker; ResumeLive loads the history the same way, then starts the worker
  through the plan 0007 start-from-checkpoint input, `source = Live`; a refusal
  is logged, stored in `supervisor_resume_error`, and the state returns to
  `SupervisorSession`. Supervisor Start sets `observatory =
  Some(ObservatoryModel::default())`. The event drain converts each plan 0006
  telemetry `SimEvent` into an `ObservatoryRecord` and folds it.

## Invariants

| id | invariant | guards | enforced at |
|---|---|---|---|
| INV-1 | Every `ObservatoryModel` collection stays within its named cap under any number of folded records; decimation keeps first and last tick; living occupants are never evicted. | memory and frame time on long runs | `observatory-model` tests `observatory::tests` |
| INV-2 | Folding the same plan 0006 records live and loading them through `store_read::load_run` yield equal models apart from `source` and `ticks_per_sec`. | live and browse show the same numbers | `observatory-store-read` test `tests/observatory_browse.rs` |
| INV-3 | Browsing never writes: detection and load leave the database file byte-identical, a missing path is not created, the browse connection refuses an `INSERT`, and startup for a supervisor or missing database calls no read-write open. | the user's run databases | `observatory-store-read` test `tests/observatory_browse.rs`; `session-wiring` bin tests `observatory_session_` |
| INV-4 | The redraw path makes no GPU dispatch, readback or database access: `tables.rs`, `plots.rs`, `history.rs` name none of `GpuKernel`, `wgpu`, `rusqlite`, `Connection`, `store_read::`, `read_full_state_blocking`; `ui.rs` names none of `store_read::load_run`, `store_read::detect_supervisor_run`, `store_read::open_read_only`, `GpuKernel`, `read_full_state_blocking`; `gpu_orchestration.rs` names no `store_read::`; the existing main-thread kernel needle test still passes. | kernel ownership on the worker; frame time | `observatory-panel` test `tests/observatory_purity.rs`; `session-wiring` tests `gui_supervisor_` and `observatory_session_` |
| INV-5 | The slot table shows per slot occupant, age, alive or dead, meals, deaths, window score against the peer median as a signed difference (or `n/a`), and the parent with source. | who occupies each slot and how it compares | `tables-view` tests `observatory::tables::tests` |
| INV-6 | The boundary inspector shows for a replacement every eligible slot with its ground and numbers, the picked slot and tie-break, the parent with source and state, and every config delta; for an idle boundary the idle reason. | why this slot and this parent | `tables-view` tests `observatory::tables::tests` |
| INV-7 | The charts draw food and deaths per agent-tick, window-score spread, retired-occupant lifespan and archive best and median over ticks, with a marker at every replacement and season flip tick and none elsewhere. | how the population changes over time | `plots-view` tests `observatory::plots::tests` |
| INV-8 | The learning view draws cumulative intake against occupant age for the selected occupants and lists each occupant's late-window measures. | within-life learning versus selection | `plots-view` tests `observatory::plots::tests` |
| INV-9 | The lineage view is a forest from parent references with each occupant once; an evicted parent yields a marked root; a cycle terminates; the archive browser lists outcome, age and config sorted by outcome descending. | who descends from whom; what the archive holds | `history-view` tests `observatory::history::tests` |
| INV-10 | The header shows tick, budget (or unlimited), ticks per second (live only) and ticks to next boundary; the season strip shows the nourishing kind and ticks to next flip; differences never underflow. | where the run is | `tables-view` tests `observatory::tables::tests` |
| INV-11 | The timeline lists events newest first, filterable by kind and tick range, with text truncated on a char boundary. | what happened when | `history-view` tests `observatory::history::tests` |
| INV-12 | A database holding a supervisor run opens to a choice of Resume live and Browse read-only; Browse starts no worker; Resume starts it through the plan 0007 checkpoint path; a refused resume returns to the dialog with the error. | session handling | `observatory-panel` tests `ui::tests::observatory_`; `session-wiring` bin tests `observatory_session_` |
| INV-13 | The six-label supervisor section is gone; the generation fitness and hazard charts, evolution tree and node detail still render in generation mode; `EvolutionSnapshot::default()` has no observatory, no supervisor session and `SessionChoice::None`. | scope boundary with generation-mode-removal | `observatory-panel` tests `ui::tests::observatory_`; integration test `evolution_snapshot_default_tree_pane_fraction` |

## State

| state | owner | lifecycle | concurrency |
|---|---|---|---|
| `ObservatoryModel` | main thread, `EvolutionSnapshot.observatory` | created on supervisor Start, Resume or Browse; dropped on Reset | main thread only; fed from drained `SimEvent`s, never shared with the worker |
| `ObservatoryViewState`, `SessionChoice` | main thread, `EvolutionSnapshot` | view state persists across frames; the choice is taken (reset to `None`) by the next `handle_evolution_action` | main thread only |
| browse `Connection` | `store_read` function scope | opened and closed inside one detect or load call, only from startup or an action handler, never per frame | read-only, `query_only`; may coexist with a writer under WAL |
| `SupervisorRunSummary` | `EvolutionSnapshot.supervisor_session` | set at startup detection, cleared on Resume, Browse or Reset | main thread only |
| file ownership | one editing task per file | `observatory-model` creates `mod.rs` and doc-only stubs `store_read.rs`, `tables.rs`, `plots.rs`, `history.rs`; afterwards only the task named in Interfaces edits each | every stub owner depends on `observatory-model`, so no two tasks edit one file concurrently |

## Untrusted inputs

| input | bounds |
|---|---|
| database at `--db` (legacy, foreign, truncated or partially written) | never created or written; missing plan 0006 tables give `Ok(None)`; malformed rows give `BrowseError` shown in the UI, never a panic; rows read limited by `LIMIT` to the caps; display strings truncated to `MAX_DISPLAY_TEXT_BYTES` on a char boundary |
| plan 0006 telemetry events from the worker | folded through the same capped `fold`; non-finite floats render as `n/a`; slot indexes at or beyond `MAX_SLOTS` are ignored |
| plan 0007 resume refusal | shown truncated to `MAX_DISPLAY_TEXT_BYTES`; never retried automatically |

## Feasibility

| task | constraints and sources | compatibility witness | preflight check |
|---|---|---|---|
| observatory-model | `lib.rs` declares the supervisor modules the model converts from; `ui.rs` `SupervisorSnapshot` shows the existing fold shape and `FoodKind` path; `serde` is a dependency; `observatory/` does not exist yet | grep of `pub mod supervisor_store;`, `pub mod supervisor_run;`, `pub struct SupervisorSnapshot {`, the `nourishing_kind` field, `serde = ` and absence of the directory | pf-observatory-model |
| observatory-store-read | `supervisor_store.rs` already opens read-only with `SQLITE_OPEN_READ_ONLY` with `SQLITE_OPEN_NO_MUTEX` and exposes `create` and `ensure_schema` for fixtures; `rusqlite` 0.34 bundled and `tempfile` 3 available | grep of those lines; test file absent | pf-observatory-store-read |
| tables-view | egui 0.31; `egui::Grid::new` and the CLAUDE.md egui notes; REVIEW-CHECKLIST present | grep of `egui = "0.31"`, `egui::Grid::new`, `is INFINITY inside` | pf-observatory-views |
| plots-view | painter chart pattern `allocate_exact_size(... Sense::hover())`, `painter_at(rect)` and `vline` exist in `ui.rs`, so charts with axis markers need no new dependency; `run_summary.rs` holds the late-window measures | grep of those exact lines | pf-observatory-views |
| history-view | egui grid, `ScrollArea::vertical()` and collapsing patterns in `ui.rs` | grep as above | pf-observatory-views |
| observatory-panel | `render_running_dashboard` with `if let Some(sup) = &evo.supervisor {` is the replacement site; `HasSession` exists in `ui.rs` and is matched in `ui_chrome.rs`; generation chart and tree functions exist; default-snapshot integration test exists; purity test absent | grep of those lines | pf-observatory-panel |
| session-wiring | `App::new` calls `check_existing_session(db_path)` (read-write open in `governor.rs`); `evolution.rs` holds the supervisor Start arm and generation Resume arm; `main.rs` calls `handle_evolution_action` every frame; `gpu_orchestration.rs` holds the `SupervisorSample` fold and the kernel needle test | grep of those lines | pf-session-wiring |
