# 0006 Architecture: supervisor decision telemetry

## Invariants

| id | invariant | guards | enforced at |
|---|---|---|---|
| INV-1 | `Inheritance::default()` and a `SupervisorConfig` JSON without `inheritance` give `WeightCopy`; `"darwinian"` and `--supervisor-inheritance darwinian` still select `Darwinian`; serde names stay `darwinian` / `weight_copy`. | the user's default flip | `crates/xagent-shared/src/config.rs` unit tests (inheritance-default) |
| INV-2 | `docs/measurements/continuous-supervisor/INHERITANCE-DEFAULT.md` exists and states: the user overrides the 0005 `REJECT` verdict, overrides the 0005 rule that weight-copy is never adopted, the supervisor becomes the only mode, and this is the user's decision, not a new measurement. `DECISION.md` still starts with the line `Verdict: REJECT`. | honest record; 0005 history intact | focused grep check (inheritance-default) |
| INV-3 | Observation consumes no random draw and mutates no decision state: a seeded core driven with and without every new query (`standings`, `last_boundary`, `next_boundary_tick`) yields identical `SampleDecision`s, outcomes, newborn brains and archive. | seeded decisions unchanged | `supervisor::tests` (boundary-detail) |
| INV-4 | Boundary detail carries the numbers the decision used: every `BelowMedian` candidate has `score < median` from the same `eligible_slots` pass; `Death` candidates are exactly the `pending_death` slots; the picked slot is `pool[drawn_index]`; an archive parent is `archive.entries()[drawn_index]`; an idle boundary's reason counts (`too_young + no_full_window + at_or_above_median + no_peer_median`) equal the slot count. | inspectability matches behaviour | `supervisor::tests` (boundary-detail) |
| INV-5 | A boundary record's config difference lists exactly the top-level `BrainConfig` fields whose `serde_json` values differ between parent and newborn config, in field-name order, each with both values. | config diff correctness | `supervisor_telemetry::tests` (telemetry-records) |
| INV-6 | Series values follow the formulas in "Series metrics": per-agent-tick rates over the cadence interval, window-score spread over finite full-window scores, retired lifespan at replacements, archive best and median outcome, and per-closed-season late-window measures equal to the difference of `RunSummaryAccumulator` closed-season totals; builder memory per slot is O(1). | metric correctness and bounds | `supervisor_telemetry::tests` (telemetry-records) |
| INV-7 | `ensure_schema` is additive and idempotent: run twice on a database holding only the 0005 tables and rows, it keeps every old row, creates each table in "Telemetry tables" once, and every new table has `run_id INTEGER NOT NULL REFERENCES run(id)`. Writes are batched in one transaction per sample. | old databases keep loading | `supervisor_store::tests` (telemetry-store) |
| INV-8 | Recording on and off give identical `SampleReport`s, birth brains, `supervisor_slot`, `supervisor_archive` and `supervisor_event` rows; a telemetry build or write failure never aborts or alters the run: it turns recording off and is reported once. | decisions independent of recording | `tests/supervisor_telemetry_run.rs` (run-recording) |
| INV-9 | The worker sends one `SimEvent::SupervisorTelemetry` per handled sample with `try_send` (dropped when the channel is full) and boundary records inside it are also sent as ordered control `SimEvent::SupervisorBoundary` with blocking `send`; no GPU dispatch or readback is added (it reuses the sample's `read_full_state_blocking` copy and `cached_food_state`). | bounded latest-wins boundary | `sim_runtime` unit tests (worker-telemetry) |
| INV-10 | The seeded headless supervisor digest over `supervisor_slot`, `supervisor_archive` and `supervisor_event` is equal across two runs and with recording on and off; the fixture records `inheritance: "weight_copy"` and the capture adapter, and the computed digest equals the fixture whenever the current adapter description equals the recorded one. | bit reproducibility after the flip | `tests/supervisor_digest.rs` (supervisor-digest-gate) |
| INV-11 | Seeded supervisor runs with the raw-vision and cortex brains hash (kernel physics + brain state + decision rows) to their lines in `tests/fixtures/supervisor_sense_parity.golden`. | sense-block layout under the supervisor | `tests/supervisor_sense_parity.rs` (supervisor-sense-parity) |
| INV-12 | A seeded supervisor run dispatched as one 100-tick `dispatch_batch` per sample and as `kernel_batch_size()` chunks through `dispatch_ticks` gives bit-identical physics at every sample and identical reports and birth brains; newborns keep both modulator flags when the base config turns them off. | fused/split GPU parity under the supervisor | `tests/supervisor_gpu_parity.rs` (supervisor-gpu-parity) |
| INV-13 | With the default world config a supervisor run has no nourishing kind and no season report; with food kinds on, seasons flip exactly at the `SEASON_STREAM` `TickSchedule` ticks, alternate kinds, the kernel's `nourishing_food_kind()` equals the report, and no flip happens elsewhere. | food-kind default and seasons | `tests/supervisor_food_kind.rs` (supervisor-food-kind-gate) |

## Interfaces (binding)

### Core additions (`supervisor.rs`, boundary-detail)

- `pub enum EligibilityGround { Death, BelowMedian { score: f32, median: f32 }, TooYoung, NoFullWindow, NoPeerMedian, AtOrAboveMedian { score: f32, median: f32 } }`
- `pub struct SlotStanding { pub slot: usize, pub occupant_id: u64, pub age_ticks: u64, pub alive: bool, pub meals: u32, pub deaths: u32, pub window_score: Option<f32>, pub peer_median: Option<f32>, pub ground: EligibilityGround, pub parent: Option<ParentRef> }` — meals/deaths are the current life's latest counters (0 before a sample).
- `pub struct TieBreak { pub pool: Vec<usize>, pub drawn_index: usize }`, `pub struct ParentDraw { pub pool_len: usize, pub drawn_index: usize }` (archive draw only; `None` for peer parents).
- `pub enum BoundaryDetail { Replaced { tick: u64, candidates: Vec<SlotStanding>, slot: usize, reason: ReplaceReason, tiebreak: TieBreak, parent: ParentRef, parent_draw: Option<ParentDraw>, retired_occupant: u64, retired_born_tick: u64 }, Idle { tick: u64, standings: Vec<SlotStanding>, reason: IdleReason } }` with `pub struct IdleReason { pub too_young: usize, pub no_full_window: usize, pub no_peer_median: usize, pub at_or_above_median: usize }`.
- `Supervisor::standings(&self, tick: u64) -> Vec<SlotStanding>`, `Supervisor::last_boundary(&self) -> Option<&BoundaryDetail>`, `Supervisor::next_boundary_tick(&self) -> u64`, `pub parent: Option<ParentRef>` on `SlotLedger` (`None` for initial occupants), `pub parent_config: BrainConfig` on `ReplacementOutcome`.
- `eligible_slots` is rewritten to produce the standings it decides from; `choose_candidate` and `plan_parent` record the index they drew. The number and order of draws on every stream is unchanged.

### Telemetry records (`supervisor_telemetry.rs`, telemetry-records)

- `pub const TELEMETRY_CADENCE_SAMPLES: u64 = 10;` (one snapshot, series and intake set every 1,000 ticks, plus every boundary and season tick).
- `pub struct ConfigFieldDiff { pub field: String, pub parent: serde_json::Value, pub newborn: serde_json::Value }`; `pub fn config_diff(parent: &BrainConfig, newborn: &BrainConfig) -> Vec<ConfigFieldDiff>`.
- `pub struct BoundaryRecord { pub detail: BoundaryDetail, pub newborn_occupant: Option<u64>, pub config_diff: Vec<ConfigFieldDiff>, pub retired_lifespan_ticks: Option<u64> }`.
- `pub enum Metric` with `as_str()` names exactly: `food_per_agent_tick`, `deaths_per_agent_tick`, `window_score_spread`, `retired_lifespan_ticks`, `archive_best_outcome`, `archive_median_outcome`, `late_toward_nourishing`, `late_away_harmful`, `late_nourishing_share`; `pub struct SeriesPoint { pub tick: u64, pub metric: Metric, pub value: f64 }`.
- `pub struct IntakeRow { pub occupant_id: u64, pub slot: usize, pub age_ticks: u64, pub meals: u32 }`.
- `pub struct ProgressRecord { pub tick: u64, pub total_ticks: Option<u64>, pub agent_tick_budget: u64, pub ticks_to_boundary: u64, pub nourishing: Option<FoodKind>, pub next_flip_tick: Option<u64>, pub ticks_per_second: Option<f64> }`.
- `pub struct TelemetryFrame { pub tick: u64, pub slots: Vec<SlotStanding>, pub boundary: Option<BoundaryRecord>, pub series: Vec<SeriesPoint>, pub intake: Vec<IntakeRow>, pub progress: ProgressRecord, pub persist: bool }` — `persist` is true on cadence, boundary and season ticks.
- `pub struct TelemetryBuilder` (`new(agent_count)`, `on_sample(...) -> TelemetryFrame`, `on_season_closed(...)`): pure, no GPU, SQLite or I/O.
- `run_summary.rs` gains `pub fn closed_totals(&self) -> Option<&KindWindowTotals>` (totals of closed seasons only; read-only).

### Series metrics

- `food_per_agent_tick`, `deaths_per_agent_tick`: increase of the summed per-slot meal / death counters since the previous cadence point (a slot whose occupant changed contributes its new life's counters) divided by `agent_count × ticks since the previous point`.
- `window_score_spread`: `max − min` of finite full-window scores; omitted with fewer than two.
- `retired_lifespan_ticks`: retired occupant's `boundary tick − born_tick`, at replacements only.
- `archive_best_outcome`, `archive_median_outcome`: rank-0 outcome and median outcome of the archive (median rule of `supervisor.rs`); omitted while empty.
- `late_toward_nourishing = Δlate_toward_nourishing_sum / Δlate_toward_nourishing_samples`, `late_away_harmful` likewise, `late_nourishing_share = Δlate_nourishing_meals / (Δlate_nourishing_meals + Δlate_harmful_meals)` over the season just closed; emitted at season flips with food kinds on; omitted when a denominator is 0.

### Telemetry tables (`supervisor_store.rs`, telemetry-store)

All `CREATE TABLE IF NOT EXISTS`, each with `run_id INTEGER NOT NULL REFERENCES run(id)`:

- `supervisor_slot_snapshot(run_id, tick, slot, occupant_id, age_ticks, alive, meals, deaths, window_score REAL NULL, peer_median REAL NULL, ground TEXT, parent_kind TEXT NULL, parent_ref INTEGER NULL, PRIMARY KEY (run_id, tick, slot))`
- `supervisor_boundary(run_id, tick, outcome TEXT /* replace|idle */, slot, reason, tiebreak_pool TEXT /* JSON array */, tiebreak_index, parent_kind, parent_ref, parent_state TEXT /* birth|learned */, parent_pool_len, parent_index, retired_occupant, retired_lifespan_ticks, newborn_occupant, config_diff TEXT /* JSON */, idle_too_young, idle_no_full_window, idle_no_peer_median, idle_at_or_above_median, PRIMARY KEY (run_id, tick))` (replace-only and idle-only columns NULL otherwise)
- `supervisor_boundary_candidate(run_id, tick, slot, occupant_id, ground TEXT /* death|below_median */, score REAL NULL, median REAL NULL, age_ticks, PRIMARY KEY (run_id, tick, slot))`
- `supervisor_series(run_id, tick, metric TEXT, value REAL, PRIMARY KEY (run_id, tick, metric))`
- `supervisor_occupant_intake(run_id, occupant_id, age_ticks, slot, meals, PRIMARY KEY (run_id, occupant_id, age_ticks))`
- `supervisor_progress(run_id, tick, total_ticks NULL, agent_tick_budget, ticks_to_boundary, nourishing_kind TEXT NULL, next_flip_tick NULL, ticks_per_second REAL NULL, PRIMARY KEY (run_id, tick))`

Writer: `SupervisorStore::record_telemetry(&mut self, frame: &TelemetryFrame) -> Result<(), StoreError>` (one transaction; no-op unless `frame.persist`). The 0005 tables and their writers are unchanged.

### Driver (`supervisor_run.rs`, `headless.rs`, run-recording)

- `SupervisorRun::set_recording(&mut self, on: bool)` (default: on when a store is present), `SupervisorRun::note_ticks_per_second(&mut self, rate: f64)`, `SupervisorRun::take_frame(&mut self) -> Option<TelemetryFrame>` (frame of the last `on_sample`; built whether or not recording is on).
- `SampleReport`, `BoundaryReport` and `SupervisorRun::start`'s signature are unchanged.
- `headless::run_supervisor_with_recording(config: FullConfig, db_path: &str, options: &HeadlessOptions, record: bool) -> Result<(), HeadlessError>`; the existing supervisor path calls it with `true` and feeds wall-clock ticks per second.

### Worker (`sim_runtime.rs`, worker-telemetry)

- `SimEvent::SupervisorTelemetry(Box<TelemetryFrame>)` (try_send, latest-wins) and `SimEvent::SupervisorBoundary(Box<BoundaryRecord>)` (control, blocking send). `gpu_orchestration.rs` gains match arms that ignore both (the observatory UI is 0008). `partition_events` treats them as control events and is otherwise unchanged.

## State

| state | owner | lifecycle | concurrency |
|---|---|---|---|
| `Supervisor` standings and `last_boundary` | `Supervisor` (pure core) | overwritten at each boundary; read-only queries | single-threaded, owned by `SupervisorRun` |
| `TelemetryBuilder` (previous cadence counters per slot, previous closed-season totals, telemetry `RunSummaryAccumulator`) | `SupervisorRun` | created at `start`, dropped at `finish` | same thread as the run (headless thread or sim worker) |
| telemetry rows | `SupervisorStore` (SQLite, WAL) | written per persisted frame in one transaction; never updated | worker thread is the only writer; readers open read-only |
| `TelemetryFrame` in flight | `SimEvent` channel | moved worker → main; dropped under back-pressure | bounded `sync_channel(EVENT_CAPACITY)` |

## Untrusted inputs

| input | bound |
|---|---|
| physics rows and food state from the GPU | lengths checked as today; non-finite counters give `None` scores and omitted series points, never a panic |
| existing database file | `ensure_schema` only adds tables; `SupervisorStore::create` still refuses a database that holds a run |
| config JSON `inheritance` | serde enum; an unknown value is a parse error, a missing value is `WeightCopy` |
| fixture and golden files | parsed with explicit field checks; missing/malformed fixture fails the test |

## Feasibility

| task | constraints and sources | compatibility witness | preflight check |
|---|---|---|---|
| inheritance-default | `#[default]` on `Inheritance::Darwinian` and `default_inheritance()` in `crates/xagent-shared/src/config.rs`; `DECISION.md` first line; supervisor tests that rely on the default (`sim_runtime` supervisor tests) work under either policy because `SupervisorRun::replace` already reads learned state | config.rs anchors and DECISION.md line | pf-inheritance-default |
| boundary-detail | `eligible_slots`, `choose_candidate`, `plan_parent`, `median`, `ReplacementOutcome`, `SlotLedger` in `supervisor.rs`; all constructions of these structs are inside `supervisor.rs` | supervisor.rs anchors | pf-boundary-detail |
| telemetry-records | `KindWindowTotals`, `WINDOW_FRACTION_NUMERATOR`, `begin_season` in `run_summary.rs`; `lib.rs` module list; `ArchiveEntry` | run_summary.rs/lib.rs anchors and absent module | pf-telemetry-records |
| telemetry-store | `ensure_schema`, `STORE_PRAGMAS`, `record_ticks` in `supervisor_store.rs`; `tempfile` dev-dependency | store anchors | pf-telemetry-store |
| run-recording | `on_sample`, `finish` in `supervisor_run.rs`; `run_supervisor`, `supervisor_step`, `run_headless_with_options` in `headless.rs`; `HeadlessOptions` literals in tests stay valid because no field is added | driver anchors, GPU adapter | pf-run-recording |
| worker-telemetry | `SimEvent::SupervisorSample`, `on_supervisor_sample`, `EVENT_CAPACITY`, `publish_snapshot` in `sim_runtime.rs`; exhaustive `SimEvent` match in `gpu_orchestration.rs`; `cached_food_state` | worker anchors, GPU adapter | pf-worker-telemetry |
| supervisor-digest-gate | `Fnv1a` and `hash_section` pattern in `seeded_headless_digest.rs`; supervisor tables; headless entry; capture needs a GPU run (architect shell) | digest anchors, GPU adapter, test build action | pf-supervisor-digest-gate |
| supervisor-sense-parity | `RECORD_VARIABLE`, `check_golden`, `kernel_state_bytes` pattern in `sense_block_parity.rs`; golden capture needs a GPU run (architect shell) | parity anchors, GPU adapter, test build action | pf-supervisor-sense-parity |
| supervisor-gpu-parity | `dispatch_batch` and `dispatch_ticks` in `gpu_kernel.rs`; modulator-flag inheritance test pattern; `on_sample` | kernel anchors, GPU adapter | pf-supervisor-gpu-parity |
| supervisor-food-kind-gate | `food_kind_draw_off_by_default` pattern; `flip_season`; `SEASON_STREAM` | food-kind anchors, GPU adapter | pf-supervisor-food-kind |
