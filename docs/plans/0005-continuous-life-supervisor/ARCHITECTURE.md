# Architecture: continuous life supervisor

## Design

```
FullConfig.governor.evolution_mode ──► run_headless_with_options (headless.rs) / App Start (evolution.rs)
   Generations (default) ─► existing generation loop, unchanged; optional RunSummary (measurement only)
   Supervisor ───────────► SupervisorRun (supervisor_run.rs, kernel-facing, shared by headless + sim worker)
                              │ owns: Supervisor core (supervisor.rs, pure), WorldState (spawns),
                              │       Option<SupervisorStore> (supervisor_store.rs), Option<RunSummaryAccumulator>
                              │ borrows: &mut GpuKernel (one kernel, population_size agents, one world)
                              ▼
  every SUPERVISOR_SAMPLE_TICKS (100) ticks: physics row per slot ─► ledger sample
     season tick?   ─► GpuKernel::set_nourishing_food_kind(other kind)      (food kinds on only)
     boundary tick? ─► eligibility ─► seeded tie-break ─► one slot
                       ─► retire: archive (own life outcome) ─► parent draw ─► mutate
                       ─► GpuKernel::replace_agent_slot (write_agent_state + write_agent_heritable_config
                          + write_agent_physics_fields with the slot's spawn row)
GUI: sim_runtime.rs worker owns SupervisorRun + kernel; clamps dispatches to the 100-tick grid,
     blocking readback on the worker only; SimEvent::SupervisorSample(report) ─► main thread
     (gpu_orchestration.rs) ─► EvolutionSnapshot.supervisor: Option<SupervisorSnapshot> (ui.rs)
Evidence: RunSummary JSON (--run-summary), FreeLifeReport (0001, BrainSource::Archive),
          bin supervisor_verdict DIR|BUNDLE [--write-bundle FILE] [--expect-verdict TEXT] (no GPU)
```

## Interfaces (binding)

### Config (`crates/xagent-shared/src/config.rs`, re-exported from `lib.rs`)

- `pub enum EvolutionMode { Generations, Supervisor }`, `#[derive(Clone, Copy,
  Debug, Default, PartialEq, Eq, Serialize, Deserialize)]`, `#[serde(rename_all
  = "snake_case")]`, `Generations` is `#[default]`.
- `pub enum Inheritance { Darwinian, WeightCopy }`, same derives and serde
  casing (`"darwinian"`, `"weight_copy"`), `Darwinian` is `#[default]`.
- `pub const SUPERVISOR_SAMPLE_TICKS: u64 = 100;` (the headless readback
  interval; every supervisor tick value is a multiple of it).
- `pub struct SupervisorConfig` (`Clone, Debug, PartialEq, Serialize,
  Deserialize`, every field `#[serde(default = ...)]`, `impl Default` equal to
  the serde defaults):

  | field | type | default | reason |
  |---|---|---|---|
  | `boundary_interval_ticks` | `u64` | 4,000 | one replacement per 4,000 ticks matches the generation reference's turnover (10 agents per 40,000-tick generation) |
  | `min_age_ticks` | `u64` | 20,000 | half a 40,000-tick generation before an agent can be judged an underperformer |
  | `score_window_ticks` | `u64` | 10,000 | a quarter generation, the within-life tracker's window |
  | `archive_capacity` | `usize` | 32 | bounded parent pool, several generations of champions |
  | `season_period_ticks` | `u64` | 40,000 | one food-kind season per generation length |
  | `inheritance` | `Inheritance` | `Darwinian` | the weight copy is a comparison arm only |
  | `agent_tick_budget` | `u64` | 0 | 0 = unlimited |

- `SupervisorConfig::validate(&self, agent_count: usize) -> Result<(),
  SupervisorConfigError>` (`Debug + Display + Error`, one variant per rule):
  `agent_count >= 2`; `boundary_interval_ticks` and `season_period_ticks` are
  multiples of `SUPERVISOR_SAMPLE_TICKS` and at least `2 *
  SUPERVISOR_SAMPLE_TICKS`; `score_window_ticks` is a nonzero multiple of it;
  `min_age_ticks >= score_window_ticks`; `1 <= archive_capacity <=
  MAX_ARCHIVE_CAPACITY (1024)`; `agent_tick_budget` is a multiple of
  `agent_count * SUPERVISOR_SAMPLE_TICKS` (checked multiplication).
- `GovernorConfig` gains `#[serde(default)] pub evolution_mode: EvolutionMode`
  and `#[serde(default)] pub supervisor: SupervisorConfig`; `impl Default`
  fills both with their defaults. `BrainConfig` is unchanged.

### Governor hooks (`crates/xagent-sandbox/src/governor.rs`)

- `init_schema` runs `ALTER TABLE run ADD COLUMN mode TEXT DEFAULT
  'generations';` with the existing `let _ = db.execute_batch(...)` pattern;
  `Governor::resume` runs the same migration and, when the latest run's
  `COALESCE(mode, 'generations')` is `'supervisor'`, logs a `log::warn!`
  naming the mode and returns `Err(rusqlite::Error::QueryReturnedNoRows)` (no
  generation run to resume). Generation runs keep inserting without a mode (the default
  fills it), so every generation-mode byte the digest hashes is unchanged.
- `pub(crate)` visibility for `composite_fitness`, `encode_brain_state`,
  `decode_brain_state` and `init_schema`; new `pub(crate) fn
  exploration_grid_cells() -> f32` returning `(HEATMAP_RES * HEATMAP_RES / 4)
  as f32`, used by `evaluate` in place of its inline expression.
- Existing `GovernorConfig { .. }` literals in `governor.rs` tests and
  `tests/integration.rs` end with `..GovernorConfig::default()`.

### Run summary (`crates/xagent-sandbox/src/run_summary.rs`, pure)

- `RunSummary { mode: EvolutionMode, inheritance: Option<Inheritance>, seed:
  u64, agent_count: usize, ticks: u64, agent_ticks: u64, food_kinds_enabled:
  bool, nourishing_meals: u64, harmful_meals: u64, deaths: u64, seasons: u64,
  kinds: Option<KindWindowTotals> }` and `KindWindowTotals {
  late_toward_nourishing_sum: f64, late_toward_nourishing_samples: u64,
  late_away_harmful_sum: f64, late_away_harmful_samples: u64,
  late_nourishing_meals: u64, late_harmful_meals: u64, early_nourishing_meals:
  u64, early_harmful_meals: u64 }`, both `Clone, Debug, PartialEq, Serialize,
  Deserialize`. `inheritance` is `None` in generation mode.
- `RunSummaryAccumulator::new(mode, inheritance, seed, agent_count,
  food_kinds_enabled)`; `begin_season(&mut self, start_tick: u64, nourishing:
  Option<FoodKind>)` (closes the open season); `reset_agent_baseline(agent)`
  (slot replaced: next counters are deltas from 0); `reset_all_baselines()`
  (generation start); `record_physics_sample(&mut self, tick: u64, physics:
  &[f32], food: Option<&[f32]>)` (meal and death deltas per agent from
  `P_FOOD_COUNT`, `P_HARMFUL_MEAL_COUNT` (0 when food kinds are off) and
  `P_DEATH_COUNT`; with kinds on and `food` present, one turn sample per alive
  agent: `P_MOTOR_TURN_OUT` against `nearest_food_of_kind` of the nourishing
  and harmful kinds within `SCENT_RANGE`, reading only the public
  `NearestFood.bearing` and signing it with a crate-local private helper
  `approach_sign_of(bearing) = -signum(bearing)` (`0.0` maps to `0.0`), which
  re-states 0004's convention because 0004 exports no sign helper, position
  `P_POS_*`, facing `[P_FACING_X, P_FACING_Z]`); `finish(self, ticks: u64) ->
  RunSummary` (`agent_ticks = ticks * agent_count`, checked).
- Windows are counted in sample intervals inside each season: with `n`
  intervals in a season (its actual length, the last season truncated at the
  run end), the early window is the first `n * 3 / 10` intervals and the late
  window the last `n * 3 / 10` (integer division). An interval's meals and its
  end-of-interval turn sample belong to the window of that interval. A season
  record is classified when it closes.

### Supervisor core (`crates/xagent-sandbox/src/supervisor.rs`, pure)

- Streams: `stream_seed(run_seed: u64, tag: u64) -> u64` is splitmix64 of
  `run_seed ^ tag`; named tags `BOUNDARY_STREAM`, `SEASON_STREAM`,
  `TIEBREAK_STREAM`, `PARENT_STREAM` (parent draw and mutation),
  `INITIAL_STREAM` (initial brains); each a `SmallRng::seed_from_u64`. The run
  seed is `config.world.seed`.
- `TickSchedule::new(seed, mean_ticks)`: gaps of `SUPERVISOR_SAMPLE_TICKS * k`
  with `k` uniform in `[ceil(m / 2), floor(3 * m / 2)]`, `m = mean_ticks /
  SUPERVISOR_SAMPLE_TICKS`; `next_tick()`, `advance()`. Boundaries and seasons
  each own one schedule; neither reads the other's stream.
- Ledger per slot: occupant id (monotone from 0), `born_tick`, birth brain and
  `BrainConfig`, pending-death flag, life cell set, and a ring of the last
  `score_window_ticks / SUPERVISOR_SAMPLE_TICKS + 1` samples of `SlotCounters {
  alive, deaths, food, distance, energy_spent, cell: Option<u16> }`
  (`SlotCounters::from_physics(row)`; the cell uses `Agent::record_heatmap`'s
  mapping and is `None` while dead).
- Window score at boundary `t` (slot age `>= score_window_ticks`):
  `composite_fitness(Δdeaths, Δfood, distinct cells in (t − W, t],
  exploration_grid_cells(), Δdistance, Δenergy_spent)` over the window `W`.
  Life outcome: the same function over the slot's whole life (counters reset at
  birth, life cell set).
- Eligibility at a boundary: a slot with a pending death (any death since it
  was born) is death-eligible; otherwise a slot with `age >= min_age_ticks`
  whose window score is strictly below the median of its peers' window scores
  (peers = the other slots with `age >= score_window_ticks`; even count = mean
  of the middle two; no peer = not eligible) is underperformer-eligible. No
  stored or archived score is consulted.
- Tie-break: candidates sorted by slot index; death-eligible first, else
  underperformers; one `random_range(0..len)` draw from the tie-break stream
  when at least one candidate exists. At most one slot per boundary.
- Archive: entries `{ entry_id (monotone from 1), occupant_id, born_tick,
  retired_tick, outcome, config, birth_brain }`; rank order `outcome` descending
  (`f32::total_cmp`), then `entry_id` ascending; insert when below capacity, or
  when full and `outcome` beats the worst (which is evicted); non-finite
  outcomes are never archived. Retired slots are archived at retirement; at the
  run end, living slots with `age >= min_age_ticks` are archived in slot order.
  `ArchiveEntry` and `SlotLedger` fields are `pub` (read by the store and the
  driver); `Archive::entries()` yields entries in rank order and
  `Archive::get(entry_id)` looks one up.
- Parent (Darwinian): uniform draw over the archive in rank order from the
  parent stream; when the archive is empty, the living peer (not the replaced
  slot) with the highest window score among peers with full windows, else the
  highest life outcome, ties to the lowest slot index; its ledger birth brain.
  Parent (`WeightCopy`): that best living peer, and the driver supplies its
  current learned state. Newborn: `mutate_steering_weights(parent_brain,
  mutation_strength, rng)` and `mutate_sensory_genes(parent_config,
  mutation_strength, &MutationMomentum::new(momentum_decay), rng)` with the
  parent stream and `GovernorConfig.mutation_strength` / `momentum_decay`.
- Initial population: `template = fresh_brain_state(&config.brain, rng)`,
  brains `steering_population(&template, n, 1, mutation_strength, rng)` from
  the initial stream, configs `config.brain`, `born_tick` 0.

### Store (`crates/xagent-sandbox/src/supervisor_store.rs`)

- `ensure_schema(db)`: `governor::init_schema` plus, idempotently,
  `supervisor_slot (run_id, slot, occupant_id, born_tick, config_json,
  birth_brain BLOB, PRIMARY KEY (run_id, slot))`, `supervisor_archive (id
  INTEGER PRIMARY KEY, run_id, entry_id, occupant_id, born_tick, retired_tick,
  outcome REAL, config_json, birth_brain BLOB, UNIQUE (run_id, entry_id))`
  (rows of evicted entries are deleted, so the table mirrors the bounded
  archive) and `supervisor_event (id INTEGER PRIMARY KEY, run_id, tick, kind,
  slot, reason, occupant_id, parent_kind, parent_ref, outcome,
  nourishing_kind)` with `kind` in `replace`, `idle`, `season`, `archive`,
  `evict`.
- `SupervisorStore::create(db_path, config: &FullConfig) -> Result<Self,
  StoreError>`: WAL, `busy_timeout`, `ensure_schema`; refuses a database whose
  `run` table already has a row; inserts the run row with `mode =
  'supervisor'`. Blobs are `encode_brain_state` output; each boundary's writes
  are one transaction.
- `load_archive_entry_read_only(db_path, rank: usize) -> Result<StoredArchiveEntry,
  ArchiveLoadError>`: `SQLITE_OPEN_READ_ONLY`, `busy_timeout`, no schema or
  write statement; the latest `mode = 'supervisor'` run; entries in rank order
  (`outcome DESC, entry_id ASC`), `OFFSET rank`; blob through
  `decode_brain_state`, config parsed strictly.

### Kernel (`crates/xagent-brain/src/gpu_kernel.rs`)

- `pub fn set_nourishing_food_kind(&mut self, kind: FoodKind) -> Result<(),
  FoodKindSetError>`: `Err` when food kinds are off; otherwise sets the current
  kind, which `write_world_config_scratch` packs into `WC_FOOD_KINDS` on the
  next world-config upload. It writes no agent buffer and does not touch the
  draw counter.
- `pub struct SlotSpawn { position: glam::Vec3, max_energy, max_integrity: f32,
  memory_capacity, processing_slots: usize }`; `pub fn
  slot_spawn_physics_row(spawn: &SlotSpawn) -> Vec<(usize, f32)>` returns all
  `PHYS_STRIDE` `(slot, value)` pairs of the row `upload_agents` builds, which
  `upload_agents` now uses (identical bytes); `pub fn replace_agent_slot(&self,
  index: u32, birth: &AgentBrainState, config: &BrainConfig, spawn:
  &SlotSpawn) -> Result<(), SlotReplaceError>` validates the index and the
  state lengths, then calls only `write_agent_state`,
  `write_agent_heritable_config` and `write_agent_physics_fields`.

### Shared driver (`crates/xagent-sandbox/src/supervisor_run.rs`)

- `SupervisorRun::start(kernel: &mut GpuKernel, config: &FullConfig, store:
  Option<SupervisorStore>, measure: bool) -> Result<Self, SupervisorRunError>`:
  validates the config, `kernel.agent_count() == population_size` and
  `SUPERVISOR_SAMPLE_TICKS % kernel_batch_size() == 0`; builds its own
  `WorldState::new(config.world)`, uploads world and agents (spawn rows from
  `Agent::new` at `safe_spawn_position`), `reset_agents_seeded`, writes the
  initial brains and heritable configs, records the slots.
- `tick()`, `next_sample_tick()`, `total_ticks() -> Option<u64>`,
  `is_finished()`, `birth_state(slot)`.
- `on_sample(&mut self, kernel, tick, physics: &[f32], food: Option<&[f32]>)
  -> Result<SampleReport, SupervisorRunError>`: `tick` must equal
  `next_sample_tick()`; order: ledger sample and measurement, season flip,
  boundary replacement, store writes. `SampleReport { tick, boundary:
  Option<BoundaryReport>, season: Option<FoodKind>, finished }` and
  `BoundaryReport { tick, slot: Option<usize>, reason: Option<ReplaceReason>,
  occupant_id: Option<u64>, parent: Option<ParentRef>, archive_len,
  best_outcome: Option<f32> }` are `Clone, Debug, PartialEq`.
- `finish(self) -> Result<SupervisorFinish { summary: Option<RunSummary> },
  SupervisorRunError>` archives eligible living slots and records the ticks.

### Drivers, CLI, GUI

- `headless.rs`: `pub struct HeadlessOptions { pub run_summary:
  Option<PathBuf> }` (`Default`); `pub fn run_headless_with_options(config,
  db_path, resume, &HeadlessOptions) -> Result<(), HeadlessError>`;
  `run_headless` delegates with default options and keeps its signature.
  Supervisor mode loops `dispatch_batch(tick, 100)` → `try_collect_state_snapshot`
  → `on_sample(cached_state, cached_food_state)`. `pub fn
  max_generations_for_agent_ticks(budget, population, tick_budget) ->
  Result<u64, HeadlessError>` returns the `max_generations` that makes the
  generation loop run exactly `budget / (population * tick_budget)`
  generations. `BrainSource::Archive { db_path: String, rank: usize }` and
  `FreeLifeReport.archive_entry_id: Option<i64>` (`#[serde(default)]`),
  `brain_source` `"archive"`.
- `main.rs`: `--evolution-mode <generations|supervisor>`,
  `--agent-tick-budget N`, `--supervisor-inheritance <darwinian|weight-copy>`,
  `--run-summary PATH`, `--free-life-archive-db PATH`,
  `--free-life-archive-rank N` (requires the db; conflicts with
  `--free-life-champion-db`). A pure `apply_evolution_cli(cli, &mut config) ->
  Result<(), String>`; errors exit with status 1 and the message on stderr.
- `sim_runtime.rs`: `SimInit.supervisor: Option<SupervisorWorkerInit { config:
  FullConfig, db_path: Option<String> }>`; `SimEvent::SupervisorSample(SampleReport)`
  (a control event, always delivered); `clamp_ticks_to_supervisor_sample`.
- `ui.rs`: `SupervisorSnapshot` and `EvolutionSnapshot.supervisor:
  Option<SupervisorSnapshot>` (default `None`).

### Verdict (`crates/xagent-sandbox/src/bin/supervisor_verdict.rs`)

`supervisor_verdict INPUT [--write-bundle FILE] [--expect-verdict TEXT]`. INPUT
is a directory holding exactly the 32 files `{generations,supervisor,weight-copy}-seed{5..8}.json`,
`{generations,supervisor}-kinds-seed{5..8}.json` (`RunSummary`) and
`free-life-{generations,supervisor,weight-copy}-seed{5..8}.json`
(`FreeLifeReport`), or a bundle file `{ "format":
"xagent-supervisor-verdict-bundle", "version": 1, "runs", "free_lives",
"metrics", "verdict" }` whose stored metrics and verdict must equal the
recomputed ones. It prints `Verdict: ADOPT|OPTION|REJECT` and a metrics table
(including the weight-copy arm), and applies the PLAN.md rule.

## Invariants

| id | invariant | guards | enforced at |
|---|---|---|---|
| INV-01 | Default-off bit identity: `evolution_mode` is `Generations` and every `SupervisorConfig` field takes its default when omitted from JSON; with the flag off, the generation path's DB rows the digest hashes, the 0001 strict seeded digest, the 0004 parity golden and the 0004 `food_kind_draw` default check are unchanged; no WGSL file changes. | Generation mode, stored champions and the evolution reference stay untouched | `config.rs`, `governor.rs`, `headless.rs`, `gpu_kernel.rs` (`upload_agents` row), checks `seeded-digest-strict`, `sense-parity`, `food-kind-default` |
| INV-02 | Supervisor settings live only in `GovernorConfig.supervisor`/`evolution_mode`: `BrainConfig` JSON has no supervisor key, nothing mutates or inherits them, and `SupervisorConfig::validate` rejects every infeasible value listed in Interfaces before any dispatch. | No hidden heritable parameter; no run on an infeasible schedule | `config.rs`, `supervisor_run.rs` (`start`), `main.rs` |
| INV-03 | Seeded schedules: boundary and season ticks come from separate streams derived from the run seed, lie on the 100-tick sample grid, and the same seed gives the same ticks; changing one stream's tag never moves the other's ticks. | Reproducible runs; seasons cannot perturb selection timing | `supervisor.rs` |
| INV-04 | Eligibility and tie-break: a slot is eligible only for a pending death or, past `min_age_ticks`, for a window score strictly below its peers' median over the same window; no stored or archived score is read; death-eligible slots go first; exactly one slot is replaced at a boundary with any candidate (none otherwise), chosen by the tie-break stream. | Fair, local selection; one refill at a time | `supervisor.rs` |
| INV-05 | Darwinian newborns: with `Darwinian`, a newborn's brain is `mutate_steering_weights` of a parent's birth brain (archive draw, or the best living peer's birth brain while the archive is empty) and its config is `mutate_sensory_genes` of the parent's config; no learned weight reaches a newborn. Only `WeightCopy`, never the default, copies a living agent's learned state. | Inheritance passes the genome, not what a life learned | `supervisor.rs`, `supervisor_run.rs` |
| INV-06 | Bounded archive ranked by own life outcome: at most `archive_capacity` entries, ordered by each entry's own whole-life `composite_fitness` then entry id; a full archive admits only an entry that beats the worst, which is evicted (in memory and in `supervisor_archive`); non-finite outcomes are never archived. | Parent pool is the best birth brains, bounded | `supervisor.rs`, `supervisor_store.rs` |
| INV-07 | Selection-only score: the window score and life outcome are computed on the CPU from physics readbacks and used only to choose slots and rank the archive; `supervisor.rs` has no GPU type; a boundary writes only the replaced slot, so every other slot's brain state and physics row are identical just before and just after it. | The only learning signal stays the agent's own homeostatic change | `supervisor.rs`, `supervisor_run.rs` |
| INV-08 | Hardware budget: a supervisor run uses one `GpuKernel` with `population_size` agents and one world, and dispatches exactly `ticks / kernel_batch_size()` kernel batches for `ticks` simulated ticks. | Hardware is the binding limit | `supervisor_run.rs`, `headless.rs`, `sim_runtime.rs` |
| INV-09 | Slot replacement: `replace_agent_slot` uses only the three existing single-slot writes; afterwards the slot's physics row equals `slot_spawn_physics_row(spawn)` (all counters zero, alive, full energy and integrity) and its brain equals the written birth brain; invalid index or state length is an error with no write. | Newborns start clean without a population reset | `gpu_kernel.rs` |
| INV-10 | Kind setter: `set_nourishing_food_kind` changes the kind the kernel uses from the next upload, resets no agent, leaves the 0004 draw counter where it was (the next reset draws the next index), and is an error with food kinds off. In supervisor mode the kind flips only at season ticks. | Seasons without generations | `gpu_kernel.rs`, `supervisor_run.rs` |
| INV-11 | Persistence compatibility: supervisor tables are created idempotently; `run.mode` is added by idempotent `ALTER TABLE` with default `'generations'`; a database with the pre-plan schema still resumes in generation mode and reports mode `generations`; resuming a supervisor run is refused; `SupervisorStore::create` refuses a non-fresh database; the read-only archive loader leaves the main database file's bytes unchanged. | Old databases keep loading; runs never mix | `governor.rs`, `supervisor_store.rs` |
| INV-12 | Driver selection: `run_headless_with_options` and the GUI Start pick the driver from `evolution_mode`; `run_headless` keeps its signature and generation behaviour; a supervisor run with the same seed and config produces identical `supervisor_event` and `supervisor_archive` rows twice; the generation run summary equals the sums of the run's `agent_result` rows; `EvolutionSnapshot::default().supervisor` is `None`. | Both modes stay usable; reproducible evidence | `headless.rs`, `main.rs`, `evolution.rs`, `ui.rs` |
| INV-13 | GUI reproducibility and thread boundary: the simulation worker clamps dispatches to the sample grid and performs every supervisor readback; the main thread never touches `GpuKernel` or `SupervisorRun`; the worker's `SampleReport` sequence for a seed equals the headless driver's for the same seed at any speed multiplier. | GUI runs are as reproducible as headless ones; the redraw path never blocks on readback | `sim_runtime.rs`, `gpu_orchestration.rs` |
| INV-14 | Verdict by code: the verdict comes only from the 32 validated inputs (seeds 5-8, 10 agents, 32,000,000 agent-ticks per in-run summary, 300,000-tick free lives, matching modes, inheritance, food-kind flags and brain sources) with the PLAN.md rule; any missing, duplicate, malformed, mismatched or undefined input is an error; `--expect-verdict` exits non-zero on mismatch; `DECISION.md`'s first line matches the bundle. | The record cannot drift from the evidence | `supervisor_verdict.rs`, `DECISION.md`, `results.json` |
| INV-16 | Measurement only and season-relative: `RunSummaryAccumulator` reads copies of physics and food state and writes no kernel, brain, physics, food or config state; meal and death totals equal the sum of per-agent counter increments with every baseline reset honoured; early and late windows are the first and last 30% of each season's intervals, judged against the kind that is nourishing in that season. | The evidence measures without changing the run; the food-kind gate follows the current season | `run_summary.rs`, `headless.rs`, `supervisor_run.rs` |
| INV-15 | GPU proofs fail closed: every new GPU test panics instead of skipping when `XAGENT_REQUIRE_GPU` is set and no adapter exists, and every GPU-backed check sets it. | Proofs that pass without running | each new GPU test; task and plan checks |

## State

| state | owner | lifecycle | concurrency |
|---|---|---|---|
| `EvolutionMode`, `SupervisorConfig` | `FullConfig.governor` (JSON, CLI) | Fixed before a run; stored in `run.governor_config` | Read-only after start |
| Supervisor core (schedules, streams, ledger, archive) | `SupervisorRun` | Created by `start`, advanced only by `on_sample`, consumed by `finish` | Owned by the thread that owns the kernel (headless loop or sim worker) |
| `WorldState` for newborn spawns | `SupervisorRun` | Built from `config.world` at `start` | Same owning thread |
| Kernel buffers of a replaced slot | `GpuKernel::replace_agent_slot` | Written between dispatches at a boundary | Host writes only between dispatches |
| Current nourishing kind | `GpuKernel` (0004 state) | Set by draws on reset (generation mode) or `set_nourishing_food_kind` (supervisor mode) | Kernel-owning thread |
| `supervisor_slot`, `supervisor_archive`, `supervisor_event`, `run.mode` | `SupervisorStore` / `governor::init_schema` | Created idempotently; written per boundary in one transaction | One writer connection on the kernel-owning thread; loaders open read-only |
| `RunSummaryAccumulator` | the driver (generation loop or `SupervisorRun`) | One run; serialised to `--run-summary` | Single-threaded CPU |
| `EvolutionSnapshot.supervisor` | main thread (`gpu_orchestration.rs`) | Updated from `SimEvent::SupervisorSample` | Main thread only; never reads the kernel |
| `results.json`, `DECISION.md` | `supervisor-decision` (architect) | Written once from validated inputs | None |

## Untrusted inputs

| input | bounds |
|---|---|
| Config JSON `governor.evolution_mode`, `governor.supervisor.*` | serde enums (unknown variant fails to parse); numeric fields must pass `SupervisorConfig::validate` before any kernel is built. |
| CLI `--evolution-mode`, `--supervisor-inheritance` | clap value enums; anything else exits 1. |
| CLI `--agent-tick-budget N` | `u64`; in supervisor mode validated as above; in generation mode it must be a multiple of `population_size * tick_budget` and cannot be combined with `--generations`; violations exit 1. |
| CLI `--resume` with supervisor mode | refused, exit 1. |
| CLI `--run-summary PATH`, `--free-life-archive-db PATH`, `--free-life-archive-rank N` | Path written once / opened read-only; I/O and SQL errors are returned, never panicked; a rank beyond the archive is an error. |
| Archive blobs and `config_json` read back | `decode_brain_state` size validation; strict `BrainConfig` parse; a brain whose length does not match the layout is refused before any write. |
| Physics rows passed to `on_sample` | length must be `agent_count * PHYS_STRIDE`, else an error; non-finite values make a slot's score non-finite, which is never archived and never below a median (treated as not eligible). |
| `supervisor_verdict` input | exactly the 32 named files or a bundle with exactly those entries; each parses, has the expected mode, inheritance, seed, food-kind flag, 10 agents, 32,000,000 agent-ticks (summaries) or 300,000 ticks and the expected `brain_source` (free lives); finite numbers; every metric defined; expected text one of `ADOPT`, `OPTION`, `REJECT`. Anything else is an error with a non-zero exit. |

## Feasibility

| task | constraints and sources | compatibility witness | preflight check |
|---|---|---|---|
| evolution-mode-config | `GovernorConfig` derives serde with per-field defaults and an explicit `impl Default` (`crates/xagent-shared/src/config.rs`), embedded in `FullConfig`; `resume` parses the stored config with `unwrap_or_default`, so old JSON loads with the new serde defaults; the only full `GovernorConfig` literals are in `governor.rs` tests and `tests/integration.rs` (others use `..Default::default()`). This task edits only those literals in the two sandbox files. | The config struct, re-export list and every literal site exist at baseline | `pf-evolution-mode-config` |
| run-mode-column | `governor.rs` has the private `composite_fitness`, blob codec and `init_schema` with the `let _ = db.execute_batch("ALTER TABLE run ...")` migration pattern (also repeated in `resume`), `resume`'s latest-run query, and the inline grid-size expression in `evaluate`; `tempfile` is a dev-dependency for file databases; the digest test is in the 0001 plan. | Every hook site exists at baseline; adapter present for the digest check | `pf-run-mode-column` |
| run-summary | 0004 task `food-kind-metrics` declares the public `nearest_food_of_kind` and `pub struct NearestFood { pub distance: f32, pub bearing: f32 }` (its sign helper is not public, so run-summary re-states `-signum(bearing)` locally and pins it with a unit test); 0004 adds `P_HARMFUL_MEAL_COUNT`; `SCENT_RANGE` is re-exported by `xagent-shared`; `P_FOOD_COUNT` 18, `P_DEATH_COUNT` 23, `P_MOTOR_TURN_OUT` 28; the module slot is free. | The helper signatures, physics slots and range constant exist (baseline or committed 0004 text) | `pf-run-summary` |
| supervisor-core | `composite_fitness` (made crate-visible by run-mode-column), `HEATMAP_RES` 64, public `mutate_steering_weights`, `mutate_sensory_genes`, `steering_population`, `fresh_brain_state`, `MutationMomentum::new(decay)`, `rand` dependency; CPU only. | All APIs exist at baseline | `pf-supervisor-core` |
| supervisor-store | Blob codec and `node_brain` schema in `governor.rs`, WAL/busy-timeout pragma line, rusqlite 0.34 bundled (read-only open flag), `tempfile` dev-dependency, the 0001 read-only loader convention. | Codec, schema pattern and SQLite features exist | `pf-supervisor-store` |
| kernel-supervisor-hooks | `write_agent_state`, `write_agent_heritable_config`, `write_agent_physics_fields` are the existing single-slot writes; `upload_agents` builds the spawn row; `write_world_config_scratch` re-uploads the world config every dispatch, so a kind field set between dispatches reaches the uniform; `set_freeze_steering_weights` is the setter precedent; 0004 adds `nourishing_food_kind` and `food_kinds_slot_value`. | The writes, the row builder, the upload path and the 0004 kernel text exist; adapter present | `pf-kernel-supervisor-hooks` |
| supervisor-run | `kernel_batch_size` (10 at default strides, divides 100), `dispatch_batch`, `try_collect_state_snapshot`, `cached_state`, `read_agent_state`, `probe_kernel_batches`, `reset_agents_seeded`, `WorldState::safe_spawn_position` (seeded spawn stream), headless `HEATMAP_INTERVAL` 100. | All kernel and world APIs exist; adapter present | `pf-supervisor-run` |
| headless-supervisor-driver | `run_headless`'s signature (used by the digest test), its kernel construction and snapshot spin loop; 0001's `run_free_life` lives in the same file. | Entry point and loop pattern exist; adapter present | `pf-headless-supervisor-driver` |
| generation-run-summary | The generation loop's per-chunk sample site, its final per-agent readout of `P_DEATH_COUNT`, `evaluate`, `evolution_complete` (which fixes how many generations run for a `max_generations`), and `cached_food_state`. | The measurement sites exist; adapter present | `pf-generation-run-summary` |
| archive-brain-source | 0001's `BrainSource`, `FreeLifeReport` and `load_champion_read_only` shapes (committed 0001 architecture) and `run_headless` in `headless.rs`. | The committed 0001 interface text and the entry point exist; adapter present | `pf-archive-brain-source` |
| evolution-cli | clap `#[arg(long)]` CLI with `seed: Option<u64>`, `resolve_config`, the stderr banner and the `run_headless` call in `main.rs`; 0001's free-life champion flags; `tempfile` for the exit test. | CLI and dispatch sites exist | `pf-evolution-cli` |
| worker-supervisor | `SimInit`, `clamp_ticks_to_generation_budget` (the clamp precedent), `GenerationBudgetReached` (control-event precedent), `advance_compute`, the `test_init` GPU test fixture, `read_full_state_blocking`, `dispatch_ticks`; `gpu_orchestration.rs` holds the only other `SimInit` literal and the exhaustive `SimEvent` match, which this task extends minimally (`supervisor: None`, one new arm). | Worker structure, readback APIs and the two main-thread sites exist; adapter present | `pf-worker-supervisor` |
| gui-supervisor-wiring | `EvolutionSnapshot` and its `Default`, `render_running_dashboard`, `start_sim_worker`'s `SimInit` literal, the `GenerationBudgetReached` drain arm, `EvolutionAction::Start`, and the existing snapshot default test. | The main-thread sites exist | `pf-gui-supervisor-wiring` |
| supervisor-verdict | One explicit `[[bin]]`; 0004 adds `default-run = "xagent"` and the auto-discovered `src/bin/food_kind_verdict.rs`, so `src/bin/supervisor_verdict.rs` is auto-discovered too; clap is a dependency; 0001's `WindowMetrics` has `sighting_meal_share` and `turn_toward_food`; CPU only. | The bin layout, clap and the metric fields exist (baseline or committed text) | `pf-supervisor-verdict` |
| supervisor-decision | Release GPU runs through the sandbox binary (`--bench` route at baseline; the new flags after evolution-cli); `--dump-config` gives the base JSON; `target/` is ignored; the record directory is free. | The action builds the release binary; the witness runs a 2,000-tick 10-agent release bench on the GPU | `pf-supervisor-decision` |
