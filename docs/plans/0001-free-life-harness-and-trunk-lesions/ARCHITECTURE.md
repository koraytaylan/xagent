# Architecture: free-life harness and trunk lesions

This document is the binding specification. Task packets refer to its
sections instead of repeating them.

## Components

| Component | File(s) | Owner task id |
|---|---|---|
| Baseline digest test and fixture | `crates/xagent-sandbox/tests/seeded_headless_digest.rs`, `crates/xagent-sandbox/tests/fixtures/seeded_headless_digest.json` | seeded-headless-digest |
| Kernel measurement layout (Rust and WGSL constants, types, packers, decoder) | `crates/xagent-brain/src/buffers.rs`, `crates/xagent-brain/src/shaders/kernel/common.wgsl` | kernel-measurement-layout |
| Lesion branches, probe writes, kernel setters and readback, GPU tests | `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`, `crates/xagent-brain/src/gpu_kernel.rs`, `crates/xagent-brain/tests/trunk_lesions.rs`, `crates/xagent-brain/tests/brain_tick_probe.rs` | kernel-lesions-and-probe |
| Read-only champion loader | `crates/xagent-sandbox/src/governor.rs` | read-only-champion-loader |
| Sample, metrics, tick log | `crates/xagent-sandbox/src/lib.rs`, `crates/xagent-sandbox/src/free_life/mod.rs`, `crates/xagent-sandbox/src/free_life/metrics.rs`, `crates/xagent-sandbox/src/free_life/tick_log.rs` | free-life-metrics-and-log |
| Free-life runner and its tests | `crates/xagent-sandbox/src/headless.rs`, `crates/xagent-sandbox/tests/free_life.rs` | free-life-runner |
| CLI flags and bin tests | `crates/xagent-sandbox/src/main.rs` | free-life-cli |
| Binary exit test | `crates/xagent-sandbox/tests/free_life_cli.rs` | free-life-cli-exit-test |
| Lesion matrix, decision record, record test | `docs/measurements/trunk-lesions/lesion-matrix.json`, `docs/measurements/trunk-lesions/DECISION.md`, `crates/xagent-sandbox/tests/lesion_matrix_record.rs` | trunk-lesion-matrix |

Dependency order: seeded-headless-digest → kernel-measurement-layout →
kernel-lesions-and-probe; kernel-lesions-and-probe, read-only-champion-loader
and free-life-metrics-and-log → free-life-runner → free-life-cli →
trunk-lesion-matrix; free-life-cli → free-life-cli-exit-test (parallel with
trunk-lesion-matrix). read-only-champion-loader and free-life-metrics-and-log
have no prerequisites and run in parallel with the kernel chain.

Source rule for every task: no plan, task, workstream or decision-doc name in
source, comments or strings (`contributing_guard.rs` fails on `plan 0NNN` and
`000N-x` slugs); every non-trivial literal is a named, documented constant.

## Kernel measurement layout (kernel-measurement-layout)

Brain-config uniform grows from `array<vec4<f32>, 4>` (`CONFIG_SIZE = 16`, all
slots used) to `array<vec4<f32>, 6>` (`CONFIG_SIZE = 24`). Same names in
`buffers.rs` and `common.wgsl`:

| Slot | Constant | Meaning when `1.0` |
|---|---|---|
| 16 | `CFG_LESION_TURN_WEIGHTS` | the 128-d encoding turn dot (`s_turn_dot`) is left out of `turn` |
| 17 | `CFG_LESION_FORWARD_WEIGHTS` | the 128-d encoding forward dot (`s_forward_dot`) is left out of `forward`; the forward bias stays |
| 18 | `CFG_LESION_MEMORY_BLEND` | the recalled-memory motor blend is skipped |
| 19 | `CFG_LESION_CRITIC` | every value the critic forms (the TD value and the replay value) is the value bias alone |
| 20 | `CFG_BRAIN_TICK_PROBE` | the brain pass writes the probe block |
| 21-23 | padding | always `0.0` |

Rust items in `buffers.rs`:

- `pub struct TrunkLesions { pub turn_weights: bool, pub forward_weights:
  bool, pub memory_blend: bool, pub critic: bool }` deriving `Clone, Copy,
  Debug, Default, PartialEq, Eq`, with `NONE`, `ALL` and `is_none()`; not
  serde, not part of `BrainConfig` or `AgentBrainState`.
- `pub fn pack_trunk_lesions(cfg: &mut [f32], lesions: TrunkLesions)`,
  `pub fn pack_brain_tick_probe(cfg: &mut [f32], enabled: bool)`.
- Probe layout: `DECISION_PROBE = DECISION_MOTOR + 4`, `BRAIN_TICK_PROBE_LEN =
  148`, `DECISION_STRIDE = DECISION_PROBE + BRAIN_TICK_PROBE_LEN` (Rust and
  WGSL), field offsets `PROBE_ENCODING = 0` (128), `PROBE_PATHWAY_INPUTS =
  128` (10), then one slot each, in order: `PROBE_REWARD`, `PROBE_VALUE`,
  `PROBE_TD_ERROR`, `PROBE_FORWARD_NOISE`, `PROBE_TURN_NOISE`,
  `PROBE_TURN_INNOVATION`, `PROBE_ENCODING_TURN_OUTPUT`,
  `PROBE_SCENT_TURN_OUTPUT`, `PROBE_VISION_TURN_OUTPUT`,
  `PROBE_ENCODING_FORWARD_OUTPUT` (= 147).
- `pub struct BrainTickProbe { pub encoding: [f32; ENCODED_DIMENSION], pub
  pathway_inputs: [f32; 10], pub reward, pub value, pub td_error, pub
  forward_noise, pub turn_noise, pub turn_innovation, pub
  encoding_turn_output, pub scent_turn_output, pub vision_turn_output, pub
  encoding_forward_output: f32 }` deriving `Clone, Copy, Debug, PartialEq`,
  with `BrainTickProbe::from_decision_record(record: &[f32]) -> Option<Self>`
  (`None` unless `record.len() >= DECISION_STRIDE`).
- `pub const TD_TRACE_DECAY: f32` mirroring `TD_DISCOUNT * TD_LAMBDA`.
- Layout tests: every slot `< CONFIG_SIZE`, WGSL parity of every new
  constant, the `brain_config` array-size test, the packers, and the decoder.

WGSL items in `common.wgsl`: the constants above, the array size, and boolean
readers `lesion_turn_weights()`, `lesion_forward_weights()`,
`lesion_memory_blend()`, `lesion_critic()`, `brain_tick_probe_enabled()`
(each `bc_f32(slot) >= 0.5`). No other file changes behaviour: new slots are
zero and the decision stride is only larger, so the seeded digest stays equal.

## Lesion branches and probe (kernel-lesions-and-probe)

In `coop_predict_and_act` and the replay step of `coop_learn_and_store`
(`brain_passes.wgsl`), with uniform branches only and every barrier still
reached by all invocations:

- turn lesion: `turn = scent_dot + vision_dot` (no `s_turn_dot`);
- forward lesion: `forward = bias` (no `s_forward_dot`);
- memory lesion: the memory blend block is skipped;
- critic lesion: the TD value and the replay value are `O_VALUE_BIAS` alone.

Traces, encoder credit, weight updates and L2 balls are unchanged; with every
flag `0.0` the arithmetic is exactly today's. While `brain_tick_probe_enabled()`
the pass writes the probe block of the decision buffer every brain tick:

| Field | Source |
|---|---|
| encoding (128) | `s_memory_key`, the centred encoding the critic and turn channel read |
| pathway inputs (10) | `O_SCENT_WHITENED` (2), then `O_VISION_PATHWAY_INPUT` (8) as the visual trace update reads it this tick |
| reward | `s_pred_td[S_REWARD]` |
| value | this tick's value, after the critic lesion |
| TD error | `s_pred_td[S_TD_ERROR]` |
| forward noise, turn noise, turn innovation | `s_explore[0]`, `s_explore[1]`, `s_explore[2]` (executed) |
| encoding turn, scent turn, vision turn, encoding forward outputs | `s_turn_dot`, `scent_dot`, `vision_dot`, `s_forward_dot` as computed, even when lesioned |

`GpuKernel` gains `trunk_lesions: TrunkLesions` and `brain_tick_probe: bool`
(off at `new`), re-packed by `packed_brain_config` on every reset; `pub fn
set_trunk_lesions(&mut self, lesions: TrunkLesions)`, `pub fn
set_brain_tick_probe(&mut self, enabled: bool)` write their slots at once
(like `set_freeze_steering_weights`); getters `trunk_lesions()` and
`brain_tick_probe_enabled()`; `pub fn read_brain_tick_probes_blocking(&self,
out: &mut Vec<BrainTickProbe>)` copies the whole decision buffer once, maps it
once and decodes every agent; on any failure it logs and leaves `out` empty, so
callers detect failure by `out.len() != agent_count`.

## Read-only champion loader (read-only-champion-loader)

In `governor.rs`: `pub struct StoredChampion { pub node_id: i64, pub
generation: i64, pub config: BrainConfig, pub brain: AgentBrainState }`;
`pub enum ChampionLoadError` (`Debug + Display + Error`: open/query failures,
no stored champion, node without brain, malformed brain, malformed config);
`pub fn load_champion_read_only(db_path: &str, node_id: Option<i64>) ->
Result<StoredChampion, ChampionLoadError>`. Opens with
`SQLITE_OPEN_READ_ONLY` on the plain path, sets `busy_timeout`, runs no
schema, pragma-write or data-write statement. It never uses `immutable=1` or
`nolock`: `immutable=1` would miss rows still in the WAL. A read-only open of
a WAL database may create `-wal` and `-shm` sidecar files that remain after
the load; that is accepted, and the guarantee is that the main database file's
bytes are unchanged. `None` selects the `node_brain` row whose node has the
greatest `(generation, id)`. The blob goes through `decode_brain_state`;
`config_json` is parsed strictly (no default fallback).

## Free-life types (free-life-metrics-and-log)

`lib.rs` gains `pub mod free_life;`; `free_life/mod.rs` holds module docs and
`pub mod metrics; pub mod tick_log;` only. All types below derive `Clone,
Debug, PartialEq, Serialize, Deserialize` and have `pub` fields; counts are
`u64`, continuous metrics `f64`.

- `metrics::FreeLifeSample { brain_tick: u64, alive: bool, meals_total: u64,
  deaths_total: u64, on_hazard: bool, motor_turn, food_distance,
  food_bearing, hazard_distance, hazard_bearing: f32, encoding: Vec<f32>
  (ENCODED_DIMENSION), pathway_inputs: [f32; 10], reward, value, td_error,
  forward_noise, turn_noise, turn_innovation, encoding_turn_output,
  scent_turn_output, vision_turn_output, encoding_forward_output: f32 }`.
- `metrics::MetricsConfig { trace_decay: f64, total_brain_ticks: u64 }`
  (also `Copy`).
- `metrics::AgentAccumulator::new(config: MetricsConfig) -> Self`,
  `push(&mut self, sample: &FreeLifeSample)`, `finish(self) -> AgentMetrics`.
- `metrics::AgentMetrics { whole: WindowMetrics, late: WindowMetrics }`.
- `metrics::WindowMetrics { samples, alive_samples, meals, deaths,
  hazard_deaths, sightings, sightings_ending_in_meal: u64,
  sighting_meal_share, turn_toward_food, turn_away_from_hazard: f64,
  pathway_food_correlation: PathwayCorrelation, turn_credit: [f64; 10],
  forward_credit: [f64; 10], value_gap_in_view, value_gap_centred: f64 }`;
  `PathwayCorrelation { encoding, scent, vision: f64 }`.
- `tick_log::TickLogWriter::create(path: &Path, expected_rows: u64) ->
  io::Result<Self>`, `write_sample(&mut self, agent: u32, sample:
  &FreeLifeSample) -> io::Result<()>`, `finish(self) -> io::Result<()>`
  (error when rows written differ from expected).
- `tick_log::read_tick_log(path: &Path) -> Result<TickLog, TickLogError>`;
  `TickLog { columns: Vec<String>, rows: u64, values: Vec<f32> }` row-major
  (row r, column c at `r * columns.len() + c`), `column_index(&self, name)`.

Tick log format: one UTF-8 JSON header line `{"format":
"xagent-free-life-tick-log", "version": 1, "endianness": "little", "columns":
[...], "rows": N}` and `\n`, then `rows` rows of little-endian `f32`, one row
per (brain tick, agent), agents in index order within a tick. Columns in order:
`brain_tick, agent, alive, meals_total, deaths_total, on_hazard, motor_turn,
food_distance, food_bearing, hazard_distance, hazard_bearing, encoding_0 ..
encoding_127, pathway_input_0 .. pathway_input_9, reward, value, td_error,
forward_noise, turn_noise, turn_innovation, encoding_turn_output,
scent_turn_output, vision_turn_output, encoding_forward_output`. Integers are
stored exactly as `f32` below 2^24; the writer errors above that. The reader
caps the header length, requires the exact column list, and requires payload
bytes `== rows * columns * 4` by checked arithmetic.

## Metric definitions (free-life-metrics-and-log)

Constants (named, documented): `FOOD_IN_VIEW_DISTANCE = 5.5`, `VIEW_HALF_ANGLE
= π/4`, `HAZARD_AHEAD_DISTANCE = 10.0`, `CENTRED_BEARING = 0.15`,
`OFF_AXIS_BEARING = 0.45`, `VALUE_DISTANCE_BINS = 2` (equal bins over
`[0, FOOD_IN_VIEW_DISTANCE)`), `SIGHTING_GRACE_BRAIN_TICKS = 10`,
`LATE_WINDOW_FRACTION = 0.3`. Sign convention of the kernel's approach and
avoidance counters (`kernel_tick.wgsl`, `phase_physics.wgsl`): a negative
`food_bearing` is food to the right and a positive `motor_turn` turns right,
so `food_side = -sign(food_bearing)`; `away_side = sign(hazard_bearing)`;
`sign(0) = 0`. `motor_turn` is `P_MOTOR_TURN_OUT`, the executed turn.

Per agent, for `whole` (every sample) and `late` (samples with `brain_tick >=
total_brain_ticks - round(LATE_WINDOW_FRACTION * total_brain_ticks)`, integer
arithmetic on the rounded length, `brain_tick` counted from 0):

- `meals`, `deaths`: increments of the cumulative totals between consecutive
  samples (counted on every sample, alive or not; the first sample is the
  baseline). `hazard_deaths`: deaths whose previous sample was on hazard
  ground. Behavioural metrics below use alive samples only.
- `sightings`: onsets of food in view (`distance <= 5.5`, `|bearing| < π/4`)
  after at least one alive sample without food in view; a sighting ends in a
  meal when the meal total rises before food has been out of view for more
  than 10 samples; a death closes an open sighting without a meal;
  `sighting_meal_share = ending_in_meal / sightings` (0 when none).
- `turn_toward_food`: mean `motor_turn * food_side` over food-in-view samples.
  `turn_away_from_hazard`: mean `motor_turn * away_side` over samples with
  hazard within 10 units and `|bearing| < π/4`, off hazard ground.
- `pathway_food_correlation`: Pearson correlation of each turn output
  (encoding, scent, vision) with `food_side` over food-in-view samples; 0 when
  either variance is 0.
- `turn_credit[k]`, `forward_credit[k]`: per sample, `credit_k += td_error *
  trace_k`, then `trace_k = decay * trace_k + noise * x_k / (1 + |x|^2)` with
  `x` the ten pathway inputs and `noise` the turn innovation (turn) or the
  executed forward noise (forward); traces reset to 0 at a death; traces run
  over all samples, credit sums only the window's samples, and the report is
  the sum divided by the window's sample count.
- `value_gap_in_view`: mean value with food in view minus mean value without.
  `value_gap_centred`: per distance bin, mean value with `|bearing| < 0.15`
  minus mean value with `|bearing| > 0.45` over food-in-view samples, averaged
  over bins that have both; 0 when none has both.
- Every reported value is finite; undefined means are 0.

## Free-life run (free-life-runner)

In `headless.rs`: `pub enum BrainSource { Fresh, Champion { db_path: String,
node_id: Option<i64> } }`; `pub struct FreeLifeOptions { pub agent_count:
usize, pub ticks: u64, pub seed: u64, pub brain_source: BrainSource, pub
lesions: TrunkLesions, pub log_path: Option<PathBuf> }`;
`DEFAULT_FREE_LIFE_AGENTS = 10`, `DEFAULT_FREE_LIFE_TICKS = 300_000`; `pub fn
parse_lesions(list: &str) -> Result<TrunkLesions, FreeLifeError>` (comma list
of `turn-weights`, `forward-weights`, `memory-blend`, `critic`, `all`, `none`;
trimmed; empty = none; unknown name = error naming it); `pub fn
lesion_names(lesions: TrunkLesions) -> Vec<&'static str>` in that order; `pub
enum FreeLifeError` (`Debug + Display + Error`: no GPU, vision stride, agent
count, ticks, unknown lesion, champion load, champion layout, readback length,
log I/O); `pub struct FreeLifeReport { pub agent_count: usize, pub ticks: u64,
pub seed: u64, pub brain_source: String ("fresh" | "champion"), pub
champion_node_id: Option<i64>, pub champion_generation: Option<i64>, pub
lesions: Vec<String>, pub kernel_batch_size: u32, pub brain_ticks: u64, pub
agents: Vec<AgentMetrics> }` deriving `Clone, Debug, PartialEq, Serialize,
Deserialize`; `pub fn run_free_life(config: &FullConfig, options:
&FreeLifeOptions) -> Result<FreeLifeReport, FreeLifeError>`:

1. Reject no GPU adapter, `agent_count` outside `1..=MAX_AGENTS`, the chosen
   brain config's `vision_stride != 1`, `ticks == 0` or `ticks %
   kernel_batch_size() != 0`.
2. Brain config: `config.brain`, or the champion's config (its brain-state
   length must equal `BrainLayout::from_config(&config).brain_stride`).
   `WorldState::new(config.world with seed)`; agents at
   `safe_spawn_position`; `GpuKernel::new` with that brain config;
   `upload_world`; `upload_agents`; `rng = SmallRng::seed_from_u64(seed)`;
   `reset_agents_seeded(&brain_config, rng.random())`.
3. Birth brain: one `fresh_brain_state(&brain_config, &mut rng)`, or the
   champion's brain, written to every agent; then `write_agent_heritable_config`
   with the brain config.
4. `set_trunk_lesions(options.lesions)`, `set_brain_tick_probe(true)`. Nothing
   else in the crate calls these setters.
5. For each of `ticks / kernel_batch_size()` batches: `dispatch_batch`, probe
   and physics readback, one `FreeLifeSample` per agent into its
   `AgentAccumulator` (`MetricsConfig { trace_decay: TD_TRACE_DECAY,
   total_brain_ticks }`) and the optional `TickLogWriter` (expected rows =
   agents x batches, checked).
6. Finish into the report. Identical options and config give an identical
   report.

## CLI (free-life-cli)

`--free-life` (mode), `--free-life-agents N` (default 10), `--free-life-ticks N`
(default 300000), the existing `--seed`, `--free-life-champion-db PATH`,
`--free-life-champion-node ID` (requires the db), `--free-life-lesion LIST`
(default `none`), `--free-life-log PATH`, `--free-life-out PATH`. A pure
`free_life_options_from_cli(cli: &Cli, config: &FullConfig) ->
Result<FreeLifeOptions, String>` builds the options. The branch runs after
config resolution and before any stdout config print; the report JSON goes to
`--free-life-out` or stdout; any error goes to stderr with exit status 1. An
agent count outside the range prints `--free-life-agents must be in
1..={MAX_AGENTS}, got {N}`. `tests/free_life_cli.rs` runs the built binary
(`CARGO_BIN_EXE_xagent`) with `--free-life --free-life-agents 0` and asserts
exit status 1 and that message on stderr.

## Matrix and decision rule (trunk-lesion-matrix)

Arms: `control` (no lesion), `turn-weights`, `forward-weights`,
`memory-blend`, `critic`, `all`. Sources: `fresh`, and `champion` = the
default champion of the reference lineage of the same seed (`--dump-config`
default with `governor.tick_budget = 40000`, saved as
`target/trunk-lesions/base.json`, run with `--config
target/trunk-lesions/base.json --seed S --no-render --db
target/trunk-lesions/reference-sS.db --generations 80` into that ignored
database under `target/trunk-lesions/`).
Seeds 5-8, 10 agents, 300,000 ticks: 48 runs. Each run's report is written
under ignored `target/trunk-lesions/runs/`, and the 48 reports are joined by
shell into one JSON array committed as
`docs/measurements/trunk-lesions/lesion-matrix.json`.

For lesion L, source X and metric m: per seed, `d_s = mean over agents(m | L)
- mean over agents(m | control)`; `d = mean(d_s)`; `se = sd(d_s) / 2` (sample
standard deviation, four seeds). A change is significant when `|d| > 2 * se`
and `d != 0`.

- Behaviour needs L when, in either source, `whole.meals` falls or
  `whole.deaths` rises significantly.
- Within-life learning needs L when, in fresh brains,
  `late.pathway_food_correlation.scent`, `late.pathway_food_correlation.vision`
  or `late.turn_toward_food` falls significantly.

`DECISION.md` holds a verdict table `| part | behaviour | within-life learning
|` with one row per part (`turn-weights`, `forward-weights`, `memory-blend`,
`critic`) and cells exactly `needed` or `not needed`, followed by `d ± se` for
the metrics and all arms including `all`, the reference champions used, and
the reproduction commands. `lesion_matrix_record.rs` recomputes the verdicts
from `lesion-matrix.json` and fails on any disagreement.

## Invariants

| id | invariant | guards | enforced at |
|---|---|---|---|
| INV-1 | With every lesion off and the probe off, a seeded headless evolution run (seed 5, 2 generations, tick budget 2,000) produces exactly the digest captured at baseline `0976448`. | `seeded_headless_digest.rs` strict mode | seeded-headless-digest (capture); kernel-measurement-layout and kernel-lesions-and-probe checks; integration `seeded-digest-strict` |
| INV-2 | Lesion and probe switches default off and are not heritable: no field in `BrainConfig`, `GovernorConfig`, `AgentBrainState` or any database table; they survive agent resets; only the free-life runner sets them. | `trunk_lesions.rs` default/persistence tests; `tests/free_life.rs` | kernel-lesions-and-probe, free-life-runner |
| INV-3 | Each lesion removes exactly its term and nothing else. | `trunk_lesions.rs` per-switch GPU tests | kernel-lesions-and-probe |
| INV-4 | The probe is passive and exact: probe on and off give identical physics and brain state; probe fields equal the brain's own values. | `brain_tick_probe.rs` | kernel-lesions-and-probe |
| INV-5 | Loading a champion never writes the database: read-only flag (plain `SQLITE_OPEN_READ_ONLY`, never `immutable=1` or `nolock`), `busy_timeout`, no schema statement, and the main database file's bytes unchanged by the load (its `-wal` and `-shm` sidecars may appear and remain; only the main file is compared); malformed blob or config, missing row or missing table is an error. | `governor::tests::read_only_champion_*` | read-only-champion-loader |
| INV-6 | Metrics follow "Metric definitions" exactly. | `free_life::metrics` unit tests on hand-computed sequences | free-life-metrics-and-log |
| INV-7 | The tick log follows "Tick log format"; the reader rejects any other size, header or column list and round-trips every column. | `free_life::tick_log` unit tests | free-life-metrics-and-log |
| INV-8 | A free life is seeded and exact: identical options give identical reports; one sample per brain tick; a champion whose layout disagrees with its config is rejected. | `tests/free_life.rs` | free-life-runner |
| INV-9 | CLI inputs are bounded: agents `1..=MAX_AGENTS`, ticks positive, a champion node requires a database, unknown lesion names rejected; errors exit non-zero with a stderr message (the built binary run with `--free-life --free-life-agents 0` exits 1 and prints `--free-life-agents must be in 1..=MAX_AGENTS`). | `main.rs` bin tests; `tests/free_life_cli.rs` | free-life-cli (bounds); free-life-cli-exit-test (binary exit status and stderr) |
| INV-10 | `lesion-matrix.json` holds the complete matrix (6 arms x 2 sources x 4 seeds, 10 agents and 300,000 ticks each) and `DECISION.md`'s verdicts equal those recomputed by the decision rule. | `lesion_matrix_record.rs` | trunk-lesion-matrix |
| INV-11 | Rust and WGSL agree on the new uniform slots, `CONFIG_SIZE = 24`, the probe layout and `DECISION_STRIDE`; the packers and decoder place every field at its named offset. | `buffers::tests` | kernel-measurement-layout |

## State

| state | owner | lifecycle | concurrency |
|---|---|---|---|
| Lesion and probe flags | `GpuKernel` fields plus uniform slots 16-20 | off at `GpuKernel::new`; set by the setters; re-packed on every reset; dropped with the kernel | single-threaded owner; uniform written via `queue.write_buffer` before the next dispatch |
| Probe block | decision buffer, per agent | rewritten each brain tick while the probe is on; never read otherwise | read only by blocking readback after `dispatch_batch` returns |
| Free-life accumulators and report | `run_free_life` | created per run, finished at the end | one thread; one probe and one physics readback per brain tick |
| Tick log file | `TickLogWriter` in the runner | created with a header declaring rows; flushed and closed at the end; an I/O error aborts the run with an error | single writer |
| Champion database | caller's file | opened read-only for the load, closed before the run | `busy_timeout` set; never takes a write lock |
| Reference lineage databases, run reports, matrix jobs | trunk-lesion-matrix architect, under ignored `target/trunk-lesions/` | created by the lineage and free-life runs; only the joined `lesion-matrix.json` is committed | background jobs with output in their job directories; at most 4 concurrent GPU jobs |

## Untrusted inputs

| input | bounds |
|---|---|
| CLI flags | agents `1..=MAX_AGENTS`; ticks `> 0` and a multiple of `kernel_batch_size()`; lesion names from the fixed set; a node id needs a database and a `node_brain` row; output paths created or truncated. |
| Governor database | read-only; blob decoded by the size-validated decoder; `config_json` parsed strictly as `BrainConfig`; brain-state length checked against the config's layout. |
| Tick log read back | header length capped, fields and column list exact; payload length equals `rows * columns * 4` by checked arithmetic. |
| `lesion-matrix.json` read by the record test | deserializes as `Vec<FreeLifeReport>`; every metric finite; counts and parameters as in INV-10. |
| World and brain config from `--config` | existing `FullConfig` parsing; free life additionally rejects `vision_stride != 1`. |

## Feasibility

| task | constraints and sources | compatibility witness | preflight check |
|---|---|---|---|
| seeded-headless-digest | `run_headless(config, db_path, resume, has_gpu)` drives a seeded run into a governor DB (`headless.rs`); the DB keeps `node`, `agent_result` and `node_brain` rows (`governor.rs`); `*.db` is ignored (`.gitignore`); an adapter is reachable (`GpuKernel::is_available`). Capture precedes any kernel change. | architect-route probe: release build, `--dump-config`, `--bench` throughput, sandbox guard/clippy/fmt green; action `release-bench-run` | pf-task-seeded-headless-digest |
| kernel-measurement-layout | All 16 uniform slots are used (`CONFIG_SIZE = 16`, slot 15 = freeze, WGSL `array<vec4<f32>, 4>`), so the uniform must grow to 24; `DECISION_STRIDE = DECISION_MOTOR + 4` has exactly one Rust and one WGSL definition, so appending the probe block is a two-line stride change; buffer sizes derive from the constants. | developer-route probe: `cargo test -p xagent-brain` (a superset of the task's focused `--lib` target) and crate clippy/fmt green at baseline | pf-task-kernel-measurement-layout |
| kernel-lesions-and-probe | The four lesion sites and the executed-innovation line exist exactly as quoted in `brain_passes.wgsl` (single policy evaluation site); the pathway inputs live at `O_SCENT_WHITENED` and `O_VISION_PATHWAY_INPUT`; `set_freeze_steering_weights` is the setter pattern. | developer-route probe: `cargo test -p xagent-brain` green (a superset of the task's focused `--lib gpu_kernel` and policy `--test` targets), a GPU adapter runs `--bench` | pf-task-kernel-lesions-and-probe |
| read-only-champion-loader | `decode_brain_state` validates blob sizes and is private to `governor.rs`, so the loader lives there; `node_brain(node_id, data)` and `node(config_json, generation)` hold the champion; rusqlite 0.34 (bundled) offers `OpenFlags::SQLITE_OPEN_READ_ONLY`. | developer-route probe: `governor::tests` green at baseline | pf-task-read-only-champion-loader |
| free-life-metrics-and-log | Ground truth comes from `P_NEAREST_FOOD_BEARING` (34), `P_NEAREST_DANGER_BEARING` (40), `P_IN_DANGER_BIOME` (35); the module is declared beside the existing ones in `lib.rs`. Pure CPU code. | developer-route probe: sandbox lib tests and guard green at baseline | pf-task-free-life-metrics-and-log |
| free-life-runner | Reuses the generation setup of `run_headless`, `fresh_brain_state`, `kernel_batch_size()` (one brain tick per batch at `vision_stride = 1`), `read_full_state_blocking`, and `GpuKernel::is_available`. 300,000 ticks is 30,000 samples with two blocking reads each. | developer-route probe: `headless::tests` green, `--bench` runs on the adapter | pf-task-free-life-runner |
| free-life-cli | `Cli` is a clap derive struct with `--seed: Option<u64>`, reused as the free-life seed; bin tests run as `--bin xagent`; `main.rs` keeps its single tolerated planning reference. | developer-route probe: `--bin xagent -- --skip sim_runtime::` (the main.rs `tests` module; the bin's wall-clock sim_runtime tests stay in the integration `sandbox-tests` check) green at baseline | pf-cli-seed-flag |
| free-life-cli-exit-test | The `[[bin]]` target is named `xagent` (`crates/xagent-sandbox/Cargo.toml`), so Cargo sets `CARGO_BIN_EXE_xagent` for the package's integration tests; `tempfile` is a dev-dependency; the startup banner goes to stderr (`main.rs`), so the test asserts a substring, not the whole stderr; the zero-count rejection happens in the pure option builder before any GPU use. | developer-route probe: `--test contributing_guard` integration runner green at baseline | pf-xagent-bin-target |
| trunk-lesion-matrix | The record test must not name a plan (`contributing_guard.rs` matches `000N-x` slugs), so records live in `docs/measurements/trunk-lesions/`; lineage databases and per-run reports stay in ignored `target/`; the CLI supplies `--seed`. Budget: 4 lineages x 80 generations x 40,000 ticks plus 48 x 300,000-tick free lives, about 3-5 h at the about 5,000 ticks/s the witness bench measures on the baseline iGPU. | architect-route probe: release build and `--bench` throughput; action `release-bench-run` | pf-task-trunk-lesion-matrix |
