# 0007 Supervisor live resume — architecture

Binding specification for the tasks in `tasks/`. Paths are repository-relative.
Plan 0006 (`0006-supervisor-decision-telemetry`) lands first; "the plan 1
supervisor digest" below is the seeded headless supervisor digest that 0006's
gate computes (its ARCHITECTURE.md names the function and the test file).

## Design

A checkpoint is the complete **pre-sample** state of a supervisor run at a
boundary tick `T`: the GPU buffers as they stand after the batch ending at `T`
was simulated, and every CPU-side field of `SupervisorRun` (and of the kernel)
as it stood **before** `SupervisorRun::on_sample(T)` ran. Resuming rebuilds the
kernel and the run from it and then calls `on_sample(T)` again with the
checkpointed physics and food rows, so every decision from `T` on is recomputed
by the same code from the same inputs.

- **Cadence.** A checkpoint is due at every boundary tick (the core's boundary
  schedule, `Supervisor::next_boundary_tick()`), and only when the run has a
  store. `SupervisorRun::checkpoint_due(tick) -> bool`.
- **GPU readback.** At a due tick the driver replaces the readback it already
  performs there (headless: the `request_state_snapshot` that `dispatch_batch`
  issues plus its collect; worker: `read_full_state_blocking`) with one
  `GpuKernel::read_checkpoint_blocking() -> KernelCheckpoint`, which records
  every captured buffer's copy into one staging buffer in **one command
  encoder and one `queue.submit`**, and returns the physics rows and food rows
  as well; a terminal map failure is logged and panics, as in
  `read_full_state_blocking` (approved exception, AUTHORIZATION.md). Headless calls `dispatch_ticks` (not `dispatch_batch`) on that batch.
  No compute pass is added.
- **Kernel state (`crates/xagent-brain/src/gpu_kernel/checkpoint.rs`).**
  `KernelCheckpoint { sections: Vec<KernelSection>, cpu: KernelCpuState }`,
  `KernelSection { name: &'static str /* stable id */, bytes: Vec<u8> }`. The
  captured set is every `wgpu::Buffer` owned by `GpuKernel` or its submodules
  whose contents carry over from one batch to the next (at least
  `agent_phys`, `decision`, `food_state`, `food_flags`, `sensory`,
  `_sensory_next`, `brain_state`, `brain_scratch`, `pattern`, `brain_config`,
  `trail_ring`, and any persistent submodule buffer such as `global_credit`'s).
  Buffers rebuilt from the seed (`heightmap`, `biome` via `upload_world`) or
  rewritten before every read in each dispatch (world-config uniforms,
  `dispatch_args`, grids, collision scratch) are listed in a `DERIVED_BUFFERS`
  doc table with the reason, not captured. `KernelCpuState` holds
  `food_kind_draws`, `nourishing_food_kind`, `active_config_index`,
  `freeze_steering_weights`, `trunk_lesions` and every other non-cache field
  that alters dispatch output; it has hand-written `to_bytes(&self) ->
  Vec<u8>` / `from_bytes(&[u8]) -> Result<Self, KernelRestoreError>` (no serde:
  `xagent-brain` gains no dependency and `Cargo.lock` does not change).
  `KernelCheckpoint::physics(&self) -> &[f32]` and `KernelCheckpoint::food(&self)
  -> Option<&[f32]>` return the captured physics and food rows.
  `GpuKernel::queue_submit_count(&self) -> u64` counts every queue submit the
  kernel makes. `GpuKernel::restore_checkpoint(&KernelCheckpoint)
  -> Result<(), KernelRestoreError>` writes the sections back with
  `queue.write_buffer`, restores the CPU fields, invalidates every CPU-side
  cache (packed encoder, context/food-grid/physics caches) as the existing
  single-slot writes do, and refuses an unknown, missing or wrong-length
  section.
- **Supervisor core state (`supervisor.rs`).** Every random stream the core and
  `TickSchedule` own becomes a crate-local `StreamRng { inner: SmallRng, seed:
  u64, steps: u64 }` implementing `RngCore` by delegating to `SmallRng` and
  counting state steps (`next_u32` and `next_u64` each advance Xoshiro256++ by
  one step; `fill_bytes` delegates and counts `ceil(len / 8)` steps).
  Restoring re-seeds and discards `steps` outputs. Draw values are unchanged,
  so the digest is unchanged. `Supervisor::capture() -> SupervisorState` and
  `Supervisor::restore(SupervisorState) -> Result<Supervisor, SupervisorError>`
  cover every field: slots' ledgers (occupant, born tick, birth brain, config,
  score window, life cells, pending death), the archive with its next entry id,
  both schedules, every stream, the occupant counter and every field 0006
  added. `SupervisorState` is `Serialize + Deserialize`; every float inside
  it serialises by bits through `supervisor_checkpoint::f32_bits` (see the
  envelope), never as a JSON float.
- **Run state (`supervisor_run.rs`).** `SupervisorRun::capture_state() ->
  SupervisorRunState` destructures `Self` exhaustively (a new field is a
  compile error until it is handled) and covers: the core state, the season
  schedule, `nourishing`, `tick`, `budget`, `next_occupant`, the number of
  `WorldState::safe_spawn_position` calls made so far (`spawn_draws`), the
  `RunSummaryAccumulator` when measuring (`run_summary.rs` derives
  `Serialize, Deserialize` on it and its parts) and every telemetry field 0006
  added. `SupervisorRun::resume(kernel, config, store, state, &KernelCheckpoint)
  -> Result<Self, SupervisorRunError>` builds `WorldState::new(config.world)`,
  replays `spawn_draws` calls of `safe_spawn_position`, calls
  `upload_world`, then `kernel.restore_checkpoint`, then rewrites every
  `supervisor_slot` row from the restored ledgers. It never calls
  `reset_agents_seeded`, `upload_agents` or `replace_agent_slot`.
  `supervisor_run.rs` also owns `pub fn load_resume_point(db_path) ->
  Result<ResumePoint, CheckpointError>` (`ResumePoint { run_id, tick, config:
  FullConfig, state: SupervisorRunState, kernel: KernelCheckpoint, physics:
  Vec<f32>, food: Option<Vec<f32>> }`: read via the store, decode, check
  config against the run row) and `pub fn load_checkpoint_config(db_path)`.
  `SupervisorRun::write_checkpoint(&mut self, tick: u64, &KernelCheckpoint)` captures,
  encodes and stores; it must be called before `on_sample` at a due tick.
- **Envelope (`supervisor_checkpoint.rs`).** Pure, GPU-free codec:
  `encode(&Checkpoint) -> Vec<u8>` and `decode(&[u8]) -> Result<Checkpoint,
  CheckpointError>`. Layout, all little-endian: magic `b"XAGCKPT\0"` (8 bytes),
  `u32 FORMAT_VERSION = 1`, `u64 tick`, `u32 section_count`, then per section
  `u16 name_len`, name bytes (UTF-8), `u64 len`, bytes; then `u64` FNV-1a-64 of
  every preceding byte. Sections: `config` (serde_json of `FullConfig`),
  `run` (serde_json of `SupervisorRunState`; every `f32` and `Vec<f32>` inside
  it, brain states included, serialises as its `to_bits()` `u32` through the
  crate-local `supervisor_checkpoint::f32_bits` serde helpers, so NaN
  payloads, `-0.0` and subnormals round-trip bit-exactly), and one `kernel.<name>` section per `KernelSection`.
  `CheckpointError` variants: `Truncated`, `BadMagic`, `VersionMismatch {
  found, expected }`, `ChecksumMismatch`, `TooLarge { len, max }`,
  `Malformed(String)`, `ConfigMismatch`, `Missing`. Each `Display` contains the
  word `checkpoint` and states the reason in plain words (for example
  `no supervisor checkpoint found`, `checkpoint is truncated`, `checkpoint
  format version 2 is not supported (expected 1)`); callers prepend the
  database path.
- **Storage (`supervisor_store.rs`).** Table `supervisor_checkpoint (run_id
  INTEGER PRIMARY KEY REFERENCES run(id), tick INTEGER NOT NULL, format_version
  INTEGER NOT NULL, simulation_ticks INTEGER, payload BLOB NOT NULL)` (the
  run row's `simulation_ticks` copied at write time) created idempotently in
  `ensure_schema`. `SupervisorStore::write_checkpoint(&mut self, tick,
  &[u8])` is one `INSERT OR REPLACE` in one transaction: exactly one row per
  run, overwritten each time. `read_checkpoint(db_path) -> Result<StoredCheckpoint
  { run_id, tick, format_version, payload }, StoreError>` opens read-only and selects the
  newest supervisor run's row. `SupervisorStore::reopen_for_resume(db_path,
  run_id, tick)` opens read-write and, in one transaction, deletes from every
  table that has both a `run_id` and a `tick` column (discovered from
  `sqlite_master` and `pragma_table_info`, so 0006's tables are covered) the
  run's rows with `tick >= T`, and from `supervisor_archive` rows with
  `retired_tick >= T`; it resets `run.simulation_ticks` to the value stored
  at checkpoint time when the column exists.
- **Headless (`headless.rs`, `main.rs`).** `run_headless_with_options(config,
  db, resume = true, options)` in supervisor mode reads the stored checkpoint,
  decodes it, uses the checkpoint's `config` section as the run config (the
  supplied config is ignored; one line says so when it differs), rebuilds the
  kernel with `GpuKernel::new` from that config, resumes the run, replays
  `on_sample(T)` and continues the loop. `HeadlessError::ResumeInSupervisorMode`
  is replaced by `HeadlessError::Checkpoint(CheckpointError)`.
  `HeadlessOptions` gains `stop_after_checkpoint_at_or_after: Option<u64>`:
  the loop returns `Ok` immediately after writing the first checkpoint whose
  tick is `>=` the value, without calling `finish` (simulating a closed app).
  `main.rs` drops the `--resume is not supported in supervisor mode` refusal.
- **Worker (`sim_runtime.rs` supervisor path, `gpu_orchestration.rs`).**
  `SupervisorWorkerInit` gains `resume: bool`. With `resume`, the worker reads
  and decodes the checkpoint from `db_path`, refuses (one `SimEvent::Error`,
  supervisor left `None`, paused) when `db_path` is `None`, the checkpoint is
  invalid, or the checkpoint config differs from `init.config`; otherwise it
  restores, sets `self.tick = T`, replays `on_sample(T)` and continues. At due
  ticks it calls `read_checkpoint_blocking` instead of
  `read_full_state_blocking` and writes the checkpoint before `on_sample`.
  `supervisor_run::load_checkpoint_config(db_path) -> Result<FullConfig,
  CheckpointError>` is the read-only call the GUI (plan 0008) uses to fill
  `init.config`. `gpu_orchestration.rs` sets `resume: false` (the GUI entry
  point belongs to 0008).

## Invariants

| id | invariant | guards | enforced at |
|---|---|---|---|
| INV-1 | A seeded headless supervisor run stopped right after a checkpoint and resumed from it produces the same plan 1 supervisor digest as the same run uninterrupted (both with a store, so both write checkpoints), and the uninterrupted digest equals plan 0006's pinned golden digest for that seeded config under the same adapter rule 0006's gate uses, so writing checkpoints changes no decision. | bit-identical resume; checkpointing never changes decisions | `crates/xagent-sandbox/tests/supervisor_resume_digest.rs` (resume-digest-gate) |
| INV-2 | A checkpoint that is truncated, has a flipped byte, a wrong magic, a different `FORMAT_VERSION`, an oversized length, or a config section differing from the run row is refused with a `CheckpointError` naming the reason; nothing is resumed, no kernel is dispatched and the database is not modified. | corrupt checkpoints are never resumed | `supervisor_checkpoint.rs` unit tests (checkpoint-envelope); `tests/supervisor_resume_digest.rs` (resume-digest-gate) |
| INV-3 | Each run holds at most one `supervisor_checkpoint` row, written by one `INSERT OR REPLACE` inside one transaction; a run with many boundaries ends with exactly one row. | atomic, bounded checkpoint storage | `supervisor_store.rs` tests (checkpoint-store) |
| INV-4 | `read_checkpoint_blocking` issues exactly one `queue.submit` and no compute pass; a headless run and a worker run make the same number of queue submits and kernel batches with checkpointing on and off. | no extra GPU dispatch or readback | `crates/xagent-brain/tests/kernel_checkpoint.rs` (kernel-checkpoint); `sim_runtime.rs` tests (worker-resume, via `queue_submit_count` and the worker dispatch counters) |
| INV-5 | Restoring a `KernelCheckpoint` into a freshly built kernel (after `upload_world`) and dispatching N more ticks yields physics rows, food rows and every agent's `read_agent_state` bit-equal to the original kernel after the same N ticks. | kernel state is captured completely | `crates/xagent-brain/tests/kernel_checkpoint.rs` (kernel-checkpoint) |
| INV-6 | `Supervisor::restore(sup.capture())` yields a core whose next 1,000 samples (same inputs) produce the same decisions, outcomes and stream draws as `sup`; a `StreamRng` yields the same values as a `SmallRng` with the same seed. | every supervisor random stream position and ledger field resumes | `supervisor.rs` unit tests (supervisor-core-state) |
| INV-7 | `SupervisorRun::capture_state` destructures `SupervisorRun` exhaustively, and a run resumed from its state on a restored kernel produces the same `SampleReport` sequence and store rows as the original. | no run field is forgotten | `crates/xagent-sandbox/tests/supervisor_run_resume.rs` (supervisor-run-resume) |
| INV-8 | Resuming deletes exactly the run's rows with tick `>= T` (archive rows by `retired_tick >= T`) in one transaction before any write, so replayed rows are not duplicated; rows of other runs and rows before `T` are untouched. | resumed database equals uninterrupted database | `supervisor_store.rs` tests (checkpoint-store) |
| INV-9 | The simulation worker started with `resume: true` on a checkpointed database continues the run and reports the same boundary sequence as an uninterrupted worker run; with an invalid checkpoint it emits one `SimEvent::Error`, stays paused and dispatches nothing. | worker start-from-checkpoint | `sim_runtime.rs` tests (worker-resume) |
| INV-10 | Headless `--resume` in supervisor mode resumes from the checkpoint; without a checkpoint it fails with `CheckpointError::Missing`; generation-mode resume is unchanged. | CLI resume replaces the refusal | `tests/supervisor_headless.rs`, `tests/evolution_cli_exit.rs`, `main.rs` tests (headless-resume) |

## State

| state | owner | lifecycle | concurrency |
|---|---|---|---|
| `supervisor_checkpoint` row | `SupervisorStore` of the running driver | created on first due boundary, overwritten at every later one; read on resume | one writer connection per run (WAL, `busy_timeout`); readers open read-only |
| `KernelCheckpoint` | driver call frame | built by `read_checkpoint_blocking`, encoded, dropped; on resume decoded and consumed by `restore_checkpoint` | headless thread or the simulation worker thread only; never crosses to the main thread |
| `SupervisorRunState` / `SupervisorState` | `SupervisorRun` | captured at a due tick before `on_sample`; consumed by `resume` | same thread as the run |
| `StreamRng` step counters | `Supervisor` / `TickSchedule` | grow with each draw; restored by replay | single-threaded |
| `spawn_draws` | `SupervisorRun` | counts `safe_spawn_position` calls; replayed on resume | single-threaded |

## Untrusted inputs

| input | bounds |
|---|---|
| checkpoint payload BLOB read from SQLite | `len <= MAX_CHECKPOINT_BYTES` (512 MiB) before decoding; magic, `FORMAT_VERSION == 1`, FNV-1a-64 checksum and every section length checked against the remaining bytes before use; section names UTF-8 and `<= 64` bytes; `section_count <= 256`; kernel sections must match the target kernel's buffer lengths exactly |
| `config` section JSON | parsed with serde into `FullConfig`; must equal the run row's stored governor, brain and world JSON after re-serialisation, else `ConfigMismatch` |
| `run` section | serde decode; slot count must equal `population_size`; the envelope tick `T` must be a multiple of `SUPERVISOR_SAMPLE_TICKS`; the state's `tick` (last handled sample, pre-sample) must be `< T` and its next sample tick must equal `T`; stream step counts `<= 2^40` |

## Feasibility

| task | constraints and sources | compatibility witness | preflight check |
|---|---|---|---|
| kernel-checkpoint | `GpuKernel` owns the buffers listed above and already has a one-submit blocking copy (`read_full_state_blocking`) and `upload_world`; `xagent-brain` has no serde dependency and none is added (`crates/xagent-brain/Cargo.toml`); `gpu_kernel.rs` declares submodules with `mod x;` (`crates/xagent-brain/src/gpu_kernel.rs`) | the named buffer fields, `read_full_state_blocking`, `food_kind_draws` and `upload_world` exist; `gpu_kernel/checkpoint.rs` does not | pf-kernel-checkpoint |
| supervisor-core-state | the core's streams are `SmallRng` fields (`rng`, `tiebreak_rng`, `parent_rng`) and `stream_seed` exists (`crates/xagent-sandbox/src/supervisor.rs`); `serde` is a sandbox dependency (`crates/xagent-sandbox/Cargo.toml`); `rand = "0.9"` (`Cargo.toml`) | grep of those declarations | pf-supervisor-core-state |
| checkpoint-envelope | `lib.rs` declares `pub mod supervisor_store;` and has no checkpoint module; no new dependency is needed (FNV-1a is hand-written) | grep and absence test | pf-checkpoint-envelope |
| checkpoint-store | `ensure_schema`, `SupervisorStore::create` and the `supervisor_event` table with `run_id` and `tick` exist (`crates/xagent-sandbox/src/supervisor_store.rs`); `rusqlite 0.34` bundled and `tempfile` (`crates/xagent-sandbox/Cargo.toml`) | grep | pf-checkpoint-store |
| supervisor-run-resume | `SupervisorRun` fields `season`, `next_occupant`, `on_sample` (`supervisor_run.rs`), `RunSummaryAccumulator` (`run_summary.rs`), `safe_spawn_position` (`world/mod.rs`) | grep | pf-supervisor-run-resume |
| headless-resume | `ResumeInSupervisorMode`, `supervisor_step` (`headless.rs`), the CLI refusal (`main.rs`), the refusal tests (`tests/supervisor_headless.rs`, `tests/evolution_cli_exit.rs`) | grep | pf-headless-resume |
| worker-resume | `SupervisorWorkerInit`, `start_supervisor` and the sample `read_full_state_blocking` (`sim_runtime.rs`); the `SupervisorWorkerInit {` literal (`gpu_orchestration.rs`) | grep | pf-worker-resume |
| resume-digest-gate | `run_headless_with_options` (`headless.rs`), existing supervisor headless test file; the gate file does not yet exist | grep and absence test | pf-resume-digest-gate |
