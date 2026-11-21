---
objective: >-
  Let a supervisor run survive closing the app: checkpoint and resume it
  bit-identically. Scope: persist, at seeded sample or boundary ticks,
  everything needed to continue the run exactly: every slot's current learned
  brain state and ledger (occupant, born tick, birth brain, config, score
  window, life cells, pending death), the archive, the boundary and season
  schedules, every supervisor random stream position, the nourishing kind, the
  tick and budget counters, and the world and agent physics state the kernel
  needs (terrain is reproducible from the seed; food positions, consumed flags
  and respawn timers, agent positions and vitals are not). Checkpoint writes
  must be atomic, bounded in size (overwrite or keep a small fixed number, never
  one per sample), and must not add GPU dispatches beyond the readback the
  supervisor already performs at that tick. Resume in both the headless driver
  and the simulation worker; replace today's refusal to resume a supervisor run.
  Proof: a gate showing that a seeded headless run stopped at a checkpoint and
  resumed produces the same digest (the plan 1 supervisor digest) as the same
  run uninterrupted, and that a corrupt, truncated or version-mismatched
  checkpoint is refused with a clear error rather than resumed. Owns the
  checkpoint tables and code, the headless resume path and the worker
  start-from-checkpoint path. The GUI entry point for resume belongs to
  supervisor-observatory-ui; generation-mode resume is left untouched for
  generation-mode-removal to delete.
depends_on:
  - 0006-supervisor-decision-telemetry
status: todo
toolchains:
  - cargo
  - rustc
  - rustup
  - cargo-fmt
  - rustfmt
  - cargo-clippy
  - clippy-driver
checks:
  - name: fmt
    command: cargo fmt --all -- --check
    fix: cargo fmt --all
  - name: clippy
    command: cargo clippy --workspace --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: workspace-tests
    command: env XAGENT_REQUIRE_GPU=1 cargo test --workspace
    timeout_ms: 7200000
  - name: integration-resume-digest
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_resume_digest
    timeout_ms: 3600000
preflight_checks:
  - name: gpu-adapter-present
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
  - name: pf-kernel-checkpoint
    command: |-
      sh -c 'grep -qF "pub fn read_full_state_blocking(&mut self) -> &[f32] {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "brain_state_buffer: wgpu::Buffer," crates/xagent-brain/src/gpu_kernel.rs && grep -qF "food_flags_buffer: wgpu::Buffer," crates/xagent-brain/src/gpu_kernel.rs && grep -qF "trail_ring_buffer: wgpu::Buffer," crates/xagent-brain/src/gpu_kernel.rs && grep -qF "food_kind_draws: u64," crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn upload_world(" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn read_agent_state(&self, index: u32) -> AgentBrainState {" crates/xagent-brain/src/gpu_kernel.rs && grep -qxF "mod main64_validation;" crates/xagent-brain/src/gpu_kernel.rs && test ! -e crates/xagent-brain/src/gpu_kernel/checkpoint.rs && test ! -e crates/xagent-brain/tests/kernel_checkpoint.rs'
  - name: pf-supervisor-core-state
    command: |-
      sh -c 'grep -qF "pub struct TickSchedule {" crates/xagent-sandbox/src/supervisor.rs && grep -qF "    rng: SmallRng," crates/xagent-sandbox/src/supervisor.rs && grep -qF "tiebreak_rng: SmallRng," crates/xagent-sandbox/src/supervisor.rs && grep -qF "parent_rng: SmallRng," crates/xagent-sandbox/src/supervisor.rs && grep -qF "pub fn stream_seed(run_seed: u64, tag: u64) -> u64 {" crates/xagent-sandbox/src/supervisor.rs && grep -qF "pub struct SlotLedger {" crates/xagent-sandbox/src/supervisor.rs && grep -qF "serde = { workspace = true }" crates/xagent-sandbox/Cargo.toml && grep -qF "rand = \"0.9\"" Cargo.toml'
  - name: pf-checkpoint-envelope
    command: |-
      sh -c 'grep -qxF "pub mod supervisor_store;" crates/xagent-sandbox/src/lib.rs && grep -qF "serde_json = " crates/xagent-sandbox/Cargo.toml && test ! -e crates/xagent-sandbox/src/supervisor_checkpoint.rs'
  - name: pf-checkpoint-store
    command: |-
      sh -c 'grep -qF "pub fn ensure_schema(db: &Connection) -> SqlResult<()> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub fn create(db_path: &str, config: &FullConfig) -> Result<Self, StoreError> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "CREATE TABLE IF NOT EXISTS supervisor_event (" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "retired_tick INTEGER NOT NULL," crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "rusqlite = { version = \"0.34\", features = [\"bundled\"] }" crates/xagent-sandbox/Cargo.toml && grep -qF "tempfile = \"3\"" crates/xagent-sandbox/Cargo.toml'
  - name: pf-supervisor-run-resume
    command: |-
      sh -c 'grep -qF "pub fn on_sample(" crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "season: TickSchedule," crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "next_occupant: u64," crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "pub struct RunSummaryAccumulator {" crates/xagent-sandbox/src/run_summary.rs && grep -qF "pub fn safe_spawn_position(&self) -> glam::Vec3 {" crates/xagent-sandbox/src/world/mod.rs && test ! -e crates/xagent-sandbox/tests/supervisor_run_resume.rs'
  - name: pf-headless-resume
    command: |-
      sh -c 'grep -qF "ResumeInSupervisorMode," crates/xagent-sandbox/src/headless.rs && grep -qF "fn supervisor_step(" crates/xagent-sandbox/src/headless.rs && grep -qF "pub struct HeadlessOptions {" crates/xagent-sandbox/src/headless.rs && grep -qF -- "--resume is not supported in supervisor mode" crates/xagent-sandbox/src/main.rs && grep -qF "fn supervisor_headless_refuses_resume() {" crates/xagent-sandbox/tests/supervisor_headless.rs && grep -qF "fn evolution_cli_exit_resume_supervisor() {" crates/xagent-sandbox/tests/evolution_cli_exit.rs'
  - name: pf-worker-resume
    command: |-
      sh -c 'grep -qF "pub struct SupervisorWorkerInit {" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn start_supervisor(" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "let physics = self.kernel.read_full_state_blocking().to_vec();" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "Some(SupervisorWorkerInit {" crates/xagent-sandbox/src/gpu_orchestration.rs'
  - name: pf-resume-digest-gate
    command: |-
      sh -c 'grep -qF "pub fn run_headless_with_options(" crates/xagent-sandbox/src/headless.rs && test -f crates/xagent-sandbox/tests/supervisor_headless.rs && test ! -e crates/xagent-sandbox/tests/supervisor_resume_digest.rs'
  - name: pf-integration-guard
    command: |-
      sh -c 'test -f crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF "cargo clippy --workspace --all-targets -- -D warnings" CONTRIBUTING.md'
execution_checks:
  - name: developer-route
    tasks:
      - kernel-checkpoint
      - supervisor-core-state
      - checkpoint-envelope
      - checkpoint-store
      - supervisor-run-resume
      - headless-resume
      - worker-resume
      - resume-digest-gate
    role: developer
    operations:
      - edit
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
    constraints:
      - task: kernel-checkpoint
        name: kernel-buffers-and-blocking-copy
        sources:
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-brain/Cargo.toml
        checks:
          - name: pf-kernel-checkpoint
            command: |-
              sh -c 'grep -qF "pub fn read_full_state_blocking(&mut self) -> &[f32] {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "brain_state_buffer: wgpu::Buffer," crates/xagent-brain/src/gpu_kernel.rs && grep -qF "food_flags_buffer: wgpu::Buffer," crates/xagent-brain/src/gpu_kernel.rs && grep -qF "trail_ring_buffer: wgpu::Buffer," crates/xagent-brain/src/gpu_kernel.rs && grep -qF "food_kind_draws: u64," crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn upload_world(" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn read_agent_state(&self, index: u32) -> AgentBrainState {" crates/xagent-brain/src/gpu_kernel.rs && grep -qxF "mod main64_validation;" crates/xagent-brain/src/gpu_kernel.rs && test ! -e crates/xagent-brain/src/gpu_kernel/checkpoint.rs && test ! -e crates/xagent-brain/tests/kernel_checkpoint.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: supervisor-core-state
        name: supervisor-streams-and-serde
        sources:
          - crates/xagent-sandbox/src/supervisor.rs
          - crates/xagent-sandbox/Cargo.toml
          - Cargo.toml
        checks:
          - name: pf-supervisor-core-state
            command: |-
              sh -c 'grep -qF "pub struct TickSchedule {" crates/xagent-sandbox/src/supervisor.rs && grep -qF "    rng: SmallRng," crates/xagent-sandbox/src/supervisor.rs && grep -qF "tiebreak_rng: SmallRng," crates/xagent-sandbox/src/supervisor.rs && grep -qF "parent_rng: SmallRng," crates/xagent-sandbox/src/supervisor.rs && grep -qF "pub fn stream_seed(run_seed: u64, tag: u64) -> u64 {" crates/xagent-sandbox/src/supervisor.rs && grep -qF "pub struct SlotLedger {" crates/xagent-sandbox/src/supervisor.rs && grep -qF "serde = { workspace = true }" crates/xagent-sandbox/Cargo.toml && grep -qF "rand = \"0.9\"" Cargo.toml'
      - task: checkpoint-envelope
        name: module-slot-and-json-dependency
        sources:
          - crates/xagent-sandbox/src/lib.rs
          - crates/xagent-sandbox/Cargo.toml
        checks:
          - name: pf-checkpoint-envelope
            command: |-
              sh -c 'grep -qxF "pub mod supervisor_store;" crates/xagent-sandbox/src/lib.rs && grep -qF "serde_json = " crates/xagent-sandbox/Cargo.toml && test ! -e crates/xagent-sandbox/src/supervisor_checkpoint.rs'
      - task: checkpoint-store
        name: store-schema-and-sqlite
        sources:
          - crates/xagent-sandbox/src/supervisor_store.rs
          - crates/xagent-sandbox/Cargo.toml
        checks:
          - name: pf-checkpoint-store
            command: |-
              sh -c 'grep -qF "pub fn ensure_schema(db: &Connection) -> SqlResult<()> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub fn create(db_path: &str, config: &FullConfig) -> Result<Self, StoreError> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "CREATE TABLE IF NOT EXISTS supervisor_event (" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "retired_tick INTEGER NOT NULL," crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "rusqlite = { version = \"0.34\", features = [\"bundled\"] }" crates/xagent-sandbox/Cargo.toml && grep -qF "tempfile = \"3\"" crates/xagent-sandbox/Cargo.toml'
      - task: supervisor-run-resume
        name: run-fields-summary-and-spawn
        sources:
          - crates/xagent-sandbox/src/supervisor_run.rs
          - crates/xagent-sandbox/src/run_summary.rs
          - crates/xagent-sandbox/src/world/mod.rs
        checks:
          - name: pf-supervisor-run-resume
            command: |-
              sh -c 'grep -qF "pub fn on_sample(" crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "season: TickSchedule," crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "next_occupant: u64," crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "pub struct RunSummaryAccumulator {" crates/xagent-sandbox/src/run_summary.rs && grep -qF "pub fn safe_spawn_position(&self) -> glam::Vec3 {" crates/xagent-sandbox/src/world/mod.rs && test ! -e crates/xagent-sandbox/tests/supervisor_run_resume.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: headless-resume
        name: headless-refusal-sites
        sources:
          - crates/xagent-sandbox/src/headless.rs
          - crates/xagent-sandbox/src/main.rs
          - crates/xagent-sandbox/tests/supervisor_headless.rs
          - crates/xagent-sandbox/tests/evolution_cli_exit.rs
        checks:
          - name: pf-headless-resume
            command: |-
              sh -c 'grep -qF "ResumeInSupervisorMode," crates/xagent-sandbox/src/headless.rs && grep -qF "fn supervisor_step(" crates/xagent-sandbox/src/headless.rs && grep -qF "pub struct HeadlessOptions {" crates/xagent-sandbox/src/headless.rs && grep -qF -- "--resume is not supported in supervisor mode" crates/xagent-sandbox/src/main.rs && grep -qF "fn supervisor_headless_refuses_resume() {" crates/xagent-sandbox/tests/supervisor_headless.rs && grep -qF "fn evolution_cli_exit_resume_supervisor() {" crates/xagent-sandbox/tests/evolution_cli_exit.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: worker-resume
        name: worker-supervisor-start-and-sample
        sources:
          - crates/xagent-sandbox/src/sim_runtime.rs
          - crates/xagent-sandbox/src/gpu_orchestration.rs
        checks:
          - name: pf-worker-resume
            command: |-
              sh -c 'grep -qF "pub struct SupervisorWorkerInit {" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn start_supervisor(" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "let physics = self.kernel.read_full_state_blocking().to_vec();" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "Some(SupervisorWorkerInit {" crates/xagent-sandbox/src/gpu_orchestration.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: resume-digest-gate
        name: headless-entry-and-test-slot
        sources:
          - crates/xagent-sandbox/src/headless.rs
          - crates/xagent-sandbox/tests/supervisor_headless.rs
        checks:
          - name: pf-resume-digest-gate
            command: |-
              sh -c 'grep -qF "pub fn run_headless_with_options(" crates/xagent-sandbox/src/headless.rs && test -f crates/xagent-sandbox/tests/supervisor_headless.rs && test ! -e crates/xagent-sandbox/tests/supervisor_resume_digest.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
  - name: integration-route
    tasks:
      - "$integration"
    role: architect
    operations:
      - edit
    command: cargo test -p xagent-sandbox --test contributing_guard
    timeout_ms: 3600000
    constraints:
      - task: "$integration"
        name: repository-gates-and-guard
        sources:
          - crates/xagent-sandbox/tests/contributing_guard.rs
          - CONTRIBUTING.md
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-integration-guard
            command: |-
              sh -c 'test -f crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF "cargo clippy --workspace --all-targets -- -D warnings" CONTRIBUTING.md'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
---

# 0007 Supervisor live resume

## Scope

Checkpoint a supervisor run at its boundary ticks and resume it bit-identically
in the headless driver and the simulation worker. ARCHITECTURE.md is the
binding specification; task ids are the packet slugs in `tasks/`.

1. **Kernel state (`kernel-checkpoint`).** `GpuKernel::read_checkpoint_blocking`
   copies every persistent buffer in one submit and returns a
   `KernelCheckpoint`; `restore_checkpoint` writes it into a fresh kernel.
2. **Core state (`supervisor-core-state`).** Counted random streams and
   `Supervisor::capture` / `restore` covering every ledger, archive, schedule
   and stream field.
3. **Envelope (`checkpoint-envelope`).** Pure versioned, checksummed,
   length-bounded binary codec with the `f32_bits` serde helpers and
   `CheckpointError`.
4. **Storage (`checkpoint-store`).** One overwritten `supervisor_checkpoint`
   row per run, written atomically; read-only load; resume-time rewind of
   rows at or after the checkpoint tick.
5. **Run state (`supervisor-run-resume`).** `SupervisorRun::checkpoint_due`,
   `capture_state`, `write_checkpoint` and `resume`.
6. **Headless (`headless-resume`).** `--resume` in supervisor mode resumes;
   the stop-after-checkpoint option exists for the gate.
7. **Worker (`worker-resume`).** `SupervisorWorkerInit.resume` starts the
   worker from the checkpoint; due ticks write checkpoints from the existing
   blocking readback.
8. **Gate (`resume-digest-gate`).** Stop-and-resume digest equality and
   refusal of corrupt, truncated and version-mismatched checkpoints.

## Prerequisite interfaces relied on

- From 0005 (landed): `SupervisorRun`, `Supervisor`, `TickSchedule`,
  `SupervisorStore`, `run_headless_with_options`, `HeadlessOptions`,
  `SupervisorWorkerInit`, `GpuKernel::replace_agent_slot`,
  `set_nourishing_food_kind`, `SUPERVISOR_SAMPLE_TICKS`.
- From 0006-supervisor-decision-telemetry: the supervisor-only seeded digest
  gate and its digest computation ("the plan 1 supervisor digest"), the
  telemetry fields it adds to `Supervisor`, `SupervisorRun` and
  `SupervisorStore`, its run-keyed event, series and snapshot tables (each
  with a `tick` column), and the `WeightCopy` inheritance default. Tasks read
  0006's ARCHITECTURE.md for exact names; a digest that is only defined inside
  0006's test file is copied verbatim into the gate with a comment naming its
  source.

## GPU proofs fail closed

Every check that runs a GPU-backed test sets `XAGENT_REQUIRE_GPU=1`; every new
GPU test starts with the fail-closed pattern of
`gpu_adapter_present_when_required`.

## Non-goals

- No GUI entry point for resume (plan 0008 calls `load_checkpoint_config` and
  sets `SupervisorWorkerInit.resume`).
- No change to generation-mode resume (`Governor::resume` and its
  supervisor-run refusal test stay; plan 0010 deletes them).
- No new learning rule, WGSL edit, extra dispatch, extra compute pass or
  extra readback; no new dependency or dependency version change.
- No checkpoint history: exactly one row per run.
- No resume of a run without a checkpoint, and no fallback to an older or
  partial state when a checkpoint is refused.
