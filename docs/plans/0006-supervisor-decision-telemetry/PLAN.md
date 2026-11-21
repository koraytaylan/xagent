---
objective: >-
  Make every supervisor decision and every population metric observable and
  stored in the database without changing a seeded run's decisions, switch the
  newborn inheritance default to weight-copy, and port the gates that still
  matter to the supervisor. Scope: (a) first, change the default of
  SupervisorConfig.inheritance from Darwinian to WeightCopy (Darwinian stays
  selectable) and write a short decision record next to
  docs/measurements/continuous-supervisor/DECISION.md stating that the user
  overrides the 0005 REJECT verdict and the 0005 rule that weight-copy is never
  adopted, that the supervisor becomes the only mode, and that this is the
  user's call, not a new measurement; the 0005 record stays untouched as
  history. (b) Per-slot snapshots: occupant id, age, alive or dead, meals,
  deaths, window score against the peer median, parent. (c) Boundary records:
  every eligible slot with its grounds and numbers (death, or below-median score
  with the score and the median), the slot picked and the tie-break, the parent
  and its source (archive entry or peer, birth or learned state), the newborn's
  config difference from its parent, and for idle boundaries the reason nothing
  happened. (d) Time series over ticks: food and deaths per agent-tick,
  window-score spread, lifespan of retired occupants, archive best and median
  outcome; per-occupant intake against age and live versions of the late-window
  learning measures run_summary.rs already computes. (e) Season state and next
  flip tick; tick, budget, ticks per second, ticks to next boundary. (f)
  Additive, idempotent schema migration with event and series tables keyed by
  run; everything crosses the worker boundary on the existing bounded
  latest-wins SimEvent channels. (g) A seeded headless supervisor digest gate,
  set after the weight-copy default flip, that is bit-reproducible and gives the
  same digest with recording on and off; port the sense-parity, GPU-parity and
  food-kind gates so they run against the supervisor. Owns supervisor.rs,
  supervisor_run.rs, supervisor_store.rs and only the supervisor path of
  sim_runtime.rs. Does not delete or change generation mode.
depends_on: []
status: completed
toolchains:
  - cargo
  - rustc
  - rustup
  - cargo-fmt
  - rustfmt
  - cargo-clippy
  - clippy-driver
checks:
  - name: integration-clippy
    command: cargo clippy --workspace --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: workspace-tests
    command: env XAGENT_REQUIRE_GPU=1 cargo test --workspace
    timeout_ms: 7200000
  - name: integration-supervisor-digest
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_digest
    timeout_ms: 3600000
  - name: integration-supervisor-sense-parity
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_sense_parity -- --include-ignored
    timeout_ms: 3600000
  - name: integration-supervisor-gpu-parity
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_gpu_parity
    timeout_ms: 3600000
  - name: integration-supervisor-food-kind
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_food_kind
    timeout_ms: 3600000
  - name: integration-generation-digest
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test seeded_headless_digest
    timeout_ms: 3600000
  - name: integration-generation-sense-parity
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored
    timeout_ms: 5400000
preflight_checks:
  - name: gpu-adapter-present
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
  - name: baseline-fmt
    command: cargo fmt --all -- --check
  - name: baseline-clippy
    command: cargo clippy --workspace --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: pf-inheritance-default
    command: |-
      sh -c 'grep -qF "pub enum Inheritance {" crates/xagent-shared/src/config.rs && grep -qF "fn default_inheritance() -> Inheritance {" crates/xagent-shared/src/config.rs && grep -qF "fn supervisor_config_defaults_when_omitted() {" crates/xagent-shared/src/config.rs && grep -qxF "Verdict: REJECT" docs/measurements/continuous-supervisor/DECISION.md && test ! -e docs/measurements/continuous-supervisor/INHERITANCE-DEFAULT.md'
  - name: pf-boundary-detail
    command: |-
      sh -c 'grep -qF "fn eligible_slots(" crates/xagent-sandbox/src/supervisor.rs && grep -qF "fn choose_candidate(" crates/xagent-sandbox/src/supervisor.rs && grep -qF "fn plan_parent(&mut self, slot: usize) -> ParentRef {" crates/xagent-sandbox/src/supervisor.rs && grep -qF "pub struct ReplacementOutcome {" crates/xagent-sandbox/src/supervisor.rs && grep -qF "fn median(mut values: Vec<f32>) -> Option<f32> {" crates/xagent-sandbox/src/supervisor.rs && grep -qF "pub struct SlotLedger {" crates/xagent-sandbox/src/supervisor.rs'
  - name: pf-telemetry-records
    command: |-
      sh -c 'grep -qF "pub struct KindWindowTotals {" crates/xagent-sandbox/src/run_summary.rs && grep -qF "pub const WINDOW_FRACTION_NUMERATOR: usize = 3;" crates/xagent-sandbox/src/run_summary.rs && grep -qF "pub fn begin_season(" crates/xagent-sandbox/src/run_summary.rs && grep -qF "pub mod supervisor_store;" crates/xagent-sandbox/src/lib.rs && grep -qF "pub struct ArchiveEntry {" crates/xagent-sandbox/src/supervisor.rs && test ! -e crates/xagent-sandbox/src/supervisor_telemetry.rs'
  - name: pf-telemetry-store
    command: |-
      sh -c 'grep -qF "pub fn ensure_schema(db: &Connection) -> SqlResult<()> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "CREATE TABLE IF NOT EXISTS supervisor_event (" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub fn record_ticks(&mut self, ticks: u64) -> Result<(), StoreError> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub const STORE_PRAGMAS: &str =" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "tempfile = \"3\"" crates/xagent-sandbox/Cargo.toml'
  - name: pf-run-recording
    command: |-
      sh -c 'grep -qF "pub fn on_sample(" crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "pub fn finish(mut self) -> Result<SupervisorFinish, SupervisorRunError> {" crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "fn run_supervisor(" crates/xagent-sandbox/src/headless.rs && grep -qF "fn supervisor_step(" crates/xagent-sandbox/src/headless.rs && grep -qF "pub fn run_headless_with_options(" crates/xagent-sandbox/src/headless.rs && grep -qF "pub fn read_agent_state(&self, index: u32) -> AgentBrainState {" crates/xagent-brain/src/gpu_kernel.rs && test ! -e crates/xagent-sandbox/tests/supervisor_telemetry_run.rs'
  - name: pf-worker-telemetry
    command: |-
      sh -c 'grep -qF "SupervisorSample(SampleReport)," crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn on_supervisor_sample(&mut self) {" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "const EVENT_CAPACITY: usize = 256;" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn publish_snapshot(&self, event_tx: &SyncSender<SimEvent>) {" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "SimEvent::SupervisorSample(report) => {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "pub fn cached_food_state(&self) -> Option<&[f32]> {" crates/xagent-brain/src/gpu_kernel.rs'
  - name: pf-supervisor-digest-gate
    command: |-
      sh -c 'grep -qF "struct Fnv1a(u64);" crates/xagent-sandbox/tests/seeded_headless_digest.rs && grep -qF "fn hash_section(h: &mut Fnv1a, conn: &Connection, tag: &str, sql: &str, skip: &[&str]) {" crates/xagent-sandbox/tests/seeded_headless_digest.rs && grep -qF "CREATE TABLE IF NOT EXISTS supervisor_slot (" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub fn run_headless_with_options(" crates/xagent-sandbox/src/headless.rs && test ! -e crates/xagent-sandbox/tests/supervisor_digest.rs'
  - name: pf-supervisor-sense-parity
    command: |-
      sh -c 'grep -qF "const RECORD_VARIABLE: &str = \"XAGENT_RECORD_SENSE_PARITY\";" crates/xagent-sandbox/tests/sense_block_parity.rs && grep -qF "fn check_golden(scenario: &str, hash: u64) {" crates/xagent-sandbox/tests/sense_block_parity.rs && grep -qF "fn kernel_state_bytes(kernel: &mut GpuKernel) -> Vec<u8> {" crates/xagent-sandbox/tests/sense_block_parity.rs && grep -qF "pub fn on_sample(" crates/xagent-sandbox/src/supervisor_run.rs && test ! -e crates/xagent-sandbox/tests/supervisor_sense_parity.rs'
  - name: pf-supervisor-gpu-parity
    command: |-
      sh -c 'grep -qF "pub fn dispatch_batch(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn dispatch_ticks(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "fn mutation_and_crossover_keep_modulator_flags() {" crates/xagent-sandbox/tests/modulator_flags.rs && grep -qF "pub fn on_sample(" crates/xagent-sandbox/src/supervisor_run.rs && test ! -e crates/xagent-sandbox/tests/supervisor_gpu_parity.rs'
  - name: pf-supervisor-food-kind
    command: |-
      sh -c 'grep -qF "fn food_kind_draw_off_by_default() {" crates/xagent-brain/tests/food_kind_draw.rs && grep -qF "fn flip_season(" crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "pub const SEASON_STREAM: u64 = 0x5EA5_0000_0000_0002;" crates/xagent-sandbox/src/supervisor.rs && test ! -e crates/xagent-sandbox/tests/supervisor_food_kind.rs'
  - name: pf-integration-guard
    command: |-
      sh -c 'test -f crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF "cargo test -p xagent-sandbox" CONTRIBUTING.md && grep -qF "cargo clippy --workspace --all-targets -- -D warnings" CONTRIBUTING.md'
execution_checks:
  - name: developer-route
    tasks:
      - inheritance-default
      - boundary-detail
      - telemetry-records
      - telemetry-store
      - run-recording
      - worker-telemetry
      - supervisor-gpu-parity
      - supervisor-food-kind-gate
    role: developer
    operations:
      - edit
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
    constraints:
      - task: inheritance-default
        name: inheritance-enum-and-0005-record
        sources:
          - crates/xagent-shared/src/config.rs
          - docs/measurements/continuous-supervisor/DECISION.md
        checks:
          - name: pf-inheritance-default
            command: |-
              sh -c 'grep -qF "pub enum Inheritance {" crates/xagent-shared/src/config.rs && grep -qF "fn default_inheritance() -> Inheritance {" crates/xagent-shared/src/config.rs && grep -qF "fn supervisor_config_defaults_when_omitted() {" crates/xagent-shared/src/config.rs && grep -qxF "Verdict: REJECT" docs/measurements/continuous-supervisor/DECISION.md && test ! -e docs/measurements/continuous-supervisor/INHERITANCE-DEFAULT.md'
      - task: boundary-detail
        name: eligibility-tiebreak-and-parent-draw
        sources:
          - crates/xagent-sandbox/src/supervisor.rs
        checks:
          - name: pf-boundary-detail
            command: |-
              sh -c 'grep -qF "fn eligible_slots(" crates/xagent-sandbox/src/supervisor.rs && grep -qF "fn choose_candidate(" crates/xagent-sandbox/src/supervisor.rs && grep -qF "fn plan_parent(&mut self, slot: usize) -> ParentRef {" crates/xagent-sandbox/src/supervisor.rs && grep -qF "pub struct ReplacementOutcome {" crates/xagent-sandbox/src/supervisor.rs && grep -qF "fn median(mut values: Vec<f32>) -> Option<f32> {" crates/xagent-sandbox/src/supervisor.rs && grep -qF "pub struct SlotLedger {" crates/xagent-sandbox/src/supervisor.rs'
      - task: telemetry-records
        name: season-window-totals-and-module-slot
        sources:
          - crates/xagent-sandbox/src/run_summary.rs
          - crates/xagent-sandbox/src/lib.rs
          - crates/xagent-sandbox/src/supervisor.rs
        checks:
          - name: pf-telemetry-records
            command: |-
              sh -c 'grep -qF "pub struct KindWindowTotals {" crates/xagent-sandbox/src/run_summary.rs && grep -qF "pub const WINDOW_FRACTION_NUMERATOR: usize = 3;" crates/xagent-sandbox/src/run_summary.rs && grep -qF "pub fn begin_season(" crates/xagent-sandbox/src/run_summary.rs && grep -qF "pub mod supervisor_store;" crates/xagent-sandbox/src/lib.rs && grep -qF "pub struct ArchiveEntry {" crates/xagent-sandbox/src/supervisor.rs && test ! -e crates/xagent-sandbox/src/supervisor_telemetry.rs'
      - task: telemetry-store
        name: idempotent-schema-and-sqlite
        sources:
          - crates/xagent-sandbox/src/supervisor_store.rs
          - crates/xagent-sandbox/Cargo.toml
        checks:
          - name: pf-telemetry-store
            command: |-
              sh -c 'grep -qF "pub fn ensure_schema(db: &Connection) -> SqlResult<()> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "CREATE TABLE IF NOT EXISTS supervisor_event (" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub fn record_ticks(&mut self, ticks: u64) -> Result<(), StoreError> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub const STORE_PRAGMAS: &str =" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "tempfile = \"3\"" crates/xagent-sandbox/Cargo.toml'
      - task: run-recording
        name: driver-sample-hook-and-headless-loop
        sources:
          - crates/xagent-sandbox/src/supervisor_run.rs
          - crates/xagent-sandbox/src/headless.rs
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-run-recording
            command: |-
              sh -c 'grep -qF "pub fn on_sample(" crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "pub fn finish(mut self) -> Result<SupervisorFinish, SupervisorRunError> {" crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "fn run_supervisor(" crates/xagent-sandbox/src/headless.rs && grep -qF "fn supervisor_step(" crates/xagent-sandbox/src/headless.rs && grep -qF "pub fn run_headless_with_options(" crates/xagent-sandbox/src/headless.rs && grep -qF "pub fn read_agent_state(&self, index: u32) -> AgentBrainState {" crates/xagent-brain/src/gpu_kernel.rs && test ! -e crates/xagent-sandbox/tests/supervisor_telemetry_run.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: worker-telemetry
        name: worker-event-channel-and-consumer-match
        sources:
          - crates/xagent-sandbox/src/sim_runtime.rs
          - crates/xagent-sandbox/src/gpu_orchestration.rs
          - crates/xagent-brain/src/gpu_kernel.rs
        checks:
          - name: pf-worker-telemetry
            command: |-
              sh -c 'grep -qF "SupervisorSample(SampleReport)," crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn on_supervisor_sample(&mut self) {" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "const EVENT_CAPACITY: usize = 256;" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn publish_snapshot(&self, event_tx: &SyncSender<SimEvent>) {" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "SimEvent::SupervisorSample(report) => {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "pub fn cached_food_state(&self) -> Option<&[f32]> {" crates/xagent-brain/src/gpu_kernel.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: supervisor-gpu-parity
        name: fused-and-split-dispatch-apis
        sources:
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-sandbox/tests/modulator_flags.rs
          - crates/xagent-sandbox/src/supervisor_run.rs
        checks:
          - name: pf-supervisor-gpu-parity
            command: |-
              sh -c 'grep -qF "pub fn dispatch_batch(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn dispatch_ticks(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "fn mutation_and_crossover_keep_modulator_flags() {" crates/xagent-sandbox/tests/modulator_flags.rs && grep -qF "pub fn on_sample(" crates/xagent-sandbox/src/supervisor_run.rs && test ! -e crates/xagent-sandbox/tests/supervisor_gpu_parity.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: supervisor-food-kind-gate
        name: season-stream-and-kernel-kind-draw
        sources:
          - crates/xagent-brain/tests/food_kind_draw.rs
          - crates/xagent-sandbox/src/supervisor_run.rs
          - crates/xagent-sandbox/src/supervisor.rs
        checks:
          - name: pf-supervisor-food-kind
            command: |-
              sh -c 'grep -qF "fn food_kind_draw_off_by_default() {" crates/xagent-brain/tests/food_kind_draw.rs && grep -qF "fn flip_season(" crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "pub const SEASON_STREAM: u64 = 0x5EA5_0000_0000_0002;" crates/xagent-sandbox/src/supervisor.rs && test ! -e crates/xagent-sandbox/tests/supervisor_food_kind.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
  - name: golden-capture-route
    tasks:
      - supervisor-digest-gate
      - supervisor-sense-parity
    role: architect
    operations:
      - edit
      - shell
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
    actions:
      - name: build-sandbox-tests
        command: cargo build -p xagent-sandbox --tests
        timeout_ms: 3600000
    constraints:
      - task: supervisor-digest-gate
        name: digest-hasher-and-supervisor-tables
        sources:
          - crates/xagent-sandbox/tests/seeded_headless_digest.rs
          - crates/xagent-sandbox/src/supervisor_store.rs
          - crates/xagent-sandbox/src/headless.rs
        checks:
          - name: pf-supervisor-digest-gate
            command: |-
              sh -c 'grep -qF "struct Fnv1a(u64);" crates/xagent-sandbox/tests/seeded_headless_digest.rs && grep -qF "fn hash_section(h: &mut Fnv1a, conn: &Connection, tag: &str, sql: &str, skip: &[&str]) {" crates/xagent-sandbox/tests/seeded_headless_digest.rs && grep -qF "CREATE TABLE IF NOT EXISTS supervisor_slot (" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub fn run_headless_with_options(" crates/xagent-sandbox/src/headless.rs && test ! -e crates/xagent-sandbox/tests/supervisor_digest.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: supervisor-sense-parity
        name: golden-file-pattern-and-kernel-state
        sources:
          - crates/xagent-sandbox/tests/sense_block_parity.rs
          - crates/xagent-sandbox/src/supervisor_run.rs
        checks:
          - name: pf-supervisor-sense-parity
            command: |-
              sh -c 'grep -qF "const RECORD_VARIABLE: &str = \"XAGENT_RECORD_SENSE_PARITY\";" crates/xagent-sandbox/tests/sense_block_parity.rs && grep -qF "fn check_golden(scenario: &str, hash: u64) {" crates/xagent-sandbox/tests/sense_block_parity.rs && grep -qF "fn kernel_state_bytes(kernel: &mut GpuKernel) -> Vec<u8> {" crates/xagent-sandbox/tests/sense_block_parity.rs && grep -qF "pub fn on_sample(" crates/xagent-sandbox/src/supervisor_run.rs && test ! -e crates/xagent-sandbox/tests/supervisor_sense_parity.rs'
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
              sh -c 'test -f crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF "cargo test -p xagent-sandbox" CONTRIBUTING.md && grep -qF "cargo clippy --workspace --all-targets -- -D warnings" CONTRIBUTING.md'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
---

# 0006 Supervisor decision telemetry

## Scope

ARCHITECTURE.md is the binding specification; task ids are the packet slugs
in `tasks/`.

1. **Inheritance default (`inheritance-default`), first.** `Inheritance`'s
   serde and `Default` default becomes `WeightCopy`; `Darwinian` stays a
   selectable variant (config JSON `"darwinian"`, CLI
   `--supervisor-inheritance darwinian`). A short record
   `docs/measurements/continuous-supervisor/INHERITANCE-DEFAULT.md` states that
   the user overrides the 0005 `REJECT` verdict and the 0005 rule that
   weight-copy is never adopted, that the supervisor becomes the only mode, and
   that this is the user's decision, not a new measurement. `DECISION.md` and
   `results.json` stay byte-for-byte untouched.
2. **Boundary detail (`boundary-detail`).** The pure core exposes, without any
   extra random draw, each slot's standing (age, alive, meals, deaths, window
   score, peer median, eligibility ground), the full detail of the last
   boundary (every eligible slot with grounds and numbers, the tie-break pool
   and drawn index, the parent draw, or the idle reason), the next boundary
   tick, each occupant's parent, and the parent's config on the replacement
   outcome.
3. **Telemetry records (`telemetry-records`).** A new pure module
   `crates/xagent-sandbox/src/supervisor_telemetry.rs` builds per-slot
   snapshots, boundary records (with the newborn's config difference from its
   parent), time-series points, per-occupant intake-against-age rows, live
   per-season late-window measures and progress records from data the driver
   already has. `run_summary.rs` gains one read-only getter.
4. **Telemetry store (`telemetry-store`).** Additive, idempotent tables keyed by
   `run_id` and writers in `supervisor_store.rs`; existing 0005 tables and
   databases keep loading unchanged.
5. **Recording in the driver (`run-recording`).** `SupervisorRun` builds the
   records at each sample and writes them when recording is on (default on with
   a store); the headless supervisor loop supplies ticks per second and can turn
   recording off. Recording never changes a decision.
6. **Worker boundary (`worker-telemetry`).** The simulation worker publishes a
   telemetry frame per sample on the existing bounded event channel
   (latest-wins `try_send`, dropped under back-pressure) and boundary records as
   ordered control events; the main thread ignores the new variant until
   0008 consumes it.
7. **Gates (`supervisor-digest-gate`, `supervisor-sense-parity`,
   `supervisor-gpu-parity`, `supervisor-food-kind-gate`).** A seeded headless
   supervisor digest gate captured after the default flip, bit-reproducible and
   equal with recording on and off; supervisor ports of the sense-parity golden,
   the GPU dispatch parity and the food-kind default/season checks.

## Non-goals

- No deletion or behaviour change of generation mode: `EvolutionMode` keeps
  `Generations` as its default here, and the generation digest, the
  `sense_block_parity` golden and `food_kind_draw` stay green unchanged.
  Removing generation mode is 0010.
- No change to which slot is replaced, which parent is drawn, how the newborn
  is mutated, any random stream, the season schedule or the kernel; no WGSL
  edit; no new GPU dispatch or readback.
- No GUI view (0008), no checkpoint/resume (0007; supervisor resume stays
  refused), no tick-window replay (0009).
- No new learning rule, reward term or selection measure.
- `DECISION.md` and `results.json` under
  `docs/measurements/continuous-supervisor/` are not edited.

## Prerequisite interfaces relied on (baseline)

- `crates/xagent-sandbox/src/supervisor.rs`: `Supervisor`, `SlotLedger`,
  `ReplacementPlan`, `ReplacementOutcome`, `ParentRef`, `ReplaceReason`,
  `Archive`, `TickSchedule`, `SEASON_STREAM`.
- `crates/xagent-sandbox/src/supervisor_run.rs`: `SupervisorRun::{start,
  on_sample, finish}`, `SampleReport`, `BoundaryReport` (both struct shapes stay
  unchanged: `ui.rs` tests build them by literal).
- `crates/xagent-sandbox/src/supervisor_store.rs`: `ensure_schema`,
  `SupervisorStore`.
- `crates/xagent-sandbox/src/sim_runtime.rs`: `SimEvent`, bounded
  `sync_channel` of `EVENT_CAPACITY`, `partition_events`.
- `crates/xagent-sandbox/tests/seeded_headless_digest.rs` (hash pattern) and
  `crates/xagent-sandbox/tests/sense_block_parity.rs` (golden-line pattern).

## GPU proofs fail closed

Every GPU-backed check sets `XAGENT_REQUIRE_GPU=1`; every new GPU test returns
early only when no adapter exists and the variable is unset, and panics when it
is set (the `gpu_guard_decision` pattern of `tests/supervisor_run.rs`).
