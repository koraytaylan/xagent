---
id: supervisor-sense-parity
title: "Supervisor sense-parity golden"
workstream: "0600"
kind: task
role: architect
depends_on:
  - inheritance-default
gated: false
status: todo
merged_as: ""
touches:
  - crates/xagent-sandbox/tests/supervisor_sense_parity.rs
  - crates/xagent-sandbox/tests/fixtures/supervisor_sense_parity.golden
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_sense_parity -- --include-ignored"
    timeout_ms: 3600000
  - name: fmt
    command: "cargo fmt --all -- --check"
    fix: "cargo fmt --all"
  - name: clippy
    command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
---

# Supervisor sense-parity golden

## Outcome

`crates/xagent-sandbox/tests/supervisor_sense_parity.rs` is the supervisor port
of the GPU trajectory parts of `crates/xagent-sandbox/tests/sense_block_parity.rs`.
Two seeded supervisor runs, one with the raw-vision brain and one with the
visual-cortex brain, each hash their physics, birth brains and boundary report
fields to one line of `crates/xagent-sandbox/tests/fixtures/supervisor_sense_parity.golden`.
Architecture INV-11 is proven by the focused-proof check, which compares both
lines and a reproducibility test that runs one scenario twice.

## Decisions

- Helpers are copied, not shared, from
  `crates/xagent-sandbox/tests/sense_block_parity.rs`: `fnv1a` (the
  `fnv1a(mut hash, bytes)` form), `hash_bytes`, `f32_bytes`, `crate_path`,
  `record_mode`, `require_gpu` and `check_golden`. The constants
  `RECORD_VARIABLE = "XAGENT_RECORD_SUPERVISOR_SENSE_PARITY"` and
  `GOLDEN_PATH = "tests/fixtures/supervisor_sense_parity.golden"` are new in the
  new file. No shared helper crate and no new dependency.
- `check_golden(scenario, hash)` behaves as in `sense_block_parity.rs` except
  for the absent-file case, which is specified here:
  - Record mode (`XAGENT_RECORD_SUPERVISOR_SENSE_PARITY=1`): creates the golden
    file when it is absent, then replaces the line whose first word is
    `scenario` or appends one if none exists. Every other line is kept in
    order. It never fails on a mismatch.
  - Not record mode: panics when the file is absent, when no line starts with
    `scenario `, or when the hex value differs. The panic message names the
    scenario and both values.
  - `require_gpu` is fail-closed: it returns early only when no adapter exists
    and `XAGENT_REQUIRE_GPU` is unset, and panics when it is set.
- Scenario names written to the golden file: `supervisor_raw_vision` and
  `supervisor_cortex`. Each line is `<name> <16 lowercase hex digits>`.
- Brain configs follow `raw_vision_trajectory_matches_golden` and
  `cortex_trajectory_matches_golden`: `BrainConfig { visual_cortex_enabled: false, ..Default::default() }`
  for raw vision and `visual_cortex_enabled: true` for cortex.
- Seeded configuration: `FullConfig::default()` with `world.seed = 5`, the
  default world food-kind setting (food kinds off, asserted before the run),
  `governor.evolution_mode = EvolutionMode::Supervisor`,
  `governor.population_size = 10`, the brain config above on the seeded
  population, and small supervisor settings as in `config()` of
  `crates/xagent-sandbox/tests/supervisor_run.rs`: boundary interval 1,000
  ticks, min age 2,000 ticks, score window 1,000 ticks, season period 3,000
  ticks, archive capacity as that helper sets it.
- `agent_tick_budget = 30_000` (10 agents x 3,000 ticks). A scenario must
  produce at least one replacement; the test asserts it. If the first record
  run has none, the architect raises `agent_tick_budget` in steps of 10,000
  agent-ticks until one replacement occurs, keeps the value in the test source,
  and records the final value in the step report. This is the only permitted
  tuning.
- Driver: one `GpuKernel` per scenario and `SupervisorRun::start` as
  `tests/supervisor_run.rs` `drive()` and `advance()` do, with
  `dispatch_batch(tick, 100)` per sample and `store` set to `None`.
- Hash per scenario: one byte buffer, built in this order, then `hash_bytes`:
  1. At every sample, `kernel.read_full_state_blocking()` truncated per agent to
     the first `PRE_SCENARIO_PHYSICS_SLOTS` values (`P_HARMFUL_MEAL_COUNT`), as
     `f32_bytes`, after asserting the buffer length as `kernel_state_bytes`
     does.
  2. For every replacement, in order of occurrence, the newborn's
     `birth_state(slot)?.brain_state` bytes as `f32_bytes`.
  3. For every `SampleReport` boundary, the boundary `slot` and occupant id
     fields as little-endian `u64` values, in sample order.
  The harmful-meal slot is not asserted in this test.
- Test `supervisor_raw_vision_matches_golden` and
  `supervisor_cortex_matches_golden`, both `#[ignore = "requires a GPU; run explicitly with --include-ignored"]`:
  call `require_gpu()`, run the scenario, assert at least one replacement, then
  `check_golden(scenario, hash)`.
- Test `supervisor_sense_parity_is_reproducible` (ignored, GPU): runs the raw
  vision scenario twice and asserts the two hashes are equal, without touching
  the golden file.
- Non-ignored pure test `golden_file_parses`: reads `GOLDEN_PATH`, asserts every
  non-empty line is `name hex16` with 16 lowercase hex digits, and asserts both
  scenario names appear. It fails if the file is absent.
- The source carries no plan numbers, task slugs or workstream words; comments
  describe the supervisor sense-parity golden only.

## Non-goals

- No change to `crates/xagent-sandbox/tests/sense_block_parity.rs` or
  `crates/xagent-sandbox/tests/fixtures/sense_block_parity.golden`.
- No product code: nothing under `crates/xagent-sandbox/src/`, no WGSL, no new
  GPU dispatch or readback beyond what `supervisor_run.rs` already performs.
- No change to `Cargo.toml`, the digest gate, the GPU-parity or food-kind tests.
- No hand-written golden hash: the golden lines come only from the record-mode
  run in step 2.

**Steps:**

1. Write `crates/xagent-sandbox/tests/supervisor_sense_parity.rs` as described in
   Decisions: the copied helpers, `check_golden` with the absent-file record
   behaviour, the seeded supervisor configuration, the hash over the three
   parts, the two golden scenarios, the reproducibility test and the pure
   `golden_file_parses` test. Keep `GOLDEN_PATH` as the only fixture path.
2. Architect shell: from the repository root, run
   `env XAGENT_REQUIRE_GPU=1 XAGENT_RECORD_SUPERVISOR_SENSE_PARITY=1 cargo test -p xagent-sandbox --test supervisor_sense_parity -- --include-ignored`.
   Confirm it reports no failed assertion, including the at-least-one-replacement
   assertion. If a scenario has no replacement, raise `agent_tick_budget` as in
   Decisions and rerun the record command. Then confirm
   `crates/xagent-sandbox/tests/fixtures/supervisor_sense_parity.golden` holds
   exactly the two lines `supervisor_raw_vision <hex16>` and
   `supervisor_cortex <hex16>`, with the hex values copied from the test
   output, not typed by hand.
3. Run the focused-proof check without record mode. Every ignored test and
   `golden_file_parses` must pass, including
   `supervisor_sense_parity_is_reproducible`.
4. Run the fmt and clippy checks and fix any reported formatting or lint in the
   new test file only, keeping the test bodies and the golden file unchanged.
5. Run the guard check and confirm it passes with no change to `CONTRIBUTING.md`.

- **Done when:** the focused-proof check (`supervisor_sense_parity` with `XAGENT_REQUIRE_GPU=1` and `--include-ignored`) passes with tests `supervisor_raw_vision_matches_golden`, `supervisor_cortex_matches_golden`, `supervisor_sense_parity_is_reproducible` and `golden_file_parses` green, each seeded supervisor scenario hash equals its line in `crates/xagent-sandbox/tests/fixtures/supervisor_sense_parity.golden` and one scenario run twice gives equal hashes, proving INV-11.
