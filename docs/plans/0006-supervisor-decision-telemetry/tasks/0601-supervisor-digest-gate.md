---
id: supervisor-digest-gate
title: "Seeded headless supervisor digest gate"
workstream: "0600"
kind: task
role: architect
depends_on:
  - inheritance-default
  - run-recording
gated: false
status: done
merged_as: "506305c6180113870b2bfbb09ab4b942f9ba2fe1"
touches:
  - crates/xagent-sandbox/tests/supervisor_digest.rs
  - crates/xagent-sandbox/tests/fixtures/supervisor_digest.json
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_digest"
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

# Seeded headless supervisor digest gate

## Outcome

`crates/xagent-sandbox/tests/supervisor_digest.rs` is a bit-reproducibility gate
for the supervisor path after the newborn inheritance default became
weight-copy. A seeded headless supervisor run is reduced to an FNV-1a digest of
its stored supervisor tables, and that digest is equal across two recorded runs
and one unrecorded run. The fixture
`crates/xagent-sandbox/tests/fixtures/supervisor_digest.json` holds the digest
captured on the architect's GPU together with the adapter description it was
captured on.

## Decisions

- Helpers are copied, not shared: `Fnv1a` (`new`, `write`, `write_len`,
  `write_value` with the same tags 0 to 4), `fnv1a`, `hash_section`,
  `env_flag`, `adapter_description` and `is_digest_hex` follow
  `crates/xagent-sandbox/tests/seeded_headless_digest.rs` exactly. No shared
  helper crate and no new dependency; `tempfile`, `rusqlite`, `serde_json`,
  `wgpu` and `pollster` are already dev-dependencies or dependencies of
  `xagent-sandbox`.
- Seeded configuration: `FullConfig::default()` with `world.seed = 5`,
  `governor.evolution_mode = EvolutionMode::Supervisor`,
  `governor.population_size = 10` (set explicitly), and
  `governor.supervisor.agent_tick_budget = 600_000` (10 agents x 60,000 ticks;
  600,000 is a multiple of 10 x `SUPERVISOR_SAMPLE_TICKS` = 1,000, so
  `SupervisorConfig::validate` accepts it). Every other supervisor setting stays
  at its default: boundary interval 4,000 ticks (a multiple of 100 and at least
  200), min age 20,000 ticks (at least the 10,000-tick score window), season
  period 40,000 ticks, archive capacity 32. Agents can be replaced after their
  20,000-tick minimum age, so a 60,000-tick life yields several boundaries
  where replacements occur.
- Each run goes to a fresh `tempfile::TempDir` through
  `xagent_sandbox::headless::run_supervisor_with_recording(config, path,
  &HeadlessOptions::default(), record)` and `.expect`s its success.
- Digest sections, in this order, each opened read-only and hashed through
  `hash_section` with the tags `supervisor_slot`, `supervisor_archive` and
  `supervisor_event`:
  - `SELECT * FROM supervisor_slot ORDER BY slot`, skipping `run_id`.
  - `SELECT * FROM supervisor_archive ORDER BY entry_id`, skipping `id` and
    `run_id`.
  - `SELECT * FROM supervisor_event ORDER BY id`, skipping `id` and `run_id`.
  The `supervisor_*_snapshot`, `supervisor_boundary*`, `supervisor_series`,
  `supervisor_occupant_intake` and `supervisor_progress` tables are not hashed.
- Before hashing, the digest function asserts the archive holds at least two
  rows, so a run that produced no replacements fails rather than hashing an
  empty history.
- Test `supervisor_digest_is_reproducible_and_recording_independent`, in this
  order:
  1. Asserts `seeded_config().governor.supervisor.inheritance` matches
     `Inheritance::WeightCopy` (the default flip is in force).
  2. Fail-closed GPU guard: when `GpuKernel::is_available()` is false, it
     panics if `XAGENT_REQUIRE_GPU=1` and otherwise prints a skip line and
     returns.
  3. Runs three times: `record = true`, `record = true`, `record = false`. It
     asserts the first two digests are equal and the first equals the third,
     and that the digest is 16 lowercase hex digits. It prints `digest: <hex>`.
  4. If `XAGENT_CAPTURE_SUPERVISOR_DIGEST=1`, it prints `adapter: <desc>` from
     `adapter_description()` and returns without reading the fixture.
  5. Otherwise it reads the fixture JSON with `serde_json` and asserts
     `seed == 5`, `agent_tick_budget == 600000`, `inheritance == "weight_copy"`,
     that `digest` is 16 hex digits and that `adapter` is a string. If
     `adapter_description()` equals the fixture `adapter`, or
     `XAGENT_REQUIRE_SEEDED_DIGEST=1`, it asserts the computed digest equals the
     fixture `digest`. Otherwise it prints that the adapter differs and skips
     only that comparison.
- Fixture file format: a JSON object with exactly the keys `seed` (5),
  `agent_tick_budget` (600000), `inheritance` (`"weight_copy"`), `digest` (the
  captured 16-hex string) and `adapter` (the captured `name (backend)` string).
- The source and the fixture carry no plan numbers, task slugs or workstream
  words; comments describe the supervisor digest only.
- The digest test does not depend on any file other than the fixture and the
  test file; it uses no golden, no Python and no network.

## Non-goals

- No product code: no change under `crates/xagent-sandbox/src/`, no change to
  `supervisor_store.rs`, `headless.rs`, `supervisor_run.rs` or the config
  crate. The test uses the interfaces the prerequisites provide.
- No change to `seeded_headless_digest.rs` or its fixture; the generation digest
  stays as it is.
- No change to `DECISION.md`, `results.json` or the measurement records.
- No sense-parity, GPU-parity or food-kind checks (other tasks).
- No raw or reproduced-by-hand digest: the fixture value comes only from the
  capture command's printed output.

**Steps:**

1. Write `crates/xagent-sandbox/tests/supervisor_digest.rs` as described in
   Decisions: the copied helpers, the seeded configuration, the three-run
   digest with the archive-row assertion, and the test body in the order given.
   Use only `tests/fixtures/supervisor_digest.json` as the fixture path.
2. Architect shell: run
   `env XAGENT_REQUIRE_GPU=1 XAGENT_CAPTURE_SUPERVISOR_DIGEST=1 cargo test -p xagent-sandbox --test supervisor_digest -- --nocapture`
   from the repository root, and confirm the output contains one `digest:` line,
   one `adapter:` line and no failed assertion (including the archive-row
   count). If the capture shows fewer than two archive rows, the architect
   raises `governor.supervisor.agent_tick_budget` (in the seeded config and in
   the test's fixture-budget assertion together) to the next multiple of
   10 x `SUPERVISOR_SAMPLE_TICKS` (1,000) that yields at least two rows, keeping
   `SupervisorConfig::validate` aligned, and re-runs the capture before recording
   the fixture. Then create `crates/xagent-sandbox/tests/fixtures/supervisor_digest.json`
   with `seed` 5, the final `agent_tick_budget`, `inheritance` `"weight_copy"`,
   and the printed `digest` and `adapter` values copied verbatim.
3. Run the focused-proof check. Its three-run test must pass with the fixture
   comparison either matched or skipped with the printed adapter-differs line.
4. Run the fmt and clippy checks and fix any reported formatting or lint in the
   test file only, keeping the test body unchanged.
5. Run the guard check, the contributing guard in
   `crates/xagent-sandbox/tests/contributing_guard.rs`, and confirm it passes
   with no change to `CONTRIBUTING.md`.

- **Done when:** the focused-proof check (`env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_digest`) passes with the test `supervisor_digest_is_reproducible_and_recording_independent` green, showing two recorded seeded supervisor runs and one unrecorded run produce the same digest over `supervisor_slot`, `supervisor_archive` and `supervisor_event`, and the fixture digest is asserted equal whenever the adapter matches the recorded adapter, proving INV-10.
