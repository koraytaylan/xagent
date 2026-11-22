---
id: sense-parity-supervisor
title: "Run the sense-parity gate only against supervisor runs"
workstream: "0100"
kind: task
depends_on: []
gated: false
role: architect
touches:
  - crates/xagent-sandbox/tests/sense_block_parity.rs
  - crates/xagent-sandbox/tests/fixtures/sense_block_parity.golden
  - crates/xagent-sandbox/tests/fixtures/sense_block_champion.db
status: done
merged_as: "b1701677123ff9dbc6e4382b850203fa79dfc6be"
checks:
  - name: sense-parity-focused
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored --test-threads=1
    timeout_ms: 5400000
  - name: sense-parity-generation-free
    command: |-
      sh -c '! grep -nE "\bGovernor\b|xagent_sandbox::governor|max_generations|sense_block_champion" crates/xagent-sandbox/tests/sense_block_parity.rs && test ! -e crates/xagent-sandbox/tests/fixtures/sense_block_champion.db'
  - name: all-targets-compile
    command: cargo check --workspace --all-targets
    timeout_ms: 3600000
---

# 0104 Run the sense-parity gate only against supervisor runs

## Outcome

`crates/xagent-sandbox/tests/sense_block_parity.rs` has no generation member
left: no `Governor`, no generation headless driver, no `GovernorConfig`
generation fields (`max_generations`, `tick_budget`), and no generation database
fixture. The two GPU scenarios that touched generation state now run a seeded
supervisor headless run and replay its rank-0 archive brain. The three
scenarios that never used generation state keep their golden lines.

The baseline already has the supervisor entry points this task needs, all
public:

- `xagent_sandbox::headless::run_headless_with_options(config, db_path, false, &HeadlessOptions::default())`,
  which returns `Result<(), HeadlessError>`. It dispatches on
  `config.governor.evolution_mode`, so the config keeps
  `evolution_mode = EvolutionMode::Supervisor` (the field is deleted later by
  `config-generation-fields`, which runs after this task).
- `xagent_sandbox::supervisor_store::load_archive_entry_read_only(db_path: &str, rank: usize) -> Result<StoredArchiveEntry, ArchiveLoadError>`.
  `StoredArchiveEntry` has a `brain: AgentBrainState` field.
- The supervisor run setup is `supervisor_config()` in
  `crates/xagent-sandbox/tests/supervisor_headless.rs` (lines 67-83). It sets
  `evolution_mode`, `population_size`, the `SupervisorConfig` fields and
  `world.food_kinds_enabled` and `world.seed` from the constants at lines 10-26.

## Decisions

- Headless scenario: the seeded supervisor run uses the configuration of
  `supervisor_config()` copied verbatim into `sense_block_parity.rs`, with its
  constants copied with the same values under test-local names. The run's
  `HEADLESS_POPULATION`, `HEADLESS_TICK_BUDGET` and `HEADLESS_GENERATIONS`
  constants and every `GovernorConfig` literal are removed.
- Headless hash: `headless_hash` stops reading `node_brain` and `node`. It hashes
  the rows of `supervisor_archive` ordered by `entry_id`, taking the columns
  `entry_id, occupant_id, born_tick, retired_tick, outcome, config_json,
  birth_brain` in that order. Integers are little-endian bytes, `outcome` is
  its `f64` little-endian bytes, text is its UTF-8 bytes, a blob is its bytes,
  and a NULL blob contributes no bytes. It asserts the hashed bytes are not
  empty. Wall-clock run columns are not hashed.
- Champion replay: no database copy and no compaction. After the headless run,
  the replay calls `load_archive_entry_read_only(db_path, 0)` on the run's own
  database and writes `entry.brain` into every agent. It keeps `REPLAY_TICKS`
  (2000), the default brain configuration, and the stride assertion, which now
  reads `entry.brain.brain_state.len()`. The replay's `Connection` and
  `lowest_champion` are removed.
- Golden lines: only `headless_run` and `champion_replay` change, because they
  now hash a supervisor run instead of a generation database. The reason is
  that generation mode is removed, so a generation-database hash no longer
  describes the scenario. `steering_population`, `trajectory_raw_vision` and
  `trajectory_cortex` do not read generation state and must stay byte-identical.
- The fail-closed GPU guard (`require_gpu`, asserting the adapter exists under
  `XAGENT_REQUIRE_GPU=1`) is kept unchanged.
- The fixture is deleted by the architect (a shell `rm`). The record code path
  that rewrote it (`record_champion_fixture`) and the constants
  `CHAMPION_FIXTURE_PATH` and `CHAMPION_FIXTURE_MAX_BYTES` are removed.
- Module doc comments that describe the champion fixture or the generation
  headless run are rewritten to describe the supervisor run, with no
  `sense_block_champion` text left.

**Steps:**

1. Read `crates/xagent-sandbox/tests/sense_block_parity.rs` and the supervisor
   pieces named in the Outcome: `supervisor_config()` in
   `crates/xagent-sandbox/tests/supervisor_headless.rs`, `run_headless_with_options`
   and `HeadlessOptions` in `crates/xagent-sandbox/src/headless.rs`, and
   `StoredArchiveEntry` and `load_archive_entry_read_only` in
   `crates/xagent-sandbox/src/supervisor_store.rs`. Confirm that `ArchiveLoadError`
   derives `Debug` before using `.expect` on its result. If it does not, match
   the error and panic with its `Display` text instead. Then leave the file
   unchanged until the next step.
2. Port the headless scenario in `sense_block_parity.rs`. Replace the
   `headless::run_headless` call and the `GovernorConfig` literal with the
   supervisor configuration from the Decisions, keeping
   `evolution_mode = EvolutionMode::Supervisor`. Call
   `headless::run_headless_with_options(config, db_path, false, &HeadlessOptions::default())`
   and `.expect` its result. Rewrite `headless_hash` as the supervisor archive
   hash from the Decisions. Keep the scenario's `check_golden("headless_run", ...)`
   call. Remove the `Governor` and `GovernorConfig` imports and the
   `HEADLESS_POPULATION`, `HEADLESS_TICK_BUDGET` and `HEADLESS_GENERATIONS`
   constants.
3. Port the champion replay in the same test function. Remove
   `record_champion_fixture`, `lowest_champion`, the fixture size assertion, the
   temporary copy of the database and the `CHAMPION_FIXTURE_*` constants. Load
   the rank-0 entry with `load_archive_entry_read_only` on the run's own
   database, assert `entry.brain.brain_state.len()` equals
   `BrainLayout::from_config(&brain).brain_stride`, write `entry.brain` into
   every agent, run `REPLAY_TICKS` ticks and keep `check_golden("champion_replay", ...)`.
   Rewrite the module doc comment so it names the supervisor run and no longer
   names `sense_block_champion.db` or a generation database.
4. Delete the fixture with a shell `rm crates/xagent-sandbox/tests/fixtures/sense_block_champion.db`.
   Remove the imports that are now unused, so the test file compiles without
   warnings under `cargo check --workspace --all-targets`.
5. Re-record the two changed golden lines with the test's record mode. Run
   `XAGENT_RECORD_SENSE_PARITY=1 XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity headless_run_and_champion_replay_match_golden -- --exact --include-ignored --test-threads=1`
   (the environment variable is `XAGENT_RECORD_SENSE_PARITY`, documented in the
   test's module comment). The filter limits the rewrite to the two lines of
   that test. Afterwards `git diff` of
   `crates/xagent-sandbox/tests/fixtures/sense_block_parity.golden` must show
   changes only on the `headless_run` and `champion_replay` lines.
6. Make the test file's remaining references consistent with the port: remove
   any leftover `governor`, `generation` or `champion` naming that the port
   made unused, and do not touch the `steering_population`,
   `trajectory_raw_vision` or `trajectory_cortex` scenarios or their golden
   lines.

- **Done when:** the checks `sense-parity-focused`, `sense-parity-generation-free`
  and `all-targets-compile` pass on the host, proving that the sense-parity gate
  guards supervisor runs with no generation fixture or generation type (INV-7)
  and that the unchanged `steering_population`, `trajectory_raw_vision` and
  `trajectory_cortex` golden lines still match byte-identically (INV-6).
