---
id: parity-golden
title: "Record the pre-refactor parity golden"
workstream: "0100"
kind: task
role: architect
depends_on: []
gated: false
status: todo
merged_as: ""
touches:
  - crates/xagent-sandbox/tests/sense_block_parity.rs
  - crates/xagent-sandbox/tests/fixtures/sense_block_parity.golden
  - crates/xagent-sandbox/tests/fixtures/sense_block_champion.db
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored"
    timeout_ms: 5400000
  - name: fmt
    command: "cargo fmt --all -- --check"
    fix: "cargo fmt --all"
  - name: clippy
    command: "cargo clippy --workspace --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
    timeout_ms: 3600000
---

# 0101 — Record the pre-refactor parity golden

## Scope and non-goals

- Add `crates/xagent-sandbox/tests/sense_block_parity.rs`, the integration test that hashes seeded behaviour and compares it with a committed golden.
- Record, on the tree exactly as plans 0001 and 0002 left it, the golden `crates/xagent-sandbox/tests/fixtures/sense_block_parity.golden` and the champion fixture `crates/xagent-sandbox/tests/fixtures/sense_block_champion.db`.
- First task of plan 0003: it completes before any other task changes code, since every later task runs the same focused-proof command as its parity check (INV-01).
- No product code change: no edits under `crates/xagent-brain/src` or `crates/xagent-sandbox/src`, no config default change, no new flag. No edit to any other test file, the free-life harness or the governor schema.
- No tolerance in the comparison. A scenario that does not reproduce is fixed by reducing harness nondeterminism, never by loosening the compare.
- No re-recording after commit. Only modulator-study (0402) may re-record, and only when a landed modulator changes default behaviour.
- The golden is host-specific. Only the host runs the ignored GPU scenarios; CI runs only the CPU scenario `steering_population`. The module doc states this.

## Test design (fixed)

- **Module doc (`//!`):** purpose (byte-level regression baseline for the sense/motor block), scenario list, record and compare modes, host-specific GPU hashes. Technical terms only: no plan, task, workstream or decision-record names anywhere in the file, including assertion and panic strings.
- **Hash:** FNV-1a 64-bit (offset basis `0xcbf29ce484222325`, prime `0x100000001b3`) over little-endian bytes, printed as 16 lowercase hex digits. Give the constants doc comments.
- **Golden format:** one line per scenario, `<scenario-name> <16 lowercase hex digits>`. Compare mode names the scenario on a mismatch, and fails if a scenario line is missing or the golden file is absent.
- **Modes:** compare by default. With `XAGENT_RECORD_SENSE_PARITY=1` the test rewrites only its own scenario line (read-modify-write; create the file if absent) and does not compare. Record runs use `--test-threads=1`.
- **Scenarios** (names fixed):
  - `steering_population`: CPU, normal `#[test]` (not ignored). `fresh_brain_state` and `steering_population` with a fixed `SmallRng` seed; hash every returned `brain_state`. Model on `seeded_steering_population_is_reproducible` in `docs/plans/0003-uniform-sense-motor-route/execution/parity_reproducibility_probe.rs`.
  - `trajectory_raw_vision` and `trajectory_cortex`: `#[ignore]` GPU. 10 agents, world seed 5, brain seed 12345, 3000 ticks as 100-tick `dispatch_ticks` calls, each followed by `poll_wait()`; `visual_cortex_enabled` false for the first, true for the second. Agent 0 gets energy and max energy `1.0` in the `upload_agents` tuple so a death occurs. Assert agent 0's death counter (`P_DEATH_COUNT`; per-agent indexing: grep `crates/xagent-brain/src/buffers.rs` and the existing kernel tests) is greater than zero. Fail, never skip, if the counter cannot be read. Hash the full physics state (`read_full_state_blocking`) plus every agent's `brain_state` and `patterns`.
  - Route scenarios: decided in step 1. Each is named `trajectory_<route>_route`, runs with that route's flag on, and is otherwise identical to the default trajectory.
  - `headless_run` and `champion_replay`: ONE `#[ignore]` GPU test. Run `headless::run_headless(FullConfig { brain: BrainConfig::default(), world: WorldConfig { seed: 5, .. }, governor: GovernorConfig { population_size: 4, tick_budget: 2000, max_generations: 2, .. } }, <tempdir db path>, false, true)`. Hash `node_brain` rows ordered by `node_id` (id as LE i64, then `data`) and `node` rows ordered by `id` (id, `best_fitness`, `avg_fitness`, with `None` mapped to NaN bits as the probe does). That hash is `headless_run`.
    - Record mode only: copy this run's database to the champion fixture after deleting all `generation_recording` rows and running `VACUUM`. The fixture must be under 2 MB; if not, lower `tick_budget` here and record again.
    - Both modes: copy the committed fixture into a fresh tempdir, open it with `Governor::resume`, find the lowest `node.id` for which `champion_brain(node_id)` is `Some`, and assert its `brain_state.len()` equals `BrainLayout::from_config(&BrainConfig::default()).brain_stride`. Write that state to every agent of a fresh 10-agent kernel with `write_agent_state` (after `upload_world` and `upload_agents`, as in the trajectory scenario), run 2000 ticks chunked, and hash the full state as the trajectory scenarios do. That hash is `champion_replay`.
- **GPU tests** are `#[ignore = "requires a GPU; run explicitly with --include-ignored"]`, begin by asserting `GpuKernel::is_available()` with a message naming the requirement, and never skip.
- Test names describe what they verify (for example `headless_run_and_champion_replay_match_golden`); each doc comment matches what its test asserts.
- No `TODO`, no commented-out code. Named `const`s with doc comments for every numeric literal other than 0, 1, or the hash constants' own definitions. No lossy `as` casts; use `try_into()` for counts.
- APIs (existing): `GpuKernel::new`, `reset_agents_seeded`, `upload_world`, `upload_agents`, `dispatch_ticks`, `poll_wait`, `read_full_state_blocking`, `write_agent_state`, `WorldState::new`, `headless::run_headless`, `Governor::resume`, `Governor::champion_brain`, `agent::fresh_brain_state`, `agent::steering_population`, `BrainLayout::from_config`. Dev-dependency `tempfile` and normal `rand` and `rusqlite` are already declared for `xagent-sandbox`.

**Steps:**

1. Read the plan 0002 decision records (`docs/decisions/pathway-critic-route.md`, `docs/decisions/pathway-forward-route.md`, `docs/decisions/pathway-critic-inputs.md`, each with a `.json` sibling) and the route flags `critic_input` and `forward_pathway_enabled` in `BrainConfig` (`crates/xagent-shared/src/config.rs`) as they stand after plan 0002 merges. A route is adopted when its `DEFAULT_*` constant in that file is flipped; adopted routes are default-on and already covered by the default trajectory scenarios. For each route whose verdict is passed and whose flag defaults off, add one `trajectory_<route>_route` scenario that sets the flag on. If every passed route is default-on, add none. If a record is absent after the dependency merges, report a blocker; do not silently add none. Record the chosen scenarios in the module doc in technical terms.
2. Write `crates/xagent-sandbox/tests/sense_block_parity.rs` as designed above.
3. Record on the unmodified tree (shell, architect only): `env XAGENT_RECORD_SENSE_PARITY=1 XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored --test-threads=1`. This writes the golden lines and the DB fixture. Then run the focused-proof command twice in compare mode (shell, architect only); both runs must pass. If any scenario differs between runs, reduce only harness nondeterminism (for example fewer agents or ticks in that scenario), re-record with the same command, and run compare mode twice more. Never weaken the comparison.
4. Run `cargo fmt --all -- --check` (fix with `cargo fmt --all`), `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test -p xagent-sandbox --test contributing_guard` (shell, architect only). Leave the three touched files as the only changes for the host to commit.

## Done when

- **Done when:** `env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored` passes in compare mode on the tree with no code change from this plan, with the committed golden holding the lines `steering_population`, `trajectory_raw_vision`, `trajectory_cortex`, `headless_run` and `champion_replay` (plus any route scenarios from step 1), and the committed champion fixture under 2 MB: the byte-identical baseline of INV-01.
