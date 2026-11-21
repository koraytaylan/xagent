---
id: governor-removal
title: "Delete the Governor module and per-generation replay"
workstream: "0300"
kind: task
depends_on:
  - study-binaries
  - integration-tests
  - generation-test-files
  - sense-parity-supervisor
  - store-legacy-guard
  - worker-session-supervisor-only
  - ui-replay-supervisor-only
  - cli-supervisor-only
  - headless-supervisor-only
  - headless-champion-source
gated: false
role: architect
touches:
  - crates/xagent-sandbox/src/governor.rs
  - crates/xagent-sandbox/src/replay.rs
  - crates/xagent-sandbox/src/lib.rs
  - crates/xagent-sandbox/tests/pathway_route_state.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "sh -c 'test ! -e crates/xagent-sandbox/src/governor.rs && test ! -e crates/xagent-sandbox/tests/pathway_route_state.rs && ! grep -nF \"pub mod governor;\" crates/xagent-sandbox/src/lib.rs && ! grep -rnE --include=*.rs \"crate::governor|xagent_sandbox::governor|generation_recording\" crates/'"
    timeout_ms: 300000
  - name: sandbox-lib-unit
    command: "cargo test -p xagent-sandbox --lib"
    timeout_ms: 3600000
  - name: all-targets-compile
    command: "cargo check --workspace --all-targets"
    timeout_ms: 3600000
---

# 0301 Delete the Governor module and per-generation replay

## Outcome

`crates/xagent-sandbox/src/governor.rs` is deleted with the Governor state
machine, its generation SQL persistence, the generation recording writer, the
champion loader, `reset_database`, `check_existing_session` and its tests.
`pub mod governor;` leaves `lib.rs`. `replay.rs` keeps only `TickRecord`, which
the 0009 tick-window replay imports (`crate::replay::TickRecord`, see
`docs/plans/0009-supervisor-tick-window-replay/tasks/0101-tick-window-core.md`);
`GenerationRecording`, `FoodEvent` and `VISION_KEYFRAME_INTERVAL` are deleted,
because only the generation recording used them. `tests/pathway_route_state.rs`,
which needs `Governor`, is deleted; its legacy-brain-upgrade cases were ported to
`supervisor_store` unit tests by `store-legacy-guard`.

This task runs after every caller-side task of the plan. It does not edit
`GovernorConfig` or `EvolutionMode`; those belong to `config-generation-fields`.

## Decisions

- **Survivors in `governor.rs`.** `composite_fitness` and `exploration_grid_cells`
  move to `supervisor.rs`; the brain-blob codec and `init_schema` move to
  `supervisor_store.rs`. Those moves are made by `store-legacy-guard`, which is a
  dependency. This task deletes the file without moving anything.
- **`replay.rs`.** Keep `TickRecord` with its fields unchanged and its derives
  (`Clone, Debug`). Delete `FoodEvent`, `VISION_KEYFRAME_INTERVAL`, the
  `GenerationRecording` struct and its `impl` block. Replace the module doc comment
  (it describes generation recordings) with a one-line doc saying the module holds
  the per-tick agent record used by supervisor tick-window replay. Rewrite the
  `vision_color` doc comment so it no longer links `VISION_KEYFRAME_INTERVAL`; it
  still says the field is stored only at keyframes. `replay.rs` has no unit tests
  to remove.
- **Callers.** No file outside this task's touches may import `governor` or
  `GenerationRecording`. If any such file remains (for example `main.rs`,
  `replay_coord.rs`, `ui.rs`, `app.rs`, `evolution.rs`, `headless.rs` or
  `tests/integration.rs`), the earlier task that owns it has not landed. Do not
  edit that file here. Stop and report it as a blocker for plan correction.
- **Deletion method.** Delete `governor.rs` and `pathway_route_state.rs` with the
  shell `rm`, which is why this task is architect-owned.

**Steps:**

1. Search `crates/` for `governor`, `GenerationRecording`, `FoodEvent`,
   `VISION_KEYFRAME_INTERVAL` and `replay::` users. Every hit outside this task's
   touches must be gone, because the owning earlier task removed it. Any
   remaining hit outside the touches is a blocker: report the file and stop.
2. Delete `crates/xagent-sandbox/src/governor.rs` and
   `crates/xagent-sandbox/tests/pathway_route_state.rs` with `rm`. Remove the line
   `pub mod governor;` from `crates/xagent-sandbox/src/lib.rs`. Leave
   `pub mod replay;` in place.
3. In `crates/xagent-sandbox/src/replay.rs`, keep only `TickRecord`, as in
   Decisions. Delete `FoodEvent`, `VISION_KEYFRAME_INTERVAL`, `GenerationRecording`
   and its `impl`. Rewrite the module doc and the `vision_color` doc comment. Add
   no `#[allow(dead_code)]` and no planning references.
4. Run the check `focused-proof`. It must exit 0, which proves `governor.rs` and
   `pathway_route_state.rs` are gone, `pub mod governor;` is absent from `lib.rs`,
   and no `crate::governor`, `xagent_sandbox::governor` or `generation_recording`
   reference remains in `crates/`.
5. Run the check `sandbox-lib-unit`. Every remaining library unit test must pass,
   including the `supervisor_store` and `supervisor` tests that now hold the moved
   codec and fitness helpers.
6. Run the check `all-targets-compile`. It must succeed with no error, so every
   integration test and binary still compiles against the trimmed `replay.rs`.
   Remove any `use` line that the compiler reports as unused in the four touched
   files. Then run all three checks on the host.

- **Done when:** the checks `focused-proof`, `sandbox-lib-unit` and
  `all-targets-compile` pass on the host, proving the Governor state machine,
  its generation persistence and the per-generation replay are deleted (INV-1),
  and that no test constructs or keeps a Governor alive (INV-7).

## Non-goals

- No change to `GovernorConfig`, the eight dead config keys or `EvolutionMode`
  (`config-generation-fields`).
- No change to `TickRecord` fields, to the supervisor digest or to the
  supervisor run path.
- No edit of docs (`docs-supervisor-only`), of history under `docs/decisions/`
  or `docs/measurements/`, or of any other plan.
- No dependency or version change.
