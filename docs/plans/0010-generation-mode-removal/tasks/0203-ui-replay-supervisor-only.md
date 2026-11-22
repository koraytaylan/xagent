---
id: ui-replay-supervisor-only
title: "GUI views and per-generation replay wiring removed"
workstream: "0200"
kind: task
depends_on:
  - worker-session-supervisor-only
gated: false
role: developer
touches:
  - crates/xagent-sandbox/src/ui.rs
  - crates/xagent-sandbox/src/ui_chrome.rs
  - crates/xagent-sandbox/src/render_pipeline.rs
  - crates/xagent-sandbox/src/replay_coord.rs
  - crates/xagent-sandbox/src/app.rs
  - crates/xagent-sandbox/src/evolution.rs
status: done
merged_as: "36008f95d89005fc07cee22a37f8bca5814dbf55"
checks:
  - name: ui-unit
    command: "cargo test -p xagent-sandbox --lib ui::"
    timeout_ms: 3600000
  - name: bin-unit
    command: "cargo test -p xagent-sandbox --bin xagent"
    timeout_ms: 3600000
  - name: ui-generation-free
    command: "sh -c 'cd crates/xagent-sandbox/src && ! grep -nE \"Previous Session|GenerationRecording|last_recording|render_fitness_chart|render_tree|EvolutionMode|evolution_mode|max_generations|elitism_count|eval_repeats|num_islands|migration_interval|HasSession\" ui.rs ui_chrome.rs render_pipeline.rs replay_coord.rs app.rs evolution.rs'"
  - name: all-targets-compile
    command: cargo check --workspace --all-targets
    timeout_ms: 3600000
---

# 0203 GUI views and per-generation replay wiring removed

## Outcome

The egui UI carries no generation surface and the main thread no
per-generation replay. The fitness and hazard charts, evolution tree, node
detail, "Previous Session Found" dialog, generation Resume, mode selector and
editors of removed fields are gone, and so are the `EvolutionSnapshot`,
`EvolutionAction` and `EvolutionState` members that served only them. The
legacy-database modal from `worker-session-supervisor-only` and the plan 0007,
0008 and 0009 supervisor views, resume and browse offer and tick-window replay
are kept.

## Decisions

- **UI (`ui.rs`, `ui_chrome.rs`).** Delete `render_fitness_chart`, the hazard
  chart, `render_tree`, node detail, the "Previous Session Found" dialog (the
  `HasSession` state and its `ui_chrome.rs` arm), the generation Resume button,
  the mode selector and the editors of the eight removed `GovernorConfig`
  fields. Delete the snapshot, action and state members that served only them
  and their unit tests; keep the observatory, session dialog, browse and
  tick-window replay views and their tests. Render the legacy-database modal
  from the snapshot fields the session task added, with "Quit" and "Start in a
  new file" and the message text unchanged. Follow the egui notes in CLAUDE.md.
- **Arms (`evolution.rs`).** Delete the logged no-op arms the session task left
  for the variants this task deletes; change nothing else there.
- **Replay (`app.rs`, `render_pipeline.rs`, `replay_coord.rs`).** Delete
  `App.recording` and `App.last_recording`, their writers and the "Replay Gen N"
  wiring in `render_pipeline.rs`. `replay_coord.rs` is the per-generation replay
  coordinator (`impl App` over `GenerationRecording`); if plan 0009 put
  tick-window replay methods in it, move them into `app.rs` unchanged (they must
  not name `GenerationRecording`); then reduce `replay_coord.rs` to one `//!`
  line saying the module is empty. `cli-supervisor-only` deletes the file and
  `mod replay_coord;` from `main.rs`; this task does not edit `main.rs`.
- **Existing tests.** `tests/integration.rs`
  `evolution_snapshot_default_tree_pane_fraction` was settled by
  `integration-tests`; if a remaining unit test in a touched file asserts a
  deleted member, delete that test with the member.

**Steps:**

1. In `ui.rs` delete the generation charts, tree, node detail, Previous Session
   Found dialog, generation Resume, mode selector, removed-field editors and the
   snapshot, action and state members that served only them, with their tests.
2. In `ui_chrome.rs` delete the arms and labels for the deleted states and
   render the legacy-database modal as in Decisions.
3. In `evolution.rs` delete the no-op arms for the deleted variants.
4. In `app.rs` and `render_pipeline.rs` delete the per-generation recording
   fields, writers and replay wiring; move any tick-window method out of
   `replay_coord.rs` into `app.rs` and reduce `replay_coord.rs` to its one doc line.
5. Remove unused imports, add no `#[allow(dead_code)]` and no planning reference
   in source.

- **Done when:** `ui-unit`, `bin-unit`, `ui-generation-free` and `all-targets-compile` pass, proving the GUI carries no generation view, state or per-generation replay wiring (INV-1, INV-7) while the supervisor views and the legacy-database refusal stay (INV-2).
