---
id: inheritance-default
title: "Weight-copy inheritance default and override record"
workstream: "0100"
kind: task
role: developer
depends_on: []
gated: false
status: todo
merged_as: ""
touches:
  - crates/xagent-shared/src/config.rs
  - crates/xagent-sandbox/src/main.rs
  - docs/measurements/continuous-supervisor/INHERITANCE-DEFAULT.md
checks:
  - name: focused-proof
    command: cargo test -p xagent-shared --lib supervisor_
  - name: cli-proof
    command: cargo test -p xagent-sandbox --bin xagent evolution_cli
  - name: record-proof
    command: |-
      sh -c 'f=docs/measurements/continuous-supervisor/INHERITANCE-DEFAULT.md && grep -qF "overrides the 0005 REJECT verdict" $f && grep -qF "weight-copy is never adopted" $f && grep -qF "the supervisor becomes the only mode" $f && grep -qF "the user'"'"'s decision, not a new measurement" $f && head -n1 docs/measurements/continuous-supervisor/DECISION.md | grep -qxF "Verdict: REJECT"'
  - name: supervisor-worker-proof
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --bin xagent sim_runtime::tests::supervisor
    timeout_ms: 3600000
  - name: fmt
    command: cargo fmt --all -- --check
    fix: cargo fmt --all
  - name: clippy
    command: cargo clippy --workspace --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: guard
    command: cargo test -p xagent-sandbox --test contributing_guard
---

# Weight-copy inheritance default and override record

## Outcome

`Inheritance::default()` and a `SupervisorConfig` JSON without `inheritance`
give `WeightCopy`. `"darwinian"` and `--supervisor-inheritance darwinian` still
select `Darwinian`. The record `INHERITANCE-DEFAULT.md` states the user's
override. `DECISION.md` and `results.json` are not edited.

## Decisions

- Only the `#[default]` attribute moves from `Darwinian` to `WeightCopy`.
  `default_inheritance()` keeps returning `Inheritance::default()`. Serde names
  stay `darwinian` and `weight_copy`.
- `Inheritance` doc comments: `WeightCopy` is described as the default newborn
  inheritance (no longer "comparison arm only"); `Darwinian` is described as
  the selectable mutated-birth-genome alternative. Plain prose, no plan or
  task references.
- The existing test `supervisor_config_defaults_when_omitted` and the existing
  CLI tests stay unchanged; they remain consistent with the new default.
- `evolution_cli_inheritance_with_generations_is_error` does not assert that
  Darwinian is selected, so a new CLI test is added rather than edited.

## Non-goals

- No change to `EvolutionMode` (generation mode stays the default here; its
  removal is a later plan).
- No edit to `docs/measurements/continuous-supervisor/DECISION.md` or
  `docs/measurements/continuous-supervisor/results.json`.
- No change to supervisor physics, decisions, random streams or the worker.
- No plan numbers, task slugs or workstream words in Rust source
  (`crates/xagent-sandbox/tests/contributing_guard.rs` scans `crates/**/*.rs`
  and `*.wgsl`; `docs/` is not scanned).

**Steps:**

1. In `crates/xagent-shared/src/config.rs`, move `#[default]` from
   `Inheritance::Darwinian` to `Inheritance::WeightCopy`. Rewrite the two
   variant doc comments as described in Decisions. Leave `default_inheritance()`
   and `SupervisorConfig` unchanged.
2. In the same file's `mod tests`, add four tests, each named with the prefix
   `supervisor_inheritance_`:
   - `supervisor_inheritance_default_is_weight_copy`: `Inheritance::default()`
     equals `Inheritance::WeightCopy`.
   - `supervisor_inheritance_omitted_json_is_weight_copy`:
     `serde_json::from_str::<SupervisorConfig>("{}")` has `inheritance` equal to
     `Inheritance::WeightCopy`.
   - `supervisor_inheritance_darwinian_json_selects_darwinian`:
     `serde_json::from_str::<SupervisorConfig>(r#"{"inheritance":"darwinian"}"#)`
     has `inheritance` equal to `Inheritance::Darwinian`.
   - `supervisor_inheritance_weight_copy_serialises_snake_case`:
     `serde_json::to_string(&Inheritance::WeightCopy)` equals `"\"weight_copy\""`.
3. In `crates/xagent-sandbox/src/main.rs`, reword the doc comment on the
   `supervisor_inheritance` CLI field so that its default note reads that the
   value is kept from the config, whose default is weight-copy. Add a test
   `evolution_cli_darwinian_still_selectable` in the existing `mod tests`
   (import `Inheritance` inside the function). It calls `evo` with
   `--evolution-mode supervisor --supervisor-inheritance darwinian`, expects
   `apply_evolution_cli` to succeed, and expects
   `config.governor.supervisor.inheritance` to equal `Inheritance::Darwinian`.
   Do not change any other line of `main.rs`.
4. Create `docs/measurements/continuous-supervisor/INHERITANCE-DEFAULT.md`,
   at most about 30 lines of plain prose. It must state, using these exact
   phrases:
   - newborn inheritance now defaults to weight-copy;
   - the user "overrides the 0005 REJECT verdict" recorded in `DECISION.md`,
     and the 0005 rule that "weight-copy is never adopted";
   - "the supervisor becomes the only mode", and generation mode is removed by
     a later plan;
   - this is "the user's decision, not a new measurement", and no new runs back
     it;
   - `DECISION.md` and `results.json` are kept unchanged as history.
5. Make no other edits. In particular, do not touch `DECISION.md`,
   `results.json`, or any file outside the three touches, and add no
   plan, task, or workstream words to Rust source, so the guard ratchet keeps
   its current count for `crates/xagent-shared/src/config.rs`.

- **Done when:** the focused-proof, cli-proof, record-proof, supervisor-worker-proof, fmt, clippy and guard checks all pass, the four `supervisor_inheritance_` tests and `evolution_cli_darwinian_still_selectable` pass, the record contains the required phrases and `DECISION.md` still starts with `Verdict: REJECT`, proving INV-1 and INV-2.
