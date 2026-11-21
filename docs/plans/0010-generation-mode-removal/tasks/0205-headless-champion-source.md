---
id: headless-champion-source
title: "Free-life champion birth-brain source removed"
workstream: "0200"
kind: task
depends_on:
  - headless-supervisor-only
gated: false
role: developer
touches:
  - crates/xagent-sandbox/src/headless.rs
  - crates/xagent-sandbox/tests/archive_free_life.rs
  - crates/xagent-sandbox/tests/free_life.rs
  - crates/xagent-sandbox/tests/food_kind_free_life.rs
  - crates/xagent-sandbox/tests/lesion_matrix_record.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test archive_free_life --test free_life --test food_kind_free_life"
    timeout_ms: 5400000
  - name: headless-unit
    command: "cargo test -p xagent-sandbox --lib headless::"
    timeout_ms: 3600000
  - name: champion-free
    command: |-
      sh -c '! grep -nE "EvolutionMode|evolution_mode|crate::governor|xagent_sandbox::governor|[Cc]hampion|CHAMPION|remove_file|reset_database" crates/xagent-sandbox/src/headless.rs crates/xagent-sandbox/tests/archive_free_life.rs crates/xagent-sandbox/tests/free_life.rs crates/xagent-sandbox/tests/food_kind_free_life.rs'
  - name: all-targets-compile
    command: "cargo check --workspace --all-targets"
    timeout_ms: 3600000
---

# 0205 Free-life champion birth-brain source removed

## Outcome

The free-life harness has only the `fresh` and `archive` birth-brain sources;
the champion source that read a generation database through `governor` is
gone with its errors, report fields and tests, and `headless.rs` no longer
imports `governor`.

## Decisions

- **`headless.rs`.** Delete `BrainSource::Champion`, `SOURCE_CHAMPION`,
  `FreeLifeError::ChampionLoad`, `FreeLifeError::ChampionLayout`,
  `FreeLifeError::FoodKindChampion` and their `Display` arms, the champion branch
  of birth-brain loading, the `node_id` and `generation` birth fields used only
  by it, `FreeLifeReport.champion_node_id` and `champion_generation`, and the
  remaining `crate::governor` imports (`load_champion_read_only`,
  `ChampionCapture`, `ChampionLoadError`). The report label set is `fresh` and
  `archive`. In `run_free_life_inner` set `fresh_brains: true`, because
  `run_free_life_food_kind` admits only `BrainSource::Fresh`.
- **Tests.** In `free_life.rs` delete the Governor fixture helpers
  (`FIXTURE_DB`, the helper that builds a governor database,
  `stored_generation`), `free_life_champion_source_reports_node_and_generation`,
  `free_life_champion_wrong_layout_is_rejected`,
  `free_life_missing_champion_node_is_reported` and the `champion_node_id`
  assertion. In `food_kind_free_life.rs` delete
  `food_kind_free_life_refuses_champion_brain`. In `archive_free_life.rs` delete
  the `evolution_mode` assignment and the `EvolutionMode` import, the
  `champion_node_id` and `champion_generation` assertions and the two
  `champion_*` keys of the expected JSON. Every other test stays.
- **`lesion_matrix_record.rs`.** This test only reads the frozen history record
  `docs/measurements/trunk-lesions/lesion-matrix.json` and its `DECISION.md`,
  which stay unedited (Non-goals). In `check_report` delete only the
  `let champion = r.brain_source == "champion"` line and the
  `r.champion_node_id` and `r.champion_generation` assertions, which read the
  deleted report fields. `SOURCES` keeps `["fresh", "champion"]`,
  `matrix_is_complete_and_finite` stays exactly as on the base, including its
  existing total-count assertion
  `assert_eq!(matrix.len(), SOURCES.len() * all_arms().len() * SEEDS.len())`
  that requires all 48 recorded runs (the base never had a per-source
  filter-and-count form there; do not introduce one), and the
  verdict recomputation stays champion-inclusive as `DECISION.md` documents:
  the `"champion"` string there is a label of historical recorded runs, not a
  birth-brain source, so this file is outside `champion-free` and INV-7 (it
  builds no Governor, runs no generation driver and ships no generation
  database). Narrowing `SOURCES` or editing the measurement record is not
  required and is out of scope.

**Steps:**

1. In `headless.rs` delete the champion source, constant, error variants,
   `Display` arms, birth branch, birth fields, report fields and governor
   imports, and set `fresh_brains: true` as in Decisions.
2. In the four test files make the edits in Decisions.
3. Remove imports and helpers left unused, with no `#[allow(dead_code)]` and no
   planning reference in source.

- **Done when:** `focused-proof` passes with the remaining `archive_free_life`, `free_life` and `food_kind_free_life` tests reported ok, and `headless-unit`, `champion-free` and `all-targets-compile` pass, proving no champion source or generation-database fixture survives in headless or its tests (INV-1, INV-7); `lesion_matrix_record.rs` keeps its historical `"champion"` run labels as Decisions state, and `all-targets-compile` proves it compiles without the deleted report fields.
