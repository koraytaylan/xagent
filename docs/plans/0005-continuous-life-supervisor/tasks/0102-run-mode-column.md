---
id: run-mode-column
title: "Run mode column, supervisor-run resume refusal and crate-visible scoring hooks"
workstream: "0100"
kind: task
role: developer
depends_on: [evolution-mode-config]
gated: false
touches: [crates/xagent-sandbox/src/governor.rs]
status: todo
merged_as: ""
checks:
  - {name: focused-proof, command: "cargo test -p xagent-sandbox --lib governor::tests::evolution_mode_"}
  - {name: governor-tests, command: "cargo test -p xagent-sandbox --lib governor::tests::", timeout_ms: 3600000}
  - {name: seeded-digest-strict, command: "env XAGENT_REQUIRE_GPU=1 XAGENT_REQUIRE_SEEDED_DIGEST=1 cargo test -p xagent-sandbox --test seeded_headless_digest", timeout_ms: 3600000}
  - {name: sandbox-clippy, command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings", timeout_ms: 3600000}
  - {name: sandbox-fmt, command: "cargo fmt -p xagent-sandbox -- --check", fix: "cargo fmt -p xagent-sandbox"}
  - {name: sandbox-guard, command: "cargo test -p xagent-sandbox --test contributing_guard"}
---

# Run mode column, supervisor-run resume refusal and crate-visible scoring hooks

## Outcome

The `run` table gains an idempotent `mode` column (default `'generations'`).
`Governor::resume` refuses a latest run whose mode is `'supervisor'`, and still
resumes pre-plan databases as generation mode. `composite_fitness`, the
brain-blob codec and `init_schema` become `pub(crate)`, and
`exploration_grid_cells()` replaces the inline grid expression in `evaluate`.
The digest is unchanged. Spec: ARCHITECTURE.md "Governor hooks" and INV-11.

## Scope

Only `crates/xagent-sandbox/src/governor.rs`:

- `init_schema`, beside the other run migrations (after the
  `simulation_ticks` line, ~2136): add
  `let _ = db.execute_batch("ALTER TABLE run ADD COLUMN mode TEXT DEFAULT 'generations';");`.
  Add the same line to the migration block at the top of `Governor::resume` (~692).
- `Governor::resume`: read `COALESCE(mode, 'generations')` of the latest run
  (extend the `FROM run ORDER BY id DESC LIMIT 1` query at ~702, or run a second
  query on that same row). If the value is `'supervisor'`, call `log::warn!`
  naming the mode and return `Err(rusqlite::Error::QueryReturnedNoRows)`.
- `pub(crate)` on `composite_fitness` (~219), `encode_brain_state` (~155),
  `decode_brain_state` (~175) and `init_schema` (~2076).
- New `pub(crate) fn exploration_grid_cells() -> f32` with a doc comment: the
  per-generation exploration denominator, a quarter of the heatmap grid. It
  returns `(HEATMAP_RES * HEATMAP_RES / 4) as f32`. `evaluate` (~875) calls it
  in place of `let total_grid_cells = (HEATMAP_RES * HEATMAP_RES / 4) as f32;`.
- Generation-mode INSERT statements are unchanged; the default fills `mode`.
- Tests in the existing governor `tests` module, names exact, file databases
  via `tempfile`:
  - `evolution_mode_pre_plan_database_resumes`: create the file DB with raw SQL
    holding the baseline `run` and `node` tables WITHOUT `mode` (column lists
    copied from `init_schema`), a run row whose `governor_config` JSON lacks
    `evolution_mode` and `supervisor`, and one root node. `Governor::resume` is
    `Ok`, `config.evolution_mode` is `Generations`, and `SELECT mode FROM run`
    returns `'generations'`.
  - `evolution_mode_migration_is_idempotent`: `init_schema` twice on one
    connection; `PRAGMA table_info(run)` lists exactly one `mode` column.
  - `evolution_mode_resume_refuses_supervisor_run`: a run row with
    `mode = 'supervisor'` makes `resume` return `Err`.
  - `evolution_mode_new_run_records_generations`: `Governor::new_seeded` on a
    file DB, then `SELECT mode FROM run` returns `'generations'`.
  - `evolution_mode_exploration_grid_cells_is_quarter_heatmap`:
    `exploration_grid_cells() == (HEATMAP_RES * HEATMAP_RES / 4) as f32 == 1024.0`.

## Non-goals

- No supervisor tables (task `supervisor-store`).
- No change to INSERT statements, node, agent_result or node_brain rows,
  fitness math, or `config.rs`.
- No change to `evaluate`'s numeric result; only the expression's home moves.

## Prerequisites

- `evolution-mode-config`: `GovernorConfig.evolution_mode` and
  `EvolutionMode`, and the `governor.rs` struct literals already completed.
- Plan 0001's strict seeded digest test. It hashes node, agent_result and
  node_brain only, so a `run.mode` column cannot change its input.

## Decisions

- A refused resume returns `QueryReturnedNoRows`; no new error type is added.
- `COALESCE(mode, 'generations')` treats any NULL mode, such as one from an
  older row, as generation mode.
- `exploration_grid_cells` is a function, not a `const`, so the existing
  usize-to-f32 cast stays exactly as written.

## Steps

**Steps:**
1. In `governor.rs`, add the `mode` migration to `init_schema` and to the
   migration block of `Governor::resume`.
2. Make `composite_fitness`, `encode_brain_state`, `decode_brain_state` and
   `init_schema` `pub(crate)`. Add `exploration_grid_cells` with its doc comment
   and use it in `evaluate`.
3. In `Governor::resume`, read `COALESCE(mode, 'generations')` for the latest
   run. On `'supervisor'`, `log::warn!` the mode and return the
   `QueryReturnedNoRows` error.
4. Add `evolution_mode_pre_plan_database_resumes` and
   `evolution_mode_migration_is_idempotent` to the governor `tests` module.
5. Add `evolution_mode_resume_refuses_supervisor_run` and
   `evolution_mode_new_run_records_generations`.
6. Add `evolution_mode_exploration_grid_cells_is_quarter_heatmap`.

- **Done when:** check focused-proof passes with the five tests
  `evolution_mode_pre_plan_database_resumes`, `evolution_mode_migration_is_idempotent`,
  `evolution_mode_resume_refuses_supervisor_run`,
  `evolution_mode_new_run_records_generations` and
  `evolution_mode_exploration_grid_cells_is_quarter_heatmap` reported `ok`;
  check governor-tests passes; check seeded-digest-strict passes with its
  digest unchanged. Together they prove INV-11 (idempotent `mode` column,
  pre-plan database resumes as `generations`, supervisor run refused) and
  INV-01 (strict seeded digest unchanged).

## Evidence and risks

- The worker has no shell. Proof is the host-run checks above. Sandbox-clippy,
  sandbox-fmt and sandbox-guard also run host-side but are not Done-when items.
- If `init_schema` or `resume` differs from the line numbers above, the worker
  searches for the named function and continues. If the named function is
  missing, the worker reports the file and symbol without editing outside
  `touches`.
- The pre-plan test copies column lists from `init_schema`. If that copy
  diverges from the baseline schema, the test is wrong, not the migration.
