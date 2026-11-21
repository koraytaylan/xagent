# 0010 Generation mode removal — architecture

Binding specification for the tasks in `tasks/`. Paths are repository-relative.
This plan runs after 0006–0009 have landed; every task reads the then-current
file before editing and settles what it finds against the rules below, not
against line numbers recorded here.

## Dependents inventory (settled before any task starts)

Each entry is settled as **delete** (it served only generations) or **port**
(it guards something that still matters). Nothing generation-shaped survives in
test-only form: no test may construct a `Governor`, run a generation driver,
or open a generation database except to prove it is refused.

| Dependent | Verdict | What survives / where | Task |
| --- | --- | --- | --- |
| `crates/xagent-sandbox/tests/integration.rs` | port | Tests of surviving APIs stay; every test of the Governor, generation config fields, `EvolutionMode`, the generation handoff, the generation UI members, per-generation replay, `reset_database` and `check_existing_session` is deleted. Literals of `GovernorConfig` are rewritten as `GovernorConfig { population_size, mutation_strength, momentum_decay, supervisor, ..Default::default() }` using only the surviving fields. | integration-tests |
| `tests/seeded_headless_digest.rs` + `tests/fixtures/seeded_headless_digest.json` | delete | Replaced by the 0006 supervisor digest gate `tests/supervisor_digest.rs`, which is in PLAN.md `checks` (`supervisor-seeded-digest`) and in the deleting task's own checks. | generation-test-files |
| `tests/generation_run_summary.rs` | delete | Only measured generation runs and `max_generations_for_agent_ticks`; the supervisor run summary is covered by `supervisor_headless.rs` and `supervisor_run.rs`. | generation-test-files |
| `tests/evolution_cli_exit.rs` | port | Becomes the removed-flag and bad-budget exit test (INV-5). | cli-supervisor-only |
| `tests/sense_block_parity.rs` + golden + `sense_block_champion.db` | port | Sense-layout parity stays; every scenario runs a supervisor headless run; the generation-database champion fixture is deleted and the champion replay reads the rank-0 archive brain of the seeded supervisor run through `supervisor_store::load_archive_entry_read_only`. If 0006 already ported it, only residual generation references and the fixture are removed. | sense-parity-supervisor |
| `tests/food_kind_free_life.rs` | port | Keeps the frozen control and summary-file tests; the champion-refusal test and `FreeLifeError::FoodKindChampion` go with the champion source. | headless-supervisor-only |
| `tests/archive_free_life.rs` | port | Keeps the archive source; `evolution_mode` assignments and the `champion_*` report fields go. | headless-supervisor-only |
| `tests/free_life.rs` | port | Fresh-source tests stay; the Governor fixture and champion-source tests go. | headless-supervisor-only |
| `tests/supervisor_headless.rs`, `tests/supervisor_run.rs` | port | `generation_mode_is_the_default_driver` and `summary.mode` asserts go; the rest stays. | headless-supervisor-only |
| `tests/pathway_route_state.rs` | port then delete | Its three legacy-brain-upgrade cases move into `supervisor_store.rs` unit tests against the moved codec (store-legacy-guard); the file, which needs `Governor`, is deleted with the Governor (governor-removal). | store-legacy-guard, governor-removal |
| Validation A/B flags (`--validate-innate-instincts`, `--validate-danger-percept`, `--validation-*`) and `headless::{validate_*, run_innate_instinct_ab, run_headless_with_flags}` with their bootstrap statistics | delete | Nothing; they ran generation A/B arms. | cli-supervisor-only (flags), headless-supervisor-only (code) |
| Free-life champion-db source (`--free-life-champion-db`, `--free-life-champion-node`, `BrainSource::Champion`, `governor::load_champion_read_only`) | delete | `BrainSource::Archive` replaces it. | cli-supervisor-only, headless-champion-source, governor-removal |
| `src/bin/food_kind_verdict.rs`, `pathway_route_gate.rs`, `pathway_return_fit.rs`, `supervisor_verdict.rs` | delete | Their decision records under `docs/decisions/` and `docs/measurements/` stay as history, unedited. | study-binaries |
| 0005 measurement code that only compared against generations (generation-mode run summary in `headless.rs`, `max_generations_for_agent_ticks`, `RunSummary.mode`) | delete | The supervisor run summary stays. | headless-supervisor-only |
| `src/replay.rs`, `src/replay_coord.rs`, the `generation_recording` table | delete | Record types the 0009 tick-window replay imports are kept and moved only if their current home is deleted; the generation recording writer goes. | ui-replay-supervisor-only, cli-supervisor-only, governor-removal |
| `governor.rs` survivors: `composite_fitness`, `exploration_grid_cells`, brain-blob codec incl. the legacy upgrade path, `run` table schema | port | Fitness helpers move into `supervisor.rs`; codec and the run/supervisor schema move into `supervisor_store.rs`. | store-legacy-guard |

## Removed surface (shared contract)

Every task treats these as gone at integration; a task removes the members
inside its own touches and stops referencing members owned by later tasks:

- `Governor`, `AdvanceResult`, `ChampionCapture`, `TreeNode`, `check_existing_session`, `reset_database`, `load_champion_read_only`, every generation SQL table writer (`node`, `agent_result`, `mutation`, `generation_recording`, `behavior_metric`, `node_brain`).
- `SimEvent::GenerationBudgetReached`, `clamp_ticks_to_generation_budget`, `PendingGeneration`, `App.pending_generation`, `App.generation_epoch`, `SimInit` generation fields, champion-state requests that served generation handoff, `InheritedBrain` if only generations used it.
- `GovernorConfig` fields `tick_budget`, `elitism_count`, `max_generations`, `patience`, `eval_repeats`, `num_islands`, `migration_interval`, `evolution_mode`, and the `EvolutionMode` enum. Surviving fields: `population_size`, `mutation_strength`, `momentum_decay`, `supervisor`.
- GUI: fitness and hazard charts, evolution tree, node detail, "Previous Session Found" dialog, generation Resume and Reset, mode selector, editors of removed fields, per-generation replay; `EvolutionSnapshot` / `EvolutionAction` / `EvolutionState` members serving only these.
- CLI: `--generations`, `--dump-tree`, `--evolution-mode`, `--validate-innate-instincts`, `--validate-danger-percept`, `--validation-generations`, `--validation-population`, `--validation-replicates`, `--validation-tick-budget`, `--validation-seed-start`, `--free-life-champion-db`, `--free-life-champion-node`; `--resume` is removed unless 0007 made it resume a supervisor checkpoint, in which case it stays with that meaning.

## Interfaces relied on

- 0006: the supervisor digest gate `crates/xagent-sandbox/tests/supervisor_digest.rs`; supervisor telemetry tables written through `SupervisorStore`.
- 0007: checkpoint/resume code and any `--resume` meaning it set; its store opens go through `SupervisorStore` or call `supervisor_store::inspect_database` first.
- 0008: the observatory views in `ui.rs` and the browse/resume offer; it is the surviving session UI.
- 0009: tick-window replay; record types it imports from `replay.rs` survive.

## Legacy database rule

`supervisor_store::inspect_database(path: &Path) -> Result<DatabaseKind, StoreError>` with
`pub enum DatabaseKind { Missing, Empty, Supervisor, LegacyGenerations { generation: Option<u64> } }`
opens the file read-only (`OpenFlags::SQLITE_OPEN_READ_ONLY`, no pragma, no
schema statement). A file is `LegacyGenerations` when a `node` table has at
least one row, or a `run` table has a row whose `mode` column is missing, NULL
or not `'supervisor'`; `generation` is `MAX(node.generation)` when readable.
`LegacyGenerations` is refused everywhere with the `Display` text of
`StoreError::LegacyGenerations`:
`"<path> holds a generation-mode run (generation <n>). This version runs only the continuous supervisor and cannot run it. The file was not changed; choose another database with --db."`
(the parenthesis is omitted when `generation` is `None`).

## Invariants

| id | invariant | guards | enforced at |
| --- | --- | --- | --- |
| INV-1 | No generation state machine, generation persistence, worker generation handoff or `EvolutionMode` remains: `crates/` contains none of `GenerationBudgetReached`, `EvolutionMode`, `pending_generation`, `generation_epoch`, `crate::governor`, `xagent_sandbox::governor`, `run_generations`, `generation_recording`. | the objective's deletion list | each owning task's grep check; PLAN check `generation-surface-absent` |
| INV-2 | A legacy generations database is detected read-only and refused with the plain message, by the store, the headless driver and the GUI; the file's bytes are identical afterwards and no new file (such as `-wal`/`-shm`) is created beside it (open with the `immutable=1` URI or equivalent). | user's `xagent.db` (generation 3670) | `supervisor_store` unit tests; `supervisor_headless.rs`; GUI session unit test in `evolution.rs` |
| INV-3 | No code path deletes, truncates or overwrites a database file without an explicit two-step confirmation in the GUI; headless never deletes one. | today's Reset deleting the file | GUI unit test; `reset_database` absent |
| INV-4 | Config JSON carrying generation-era `governor` keys loads, the dead keys are ignored, and one warning line naming all of them is logged once per load. | old config files | `xagent-shared` config tests; `main.rs` unit test |
| INV-5 | Each removed CLI flag exits with status 2 and a message naming the flag and saying generation mode was removed, before any database file is created. | scripts using old flags | `tests/evolution_cli_exit.rs` |
| INV-6 | The removal does not change a seeded supervisor run: the 0006 supervisor digest gate passes unchanged. | bit-reproducibility | `supervisor-seeded-digest` check |
| INV-7 | No generation mode lives on in test-only form: no test constructs a Governor, runs a generation driver or ships a generation-database fixture except a raw-SQL legacy fixture built to prove refusal. | objective | owning tasks' grep checks |
| INV-8 | The brain-blob codec keeps decoding legacy (pre-upgrade) blobs as before. | archive blobs, checkpoint blobs | `supervisor_store` unit tests |
| INV-9 | The redraw path still makes no GPU dispatch or readback, and worker-to-main channels stay bounded latest-wins. | CLAUDE.md worker boundary | `sim_runtime.rs` unit tests |
| INV-10 | README.md, CLAUDE.md, CONTRIBUTING.md, EVOLUTION_JOURNEY.md and `crates/xagent-sandbox/README.md` describe the supervisor-only system; CLAUDE.md test counts equal the `sandbox-tests` output; the contributing-guard baseline matches the tree. | docs | `docs-supervisor-only` and `guard-baseline` checks |

## State

| state | owner | lifecycle | concurrency |
| --- | --- | --- | --- |
| Supervisor run database | `SupervisorStore` (worker thread or headless driver) | created only after `inspect_database` returns `Missing` or `Empty`; never deleted by code | one writer; read-only browsers open read-only |
| Legacy generations database | the user | read only by `inspect_database`; never opened writable | none |
| Simulation state | `sim_runtime.rs` worker | supervisor session start to stop | bounded latest-wins `SimEvent` channels; main thread never dispatches |
| Ignored-key warning | `main.rs` config loader | one line per config load | main thread only |

## Untrusted inputs

| input | bound |
| --- | --- |
| `--config` JSON | parsed by serde; unknown keys ignored; dead-key names reported only if they are in the fixed list of eight |
| `--db` file | classified read-only; any SQLite error from `inspect_database` is reported and refused, never "repaired" |
| Removed CLI flags | accepted only by hidden clap args that exit 2 before any side effect |

## Feasibility

| task | constraints and sources | compatibility witness | preflight check |
| --- | --- | --- | --- |
| study-binaries | the four `src/bin/*.rs` exist; bins are auto-discovered (`Cargo.toml` names only `xagent`); no test runs them | `test -f` and grep on `Cargo.toml` and tests | pf-study-binaries |
| integration-tests | `integration.rs` builds `GovernorConfig` literals and `Governor::new`; surviving APIs are reachable from the lib | grep on `integration.rs` and `ui.rs` | pf-integration-tests |
| generation-test-files | both test files and the digest fixture exist; nothing in `src/` reads the fixture | `test -f`, grep | pf-generation-test-files |
| sense-parity-supervisor | parity test reads a generation DB fixture via `Governor`; the store exposes `load_archive_entry_read_only` | grep on test and store | pf-sense-parity-supervisor |
| store-legacy-guard | store imports codec/schema from `governor`; `create` opens writable before checking; supervisor imports fitness from `governor`; rusqlite 0.34 bundled exposes `OpenFlags` | grep on store, supervisor, governor, Cargo.toml | pf-store-legacy-guard |
| worker-session-supervisor-only | worker event variant, clamp, `PendingGeneration` and `reset_database` exist in the worker, session and governor sources | grep on `sim_runtime.rs`, `gpu_orchestration.rs`, `evolution.rs`, `governor.rs` | pf-gui-cli-supervisor-only |
| ui-replay-supervisor-only | "Previous Session Found" exists in `ui.rs`; the generation replay wiring is in the bin | grep on `ui.rs` and `main.rs` | pf-gui-cli-supervisor-only |
| cli-supervisor-only | generation CLI fields and `mod replay_coord;` exist in `main.rs`; `evolution_cli_exit.rs` exists | grep on `main.rs`, `test -f` | pf-gui-cli-supervisor-only |
| headless-supervisor-only | `run_generations`, `dump_tree`, validators, `RunSummary.mode` and `generation_mode_is_the_default_driver` exist | grep on headless, run_summary and tests | pf-headless-supervisor-only |
| headless-champion-source | `BrainSource` and the champion tests in `food_kind_free_life.rs` and `free_life.rs` exist | grep on headless and tests | pf-headless-champion-source |
| governor-removal | `lib.rs` declares `governor` and `replay`; `pathway_route_state.rs` imports `Governor` | grep on `lib.rs`, governor, test | pf-governor-removal |
| config-generation-fields | `EvolutionMode`, the eight dead fields and the four survivors exist; no `deny_unknown_fields` in config | grep on `config.rs`, `lib.rs` | pf-config-generation-fields |
| sandbox-mode-test-users | the five supervisor tests set `evolution_mode = EvolutionMode::Supervisor`; `sense_block_parity.rs` exists (its line comes from sense-parity-supervisor) | grep on the six tests | pf-sandbox-mode-test-users |
| sandbox-governor-default-users | `EvolutionSnapshot::default` reads `GovernorConfig::default().tick_budget`; `steering_population` exists | grep on `ui.rs`, `agent/mod.rs` | pf-sandbox-governor-default-users |
| guard-baseline | guard baseline row for `governor.rs` exists | grep on guard | pf-docs-supervisor-only |
| docs-supervisor-only | CLAUDE.md governor line and 221 count; CONTRIBUTING governor rule | grep on docs | pf-docs-supervisor-only |
