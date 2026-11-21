---
id: config-generation-fields
title: "Remove generation-only GovernorConfig fields and EvolutionMode"
workstream: "0300"
kind: task
depends_on:
  - governor-removal
gated: false
role: developer
touches:
  - crates/xagent-shared/src/config.rs
  - crates/xagent-shared/src/lib.rs
status: todo
merged_as: ""
checks:
  - name: shared-tests
    command: "cargo test -p xagent-shared"
    timeout_ms: 1800000
  - name: config-generation-free
    command: |-
      sh -c '! grep -nE "EvolutionMode|\.(tick_budget|elitism_count|max_generations|patience|eval_repeats|num_islands|migration_interval|evolution_mode)\b|^\s*(pub )?(tick_budget|elitism_count|max_generations|patience|eval_repeats|num_islands|migration_interval|evolution_mode):|fn default_(eval_repeats|num_islands|migration_interval)\b" crates/xagent-shared/src/config.rs crates/xagent-shared/src/lib.rs'
  - name: all-targets-compile
    command: "cargo check --workspace --all-targets"
    timeout_ms: 3600000
---

# 0302 Remove generation-only GovernorConfig fields and EvolutionMode

## Outcome

`GovernorConfig` keeps exactly four keys: `population_size`,
`mutation_strength`, `momentum_decay` and `supervisor`, with their defaults
unchanged. The eight generation-era keys (`tick_budget`, `elitism_count`,
`max_generations`, `patience`, `eval_repeats`, `num_islands`,
`migration_interval`, `evolution_mode`) and the `EvolutionMode` enum are gone
from `crates/xagent-shared/src/config.rs`, and `EvolutionMode` is no longer
re-exported from `crates/xagent-shared/src/lib.rs`.

Old config JSON that still carries those keys keeps loading. serde ignores
unknown fields by default, so no `deny_unknown_fields` is added. The one-line
warning about dead keys is emitted by the sandbox config loader, not here, so
this task adds no dependency: `Cargo.toml` and `Cargo.lock` are not touched.

This task runs after `governor-removal`. Every other user of the removed
fields and of `EvolutionMode` was removed by an earlier task, so the workspace
compiles once this task lands.

## Decisions

- **Removed.** The eight keys listed above, the `EvolutionMode` enum with its
  derives and doc comment, the default functions used only by removed fields
  (`default_eval_repeats`, `default_num_islands`, `default_migration_interval`),
  and the `EvolutionMode` re-export in `lib.rs`.
- **Kept, unchanged.** `population_size` (`default_population_size`, 10),
  `mutation_strength` (`default_mutation_strength`, 0.1), `momentum_decay`
  (`default_momentum_decay`, 0.9) and `supervisor` (`SupervisorConfig`, with its
  own `#[serde(default)]`). `SupervisorConfig` and its defaults are not edited.
- **No `deny_unknown_fields`.** Do not add it to `GovernorConfig`, `FullConfig`
  or any other type in `config.rs`.
- **Docs.** The `GovernorConfig` doc comment changes from "Configuration for
  the evolution governor." to supervisor wording, e.g. "Settings for the
  supervisor-only evolution driver: population, mutation strength, momentum
  decay and the supervisor schedule. Config files that still carry the removed
  generation keys load with those keys ignored." Doc comments of the kept fields
  that say "per generation" or "per-island" are reworded to match (for example
  "per run" and "mutation momentum").
- **Test edits.** The `EvolutionMode` assertions are deleted.
  `governor_config_tuned_defaults` asserted only `tick_budget`; step 4 replaces
  that assertion and keeps the test name.

**Steps:**

1. In `crates/xagent-shared/src/config.rs`, delete the `EvolutionMode` enum
   together with its doc comment and derive attributes (the block starting at
   `/// How evolution is driven.` and ending at its closing brace). Delete the
   `evolution_mode` field and its `#[serde(default)]` attribute from
   `GovernorConfig`.
2. In the same file, delete the fields `tick_budget`, `elitism_count`,
   `max_generations`, `patience`, `eval_repeats`, `num_islands` and
   `migration_interval` from `GovernorConfig`, with their doc comments and
   attributes. Delete the functions `default_eval_repeats`,
   `default_num_islands` and `default_migration_interval`, with their comments.
   Keep `population_size`, `mutation_strength`, `momentum_decay` and `supervisor`
   exactly as they are, except for the doc-comment rewording in Decisions.
   Replace the `impl Default for GovernorConfig` body with:

   ```rust
   Self {
       population_size: default_population_size(),
       mutation_strength: 0.1,
       momentum_decay: 0.9,
       supervisor: SupervisorConfig::default(),
   }
   ```

3. In `crates/xagent-shared/src/lib.rs`, remove `EvolutionMode, ` from the
   `pub use config::{...}` block and reflow that block as `cargo fmt` would.
   Do not change any other re-export.
4. In the `#[cfg(test)] mod tests` of `config.rs`, make these edits and nothing
   else in those tests:
   - `supervisor_config_defaults_when_omitted`: delete the two
     `assert_eq!(...evolution_mode, EvolutionMode::Generations);` lines. Keep the
     JSON literal with the dead keys: it proves they are ignored.
   - `supervisor_config_round_trip`: delete the line
     `gov.evolution_mode = EvolutionMode::Supervisor;` and the line
     `assert_eq!(back.evolution_mode, EvolutionMode::Supervisor);`.
   - `supervisor_config_not_in_brain_config`: delete the `"evolution_mode",`
     entry from its key array. The other entries stay.
   - `governor_config_tuned_defaults`: replace
     `assert_eq!(config.tick_budget, 1_000_000);` with
     `assert_eq!(config.mutation_strength, 0.1);` and
     `assert_eq!(config.momentum_decay, 0.9);`.
5. Add these two tests to the same `tests` module, directly after
   `supervisor_config_round_trip`:

   ```rust
   #[test]
   fn legacy_generation_governor_json_loads() {
       let json = r#"{
           "governor": {
               "population_size": 12,
               "tick_budget": 1000000,
               "elitism_count": 3,
               "max_generations": 0,
               "patience": 5,
               "mutation_strength": 0.25,
               "eval_repeats": 2,
               "num_islands": 1,
               "migration_interval": 5,
               "momentum_decay": 0.8,
               "evolution_mode": "generations",
               "supervisor": { "archive_capacity": 7 }
           }
       }"#;
       let full: FullConfig = serde_json::from_str(json).expect("legacy governor JSON loads");
       assert_eq!(full.governor.population_size, 12);
       assert_eq!(full.governor.mutation_strength, 0.25);
       assert_eq!(full.governor.momentum_decay, 0.8);
       assert_eq!(full.governor.supervisor.archive_capacity, 7);
       let out = serde_json::to_string(&full).unwrap();
       for key in [
           "tick_budget",
           "elitism_count",
           "max_generations",
           "patience",
           "eval_repeats",
           "num_islands",
           "migration_interval",
           "evolution_mode",
       ] {
           assert!(!out.contains(key), "re-serialized config keeps {key}");
       }
   }

   #[test]
   fn governor_config_keys_are_supervisor_only() {
       let value = serde_json::to_value(GovernorConfig::default()).unwrap();
       let mut keys: Vec<&str> = value
           .as_object()
           .unwrap()
           .keys()
           .map(String::as_str)
           .collect();
       keys.sort_unstable();
       assert_eq!(
           keys,
           ["momentum_decay", "mutation_strength", "population_size", "supervisor"]
       );
   }
   ```

   The JSON omits `brain` and `world`, which are `#[serde(default)]` in
   `FullConfig`. Do not write `"brain": {}`: `BrainConfig` has required fields.
6. Run the checks `shared-tests`, `config-generation-free` and
   `all-targets-compile` on the host. `shared-tests` must report all tests passed,
   including `legacy_generation_governor_json_loads`,
   `governor_config_keys_are_supervisor_only` and
   `governor_config_tuned_defaults`. `config-generation-free` must exit 0.
   `all-targets-compile` must finish with no error.

- **Done when:** the checks `shared-tests` (including the tests
  `legacy_generation_governor_json_loads` and
  `governor_config_keys_are_supervisor_only`), `config-generation-free` and
  `all-targets-compile` pass on the host, proving that old config JSON carrying
  the generation-era keys still loads with those keys ignored (INV-4), and that
  `EvolutionMode` and the generation-only `GovernorConfig` fields are gone
  (INV-1).

## Non-goals

- No `deny_unknown_fields` and no logging dependency. The dead-key warning
  belongs to `cli-supervisor-only`.
- No change to surviving defaults, `SupervisorConfig`, `Cargo.toml` or
  `Cargo.lock`, or to any file outside touches.
