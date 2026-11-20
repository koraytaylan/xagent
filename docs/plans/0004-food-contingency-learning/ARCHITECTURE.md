# Architecture: food contingency learning

## Design

```
WorldConfig.food_kinds_enabled ──► GpuKernel (draws nourishing kind on new + every agent reset)
                                     │  packs WC_FOOD_KINDS (slot 27): 0 off, 1 First nourishing, 2 Second nourishing
                                     ▼
 common.wgsl helpers: food_kinds_enabled(), food_item_kind(i), food_item_is_nourishing(i), food_item_color(i)
     │                          │                                   │
     ▼                          ▼                                   ▼
 phase_food_claim.wgsl     vision shaders (colour by kind)     phase_vision.wgsl: pair 1 = First odour
 (nourishing: +energy,      phase_vision*.wgsl                  (scent channels); pair 2 = Second odour,
  P_FOOD_COUNT; harmful:                                         written to the extra sensory channels
  −integrity, death mark,                                        right after scent
  P_HARMFUL_MEAL_COUNT)                                              │
                                                                    ▼
 BrainConfig.second_odour_sense_enabled ─► senses_for_config() ─► [SMELL, VISION, SECOND_ODOUR]
    │                                         │                      │
    ▼                                         ▼                      ▼
 BrainLayout::from_config                 GpuKernel::new ─►     coop_feature_extract / phase_brain_features:
 (sensory_stride +2, feature_count +2,    new_with_senses;      extra channels copied after the scent features
  third sense block appended)             EXTRA_SENSORY_CHANNELS = 2   │
                                                                    ▼
                                                 0003 sense block: adapt → whiten → Turn route (LearnedOnly)
free-life harness (0001) ──► food_kind_metrics (per-kind turns, meals by quarter, link accumulators, summaries)
   --food-kinds --freeze-policy  └► kind_readout (k-fold L2 logistic, shuffled control)
   --food-kind-summary FILE
bin food_kind_verdict INPUT [--write-bundle FILE] [--expect-verdict TEXT]   (no GPU)
```

Decisions:

- **Kind identity.** A food item's kind is its index parity (even = `First`,
  odd = `Second`). Respawned items keep their index, so the two kinds stay
  equal in number. There is no per-item buffer, and the food layout
  (`FOOD_STATE_STRIDE` = 4) does not change.
- **Assignment draw.** The nourishing kind is
  `WorldConfig::nourishing_food_kind(draw_index)`: a splitmix64 hash of `seed`
  and `draw_index`, low bit 0 → `First`. `GpuKernel` keeps a draw counter. It
  draws index 0 at construction and the next index on every agent reset that
  actually runs, but only when food kinds are on. The current kind is available
  from `GpuKernel::nourishing_food_kind()`. Every generation and every seeded
  run therefore gets a redraw, deterministically.
- **Harm magnitude.** The loss is `HARMFUL_MEAL_INTEGRITY_PER_ENERGY =
  ENERGY_WEIGHT / INTEGRITY_WEIGHT` (declared in `common.wgsl`, mirrored in
  `buffers.rs`) × `food_energy / max_energy × max_integrity`. A harmful meal's
  raw gradient is therefore the negative of a nourishing meal's. There is no
  new tunable parameter, and `WORLD_CONFIG_SIZE` stays 28.
- **Second-kind colour.** `FOOD_SECOND_KIND_COLOR = (0.85, 0.3, 0.75, 1.0)`
  (magenta). The baseline food colour is `(0.7, 0.95, 0.2, 1.0)`. The two
  differ in every RGB hemifield mean.
- **Second odour as a declared sense** (PLAN.md "Owner decisions on the 0003
  interface"). In `crates/xagent-brain/src/senses.rs`:
  - `SECOND_ODOUR` is name `"second_odour"`, channels `SCENT_CHANNELS`, input
    `NonVisualFeatures { offset: SECOND_ODOUR_FEATURE_OFFSET }` with
    `SECOND_ODOUR_FEATURE_OFFSET = SCENT_FEATURE_OFFSET + SCENT_CHANNELS` (27),
    adaptation rate `SMELL.adaptation_rate`, whitening `SMELL.whitening`,
    `LearningRate::Actor`, and routes `SECOND_ODOUR_ROUTES = &[Turn /
    LearnedOnly]`;
  - `senses_for_config(config)` selects `DECLARED_SENSES` or
    `SENSES_WITH_SECOND_ODOUR`;
  - `extra_sensory_channels(config)` returns `SECOND_ODOUR.channels` when the
    flag is on and `0` otherwise.

  The WGSL override `EXTRA_SENSORY_CHANNELS` (default `0u`) grows
  `SENSORY_STRIDE` and `NON_VISUAL_FEATURE_COUNT`. The extra channels follow
  the scent channels in the sensory buffer (`scent_base + SCENT_CHANNELS`).
  The serial vision pass writes the second odour there, guarded by
  `EXTRA_SENSORY_CHANNELS >= SCENT_CHANNELS`. Both feature extractors copy the
  `EXTRA_SENSORY_CHANNELS` values right after the scent features and before
  the danger percept, which keeps the danger write sequential and correct.
- **Scalar scent text.** The textual scalar call `let scent =
  sense_scent(pos, facing, smell_strength);` stays exactly once, so
  `with_prepared_scent` in `gpu_kernel.rs` is unchanged. The kind filter lives
  inside `sense_scent` and `vision_prepare_scent`. The second pair uses a
  separate serial `sense_second_odour`.
- **Parity golden preserved.** `PHYS_STRIDE` grows to 54. The 0003 parity test
  then hashes per agent only physics slots `[0, P_HARMFUL_MEAL_COUNT)`, which
  is the same byte stream as before, and asserts that the new slot is zero.
- **Verdict as its own binary.** `src/bin/food_kind_verdict.rs` is an
  auto-discovered second binary. `default-run = "xagent"` keeps
  `cargo run -p xagent-sandbox` meaning the main binary. The verdict reads
  either the directory of 8 per-run files or a bundle file. It writes the
  validated bundle with `--write-bundle`. The committed evidence is one bundle,
  `results.json`.
- **Control.** The frozen-policy arm is the same run with
  `GpuKernel::set_freeze_steering_weights(true)`.
- **GPU tests fail closed.** New GPU tests panic without an adapter when
  `XAGENT_REQUIRE_GPU` is set, and every GPU check sets it.

## Invariants

| id | invariant | guards | enforced at |
|---|---|---|---|
| INV-01 | Default-off parity: `food_kinds_enabled` and `second_odour_sense_enabled` default to `false`, including when omitted from JSON. When both are off: `WC_FOOD_KINDS` is 0; `EXTRA_SENSORY_CHANNELS` is 0; `senses_for_config` returns `DECLARED_SENSES`; every meal adds exactly `food_energy_value` and increments `P_FOOD_COUNT`; every item renders `(0.7, 0.95, 0.2, 1.0)`; the first nostril pair smells every uneaten item; the `BrainLayout` strides and sense offsets equal the layout without the new sense; and the 0003 parity golden still matches unchanged, with the new physics slot zero. | The evolution reference and stored champions stay untouched | `crates/xagent-shared/src/config.rs`, `crates/xagent-brain/src/senses.rs`, `crates/xagent-brain/src/buffers.rs`, `crates/xagent-brain/src/gpu_kernel.rs`, `phase_food_claim.wgsl`, the vision shaders, `crates/xagent-sandbox/tests/sense_block_parity.rs` |
| INV-02 | No value label: kind = index parity. The nourishing kind comes from a deterministic seed/draw hash; over 1,000 seeds, either kind's share is within [0.45, 0.55]. While kinds are on, the kernel redraws at construction and on every agent reset. No sensory channel, colour or odour encodes which kind is nourishing. | The agent must learn the contingency, not read it | `WorldConfig::nourishing_food_kind`, `GpuKernel` draw counter, the vision shaders |
| INV-03 | Harm only through the body: a harmful meal adds no energy. It subtracts `food_energy/max_energy × HARMFUL_MEAL_INTEGRITY_PER_ENERGY × max_integrity` from integrity, clamps to `[0, max_integrity]`, and marks death in the same scope when integrity reaches 0. It increments `P_HARMFUL_MEAL_COUNT`, which survives respawn like `P_FOOD_COUNT`, and not `P_FOOD_COUNT`. No reward term or flag is added. | The homeostatic-only learning contract | `phase_food_claim.wgsl`, `phase_death.wgsl`, `kernel_tick.wgsl`, `common.wgsl`, `buffers.rs` |
| INV-04 | Kind cues: while kinds are on, `Second` items render `FOOD_SECOND_KIND_COLOR` in every vision path. Nostril pair 1 smells only `First` items and pair 2 only `Second` items, with the same odour law as baseline smell. Pair 2 lands in the extra sensory channels right after scent. The serial and parallel pair-1 paths agree exactly. | The kinds are distinguishable by colour and odour only | `phase_vision.wgsl`, `phase_vision_parallel.wgsl`, `phase_vision_object_queries.wgsl`, `phase_vision_scent_parallel.wgsl` |
| INV-05 | Declared sense: the second pair enters only through the `SECOND_ODOUR` descriptor, read through the existing `NonVisualFeatures` input kind at feature offset 27. Its only route is `Turn` / `LearnedOnly`, and there is no sense-specific block code. With the flag on, the layout appends one `ClosedForm2` block after the 0003 route block without moving any base-sense slot relative to the tail. `sensory_stride` and `feature_count` each grow by exactly 2, the kernel emits `EXTRA_SENSORY_CHANNELS = 2`, and both feature extractors copy the two extra channels to features 27 and 28. Its block weights are zero in every birth brain and stay zero through mutation, crossover and inheritance. They become nonzero only through within-life learning. `second_odour_sense_enabled` is never mutated and is copied from parent A in crossover. | No hand-built pathway; learned-only weights; a feasible route through 0003 | `crates/xagent-brain/src/senses.rs`, `crates/xagent-brain/src/buffers.rs`, `crates/xagent-brain/src/gpu_kernel.rs`, `brain_passes.wgsl`, `phase_brain_features.wgsl`, `crates/xagent-sandbox/src/agent/mod.rs` |
| INV-06 | Measurement only: metrics, readout and link accumulators read kernel snapshots and copied samples. They never write brain, physics, food or config state. | The probes do not change what they measure | `food_kind_metrics.rs`, `kind_readout.rs`, `headless.rs` (free-life runner) |
| INV-07 | Control parity: the frozen arm shares the learning arm's seed, fresh brains, nourishing kind and every config value except `freeze_steering_weights`. Under freeze, the `second_odour` turn weights stay exactly 0 for a whole run. | A fair within-life comparison | `headless.rs` (free-life runner) |
| INV-08 | Verdict by code: the verdict comes from exactly 8 summaries (seeds 5–8 × learning/frozen, 300,000 ticks, 10 fresh agents, the same nourishing kind per seed) with the thresholds fixed in PLAN.md. FAIL names the first failing link in the order readout → credit → weight change. A missing, duplicate, malformed or mismatched input is an error. `--expect-verdict` exits non-zero on mismatch. `DECISION.md`'s verdict line matches the bundle. | The decision record cannot drift from the evidence | `crates/xagent-sandbox/src/bin/food_kind_verdict.rs`, `DECISION.md`, `results.json` |
| INV-09 | Scenario brain guard: `--food-kinds` with a stored champion brain is refused with an error before any dispatch. | Prevents layout mismatch | `headless.rs` (free-life runner) |
| INV-10 | GPU proofs fail closed: every new GPU test panics instead of skipping when `XAGENT_REQUIRE_GPU` is set and no adapter exists, and every GPU-backed check sets it. | Proofs that pass without running | each new GPU test file; task and plan checks |

## State

| state | owner | lifecycle | concurrency |
|---|---|---|---|
| `WorldConfig.food_kinds_enabled`, `BrainConfig.second_odour_sense_enabled` | config (serde, CLI) | Set before kernel construction and immutable during a run. The brain flag is never mutated. | Read-only after construction |
| Sense table selection (`senses_for_config`) | `senses.rs` | Compile-time tables; selection is a pure function of the flag | Immutable |
| Nourishing-kind draw counter and current kind | `GpuKernel` | Index 0 at `new`, +1 per agent reset that runs (a deferred `try_reset_agents` consumes none), only while kinds are on | Owned by the single thread that owns the kernel (sim worker or headless loop) |
| `WC_FOOD_KINDS` (slot 27) | `GpuKernel::write_world_config_scratch` | Rewritten with every world-config upload | GPU uniform, written only by the host between dispatches |
| `EXTRA_SENSORY_CHANNELS` override | `vision_override_constants` from `BrainLayout` | Fixed at pipeline creation | Immutable per kernel |
| `P_HARMFUL_MEAL_COUNT` (physics slot 53, `PHYS_STRIDE` 54) | `resolve_food_claim` (one thread per agent) | Generation-cumulative, saved and restored across respawn like `P_FOOD_COUNT`, zeroed by agent reset | Single writer per agent; read back only by CPU snapshots |
| Extra sensory channels (second odour) | serial vision pass, thread per agent | Rewritten every vision batch | Single writer per agent slot |
| `second_odour` sense block slots | 0003 sense block | Weights zero at birth. Mean and covariance follow the block's rules. | Brain workgroup, under the block's existing barriers |
| Per-run food-kind accumulators and readout samples | harness (`run_free_life` in `headless.rs`) via `food_kind_metrics` / `kind_readout` | Live for one run, then serialised to the summary JSON | Single-threaded CPU |
| `results.json`, `DECISION.md` | food-contingency-decision (architect) | Written once by the decision task; durable | None |

## Untrusted inputs

| input | bounds |
|---|---|
| Config JSON (`food_kinds_enabled`, `second_odour_sense_enabled`) | serde bools with default `false`; a non-bool value fails to parse. |
| CLI `--food-kinds`, `--freeze-policy` | Flags only. `--food-kinds` with a champion brain source is refused (INV-09). |
| CLI `--food-kind-summary <file>` | Path written once. An I/O error is returned, not panicked. |
| `food_kind_verdict <input> [--write-bundle <file>] [--expect-verdict <text>]` | The input is a directory that holds exactly `learning-seed{5,6,7,8}.json` and `frozen-seed{5,6,7,8}.json`, or a bundle file with exactly those 8 runs. Each run must parse and have 10 agents, 300,000 ticks, fresh brains, finite numbers, and a nourishing kind matching the other arm of its seed. The expected text must be one of `PASS`, `FAIL readout`, `FAIL credit`, `FAIL weight-change`. Anything else is an error with a non-zero exit. |
| Readout samples | Each class needs at least 2 × the fold count samples, or the score is absent (never a fake 0.5). Empty, ragged or non-finite features make it absent too. |
| `BrainLayout::with_senses(config, senses)` arguments | Panics if `config.second_odour_sense_enabled` disagrees with whether `senses` contains `SECOND_ODOUR`, so the strides can never disagree with the table. |

## Feasibility

| task | constraints and sources | compatibility witness | preflight check |
|---|---|---|---|
| food-kind-config | `WorldConfig`/`BrainConfig` derive serde with per-field defaults (`crates/xagent-shared/src/config.rs`). `crossover_config` and `mutate_config_with_strength_rng` are full `BrainConfig` literals (`crates/xagent-sandbox/src/agent/mod.rs`) and must gain the field. Re-exports are in `crates/xagent-shared/src/lib.rs`. | The serde default pattern and the crossover literal exist at baseline | `pf-config-serde-defaults` |
| second-odour-sense | 0003's `senses.rs` defines `DECLARED_SENSES = &[SMELL, VISION]` and `SMELL` with `NonVisualFeatures { offset: SCENT_FEATURE_OFFSET }` (0003 task 0201). Appended senses may declare only `Turn` (0003 task 0202), and `with_senses(config, senses)` is the builder. At baseline `SCENT_FEATURE_OFFSET` is 25 (`senses.rs`, `usize`; 0003 removed the WGSL copy) and `SCENT_CHANNELS` is 2, so offset 27 is the first slot after scent. `BrainConfig` is exported from `xagent-shared`. | The committed 0003 task text fixes the descriptor shape, the Turn-only rule and the builder; the baseline constants fix offset 27 | `pf-sense-interface` |
| food-kind-layout | `WORLD_CONFIG_SIZE` = 28 with slots 0–26 used, so slot 27 is free with no size change (`wconfig` is `array<vec4<f32>, 7>`). `PHYS_STRIDE` = 53 in Rust and WGSL; a slot appended at 53 gives 54. `ENERGY_WEIGHT` 0.6 and `INTEGRITY_WEIGHT` 0.4 exist. The rounding validation lists `P_FOOD_COUNT`. The 0003 parity test hashes the full physics state (0003 task 0101), so hashing the prefix `[0, 53)` keeps its byte stream. | The free slot 27, the stride mirrors, the weights and the parity hashing rule exist; an adapter is present for the parity check | `pf-task-food-kind-layout` |
| extra-sensory-layout | `SENSORY_STRIDE` and `NON_VISUAL_FEATURE_COUNT` are override expressions in `common.wgsl` (over `NON_VISUAL_FEATURE_CHANNELS`, 0003's rename of the WGSL `SCENT_CHANNELS`), and the danger percept already grows `feature_count` by a flag-gated width in `with_retina_flagged` (`buffers.rs`), so the extra-channel growth uses the same pattern. `from_config` exists; `with_senses(config, senses)` and the Turn-only rule for appended senses come from 0003 task 0202. | The two override lines, the danger-width pattern, `from_config` and the committed 0003 builder text exist; an adapter is present | `pf-task-extra-sensory-layout` |
| kernel-food-kind-draw | `reset_agents_seeded` and `reset_agents` are the reset hooks. `write_world_config_scratch` already overrides a slot (`WC_PHASE_MASK`) after `fill_world_config`, so `WC_FOOD_KINDS` is one more override. `GpuKernel::is_available` backs the fail-closed helper. | The reset APIs, the scratch override and the availability probe exist; an adapter is present | `pf-task-kernel-food-kind-draw` |
| second-odour-kernel | `vision_override_constants` already emits the flag-derived `DANGER_PERCEPT_FEATURES_ACTIVE`, so `EXTRA_SENSORY_CHANNELS` is one more insert; `vision_override_map_covers_root_constants` guards derived values; `dispatch_ticks` runs a built kernel. 0003 adds `new_with_senses` (0003 task 0204). | The override generator, its guard test, `dispatch_ticks` and the 0003 constructor signature exist; an adapter is present | `pf-task-second-odour-kernel` |
| kernel-food-kind-meals | One claim site adds `WC_FOOD_ENERGY` and bumps `P_FOOD_COUNT`. Its callers are `kernel_tick.wgsl`, `physics_tick.wgsl` and `persistent_tick.wgsl`. `P_FOOD_COUNT` is saved and restored across respawn in `kernel_tick.wgsl` and `phase_death.wgsl`. | The claim site, all three callers and both save/restore sites exist at baseline; an adapter is present | `pf-task-kernel-food-kind-meals` |
| food-kind-colour | The baseline food colour literal `hit_color = vec4<f32>(0.7, 0.95, 0.2, 1.0);` appears once in each of the three vision shaders, and the `vision_validation` regression module exists. | The three colour sites and the vision regression module exist; an adapter is present | `pf-task-food-kind-colour` |
| food-kind-senses | The scalar scent call appears exactly once, which keeps the `with_prepared_scent` text redirect valid. The two scent writes end at `scent_base + 1u`, so the extra channels start at `scent_base + SCENT_CHANNELS`. `vision_prepare_scent` is the parallel accumulator, and its guard `if alive && food_flags[food] == 0u {` takes the same filter. | The single scalar call, the scent write site, the parallel helper and its guard exist; an adapter is present | `pf-task-food-kind-senses` |
| second-odour-features | Both feature extractors (`coop_feature_extract` in `brain_passes.wgsl`, `phase_brain_features.wgsl`) write the two scent features last, before the danger block, so the extra channels go right after them. `dispatch_feature_phase_for_test` and `read_brain_scratch_blocking` expose the scratch route; `read_agent_state` exposes the block slots of the fused route. | The two scent-feature write lines, the danger guard and the three readback APIs exist; an adapter is present | `pf-task-second-odour-features` |
| second-odour-inheritance | `fresh_brain_state`, `mutate_steering_weights`, `steering_population` and `mutate_brain_state_seeded` are public in `crates/xagent-sandbox/src/agent/mod.rs`; `rand` is a normal dependency; 0003 task 0203 makes steering mutation kick only `Mutated` `Turn` routes. CPU only. | The four public APIs, the rand dependency and the committed 0003 mutation text exist | `pf-inheritance-apis` |
| food-kind-metrics | Metrics read `P_MOTOR_TURN_OUT`, food positions (`FOOD_STATE_STRIDE` 4) and bearings in the kernel's convention. They serialise with serde_json, which the sandbox already depends on. A new module is declared in `crates/xagent-sandbox/src/lib.rs`. | The physics and food layout constants, the lib module list and serde_json exist | `pf-metrics-sources` |
| food-kind-verdict | The sandbox declares one explicit `[[bin]]` (`xagent`, `src/main.rs`) with no `default-run`; `src/bin/` already holds 0002's `pathway_return_fit` and `pathway_route_gate`, and `src/bin/food_kind_verdict.rs` is free. Cargo auto-discovers `src/bin/food_kind_verdict.rs`, and `default-run = "xagent"` keeps the plain `cargo run -p xagent-sandbox` working. clap is already a dependency. | The single explicit bin, the free `src/bin/food_kind_verdict.rs` slot and the clap dependency exist | `pf-verdict-binary-slot` |
| kind-readout | A pure CPU module in `crates/xagent-brain/src/lib.rs`. `kind_readout.rs` does not exist yet. No new dependency. | The brain lib module list exists and the file is free | `pf-readout-crate` |
| food-kind-free-life | The control uses `GpuKernel::set_freeze_steering_weights`. Food positions come from `cached_food_state`. CLI flags use clap `#[arg(long)]` in `main.rs`. The binary is `src/main.rs` (`xagent`). The kernel picks the sense table from the flag alone (`GpuKernel::new`). | The freeze and food-snapshot APIs and the clap CLI exist; an adapter is present | `pf-task-food-kind-free-life` |
| food-contingency-decision | A release GPU run through the sandbox binary (the `--bench` route at baseline; the free-life route after food-kind-free-life). It needs a GPU adapter and fails closed without one. | The baseline `--bench` CLI and `run_bench` exist. The action builds the release binary, and the witness command runs a 2,000-tick, 2-agent release bench on the GPU. | `pf-decision-run-route` |
