# 0003 — Uniform sense-motor route: architecture

## Starting point (baseline, plus plan 0002)

Two hand-built pathways feed the turn channel today:

- **Smell** (`brain_passes.wgsl`, thread 0 of the exploration block): two
  nostril features at `SCENT_FEATURE_OFFSET` in the non-visual tail, whitened
  by the closed-form 2×2 `whiten_scent` against a packed covariance
  `[l², l·r, r²]`, statistics at rate `SCENT_WHITENING_RATE`, weights updated at
  `ACTION_WEIGHT_LEARNING_RATE`. Slots `O_SCENT_MEAN` (2), `O_SCENT_COVARIANCE`
  (3), `O_SCENT_WHITENED` (2), `O_SCENT_TURN_WEIGHTS` (2), `O_TRACE_SCENT` (2).
- **Vision** (`vision_pathway_step`): eight hemifield means of the adapted raw
  field (silent when the visual cortex is on), full 8×8 covariance at
  `VISION_PATHWAY_RATE`, ZCA whitening refreshed by Jacobi every
  `VISION_WHITENING_REFRESH` brain ticks, learning rate scaled by the
  heritable `O_VISION_PLASTICITY` gene. Slots `O_VISION_PATHWAY_MEAN` (8),
  `_COVARIANCE` (64), `_WHITENING` (64), `_INPUT` (8), `O_VISION_TURN_WEIGHTS`
  (8), `O_TRACE_VISION` (8).
- Both credit the executed fresh turn-noise innovation `s_explore[2]`,
  normalised by `1 + |input|²`, are gated by `steering_weights_learn()`, settle
  at death in `settle_turn_pathways_at_death`, and are mutated by
  `mutate_steering_weights` (smell first, then vision, one draw per weight).
- Plan 0002 adds `BrainConfig::critic_input` (`encoding` | `pathway` |
  `encoding_and_pathway`) and `BrainConfig::forward_pathway_enabled`, one
  contiguous `PATHWAY_ROUTE_SLOT_COUNT` (1321) slot block after
  `O_TRACE_VISION` (`O_FORWARD_PATHWAY_WEIGHTS` 10, `O_TRACE_FORWARD_PATHWAY`
  10, `O_CRITIC_PATHWAY_WEIGHTS` 10, `O_TRACE_CRITIC_PATHWAY` 10,
  `O_PREV_PATHWAY_INPUT_NORM` 1, `O_RECENT_PATHWAY_KEYS` 1280) whose 10 inputs
  are `[smell 2, vision 8]`, a legacy-blob upgrade in `decode_brain_state`, and
  moves the whitening before the value reduction. Its verdicts are in
  `docs/decisions/pathway-critic-inputs.{md,json}`,
  `docs/decisions/pathway-critic-route.{md,json}` and
  `docs/decisions/pathway-forward-route.{md,json}`; a route is adopted (its
  `DEFAULT_*` constant flipped) only if all its gates passed.

Text anchors on this code are load-bearing: `compose_brain_passes`
(`gpu_kernel.rs`), `context_gather.rs`, `brain_sections.rs` and the whitening
and visual-pathway validation modules splice or search `brain_passes.wgsl`
source. Any rename moves those anchors in the same task.

## Chosen design

### Sense descriptor (`crates/xagent-brain/src/senses.rs`)

```rust
pub enum SenseInput { NonVisualFeatures { offset: usize }, VisualHemifields }
pub enum Whitening { ClosedForm2 { initial_variance: f32 }, Jacobi { refresh_ticks: u32, initial_variance: f32 } }
pub enum LearningRate { Actor, ActorTimesGene { gene_offset: usize } }
pub enum MotorRoute { Turn, Forward, Critic }
pub enum Heritability { Mutated, LearnedOnly }
pub struct RouteDescriptor { pub route: MotorRoute, pub heritability: Heritability }
pub struct SenseDescriptor {
    pub name: &'static str,
    pub channels: usize,
    pub input: SenseInput,
    pub adaptation_rate: f32,
    pub whitening: Whitening,
    pub learning_rate: LearningRate,
    pub routes: &'static [RouteDescriptor],
}
pub const DECLARED_SENSES: &[SenseDescriptor] = &[SMELL, VISION];
pub const MAX_SENSE_SLOTS: usize = 4;
pub const MAX_SENSE_CHANNELS: usize = 8;
```

Exact field names may be refined by `sense-descriptor`; the variants and the
rules below are fixed. Routes: `Turn` (Mutated) for both senses always;
`Forward` (LearnedOnly) for both iff `docs/decisions/pathway-forward-route`
says adopted and `DEFAULT_FORWARD_PATHWAY_ENABLED` is true; `Critic`
(LearnedOnly) for both iff `docs/decisions/pathway-critic-route` says adopted
and `DEFAULT_CRITIC_INPUT` is not `Encoding`. A record/default disagreement or
a missing record blocks the task. `sense-descriptor` records the verdicts in
`0201-ROUTE-WIRING.md`. A rejected route is not declared; its flag-gated code
stays as plan 0002 left it, outside the block, reading its inputs through the
layout.

### Derived layout

`BrainLayout` gains the per-sense offsets (`sense_blocks()`), computed by one
deterministic rule from `DECLARED_SENSES`: per sense, in declared order, mean
(n), covariance (3 for `ClosedForm2`, n² for `Jacobi`), whitening matrix (n²,
`Jacobi` only), whitened input (n), turn weights (n), turn trace (n); then plan
0002's route block (present whatever the verdicts) placed right after the base
senses with its constants unchanged, each sense's forward/critic weights and
traces being its declared-order slice of that block; any sense declared later
appends after the route block and before `O_SENSORY_MEAN`, so declaring a
sense never moves an existing slot. The public
`O_SCENT_*`, `O_VISION_PATHWAY_*`, `O_VISION_TURN_WEIGHTS`, `O_TRACE_SCENT`,
`O_TRACE_VISION` (and plan 0002's route constants) stay as names but are
computed from the descriptor table by `const fn`, so plan 0001/0002 callers
keep compiling. `FIXED_TAIL_SIZE`, `BRAIN_STRIDE` and
`feature_count_for_brain_stride` keep their values for the declared sense set.

### WGSL block

Per-sense parameters reach WGSL only as pipeline override constants emitted by
`vision_override_constants(layout)` from `layout.sense_blocks()`:
`SENSE_COUNT` plus, for each slot `k < MAX_SENSE_SLOTS`, input kind/offset,
channel count, whitening kind, refresh cadence, rates, learning-rate source,
route mask and the relative slot offsets. `common.wgsl` declares them with
defaults equal to the declared table (a Rust test parses and compares them).
`brain_passes.wgsl` and `common.wgsl` contain one block — gather input,
centre, whiten, store, update statistics, add each route's dot product, trace
update, TD weight update with L2-ball clamp, terminal settle at death — looped
over `0..SENSE_COUNT` in declared order. Float expression order is preserved
per strategy (the closed-form 2×2 and Jacobi code paths keep today's
operations), so smell and vision stay bit-identical. The cooperative
schedules (`XAGENT_BRAIN_COOPERATIVE_WHITENING`, cooperative Jacobi, visual
pathway parallelization) refresh/prepare every `Jacobi` sense through the same
block.

### Refactor sequence (each step byte-identical, at most six files)

Many Rust modules splice `brain_passes.wgsl` by text anchor, so the WGSL
change is staged:

1. `sense-override-constants` — `vision_override_constants` emits the sense
   table as overrides; `common.wgsl` declares them (defaults = table);
   `GpuKernel::new_with_senses`. Nothing reads them yet.
2. `generic-sense-helpers` — `common.wgsl` gains helpers parameterised by sense
   index (closed-form whitening, Jacobi refresh, learning rate, route settle at
   death); the old named helpers (`whiten_scent`, `refresh_vision_whitening`,
   `vision_pathway_learning_rate`, `settle_turn_pathways_at_death`) become thin
   wrappers so every anchor keeps matching; death callers use the generic
   settle.
3. `generic-sense-block-sites` — the exploration, TD and trace sites of
   `brain_passes.wgsl` loop over senses; the production composer
   (`compose_brain_passes`), `context_gather.rs`, `brain_sections.rs` and the
   anchors in `whitening_validation.rs` / `whitening_storage_validation.rs`
   follow. Transitional: its lib-test check skips exactly the two
   composition tests `visual_pathway_composition_preserves_other_passes` and
   `jacobi_composition_keeps_other_helpers_and_scratch`, whose candidate
   fragments step 4 rewrites; the plan's `brain-lib-suite` runs them unskipped.
4. `generic-schedule-candidates` — the cooperative whitening (production),
   cooperative Jacobi, visual-pathway parallelization and isolated-refresh
   fragments and their validation modules become generic over `Jacobi` senses;
   every named alternate-schedule validation passes unweakened.
5. `remove-sense-specific-wgsl` — wrappers, the hand-written
   `O_SCENT_*`/`O_VISION_*` WGSL overrides and the now-unused WGSL pathway
   constants are deleted (the descriptor table in `senses.rs` becomes their
   single source); the source-identifier test lands; README updated.

Accepted naming exception: the composed cooperative-refresh entry keeps its
name `cooperative_refresh_vision_whitening(brain_base, tid)` because four
modules anchor on it; its body is generic over every `Jacobi` sense. INV-05's
source test covers `brain_passes.wgsl` and `common.wgsl`. Descriptor kind
codes (input kind, whitening kind, learning-rate source) are methods on the
`senses.rs` enums, so the override emitter in `gpu_kernel.rs` never changes
when a sense or input kind is added.

### Parity method

Task 0101 adds `crates/xagent-sandbox/tests/sense_block_parity.rs` and, with
shell, records on the pre-refactor tree (plan 0002 merged, nothing of this plan
changed) `crates/xagent-sandbox/tests/fixtures/sense_block_parity.golden`
(FNV-1a 64 hex digests, one named line per scenario) and
`crates/xagent-sandbox/tests/fixtures/sense_block_champion.db` (the governor
database written by the seeded headless scenario). Scenarios: seeded 10-agent
GPU trajectories (cortex off, cortex on), the seeded headless evolution run
(`headless::run_headless`, 2 generations), replay of the committed champion
database's champion brain, and the CPU steering-population draw. Record mode
is `XAGENT_RECORD_SENSE_PARITY=1`; compare mode is the default. GPU scenarios
are `#[ignore]` and fail (not skip) without an adapter; the CPU scenario is a
normal test. The golden is host-specific; only a task whose verdict changes
default behaviour (0402 landing) may re-record it, and must say so.

### Investigations

- **Pooling (`pooling-study`, then `learned-pooling-port`)**: `crates/xagent-sandbox/src/pooling_study.rs` (fit
  library, unit-tested on synthetic data) and
  `crates/xagent-sandbox/examples/pooling_study.rs` (runner: seeded fresh-brain
  free lives through plan 0001's harness API, seeds 5–8, 10 agents, 300,000
  ticks, logging per brain tick the 32 adapted column means plus the harness's
  per-tick fields). Arms scored on the same logs: hemifields whitened learned
  (the reference), 32 columns whitened learned, and learned pooling
  (credit-trained pooling weights from columns to 2 × 4 pooled channels, and
  an Oja/PCA-learned pooling to 8 components). Score: correlation of the
  policy's turn with the food side over the last 30% of life, food in view.
  Port iff the best learned-pooling arm's mean ≥ the hemifield arm's mean on
  the same logs; the port is a new `SenseInput` variant behind a default-off
  `BrainConfig` flag whose extra state slots enter the layout only when the
  flag is on, so the default stride and parity hold. The decision record
  states `Verdict: PORTED` or `Verdict: NOT PORTED` on its own line.
  `learned-pooling-port` applies the verdict: PORTED → implement the winning
  arm as that input kind with its GPU test; NOT PORTED → no source change, and
  its check proves the recorded verdict and the unchanged default (parity).
- **Modulators (`modulator-flags`, `modulator-study`)**: `BrainConfig::klinotaxis_turn_gain_enabled`
  and `BrainConfig::motor_fatigue_enabled` (serde default `true`, not mutated,
  packed into new `CFG_*` slots). Off makes the factor exactly `1.0` in both
  the executed command and the credited noise. Arms: control, klinotaxis off,
  fatigue off; within-life metrics from paired fresh-brain free lives (seeds
  5–8, 10 agents, 300,000 ticks, plan 0001 harness) and the per-seed
  80-generation reference (plan 0002's procedure: `--dump-config` default with
  `governor.tick_budget = 40000`, `--generations 80`, seeds 5–8, arm and control
  at the same commit). Land a modulator's removal iff it is not worse, judged
  as plan 0002 judges its routes: arm minus control mean meals per agent ≥
  −SE, mean deaths per agent ≤ +SE (SE the Welch standard error of the 40-vs-40
  difference), pooled share of sightings ending in a meal ≥ control − SE, **and**
  in every seed the arm's mean `composite_fitness` over generations 60–78 is at
  least the control's. Each modulator is judged against the control alone. Landed = the flag's default flipped to `false` in `BrainConfig` and
  every preset (the modulator stays selectable by setting the flag), the parity
  golden re-recorded because default behaviour changed, and the reason
  recorded. Reverted = the default stays `true`, so default behaviour and the
  golden are unchanged; the flag stays as the measurement switch. Flags never
  move `brain_state` slots, so stored databases load either way.

## Invariants

| id | invariant | guards | enforced at |
|---|---|---|---|
| INV-01 | With default configuration, smell and vision re-expressed through the block produce byte-identical seeded GPU trajectories (cortex off and on), a byte-identical seeded headless run, and an identical replay of the stored champion database, against the golden recorded before any change of this plan. | refactor drift, float reordering, lost death/refresh paths | `crates/xagent-sandbox/tests/sense_block_parity.rs` (`parity-golden` records; every later code task runs it) |
| INV-02 | Every sense slot offset derives from the descriptor table through `BrainLayout`; derived smell/vision/route-block offsets equal their pre-plan values, `BRAIN_STRIDE` and `FIXED_TAIL_SIZE` are unchanged, and a newly declared sense appends without moving an existing slot. | DB incompatibility, hardcoded offsets | `crates/xagent-brain/src/buffers.rs` tests (`derived-sense-layout`); `crates/xagent-brain/tests/sense_block_extension.rs` |
| INV-03 | Smell and visual turn weights are mutated exactly as before (same RNG draw order and kick); every other sense/route weight, including any newly declared sense, is learned-only: zero at birth and never touched by mutation. | heritability drift | `crates/xagent-sandbox/src/agent/mod.rs` tests and the CPU parity scenario (`descriptor-driven-mutation`) |
| INV-04 | The descriptor declares a route only if plan 0002 adopted it: turn always, forward iff the forward-route record and default say adopted, critic iff the critic-route record and default say adopted. | wiring a failed route | `crates/xagent-brain/src/senses.rs` tests + `0201-ROUTE-WIRING.md` (`sense-descriptor`) |
| INV-05 | `brain_passes.wgsl` and `common.wgsl` contain no per-sense pathway code: none of the hand-written per-sense slot overrides (`O_SCENT_*`, `O_VISION_PATHWAY_*`, `O_VISION_TURN_WEIGHTS`, `O_TRACE_SCENT`, `O_TRACE_VISION`), per-sense pathway rate/refresh/variance constants (`SCENT_*` / `VISION_PATHWAY_*` whitening constants) or per-sense pathway helpers (`whiten_scent`, `vision_pathway_step`, `refresh_vision_whitening`, `vision_pathway_learning_rate`, `settle_turn_pathways_at_death`); names that describe sensor hardware (vision grid, scent field, sensory genes) or an input kind (hemifield pooling) may remain; per-sense parameters come only from override constants generated from `BrainLayout`; a sense declared only in a test composes, starts at zero and learns with no WGSL edit. | new code path per sense | source and override-default tests in `gpu_kernel.rs` (`sense-override-constants`, `remove-sense-specific-wgsl`); `crates/xagent-brain/tests/sense_block_extension.rs` (`declared-sense-extension-test`) |
| INV-06 | Learned pooling is ported only if its offline score reaches the hemifield score on the same logs; the port, if any, is default-off, so vision's default input stays the hemifields. | adopting a worse input | `crates/xagent-sandbox/src/pooling_study.rs` tests + `0301-POOLING-DECISION.md` (`pooling-study`); verdict check (`learned-pooling-port`) |
| INV-07 | Each modulator flag defaults to today's behaviour until its verdict; flag off makes its factor exactly 1.0 in the executed motor command and in the credited noise; the flags are not heritable and old configs deserialize to the default. | silent behaviour change, credit/execution mismatch | `crates/xagent-sandbox/tests/modulator_flags.rs`, `crates/xagent-shared/src/config.rs` tests (`modulator-flags`) |
| INV-08 | Each modulator is landed or reverted by the recorded rule, with the per-seed within-life and 80-generation numbers recorded; the parity golden is re-recorded only if a landing changed default behaviour. | unrecorded or rule-free verdicts | `0402-MODULATOR-DECISION.md` + `crates/xagent-sandbox/tests/modulator_flags.rs` (`modulator-study`) |
| INV-09 | Every alternate brain schedule (cooperative whitening, cooperative Jacobi, visual-pathway parallelization, isolated and workgroup whitening refresh, context gather) stays complete-state identical to the serial production block for every `Jacobi` sense. | schedule drift hidden by the refactor | named GPU validations in `crates/xagent-brain/src/gpu_kernel/*_validation.rs` (`generic-sense-block-sites`, `generic-schedule-candidates`) |

## State

| state | owner | lifecycle | concurrency |
|---|---|---|---|
| Per-sense `brain_state` slots (mean, covariance, whitening, whitened input) | GPU kernel, thread 0 of the agent's brain workgroup (cooperative schedules: all threads behind barriers) | survive death; initial variance written by `init_brain_state_for` from the descriptor | one workgroup per agent; no cross-agent access |
| Per-sense route weights | GPU kernel (TD update, terminal settle); CPU mutation for `Mutated` routes between generations | turn weights of smell/vision inherited through the birth brain and mutated; all other route weights zero at birth, learned within the life | CPU writes only while no dispatch is in flight (existing `write_agent_state` contract) |
| Per-sense route traces | GPU kernel | episodic, zeroed at death after the terminal settle | as above |
| `DECLARED_SENSES` table | `senses.rs` | compile-time constant | immutable |
| Parity golden and champion DB fixture | task 0101 (0402 may re-record the golden) | committed, read-only in tests; the DB is copied to a temp dir before opening | tests run in one process; no shared writable state |
| Modulator flags | `BrainConfig` → config uniform | population-wide per run, not mutated | uploaded with the config uniform |
| Study logs and outputs | tasks 0301/0402 (architect shell) | scratch under the job directory; only the decision records are durable | at most two GPU study jobs at once, after a memory check |

## Untrusted inputs

| input | source | bounds |
|---|---|---|
| Stored champion brain blobs | governor SQLite `node_brain` | `decode_brain_state` validates lengths; the parity test asserts the decoded length equals `BrainLayout::default().brain_stride` before writing it to the kernel |
| Config JSON | `--config` / stored `config_json` | new flags use serde defaults (`true`); unknown fields ignored as today |
| Free-life logs read by the pooling fit | files written by the 0301 runner | record width checked against the header (32 column means + harness fields); truncated or mismatched records rejected with an error, not skipped silently |
| Sense descriptor table | source | compile-time assertions: `DECLARED_SENSES.len() <= MAX_SENSE_SLOTS`, `channels <= MAX_SENSE_CHANNELS`, `ClosedForm2` only with 2 channels |

## Feasibility

| task | constraints and sources | compatibility witness | preflight check |
|---|---|---|---|
| parity-golden | `run_headless` (`headless.rs`), `champion_brain`/`store_champion_brain` (`governor.rs`), `steering_population`/`fresh_brain_state` (`agent/mod.rs`) and the kernel API used by `integration.rs` exist at baseline; a seeded 10-agent run must reproduce byte for byte on the host | `execution/parity_reproducibility_probe.rs` seeded as the future test file runs both GPU scenarios twice and the CPU draw twice and asserts byte equality | `pf-task-parity-golden` |
| sense-descriptor | smell constants in `common.wgsl` (`SCENT_FEATURE_OFFSET 25`, `SCENT_WHITENING_RATE 0.01`, `VISION_PATHWAY_RATE 0.01`, `VISION_WHITENING_REFRESH 20`) and `buffers.rs` (`SCENT_INITIAL_VARIANCE`, `VISION_PATHWAY_INITIAL_VARIANCE`, `O_VISION_PLASTICITY`); plan 0002 verdict records and `DEFAULT_*` constants | the table restates existing constants; pure data plus compile-time assertions | `pf-task-sense-descriptor` |
| derived-sense-layout | offset cascade `O_SCENT_MEAN = O_RECENT_NORMS + RECENT_CAP` … `O_TRACE_VISION` in `buffers.rs`; smell core 11 slots, vision core 160 slots; plan 0002 inserts its 1321-slot route block between `O_TRACE_VISION + 8` and `O_SENSORY_MEAN` | the cascade is a fixed sum reproducible by the per-sense rule (2+3+2+2+2, 8+64+64+8+8+8) followed by the route block; existing layout tests green | `pf-task-derived-sense-layout` |
| descriptor-driven-mutation | `mutate_steering_weights` iterates `(scent, 2)` then `(vision, 8)` with one draw per weight and `rescale_into_ball` | iterating `Mutated` routes in declared order reproduces the draw sequence; CPU parity scenario detects any change | `heritability-source-anchors` |
| sense-override-constants | `vision_override_constants(layout)` already feeds every pipeline and validation module; WGSL `override` declarations with defaults (`common.wgsl`) | adding overrides is additive; a default unused override changes no code generation | `pf-task-sense-override-constants` |
| generic-sense-helpers | `whiten_scent`, `refresh_vision_whitening`, `vision_pathway_learning_rate`, `settle_turn_pathways_at_death` in `common.wgsl`; callers in `phase_death.wgsl`, `kernel_tick.wgsl` | wrappers keep every name and call text that Rust anchors search | `pf-task-generic-sense-helpers` |
| generic-sense-block-sites | exploration/TD/trace sites in `brain_passes.wgsl`; `REFRESH_BLOCK` in `gpu_kernel.rs`; anchors in `context_gather.rs`, `brain_sections.rs`, `whitening_validation.rs`, `whitening_storage_validation.rs` | the named validations pass on baseline, so the same names isolate regressions; two composition tests are skipped transitionally (see Refactor sequence) | `pf-task-generic-sense-block-sites` |
| generic-schedule-candidates | candidate fragments `brain_cooperative_visual_pathway.wgsl`, `brain_cooperative_jacobi.wgsl`, `whitening_refresh.wgsl`, production `brain_cooperative_whitening.wgsl`; modules `visual_pathway_validation.rs`, `cooperative_jacobi_validation.rs` | the same named validations that pass on baseline must pass unskipped | `whitening-validations-baseline` |
| remove-sense-specific-wgsl | hand-written `override O_SCENT_*`/`O_VISION_*` cascade in `common.wgsl`; README brain-state section | deletion after every reader moved to generated overrides; source test proves absence | `pathway-source-anchors` |
| declared-sense-extension-test | `GpuKernel::new` sizes buffers from `BrainLayout::from_config`; integration tests in `crates/xagent-brain/tests` drive the kernel | a test-only layout with one extra feature sense goes through `GpuKernel::new_with_senses` | `pf-task-declared-sense-extension-test` |
| modulator-flags | klinotaxis and fatigue factors in `brain_passes.wgsl`; `CONFIG_SIZE = 16` fully used at baseline (plans 0001/0002 extend it; take the next free slots); `BrainConfig` bool-flag pattern (`danger_percept_enabled`); the full `BrainConfig` literals in `agent/mod.rs` (mutation, crossover) must copy the new flags | flags take the next free `CFG_*` slots, growing `CONFIG_SIZE` by one vec4 if needed (existing alignment tests) | `pf-task-modulator-flags` |
| modulator-study | headless evolution CLI (`--no-render --seed --generations --config --db` in `main.rs`), plan 0001 free-life harness, investigation reference numbers | architect shell runs the paired arms; only the decision record and the default flip/revert are committed | `pf-task-modulator-study` |
| pooling-study | `read_brain_scratch_blocking` exposes adapted features per brain tick; `vision_hemifields` defines the reference pooling; reference scores in the investigation | the runner reads features through the public kernel API; the fit is CPU-only | `pf-task-pooling-study` |
| learned-pooling-port | the generic block's input kinds (`SenseInput`), layout rule and config-flag pattern | a new input kind plus flag-gated slots reuse the block; NOT PORTED needs no change | `pf-task-learned-pooling-port` |
