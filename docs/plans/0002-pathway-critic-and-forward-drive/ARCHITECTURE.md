# Architecture: pathway critic and forward drive

## Baseline facts (verified against source at the plan baseline)

- The critic is `V = O_VALUE_BIAS + O_VALUE_WEIGHTS · s_memory_key` (the
  centred 128-d encoding) in `coop_predict_and_act`
  (`crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`). Its step is
  `CRITIC_LEARNING_RATE / (1 + |previous centred encoding|²) · δ · trace`, its
  L2 ball is `MAX_WEIGHT_NORM`, and the store pass replays one settled
  recent-experience moment per brain tick from `O_RECENT_KEYS`.
- The value reduction runs before thread 0 whitens this tick's scent
  (`O_SCENT_WHITENED`) and visual hemifields (`vision_pathway_step` writes
  `O_VISION_PATHWAY_INPUT`); preflight `witness-value-before-whitening`.
- The forward command is `O_ACT_BIASES[0] + O_ACTION_FORWARD_WEIGHTS · s_encoded`
  with no pathway. Forward noise is a fresh uniform draw every brain tick (no
  persistence), so its fresh innovation as executed is exactly
  `s_explore[0u] = noise_forward · exploration_rate · fatigue_factor`
  (preflight `witness-forward-noise-fresh`).
- The turn pathways' rule: trace `z ← γλ·z + s_explore[2u] · x / (1 + |x|²)`
  per sense (smell: 2 inputs, vision: 8 inputs); weights
  `+= rate · δ · z` (smell `ACTION_WEIGHT_LEARNING_RATE`, vision
  `vision_pathway_learning_rate`), each sense in its own `MAX_WEIGHT_NORM` ball;
  at death `settle_turn_pathways_at_death` applies `TERMINAL_DEATH_TD_ERROR`
  through the traces, then zeroes them.
- `CONFIG_SIZE` was 16 before plan 0001; it is 24 with slots 16-20 used at HEAD (preflight
  `witness-config-size`); plan 0001 may add lesion slots before this plan runs.
- The brain-state fixed tail ends with `O_TRACE_VISION`, then
  `O_SENSORY_MEAN` (layout-sized), in both `buffers.rs` and `common.wgsl`.
  `feature_count_for_brain_stride` infers the layout from a blob's length, and
  `inherited_brain_fits` skips any champion whose length differs.
- Every CLI mode receives the `FullConfig` from `resolve_config` (`--config`),
  and no config type denies unknown fields.
- `brain_passes.wgsl` is also consumed as text. `compose_brain_passes(true)`
  in `crates/xagent-brain/src/gpu_kernel.rs` (production, enabled by
  `XAGENT_BRAIN_COOPERATIVE_WHITENING=1`) asserts that its `REFRESH_BLOCK`
  (inside `vision_pathway_step`) and `EXPLORATION_BOUNDARY` markers appear
  exactly once, removes the refresh block and inserts
  `cooperative_refresh_vision_whitening` just before the exploration boundary.
  Non-ignored lib tests in `crates/xagent-brain/src/gpu_kernel/*.rs` (for
  example `visual_pathway_composition_preserves_other_passes`, which asserts
  the exact `vision_pathway_step` call text and the order refresh <
  exploration < call) match further marker texts. CI does not run
  `cargo test -p xagent-brain --lib`; this plan does (preflight
  `witness-composition-markers`).
- The value pass (`coop_predict_and_act`) and the store pass
  (`coop_learn_and_store`) run in the same entry-point invocation in every
  brain entry (`brain_tick.wgsl`, `brain_inner.wgsl`, `exact_tiled_brain.wgsl`,
  `phase_brain_tail_from_scratch.wgsl`), so a module-scope `var<private>`
  variable written by thread 0 in the first is readable by thread 0 in the
  second (no workgroup storage is needed or allowed for
  `s_critic_pathway_input`; see D5 and AUTHORIZATION.md).

## Decisions

### D1. Pathway input vector

`PATHWAY_INPUT_COUNT = SCENT_CHANNELS + VISION_PATHWAY_INPUTS = 10`. The vector
`x` is `[O_SCENT_WHITENED[0..2], O_VISION_PATHWAY_INPUT[0..8]]` of the current
brain tick. With the visual cortex on, the visual part is zero (unchanged
baseline behaviour).

### D2. Flags (route-flags; config slots in brain-slot-layout)

In `BrainConfig` (`crates/xagent-shared/src/config.rs`, task route-flags):

- `critic_input: CriticInput` with `#[serde(default = ...)]`, enum
  `CriticInput { Encoding, Pathway, EncodingAndPathway }`, serde
  `rename_all = "snake_case"` (`"encoding"`, `"pathway"`,
  `"encoding_and_pathway"`). Default from the documented constant
  `DEFAULT_CRITIC_INPUT = CriticInput::Encoding`.
- `forward_pathway_enabled: bool`, default from
  `DEFAULT_FORWARD_PATHWAY_ENABLED = false`.
- Both locked per batch, never perturbed by `mutate_config`, never shown as
  genes. Task brain-slot-layout packs them in `build_config_for` into two new
  config slots `CFG_CRITIC_INPUT` (0.0 / 1.0 / 2.0) and
  `CFG_FORWARD_PATHWAY_ENABLED` (0.0 / 1.0) at the next two free indices after
  the highest `CFG_*` index present; `CONFIG_SIZE` grows to the next multiple of 4
  if needed, and the WGSL `brain_config` uniform grows with it. Mirrored by the
  same names in `common.wgsl`. Plan 0001's architecture grows the uniform to
  `CONFIG_SIZE = 24` with slots 16-20 used and 21-23 padding, so the expected
  placement is `CFG_CRITIC_INPUT = 21`, `CFG_FORWARD_PATHWAY_ENABLED = 22`,
  `CONFIG_SIZE` unchanged; the worker confirms against the integrated code.
- `crates/xagent-sandbox/src/agent/mod.rs` builds two full `BrainConfig`
  literals (mutation and crossover); both copy the flags unchanged from the
  parent (crossover: from `a`), as they do for the other per-batch gates.

### D3. Slot block (brain-slot-layout)

One contiguous block inserted after `O_TRACE_VISION` and before
`O_SENSORY_MEAN`, in this order (Rust and WGSL identical):

| Offset | Size | Lifecycle |
|---|---|---|
| `O_FORWARD_PATHWAY_WEIGHTS` | 10 | learned, zero at birth, survives death |
| `O_TRACE_FORWARD_PATHWAY` | 10 | episodic, zeroed at death |
| `O_CRITIC_PATHWAY_WEIGHTS` | 10 | learned, zero at birth, survives death |
| `O_TRACE_CRITIC_PATHWAY` | 10 | episodic, zeroed at death |
| `O_PREV_PATHWAY_INPUT_NORM` | 1 | squared size of the previous brain tick's critic pathway input `c` (D5, D6); survives death |
| `O_RECENT_PATHWAY_KEYS` | `RECENT_CAP × 10` = 1280 | ring keys, slot-major, survive death like `O_RECENT_KEYS` |

`PATHWAY_ROUTE_SLOT_COUNT = 1321`. `1321 mod (ENCODED_DIMENSION + 1) = 31`, so
no baseline-length blob is also a current-length blob (preflight
`witness-slot-block-unambiguous`).

### D4. Legacy blob upgrade (brain-slot-layout; decode in legacy-decode-and-birth-state)

`buffers.rs` gains `upgrade_legacy_brain_state(brain_state: &[f32]) ->
Option<Vec<f32>>`: a length that matches a current layout is returned
unchanged; a length that matches a layout with
`FIXED_TAIL_SIZE − PATHWAY_ROUTE_SLOT_COUNT` gets `PATHWAY_ROUTE_SLOT_COUNT`
zeros inserted at that legacy layout's `O_SENSORY_MEAN` offset, every other
value preserved; any other length returns `None`. `decode_brain_state` in
`governor.rs` applies it, so resume, inheritance and the free-life champion
loader all see current-length brains.

### D5. Kernel ordering (kernel-routes): nothing moves

No existing kernel text moves. The whitening of scent and of the visual
hemifields stays inside the thread-0 exploration block, after the value
reduction, and `vision_pathway_step`, its refresh block, its call text, the
exploration boundary comment and every other text that
`compose_brain_passes` or a `gpu_kernel` validation test matches keep their
text, count and order. Moving the whitening ahead of the value (the rejected
alternative) would put the production cooperative refresh after the tick's
whitening and break the text-validation tests outside this plan's ownership.

Consequences, fixed here:

- The critic's pathway input at brain tick `t` is `c_t`: the ten values held
  in `O_SCENT_WHITENED[0..2]` and `O_VISION_PATHWAY_INPUT[0..8]` when the value
  is formed, which are the whitened inputs `x_{t−1}` stored by the previous
  brain tick (zero at birth). Thread 0 copies `c_t` into a new module-scope `var<private>` array
  `s_critic_pathway_input` (length `PATHWAY_INPUT_COUNT`, literal `10` in the
  declaration) before whitening overwrites the slots; thread 0 of the same
  invocation reads it for the pathway dot, normaliser and trace, and its store
  pass writes it to the ring. Every access is inside a `tid == 0u` block, so no
  workgroup memory is added and the fixed shared-memory estimates hold.
- The forward pathway reads this tick's `x_t`, which is in those slots after
  the existing `vision_pathway_step` call in the exploration block.
- New code is inserted only inside uniform branches on the two config slots.
  Edits to existing lines are limited to those D6 needs (the critic's
  normaliser reads `S_CRITIC_NORMALIZER`, and the mode gates that keep the
  128-d critic value dot, weight step, trace, replay and death step unread and
  unwritten in pathway mode) and must not alter any matched marker; `cargo test -p xagent-brain --lib` (INV-13) proves the composition
  and the validation tests still hold. Routes-off arithmetic is unchanged, so
  routes-off behaviour is bit-identical (INV-1) on the scalar path and the
  cooperative composition still applies unchanged.

### D6. Critic input route (kernel-routes)

Mode `m = CFG_CRITIC_INPUT`; `c` is the critic pathway input of D5 (the
previous brain tick's whitened inputs). `V = bias + [m ≠ pathway] ·
O_VALUE_WEIGHTS · s_memory_key + [m ≠ encoding] · O_CRITIC_PATHWAY_WEIGHTS ·
c`. TD error, bias update and clamp unchanged. Critic step normaliser
`S_CRITIC_NORMALIZER = 1 / (1 + [m ≠ pathway]·|previous centred encoding|² +
[m ≠ encoding]·O_PREV_PATHWAY_INPUT_NORM)`; in encoding mode it equals
`S_STEP_NORMALIZER` bit for bit, and the turn channel keeps
`S_STEP_NORMALIZER` in every mode. `O_PREV_PATHWAY_INPUT_NORM` holds `|c|²`
of the previous brain tick, rewritten after use. Critic pathway weights
`+= CRITIC_LEARNING_RATE · S_CRITIC_NORMALIZER · δ · O_TRACE_CRITIC_PATHWAY`,
kept in their own `MAX_WEIGHT_NORM` ball; trace `z ← γλ·z + c`. In pathway
mode the 128-d critic weights and trace are neither read nor written. Replay:
the store pass writes `s_critic_pathway_input` (this tick's `c`) to
`O_RECENT_PATHWAY_KEYS` at the same `store_slot` as `O_RECENT_KEYS` (modes
other than encoding), and the replayed value and step use the included input
sets with normaliser `1 + included squared key sizes`. Death: the terminal critic step reaches the
included weight sets, then both critic traces are zeroed. The plan 0001
critic lesion (value held at its bias) holds the whole value, pathway part
included. In encoding mode none of the new critic slots is read or written.

### D7. Forward pathway route (kernel-routes)

When `CFG_FORWARD_PATHWAY_ENABLED = 1`: forward pre-tanh command adds
`O_FORWARD_PATHWAY_WEIGHTS · x` (before the memory blend), where `x` is this
tick's whitened input read from the slots after the existing
`vision_pathway_step` call (D5). The weight step uses this tick's `δ` and the
trace as it stood before this tick's trace update, in the TD block beside the
smell turn-weight step but outside `steering_weights_learn()`. Trace per sense:
smell `z ← γλ·z + s_explore[0u] · x_s / (1 + |x_s|²)`, vision
`z ← γλ·z + s_explore[0u] · x_v / (1 + |x_v|²)`. Weights `+= rate · δ · z`
with smell at `ACTION_WEIGHT_LEARNING_RATE` and vision at
`vision_pathway_learning_rate(brain_base)`, each sense in its own
`MAX_WEIGHT_NORM` ball. Death: `TERMINAL_DEATH_TD_ERROR` through the traces,
then traces zeroed. Not covered by the steering freeze (it is not a turn
weight) and not by the 128-d forward-weight lesion. When the flag is 0, none of
the forward pathway slots is read or written.

### D8. Offline return fit (return-fit-tool, critic-input-decision)

Per agent life from the plan 0001 per-tick log: target return
`G_t = Σ_k γ^k r_{t+k+1}` with `γ = TD_DISCOUNT = 0.97`, cut at each death with
the terminal outcome `TERMINAL_DEATH_TD_ERROR = −1` as the last reward; the
final 300 brain ticks of the log are dropped (truncated returns). Input sets:
`pathway` (10), `encoding` (128, as logged), `encoding_and_pathway` (138), each
with an intercept. The pathway columns of row `t` are the logged whitened
pathway inputs of row `t − 1` of the same agent, which is what the critic
reads (D5); the first row of each agent's log and the first row after each
death are dropped from all three sets, so every set fits the same rows. Ridge least squares, λ chosen from
`{1e-3, 1e-2, 1e-1, 1, 10}` on an inner split of the training blocks. Blocks of
1000 brain ticks; a block is held out when `splitmix64(seed, agent, block) mod
10 < 3`. Scores on held-out ticks: R², in-view minus not-in-view mean fitted
value, centred minus off-axis mean fitted value, using the same in-view and
centred predicates as the plan 0001 critic gap metrics. Winner rule: among sets
whose pooled in-view gap and centred gap are both positive, the highest pooled
mean held-out R²; `encoding_and_pathway` wins over `pathway` only if its paired
per-agent R² advantage exceeds one standard error; if no set has both gaps
positive, the winner is `encoding` (route stays off).

### D9. Gates (route-gate-tool, route-gate-decisions)

Arms: fresh birth brains, seeds 5–8, 10 agents each (40 agents), 300,000
ticks, `--dump-config` default config, paired by seed with a control arm of
current code (both flags off). The critic arm sets `critic_input` to the
critic-input-decision winner (no critic arm when the winner is `encoding`); the forward arm sets
`forward_pathway_enabled`. Each route arm differs from control in its own flag
only.

- G1 (forward only), conditional credit: per agent,
  `C = Σ_k g_k · m_k`, where `g_k` is the life mean of `δ_t · e_{t−1,k}`,
  `e` the forward pathway trace rebuilt from the log (executed forward noise
  times the whitened input, normalised per sense as in D7, decayed by γλ,
  reset at deaths), `δ_t` the logged TD error of the brain tick, and `m_k` the
  mean of input `k` on ticks with food in view and centred (plan 0001's
  `FOOD_IN_VIEW_DISTANCE`, `VIEW_HALF_ANGLE`, `CENTRED_BEARING`) minus its life
  mean. Pass: `C > 0` in at least 21 of 40 agents and pooled mean `C > 0`.
  Unconditional forward credit is reported, never gated.
- G2, conversion: pooled share of sightings ending in a meal higher than
  control, and higher in at least 3 of 4 seeds.
- G3, not worse: route minus control mean meals per agent `≥ −SE` and mean
  deaths per agent `≤ +SE`, SE the Welch standard error of the 40-vs-40
  difference.
- G4, 80 generations: only if G1–G3 (G2–G3 for the critic) pass. Headless
  `--generations 80`, `--dump-config` default with `governor.tick_budget =
  40000`, seeds 5–8, route on and control off at the same commit; pass when in
  every seed the route's mean `composite_fitness` over generations 60–78 is at
  least the control's.

A route is adopted (its `DEFAULT_*` constant flipped in `config.rs`) only when
all its gates pass; otherwise the record states the failing gate and the
default stays off. `verify` re-derives each verdict from the recorded numbers
and checks the code default against it.

## Interfaces consumed from plan 0001

From plan 0001's architecture (exact names confirmed by workers against the
integrated code; this plan never edits plan 0001 files):

- `crates/xagent-sandbox/src/free_life/` (`metrics.rs`, `tick_log.rs`): the
  per-agent `AgentMetrics` (serde; `whole` and `late` `WindowMetrics` with
  meals, deaths, sightings, `sighting_meal_share`, `forward_credit[10]`, value
  gaps) collected by the runner's `FreeLifeReport` (in `headless.rs`, `agents:
  Vec<AgentMetrics>`), the
  tick log (JSON header line naming columns, then little-endian `f32` rows of
  the brain-tick probe: centred encoding 128, whitened pathway inputs 10,
  reward, value, TD error, executed forward noise, executed turn noise and
  innovation, the pathway turn outputs, plus ground truth), and the named
  constants `FOOD_IN_VIEW_DISTANCE = 5.5`, `VIEW_HALF_ANGLE = π/4`,
  `CENTRED_BEARING = 0.15`, `OFF_AXIS_BEARING = 0.45`.
- The free-life runner in `headless.rs` and its CLI flags in `main.rs`
  (config from `--config`, seed from `--seed`).
- The lesion switches (`CFG_LESION_*`, `TrunkLesions`) and the probe slot.
- The read-only champion loader in `governor.rs`, which goes through
  `decode_brain_state` (so D4's upgrade reaches it).

Plan 0001's own `forward_credit[10]` normalises by the full ten-input size;
this plan's G1 rebuilds the trace per sense as D7 does, and reports plan
0001's metric beside it.

## Invariants

| id | invariant | guards | enforced at |
|---|---|---|---|
| INV-1 | With `critic_input = encoding` and `forward_pathway_enabled = false`, a seeded free life's per-agent metrics (the report's `agents` array) and tick-log rows (payload after the header line) hash to the fingerprint recorded at the plan baseline on the recorded GPU adapter; on any other adapter the replay prints a skip naming both adapters and passes, and the digest comparison itself is unit-tested against altered fixture values. | Routes-off behaviour is unchanged by the layout growth, the kernel routes and any adoption, without making CI's lavapipe run fail on a fingerprint from another adapter. | `crates/xagent-sandbox/tests/pathway_default_fingerprint.rs` (default-fingerprint), run again by brain-slot-layout, kernel-routes and route-gate-decisions |
| INV-2 | The slot block of D3 sits contiguously between `O_TRACE_VISION + VISION_PATHWAY_INPUTS` and `O_SENSORY_MEAN` with identical offsets in Rust and WGSL, `FIXED_TAIL_SIZE` grows by exactly `PATHWAY_ROUTE_SLOT_COUNT`, a fresh brain has the block zero, and `build_config_for` packs both flags into `CFG_CRITIC_INPUT` and `CFG_FORWARD_PATHWAY_ENABLED`. | Shader and host agree on every new slot and flag. | `buffers.rs` unit tests (brain-slot-layout) |
| INV-3 | A baseline-length brain blob upgrades and decodes to a current-length brain with the block zero and every other value preserved; current blobs are unchanged; other lengths are rejected. | Stored champion databases keep loading. | `buffers.rs` unit tests (brain-slot-layout), `crates/xagent-sandbox/tests/pathway_route_state.rs` (legacy-decode-and-birth-state) |
| INV-4 | Every new weight is zero at birth and the flags are inherited unchanged: brain mutation, config mutation and crossover never write the block and never alter either flag; with both routes off the kernel never writes the block. | Learned-only, zero-at-birth weights; flags locked per batch and not heritable. | `crates/xagent-sandbox/tests/pathway_route_state.rs` (legacy-decode-and-birth-state), `crates/xagent-sandbox/tests/pathway_routes.rs` (kernel-routes) |
| INV-5 | The flags deserialize to their `DEFAULT_*` constants when absent, round-trip through serde for every value, and reject unknown `critic_input` names. | Flags are default-off and parse safely. | `config.rs` unit tests (route-flags) |
| INV-6 | With the forward flag on, the forward traces, weight steps, per-sense balls, death settle and forward contribution follow D7 exactly. | The forward pathway uses the existing pathway rule with forward-noise credit. | `crates/xagent-sandbox/tests/pathway_routes.rs` (kernel-routes) |
| INV-7 | In each critic mode the value, normaliser, weight steps, replay and death settle follow D6, and encoding mode leaves the critic pathway slots untouched. | The critic reads exactly the selected input set. | `crates/xagent-sandbox/tests/pathway_routes.rs` (kernel-routes) |
| INV-8 | The return-fit tool computes returns, splits, ridge fits, R² and value gaps as D8 on synthetic data with known answers, and `verify` fails a record whose winner disagrees with the D8 rule. | The offline decision follows the agreed method. | `crates/xagent-sandbox/src/bin/pathway_return_fit.rs` unit tests (return-fit-tool) |
| INV-9 | The route-gate tool computes conditional credit and G1–G4 as D9 on synthetic data, and `verify` fails a record whose verdict disagrees with its numbers or whose route default disagrees with the verdict. | Gates are conditional-credit based and adoption follows the gate. | `crates/xagent-sandbox/src/bin/pathway_route_gate.rs` unit tests (route-gate-tool) |
| INV-10 | `docs/decisions/pathway-critic-inputs.json` passes `pathway_return_fit verify`. | The critic-input decision record matches its rule. | `pathway_return_fit verify` (critic-input-decision) |
| INV-11 | Both route records pass `pathway_route_gate verify`, so each default is on only if its gates passed. | A negative result is recorded and the route stays off. | `pathway_route_gate verify` (route-gate-decisions) |
| INV-12 | No source file names a plan, task or decision document. | Repository source-agnosticism rule. | `crates/xagent-sandbox/tests/contributing_guard.rs` (every code task) |
| INV-13 | Every text marker that `compose_brain_passes` and the `gpu_kernel` validation tests match in the kernel shaders keeps its text, count and order, so `compose_brain_passes(true)` still composes and `cargo test -p xagent-brain --lib` passes. | The production cooperative-whitening path and its scalar-parity validation stay valid after the routes land. | `crates/xagent-brain/src/gpu_kernel.rs` and `crates/xagent-brain/src/gpu_kernel/*.rs` lib tests, run by kernel-routes and at integration |
| INV-14 | Every `pathway_routes` test sets both route flags explicitly and never takes them from `BrainConfig::default()`. | Adopting a route default cannot break the route tests. | `crates/xagent-sandbox/tests/pathway_routes.rs` (kernel-routes), re-run by route-gate-decisions |

## State

| state | owner | lifecycle | concurrency |
|---|---|---|---|
| `critic_input`, `forward_pathway_enabled` | `BrainConfig`, written by config load only | set per batch from `--config`; copied unchanged to offspring | read-only uniform on the GPU |
| `O_FORWARD_PATHWAY_WEIGHTS`, `O_CRITIC_PATHWAY_WEIGHTS` | per-agent `brain_state`, written only by thread 0 of the agent's brain workgroup and by the death paths | zero at birth; learned through life; survive death; never mutated or inherited as non-zero | one workgroup per agent; reads after the existing `workgroupBarrier`s |
| `O_TRACE_FORWARD_PATHWAY`, `O_TRACE_CRITIC_PATHWAY` | same | zero at birth; accumulate each brain tick; zeroed at death after the terminal step | same |
| `O_PREV_PATHWAY_INPUT_NORM` | same | zero at birth; rewritten each brain tick at trace time | same |
| `O_RECENT_PATHWAY_KEYS` | same, store pass | slot written with the ring slot it shadows; survives death | written by threads `< PATHWAY_INPUT_COUNT` before the existing `storageBarrier` |
| `s_critic_pathway_input` (`var<private>`, `PATHWAY_INPUT_COUNT`) | `coop_predict_and_act` thread 0 | written at value time (critic modes other than encoding) from the stored whitened slots, read only by thread 0 of the same invocation (pathway dot, normaliser, trace, store-pass ring write); never persisted | single-invocation program order; no other thread reads it |
| Fingerprint fixture | default-fingerprint | recorded once at the plan baseline, read-only afterwards | test reads only |
| Decision JSON records | critic-input-decision, route-gate-decisions | written once by the measuring architect; read by `verify` | none |

## Untrusted inputs

| input | bound |
|---|---|
| Stored brain blobs (`node_brain`) | header and payload size checked by `decode_brain_state`; only current or exact baseline lengths accepted; others rejected and logged |
| `--config` JSON | serde-typed; `critic_input` accepts only the three names; unknown fields ignored as today |
| Per-tick logs and metrics files read by the tools | record count and width checked against the reader; non-finite values rejected with an error naming the file; a fit with an empty held-out or training set fails instead of reporting |
| Governor databases read by the gate tool | opened read-only with `busy_timeout`; generations 60–78 must all be present or the tool fails |
| Decision JSON read by `verify` | schema-checked; a missing field or unknown input set fails |

## Feasibility

| task | constraints and sources | compatibility witness | preflight check |
|---|---|---|---|
| default-fingerprint | Pinned config passes through `resolve_config` (`crates/xagent-sandbox/src/main.rs`); extra route keys are ignored until route-flags defines them because no config type denies unknown fields (`crates/xagent-shared/src/config.rs`); the adapter name comes from the `[GpuKernel] Adapter:` info log in `crates/xagent-brain/src/gpu_kernel.rs`; GPU available to host checks. | architect witness fails without an adapter (`gpu_adapter_present_when_required` under `XAGENT_REQUIRE_GPU=1`) and runs `--dump-config` | `preflight-default-fingerprint` |
| route-flags | No config type denies unknown fields, so new serde-defaulted fields keep old JSON loading (`config.rs`); two full `BrainConfig` literals in `agent/mod.rs` (mutation and crossover) must carry the new fields. | developer witness compiles the sandbox test targets | `preflight-route-flags` |
| brain-slot-layout | `CONFIG_SIZE` 24 at HEAD (`buffers.rs`, grown from 16 by plan 0001), slots 16-20 used, padding 21-23; block inserted before `O_SENSORY_MEAN` in both `buffers.rs` and `common.wgsl`; 1321 mod 129 ≠ 0. | developer witness compiles the sandbox test targets | `preflight-brain-slot-layout` |
| legacy-decode-and-birth-state | `decode_brain_state` is the single blob decode in `governor.rs`, used by `champion_brain` and plan 0001's read-only loader. | developer witness compiles the sandbox test targets | `witness-decode-brain-state` |
| kernel-routes | Value computed before whitening (`brain_passes.wgsl`), so the critic reads the previous tick's stored inputs and nothing moves (D5); composition markers in `gpu_kernel.rs` and the validation tests (`gpu_kernel/visual_pathway_validation.rs`); forward noise is fresh per tick; existing pathway trace rule; death settle helper in `common.wgsl`; `XAGENT_REQUIRE_GPU` test convention (`tests/integration.rs`). | developer witness compiles the sandbox test targets; constraint check `brain-lib-baseline` runs `cargo test -p xagent-brain --lib` green at the baseline | `preflight-kernel-routes` |
| return-fit-tool | `src/bin/*.rs` auto-discovered (no `autobins` in `crates/xagent-sandbox/Cargo.toml`); `TD_DISCOUNT = 0.97` (`common.wgsl`). | developer witness compiles the sandbox test targets | `preflight-return-fit-tool` |
| route-gate-tool | Bin auto-discovery; `rusqlite` 0.34 dependency; `node.generation` and `agent_result.composite_fitness` columns (`governor.rs`). | developer witness compiles the sandbox test targets | `preflight-route-gate-tool` |
| critic-input-decision | Logged free lives through the shared CLI config; returns use `TD_DISCOUNT`. | architect witness fails without an adapter (`gpu_adapter_present_when_required` under `XAGENT_REQUIRE_GPU=1`) and runs `--dump-config` | `preflight-critic-input-decision` |
| route-gate-decisions | Free lives and `--generations 80` headless runs through the shared CLI config; fitness by generation readable from the governor schema; 80-generation reference procedure in the investigation review; route tests pin flags explicitly (INV-14). | architect witness fails without an adapter (`gpu_adapter_present_when_required` under `XAGENT_REQUIRE_GPU=1`) and runs `--dump-config` | `preflight-route-gate-decisions` |
