# Architecture — Plan 0001 (deltas)

> Edits in `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/phase_death.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/common.wgsl`,
> `crates/xagent-sandbox/src/headless.rs`,
> `crates/xagent-sandbox/src/governor.rs` (gated),
> `crates/xagent-sandbox/tests/integration.rs`, plus experiment configs under
> `experiments/` and baseline records in
> `docs/superpowers/specs/2026-06-10-learning-baseline.md`. Line numbers are
> hints; locate by symbol.

## 0001 — Hazard observability and baseline

Today `read_agent_telemetry_blocking` (`gpu_kernel.rs:1888-1903`) copies the
full `sensory_stride` floats per agent and discards everything after the
vision colors, and the probe harness (`tests/integration.rs`, Learning Probe
Tests) has no danger arena.

Edits:

- **Telemetry tail** (`gpu_kernel.rs:193-210`, `:1888-1903`): `AgentTelemetry`
  gains

  ```rust
  /// Non-visual sensory tail exactly as packed by `phase_vision_senses`:
  /// [velocity(3), facing(3), angular(1), energy, integrity,
  ///  energy_delta, integrity_delta, touch(4 contacts × 4)].
  /// One vision batch stale, like all of `sensory_buffer`.
  pub sensory_non_visual: Vec<f32>,
  ```

  sliced from the already-read staging copy at
  `self.layout.vision_color_count + self.layout.vision_depth_count` — no new
  GPU readback, two lines plus the struct field. `AgentTelemetry` is
  constructed in exactly one place, so no other call sites change.

- **Hazard probe arena** (`tests/integration.rs`, new "Hazard Probe Tests"
  section after the Learning Probe Tests): a half-plane danger biome (biome
  column < 128 ⇔ x < 0 → `BIOME_DANGER`, the rest food-rich) built on the
  existing `ProbeArena` plumbing (`build_probe_arena`, `reset_bodies_with` —
  the all-danger override pattern already exists in
  `td_traces_bounded_across_deaths`). Agents are re-positioned to
  `x = −10`, spread along z, facing +Z (parallel to the boundary, so
  straight-line walking never exits on its own). Per episode each agent
  resolves to exactly one of: **exited** (first sampled tick with
  `P_POS_X ≥ 0`, alive, death count unchanged), **died** (`P_DEATH_COUNT`
  delta — respawn teleports to the safe side, so death must be checked
  before the position), or **timed out**. Episode cap 600 ticks cleanly
  separates the three at default damage rates (death by hazard at ~tick 200
  for a non-escaping agent); position sampling every 5 ticks bounds latency
  resolution and readback cost.

- **Pinned baseline**: run once, print
  `exit_fraction` / `mean_exit_latency` / `death_fraction`, then commit
  ±50%-relative assertion bands around the recorded values plus two
  structural assertions (outcomes don't double-count; the arena is actually
  hazardous). Same re-pin protocol as
  `learning_probe_mirrored_steering_is_chance`. The recorded numbers and the
  gate ("workstream 0002 must move exit latency or death fraction outside
  the bands to claim a behavioral win") land in the baseline spec.

## 0002 — Survival-signal grounding

### Terminal death TD update

Today `agent_death_respawn` (`kernel_tick.wgsl:283-392`) zeroes the three
trace vectors, the trace biases, and `O_PREV_VALUE` with no terminal
evaluation, restores full energy, and runs before `brain_tick_inner` in the
cycle — the dying transition never produces a TD error.

Edits (`common.wgsl` TD constants block; `kernel_tick.wgsl` and
`phase_death.wgsl` immediately above their trace-zeroing blocks):

```wgsl
// Terminal TD error applied through the dying episode's eligibility traces
// at the moment of death, before they are cleared for the next life. Death
// must be the single worst lesson the learner can receive, but never
// stronger than the per-transition bound that protects against artifacts.
const TERMINAL_DEATH_TD_ERROR: f32 = -MAX_TD_ERROR;
```

The update applies the standard TD(λ) step with `δ = TERMINAL_DEATH_TD_ERROR`
to all heads through their existing traces — value bias and weights via
`CRITIC_LEARNING_RATE` (× `TD_VECTOR_SCALE` per dimension), both actor biases
and weight vectors via `ACTION_WEIGHT_LEARNING_RATE` — then the existing
zeroing block runs unchanged. Properties that make this safe:

- `agent_death_respawn` is thread-0-only with no barriers; barrier
  uniformity is untouched.
- The weight L2-ball clamps run in the next brain tick's pass 6, bounding
  magnitudes.
- The post-respawn brain tick in the same cycle applies its δ through
  freshly zeroed traces, so the terminal kick is the *only* bias movement in
  the death cycle — which is what makes the kick exactly assertable in a
  test (preset traces → exact expected deltas).
- Deaths early in a life apply the kick through near-zero traces — a short
  life teaches little, which is correct.

### Hazard and terrain-edge touch contacts

Today `phase_vision_senses` (`phase_vision.wgsl:173-312`) zeroes the four
touch slots and fills them from food and agent scans only. The CPU reference
(`agent/senses.rs::detect_touch`) also emits `TOUCH_TERRAIN_EDGE` (inward
direction, closeness intensity, 3-unit range) and `TOUCH_HAZARD` (no planar
direction, fixed 0.5 intensity).

Edits (`common.wgsl` touch constants; `phase_vision.wgsl` touch section):

- New constant:

  ```wgsl
  // Hazard contacts have no meaningful planar direction (the hazard is the
  // terrain underfoot), so they carry a fixed mid-scale intensity instead
  // of a closeness value. Matches the CPU reference in agent/senses.rs.
  const TOUCH_HAZARD_INTENSITY: f32 = 0.5;
  ```

- **Hazard first**: before the food scan, `sample_biome(pos.x, pos.z) ==
  BIOME_DANGER` writes slot 0 as `[0, 0, TOUCH_HAZARD_INTENSITY,
  TOUCH_HAZARD/4]` and starts `touch_count` at 1 — present-moment damage can
  never be evicted by lower-stakes contacts when the four slots fill (in
  danger biomes there is no food, but agent contacts are possible).
- **Edges last**: after the agent scan, four wall checks against
  `wc_f32(WC_WORLD_HALF_BOUND)` with `TOUCH_EDGE_RANGE`, direction pointing
  inward, intensity `1 − distance/range`, tag `TOUCH_TERRAIN_EDGE/4`. The
  per-wall distance/direction tables use `var` arrays (naga requires a
  mutable binding for dynamic indexing).

`phase_vision_senses` is one-thread-per-agent with no barriers — no
uniformity concerns. The feature layout does not change (the slots already
exist; only their population does).

### Same-cycle interoception

Today `coop_feature_extract` (`brain_passes.wgsl:70-109`) reads all 25
non-visual features from the batch-lagged `sensory_buffer`, including the
four interoception values — while `coop_habituate_homeo`
(`brain_passes.wgsl:148-183`) already reads energy/integrity same-cycle from
`physics_state` for the reward path. The state the policy conditions on lags
the reward it is blamed for.

Edit (`brain_passes.wgsl`, thread-0 block of `coop_feature_extract`): the
four interoception slots are populated from `physics_state` —
`P_ENERGY / max(P_MAX_ENERGY, 1e-6)`, `P_INTEGRITY / max(P_MAX_INTEGRITY,
1e-6)`, `P_ENERGY − P_PREV_ENERGY`, `P_INTEGRITY − P_PREV_INTEGRITY` — using
the established same-cycle thread-0 read pattern. The angular-velocity slot
and everything else keep reading `sensory_buffer`; `touch_offset` derivation
is unchanged. The deltas now cover the last physics sub-tick — the one that
contains any eat event or hazard damage from this cycle (`P_PREV_ENERGY` is
snapshotted at the top of each `agent_physics` sub-tick; food detect and
biome damage run after the snapshot). No unit test can observe workgroup
memory directly; the behavioral gate is the hazard-probe re-measure plus the
existing suite staying green.

## 0003 — Learning visibility and lag economics

### Quarter-split learning metric

`run_headless` (`headless.rs:28-270`) already reads `cached_state` every
chunk inside the generation loop and prints per-generation metrics in
`log_learning_metrics` (`headless.rs:278-323`). Quarter sampling is
therefore free.

Edits (`headless.rs`):

- A pure helper, unit-testable without a GPU:

  ```rust
  /// Food rates (per 1k alive-ticks) of the first and last generation
  /// quarters, from cumulative (food, alive_ticks) samples taken at the
  /// four quarter boundaries. Rising last-over-first is the direct signal
  /// that the population improves within a lifetime instead of only
  /// across generations.
  fn quarter_rates(samples: &[(u64, u64); 4]) -> (f64, f64)
  ```

  First-quarter rate from sample 0; last-quarter rate from the deltas
  between samples 2 and 3; zero-alive-ticks guards return 0.0.

- The generation loop samples cumulative `P_FOOD_COUNT` / `P_TICKS_ALIVE`
  population sums whenever `ticks_done` crosses a quarter boundary (the
  final state read that already exists for fitness extraction supplies the
  fourth sample), and `log_learning_metrics` prints
  `Learn q1→q4: {first:.3} → {last:.3}` alongside the existing
  `Food/1k-ticks` figure.

### Stride/lag sweep + discount documentation

The `TD_DISCOUNT` comment (`common.wgsl:280-282`) claims the 33-brain-tick
horizon "matches the travel time from the edge of vision range to food" —
true only at `brain_tick_stride = 1` (45 physics ticks ≈ 33–45 brain ticks);
at the default stride 10 an approach is ~4.5 brain ticks and the horizon is
~11 s of real time. The comment is corrected (value unchanged) to state the
real-time horizon, the actual approach time, and the recalibration formula
(γ ≈ 1 − stride/330 to preserve an 11 s horizon) for any future default
change.

The sweep itself is configuration plus the existing headless A/B protocol —
three `FullConfig` JSONs under `experiments/` sharing
`world.seed = 42`, `governor.tick_budget = 120000`,
`governor.population_size = 12`, `governor.max_generations = 16`:

| Arm | `brain_tick_stride` | `vision_stride` | Sensory lag |
|---|---|---|---|
| `lag100-control` | 10 | 10 | 100 ticks |
| `lag10` | 2 | 5 | 10 ticks |
| `lag2` | 1 | 2 | 2 ticks |

Compared on `Food/1k-ticks` trend, `Learn q1→q4` trend, `Deaths`, and
`ticks/sec`. Decision rule in SCOPE (locked decisions). The mirrored-probe
evidence (steering at chance at lag 1) is why this is an experiment with a
budget rather than the expected fix.

## 0004 — Gated follow-ups

### Fitness rework (gated on 0001–0003 data)

`governor.rs:473-479` today:

```rust
let survival = 1.0 / (1.0 + r.death_count as f32 * 0.5);
let foraging = (r.food_consumed as f32 / food_target).min(1.0);
// …
r.composite_fitness = survival * 0.4 + foraging * 0.3 + exploration * 0.3;
```

Two fully specified variants; the data picks one (see SCOPE locked
decisions):

- **Variant B — multiplicative survival** (kamikaze confirmed):
  `composite = survival * (foraging * 0.5 + exploration * 0.5)` — an agent
  that forages by dying repeatedly no longer outscores one that forages
  carefully.
- **Variant A — improvement term** (learning invisible): `Agent` gains
  `food_rate_first_quarter` / `food_rate_last_quarter` (populated per-agent
  in `run_headless` by lifting the quarter sampling from
  population-cumulative to per-agent), threaded into the fitness record;
  `improvement = ((last − first) / max(first, 0.05)).clamp(−1, 1) · 0.5 + 0.5`;
  `composite = survival·0.3 + foraging·0.25 + exploration·0.2 +
  improvement·0.25`.

Either variant updates the governor's composite unit tests (the
`mock_fitness` helpers construct records directly) and lands with a
fixed-seed 16-generation comparison against the sweep control.

### Predictor forward-model objective (gated, Phase-2 discipline)

Today pass 7a trains the prediction made *this* tick against *this* tick's
input through *this* tick's input (`brain_passes.wgsl:719-730`) — gradient
descent toward `W·x ≈ x` — while the novelty error uses `O_PREV_PREDICTION`
vs the current state (`brain_passes.wgsl:323-330`).

Edits (`brain_passes.wgsl`):

- **Train-then-predict in pass 6** (threads 0..`PREDICTOR_DIMENSION`): first
  train on the completed transition — `O_PREV_PREDICTION` (not yet
  overwritten) against current `s_encoded`, gradient through
  `O_PREV_ENCODED` (not yet overwritten) — then matmul the new prediction
  from current `s_encoded`. Each thread owns its weight row for both the
  update and the matmul, so no barrier is needed between them.
- **The predictor moves to encoded space**: it reads and predicts
  `s_encoded` (pre-habituation), not `s_habituated` — the forward model
  learns world dynamics, habituation stays the attention layer downstream,
  and the recalled-context blend becomes consistent (patterns store
  encoded-space vectors).
- **One error, computed once**: the pass-6 thread-0 novelty loop targets
  `s_encoded`; the duplicate end-of-pass error recomputation is deleted;
  pass 7's context-weight adaptation consumes the shared `s_pred_error`.
  Pass 7a is deleted (training moved); pass 3's `O_PREV_ENCODED` overwrite
  moves to a new step 7g at the end of pass 7 (habituation reads the same
  value it always did — last tick's encoded state).

Expected effect is honestly small (no efference copy → near-identity is
near-optimal in quiet stretches); the claim is objective consistency. Keep
only if every probe band holds and the fixed-seed headless control does not
regress; revert wholesale otherwise and record the negative result.

## Test strategy

- `telemetry_exposes_non_visual_sensory_tail` (`tests/integration.rs`): tail
  length equals `sensory_stride − vision_color_count − vision_depth_count`;
  normalized energy at tail index 7 lands in (0.5, 1.0] after one tick.
- `hazard_probe_exit_latency_baseline` (`tests/integration.rs`): three
  episodes × 16 agents in the half-plane arena; prints and pins
  `exit_fraction` / `mean_exit_latency` / `death_fraction`; structural
  assertions (no double-counted outcomes; arena actually hazardous).
- `death_applies_terminal_td_update_through_traces`
  (`tests/integration.rs`): warm-up tick on safe biome, flip to all-danger
  with `integrity_scale = 200` (one-tick kill), preset
  `value_bias = 0.5`, `trace_biases = [5, 1, 1]`; after the death tick,
  `value_bias == 0.45 ± 1e-3` and both actor biases dropped by exactly
  0.10 — the only bias movement in the death cycle is the terminal kick, so
  the assertion is exact, falsifiable, and adapter-stable.
- `gpu_touch_emits_hazard_contact_in_danger_biome` (`tests/integration.rs`):
  all-danger arena, one batch → slot 0 tag ≈ 0.75, intensity 0.5, zero
  planar direction.
- `gpu_touch_emits_terrain_edge_contact_near_wall`
  (`tests/integration.rs`): agent at `x = 126.5` (1.5 units from the +X
  wall) → some slot with tag ≈ 0.5, direction x < −0.9, intensity ≈ 0.5.
- `quarter_rates_computes_first_and_last_quarter_food_rates` and
  `quarter_rates_handles_zero_alive_ticks` (`headless.rs` unit tests, no
  GPU): exact arithmetic on crafted samples.
- Same-cycle interoception has no direct unit observable (workgroup
  memory); its gate is the hazard-probe re-measure plus the full suite —
  in particular `learning_probe_free_run_foraging_baseline` and the
  mirrored steering band must not move.
- Regression gates for every 0002+ change: the full existing suite,
  the pinned hazard bands, and (for the predictor task) the fixed-seed
  headless control numbers.

`cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D
warnings`, `cargo test -p xagent-sandbox` stay green throughout. GPU tests
self-skip without an adapter; CI runs Mesa lavapipe.

## Interaction with prior work

- **Plan 2026-06-10 (emergent learning pathway)** and the **2026-06-10
  learning-baseline spec**: this plan extends their measurement discipline
  to the danger side. The TD(λ) machinery this plan grounds is that plan's
  Phase 1; the encoder hands-off decision honors its Phase-2 negative
  result; the mirrored-probe correction from its Phase 3 is what demotes
  the lag hypothesis to an experiment. New baselines and after-numbers are
  recorded in the same spec file.
- **Issue #115 / `MAX_SENSORY_LAG_TICKS`**: the lag tripwire and its
  documentation stay; the sweep (workstream 0003) is the measured follow-up
  the kernel comment defers to. Any default-stride change re-pins
  `brain_config_tuned_defaults` and `default_sensory_lag_is_within_bound`.
- **CPU sensory reference (`agent/senses.rs`)**: the GPU touch port copies
  its semantics (tags, ranges, directions, intensities) so the two paths
  describe one world.
- **The four 2026-06-12 reviews**: origin of the findings; SCOPE records
  which claims were adopted versus rejected so they are not re-litigated.
