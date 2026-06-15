# Architecture - Plan 0007 (deltas)

> Edits in `crates/xagent-sandbox/src/app.rs`,
> `crates/xagent-sandbox/src/gpu_orchestration.rs`,
> `crates/xagent-sandbox/src/sim_runtime.rs`,
> `crates/xagent-sandbox/src/governor.rs`,
> `crates/xagent-shared/src/config.rs`,
> `crates/xagent-brain/src/buffers.rs`,
> `crates/xagent-brain/src/shaders/kernel/common.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`,
> and `crates/xagent-sandbox/tests/integration.rs`. Line numbers are hints;
> locate by symbol.

## 0001 - Runtime Genome Authority

Today `PendingUpload` carries only `agent_data`
(`app.rs:37-43`), and `build_pending_upload` drops the full per-agent
`BrainConfig` (`gpu_orchestration.rs:31-56`). `Worker::new` creates a kernel
from one population-wide `brain_config`, uploads world/agent physics, and never
patches per-agent brain-state tail slots (`sim_runtime.rs:317-331`). On
generation reset, `reset_population` first reseeds/inherits champion state and
then stops (`sim_runtime.rs:398-445`). The headless path performs the missing
step by calling `write_agent_heritable_config` for each agent
(`headless.rs:159-164`).

Edits:

- **Carry full per-agent configs to the worker** (`app.rs:37-43`):

```rust
/// Full genome for each uploaded agent, parallel to `agent_data`.
pub(crate) agent_configs: Vec<BrainConfig>,
```

- **Populate `agent_configs` from `self.agents`** (`gpu_orchestration.rs:31-56`):

```rust
agent_configs: self.agents.iter().map(|agent| agent.brain_config.clone()).collect(),
```

- **Patch per-agent heritable slots after every upload/inheritance**
  (`sim_runtime.rs:317-331`, `sim_runtime.rs:431-445`):

```rust
fn patch_agent_configs(kernel: &GpuKernel, configs: &[BrainConfig]) {
    for (index, config) in configs.iter().enumerate() {
        let Ok(agent_index) = u32::try_from(index) else { break };
        if agent_index >= kernel.agent_count() {
            break;
        }
        kernel.write_agent_heritable_config(agent_index, config);
    }
}
```

Call this helper after `kernel.upload_agents(...)` in `Worker::new`, and again
after inherited champion/mutant states are written in `reset_population`. The
second call is required because `write_agent_state` overwrites the whole
brain-state buffer, including the tail values that store movement speed,
fatigue floor, curiosity, and habituation sensitivity.

- **Make mutation provenance match effective genes** (`governor.rs:1689-1735`):
  add `movement_speed` to `record_mutations`, and keep the mutation table from
  implying that population-uniform-only fields were evaluated per agent. Until
  independent arenas exist, classify fields as:

| Field | Runtime authority in this plan |
|---|---|
| `memory_capacity`, `processing_slots` | Per-agent physics upload |
| `habituation_sensitivity`, `max_curiosity_bonus`, `fatigue_floor`, `movement_speed` | Per-agent brain-state tail patch |
| `learning_rate`, `decay_rate`, `distress_exponent`, `metabolic_rate`, `integrity_scale` | Population uniform; do not treat within-generation variants as independently evaluated |

Properties that make this safe:

- The patch uses the existing public GPU API `write_agent_heritable_config`,
  already designed for these slots (`gpu_kernel.rs:1912-1952`).
- The call occurs during startup/reset, outside per-tick hot paths.
- No WGSL layout changes are required for this workstream.
- `u32::try_from` preserves numeric safety at the worker boundary.

## 0002 - Behavioral Evidence Telemetry

Today `generation_recording` stores `(node_id, agent_count, tick_count, data)`
only (`governor.rs:1670-1675`). `store_recording` writes 15 floats per agent per
sample (`governor.rs:1276-1281`), but the DB lacks food-bearing, danger dwell,
and an authoritative all-agent staleness/straightness signal. Replay sampling is
also cadence-sampled at collected snapshots, not every physics tick
(`replay_coord.rs:13-19`), so the recording must say what it is.

Edits:

- **Version the recording format** (`governor.rs:1670-1675`):

```sql
ALTER TABLE generation_recording ADD COLUMN format_version INTEGER DEFAULT 1;
ALTER TABLE generation_recording ADD COLUMN record_stride INTEGER DEFAULT 15;
```

New recordings use `format_version = 2` and a named stride constant in Rust.
Legacy blobs remain loadable by treating missing/`1` as the current 15-float
format.

- **Add a per-generation behavior summary table** (`governor.rs:init_schema`):

```sql
CREATE TABLE IF NOT EXISTS behavior_metric (
    node_id INTEGER PRIMARY KEY REFERENCES node(id),
    sample_count INTEGER NOT NULL,
    mean_abs_turn REAL NOT NULL,
    turn_sign_persistence REAL NOT NULL,
    straightness REAL NOT NULL,
    food_distance_delta REAL,
    food_bearing_alignment REAL,
    danger_dwell_fraction REAL,
    danger_exit_latency_ticks REAL
);
```

- **Publish navigational state from the GPU.** Add physics slots for nearest
  food vector and danger flag near `P_NEAREST_FOOD_DISTANCE` (`buffers.rs:159-170`):

```rust
/// Signed bearing from facing direction to nearest in-range food, radians.
pub const P_NEAREST_FOOD_BEARING: usize = 34;
/// Whether the agent is currently in a danger biome.
pub const P_IN_DANGER_BIOME: usize = 35;
```

The exact indices are assigned during implementation by extending
`PHYS_STRIDE`; update Rust/WGSL constants together.

- **Compute these slots in the fused kernel.** `agent_food_detect` already scans
  food positions (`kernel_tick.wgsl:237-253`) and publishes nearest distance
  (`kernel_tick.wgsl:288-294`). Extend the same reduction to carry nearest
  `dx/dz`, then thread 0 computes bearing from current facing. `agent_physics`
  already samples `biome_type` (`kernel_tick.wgsl:168-173`); write the danger
  flag there.

- **Record behavior summaries on generation completion.** Use the cadence-sampled
  `GenerationRecording` as the source of motion metrics and the new GPU slots as
  the source of food/danger metrics. Persist `q1_food_rate` and `q4_food_rate`
  for all new databases and make the migration observable in tests.

Properties that make this safe:

- The food scan already has a workgroup-uniform reduction barrier sequence; carry
  vector components through the same winning distance index rather than adding a
  divergent path.
- Recording format versioning preserves existing `xagent.db` compatibility.
- Metrics are written once per generation, not per frame.

## 0003 - Control-Rate Curriculum

Today the run uses `movement_speed=20.0`, `brain_tick_stride=10`, and
`vision_stride=10`. `BrainConfig::sensory_lag_ticks` documents the one-batch lag
as `vision_stride * brain_tick_stride` (`config.rs:343-348`, `config.rs:381-382`).
At default tick rate 30 Hz (`config.rs:432-443`), an agent can travel 66.7 world
units at full forward during one visual lag window, while vision range is 30.

Edits:

- **Add an explicit learning curriculum preset** (`config.rs`):

```rust
/// Slow control-rate preset for learning probes and early evolution.
pub fn learning_curriculum() -> Self {
    Self {
        movement_speed: 8.0,
        brain_tick_stride: 2,
        vision_stride: 5,
        ..Self::default()
    }
}
```

- **Lower movement speed bounds behind tests.** Update mutation clamps from
  `[20.0, 100.0]` to `[4.0, 30.0]` only after tests and docs are updated. Keep
  energy-cost normalization explicit; if the denominator remains 20.0 in
  `kernel_tick.wgsl:160-166`, document that it is the legacy cost baseline, not
  the new default.

- **Expose speed-lag diagnostics.** Add a helper:

```rust
/// Maximum full-forward travel while one visual batch is stale.
pub fn full_forward_lag_distance(&self, tick_rate: f32) -> f32 {
    self.movement_speed * self.sensory_lag_ticks() as f32 / tick_rate.max(1e-6)
}
```

Properties that make this safe:

- The preset is opt-in until the `0005` probes pass.
- Smaller strides increase GPU work, so throughput is measured before changing
  defaults.
- Lower speed bounds make the user's hypothesis falsifiable rather than hidden
  behind the current minimum of 20.

## 0004 - Turn-Attractor And Klinotaxis Repair

Today policy output is evaluated, noise is added, fatigue scales both channels,
then klinotaxis multiplies turn by a positive factor (`brain_passes.wgsl:650-672`).
The factor can never reverse sign. Position-based staleness currently feeds
fatigue, but replay staleness is selected-agent telemetry and not reliable for
all-agent DB decisions (`gpu_orchestration.rs:254-260`, `gpu_orchestration.rs:330`).

Edits:

- **Add authoritative turn persistence state.** Track a short EMA of signed turn
  and absolute turn in brain state, using new fixed-tail slots near the existing
  position-ring fields. Publish an all-agent `P_TURN_PERSISTENCE_OUT` physics
  slot for recording.

- **Replace multiplier-only klinotaxis with a sign-changing escape path.** The
  replacement uses worsening gradient plus persistent turn to do one of two
  deterministic things:

```wgsl
let worsening = clamp(-(s_homeo[3u] - s_homeo[4u]) * KLINOTAXIS_SENSITIVITY, 0.0, 1.0);
let persistent_turn = clamp(abs(turn_ema) / max(abs_turn_ema, 1e-6), 0.0, 1.0);
let escape = worsening * persistent_turn;
turn = mix(turn, -sign(turn_ema) * max(abs(turn), 0.2), escape);
forward = mix(forward, max(forward, 0.4), escape);
```

The exact constants are pinned by tests, but the invariant is fixed: a worsening
gradient under one-sided turn must be able to reduce or reverse that sign while
preserving forward escape.

- **Keep actor-credit traces zero-mean.** Do not switch eligibility traces back
  to full motor output. The current noise-trace design (`brain_passes.wgsl:674-681`,
  `brain_passes.wgsl:714-739`) avoids reinforcing every pre-existing turn bias
  on positive credit events; this plan changes the control path, not that
  credit invariant.

Properties that make this safe:

- New state lives in per-agent brain/physics buffers and is updated in thread 0
  only; no new cooperative barriers are required.
- The sign-changing behavior is gated by worsening gradient plus measured
  persistence, so ordinary exploratory turning is not globally suppressed.

## 0005 - Food/Danger Emergence Gates

Today Plan 0004's status records a negative mirrored-steering remeasure, and
`xagent.db` cannot prove food chasing or danger avoidance. The existing
integration tests include controlled food visibility and learning probes, but
the long-run acceptance criteria do not require behavior-level proof.

Edits:

- **Add three fixed-seed GPU probes** (`integration.rs`):

| Probe | Setup | Pass condition |
|---|---|---|
| Food closure | One agent, one food item at known left/right/front bearings, curriculum config | Mean nearest-food distance decreases over the first probe window and turn sign aligns with bearing above a fixed threshold |
| Danger exit | Agent starts inside or entering danger biome | Danger dwell fraction falls and exit latency is below a threshold |
| Anti-circle | Agent starts with strong turn bias and no food | Turn persistence falls and straightness rises without increasing death count |

- **Run a short evolution smoke after probes.** A 20-generation fixed-seed run
  under the curriculum preset must beat the current DB's late baseline on at
  least two of: food per 1000 alive ticks, deaths per food, turn persistence, and
  danger dwell.

- **Record the negative if gates fail.** If the probes do not pass, stop at the
  failing workstream and write a decision note instead of continuing into a long
  evolution run.

Properties that make this safe:

- The probes are smaller and more diagnostic than a many-hour DB run.
- Long-run evolution is not used to debug basic control wiring.
- The plan can fail usefully: a failed gate identifies whether the blocker is
  runtime genetics, telemetry, time scale, turn control, or representation.

## Test Strategy

Add or update tests in `crates/xagent-sandbox/tests/integration.rs` and local
unit tests near changed helpers:

- `worker_reset_applies_per_agent_heritable_configs_after_inheritance`:
  verifies movement speed/fatigue/curiosity/habituation tail values survive the
  live reset path.
- `mutation_table_records_movement_speed`:
  verifies a changed speed produces a `movement_speed` mutation row.
- `generation_recording_v2_loads_legacy_and_new_stride`:
  verifies old 15-float blobs remain readable and new blobs expose added fields.
- `learning_curriculum_bounds_lag_distance`:
  verifies the preset's full-forward lag distance is below the food touch/vision
  scale chosen in SCOPE.
- `klinotaxis_can_reverse_persistent_bad_turn`:
  verifies the shader/controller math changes sign or damps strongly under a
  worsening gradient.
- Food closure, danger exit, and anti-circle GPU probes with the standard GPU
  self-skip guard.

CI gate remains:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p xagent-sandbox
```

## Interaction With Prior Work

- **Plan 0004 is honored, not repeated.** Potential-based scalar approach
  shaping remains useful substrate, but its negative mirrored-steering result
  means this plan adds runtime truth, direction-aware metrics, and control
  gates before claiming food seeking.
- **Plan 0005 throughput results are respected.** Lower stride settings are
  measured as a curriculum/default tradeoff; throughput cost is not ignored.
- **Existing DBs remain readable.** Recording versioning and idempotent
  migrations preserve `xagent.db` compatibility while admitting that this
  specific DB cannot answer all learning questions.
