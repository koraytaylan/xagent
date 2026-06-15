# XAgent Plan 0007 - Learning Control Grounding

This plan first makes the live GPU runtime evaluate the genes persisted in
`xagent.db`, then makes behavior measurable, then slows the perception-action
loop enough for credit assignment, then replaces multiplier-only klinotaxis with
a sign-breaking anti-circle control path, and finally requires controlled
food/danger probes before a long evolution run can be interpreted as progress.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md)
for the deltas.

**Conventions**
- Each task has a stable kebab-case **id**.
- **Depends on** lists direct prerequisites only. `-` means none.
- **Done when** is the verifiable acceptance criterion; every code task must
  keep `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo test -p xagent-sandbox` green.
- GPU tests self-skip without an adapter:

```rust
if !xagent_brain::GpuKernel::is_available() {
    eprintln!("Skipping: no GPU/fallback adapter available");
    return;
}
```

---

## 0001 - Runtime Genome Authority

### effective-agent-config-upload - Apply Per-Agent Heritable Configs In The Worker

The live worker currently drops full `BrainConfig` values at the runtime
boundary. `PendingUpload` has only `agent_data` (`app.rs:37-43`), and
`Worker::reset_population` writes inherited champion brain state without
re-applying per-agent tail values (`sim_runtime.rs:433-445`). This makes
movement speed, fatigue floor, curiosity, and habituation mutations unreliable
in interactive evolution.

**Steps:**
1. Add `agent_configs: Vec<BrainConfig>` to `PendingUpload` in `app.rs` and fill
   it from `self.agents.iter().map(|agent| agent.brain_config.clone())` in
   `build_pending_upload`.
2. Add a private `patch_agent_configs(kernel: &GpuKernel, configs: &[BrainConfig])`
   helper in `sim_runtime.rs` that calls `write_agent_heritable_config` for each
   valid agent index using `u32::try_from`.
3. Call the helper in `Worker::new` immediately after `kernel.upload_agents`.
4. Call the helper in `Worker::reset_population` after inherited champion/mutant
   states are written, because `write_agent_state` overwrites the entire
   brain-state buffer.
5. Add a GPU integration test named
   `worker_reset_applies_per_agent_heritable_configs_after_inheritance`. It
   should create two agents with distinct `movement_speed` and `fatigue_floor`,
   force inherited champion state into both, apply the worker reset path or its
   extracted helper, read back both brain states, and assert the tail slots match
   each agent's config.

- **Depends on:** -
- **Done when:** the test fails on the current worker behavior and passes after
  the patch; cargo fmt/clippy/test green.

### mutation-provenance-for-effective-genes - Record Movement Speed And Scope Genes Truthfully

`agent_result.config_json` shows `movement_speed` values up to 29.16912 in the
investigated DB, but `mutation` has no `movement_speed` rows because
`record_mutations` stops at `fatigue_floor` (`governor.rs:1689-1735`). The DB
therefore cannot explain speed changes.

**Steps:**
1. Add `movement_speed` to `record_mutations`.
2. Add a unit test named `record_mutations_includes_movement_speed`.
3. Add source docs or an enum/table near mutation recording that labels which
   `BrainConfig` fields are per-agent effective in the current shared-kernel
   runtime and which are population-uniform only.
4. Do not add mutation rows for fields that are not actually applied per agent
   unless the row also records that scope.

- **Depends on:** `effective-agent-config-upload`
- **Done when:** speed changes produce mutation provenance and the code no
  longer implies population-uniform genes were independently evaluated per
  agent; cargo fmt/clippy/test green.

## 0002 - Behavioral Evidence Telemetry

### recording-format-v2 - Version Recordings And Preserve Legacy DBs

The current `generation_recording` table stores a raw float blob with implicit
stride 15. This is enough to decode old movement traces but not enough to extend
the format safely.

**Steps:**
1. Add idempotent migrations for `format_version INTEGER DEFAULT 1` and
   `record_stride INTEGER DEFAULT 15`.
2. Replace literal `15usize` in `store_recording` and `load_recording` with
   named constants for v1 and v2 strides.
3. Update `load_recording` to use the stored stride/version when present and to
   treat missing columns as legacy v1.
4. Add `generation_recording_v2_loads_legacy_and_new_stride`.

- **Depends on:** -
- **Done when:** the existing `xagent.db` recordings still load and a synthetic
  v2 row validates its declared stride; cargo fmt/clippy/test green.

### navigation-telemetry-slots - Publish Food Bearing And Danger State

The DB cannot prove chasing or avoidance because the GPU never publishes
nearest-food bearing or danger dwell as all-agent physics fields.

**Steps:**
1. Extend `PHYS_STRIDE` in Rust and WGSL with slots for nearest-food bearing,
   nearest-food distance telemetry if needed, danger flag, and turn persistence.
2. In `agent_food_detect`, carry the nearest shaping food's vector through the
   existing reduction and compute a signed bearing relative to facing.
3. In `agent_physics`, write `P_IN_DANGER_BIOME` from the existing biome sample.
4. Update all Rust readback guards and snapshot application paths.
5. Add GPU tests that place food left/right/front and assert the bearing sign
   and distance are correct, plus a danger-biome placement test that asserts the
   danger flag.

- **Depends on:** `recording-format-v2`
- **Done when:** all new telemetry slots have deterministic tests and no legacy
  recording breaks; cargo fmt/clippy/test green.

### behavior-metric-table - Persist Per-Generation Behavior Evidence

Aggregate food and death counts do not answer whether an agent deliberately
chased food or avoided danger. Persist a small behavior summary per node so
future DB investigations do not require ad hoc blob decoding.

**Steps:**
1. Add `behavior_metric` with the schema from `ARCHITECTURE.md`.
2. Compute mean absolute turn, turn-sign persistence, straightness, food-distance
   delta, food-bearing alignment, danger dwell fraction, and danger exit latency
   from the v2 recording at generation completion.
3. Persist `q1_food_rate` and `q4_food_rate` for newly created DBs and add a test
   that a fresh in-memory governor has these columns after `init_schema`.
4. Add a small DB audit helper or test fixture that can reproduce the metrics
   used in SCOPE from a recording row.

- **Depends on:** `navigation-telemetry-slots`
- **Done when:** a generated test recording produces deterministic
  `behavior_metric` values and q-rate columns exist on fresh DBs; cargo
  fmt/clippy/test green.

## 0003 - Control-Rate Curriculum

### learning-curriculum-preset - Add A Slow Learning Preset

The investigated run's full-forward visual-lag travel is about 66.7 world units,
more than twice vision range. Add a deliberate learning preset instead of
continuing to treat that time scale as the only baseline.

**Steps:**
1. Add `BrainConfig::learning_curriculum()` with `movement_speed=8.0`,
   `brain_tick_stride=2`, and `vision_stride=5`.
2. Add `BrainConfig::full_forward_lag_distance(tick_rate: f32) -> f32`.
3. Add `learning_curriculum_bounds_lag_distance`, asserting that the preset's
   full-forward lag distance is below 3.0 world units at `WorldConfig::default`
   tick rate.
4. Wire the preset into the CLI/UI only as an explicit learning preset; do not
   make it the default in this task.

- **Depends on:** `effective-agent-config-upload`
- **Done when:** the preset is available and its lag-distance invariant is
  tested; cargo fmt/clippy/test green.

### movement-speed-range-revisit - Lower Speed Bounds Behind The Curriculum Gate

`movement_speed` is currently documented and tested as clamped to `[20.0, 100.0]`.
That prevents the user's speed hypothesis from being tested through evolution.

**Steps:**
1. Change breeding clamps and tests to `[4.0, 30.0]`.
2. Update docs/comments that mention `[20.0, 100.0]`.
3. Keep the movement energy normalization denominator explicit and documented.
4. Add a mutation test proving values below 4.0 and above 30.0 clamp correctly.

- **Depends on:** `learning-curriculum-preset`
- **Done when:** speed can evolve in the slow range and stale docs are gone;
  cargo fmt/clippy/test green.

## 0004 - Turn-Attractor And Klinotaxis Repair

### authoritative-turn-persistence - Track Turn Persistence In GPU State

The DB proves persistent turning, but the shader does not publish an
authoritative all-agent turn-persistence value. Add one before using it for
control or acceptance gates.

**Steps:**
1. Add brain-state fields for signed turn EMA and absolute turn EMA.
2. Update them in `coop_predict_and_act` after final motor values are known.
3. Publish `P_TURN_PERSISTENCE_OUT = abs(turn_ema) / max(abs_turn_ema, eps)`.
4. Record the field in v2 recordings and `behavior_metric`.
5. Add a CPU-visible shader-math/unit test or GPU probe that feeds repeated same
   sign turns and alternating sign turns, then asserts persistence high vs low.

- **Depends on:** `navigation-telemetry-slots`
- **Done when:** turn persistence is authoritative and recorded for all agents;
  cargo fmt/clippy/test green.

### sign-breaking-klinotaxis - Replace Positive-Only Turn Scaling

Current klinotaxis does `turn *= positive_factor`, so it cannot reverse a bad
turn sign. Replace it with a worsening-gradient and persistence-gated escape
path.

**Steps:**
1. Implement the sign-breaking formula described in `ARCHITECTURE.md`.
2. Preserve the existing noise-based eligibility trace invariant: traces must
   still carry exploration noise, not full motor output.
3. Add `klinotaxis_can_reverse_persistent_bad_turn`. The test should exercise
   the scalar math directly if possible; otherwise add a tiny WGSL-equivalent
   Rust helper used only for tests.
4. Add a fixed-seed anti-circle GPU probe that starts with a strong inherited
   turn bias and asserts turn persistence falls while straightness rises.

- **Depends on:** `authoritative-turn-persistence`
- **Done when:** the current positive-only implementation fails the test and
  the replacement passes; cargo fmt/clippy/test green.

## 0005 - Food/Danger Emergence Gates

### food-closure-probe - Require Bearing-Aligned Food Approach

Food count alone is not food chasing. A controlled probe must show that the
agent turns according to food bearing and closes distance.

**Steps:**
1. Add a fixed flat-world GPU probe with one agent and one food item at front,
   left, and right bearings under `BrainConfig::learning_curriculum()`.
2. Measure initial and final nearest-food distance, turn/bearing alignment, and
   food consumption if it occurs.
3. Set a red-green threshold in the test body: distance must decrease and
   alignment must exceed the chance band recorded by Plan 0004.
4. Persist the same fields into `behavior_metric` during normal runs.

- **Depends on:** `sign-breaking-klinotaxis`, `learning-curriculum-preset`,
  `behavior-metric-table`
- **Done when:** the probe can fail before the control fixes and pass after
  them; cargo fmt/clippy/test green.

### danger-exit-probe - Require Hazard Avoidance Evidence

Danger avoidance needs a direct dwell/exit metric. Death count is too delayed
and confounded by starvation.

**Steps:**
1. Add a fixed-seed GPU probe where an agent starts inside or entering a danger
   biome with a safe region nearby.
2. Measure danger dwell fraction and exit latency from `P_IN_DANGER_BIOME`.
3. Assert the curriculum/control policy exits danger under a fixed threshold and
   does not re-enter immediately.
4. Persist danger dwell and exit latency into `behavior_metric`.

- **Depends on:** `sign-breaking-klinotaxis`, `behavior-metric-table`
- **Done when:** danger avoidance has a direct pass/fail test and DB metric;
  cargo fmt/clippy/test green.

### short-evolution-gate - Run A Small Evidence-Based Evolution Gate

Only after runtime truth, telemetry, speed scale, and control probes are fixed
should a generational run be used as evidence.

**Steps:**
1. Add or document a reproducible 20-generation fixed-seed run command using the
   learning curriculum.
2. Compare against the investigated DB's late baseline: food per 1000 alive
   ticks, deaths per food, turn persistence, straightness, food-bearing
   alignment, and danger dwell.
3. Require improvement on at least two primary behavior metrics plus no
   regression in deaths per food before proposing a default change.
4. If the gate fails, add a decision note in this plan folder identifying the
   first failed workstream and stop.

- **Depends on:** `food-closure-probe`, `danger-exit-probe`
- **Done when:** either the gate passes with recorded metrics or a negative
  decision note is committed; cargo fmt/clippy/test green.

---

**End of plan 0007 TASKS.** When every "Done when" bullet is green, the plan's
end state is reached.
