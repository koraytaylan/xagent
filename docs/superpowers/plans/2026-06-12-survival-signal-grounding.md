# Survival Signal Grounding: Terminal Death Lesson, Live Hazard Touch, Same-Cycle Interoception

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "danger = bad" learnable by the within-lifetime TD(λ) learner — today the death event carries zero learning signal, hazard is invisible to touch, and interoception arrives one vision batch late — and extend the measurement harness so danger avoidance and within-lifetime improvement become pinned, falsifiable numbers like the foraging probes already are.

**Architecture:** Three small WGSL changes ground the existing homeostatic teacher in time (one terminal TD update through the dying episode's eligibility traces; hazard/terrain-edge touch contacts emitted by the live GPU sensory path; energy/integrity features read same-cycle from `physics_state` instead of the batch-lagged `sensory_buffer`). A new hazard probe pins baseline numbers first, house-style, so every change lands with before/after measurements. Outer-loop work (quarter-split learning metrics, stride sweep, fitness rework) follows behind explicit gates.

**Tech Stack:** Rust, wgpu/WGSL fused kernel (`kernel_tick.wgsl` + `brain_passes.wgsl`), egui sandbox, SQLite governor, integration probes on Mesa lavapipe.

**References:** the four 2026-06-12 reviews (`docs/reviews/2026-06-12-{claude-fable-5,gemini-31-pro,gpt-5-codex,grok-43}.md`), `docs/superpowers/specs/2026-06-10-learning-baseline.md`, `docs/superpowers/plans/2026-06-10-emergent-learning-pathway.md`, issues #13 and #115.

---

## Evidence base: the four reviews versus the code on `develop`

The reviews were written against different snapshots. Every load-bearing claim was re-verified against `develop` @ `0260f7e` before being adopted here.

The grok-43 review was later re-issued against the current snapshot: it now describes the TD(λ) architecture accurately (its buffer-size arithmetic checks out exactly), reaches the same priority ordering as this plan, and cites this plan directly — so it confirms the diagnosis but is not independent evidence for it. Its one new mechanical claim (depth dropped from features) is false; see the rejected table.

### Verified — adopted into this plan

| Claim | Review(s) | Evidence on `develop` |
|---|---|---|
| Death teaches nothing: traces and `O_PREV_VALUE` are zeroed at respawn with no terminal update, and respawn restores full energy in the same kernel cycle — dying while starving is a net homeostatic *win* | claude-fable-5 (Finding 2) | `kernel_tick.wgsl:383-391` zeroes traces with no update; `kernel_tick.wgsl:337` restores `P_ENERGY = max_energy`; cycle order at `kernel_tick.wgsl:474-507` runs respawn *before* the brain tick, so the dying transition is never evaluated |
| `TOUCH_HAZARD` / `TOUCH_TERRAIN_EDGE` are defined but never emitted by the live GPU path — danger has no touch grounding while food has visual + touch + reward | claude-fable-5 (Finding 3), gpt-5-codex (Finding 5) | `common.wgsl:213-214` defines the tags; `phase_vision.wgsl:224-311` emits only `TOUCH_FOOD` and `TOUCH_AGENT`; the CPU reference `agent/senses.rs::detect_touch` emits both missing tags |
| Interoception (energy/integrity/deltas) reaches the feature vector only via the once-per-batch vision pass — at decision time it is up to `vision_stride × brain_tick_stride` = 100 physics ticks stale; present-moment pain enters only through the reward/urgency scalars, never the state | claude-fable-5 (Finding 3), gpt-5-codex (Experiment 3) | `phase_vision.wgsl:206-222` packs them; `brain_passes.wgsl:70-109` (`coop_feature_extract`) reads only `sensory_buffer`; `coop_habituate_homeo` (`brain_passes.wgsl:148-183`) shows the same-cycle `physics_state` read pattern is already safe |
| Sensory state is frozen across all `vision_stride` brain decisions in a batch and stale by one full batch (100 physics ticks at defaults); at `movement_speed = 20` an agent travels ~67 units per vision refresh versus a 30-unit vision range; speed is heritable only upward ([20, 100]) | all four | `config.rs:497` (test pins lag = 100), `kernel_tick.wgsl:509-523` (lag comment), `common.wgsl:204` (`VISION_MAX_DIST = 30`), `config.rs:105-108` (speed clamp) |
| Fitness is satisfiable without learning: `survival·0.4 + foraging·0.3 + exploration·0.3` with `survival = 1/(1+0.5·deaths)`, while `food_count`/`ticks_alive` persist across free respawns — kamikaze foraging pays | claude-fable-5 (Finding 5), gpt-5-codex (Finding 6), grok-43 | `governor.rs:473-479`; `kernel_tick.wgsl:318-348` preserves fitness fields through respawn |
| Predictor trains as an identity autoencoder: pass 7a trains *this tick's* prediction against *this tick's* input, while the novelty error compares *last tick's* prediction against the current state — the training objective is not the measured objective | gemini-31-pro (Bug 2) | `brain_passes.wgsl:719-730` (trains same-tick), versus `brain_passes.wgsl:323-330` (novelty uses `O_PREV_PREDICTION`) |
| Memory stores a pattern every brain tick with valence = the instantaneous gradient (≈ −0.001 most ticks); eligibility records noise before fatigue/klinotaxis scale the executed motor; vision alpha is a constant 1.0 (48 of 192 color features dead) | claude-fable-5 (smaller issues), grok-43, gpt-5-codex (Finding 4) | `brain_passes.wgsl:788-813` (unconditional store), `brain_passes.wgsl:610-634` (noise published pre-scaling), `phase_vision.wgsl:40,84,130,146-150` (alpha always 1.0) |
| `TD_DISCOUNT` comment mis-calibrated ~7×: the 33-brain-tick horizon equals vision-edge travel time only at `brain_tick_stride = 1`, not the default 10 | claude-fable-5 (Finding 4, minor) | `common.wgsl:280-282`; 30 units at 20 u/s = 45 physics ticks = 4.5 brain ticks at stride 10 |

### Stale or wrong — rejected, with reasons

| Claim | Review(s) | Why rejected |
|---|---|---|
| Architecture is REINFORCE over a 64-slot history ring with `CREDIT_DECAY`/`PAIN_AMP` | gemini-31-pro | That machinery was replaced by the TD(λ) actor-critic (`brain_passes.wgsl:368-430`, "no history ring"); critiques of its timing transfer only loosely. (grok-43 originally shared this description; its re-issued version describes the TD(λ) architecture correctly) |
| Depth pixels are "dropped from the feature vector" | grok-43 (re-issued version, mechanical notes) | False: `coop_feature_extract` copies color *and* depth 1:1 into `s_features` (`brain_passes.wgsl:73-80`), and `FEATURE_COUNT` includes `VISION_DEPTH_COUNT` (`common.wgsl:30`) |
| The encoder already uses an Oja/PCA rule | gpt-5-codex (Finding 3) | Not on `develop` — Codex reviewed a worktree with Gemini's proposed patch applied. `develop` has the task-driven encoder credit (`brain_passes.wgsl:744-757`) |
| Fix the encoder with Oja's rule (PCA) | gemini-31-pro (Bug 1 fix) | Contradicted by the project's own measurements: encoder self-supervision was implemented and reverted as a measured negative (plan 2026-06-10, Phase 2), and `encoder_food_side_separability_diagnostic` shows the random encoder *preserves* food-side separability at a ≈4× margin. Codex's own Finding 3 explains why variance-preservation ≠ survival relevance. The encoder is measured non-binding; do not touch it without a probe regression first |
| "Lag isolation will confirm Finding 1 as dominant — if approach behavior emerges at stride 1, walk strides back up" | claude-fable-5 (Experiment 1), gpt-5-codex (Experiment 1) | **Already partially falsified**: `learning_probe_mirrored_steering_is_chance` trains at `brain_tick_stride = 1, vision_stride = 1` (lag = 1 tick) and still lands at chance (0.52). Lag removal alone does not produce vision-conditional steering. Lag remains *necessary* to fix for the open world (no closed-loop behavior can survive a 100-tick frozen frame), but it is not *sufficient* — so the stride sweep here is an A/B experiment with a TPS budget, not a confirmation exercise |

### What this means

The reviews converge on "the learner is sound, the world as experienced is unlearnable" — but none of them knew about the mirrored probe. Combining both bodies of evidence:

1. **The danger pathway is genuinely absent** (no touch cue, no terminal lesson, stale interoception, free respawn). This is verified missing mechanism, not hypothesis. It is also unprobed — the probe arenas are hazard-free. Highest confidence, smallest diffs, measurement gap to close first. → Tasks 1–6.
2. **Within-lifetime steering fails even at lag 1** — the open bottleneck is credit dynamics under movement nuisance (baseline spec, Phase 3), not representation, not lag alone. Fixes here are experiments, not fixes; they go behind probe gates. → Tasks 8–10.
3. **The outer loop cannot select for learning it cannot see.** Quarter-split metrics make within-lifetime improvement visible first; the fitness formula changes only after the data says which failure mode (invisible learning vs kamikaze foraging) is binding. → Tasks 7, 9.

---

## File structure

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/xagent-brain/src/gpu_kernel.rs` | Modify | expose the non-visual sensory tail in `AgentTelemetry` |
| `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl` | Modify | terminal TD update in `agent_death_respawn` before trace zeroing |
| `crates/xagent-brain/src/shaders/kernel/common.wgsl` | Modify | `TERMINAL_DEATH_TD_ERROR` + `TOUCH_HAZARD_INTENSITY` constants; corrected `TD_DISCOUNT` comment |
| `crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl` | Modify | emit `TOUCH_HAZARD` and `TOUCH_TERRAIN_EDGE` contacts |
| `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl` | Modify | same-cycle interoception in `coop_feature_extract`; (gated Task 10) predictor objective fix |
| `crates/xagent-sandbox/tests/integration.rs` | Modify | hazard probe section, terminal-death test, touch-contact tests |
| `crates/xagent-sandbox/src/headless.rs` | Modify | quarter-split learning metric |
| `crates/xagent-sandbox/src/governor.rs` | Modify (gated Task 9) | fitness rework |
| `docs/superpowers/specs/2026-06-10-learning-baseline.md` | Modify | record hazard-probe baselines and after-numbers |

Tasks 1–6 are the core sequence. Task 7 is independent of 2–6. Tasks 8–10 are gated experiments.

---

### Task 1: Expose the non-visual sensory tail in telemetry

The telemetry readback already copies the full `sensory_stride` floats per agent and then discards everything after the vision colors. Exposing the tail costs nothing and is the observability prerequisite for the touch tests (Task 4) and the hazard probe (Task 2).

**Files:**
- Modify: `crates/xagent-brain/src/gpu_kernel.rs:193-210` (`AgentTelemetry`), `crates/xagent-brain/src/gpu_kernel.rs:1888-1903` (`read_agent_telemetry_blocking`)
- Test: `crates/xagent-sandbox/tests/integration.rs`

- [ ] **Step 1: Write the failing test**

Add to the Learning Probe Tests section of `integration.rs`:

```rust
/// The telemetry sensory tail must expose the packed non-visual senses
/// (velocity 3, facing 3, angular 1, energy, integrity, energy delta,
/// integrity delta, then 4 touch contacts × 4) so probes can assert on
/// touch contacts and interoception without raw buffer plumbing.
#[test]
fn telemetry_exposes_non_visual_sensory_tail() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = probe_brain_config();
    let mut arena = build_probe_arena(&brain, 31);
    arena.kernel.dispatch_batch(0, 1);

    let telemetry = arena.kernel.read_agent_telemetry_blocking(0);
    let layout = xagent_brain::buffers::BrainLayout::new(brain.vision_width, brain.vision_height);
    let expected_len =
        layout.sensory_stride - layout.vision_color_count - layout.vision_depth_count;
    assert_eq!(
        telemetry.sensory_non_visual.len(),
        expected_len,
        "non-visual tail length must match the layout"
    );
    // Energy is packed normalized at index 7 of the tail and the agent is
    // alive at full-ish energy after one tick.
    let energy_normalized = telemetry.sensory_non_visual[7];
    assert!(
        (0.5..=1.0).contains(&energy_normalized),
        "normalized energy {energy_normalized} not in (0.5, 1.0] after one tick"
    );
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p xagent-sandbox --test integration telemetry_exposes_non_visual_sensory_tail`
Expected: FAIL — `no field sensory_non_visual on type AgentTelemetry` (compile error).

- [ ] **Step 3: Implement**

In `gpu_kernel.rs`, add the field to `AgentTelemetry` (after `vision_color`):

```rust
    /// Non-visual sensory tail exactly as packed by `phase_vision_senses`:
    /// [velocity(3), facing(3), angular(1), energy, integrity,
    ///  energy_delta, integrity_delta, touch(4 contacts × 4)].
    /// One vision batch stale, like all of `sensory_buffer`.
    pub sensory_non_visual: Vec<f32>,
```

In `read_agent_telemetry_blocking`, after the `vision_color` slice:

```rust
        let non_visual_base = self.layout.vision_color_count + self.layout.vision_depth_count;
        let sensory_non_visual: Vec<f32> = sensory[non_visual_base..].to_vec();
```

and add `sensory_non_visual,` to the `AgentTelemetry { ... }` construction at the end of the function.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p xagent-sandbox --test integration telemetry_exposes_non_visual_sensory_tail`
Expected: PASS (self-skips without an adapter; CI has lavapipe).

- [ ] **Step 5: Gates and commit**

Run: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test -p xagent-sandbox`
Expected: all green.

```bash
git add crates/xagent-brain/src/gpu_kernel.rs crates/xagent-sandbox/tests/integration.rs
git commit -m "feat: expose non-visual sensory tail in agent telemetry"
```

---

### Task 2: Hazard probe — pin the danger-avoidance baseline

House rule (plan 2026-06-10, Phase 0): the gates come first. This probe is to danger what the mirrored steering probe is to food. It measures **hazard-exit latency** and **deaths** for agents that start inside a danger half-plane, with the world boundary geometry that makes escape a ~10-unit eastward walk.

**Files:**
- Test: `crates/xagent-sandbox/tests/integration.rs` (new "Hazard Probe Tests" section, after the Learning Probe Tests)

- [ ] **Step 1: Write the probe**

```rust
// ── Hazard Probe Tests ──────────────────────────────────────────────────
//
// A half-plane danger arena: danger biome for x < 0, food-rich for x ≥ 0.
// Agents start 10 units inside the danger side, facing +Z (parallel to the
// boundary, so straight-line walking never exits on its own). Measured:
// hazard-exit latency (first tick with x ≥ 0, alive, without dying first)
// and deaths. These pin the danger-avoidance baseline the same way the
// mirrored steering probe pins food-approach.

/// Distance agents start inside the danger half-plane. Far enough that
/// exit requires sustained directed movement (~15 ticks of straight-east
/// walking at default speed), close enough that random walks exit within
/// the episode often enough to measure a latency distribution.
const HAZARD_PROBE_START_DEPTH: f32 = 10.0;
/// Episode cap in physics ticks. At hazard damage 0.5/tick (rate 1.0 ×
/// integrity_scale 0.5) an agent that never exits dies at tick 200, so
/// 600 ticks cleanly separates "exited", "died", and "wandered".
const HAZARD_PROBE_EPISODE_TICKS: u64 = 600;
/// Position sampling interval — bounds latency resolution and readback cost.
const HAZARD_PROBE_SAMPLE_TICKS: u32 = 5;
/// Training episodes for the trained variant of the probe.
const HAZARD_PROBE_EPISODES: usize = 3;

/// Outcome of one hazard episode for one agent.
struct HazardEpisodeOutcome {
    exit_latency_ticks: Option<u64>,
    died: bool,
}

/// Run one hazard episode and classify each agent's outcome.
fn run_hazard_episode(arena: &mut ProbeArena, start_tick: u64) -> Vec<HazardEpisodeOutcome> {
    use xagent_brain::buffers::{PHYS_STRIDE, P_DEATH_COUNT, P_POS_X};

    let initial_state = arena.kernel.read_full_state_blocking().to_vec();
    let initial_deaths: Vec<f32> = (0..PROBE_AGENT_COUNT)
        .map(|a| initial_state[a * PHYS_STRIDE + P_DEATH_COUNT])
        .collect();

    let mut outcomes: Vec<HazardEpisodeOutcome> = (0..PROBE_AGENT_COUNT)
        .map(|_| HazardEpisodeOutcome { exit_latency_ticks: None, died: false })
        .collect();

    let mut ticks_done: u64 = 0;
    while ticks_done < HAZARD_PROBE_EPISODE_TICKS {
        arena
            .kernel
            .dispatch_batch(start_tick + ticks_done, HAZARD_PROBE_SAMPLE_TICKS);
        ticks_done += u64::from(HAZARD_PROBE_SAMPLE_TICKS);
        let state = arena.kernel.read_full_state_blocking();
        for (a, outcome) in outcomes.iter_mut().enumerate() {
            if outcome.died || outcome.exit_latency_ticks.is_some() {
                continue;
            }
            if state[a * PHYS_STRIDE + P_DEATH_COUNT] > initial_deaths[a] {
                outcome.died = true;
            } else if state[a * PHYS_STRIDE + P_POS_X] >= 0.0 {
                outcome.exit_latency_ticks = Some(ticks_done);
            }
        }
    }
    outcomes
}

/// Build the half-plane danger arena on top of the standard probe arena:
/// biome column < 128 (x < 0) is danger, the rest food-rich; agents are
/// re-positioned to x = −HAZARD_PROBE_START_DEPTH, spread along z.
fn build_hazard_arena(brain: &BrainConfig, brain_seed: u64) -> ProbeArena {
    let mut arena = build_probe_arena(brain, brain_seed);
    let mut biomes = vec![0_u32; PROBE_BIOME_RES * PROBE_BIOME_RES];
    for row in 0..PROBE_BIOME_RES {
        for col in 0..PROBE_BIOME_RES / 2 {
            biomes[row * PROBE_BIOME_RES + col] = 2;
        }
    }
    arena.biomes = biomes;
    for (index, agent) in arena.agent_data.iter_mut().enumerate() {
        let z_spread = (index as f32 - (PROBE_AGENT_COUNT as f32 - 1.0) / 2.0) * 8.0;
        agent.0 = glam::Vec3::new(-HAZARD_PROBE_START_DEPTH, PROBE_AGENT_Y, z_spread);
    }
    arena.reset_bodies();
    arena
}

/// Baseline: untrained agents in the hazard arena. Prints exit fraction,
/// mean exit latency, and death fraction; asserts only structural sanity
/// plus wide falsifiable bands. Re-pin the bands after Tasks 3–5 land
/// (the same protocol as the steering probes).
#[test]
fn hazard_probe_exit_latency_baseline() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        ..Default::default()
    };
    let mut arena = build_hazard_arena(&brain, 37);

    let mut exits: Vec<u64> = Vec::new();
    let mut deaths = 0_usize;
    let mut tick_cursor = 0_u64;
    for episode in 0..HAZARD_PROBE_EPISODES {
        if episode > 0 {
            arena.reset_bodies();
        }
        for outcome in run_hazard_episode(&mut arena, tick_cursor) {
            if let Some(latency) = outcome.exit_latency_ticks {
                exits.push(latency);
            }
            if outcome.died {
                deaths += 1;
            }
        }
        tick_cursor += HAZARD_PROBE_EPISODE_TICKS;
    }

    let trials = HAZARD_PROBE_EPISODES * PROBE_AGENT_COUNT;
    let exit_fraction = exits.len() as f64 / trials as f64;
    let death_fraction = deaths as f64 / trials as f64;
    let mean_latency = if exits.is_empty() {
        f64::from(u32::MAX)
    } else {
        exits.iter().sum::<u64>() as f64 / exits.len() as f64
    };
    eprintln!(
        "hazard probe baseline: trials={trials} exit_fraction={exit_fraction:.3} \
         mean_exit_latency={mean_latency:.1} death_fraction={death_fraction:.3}"
    );

    // Structural sanity: every trial resolves into exit, death, or timeout.
    assert!(exits.len() + deaths <= trials, "double-counted outcomes");
    // Falsifiable floor: the arena must actually be dangerous — if nothing
    // ever dies and everything exits instantly, the geometry broke.
    assert!(
        death_fraction > 0.0 || mean_latency > 50.0,
        "arena is not hazardous: death_fraction={death_fraction}, \
         mean_exit_latency={mean_latency}"
    );
}
```

- [ ] **Step 2: Run it and record the numbers**

Run: `cargo test -p xagent-sandbox --test integration hazard_probe_exit_latency_baseline -- --nocapture`
Expected: PASS, with the `hazard probe baseline:` line printing real numbers.

- [ ] **Step 3: Pin the baseline**

Take the printed `exit_fraction`, `mean_exit_latency`, and `death_fraction` and add a pinned-band assertion at the end of the test (same re-pin discipline as `learning_probe_mirrored_steering_is_chance` — generous ±50% relative bands so adapter noise does not flake, tight enough that a real avoidance improvement trips it):

```rust
    // Pinned baseline (recorded <date>, lavapipe): re-pin when Tasks 3-5
    // improve escape. Bands are ±50% relative.
    // exit_fraction was X.XXX, mean_exit_latency was YYY.Y, death_fraction was Z.ZZZ
```

with concrete `assert!((low..=high).contains(&metric))` lines using the recorded values.

- [ ] **Step 4: Record in the baseline spec**

Append a "Hazard probe baseline" subsection to `docs/superpowers/specs/2026-06-10-learning-baseline.md` with the three numbers, the protocol summary, and the gate: *Tasks 3–5 must reduce mean exit latency or death fraction outside the pinned bands to claim a behavioral win.*

- [ ] **Step 5: Gates and commit**

Run: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test -p xagent-sandbox`

```bash
git add crates/xagent-sandbox/tests/integration.rs docs/superpowers/specs/2026-06-10-learning-baseline.md
git commit -m "test: add hazard probe and pin danger-avoidance baseline"
```

---

### Task 3: Terminal death TD update through the dying episode's traces

The one experience the learner must never miss is the transition into death. Apply a single maximal negative TD update through the traces the dying life accumulated, *then* zero them. This preserves the design philosophy: no innate fear module — death itself becomes the teacher, through the existing credit machinery.

**Files:**
- Modify: `crates/xagent-brain/src/shaders/kernel/common.wgsl` (TD constants block, after `MAX_TD_ERROR`)
- Modify: `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl` (`agent_death_respawn`, immediately before the trace-zeroing block at the `// Reset TD transients` comment)
- Modify: `crates/xagent-brain/src/shaders/kernel/phase_death.wgsl` (the standalone death path mirrors the fused one — apply the same insertion before its trace zeroing; grep `O_TRACE_CRITIC` there to find the spot)
- Test: `crates/xagent-sandbox/tests/integration.rs`

- [ ] **Step 1: Write the failing test**

The test makes the kick exactly computable: preset known trace and bias values, kill the agent in one tick (huge `integrity_scale`), and assert the biases moved by exactly `learning_rate × δ_terminal × trace`. The post-respawn brain tick cannot interfere — respawn zeroes the traces before it runs, so the TD update that tick multiplies by zero.

```rust
/// Dying must apply one terminal TD update (δ = −MAX_TD_ERROR) through the
/// dying life's eligibility traces before they are cleared. With preset
/// traces the kick is exactly computable: Δvalue_bias = 0.01·(−1)·5 = −0.05
/// and Δactor_bias = 0.1·(−1)·1 = −0.10. The post-respawn brain tick in the
/// same cycle applies δ through freshly zeroed traces, so it cannot move
/// the biases — any deviation from the exact kick is a real defect.
#[test]
fn death_applies_terminal_td_update_through_traces() {
    use xagent_brain::buffers::{
        BrainLayout, ENCODED_DIMENSION, O_ACT_BIASES, O_PREDICTOR_CONTEXT_WEIGHT,
        O_TRACE_BIASES, O_VALUE_BIAS, PHYS_STRIDE, PREDICTOR_DIMENSION, P_DEATH_COUNT,
    };

    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    /// Hazard damage is rate (1.0) × integrity_scale per physics tick; 200
    /// wipes the full 100 integrity in a single tick inside a danger biome.
    const ONE_TICK_KILL_INTEGRITY_SCALE: f32 = 200.0;

    let brain = BrainConfig {
        integrity_scale: ONE_TICK_KILL_INTEGRITY_SCALE,
        ..probe_brain_config()
    };
    // Warm-up happens on safe biome; then the world flips to all-danger.
    let mut arena = build_probe_arena(&brain, 29);
    arena.kernel.dispatch_batch(0, 1);
    arena.biomes = vec![2_u32; PROBE_BIOME_RES * PROBE_BIOME_RES];
    arena.reset_bodies();

    let layout = BrainLayout::new(brain.vision_width, brain.vision_height);
    let tail_base = layout.feature_count * ENCODED_DIMENSION
        + ENCODED_DIMENSION
        + PREDICTOR_DIMENSION * ENCODED_DIMENSION;
    let value_bias_offset = tail_base + (O_VALUE_BIAS - O_PREDICTOR_CONTEXT_WEIGHT);
    let trace_biases_offset = tail_base + (O_TRACE_BIASES - O_PREDICTOR_CONTEXT_WEIGHT);
    let act_biases_offset = tail_base + (O_ACT_BIASES - O_PREDICTOR_CONTEXT_WEIGHT);

    let agent = 0_u32;
    let mut state = arena.kernel.read_agent_state(agent);
    state.brain_state[value_bias_offset] = 0.5;
    state.brain_state[trace_biases_offset] = 5.0;
    state.brain_state[trace_biases_offset + 1] = 1.0;
    state.brain_state[trace_biases_offset + 2] = 1.0;
    let forward_bias_before = state.brain_state[act_biases_offset];
    let turn_bias_before = state.brain_state[act_biases_offset + 1];
    arena.kernel.write_agent_state(agent, &state);

    arena.kernel.dispatch_batch(1, 1);

    let physics = arena.kernel.read_full_state_blocking();
    assert!(
        physics[agent as usize * PHYS_STRIDE + P_DEATH_COUNT] >= 1.0,
        "agent did not die in the one-tick-kill arena"
    );

    let after = arena.kernel.read_agent_state(agent);
    let value_bias = after.brain_state[value_bias_offset];
    let forward_bias = after.brain_state[act_biases_offset];
    let turn_bias = after.brain_state[act_biases_offset + 1];
    assert!(
        (value_bias - 0.45).abs() < 1e-3,
        "value bias {value_bias} != 0.45: terminal critic kick missing or wrong"
    );
    assert!(
        (forward_bias - (forward_bias_before - 0.10)).abs() < 1e-3,
        "forward bias {forward_bias} (was {forward_bias_before}): terminal actor kick missing"
    );
    assert!(
        (turn_bias - (turn_bias_before - 0.10)).abs() < 1e-3,
        "turn bias {turn_bias} (was {turn_bias_before}): terminal actor kick missing"
    );
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p xagent-sandbox --test integration death_applies_terminal_td_update_through_traces`
Expected: FAIL — value bias stays 0.5 (no terminal update exists yet).

- [ ] **Step 3: Add the constant to `common.wgsl`**

In the TD(λ) credit constants block, after `MAX_TD_ERROR`:

```wgsl
// Terminal TD error applied through the dying episode's eligibility traces
// at the moment of death, before they are cleared for the next life. Death
// must be the single worst lesson the learner can receive, but never
// stronger than the per-transition bound that protects against artifacts.
const TERMINAL_DEATH_TD_ERROR: f32 = -MAX_TD_ERROR;
```

- [ ] **Step 4: Apply the terminal update in `kernel_tick.wgsl`**

In `agent_death_respawn`, immediately above the `// Reset TD transients` comment block:

```wgsl
    // Terminal lesson: the transition into death is the one experience the
    // within-lifetime learner must never miss. Apply one final TD update
    // with the maximum negative error through the eligibility traces the
    // dying life accumulated — then clear them below so no credit leaks
    // into the next life. Without this, dying carries zero learning signal
    // and the full-energy respawn makes death read as a free heal.
    let terminal_value_bias_trace = brain_state[brain_base + O_TRACE_BIASES];
    let terminal_forward_bias_trace = brain_state[brain_base + O_TRACE_BIASES + 1u];
    let terminal_turn_bias_trace = brain_state[brain_base + O_TRACE_BIASES + 2u];
    brain_state[brain_base + O_VALUE_BIAS] +=
        CRITIC_LEARNING_RATE * TERMINAL_DEATH_TD_ERROR * terminal_value_bias_trace;
    brain_state[brain_base + O_ACT_BIASES] +=
        ACTION_WEIGHT_LEARNING_RATE * TERMINAL_DEATH_TD_ERROR * terminal_forward_bias_trace;
    brain_state[brain_base + O_ACT_BIASES + 1u] +=
        ACTION_WEIGHT_LEARNING_RATE * TERMINAL_DEATH_TD_ERROR * terminal_turn_bias_trace;
    for (var i = 0u; i < ENCODED_DIMENSION; i++) {
        brain_state[brain_base + O_VALUE_WEIGHTS + i] += CRITIC_LEARNING_RATE
            * TD_VECTOR_SCALE * TERMINAL_DEATH_TD_ERROR
            * brain_state[brain_base + O_TRACE_CRITIC + i];
        brain_state[brain_base + O_ACTION_FORWARD_WEIGHTS + i] += ACTION_WEIGHT_LEARNING_RATE
            * TD_VECTOR_SCALE * TERMINAL_DEATH_TD_ERROR
            * brain_state[brain_base + O_TRACE_FWD + i];
        brain_state[brain_base + O_ACTION_TURN_WEIGHTS + i] += ACTION_WEIGHT_LEARNING_RATE
            * TD_VECTOR_SCALE * TERMINAL_DEATH_TD_ERROR
            * brain_state[brain_base + O_TRACE_TURN + i];
    }
```

The weight L2-ball clamps run in the next brain tick's pass 6, so magnitudes stay bounded. `agent_death_respawn` is thread-0-only with no barriers — barrier uniformity is untouched.

Apply the identical insertion in `phase_death.wgsl` before its trace zeroing (the standalone pipeline must stay behaviorally identical to the fused one).

- [ ] **Step 5: Run the test to verify it passes, then the suite**

Run: `cargo test -p xagent-sandbox --test integration death_applies_terminal_td_update_through_traces`
Expected: PASS.

Run: `cargo test -p xagent-sandbox`
Expected: all green — in particular `td_traces_bounded_across_deaths` (traces still zeroed after the kick) and `learning_probe_mirrored_steering_is_chance` (no deaths in that arena, so unaffected).

- [ ] **Step 6: Gates and commit**

Run: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings`

```bash
git add crates/xagent-brain/src/shaders/kernel/common.wgsl \
        crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl \
        crates/xagent-brain/src/shaders/kernel/phase_death.wgsl \
        crates/xagent-sandbox/tests/integration.rs
git commit -m "feat: apply terminal TD update through traces at death"
```

---

### Task 4: Emit hazard and terrain-edge touch contacts on the GPU

Food has three grounding paths (visual cue, touch contact, energy spike); danger has one lagged visual cue. The CPU reference (`agent/senses.rs::detect_touch`) already emits `TOUCH_HAZARD` (zero planar direction, fixed intensity — the hazard is the ground underfoot) and `TOUCH_TERRAIN_EDGE` (inward direction, closeness intensity). Port both to `phase_vision_senses`, hazard first so present-moment damage can never be evicted when the four contact slots fill.

**Files:**
- Modify: `crates/xagent-brain/src/shaders/kernel/common.wgsl` (touch constants block)
- Modify: `crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl` (`phase_vision_senses`, touch section)
- Test: `crates/xagent-sandbox/tests/integration.rs`

- [ ] **Step 1: Write the failing tests**

```rust
/// Standing in a danger biome must produce a TOUCH_HAZARD contact in the
/// sensory buffer: zero planar direction (the hazard is underfoot), fixed
/// intensity, tag 3/4. Mirrors the CPU reference in agent/senses.rs.
#[test]
fn gpu_touch_emits_hazard_contact_in_danger_biome() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = probe_brain_config();
    let mut arena = build_probe_arena(&brain, 41);
    arena.biomes = vec![2_u32; PROBE_BIOME_RES * PROBE_BIOME_RES];
    arena.reset_bodies();
    arena.kernel.dispatch_batch(0, 1);

    let telemetry = arena.kernel.read_agent_telemetry_blocking(0);
    // Touch slots start after [vel(3), facing(3), angular(1), intero(4)] = 11.
    let touch_base = 11;
    let first_slot = &telemetry.sensory_non_visual[touch_base..touch_base + 4];
    assert!(
        (first_slot[3] - 0.75).abs() < 1e-3,
        "first touch slot tag {} != 0.75 (TOUCH_HAZARD/4) — hazard contact missing",
        first_slot[3]
    );
    assert!(
        (first_slot[2] - 0.5).abs() < 1e-3,
        "hazard contact intensity {} != 0.5",
        first_slot[2]
    );
    assert!(
        first_slot[0].abs() < 1e-6 && first_slot[1].abs() < 1e-6,
        "hazard contact direction must be planar zero, got ({}, {})",
        first_slot[0], first_slot[1]
    );
}

/// Standing within TOUCH_EDGE_RANGE of a world wall must produce a
/// TOUCH_TERRAIN_EDGE contact pointing inward with closeness intensity.
#[test]
fn gpu_touch_emits_terrain_edge_contact_near_wall() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = probe_brain_config();
    let mut arena = build_probe_arena(&brain, 43);
    // World half-bound is world_size/2 = 128; 1.5 units from the +X wall.
    arena.agent_data[0].0 = glam::Vec3::new(126.5, PROBE_AGENT_Y, 0.0);
    arena.reset_bodies();
    arena.kernel.dispatch_batch(0, 1);

    let telemetry = arena.kernel.read_agent_telemetry_blocking(0);
    let touch_base = 11;
    let mut edge_slot: Option<&[f32]> = None;
    for contact in 0..4 {
        let slot = &telemetry.sensory_non_visual[touch_base + contact * 4..touch_base + contact * 4 + 4];
        if (slot[3] - 0.5).abs() < 1e-3 {
            edge_slot = Some(slot);
            break;
        }
    }
    let slot = edge_slot.expect("no TOUCH_TERRAIN_EDGE contact found near the +X wall");
    assert!(
        slot[0] < -0.9,
        "edge contact must point inward (−X), got direction x = {}",
        slot[0]
    );
    assert!(
        (slot[2] - 0.5).abs() < 0.05,
        "edge intensity {} != ~0.5 at 1.5 units from a 3-unit range wall",
        slot[2]
    );
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p xagent-sandbox --test integration gpu_touch_emits`
Expected: both FAIL — tags 0.75/0.5 never appear (the slots stay zeroed).

- [ ] **Step 3: Add the intensity constant to `common.wgsl`**

In the touch constants block:

```wgsl
// Hazard contacts have no meaningful planar direction (the hazard is the
// terrain underfoot), so they carry a fixed mid-scale intensity instead of
// a closeness value. Matches the CPU reference in agent/senses.rs.
const TOUCH_HAZARD_INTENSITY: f32 = 0.5;
```

- [ ] **Step 4: Emit the contacts in `phase_vision_senses`**

Replace the start of the touch section (the `var touch_count` declaration through the zeroing loop stays, then insert the hazard block before the food loop):

```wgsl
    // ── Touch contacts ────────────────────────────────────────────────
    var touch_count: u32 = 0u;
    let touch_base = off;

    for (var i: u32 = 0u; i < MAX_TOUCH_CONTACTS * 4u; i++) {
        sensory_buffer[touch_base + i] = 0.0;
    }

    // Hazard contact first: present-moment damage must never be evicted by
    // lower-stakes contacts when the four slots fill. Zero planar direction
    // (the hazard is the ground underfoot), fixed intensity — mirrors the
    // CPU reference in agent/senses.rs.
    if (sample_biome(pos.x, pos.z) == BIOME_DANGER) {
        sensory_buffer[touch_base]      = 0.0;
        sensory_buffer[touch_base + 1u] = 0.0;
        sensory_buffer[touch_base + 2u] = TOUCH_HAZARD_INTENSITY;
        sensory_buffer[touch_base + 3u] = f32(TOUCH_HAZARD) / 4.0;
        touch_count = 1u;
    }
```

After the existing agent-contact loop (end of the function), append the four wall checks:

```wgsl
    // Terrain-edge contacts: the world boundary pushes back. Direction
    // points inward (away from the wall), intensity rises as the wall
    // nears — mirrors the CPU reference in agent/senses.rs.
    let world_half_for_touch = wc_f32(WC_WORLD_HALF_BOUND);
    // `var` (not `let`): naga requires a mutable binding for dynamic indexing.
    var wall_distances = array<f32, 4>(
        pos.x + world_half_for_touch,   // distance to the −X wall
        world_half_for_touch - pos.x,   // distance to the +X wall
        pos.z + world_half_for_touch,   // distance to the −Z wall
        world_half_for_touch - pos.z,   // distance to the +Z wall
    );
    var inward_x = array<f32, 4>(1.0, -1.0, 0.0, 0.0);
    var inward_z = array<f32, 4>(0.0, 0.0, 1.0, -1.0);
    for (var wall: u32 = 0u; wall < 4u; wall++) {
        if (touch_count >= MAX_TOUCH_CONTACTS) { break; }
        let wall_distance = wall_distances[wall];
        if (wall_distance < TOUCH_EDGE_RANGE) {
            let slot = touch_base + touch_count * 4u;
            sensory_buffer[slot]      = inward_x[wall];
            sensory_buffer[slot + 1u] = inward_z[wall];
            sensory_buffer[slot + 2u] = 1.0 - max(wall_distance, 0.0) / TOUCH_EDGE_RANGE;
            sensory_buffer[slot + 3u] = f32(TOUCH_TERRAIN_EDGE) / 4.0;
            touch_count += 1u;
        }
    }
```

`phase_vision_senses` is one-thread-per-agent with no barriers — no uniformity concerns.

- [ ] **Step 5: Run the tests to verify they pass, then the suite**

Run: `cargo test -p xagent-sandbox --test integration gpu_touch_emits`
Expected: both PASS.

Run: `cargo test -p xagent-sandbox`
Expected: green. The mirrored steering probe and foraging baselines run in hazard-free arenas away from walls, so their numbers must not move; if any probe band trips, stop and investigate before proceeding.

- [ ] **Step 6: Gates and commit**

Run: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings`

```bash
git add crates/xagent-brain/src/shaders/kernel/common.wgsl \
        crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl \
        crates/xagent-sandbox/tests/integration.rs
git commit -m "feat: emit hazard and terrain-edge touch contacts in GPU sensory path"
```

---

### Task 5: Same-cycle interoception in the feature vector

The brain's only state features that track the reward (energy, integrity, and their deltas) currently arrive one vision batch late. Pain must be felt at decision time. Read them from `physics_state` in `coop_feature_extract` — the same-cycle thread-0 read pattern `coop_habituate_homeo` already uses safely. The `sensory_buffer` slots keep being packed for CPU readback (telemetry/UI read them); only the brain stops consuming them.

The delta semantics change deliberately: from "change across one full batch, one batch late" to "change across the last physics sub-tick, now" — eat events and hazard damage land in the same cycle they happen (physics and eating run before the brain within each cycle).

**Files:**
- Modify: `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl` (`coop_feature_extract`, thread-0 block)

- [ ] **Step 1: Implement**

Replace the five-value read at `ang_offset` (currently a contiguous loop of `sensory_buffer` reads for angular + the four interoception values) with:

```wgsl
        let ang_offset = fac_offset + 3u;
        s_features[fi] = sensory_buffer[s_base + ang_offset]; fi = fi + 1u;
        // Interoception is read same-cycle from physics_state rather than
        // from the batch-lagged sensory_buffer: pain and satiety must be
        // felt at decision time, not one vision batch later. Same-cycle
        // physics reads from thread 0 are the established pattern in
        // coop_habituate_homeo. The packed sensory_buffer slots remain for
        // CPU readback; the brain just stops consuming them. The deltas
        // cover the last physics sub-tick, which is the one that contains
        // any eat event or hazard damage from this cycle.
        let interoception_base = agent_id * PHYS_STRIDE;
        let current_max_energy = max(physics_state[interoception_base + P_MAX_ENERGY], 1e-6);
        let current_max_integrity = max(physics_state[interoception_base + P_MAX_INTEGRITY], 1e-6);
        let current_energy = physics_state[interoception_base + P_ENERGY];
        let current_integrity = physics_state[interoception_base + P_INTEGRITY];
        s_features[fi] = current_energy / current_max_energy; fi = fi + 1u;
        s_features[fi] = current_integrity / current_max_integrity; fi = fi + 1u;
        s_features[fi] = current_energy - physics_state[interoception_base + P_PREV_ENERGY]; fi = fi + 1u;
        s_features[fi] = current_integrity - physics_state[interoception_base + P_PREV_INTEGRITY]; fi = fi + 1u;
        let touch_offset = ang_offset + 5u;
```

The `touch_offset` derivation is unchanged — the `sensory_buffer` layout does not move, only the source of four `s_features` values.

- [ ] **Step 2: Run the suite**

Run: `cargo test -p xagent-sandbox`
Expected: green. No unit test can observe `s_features` directly (workgroup memory); the behavioral gate is Task 6's probe re-run. The existing probes must not regress — `learning_probe_free_run_foraging_baseline` and the mirrored probe both tolerate this change (interoception in a hazard-free arena is a slow monotone drain either way), but verify, don't assume.

- [ ] **Step 3: Gates and commit**

Run: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings`

```bash
git add crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
git commit -m "feat: read interoception same-cycle from physics state in feature extract"
```

---

### Task 6: Re-measure the hazard probe and record the verdict

- [ ] **Step 1: Re-run the hazard probe**

Run: `cargo test -p xagent-sandbox --test integration hazard_probe_exit_latency_baseline -- --nocapture`

Three outcomes:
- Metrics improved beyond the pinned bands → re-pin the bands at the new values (same protocol as the steering probe re-pins) and record the before/after pair.
- Metrics unchanged → the assertion bands hold; record that the grounding changes did not by themselves produce avoidance (expected possibility: the mirrored-probe evidence says credit dynamics may still be the binding constraint — the danger pathway now exists, which is the prerequisite, not the guarantee).
- Metrics worsened → stop, bisect Tasks 3–5 by reverting one at a time, and record which change regressed.

- [ ] **Step 2: Evolution-scale check**

Run a fixed-seed headless comparison (16 generations each, same protocol as the baseline spec's evolution-scale section):

```bash
cargo run --release -p xagent-sandbox -- --headless --config experiments/control.json --db experiments/control.db
```

where `experiments/control.json` is the current default `FullConfig` (generate with `cargo run -p xagent-sandbox -- --dump-config > experiments/control.json`, then set `governor.tick_budget` to 120000, `governor.population_size` to 12, `world.seed` to 42). Compare `Food | Deaths | Food/1k-ticks` lines against the Phase-1 numbers in the baseline spec. Deaths should trend down relative to food if the terminal lesson is doing work.

- [ ] **Step 3: Record in the baseline spec and commit**

Append the numbers to the hazard-probe section of `docs/superpowers/specs/2026-06-10-learning-baseline.md`.

```bash
git add docs/superpowers/specs/2026-06-10-learning-baseline.md crates/xagent-sandbox/tests/integration.rs
git commit -m "doc: record hazard probe results after survival-signal grounding"
```

---

### Task 7: Quarter-split learning metric in headless runs

Within-lifetime improvement — food rate in the last quarter of a generation versus the first — is the number two reviews independently proposed as the missing "is anything being learned in one lifetime" signal, and the prerequisite for any fitness rework (Task 9). The generation loop already reads `cached_state` every chunk, so quarter sampling costs nothing.

**Files:**
- Modify: `crates/xagent-sandbox/src/headless.rs`

- [ ] **Step 1: Write the failing unit test (no GPU needed)**

At the bottom of `headless.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_rates_computes_first_and_last_quarter_food_rates() {
        // Cumulative (food, alive_ticks) samples at the four quarter
        // boundaries: q1 ate 4 in 1000 alive-ticks (rate 4.0/1k); the last
        // quarter ate 12−8 = 4 in 4000−3200 = 800 alive-ticks (rate 5.0/1k).
        let samples = [(4_u64, 1000_u64), (6, 2100), (8, 3200), (12, 4000)];
        let (first_quarter_rate, last_quarter_rate) = quarter_rates(&samples);
        assert!((first_quarter_rate - 4.0).abs() < 1e-9);
        assert!((last_quarter_rate - 5.0).abs() < 1e-9);
    }

    #[test]
    fn quarter_rates_handles_zero_alive_ticks() {
        let samples = [(0_u64, 0_u64), (0, 0), (0, 0), (0, 0)];
        let (first_quarter_rate, last_quarter_rate) = quarter_rates(&samples);
        assert_eq!(first_quarter_rate, 0.0);
        assert_eq!(last_quarter_rate, 0.0);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p xagent-sandbox quarter_rates`
Expected: FAIL — `quarter_rates` not defined.

- [ ] **Step 3: Implement**

```rust
/// Food rates (per 1k alive-ticks) of the first and last generation
/// quarters, from cumulative (food, alive_ticks) samples taken at the four
/// quarter boundaries. The last-quarter rate uses the deltas between the
/// third and fourth samples. Rising last-over-first is the direct signal
/// that the population improves within a lifetime instead of only across
/// generations.
fn quarter_rates(samples: &[(u64, u64); 4]) -> (f64, f64) {
    let (first_food, first_alive) = samples[0];
    let first_quarter_rate = if first_alive > 0 {
        first_food as f64 / first_alive as f64 * 1000.0
    } else {
        0.0
    };
    let last_food = samples[3].0.saturating_sub(samples[2].0);
    let last_alive = samples[3].1.saturating_sub(samples[2].1);
    let last_quarter_rate = if last_alive > 0 {
        last_food as f64 / last_alive as f64 * 1000.0
    } else {
        0.0
    };
    (first_quarter_rate, last_quarter_rate)
}
```

In `run_headless`'s generation loop, before `while ticks_done < tick_budget`:

```rust
        let quarter_length = (tick_budget / 4).max(1);
        let mut quarter_samples: [(u64, u64); 4] = [(0, 0); 4];
        let mut next_quarter: usize = 0;
```

Inside the loop, after the heatmap sampling block (where `state` is already in scope):

```rust
            while next_quarter < 4 && ticks_done >= quarter_length * (next_quarter as u64 + 1) {
                let mut cumulative_food = 0_u64;
                let mut cumulative_alive = 0_u64;
                for i in 0..agents.len() {
                    let base = i * PHYS_STRIDE;
                    cumulative_food += state[base + P_FOOD_COUNT] as u64;
                    cumulative_alive += state[base + P_TICKS_ALIVE] as u64;
                }
                quarter_samples[next_quarter] = (cumulative_food, cumulative_alive);
                next_quarter += 1;
            }
```

After the loop (the final state read already exists for fitness extraction), force the fourth sample from the final state, then print alongside the existing metrics line in `log_learning_metrics` by passing the two rates in:

```rust
        while next_quarter < 4 {
            let mut cumulative_food = 0_u64;
            let mut cumulative_alive = 0_u64;
            for i in 0..agents.len() {
                let base = i * PHYS_STRIDE;
                cumulative_food += state[base + P_FOOD_COUNT] as u64;
                cumulative_alive += state[base + P_TICKS_ALIVE] as u64;
            }
            quarter_samples[next_quarter] = (cumulative_food, cumulative_alive);
            next_quarter += 1;
        }
        let (first_quarter_rate, last_quarter_rate) = quarter_rates(&quarter_samples);
```

and extend the `println!` in `log_learning_metrics` (pass the two rates as parameters):

```rust
    println!(
        "  Food: {total_food} | Deaths: {total_deaths} | Food/1k-ticks: {food_per_1k:.3} \
         | Learn q1→q4: {first_quarter_rate:.3} → {last_quarter_rate:.3}{weight_norms}"
    );
```

- [ ] **Step 4: Run tests, gates, commit**

Run: `cargo test -p xagent-sandbox quarter_rates && cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test -p xagent-sandbox`

```bash
git add crates/xagent-sandbox/src/headless.rs
git commit -m "feat: print first-vs-last quarter food rates per generation in headless runs"
```

---

### Task 8: Stride/lag sweep at evolution scale + discount documentation

The mirrored probe shows lag removal alone does not unlock steering, so this is an A/B experiment with a TPS budget, not a confirmation exercise. The open-world question stands: at lag 100 an agent crosses 2× its vision range blind between frames, and the danger pathway built in Tasks 3–5 needs timely cues to matter in the wild.

**Files:**
- Create: `experiments/lag100-control.json`, `experiments/lag10.json`, `experiments/lag2.json`
- Modify: `crates/xagent-brain/src/shaders/kernel/common.wgsl` (comment only)

- [ ] **Step 1: Fix the `TD_DISCOUNT` comment (documentation accuracy, no value change)**

Replace the comment above `TD_DISCOUNT` in `common.wgsl`:

```wgsl
// Per-brain-tick discount. Horizon 1/(1−γ) ≈ 33 brain ticks ≈ 11 s of
// real time at the default strides (brain tick every 10 physics ticks at
// 30 Hz) — several food approaches long. A vision-edge approach itself is
// ~45 physics ticks ≈ 4.5 brain ticks at default speed; the horizon is
// intentionally longer so the critic bridges sparse encounters. At
// brain_tick_stride = 1 the same constant gives a 1.1 s horizon — if the
// default stride changes, recalibrate γ to keep the real-time horizon
// (γ = 1 − stride/330 approximately).
const TD_DISCOUNT: f32 = 0.97;
```

- [ ] **Step 2: Create the experiment configs**

Run: `cargo run -p xagent-sandbox -- --dump-config > experiments/lag100-control.json`

Edit the three files so all share `world.seed = 42`, `governor.tick_budget = 120000`, `governor.population_size = 12`, `governor.max_generations = 16`, and differ only in:

| File | `brain.brain_tick_stride` | `brain.vision_stride` | Sensory lag |
|---|---|---|---|
| `lag100-control.json` | 10 | 10 | 100 ticks |
| `lag10.json` | 2 | 5 | 10 ticks |
| `lag2.json` | 1 | 2 | 2 ticks |

- [ ] **Step 3: Run the three arms**

```bash
cargo run --release -p xagent-sandbox -- --headless --config experiments/lag100-control.json --db experiments/lag100.db
cargo run --release -p xagent-sandbox -- --headless --config experiments/lag10.json --db experiments/lag10.db
cargo run --release -p xagent-sandbox -- --headless --config experiments/lag2.json --db experiments/lag2.db
```

Collect per arm: `Food/1k-ticks` trend, `Learn q1→q4` trend (Task 7), `Deaths`, and `ticks/sec`.

- [ ] **Step 4: Decide and record**

Decision rule: adopt the smallest-lag configuration whose ticks/sec cost versus control is under ~30% **and** whose foraging or learn-delta trend beats control on the fixed seed. If a new default is adopted, change `default_brain_tick_stride` / `default_vision_stride` in `config.rs` (with the `brain_config_tuned_defaults` and `default_sensory_lag_is_within_bound` tests re-pinned) and recalibrate `TD_DISCOUNT` per the comment's formula in the same commit. If no arm wins, record the negative result in the baseline spec — that is itself the answer to the reviews' Experiment 1.

```bash
git add experiments/ docs/superpowers/specs/2026-06-10-learning-baseline.md crates/xagent-brain/src/shaders/kernel/common.wgsl
git commit -m "doc: record stride/lag sweep results and correct TD discount rationale"
```

---

### Task 9 (gated on Tasks 6–8 data): Fitness rework

Do not execute until the quarter metrics and hazard numbers exist. The data picks the variant:

- **Kamikaze confirmed** (deaths rise with foraging at evolution scale, terminal lesson notwithstanding) → Variant B.
- **Learning invisible** (q1→q4 flat while across-generation foraging rises) → Variant A.
- Both → B first, then A as a separate measured change. One change per measurement, per house discipline.

**Files:**
- Modify: `crates/xagent-sandbox/src/governor.rs:471-479`
- Modify: `crates/xagent-sandbox/src/headless.rs` and `crates/xagent-sandbox/src/agent/mod.rs` (Variant A plumbing)

**Variant B — survival becomes multiplicative (kamikaze foraging stops paying):**

```rust
            let survival = 1.0 / (1.0 + r.death_count as f32 * 0.5);
            let foraging = (r.food_consumed as f32 / food_target).min(1.0);
            // … exploration unchanged …
            // Survival gates the whole score multiplicatively: an agent
            // that forages by dying repeatedly no longer outscores one
            // that forages carefully. Weights inside the gate re-balance
            // foraging vs exploration to keep their prior 1:1 ratio.
            r.composite_fitness = survival * (foraging * 0.5 + exploration * 0.5);
```

**Variant A — within-lifetime improvement term:** add `food_rate_first_quarter: f32` and `food_rate_last_quarter: f32` to `Agent` (default 0.0), populate them in `run_headless` from the Task 7 per-agent quarter samples (lift the sampling from population-cumulative to per-agent: same loop, per-agent arrays), pipe them through the fitness record next to `food_consumed`, and change the composite to:

```rust
            // Improvement: last-quarter food rate versus first-quarter,
            // mapped to [0, 1] with 0.5 = no change. Selects for lineages
            // whose lifetimes end better than they start — learning —
            // rather than only for lineages that end well.
            let improvement_span = r.food_rate_first_quarter.max(0.05);
            let improvement = (((r.food_rate_last_quarter - r.food_rate_first_quarter)
                / improvement_span)
                .clamp(-1.0, 1.0))
                * 0.5
                + 0.5;
            r.composite_fitness =
                survival * 0.3 + foraging * 0.25 + exploration * 0.2 + improvement * 0.25;
```

Both variants need: the governor unit tests that assert composite math updated to the new formula (the `mock_fitness` helpers in `governor.rs` tests construct records directly — extend them with the new fields for Variant A), a fixed-seed 16-generation headless run against the Task 8 control numbers in the PR body, and a `doc:` note in the baseline spec.

---

### Task 10 (gated, candidate): Predictor forward-model objective fix

**Verified bug, honestly small expected effect.** Pass 7a trains this tick's prediction against this tick's input — an identity autoencoder objective — while novelty measures last tick's prediction against this tick's state. The objectives disagree; that is wrong on principle. But because the predictor has no efference copy (motor is not an input), the best achievable forward model in a mostly-static world is close to identity anyway — so expect consistency, not fireworks. Land it like Phase 2 of the 2026-06-10 plan: implement, measure, keep only if nothing regresses.

**Files:**
- Modify: `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`

- [ ] **Step 1: Move predictor training into pass 6, before the prediction**

In `coop_predict_and_act`, replace the predictor matmul block:

```wgsl
    // ── Predictor: train then predict — threads 0..PREDICTOR_DIMENSION ──
    // Train the forward model on the transition that just completed: the
    // prediction made last brain tick (O_PREV_PREDICTION, not yet
    // overwritten) against the state that actually arrived (s_encoded),
    // with the gradient flowing through last tick's input (O_PREV_ENCODED,
    // overwritten only at the end of pass 7). Training before predicting
    // keeps each row's reads and writes within one thread — no barrier.
    if (tid < PREDICTOR_DIMENSION) {
        let previous_prediction = brain_state[brain_base + O_PREV_PREDICTION + tid];
        let transition_error = previous_prediction - s_encoded[tid];
        let tanh_derivative = 1.0 - previous_prediction * previous_prediction;
        let predictor_learning_rate = bc_f32(CFG_LEARNING_RATE);
        for (var j: u32 = 0u; j < ENCODED_DIMENSION; j = j + 1u) {
            let previous_input = brain_state[brain_base + O_PREV_ENCODED + j];
            let grad = clamp(transition_error * tanh_derivative * previous_input, -1.0, 1.0);
            var w = brain_state[brain_base + O_PREDICTOR_WEIGHTS + tid * ENCODED_DIMENSION + j]
                - predictor_learning_rate * grad;
            w = clamp(w, -3.0, 3.0);
            brain_state[brain_base + O_PREDICTOR_WEIGHTS + tid * ENCODED_DIMENSION + j] = w;
        }
        var s: f32 = 0.0;
        for (var j: u32 = 0u; j < ENCODED_DIMENSION; j = j + 1u) {
            s += s_encoded[j] * brain_state[brain_base + O_PREDICTOR_WEIGHTS + tid * ENCODED_DIMENSION + j];
        }
        s_prediction[tid] = s;
    }
```

Note the predictor now reads and predicts `s_encoded` (pre-habituation), not `s_habituated`: the forward model learns world dynamics; habituation remains the attention layer downstream. This also makes the recalled-context blend consistent — patterns store encoded-space vectors.

- [ ] **Step 2: Make all error computations target `s_encoded` and compute the error once**

In pass 6 thread 0: change the prediction-error loop target from `s_habituated[d]` to `s_encoded[d]`. Delete the second error computation near the end of thread 0's block (the loop that recomputes `error_squared_sum` against `s_habituated` and overwrites `s_pred_error`) — `s_pred_error` keeps the value set from the true forward error.

In pass 7: delete block 7a entirely (training moved to pass 6). In the thread-0 context-weight block, replace the recomputed `error_mag` with `s_pred_error`:

```wgsl
    // Thread 0: context weight adaptation, driven by the same forward
    // prediction error that drives novelty.
    if (tid == 0u) {
        brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT] +=
            learning_rate * 0.01 * (s_pred_error - 0.5);
        brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT] = clamp(
            brain_state[brain_base + O_PREDICTOR_CONTEXT_WEIGHT], 0.05, 0.5);
    }
```

- [ ] **Step 3: Move the `O_PREV_ENCODED` overwrite from pass 3 to the end of pass 7**

In `coop_habituate_homeo`, delete the line `brain_state[brain_base + O_PREV_ENCODED + tid] = enc;` (the read above it stays — habituation still compares against last tick's encoded state, same value as before).

At the very end of `coop_learn_and_store`, append:

```wgsl
    // ── 7g. Publish this tick's encoded state for the next tick's
    // habituation delta and predictor training input ────────────────────
    if (tid < ENCODED_DIMENSION) {
        brain_state[brain_base + O_PREV_ENCODED + tid] = s_encoded[tid];
    }
```

- [ ] **Step 4: Regression gate**

Run: `cargo test -p xagent-sandbox` — every probe must stay green, including the mirrored steering band and `td_critic_tracks_metabolic_drain`.

Then re-run the fixed-seed headless control (Task 8 protocol) and the hazard probe. Keep the change only if no metric regresses; revert wholesale otherwise and record the negative result (same discipline as the Phase-2 reconstruction revert). Record before/after `prediction_error` telemetry ranges in the baseline spec — the forward error should now *decrease* over a lifetime in quiet stretches instead of hovering at the state-change magnitude.

```bash
git add crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl docs/superpowers/specs/2026-06-10-learning-baseline.md
git commit -m "fix: train predictor as forward model on completed transitions"
```

---

## Explicitly deferred (documented non-scope, with the measurement that would justify each)

- **Efference copy for the predictor** (motor as predictor input): the principled completion of Task 10 — without it the forward model cannot predict self-caused change. Layout change (`O_PREDICTOR_WEIGHTS` grows by 2 columns, `BRAIN_STRIDE` shifts, inheritance blobs invalidate). Justify with: Task 10's recorded prediction-error still dominated by self-motion.
- **Memory store gating** (store only on |valence| above the metabolic noise floor, or on eat/damage events): today every brain tick stores a pattern with valence ≈ −0.001, churning the 128 slots in ~42 s. Justify with: recall-valence separation diagnostic (do food-state patterns hold positive valence at all under churn?).
- **Eligibility magnitude alignment** (record noise after fatigue/klinotaxis scaling): sign is preserved today, only magnitude mismatches. Justify with: mirrored-probe sensitivity sweep once anything moves it off chance.
- **Exploration floor decay** (Codex Experiment 6: lower the 0.10 floor once valence memories stabilize): masks learned policy if one exists; currently nothing to mask per the mirrored probe.
- **Vision alpha channel** (constant 1.0, 48 dead features): repurposing it changes `FEATURE_COUNT` consumers and CPU readback; bundle with the next vision-layout change (17×13 default flip), which is itself gated on directional steering existing.
- **Encoder changes of any kind**: measured non-binding twice (Phase-2 revert; separability diagnostic). Touch only after a probe regression implicates representation.

## Validation discipline

Unchanged from the 2026-06-10 plan: every task lands with before/after probe numbers in the PR body, fixed seeds, `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, full `cargo test -p xagent-sandbox` green. No task merges on "should work". GPU tests self-skip without an adapter; CI runs lavapipe.
