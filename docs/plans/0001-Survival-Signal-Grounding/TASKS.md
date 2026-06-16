# XAgent Plan 0001 — Survival Signal Grounding

Make "danger = bad" learnable: pin a danger-avoidance baseline, give the
TD(λ) learner a terminal death lesson through its existing traces, emit
hazard/edge touch contacts on the live GPU path, feed interoception
same-cycle, make within-lifetime learning visible per generation, sweep the
sensory lag under a TPS budget, and hold the fitness and predictor changes
behind the data those measurements produce.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md)
for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/{plan_slug}--{id}/`).
- **Depends on** lists *direct* prerequisites only (structural or
  measurement-ordering — this plan is measurement-first, so probe tasks gate
  the changes they measure).
- **Done when** is the verifiable acceptance check. Every task must keep
  `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, and
  `cargo test -p xagent-sandbox` green.
- GPU tests self-skip without an adapter (`GpuKernel::is_available()`); CI
  runs Mesa lavapipe.
- Line numbers are hints; locate every site by the named symbol (grep).

---

## 0001 — Hazard observability and baseline

### sensory-tail-telemetry — Expose the non-visual sensory tail

`read_agent_telemetry_blocking` (`gpu_kernel.rs:1888-1903`) already reads the
full `sensory_stride` floats per agent and keeps only the vision colors.
Touch contacts and interoception are invisible to tests and the UI.

**Steps:**

1. In `crates/xagent-brain/src/gpu_kernel.rs`, add to `AgentTelemetry`
   (after `vision_color`):

   ```rust
   /// Non-visual sensory tail exactly as packed by `phase_vision_senses`:
   /// [velocity(3), facing(3), angular(1), energy, integrity,
   ///  energy_delta, integrity_delta, touch(4 contacts × 4)].
   /// One vision batch stale, like all of `sensory_buffer`.
   pub sensory_non_visual: Vec<f32>,
   ```

2. In `read_agent_telemetry_blocking`, after the `vision_color` slice:

   ```rust
   let non_visual_base = self.layout.vision_color_count + self.layout.vision_depth_count;
   let sensory_non_visual: Vec<f32> = sensory[non_visual_base..].to_vec();
   ```

   and add `sensory_non_visual,` to the `AgentTelemetry { ... }`
   construction.

3. Add the test in `crates/xagent-sandbox/tests/integration.rs` (Learning
   Probe Tests section):

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
       // Energy is packed normalized at index 7 of the tail and the agent
       // is alive at full-ish energy after one tick.
       let energy_normalized = telemetry.sensory_non_visual[7];
       assert!(
           (0.5..=1.0).contains(&energy_normalized),
           "normalized energy {energy_normalized} not in (0.5, 1.0] after one tick"
       );
   }
   ```

- **Depends on:** —
- **Done when:** the test fails before step 1–2 (missing field) and passes
  after; cargo fmt/clippy/test green.

### hazard-probe-baseline — Pin the danger-avoidance baseline

The probe harness pins food-approach and foraging baselines in hazard-free
arenas; nothing measures hazard-exit latency or death economics, so the
grounding changes in workstream 0002 have no gate. House rule (plan
2026-06-10, Phase 0): the gates come first.

**Steps:**

1. In `crates/xagent-sandbox/tests/integration.rs`, add a "Hazard Probe
   Tests" section after the Learning Probe Tests:

   ```rust
   // ── Hazard Probe Tests ──────────────────────────────────────────────
   //
   // A half-plane danger arena: danger biome for x < 0, food-rich for
   // x ≥ 0. Agents start 10 units inside the danger side, facing +Z
   // (parallel to the boundary, so straight-line walking never exits on
   // its own). Measured: hazard-exit latency (first tick with x ≥ 0,
   // alive, without dying first) and deaths. These pin the
   // danger-avoidance baseline the same way the mirrored steering probe
   // pins food-approach.

   /// Distance agents start inside the danger half-plane. Far enough that
   /// exit requires sustained directed movement (~15 ticks of straight-east
   /// walking at default speed), close enough that random walks exit within
   /// the episode often enough to measure a latency distribution.
   const HAZARD_PROBE_START_DEPTH: f32 = 10.0;
   /// Episode cap in physics ticks. At hazard damage 0.5/tick (rate 1.0 ×
   /// integrity_scale 0.5) an agent that never exits dies at tick 200, so
   /// 600 ticks cleanly separates "exited", "died", and "wandered".
   const HAZARD_PROBE_EPISODE_TICKS: u64 = 600;
   /// Position sampling interval — bounds latency resolution and readback
   /// cost.
   const HAZARD_PROBE_SAMPLE_TICKS: u32 = 5;
   /// Episodes per measurement.
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

   /// Build the half-plane danger arena on top of the standard probe
   /// arena: biome column < 128 (x < 0) is danger, the rest food-rich;
   /// agents are re-positioned to x = −HAZARD_PROBE_START_DEPTH, spread
   /// along z.
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

   /// Baseline: untrained agents in the hazard arena. Prints exit
   /// fraction, mean exit latency, and death fraction; asserts structural
   /// sanity plus pinned falsifiable bands. Re-pin the bands when
   /// workstream 0002 improves escape (the same protocol as the steering
   /// probes).
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

       // Structural sanity: every trial resolves into exit, death, or
       // timeout.
       assert!(exits.len() + deaths <= trials, "double-counted outcomes");
       // Falsifiable floor: the arena must actually be dangerous — if
       // nothing ever dies and everything exits instantly, the geometry
       // broke.
       assert!(
           death_fraction > 0.0 || mean_latency > 50.0,
           "arena is not hazardous: death_fraction={death_fraction}, \
            mean_exit_latency={mean_latency}"
       );
   }
   ```

2. Run it and record the printed numbers:

   ```bash
   cargo test -p xagent-sandbox --test integration hazard_probe_exit_latency_baseline -- --nocapture
   ```

3. Pin the baseline: add `assert!((low..=high).contains(&metric))` lines for
   the three recorded values with ±50%-relative bands (generous enough that
   adapter noise does not flake, tight enough that a real avoidance
   improvement trips them), with a comment recording the date, adapter, and
   raw values.

4. Append a "Hazard probe baseline" subsection to
   `docs/superpowers/specs/2026-06-10-learning-baseline.md` with the three
   numbers, the protocol summary, and the gate: *workstream 0002 must move
   mean exit latency or death fraction outside the pinned bands to claim a
   behavioral win.*

- **Depends on:** —
- **Done when:** the probe passes with pinned bands committed around real
  recorded values; the baseline spec carries the numbers and the gate; cargo
  fmt/clippy/test green.

---

## 0002 — Survival-signal grounding

### terminal-death-update — Apply one terminal TD update through the dying traces

`agent_death_respawn` zeroes the traces and `O_PREV_VALUE` with no terminal
evaluation (`kernel_tick.wgsl:383-391`) and restores full energy in the same
cycle — the transition into death never produces a TD error, so the single
strongest "danger = bad" event teaches nothing.

**Steps:**

1. In `crates/xagent-brain/src/shaders/kernel/common.wgsl`, add after
   `MAX_TD_ERROR`:

   ```wgsl
   // Terminal TD error applied through the dying episode's eligibility
   // traces at the moment of death, before they are cleared for the next
   // life. Death must be the single worst lesson the learner can receive,
   // but never stronger than the per-transition bound that protects
   // against artifacts.
   const TERMINAL_DEATH_TD_ERROR: f32 = -MAX_TD_ERROR;
   ```

2. In `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl`
   (`agent_death_respawn`), immediately above the `// Reset TD transients`
   comment block, insert:

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

   `agent_death_respawn` is thread-0-only with no barriers — barrier
   uniformity is untouched. The weight L2-ball clamps run in the next brain
   tick's pass 6.

3. Apply the identical insertion in
   `crates/xagent-brain/src/shaders/kernel/phase_death.wgsl` before its
   trace zeroing (grep `O_TRACE_CRITIC` there) — the standalone pipeline
   must stay behaviorally identical to the fused one.

4. Add the test in `crates/xagent-sandbox/tests/integration.rs`:

   ```rust
   /// Dying must apply one terminal TD update (δ = −MAX_TD_ERROR) through
   /// the dying life's eligibility traces before they are cleared. With
   /// preset traces the kick is exactly computable:
   /// Δvalue_bias = 0.01·(−1)·5 = −0.05 and Δactor_bias = 0.1·(−1)·1 =
   /// −0.10. The post-respawn brain tick in the same cycle applies δ
   /// through freshly zeroed traces, so it cannot move the biases — any
   /// deviation from the exact kick is a real defect.
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

       /// Hazard damage is rate (1.0) × integrity_scale per physics tick;
       /// 200 wipes the full 100 integrity in a single tick inside a
       /// danger biome.
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

       // This tick kills (integrity 100 → 0), respawns, and runs one
       // post-respawn brain tick whose traces were just zeroed — so the
       // only bias change in this tick is the terminal kick.
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

- **Depends on:** hazard-probe-baseline
- **Done when:** the new test fails before the WGSL change and passes after;
  `td_traces_bounded_across_deaths` still passes (traces still zeroed after
  the kick); the mirrored steering probe band holds (no deaths in that
  arena); cargo fmt/clippy/test green.

### hazard-edge-touch — Emit TOUCH_HAZARD and TOUCH_TERRAIN_EDGE on the GPU

The live GPU sensory path emits only food and agent contacts
(`phase_vision.wgsl:224-311`); `TOUCH_HAZARD`/`TOUCH_TERRAIN_EDGE` exist in
`common.wgsl:213-214` and the CPU reference emits them
(`agent/senses.rs::detect_touch`). Danger needs a touch channel like food
has.

**Steps:**

1. In `crates/xagent-brain/src/shaders/kernel/common.wgsl` (touch constants
   block), add:

   ```wgsl
   // Hazard contacts have no meaningful planar direction (the hazard is
   // the terrain underfoot), so they carry a fixed mid-scale intensity
   // instead of a closeness value. Matches the CPU reference in
   // agent/senses.rs.
   const TOUCH_HAZARD_INTENSITY: f32 = 0.5;
   ```

2. In `phase_vision_senses` (`phase_vision.wgsl`), insert the hazard block
   between the slot-zeroing loop and the food scan:

   ```wgsl
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

3. After the existing agent-contact loop (end of the function), append the
   wall checks:

   ```wgsl
   // Terrain-edge contacts: the world boundary pushes back. Direction
   // points inward (away from the wall), intensity rises as the wall
   // nears — mirrors the CPU reference in agent/senses.rs.
   let world_half_for_touch = wc_f32(WC_WORLD_HALF_BOUND);
   // `var` (not `let`): naga requires a mutable binding for dynamic
   // indexing.
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

   `phase_vision_senses` is one-thread-per-agent with no barriers — no
   uniformity concerns. The buffer layout does not change.

4. Add the tests in `crates/xagent-sandbox/tests/integration.rs`:

   ```rust
   /// Standing in a danger biome must produce a TOUCH_HAZARD contact in
   /// the sensory buffer: zero planar direction (the hazard is underfoot),
   /// fixed intensity, tag 3/4. Mirrors the CPU reference in
   /// agent/senses.rs.
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
       // Touch slots start after [vel(3), facing(3), angular(1),
       // interoception(4)] = 11.
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
       // World half-bound is world_size/2 = 128; 1.5 units from the +X
       // wall.
       arena.agent_data[0].0 = glam::Vec3::new(126.5, PROBE_AGENT_Y, 0.0);
       arena.reset_bodies();
       arena.kernel.dispatch_batch(0, 1);

       let telemetry = arena.kernel.read_agent_telemetry_blocking(0);
       let touch_base = 11;
       let mut edge_slot: Option<&[f32]> = None;
       for contact in 0..4 {
           let slot = &telemetry.sensory_non_visual
               [touch_base + contact * 4..touch_base + contact * 4 + 4];
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

- **Depends on:** sensory-tail-telemetry, hazard-probe-baseline
- **Done when:** both tests fail before the WGSL change (tags never appear)
  and pass after; the mirrored steering and foraging probes hold their bands
  (their arenas are hazard-free and away from walls — if a band trips, stop
  and investigate); cargo fmt/clippy/test green.

### same-cycle-interoception — Feel pain at decision time

The four interoception features (energy, integrity, deltas) reach
`s_features` only via the batch-lagged `sensory_buffer`
(`brain_passes.wgsl:70-109`), while the reward path already reads
`physics_state` same-cycle (`brain_passes.wgsl:148-183`). The state the
policy conditions on lags the reward it is blamed for by up to 100 physics
ticks.

**Steps:**

1. In `coop_feature_extract` (`brain_passes.wgsl`, thread-0 block), replace
   the contiguous five-value read at `ang_offset` with:

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

   The `touch_offset` derivation is unchanged — the `sensory_buffer` layout
   does not move, only the source of four `s_features` values.

2. Run the full suite. No unit test can observe `s_features` directly
   (workgroup memory); the behavioral gate is the hazard-probe re-measure
   (`hazard-grounding-remeasure`). The existing probes must not regress —
   verify, don't assume.

- **Depends on:** hazard-probe-baseline
- **Done when:** the full suite is green including all pinned probe bands;
  cargo fmt/clippy/test green.

### hazard-grounding-remeasure — Re-measure and record the verdict

**Steps:**

1. Re-run the hazard probe with all three grounding changes in place:

   ```bash
   cargo test -p xagent-sandbox --test integration hazard_probe_exit_latency_baseline -- --nocapture
   ```

   Three outcomes:
   - Improved beyond the pinned bands → re-pin the bands at the new values
     and record the before/after pair.
   - Unchanged → the bands hold; record that grounding alone did not
     produce avoidance (a real possibility per the mirrored-probe evidence —
     the danger pathway now exists, which is the prerequisite, not the
     guarantee).
   - Worsened → stop, bisect by reverting `terminal-death-update`,
     `hazard-edge-touch`, `same-cycle-interoception` one at a time, and
     record which change regressed.

2. Evolution-scale check — generate a control config and run 16 generations:

   ```bash
   cargo run -p xagent-sandbox -- --dump-config > experiments/control.json
   # edit: governor.tick_budget = 120000, governor.population_size = 12,
   #       governor.max_generations = 16, world.seed = 42
   cargo run --release -p xagent-sandbox -- --headless --config experiments/control.json --db experiments/control.db
   ```

   Compare `Food | Deaths | Food/1k-ticks` lines against the Phase-1 numbers
   in the baseline spec. Deaths should trend down relative to food if the
   terminal lesson is doing work.

3. Append the numbers to the hazard-probe section of
   `docs/superpowers/specs/2026-06-10-learning-baseline.md`.

- **Depends on:** terminal-death-update, hazard-edge-touch,
  same-cycle-interoception
- **Done when:** the baseline spec carries the after-numbers and a verdict
  (improved / unchanged / regressed-and-bisected); pinned bands re-pinned if
  improved; cargo fmt/clippy/test green.

---

## 0003 — Learning visibility and lag economics

### quarter-learning-metric — Print first-vs-last quarter food rates per generation

Within-lifetime improvement — food rate in the last quarter of a generation
versus the first — is the missing "is anything being learned in one
lifetime" signal and the prerequisite for any fitness rework. The headless
generation loop (`headless.rs:173-208`) already reads `cached_state` every
chunk, so quarter sampling is free.

**Steps:**

1. In `crates/xagent-sandbox/src/headless.rs`, add the pure helper and unit
   tests (no GPU needed):

   ```rust
   /// Food rates (per 1k alive-ticks) of the first and last generation
   /// quarters, from cumulative (food, alive_ticks) samples taken at the
   /// four quarter boundaries. The last-quarter rate uses the deltas
   /// between the third and fourth samples. Rising last-over-first is the
   /// direct signal that the population improves within a lifetime instead
   /// of only across generations.
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

   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn quarter_rates_computes_first_and_last_quarter_food_rates() {
           // Cumulative (food, alive_ticks) samples at the four quarter
           // boundaries: q1 ate 4 in 1000 alive-ticks (rate 4.0/1k); the
           // last quarter ate 12−8 = 4 in 4000−3200 = 800 alive-ticks
           // (rate 5.0/1k).
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

2. In `run_headless`'s generation loop, before the tick loop:

   ```rust
   let quarter_length = (tick_budget / 4).max(1);
   let mut quarter_samples: [(u64, u64); 4] = [(0, 0); 4];
   let mut next_quarter: usize = 0;
   ```

   Inside the loop, after the heatmap sampling block (where `state` is
   already in scope):

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

   After the loop, where the final state is read for fitness extraction,
   fill any remaining samples from the final state the same way, then:

   ```rust
   let (first_quarter_rate, last_quarter_rate) = quarter_rates(&quarter_samples);
   ```

3. Pass the two rates into `log_learning_metrics` and extend its `println!`:

   ```rust
   println!(
       "  Food: {total_food} | Deaths: {total_deaths} | Food/1k-ticks: {food_per_1k:.3} \
        | Learn q1→q4: {first_quarter_rate:.3} → {last_quarter_rate:.3}{weight_norms}"
   );
   ```

- **Depends on:** —
- **Done when:** both unit tests pass without a GPU; a headless run prints
  the `Learn q1→q4` figures; cargo fmt/clippy/test green.

### stride-lag-sweep — Three-arm lag A/B at evolution scale + discount comment fix

The mirrored steering probe trains at lag 1 and stays at chance, so lag
removal is not the expected fix — but at lag 100 an agent crosses 2× its
vision range blind between frames, and the workstream-0002 cues need
timeliness in the wild. Budgeted experiment, decision rule locked in SCOPE.

**Steps:**

1. In `common.wgsl`, replace the comment above `TD_DISCOUNT` (value
   unchanged):

   ```wgsl
   // Per-brain-tick discount. Horizon 1/(1−γ) ≈ 33 brain ticks ≈ 11 s of
   // real time at the default strides (brain tick every 10 physics ticks
   // at 30 Hz) — several food approaches long. A vision-edge approach
   // itself is ~45 physics ticks ≈ 4.5 brain ticks at default speed; the
   // horizon is intentionally longer so the critic bridges sparse
   // encounters. At brain_tick_stride = 1 the same constant gives a 1.1 s
   // horizon — if the default stride changes, recalibrate γ to keep the
   // real-time horizon (γ ≈ 1 − stride/330).
   const TD_DISCOUNT: f32 = 0.97;
   ```

2. Create the experiment configs:

   ```bash
   cargo run -p xagent-sandbox -- --dump-config > experiments/lag100-control.json
   cp experiments/lag100-control.json experiments/lag10.json
   cp experiments/lag100-control.json experiments/lag2.json
   ```

   Edit all three to share `world.seed = 42`,
   `governor.tick_budget = 120000`, `governor.population_size = 12`,
   `governor.max_generations = 16`, differing only in:

   | File | `brain.brain_tick_stride` | `brain.vision_stride` | Lag |
   |---|---|---|---|
   | `lag100-control.json` | 10 | 10 | 100 |
   | `lag10.json` | 2 | 5 | 10 |
   | `lag2.json` | 1 | 2 | 2 |

3. Run the three arms:

   ```bash
   cargo run --release -p xagent-sandbox -- --headless --config experiments/lag100-control.json --db experiments/lag100.db
   cargo run --release -p xagent-sandbox -- --headless --config experiments/lag10.json --db experiments/lag10.db
   cargo run --release -p xagent-sandbox -- --headless --config experiments/lag2.json --db experiments/lag2.db
   ```

   Collect per arm: `Food/1k-ticks` trend, `Learn q1→q4` trend, `Deaths`,
   `ticks/sec`.

4. Decide per the locked rule: adopt the smallest-lag arm whose ticks/sec
   cost versus control is under ~30% and which beats control on the fixed
   seed. If a new default is adopted: change `default_brain_tick_stride` /
   `default_vision_stride` in `crates/xagent-shared/src/config.rs`, re-pin
   `brain_config_tuned_defaults` and `default_sensory_lag_is_within_bound`,
   and recalibrate `TD_DISCOUNT` per the comment's formula — all in the same
   commit. If no arm wins, record the negative result in the baseline spec;
   that is itself the answer to the reviews' lag experiment.

- **Depends on:** quarter-learning-metric
- **Done when:** the corrected comment is in; three result sets are recorded
  in the baseline spec with a decision (new default with re-pinned tests +
  recalibrated γ, or a documented negative); cargo fmt/clippy/test green.

---

## 0004 — Gated follow-ups

### fitness-rework — Make the outer loop demand what we want (GATED)

**Gate:** do not start until `hazard-grounding-remeasure` and
`stride-lag-sweep` data exist. The data picks the variant (SCOPE locked
decisions): kamikaze confirmed → Variant B; learning invisible → Variant A;
both → B first, then A, separately measured.

Today (`governor.rs:473-479`):

```rust
let survival = 1.0 / (1.0 + r.death_count as f32 * 0.5);
let foraging = (r.food_consumed as f32 / food_target).min(1.0);
// …
r.composite_fitness = survival * 0.4 + foraging * 0.3 + exploration * 0.3;
```

**Steps:**

1. **Variant B — multiplicative survival:**

   ```rust
   // Survival gates the whole score multiplicatively: an agent that
   // forages by dying repeatedly no longer outscores one that forages
   // carefully. Weights inside the gate re-balance foraging vs
   // exploration to keep their prior 1:1 ratio.
   r.composite_fitness = survival * (foraging * 0.5 + exploration * 0.5);
   ```

2. **Variant A — improvement term:** add `food_rate_first_quarter: f32` and
   `food_rate_last_quarter: f32` to `Agent` (default 0.0); populate them in
   `run_headless` by lifting the quarter sampling from population-cumulative
   to per-agent (same loop, per-agent arrays); thread them into the fitness
   record next to `food_consumed`; then:

   ```rust
   // Improvement: last-quarter food rate versus first-quarter, mapped to
   // [0, 1] with 0.5 = no change. Selects for lineages whose lifetimes
   // end better than they start — learning — rather than only for
   // lineages that end well.
   let improvement_span = r.food_rate_first_quarter.max(0.05);
   let improvement = (((r.food_rate_last_quarter - r.food_rate_first_quarter)
       / improvement_span)
       .clamp(-1.0, 1.0))
       * 0.5
       + 0.5;
   r.composite_fitness =
       survival * 0.3 + foraging * 0.25 + exploration * 0.2 + improvement * 0.25;
   ```

3. Update the governor composite unit tests (the `mock_fitness` /
   `mock_multi_fitness` helpers construct records directly — extend them
   with the new fields for Variant A) so the formula is asserted exactly.

4. Run a fixed-seed 16-generation headless comparison against the
   `stride-lag-sweep` control and record before/after in the baseline spec
   and the PR body.

- **Depends on:** quarter-learning-metric, hazard-grounding-remeasure,
  stride-lag-sweep
- **Done when:** one variant is merged with updated unit tests asserting the
  new formula, a fixed-seed comparison recorded, and the baseline spec
  updated; one change per measurement; cargo fmt/clippy/test green.

### predictor-forward-objective — Train the predictor as a forward model (GATED)

**Gate:** lands Phase-2-style — keep only if every probe band holds and the
fixed-seed headless control does not regress; revert wholesale otherwise and
record the negative result. Expected effect is honestly small (no efference
copy → near-identity is near-optimal in quiet stretches); the claim is
objective consistency, not behavior.

Today pass 7a trains *this tick's* prediction against *this tick's* input
(`brain_passes.wgsl:719-730`) — an identity-autoencoder objective — while
novelty compares *last tick's* prediction against the current state
(`brain_passes.wgsl:323-330`).

**Steps:**

1. In `coop_predict_and_act` (`brain_passes.wgsl`), replace the predictor
   matmul block with train-then-predict (threads
   0..`PREDICTOR_DIMENSION`):

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

   The predictor now reads and predicts `s_encoded` (pre-habituation): the
   forward model learns world dynamics, habituation stays the attention
   layer downstream, and the recalled-context blend becomes consistent
   (patterns store encoded-space vectors).

2. One error, computed once: in pass 6 thread 0, change the novelty-error
   loop target from `s_habituated[d]` to `s_encoded[d]`; delete the
   duplicate end-of-block error recomputation that overwrites
   `s_pred_error` (it keeps the value from the true forward error). In pass
   7, delete block 7a (training moved to pass 6) and feed the context-weight
   adaptation from the shared value:

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

3. Move the `O_PREV_ENCODED` overwrite: delete
   `brain_state[brain_base + O_PREV_ENCODED + tid] = enc;` from
   `coop_habituate_homeo` (the read above it stays — habituation compares
   against the same value it always did), and append at the very end of
   `coop_learn_and_store`:

   ```wgsl
   // ── 7g. Publish this tick's encoded state for the next tick's
   // habituation delta and predictor training input ─────────────────────
   if (tid < ENCODED_DIMENSION) {
       brain_state[brain_base + O_PREV_ENCODED + tid] = s_encoded[tid];
   }
   ```

4. Regression gate: full suite green including the mirrored steering band
   and `td_critic_tracks_metabolic_drain`; re-run the hazard probe and the
   fixed-seed headless control; record before/after `prediction_error`
   telemetry ranges in the baseline spec — the forward error should now
   decrease over a lifetime in quiet stretches instead of hovering at the
   state-change magnitude.

- **Depends on:** hazard-probe-baseline, hazard-grounding-remeasure
- **Done when:** either merged with all bands holding and the control
  comparison recorded, or reverted wholesale with the negative result
  recorded in the baseline spec; cargo fmt/clippy/test green.

---

**End of plan 0001 TASKS.** When every "Done when" bullet is green, death is
the worst lesson an agent can receive instead of a free heal, hazards are
felt through touch and same-cycle interoception the way food already is,
danger avoidance has a pinned probe the way foraging does, every headless
generation reports whether lifetimes end better than they start, the lag
question has a measured answer instead of a hypothesis, and the fitness and
predictor changes have either landed behind data or been recorded as
negatives.
