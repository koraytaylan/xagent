# XAgent Plan 0004 — Approach Reward Shaping

Give the within-lifetime reward a spatial approach gradient: pin the pre-change
alignment and foraging baselines, surface nearest-food-within-shaping-radius from
the food-detect pass, add potential-based shaping `F = γΦ(s′) − Φ(s)` into
`raw_gradient`, split the actor's vector scale from the critic's `1/128`, and
re-measure against a hard prove-or-kill rule; then restore the selection signal
(foraging-primary fitness, decoupled experiment resolution, a significance guard,
a governor-path within-life metric); and gate the reactive/valence rewiring, the
heritable learning constants, and the lag/vision re-measure behind the one
measurement that proves the unlock.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md)
for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/{plan_slug}--{id}/`).
- **Depends on** lists *direct* prerequisites only (`—` means none).
- **Done when** is the verifiable acceptance check. Every task must keep
  `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, and
  `cargo test -p xagent-sandbox` green (stated as "cargo fmt/clippy/test green")
  unless the task explicitly says it is documentation-only.
- GPU tests self-skip without an adapter (`GpuKernel::is_available()`); CI runs
  Mesa lavapipe.
- Line numbers are hints; locate every site by the named symbol (grep).

---

## 0001 — Approach reward shaping

### approach-shaping-baseline — Pin the pre-change alignment and foraging numbers

The diagnosis predicts that adding an approach gradient moves turn/bearing
alignment off chance and lifts free-run foraging. Both baselines already have
probes (`learning_probe_baseline_turn_alignment_is_chance`,
`integration.rs:1992`; `learning_probe_free_run_foraging_baseline`, `:2031`); this
task records their current numbers as the "before" the `0001` remeasure compares
against, so the unlock is judged against pinned numbers, not memory.

**Steps:**

1. Run the two probes and capture their `eprintln!` output:

   ```bash
   cargo test -p xagent-sandbox --test integration -- --nocapture \
     learning_probe_baseline_turn_alignment_is_chance \
     learning_probe_free_run_foraging_baseline
   ```

2. Record, in the PR body and in a new dated subsection of
   `docs/superpowers/specs/2026-06-10-learning-baseline.md`: the alignment
   `correct/scored = rate` (expected ≈ 0.5, inside `[0.38, 0.62]`) and the
   free-run `food=… deaths=… food/agent/1k-ticks=…`, plus the run's HEAD sha.
3. Note the chance band edge `0.62` (`integration.rs:2019-2023`) verbatim as the
   threshold the remeasure must beat.

- **Depends on:** —
- **Done when:** the two baseline numbers and the `0.62` threshold are recorded in
  the baseline spec subsection and PR body; no code changed; cargo fmt/clippy/test
  green. (Measurement task — no new behavioral unit.)

### visible-food-potential-input — Surface nearest-food-within-shaping-radius

`agent_food_detect` (`kernel_tick.wgsl:201-277`) reduces only the nearest food
within `eat_radius` (`:232`), for eating. The approach potential needs the
nearest food within a wider navigational `SHAPING_RADIUS`. Add a parallel
reduction and publish it (plus a slot for the previous potential) to new
`physics_state` slots the homeostasis pass reads in the same kernel cycle.

**Steps:**

1. In `crates/xagent-brain/src/buffers.rs`, append two slots after
   `P_LAST_DEATH_TICK` (`:158`) and bump `PHYS_STRIDE` (`:159`) to `34`:

   ```rust
   /// Distance (world units) to the nearest food within SHAPING_RADIUS, written
   /// by `agent_food_detect`; SHAPING_RADIUS when none in range. Read by the
   /// homeostasis pass to form the approach potential Φ.
   pub const P_NEAREST_FOOD_DIST: usize = 32;
   /// Previous brain tick's approach potential Φ(s); reset on respawn.
   pub const P_PREV_POTENTIAL: usize = 33;
   pub const PHYS_STRIDE: usize = 34;
   ```

   Add both to the `PHYS_STRIDE`-mirror assertions (`buffers.rs:824`, the slot
   list around `:1017-1039`, and the "highest offset + 1" guard at `:1036-1039`).
2. In `crates/xagent-brain/src/shaders/kernel/common.wgsl`, add
   `const SHAPING_RADIUS: f32 = 30.0;` (= `VISION_MAX_DIST`, `:204`), and confirm
   the WGSL `P_NEAREST_FOOD_DIST`/`P_PREV_POTENTIAL`/`PHYS_STRIDE` constants are
   injected from Rust (the kernel reads `PHYS_STRIDE` etc. via the host-substituted
   constants; update wherever the other `P_*` constants are emitted).
3. In `agent_food_detect` (`kernel_tick.wgsl:201-277`), carry a second per-thread
   minimum `local_best_shaping_dist_sq` (init `1e12`) over `SHAPING_RADIUS²` with
   *no* `eat_radius` gate, reduce it through the same two-phase shared-memory
   pattern as the eat candidate (`:239-266`) — reusing the barriers, adding a
   parallel pair of shared arrays or a second pass over the existing ones — and
   in the `tid == 0u && alive` block write
   `physics_state[b + P_NEAREST_FOOD_DIST] = select(SHAPING_RADIUS, sqrt(best),
   best < SHAPING_RADIUS * SHAPING_RADIUS)`. Leave the eat path (`:267-275`)
   unchanged.
4. In `agent_death_respawn` (`kernel_tick.wgsl:283-348`), inside the thread-0
   restore block (`:318-346`), set
   `physics_state[base + P_PREV_POTENTIAL] = 0.0;` and
   `physics_state[base + P_NEAREST_FOOD_DIST] = SHAPING_RADIUS;`.
5. Add a GPU integration test to `crates/xagent-sandbox/tests/integration.rs` that
   places one food at a known distance `< SHAPING_RADIUS` from a probe agent, runs
   one batch, and asserts `P_NEAREST_FOOD_DIST` reads ≈ that distance (and
   `SHAPING_RADIUS` when the food is removed), embedding the guard verbatim:

   ```rust
   if !xagent_brain::GpuKernel::is_available() {
       eprintln!("Skipping: no GPU/fallback adapter available");
       return;
   }
   ```

- **Depends on:** —
- **Done when:** `P_NEAREST_FOOD_DIST` reports the nearest-food distance within
  `SHAPING_RADIUS` (sentinel when none), the new test passes, the `PHYS_STRIDE`
  guard tests pass, and `deterministic_across_batch_sizes` stays green; cargo
  fmt/clippy/test green.

### potential-based-reward-shaping — Add F = γΦ(s′) − Φ(s) to raw_gradient

`raw_gradient` (`brain_passes.wgsl:170`) is purely interoceptive. Inject the
potential-based approach term at that one site so it propagates to the reward, the
homeostatic EMAs, and the memory valence. Keep the actor at the existing
`TD_VECTOR_SCALE` in this task (the next task raises it) so the short-horizon
alignment probe stays at chance and green.

**Steps:**

1. In `crates/xagent-brain/src/shaders/kernel/common.wgsl`, add the gain:

   ```wgsl
   /// Approach-shaping gain: Φ(s) = −APPROACH_SHAPING_GAIN · d_norm. Sized so the
   /// per-brain-tick F dominates the ~2e-4 metabolic drain but stays below the
   /// ~0.12 contact-eat spike. Initial value; promoted to a gene in WS 0004.
   const APPROACH_SHAPING_GAIN: f32 = 0.05;
   ```

2. In `coop_habituate_homeo`, thread-0 block (`brain_passes.wgsl:160-195`),
   immediately before the `raw_gradient` assignment at `:170`, form `Φ`/`F` and
   fold `F` into `raw_gradient`:

   ```wgsl
   let d_norm = clamp(physics_state[phys_base_homeo + P_NEAREST_FOOD_DIST]
                      / SHAPING_RADIUS, 0.0, 1.0);
   let potential = -APPROACH_SHAPING_GAIN * d_norm;
   let prev_potential = physics_state[phys_base_homeo + P_PREV_POTENTIAL];
   let shaping = TD_DISCOUNT * potential - prev_potential;
   physics_state[phys_base_homeo + P_PREV_POTENTIAL] = potential;
   let raw_gradient = energy_delta * ENERGY_WEIGHT
                    + integrity_delta * INTEGRITY_WEIGHT
                    + shaping;
   ```

   `phys_base_homeo` is already bound at `:161`. Do not touch `:171-195`.
3. Add a GPU integration test to `integration.rs` — `shaped_reward_rewards_approach`
   — that runs an agent one brain tick at distance `d1` from in-range food, then a
   second where `P_NEAREST_FOOD_DIST` has decreased to `d2 < d1`, and asserts the
   `shaping` contribution is positive (observe via `P_GRADIENT_OUT` deltas or a
   dedicated readback): the gradient with approach exceeds the gradient with the
   same energy/integrity deltas but a *receding* food. Embed the self-skip guard
   verbatim. This test is red before step 2 and green after.
4. Confirm `learning_probe_baseline_turn_alignment_is_chance` and
   `learning_probe_free_run_foraging_baseline` still pass (the 60/3000-tick
   horizons at the unchanged `1/128` actor scale do not flip the band).

- **Depends on:** `visible-food-potential-input`
- **Done when:** `shaped_reward_rewards_approach` is red before the edit and green
  after; the two baseline probes and `deterministic_across_batch_sizes` stay
  green; cargo fmt/clippy/test green.

### actor-vector-scale — Give the actor its own (larger) weight-step scale

The actor and critic both scale weight steps by `TD_VECTOR_SCALE = 1/128`
(`common.wgsl:306`; `brain_passes.wgsl:450-455`), throttling the actor step to
`0.10/128 ≈ 8e-4`. Split a separate, larger scale for the actor so it can latch
onto the now-sign-correct `δ` from the shaping term.

**Steps:**

1. In `common.wgsl`, beside `TD_VECTOR_SCALE`, add:

   ```wgsl
   /// Actor (forward/turn) weight-step scale, separate from the critic's
   /// TD_VECTOR_SCALE stability scale. Bounded by the MAX_WEIGHT_NORM L2 ball.
   const ACTOR_VECTOR_SCALE: f32 = 1.0 / 16.0;
   ```

2. In `coop_credit_assignment` (`brain_passes.wgsl:444-459`), change the
   `O_ACTION_FORWARD_WEIGHTS` and `O_ACTION_TURN_WEIGHTS` updates (`:452-455`) to
   multiply by `ACTOR_VECTOR_SCALE` instead of `TD_VECTOR_SCALE`; leave the
   `O_VALUE_WEIGHTS` critic update (`:450-451`) at `TD_VECTOR_SCALE`.
3. Add a GPU integration test `actor_step_scales_with_actor_vector_scale` to
   `integration.rs` that drives a known `δ` and trace and asserts the forward/turn
   weight delta equals `ACTION_WEIGHT_LEARNING_RATE · ACTOR_VECTOR_SCALE · δ ·
   trace` (within tolerance), proving the actor step uses the new scale while the
   critic step is unchanged. Embed the self-skip guard verbatim.
4. Confirm the two baseline probes still pass (the modest `1/16` scale over the
   probes' short horizons does not flip the chance band; the decisive movement
   requires the long warm-up added by the remeasure).

- **Depends on:** `potential-based-reward-shaping`
- **Done when:** `actor_step_scales_with_actor_vector_scale` passes, the critic
  update still uses `TD_VECTOR_SCALE`, and the two baseline probes stay green;
  cargo fmt/clippy/test green.

### approach-shaping-remeasure — Prove or kill the unlock

Run the diagnosis's falsification test: with shaping + actor scale, alignment
should move decisively above the `0.62` chance-band edge and free-run foraging
should rise. This is the gate that opens (or closes) `0002`/`0004`/`0005`.

**Steps:**

1. Add `learning_probe_shaped_turn_alignment_beats_chance` to `integration.rs`,
   modeled on `learning_probe_baseline_turn_alignment_is_chance` (`:1992`) but with
   a warm-up training phase in the same fixed-bearing arena before scoring:

   ```rust
   #[test]
   fn learning_probe_shaped_turn_alignment_beats_chance() {
       if !xagent_brain::GpuKernel::is_available() {
           eprintln!("Skipping: no GPU/fallback adapter available");
           return;
       }
       // Warm-up: train the policy on the shaped reward in the fixed-bearing
       // arena, then score turn/bearing alignment over >= MIN_SCORED_SAMPLES.
       const WARMUP_TICKS: usize = 3000;
       const PROBE_TICKS: usize = 60;
       const MIN_SCORED_SAMPLES: usize = 200;
       let brain = probe_brain_config();
       let mut arena = build_probe_arena(&brain, 11);
       for _ in 0..WARMUP_TICKS { arena.kernel.dispatch_batch(0, 1); }
       let (correct, scored) = score_turn_alignment(&mut arena, 0, PROBE_TICKS);
       let rate = correct as f64 / scored.max(1) as f64;
       eprintln!("shaped turn/bearing alignment {correct}/{scored} = {rate:.3}");
       assert!(scored >= MIN_SCORED_SAMPLES, "too few scored samples: {scored}");
       assert!(rate > 0.62, "alignment {rate:.3} did not beat chance band edge 0.62");
   }
   ```

   Locate `probe_brain_config`, `build_probe_arena`, `score_turn_alignment` by
   symbol (helpers near the other `learning_probe_*` tests). Tune `WARMUP_TICKS`
   upward if needed for the rate to stabilize; do not lower the `0.62` threshold.
2. Re-run `learning_probe_free_run_foraging_baseline` and record the new
   `food/agent/1k-ticks` against the `approach-shaping-baseline` number.
3. Record before/after alignment and foraging in the PR body and the baseline-spec
   subsection, with fixed seeds and the HEAD sha.
4. State the verdict: if alignment `> 0.62` **and** foraging rose, the unlock
   landed → `0002`/`0004`/`0005` open. If alignment stayed inside `[0.38, 0.62]`
   after a generous warm-up, record the negative, name the encoder/representation
   as the next suspect (per the review), and leave `0002`/`0004`/`0005` unopened.

- **Depends on:** `actor-vector-scale`, `approach-shaping-baseline`
- **Done when:** `learning_probe_shaped_turn_alignment_beats_chance` passes (rate
  `> 0.62`) and the before/after foraging delta is recorded — **or** the negative
  result is recorded with the encoder named as next suspect; cargo fmt/clippy/test
  green. (Prove-or-kill gate; its outcome opens or closes the gated workstreams.)

---

## 0002 — Reactive & valence layers on the external gradient

### klinotaxis-external-gradient — Drive klinotaxis from the food-distance gradient (GATED)

**Gate:** start only after `approach-shaping-remeasure` shows alignment moved above
`0.62`. If the remeasure recorded the negative, do not start; the reactive layer
cannot help a non-existent steering signal.

Klinotaxis reads `gradient_deviation = s_homeo[3u] − s_homeo[4u]`
(`brain_passes.wgsl:653`) — `gradient_fast − gradient_medium`, two EMAs of the
interoceptive `raw_gradient`. Drive it instead from the change in the *external*
approach potential `Φ` (`d_norm`), the across-time concentration comparison real
chemotaxis uses.

**Steps:**

1. In `coop_habituate_homeo` (`brain_passes.wgsl:160-195`), maintain a fast and a
   slow EMA of `d_norm` (reuse two free `brain_state` slots near `O_HOMEO`, or two
   spare `s_homeo` slots), mirroring the existing fast/medium gradient EMA pattern
   (`:171-176`).
2. In the motor block (`brain_passes.wgsl:648-655`), set `gradient_deviation` from
   `d_norm_fast − d_norm_slow` (an approaching agent's fast EMA leads the slow one),
   leaving `KLINOTAXIS_SENSITIVITY` and the `clamp(…, 0.3, 3.0)` envelope (`:654`)
   unchanged. Choose the exact EMA rates against the remeasure data.
3. Re-run `learning_probe_shaped_turn_alignment_beats_chance` and the foraging
   probe; record whether klinotaxis improves, holds, or regresses the rate.

- **Depends on:** `approach-shaping-remeasure`
- **Done when:** klinotaxis reads the external `d_norm` gradient and the shaped
  alignment/foraging probes are at least non-regressed (numbers recorded) — or the
  change is reverted wholesale with the negative recorded; cargo fmt/clippy/test
  green.

### memory-valence-food-in-view — Confirm/clean memory valence under shaping (GATED)

**Gate:** start only after `approach-shaping-remeasure` shows alignment moved above
`0.62`.

After `0001`, pattern valence is written from the shaped `raw_gradient`
(`brain_passes.wgsl:787,:813`), so a food-in-view approach state should now accrue
*positive* valence and recall (`sim·valence`, `:536-539`) should blend *toward*
food, fixing the "negate approach → escape" pathology (`:527`). Verify this; only
change code if recall still escapes food.

**Steps:**

1. Add a GPU integration test that drives an agent through an approach-then-eat
   episode and asserts the stored valence for the food-in-view pattern is positive
   (read the pattern valence slot `O_PAT_MOTOR + idx·3 + 2`), and that recall in a
   similar state biases motor *toward* food. Embed the self-skip guard verbatim.
2. If valence is already positive and recall approaches food, this is a
   verification-only landing (record the result; no shader change). If recall still
   escapes, separate the approach component into the stored valence so the
   food-in-view blend is net-positive, leaving the danger-avoidance behavior intact.
3. Update the stale comment at `brain_passes.wgsl:527` to match the post-shaping
   reality.

- **Depends on:** `approach-shaping-remeasure`
- **Done when:** the valence-sign test passes (food-in-view recall blends toward
  food); the comment matches behavior; cargo fmt/clippy/test green.

---

## 0003 — Selection signal restoration

### foraging-primary-fitness — Make food_per_1k_alive_ticks the primary objective

`composite_fitness` (`governor.rs:54-68`) multiplies foraging+exploration by the
survival gate `1/(1+death_count·0.5)` (`:62`), collapsing the composite into the
~0.0014 noise-floor tail at high death counts. Make `food_per_1k_alive_ticks`
primary with survival as a bounded multiplier, keeping the anti-kamikaze property
Plan 0001 added but restoring dynamic range where the agents live.

**Steps:**

1. In `composite_fitness` (`governor.rs:54-68`), compute
   `food_per_1k = food_consumed / (ticks_alive / 1000)` as the primary term
   (thread `ticks_alive` through from the caller `Governor::evaluate`,
   `governor.rs:482-498`, which already has `P_TICKS_ALIVE` access), and apply
   survival as a *bounded* multiplier (e.g. `survival = 1/(1 + death_count·k_s)`
   with the foraging term dominant), so a careful forager outscores a kamikaze one
   without the composite collapsing into the tail. Pin the exact form and the
   `k_s` value in the implementation with a doc-comment rationale; keep
   exploration as a smaller additive bonus.
2. Update the `composite_fitness` unit tests (search `composite_fitness` in
   `governor.rs` tests) so they pin the new form: assert (a) more food at equal
   deaths scores higher, (b) more deaths at equal food scores lower, (c) the live
   range for realistic (food, deaths, ticks_alive) tuples is well above the
   0.00x noise floor.
3. Note in the PR body that this supersedes Plan 0001's Variant B gate, citing the
   superseded behavior.

- **Depends on:** —
- **Done when:** `composite_fitness` makes `food_per_1k_alive_ticks` primary with
  survival bounded, the updated unit tests pass (red→green on the new assertions),
  and the live-range assertion holds; cargo fmt/clippy/test green.

### decoupled-experiment-resolution — Independent unique-config and repeat counts

Unique configs are coupled to repeats: `unique_count =
(population_size/eval_repeats).max(1)` (`governor.rs:944`), each repeated to fill
the population (`:997-998`), giving only ~2–5 unique configs/gen; `num_islands`
defaults to 3 (`config.rs:250`). Decouple the two dimensions and confirm per-repeat
seeds vary.

**Steps:**

1. In `crates/xagent-shared/src/config.rs`, add an `eval_unique_configs: usize`
   field to `GovernorConfig` with `#[serde(default = "default_eval_unique_configs")]`
   (default `8`), and change `default_num_islands` (`:250`) to `1`. Keep
   `population_size`/`eval_repeats` for backward-compatible configs.
2. In `governor.rs:941-1001`, drive the breeder off `eval_unique_configs` (clamped
   to a sane range) instead of `population/eval_repeats`; size the evaluated
   population as `eval_unique_configs · eval_repeats`.
3. Ensure each repeat gets an *independent* world seed: where `Governor::evaluate`
   groups by `agent_index/eval_repeats` (`governor.rs:559-567`), derive the world
   seed as `base_seed + repeat_index` (or a hash) so the `eval_repeats` reduce
   eval noise across *different* worlds, not identical ones. Add/extend a unit test
   asserting two repeats of the same config see different seeds.
4. Update any test that assumed `unique_count = population/eval_repeats`.

- **Depends on:** —
- **Done when:** unique-config count is set by `eval_unique_configs` independent of
  `eval_repeats`, `num_islands` defaults to `1`, per-repeat seeds provably vary
  (unit test), and existing governor tests pass; cargo fmt/clippy/test green.

### selection-significance-guard — Accept only above-noise improvements

The accept rule is a bare `if gen_avg >= parent_fitness` (`governor.rs:627`), so
the spawn bar ratchets on noise (the review measured a 0.76 % success rate and
47 % winner-disagreement between repeats). Require the improvement to clear a
multiple of the pooled standard error.

**Steps:**

1. In `Governor::evaluate`/the repeat-grouping path (`governor.rs:559-567`),
   surface the per-config across-repeat spread (variance/stderr) alongside the
   mean, so `advance` can read a `pooled_stderr` for the generation.
2. In `Governor::advance` (`governor.rs:609-655`), replace `if gen_avg >=
   parent_fitness` (`:627`) with `if gen_avg - parent_fitness > K_SIGNIF *
   pooled_stderr`, with `const K_SIGNIF: f32 = 1.0;` (one stderr; doc-comment the
   choice). Keep the backtracking/patience logic (`:642+`) otherwise unchanged.
3. Add a unit test: a child whose mean exceeds the parent by less than
   `K_SIGNIF·pooled_stderr` is rejected; one exceeding it by more is accepted.

- **Depends on:** `decoupled-experiment-resolution`
- **Done when:** the accept rule requires `gen_avg − parent > K_SIGNIF·pooled_stderr`,
  the new unit test passes, and backtracking still works (existing governor tests
  green); cargo fmt/clippy/test green.

### governor-within-life-metric — Persist q1→q4 within-life food-rate on the production path

The q1→q4 within-life food-rate exists only in `run_headless`
(`headless.rs:173-265,:311-325`); `Governor::evaluate`/`log_generation`
(`governor.rs:482,:267`) persist only end-of-life aggregates, so a production run
cannot detect within-life learning. Lift the metric onto the governor path.

**Steps:**

1. Extract the quarter-rate computation (`quarter_rates`, `headless.rs:311-325`)
   into a shared helper callable from `Governor`, or compute the per-generation
   first/last-quarter food-rate inside `Governor::evaluate` from the same
   cumulative `(food, alive_ticks)` quarter samples.
2. Persist a per-generation `q1_food_rate`/`q4_food_rate` (and their delta) via an
   idempotent migration in the generation-logging path
   (`let _ = db.execute_batch("ALTER TABLE … ADD COLUMN …");`) and write it in
   `log_generation` (`governor.rs:267`).
3. Add a unit test that a generation whose food-rate rises q1→q4 records a positive
   delta and one that is flat records ≈0.

- **Depends on:** —
- **Done when:** every governor run (not just `run_headless`) persists the
  per-generation q1→q4 within-life food-rate, the migration is idempotent, and the
  new unit test passes; cargo fmt/clippy/test green.

---

## 0004 — Heritable learning dynamics

### policy-constants-to-genes — Promote the policy/critic learning constants into BrainConfig (GATED)

**Gate:** start only after `approach-shaping-remeasure` shows alignment moved above
`0.62`. Tuning steering dynamics before steering exists is pointless and risks
destabilizing the just-validated default.

`ACTION_WEIGHT_LEARNING_RATE` (`common.wgsl:277`), `KLINOTAXIS_SENSITIVITY`
(`:281`), `TD_DISCOUNT` (`:294`), `TD_LAMBDA` (`:298`), `CRITIC_LEARNING_RATE`
(`:301`), and the new `ACTOR_VECTOR_SCALE` are `const`, so the evolvable genes
cannot tune the policy's learning dynamics. Promote them to `BrainConfig` genes.

**Steps:**

1. In `crates/xagent-shared/src/config.rs`, add six `BrainConfig` fields, each
   `#[serde(default = …)]` to the current `const` value, documented heritable with
   clamp ranges: `action_weight_learning_rate ∈ [0.01, 0.5]`,
   `critic_learning_rate ∈ [0.001, 0.1]`, `klinotaxis_sensitivity ∈ [50, 1000]`,
   `td_discount ∈ [0.9, 0.995]`, `td_lambda ∈ [0.5, 0.99]`,
   `actor_vector_scale ∈ [1.0/128.0, 1.0/4.0]`.
2. In `crates/xagent-brain/src/buffers.rs`, add `CFG_*` indices (after
   `CFG_INTEGRITY_SCALE`, mirroring `:292-294`), bump `CONFIG_SIZE`, pack them in
   `build_config_for` (`buffers.rs:545-564`), and extend the `CFG_*` mirror tests
   (`:710-712,:858`) and a round-trip test (`:754-764`).
3. In `crates/xagent-brain/src/shaders/kernel/{common,brain_passes}.wgsl`, read the
   six values via `bc_f32(CFG_…)` (`common.wgsl:355-356`) at their use sites
   instead of the `const`s, keeping the `const`s only as `default_*` echoes in Rust.
4. In `crates/xagent-sandbox/src/agent.rs` (`mutate_config_with_strength`) and the
   `mutation_outcomes` logging (`governor.rs:1514-1541`), mutate (clamped) and log
   the six new genes.

- **Depends on:** `approach-shaping-remeasure`
- **Done when:** the six learning constants are heritable `BrainConfig` genes
  (mutated, clamped, packed, read in-shader, round-tripped), the mirror/round-trip
  tests pass, and existing configs still load (serde defaults); cargo
  fmt/clippy/test green.

---

## 0005 — Sensory lag & vision geometry

### heritable-stride-revisit — Re-open Plan 0001's lag verdict post-shaping (GATED)

**Gate:** start only after `approach-shaping-remeasure` shows alignment moved above
`0.62`. Plan 0001 fixed `lag100` because higher strides cost more tps than the
learning gain — measured under the spatially-blind reward, where there was no
steering to gain from fresher state. That verdict is valid only pre-shaping.

**Steps:**

1. Make `vision_stride` heritable in `crates/xagent-shared/src/config.rs` and
   `crates/xagent-sandbox/src/agent.rs` (mutated within `[1, MAX_VISION_STRIDE]`),
   mirroring the existing `movement_speed` heritability (`config.rs:105-108`).
2. Re-run Plan 0001's three-arm stride/lag sweep *post-shaping* (control vs two
   reduced-lag arms) on a fixed seed at evolution scale, recording ticks/sec and
   foraging per arm in the baseline spec subsection.
3. Adopt a smaller default lag only if the re-measured foraging gain beats the tps
   cost under Plan 0001's budget rule (≈≤30 % tps cost AND beats control on the
   fixed seed); recalibrate `TD_DISCOUNT` in the same change to preserve the
   real-time horizon. Otherwise keep `lag100` and record the negative.

- **Depends on:** `approach-shaping-remeasure`
- **Done when:** `vision_stride` is heritable and the post-shaping sweep is recorded
  with a verdict — either a smaller lag adopted with `TD_DISCOUNT` recalibrated, or
  `lag100` retained with the negative recorded; cargo fmt/clippy/test green.

### vision-row-geometry — Make navigational-range ground food visible (GATED)

**Gate:** start only after `approach-shaping-remeasure` shows alignment moved above
`0.62`. Plan 0001 deferred the odd-grid vision change "until the learner can act on
directional vision" (`config.rs:73-83`); the remeasure passing satisfies that.

The 8×6 ray grid over a 90° FOV (`VISION_W=8`/`VISION_H=6`, `VISION_FOV_HALF=PI/4`,
`common.wgsl:13-14,:203`; ray pitch at `phase_vision.wgsl:31-37`) puts half the
rows above the horizon, so ground food beyond ~5 units falls between ray rows.

**Steps:**

1. Give the vision grid an odd height (default `vision_height` to an odd value,
   e.g. 7, or 13 per the documented 17×13 rationale at `config.rs:73-83`) so a ray
   row grazes the horizon; the vision-acuity probe (`integration.rs:2083+`) already
   documents the odd-grid rationale.
2. Verify navigational-range ground food becomes visible by extending
   `learning_probe_food_is_visible` (`integration.rs:1960`) or adding a probe at a
   distance beyond ~5 units, asserting ≥1 food pixel.
3. Re-run `learning_probe_shaped_turn_alignment_beats_chance` and the foraging
   probe; record the effect of the wider visibility.

- **Depends on:** `approach-shaping-remeasure`
- **Done when:** distal ground food is visible at navigational range (probe asserts
  food pixels beyond ~5 units), and the shaped alignment/foraging probes are
  recorded non-regressed — or the change is reverted with the negative recorded;
  cargo fmt/clippy/test green.

---

**End of plan 0004 TASKS.** When every "Done when" bullet is green — `0001` fully
landed and `0003` landed, with `0002`/`0004`/`0005` each either landed behind a
passed `0001` remeasure gate or recorded as a negative — the plan's end state is
reached.
