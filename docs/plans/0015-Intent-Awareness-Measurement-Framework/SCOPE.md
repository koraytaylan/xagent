# Scope — Plan 0015

> Build a principled system for measuring agent awareness and intent around
> food-seeking and danger-avoidance behaviors within the homeostasis-only
> learning constraint, formalizing measurement of whether actions are deliberate
> goal-directed choices versus incidental motion.

## Why this plan

The project README (§1, §3, §10) and Plan 0012 commit to homeostasis-only
evaluation as the sole learning signal, with no hand-engineered reward shaping.
Plan 0009 already ships a working **avoidance-intent** measurement: every tick
danger is in sense range, `agent_avoidance_accumulate`
(`kernel_tick.wgsl:257-274`) increments `P_AVOIDANCE_SENSE_RANGE_TICKS` (offset
42), and every tick the motor turn rotates away from the danger bearing it
increments `P_AVOIDANCE_TURNS_OPPOSING` (offset 43). Those counters flow from the
GPU state buffer onto the `Agent` cache (`gpu_orchestration.rs:256-257`,
`headless.rs:235-236`/`823-824`), into `AgentFitness` in `evaluate()`
(`governor.rs:676-677`), and into the population `compute_avoidance_intent_fraction`
(`governor.rs:203-207`), persisted as `avoidance_intent_fraction` in the
`behavior_metric` table — with a passing discrimination test
(`avoidance_intent_fraction_discriminates_turn_away`, `governor.rs:4763`). (The
zero assignments at `governor.rs:2041/2083/3555` are *test fixtures*, not the
production path.)

What is **missing** is the symmetric **approach-intent** measurement for food, and
a baseline that tells us whether either signal rises above chance under pure
homeostatic learning. The user's intent is to detect whether deliberate behaviors —
approaching food, avoiding danger — emerge from learning (agents correlating
sensed state with motor action) or are merely random-walk statistics. This
requires:

1. **An approach-intent counter symmetric to avoidance.** Food bearing/distance
   are already computed each cycle (`P_NEAREST_FOOD_DISTANCE` 32,
   `P_NEAREST_FOOD_BEARING` 34), and the motor turn is already available in
   `decision_buffer` where `agent_avoidance_accumulate` reads it. The only gap is
   a food-side counter pair and accumulation function mirroring avoidance.
2. **CPU-readable intent telemetry for measurement.** The baseline probe reads
   per-agent telemetry, so the four intent counters (approach + avoidance) must be
   exposed on `AgentTelemetry`.
3. **A population approach-intent fraction.** Mirror
   `compute_avoidance_intent_fraction` so the approach signal aggregates into the
   `behavior_metric` table the same way.
4. **A baseline distribution under homeostasis-only learning.** Capture the
   across-agent distribution of both intent fractions (incl. percentiles) at
   default config, so a follow-up plan can set deliberate-vs-incidental thresholds
   against real numbers — and so we learn whether the signal exists at all before
   building a classifier on it.

## In scope

- **0001 — Approach-Intent Telemetry Pipeline.** Add the food-side counters
  `P_APPROACH_SENSE_RANGE_TICKS` / `P_APPROACH_TURNS_TOWARD` (mirroring the
  avoidance pair), an `agent_approach_accumulate` function called from the existing
  thread-0 avoidance block, generation-cumulative save/restore in **both** death
  paths, and exposure of all four intent counters on `AgentTelemetry`. See
  [TASKS.md](TASKS.md).
- **0002 — Homeostasis-Only Intent Metrics.** Route the new approach counters from
  the GPU state buffer onto the `Agent` cache and `AgentFitness` (mirroring the
  avoidance copy sites), add `compute_approach_intent_fraction`, and persist
  `approach_intent_fraction` to `behavior_metric` with an idempotent migration. The
  avoidance plumbing is reused untouched. See [TASKS.md](TASKS.md).
- **0003 — Baseline Measurement & Documentation.** Capture the across-agent
  distribution of both intent fractions at default config (homeostasis-only,
  post-Plan-0012), recording mean/std/min/max **and p25/p50/p75** as the reference
  the deferred validation harness will consume; document the shipped telemetry in
  the brain and main READMEs. See [TASKS.md](TASKS.md).

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| Approach-intent (food) counterpart to the existing avoidance-intent machinery is missing — no food-side counter, accumulation function, fitness field, or population fraction (1,2,3) | `0001`, `0002` |
| Intent-signal strength baseline unknown under pure homeostasis-only learning (post-Plan-0012) (4) | `0003` |
| Deliberate-vs-incidental threshold and A/B validation against observable behavior not established | deferred follow-up plan (authored against this plan's measured baseline) |

## Locked decisions

- **Intent measurement is count-based, mirroring the existing avoidance machinery.**
  Each tick the target is in sense range increments a `sense_range_ticks` counter;
  each tick the motor turn rotates the correct way increments a `turns_*` counter;
  the intent fraction is `turns / sense_range_ticks`. This plan does NOT introduce a
  cosine/angle metric or change the existing avoidance computation — it copies it for
  food. Turning *toward* food is the sign-mirror of turning *away* from danger:
  avoidance uses `(motor_turn * danger_bearing) > 0.0`, approach uses
  `(motor_turn * food_bearing) < 0.0`.
- **The counters are generation-cumulative, preserved across respawn.** Exactly like
  the existing avoidance counters, both death/respawn functions
  (`agent_death_respawn` in `kernel_tick.wgsl` for the fused path, and
  `phase_death.wgsl` for the split/remainder path) save the approach counters before
  the full-stride zero loop and restore them after. Both paths must be edited
  together — diverging them on the new slots violates a tested death-path-parity
  invariant.
- **Intent measurement is purely observational, never a learning signal.** The
  counters and the intent fractions they feed are computed post-hoc from telemetry
  with zero impact on the GPU kernel's learning or fitness computation. They exist
  solely to characterize whether emergent behavior has learned deliberate steering.
  This plan respects the homeostasis-only learning constraint (Plan 0012) absolutely.
- **Deliberate-vs-incidental classification and its A/B validation harness are
  deferred to a follow-up plan.** This plan ships the measurement infrastructure and
  the baseline distribution (incl. percentiles); it deliberately does NOT author the
  classifier or the seeded A/B harness, because defensible thresholds can only be set
  once the real baseline numbers exist — and the baseline may show intent at chance,
  in which case the next plan targets the credit path, not a classifier.
- **Baseline from pure homeostatic learning (post-Plan-0012) is the reference.** The
  baseline intent distribution (workstream 0003) is measured after Plan 0012 removed
  all PBRS shaping, so it represents intent under pure homeostatic learning — the
  BEFORE state. A future plan that changes learning compares new measurements against
  this fixed snapshot.

## Out of scope

- **The seeded A/B intent-validation harness and deliberate-vs-incidental
  classifier.** Deferred to a follow-up plan, authored against the *real* baseline
  distribution this plan measures (workstream 0003). Setting classification
  thresholds, building the null random-walk control, and validating intent against
  path-coherence/encounter statistics all depend on baseline numbers that do not
  exist until 0015 runs.
- **Changing the existing avoidance-intent computation.** The avoidance counters,
  accumulation, fraction, and test already work and are reused as-is; this plan only
  adds the approach mirror.
- **A cosine/angle-based intent metric.** The shipped machinery is count-based;
  introducing a finer-grained alignment metric would mean changing the working
  avoidance path and is out of scope.
- **Retraining or re-initializing weights based on intent signals.** Intent
  measurement is post-hoc telemetry only. Agents train as before under homeostatic
  learning. No retraining loop is opened.
- **Modifying fitness computation or evolution gates based on intent metrics.**
  Intent metrics are observational. Fitness remains effort-rebased (per Plan 0009). A
  future plan may gate evolution on intent thresholds, but this plan measures without
  gating.
- **Changing the danger-percept or food-detect scan mechanics.** The scans are
  unchanged. Plan 0015 only reads their output (`P_NEAREST_FOOD_*` /
  `P_NEAREST_DANGER_*`); the physics remains as Plan 0009 defined it.
- **Adding new sensory modalities (e.g., olfaction, hearing).** Intent measurement
  works with the existing food/danger percepts. New modalities are out of scope.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
