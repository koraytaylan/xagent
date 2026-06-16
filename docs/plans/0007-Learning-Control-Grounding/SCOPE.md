# Scope - Plan 0007

> Ground xagent learning in truthful runtime genetics, observable behavior
> metrics, slower controllable perception-action loops, and red-green probes for
> food seeking and danger avoidance before asking evolution to amplify behavior.

## Why this plan

This plan is based on a direct investigation of `xagent.db` on 2026-06-15 plus
the live GPU/runtime code that produced it. The database is a 1-run sample
(`run.id=1`, seed 42, `wall_time_secs=11412.146806292`) with 251 nodes, 2500
agent results, 1952 mutation rows, and 250 replay recordings. The load-bearing
observations are:

1. **Evolution did not improve in this run.** `node.status` is one active node,
   one exhausted root, and 249 failed descendants. `run.best_score` remains the
   generation-0 root score (`0.0042712856`, `spawn_parent_id=1`), while the last
   25 evaluated generations average `0.0014031432`. The root averaged 292.9 food
   and 305.2 deaths per agent; generations 225-249 averaged 262.096 food and
   324.42 deaths. This is a stalled outer loop, not a lineage acquiring
   deliberate behavior.
2. **The root score is mostly an exploration artifact.** Generation 0 averaged
   996.9 explored cells, while generations 225-249 averaged 174.592. Food did
   not improve, deaths worsened, and the late population never recovered the
   root's exploration coverage. The governor's current foraging-primary formula
   (`composite_fitness`, `governor.rs:74-92`) is better than the old survival
   gate, but this DB still shows selection pinned to a root whose advantage is
   not food seeking.
3. **The replay proves persistent turning.** Decoding every
   `generation_recording.data` blob with the documented 15-float stride
   (`governor.rs:1276-1281`, `governor.rs:1303-1318`) gives mean straightness
   `0.007105`, mean absolute turn `0.375534`, mean turn-bias ratio `0.851702`,
   and turn-sign persistence `0.940761` across 250 recorded nodes. Mean absolute
   turn correlates with deaths (`r=0.9107`) and against fitness (`r=-0.6596`).
   The visual "circles until death" observation is therefore database-measurable.
4. **The current DB cannot prove food chasing or danger avoidance.** The
   recording stores position, yaw, vitals, motor, prediction, exploration,
   gradient, urgency, fatigue, and staleness only (`governor.rs:1276-1281`).
   It does not store food positions, nearest-food bearing, danger-zone dwell
   time, or hazard exit latency. This `xagent.db` also lacks `q1_food_rate` and
   `q4_food_rate` columns even though the current migration adds them
   (`governor.rs:1664-1666`), so within-life learning cannot be recovered from
   the node table.
5. **Several genes in `xagent.db` are not faithfully applied in the live GPU
   runtime.** The live `PendingUpload` carries only terrain, food state, and
   `(position, max_energy, max_integrity, memory_capacity, processing_slots)`
   (`app.rs:37-43`, `gpu_orchestration.rs:31-56`). `Worker::new` uploads those
   values but never patches per-agent heritable brain-state slots
   (`sim_runtime.rs:317-331`). `Worker::reset_population` writes inherited
   champion brain states to every agent (`sim_runtime.rs:433-445`) but still
   does not call `GpuKernel::write_agent_heritable_config`
   (`gpu_kernel.rs:1912-1952`). The headless path does patch these values
   (`headless.rs:159-164`), so the interactive path is the inconsistent one.
6. **Mutation provenance is incomplete.** Agent configs in `xagent.db` show
   `movement_speed` ranging from 20.0 to 29.16912, but `mutation` has no
   `movement_speed` rows because `record_mutations` checks only nine fields and
   stops at `fatigue_floor` (`governor.rs:1689-1735`). Speed correlations from
   this DB are therefore ambiguous: the configs vary, the live runtime may not
   apply them, and the mutation table cannot explain their direction.
7. **The perception-action time scale supports the user's speed concern.**
   The run's agent configs all have `brain_tick_stride=10` and
   `vision_stride=10`, so sensory lag is 100 physics ticks
   (`config.rs:84-96`, `config.rs:381-382`). With default speed 20 units/s and
   tick rate 30 Hz (`config.rs:179-180`, `config.rs:432-443`), a full-forward
   agent travels about 66.7 world units during one stale visual batch, more than
   twice `VISION_MAX_DIST=30` (`common.wgsl:203-214`). Even at partial throttle,
   the brain is often learning from observations that no longer describe its
   current local world.
8. **Current klinotaxis cannot break a bad turn sign.** The shader computes
   `klinotaxis_factor = clamp(1.0 - gradient_deviation *
   KLINOTAXIS_SENSITIVITY, 0.3, 3.0)` and then does `turn *=
   klinotaxis_factor` (`brain_passes.wgsl:665-672`). The factor is always
   positive. It can scale an existing turn, but it cannot reverse a persistent
   left/right bias or inject a sign-correct escape maneuver.
9. **Plan 0004 already falsified scalar approach shaping as a complete unlock.**
   Its status records a chance-level mirrored-steering result after potential
   based reward shaping and actor-scale changes
   (`0004-Approach-Reward-Shaping/STATUS.md`). This plan therefore does not
   repeat "add scalar distance reward" as the answer. It treats food/danger
   behavior as a probe-gated control and observability problem.

Together these findings explain why the current system can look alive but not
deliberate: the DB shows a persistent-turn attractor, the runtime may not apply
the genes the DB says it evaluated, visual state is stale at navigational scale,
and the stored data is insufficient to prove or falsify food/danger intent.

## In scope

- **0001 - Runtime genome authority.** Make the live worker apply the same
  per-agent heritable config values the headless path applies, and make mutation
  provenance include every effective per-agent gene.
- **0002 - Behavioral evidence telemetry.** Version the recording format and add
  authoritative per-agent metrics for turn persistence, straightness, nearest
  food distance/bearing, danger dwell, and within-life food rates.
- **0003 - Control-rate curriculum.** Add a learning preset with slower
  movement and lower sensory lag, then gate any default change on measured
  behavior and throughput.
- **0004 - Turn-attractor and klinotaxis repair.** Replace multiplier-only
  klinotaxis with a sign-breaking/stabilizing control path that can reduce or
  reverse persistent turning when outcomes worsen.
- **0005 - Food/danger emergence gates.** Add controlled GPU probes that must
  show food-distance closure, bearing-aligned turns, and hazard exit behavior
  before a long evolutionary run is considered meaningful.

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| Evolution did not improve (1) | `0005` |
| Root exploration artifact (2) | `0002`, `0005` |
| Persistent turning in replay (3) | `0002`, `0004` |
| DB cannot prove chasing/avoidance (4) | `0002`, `0005` |
| Live runtime does not apply per-agent genes (5) | `0001` |
| Mutation provenance incomplete (6) | `0001`, `0002` |
| Speed/sensory lag mismatch (7) | `0003`, `0005` |
| Klinotaxis cannot reverse sign (8) | `0004` |
| Scalar shaping already falsified as complete unlock (9) | `0004`, `0005` |

## Locked decisions

- **Runtime truth before tuning.** No movement-speed, curiosity, fatigue, or
  habituation conclusion is accepted until the live worker applies those
  per-agent values after inheritance and the DB records their mutation
  provenance.
- **Do not claim emergence from aggregate fitness alone.** Food seeking requires
  food-distance and bearing metrics; danger avoidance requires danger dwell and
  exit-latency metrics. Food count and death count remain summary outcomes, not
  proof of intent.
- **Slower control is a curriculum, not a blind permanent default.** The initial
  learning preset targets `movement_speed=8.0`, `brain_tick_stride=2`, and
  `vision_stride=5` so full-forward sensory-lag travel is about 2.7 world units
  instead of 66.7. It becomes the evolution default only if the probes improve
  without unacceptable throughput loss.
- **Klinotaxis must be able to change direction.** A positive turn multiplier is
  not an anti-circle mechanism. The replacement must either damp persistent turn
  toward straight motion or inject a sign-changing reorientation when the
  measured gradient worsens.
- **Certainty comes from red-green probes.** The plan's claim is not that any
  single edit guarantees intelligence. The claim is that each prerequisite for
  learning is made observable and mechanically tested: genes affect the GPU
  state, sensory/action lag is bounded, bad turns can be broken, food/danger
  signals are measured, and only then does evolution run.

## Out of scope

- **Independent per-genome arenas.** A full arena-per-repeat architecture would
  solve shared-world competition and population-uniform config issues, but it is
  larger than this plan. This plan first makes the current shared-world runtime
  truthful and measurable.
- **A new brain architecture.** Encoder sizes, memory layout, and TD(lambda)
  structure stay intact unless the probes still fail after the runtime/control
  fixes.
- **Pure UI polish.** UI display changes are included only where needed to
  expose new evidence; visual redesign is not part of this plan.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
