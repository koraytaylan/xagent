# Scope — Plan 0010

> Make Plan 0009's flag-gated effort/danger machinery actually safe to graduate:
> re-derive the effort-fitness calibration from real cumulative telemetry so the
> foraging and exploration axes stop collapsing to ≈0 at production scale, harden
> the speed-decoupling gate so it cannot pass on a worsening metric, fix the
> avoidance-intent percept so it measures avoidance rather than its opposite, and
> repair the acceptance-named tests that never assert their titular property —
> all without flipping any default until the corrected gate says so.

## Why this plan

Plan 0009 shipped four flag-gated layers (super-linear drag, path-length hazard,
effort-rebased fitness, danger percept + avoidance shaping) with a verified
byte-identical no-op at defaults. Four independent 2026-06-18 reviews (Claude
Opus 4.8, Grok 4.3, GPT-5 Codex, Gemini 3.1 Pro High) agree the default build is
correct and well-tested (93/93 green), but converge on one conclusion: the work
is **not ready to flip the flags on**. The blockers below are every review claim
that survived independent re-verification against the current tree (see
provenance). Nothing here regresses the shipped default path.

1. **The effort-rebased foraging axis collapses to ≈0 at production scale.**
   `composite_fitness` foraging is `min((food_consumed / energy) / FORAGING_ENERGY_TARGET, 1.0)`
   with `FORAGING_ENERGY_TARGET = 0.5` (`governor.rs:133`, `governor.rs:52`) and
   weight `0.85` (`governor.rs:72,154`). `energy_spent` is a GPU accumulator summed
   every tick and **preserved across respawn** (`phase_death.wgsl:50-55,85-90`;
   `kernel_tick.wgsl:543-577`), never zeroed within a generation
   (`agent/mod.rs:318-353` zeroes only on new generation). Over a ~1M-tick
   generation, depletion drain alone is `0.015/tick × ~1e6 ≈ 15,000`, so real
   `food/energy ≈ few-hundred / ~15,000 ≈ 0.003–0.02`, pinning the weight-0.85
   axis near zero for the whole population. The calibration assumed `food/energy =
   180/180 = 1.0` (`0009-FITNESS-CALIBRATION.md:80`), a ratio production never
   produces.
2. **The exploration axis collapses the same way; the "non-binding" distance
   budget is the binding constraint.** `exploration = min(coverage, cells_per_dist)`,
   `cells_per_dist = min(cells_explored / (distance_traveled / 16.0), 1.0)`
   (`governor.rs:136-139`, `EXPLORATION_DISTANCE_BUDGET = 16.0` at `governor.rs:62`).
   `distance_traveled` is the same kind of respawn-preserved accumulator
   (O(hundreds of thousands) over a generation) while `cells_explored` is capped
   at the reachable grid, so `cells_per_dist ≈ 0.025` for every agent. The
   calibration doc calls the budget a "non-binding guard"
   (`0009-FITNESS-CALIBRATION.md:42-46`) — true only for the synthetic single-life
   distances, false in production.
3. **`P_ENERGY_SPENT` under-counts true energy and the under-count scales with
   brain size.** The accumulator records only depletion + movement drain
   (`kernel_tick.wgsl:186-189`, `phase_physics.wgsl:140-143`); the per-tick brain
   metabolic drain `(METABOLIC_BASE_COST + mem_cap·METABOLIC_MEMORY_COST +
   proc_slots·METABOLIC_PROCESSING_COST)·metabolic_rate` is subtracted from real
   energy (`kernel_tick.wgsl:228`, `phase_physics.wgsl:178`) but never added to the
   slot, so two agents with identical food/movement but different brains get
   different (mis-ranked) foraging scores — a confound that undercuts the
   "effort-rebased = brain-agnostic" intent.
4. **The calibration constants were picked from three synthetic profiles, not the
   spec-mandated recorded-generation replay.** `fitness_calibration_replay_profiles`
   (`governor.rs:2351-2384`) builds hand-coded competent/aimless/camper profiles;
   the `store_recording`/`load_recording`/`generation_recording` machinery the
   replay needs exists (`governor.rs:1375,1472,1801-1808`) but is unused. The
   skipped replay is exactly the check that would have surfaced findings 1–3.
5. **The calibration test asserts ordering only, not the documented magnitudes.**
   The test asserts `effort_competent > effort_aimless` and `effort_camper < 1.0`
   (`governor.rs:2503-2516`); the doc's headline numbers (competent 0.7050,
   fast-aimless 0.3656, delta −0.3506, gap 0.3394, camper 0.8650) appear only in
   `eprintln!` (`governor.rs:2415-2498`), so a math regression preserving ordering
   passes. The test also uses `grid = 1000.0` (`governor.rs:2352`) while production
   `evaluate` uses `total_grid_cells = HEATMAP_RES²/4 = 1024` (`governor.rs:656`).
6. **The speed-decoupling gate passes when the primary metric moves the wrong
   way.** The gate is `speed_decoupled = on_corr.abs() < 0.3` and `gate_passed =
   speed_decoupled && ticks_alive_ok && danger_retained` (`headless.rs:947,951`),
   with no requirement that the baseline be strongly positive or that ON improve.
   The committed decision doc prints "Speed-fitness correlation falls … : PASS"
   while its own Analysis says the correlation **rose** 0.2238 → 0.2569
   (`0009-SPEED-DECOUPLING.md:16,37,67`). The metric the layer exists to reduce
   increased, yet the gate passed.
7. **The A/B is uncontrolled: baseline and ON draw independent unseeded RNG in
   three places.** The two arms are separate `run_headless_with_flags` calls
   (`headless.rs:423,427`), each seeding genomes via `mutate_config_with_strength`
   (`agent/mod.rs:378`), brain mutations via `mutate_brain_state` (`agent/mod.rs:500`),
   and brain weights via `reset_agents` (`gpu_kernel.rs:523`) — all `rand::rng()`,
   none seeded per-arm. A deterministic `reset_agents_seeded` exists
   (`gpu_kernel.rs:516-520`) but the harness does not use it, so cross-arm deltas
   cannot be attributed to the flags.
8. **Two of the three gate conjuncts carry no signal.** `avoidance_retained =
   on_stats.mean_avoidance_intent_fraction >= 0.0` is tautologically true and is
   **excluded** from `gate_passed` (`headless.rs:950,951`); `ticks_alive_ok` is
   `on_mean > baseline_mean × 0.8` (`headless.rs:948`) but both arms saturate
   `tick_budget = 1_000_000` (`config.rs:429`) because `P_TICKS_ALIVE` is preserved
   across respawn, so the floor cannot bind. "GATE PASSED" rests on the correlation
   conjunct alone. The entire decision machinery (`compute_correlation`,
   `compute_regression`, `format_validation_markdown`, the gate) has **zero unit
   tests** (only reachable via `--validate-speed-decoupling`).
9. **The avoidance-intent metric measures the opposite of its name.** The
   "turn opposes danger bearing" test is `(motor_turn · danger_bearing) < 0.0`
   (`kernel_tick.wgsl:258-260`), but `danger_bearing = atan2(facing×to_danger,
   facing·to_danger)` is **negative** for right-side danger while positive
   `motor_turn` turns right (`common.wgsl:375` `TURN_SPEED=3.0`, yaw at
   `kernel_tick.wgsl:86`), so a genuine turn-away (`motor_turn<0` for right danger)
   gives a positive product and is **not** counted; turning *into* the danger is
   counted. The in-source comment "positive = danger to the right"
   (`kernel_tick.wgsl:256`) is backwards. The counter is observability-only —
   `composite_fitness` never receives it (`governor.rs:115-124`) — so selection is
   uncorrupted, but `avoidance_intent_fraction` is a named gate criterion and
   reports the inverse of the truth.
10. **The fused kernel feeds the avoidance counters previous-cycle danger
    telemetry.** In `kernel_tick.wgsl`, `agent_physics` increments the counters by
    reading `P_NEAREST_DANGER_*` (`:258-260`) before `agent_danger_detect`
    (`:803-805`) writes those slots that cycle; the split `phase_physics.wgsl`
    scans danger first (`:197-247`) then increments (`:261-263`). The counters read
    stale danger in the production fused path.
11. **The danger ring-scan runs unconditionally on the flag-off path.**
    `agent_danger_detect` does an O((DANGER_SENSE_RADIUS/cell)²) biome-grid scan +
    `atan2` every cycle (`kernel_tick.wgsl:269-334`, called at `:803-805` with no
    flag guard; mirrored `phase_physics.wgsl:197-247`); `danger_percept_enabled`
    gates only feature packing (`brain_passes.wgsl:239`), not the scan. The
    default build is byte-identical in OUTPUT but not COMPUTE. The `atan2(cross_y,
    dot_val)` (`kernel_tick.wgsl:319-321`) has no `dist > EPSILON` guard, so an
    agent on a danger-cell center hits WGSL-indeterminate `atan2(0,0)`.
12. **Three acceptance-named tests do not assert their titular property.**
    `nearest_danger_bearing_points_at_danger` only `eprintln!`s the near agent's
    bearing and asserts it solely for the far agent (`integration.rs:6470,6497`);
    `recorded_telemetry_persists_in_agent_fitness` copies GPU slots into agent
    fields and asserts the copies `> 0.0`, never touching `AgentFitness`/`evaluate`/DB
    (`integration.rs:6369,6378,6384`); `danger_percept_byte_identical_when_flag_off`
    compares two flag-off runs to each other (`integration.rs:4010,4014`), proving
    determinism not byte-identity to a pre-percept build.
13. **Layout/migration/coverage gaps.** `WC_DANGER_PERCEPT_ENABLED` has neither an
    in-bounds assert nor a Rust↔WGSL parity test, and `CFG_DANGER_PERCEPT_ENABLED`
    is absent from `shader_config_constants_match_rust` (`buffers.rs:1257-1270,1496-1520`);
    no test asserts `danger_percept_enabled`/`effort_rebased_fitness` deserialize to
    `false` from a legacy blob (`config.rs:712-757` covers only visual fields); the
    `behavior_metric` `ALTER` precedes its `CREATE` (`governor.rs:1827` before `:1830`);
    a stale comment says `PHYS_STRIDE=39` (`integration.rs:3732`, real value 44) and
    `fill_world_config`'s doc says "24 world-config slots" (`buffers.rs:638`,
    `WORLD_CONFIG_SIZE=28`); the raw avoidance counters are persisted to neither
    `agent_result` nor `behavior_metric` (only the derived ratio); and the WGSL
    literals `255`, `1.414`, `20.0` are un-named (`common.wgsl:640`,
    `kernel_tick.wgsl:182,208`, `phase_physics.wgsl:136,158`).

**Provenance.** Verified against branch `claude/funny-cray-84end5` @ `5ddc976`
(the tip carrying the 0009 implementation, the four reviews, and the
`contributing_guard.rs` ratchet). Every numbered finding above was independently
re-read in the current source — line numbers are the real current locations, not
the reviews' — and only claims that survived that re-read are listed. The claims
below were raised by a review and **rejected** during verification; they are
recorded so they are not re-litigated into tasks.

**Review claims rejected during verification:**

| Claim | Source | Why rejected |
|---|---|---|
| Avoidance-counter staleness is a fused-vs-split *divergence* that the parity test leaves unguarded | Gemini F2, GPT-5 P2 | `split_serial_matches_fused_serial` locks the full 44-slot vector and passes because production `SplitSerial` re-dispatches the **same** `kernel_tick.wgsl`, not `phase_physics.wgsl`; the one-cycle staleness is real but lives in `kernel_tick.wgsl` itself (finding 10), not in an unguarded path divergence |
| The calibration test "has no asserts / only prints" | Gemini F1, Grok #3 | A follow-up commit added `effort_competent > effort_aimless` and `effort_camper < 1.0` (`governor.rs:2503-2516`); the residual is unasserted *magnitudes* (finding 5), not zero asserts |
| `compute_correlation` low-variance branch auto-passes the gate | Claude L5 context | Correlation is computed over speeds flattened across all generations with `movement_speed` re-mutated each generation, so `sum_x_sq` stays ~13 orders above the `1e-10` guard; folded into a unit-test robustness assertion (finding 8), not a standalone defect |
| The harness cleanup leaks `-wal`/`-shm` sidecars as a correctness bug | GPT-5 Rec | Confirmed the sidecars are not removed (`headless.rs:796`), but it is a benign temp-file hygiene nit folded into the decision-doc task, not a blocker |

## In scope

- **0001 — Effort-fitness scale-invariance & recalibration.** Add the brain
  metabolic drain to `P_ENERGY_SPENT` in both physics paths; build the real
  recorded-generation replay calibration and re-derive (or make scale-invariant)
  `FORAGING_ENERGY_TARGET` and `EXPLORATION_DISTANCE_BUDGET` so the production
  population does not pin both effort axes to ≈0; pin the documented magnitudes and
  the production grid denominator in the calibration test. Findings 1–5.
- **0002 — Decision-machinery hardening.** Rewrite the gate to require a
  strongly-positive baseline and strict improvement, drop the tautological
  conjunct and replace the saturated viability conjunct with an uncapped metric,
  make the A/B a seeded paired comparison, add GPU-free unit tests for the
  correlation/regression/gate math, and emit run metadata into the decision doc.
  Findings 6–8.
- **0003 — Danger-percept correctness.** Fix the avoidance-intent sign convention
  and the fused stale-telemetry ordering so the counter measures genuine
  turn-aways, gate the danger ring-scan behind the percept flag so the default
  build is a true compute no-op, and guard the bearing `atan2` against the
  on-center degeneracy. Findings 9–11.
- **0004 — Test, layout & migration integrity.** Make the three acceptance-named
  tests assert their titular property, close the `WC_*`/`CFG_*` parity and
  serde-default coverage gaps, order the `behavior_metric` migration correctly,
  fix the stale layout comments/docs, decide and document the avoidance-counter
  persistence, and name the WGSL magic numbers. Findings 12–13.

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| Foraging axis collapses at production scale (1) | `0001` |
| Exploration axis collapses; budget is binding (2) | `0001` |
| `P_ENERGY_SPENT` under-counts brain drain (3) | `0001` |
| Constants from synthetic, not recorded replay (4) | `0001` |
| Calibration test asserts ordering only; grid 1000 vs 1024 (5) | `0001` |
| Gate passes on rising correlation (6) | `0002` |
| Uncontrolled unseeded A/B (7) | `0002` |
| Tautological + saturated conjuncts; zero gate unit tests (8) | `0002` |
| Avoidance-intent sign inversion (9) | `0003` |
| Fused stale-telemetry ordering (10) | `0003` |
| Unconditional danger scan; `atan2(0,0)` (11) | `0003` |
| Three acceptance-named tests do not assert (12) | `0004` |
| Layout/migration/coverage/magic-number gaps (13) | `0004` |

## Locked decisions

- **No default is flipped in this plan.** `0009`'s `default-flip-gate` stays
  GATED. This plan makes the gate trustworthy and the calibration real; flipping
  `speed_cost_exponent`, `effort_rebased_fitness`, or `danger_percept_enabled`
  remains a separate, measurement-gated decision after `0001` and `0002` land.
- **Scale-invariance over hand-tuned constants.** The foraging/exploration axes
  are re-derived from the spec-mandated recorded-generation replay against real
  cumulative telemetry, and the chosen form must satisfy a production-scale
  assertion: a competent forager on real recorded telemetry reaches foraging ≈ 1.0
  (not ≈ 0.02). The accepted construction and the re-derived constants are written
  up in `0010-FITNESS-RECALIBRATION-DECISION.md`. Until that doc lands, the
  constants are not changed.
- **Brain-drain accounting lands before recalibration.** `P_ENERGY_SPENT` must
  include the per-tick brain metabolic drain (byte-identical edit in both shaders)
  before the replay is run, so the calibration sees the true energy denominator.
- **The avoidance-intent fix is observability-only and changes no selection
  output.** `composite_fitness` does not consume the counters (`governor.rs:115-124`),
  so the sign/timing fix corrects a reported metric and a gate criterion, never the
  evolved population. The default-off build stays byte-identical in encoded
  state/behavior/fitness.
- **The danger scan is gated, not merely measured.** The entire
  `agent_danger_detect` scan is placed behind `WC_DANGER_PERCEPT_ENABLED` so the
  default build is byte-identical in COMPUTE as well as OUTPUT, in both shaders,
  with a parity test proving flag-off determinism is preserved.
- **Each task strips the planning references in the file regions it edits and
  ratchets that file's `contributing_guard.rs` baseline down**, per CONTRIBUTING's
  on-touch cleanup rule, and introduces **zero** new planning references in new
  code. The wholesale scrub of files this plan never touches is owned by Plan
  `0011` (disjoint file set — see that plan's SCOPE), so the two plans run in
  parallel without contending for the guard baseline.
- **Both physics paths and both respawn whitelists keep moving together.** Every
  shader edit in `0001`/`0003` lands in `kernel_tick.wgsl` AND `phase_physics.wgsl`
  (and `phase_death.wgsl` for respawn), locked by the existing parity/equivalence
  tests, exactly as `0009` required.

## Out of scope

- **Flipping any `0009` default.** Gated on the corrected gate's numbers, in a
  later change. This plan only makes that decision trustworthy.
- **Reworking the encoder, memory, or TD(λ) credit path.** Plan `0007` owns the
  learner; this plan changes calibration, the validation gate, the danger percept
  metric, and test/layout integrity — never the learner architecture.
- **Adding danger as a subtractive fitness penalty.** `0009`'s locked decision
  stands: danger stays graded and observability-only so avoidance-decision data is
  retained. The avoidance-intent fix corrects a *metric*, not a selection term.
- **The wholesale planning-reference scrub of untouched files and the magic-number
  naming outside this plan's touched shaders.** Owned by Plan `0011`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
