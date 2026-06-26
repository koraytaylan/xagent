# Scope — Plan 0021

> Run hardened harnesses on production seeds to record flip-or-retire decisions
> for effort-rebased fitness and danger-percept flags, and record terminal
> decision for innate-instincts.

## Why this plan

The `2026-06-25` review (F8) flagged that three levers (`effort_rebased_fitness`,
`danger_percept_enabled`, `innate_instincts_enabled`) have undergone significant
correctness work and have hardened decision harnesses, but remain in "safe but
unused" limbo without recorded graduation decisions or explicit unlock
conditions.

1. **Effort-rebased fitness has correct math but no graduation decision.** Plan
   0014 fixed the camper inversion (removed `ticks` factor, re-derived targets,
   added `competent > camper` guards) and the calibration replay now passes
   (`crates/xagent-sandbox/src/governor.rs:135-151`). The 0010 STATUS records
   "running 0009's default-flip-gate to a decision is the follow-on," but no plan
   has picked it up. The corrected lever has sat gated-off for ~one week with no
   recorded A/B on production seeds. The hardened speed-decoupling harness
   (`crates/xagent-sandbox/src/headless.rs:419-458` `validate_speed_decoupling`)
   computes a full-population A/B (baseline vs effort_ON), measuring
   `mean_ticks_alive` stability, fitness decorrelation from `movement_speed`, and
   `danger_dwell` retention. This plan runs it to a measured decision.

2. **Danger-percept-enabled has hardened tests but no graduation decision or
   unlock gate.** Plans 0009/0010 built `danger_percept_enabled` and added it to
   the hardened speed-decoupling validation
   (`crates/xagent-sandbox/src/headless.rs:428-443` `run_headless_with_flags`
   enables it as a flag), plus two corroborating tests:
   `crates/xagent-brain/tests/intent_baseline_measurement.rs` (avoidance distribution at default config) and
   `crates/xagent-brain/tests/danger_percept_ablation_ab.rs` (paired A/B with/without danger percept
   signal). The `2026-06-25` review F8 and 0010 STATUS explicitly ask for either a
   flip-or-retire decision or documented unlock conditions. The danger-percept A/B
   harness is ready; this plan runs it on production seeds and records the verdict.

3. **Innate-instincts-enabled already has a REJECT decision doc
   (`0013-INNATE-INSTINCT-DECISION.md`) but is not recorded as terminal in
   STATUS.** Plan 0013 ran `run_innate_instinct_ab`
   (`crates/xagent-sandbox/src/headless.rs:492-560`) and recorded three gate
   failures (survival +0.19% << +10%, alignment 0.000 with `danger_percept` OFF,
   food-per-death 0.37 << 2.0). The decision doc lists revisit conditions
   (`danger_percept` ON for a fair test, weaker seeds, longer runs). The STATUS row
   states "flag stays default-off" but does not explicitly record whether the
   decision is terminal or deferred. This plan marks it terminal (flag ships
   complete, seeding mechanism kept) or documents the gate preconditions if future
   credit-path work would unlock it.

All three are "accumulation" issues: correct code, good harnesses, no recorded
decision. This plan runs the production A/Bs and writes the decision docs.

**Provenance.** Every finding re-verified against `develop` source: the effort
foraging formula and `competent > camper` guard at `governor.rs:135-151`
(confirmed); the speed-decoupling harness with the `danger_percept_enabled` flag
toggle at `headless.rs:419-458` / `headless.rs:428-443` (confirmed); the two
hardened danger tests `crates/xagent-brain/tests/intent_baseline_measurement.rs` and
`crates/xagent-brain/tests/danger_percept_ablation_ab.rs` (confirmed); the recorded prove-or-kill gate
failures in `0013-INNATE-INSTINCT-DECISION.md` against `run_innate_instinct_ab` at
`headless.rs:492-560` (confirmed). No `2026-06-25` review claim was rejected
during verification — all three flagged levers are confirmed default-off and
decision-free.

## In scope

- **0001 — Effort-Fitness-Production-A-B.** Run the hardened speed-decoupling
  harness (`validate_speed_decoupling`,
  `crates/xagent-sandbox/src/headless.rs:419-458`) on production-scale seeded
  worlds (population 100, 50 generations minimum) with `effort_rebased_fitness`
  baseline vs ON, measuring `mean_ticks_alive` stability, fitness correlation with
  `movement_speed` (target: decorrelated), and `danger_dwell_fraction` retention.
  Record 95% CI bootstrap delta, decision doc (flip-or-retire), and unlock
  conditions if deferred. See [TASKS.md](TASKS.md).
- **0002 — Danger-Percept-Production-A-B.** Run danger-percept-enabled A/B harness
  (paired runs via `run_headless_with_flags` at
  `crates/xagent-sandbox/src/headless.rs:628-664` with `danger_percept_enabled`
  flag toggle) on production-scale seeds with measurement of avoidance-intent
  fraction, survival (`mean_ticks_alive`), and steering alignment. Record 95% CI
  bootstrap delta, verdict (flip/retire/defer with measured evidence), and decision
  doc with unlock conditions if deferred. See [TASKS.md](TASKS.md).
- **0003 — Innate-Instincts-Graduation-Status.** `innate_instincts_enabled`
  already has a 0013 REJECT decision doc (`0013-INNATE-INSTINCT-DECISION.md`).
  Record that decision as either (a) terminal: flag stays default-off, seeding
  mechanism shipped complete, not revisiting unless the credit path fundamentally
  changes, or (b) gated-on-credit-path: unlock when the credit-path bottleneck
  (steering alignment above 0.62) is solved, because a seeded prior can only help if
  the credit path can act on it. Update the plan STATUS row and decision doc to
  reflect the terminal or gated choice. See [TASKS.md](TASKS.md).

## Origin -> workstream mapping

| Finding (2026-06-25) | Addressed by |
|---|---|
| Effort-rebased fitness math fixed but graduation deferred (F8, 1) | `0001` |
| Danger-percept hardened and tested but no flip-or-retire decision recorded (F8, 2) | `0002` |
| Innate-instincts already failed prove-or-kill gate but decision not marked terminal or gated-on-credit-path (F8, 3) | `0003` |

## Locked decisions

- **No shipped code changes; measurement and decision recording only.** This plan
  measures existing gated levers and records graduation decisions; it does not flip
  any defaults or integrate new mechanisms. All three flags
  (`effort_rebased_fitness`, `danger_percept_enabled`, `innate_instincts_enabled`)
  remain default-off. The plan is gate-passing measurement + decision docs, not code
  integration. Gate: any verdict of FLIP is recorded as a decision only — the actual
  default flip is a separate follow-up change.
- **Bootstrap 95% CI is the decision evidence standard.** Every A/B harness output
  must include point estimate and 95% CI (lower 2.5th, upper 97.5th percentile) for
  all metrics. A single-run or 10-gen measurement is not sufficient for flip/retire
  decisions; bootstrap N=100 replicates at production scale (100 population, 50
  generations) to establish statistical precision. The decision rule thresholds are
  documented in each workstream's ARCHITECTURE section. Gate: a decision lacking a
  reported 95% CI on every metric is not acceptable evidence and the harness is re-run.
- **Steering alignment is captured as a deferral signal, not a hard gate.** The
  0018 gradient-shaping work showed that magnitude (400× amplification) does not
  improve steering (stays at chance 0.498). The bottleneck is credit alignment
  (timing + direct supervision), not learning-rate magnitude. Danger-percept and
  effort-fitness may improve intent measurement, but steering will remain in the
  chance band [0.38, 0.62] until the credit path is addressed. Record
  `steering_alignment` in the danger-percept A/B but use it to explain deferral
  (e.g., "intent improves but steering stays at chance → unlock when credit path is
  fixed") rather than a flip gate. Gate: steering in the chance band routes a verdict
  to DEFER-WITH-GATE, never to RETIRE on the steering axis alone, respecting the 0018
  findings.

## Out of scope

- **Credit-path hypothesis or fix.** The 0018 review and triple-falsification of
  credit-path mechanisms mean the next research direction must either name a new
  mechanism class or step back and question the TD(λ) structure. That is Plan 0019 or
  a structural rethink, not part of this graduation-decision plan.
- **Cortex workgroup restructuring or throughput optimization.** The 0017 cortex
  budget is unmet on real GPU (1.1% vs 50% target); the structural limit is 7% lane
  occupancy (out of scope for 0017). That is a separate workgroup-redesign plan, not
  part of lever graduation.
- **Flipping any default or integrating new code.** This plan records decisions;
  the actual flip (if a FLIP verdict lands) is a follow-up change to the config
  presets. It is purely measurement + decision doc, not code landing. Mirrors the
  "no shipped code changes" locked decision to draw the boundary: this plan ends at a
  recorded verdict, never at a changed default.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
