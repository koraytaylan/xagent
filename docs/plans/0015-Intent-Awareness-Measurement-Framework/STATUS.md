# Plan 0015 — Intent & Awareness Measurement Framework — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete.
_Last updated: 2026-06-22, against `develop`._

- **Goal:** Add the **approach-intent** counterpart to the project's existing
  **avoidance-intent** machinery and measure the baseline. Count-based food-side
  counters (`P_APPROACH_SENSE_RANGE_TICKS` / `P_APPROACH_TURNS_TOWARD`) mirroring
  the avoidance pair, an `agent_approach_accumulate` called from the existing
  thread-0 avoidance block, generation-cumulative across both death paths, exposed
  on `AgentTelemetry`, routed into `AgentFitness` and a population
  `approach_intent_fraction`, plus a baseline distribution (incl. p25/p50/p75) of
  both intent fractions at default config, and documentation of the shipped
  telemetry. All observational only (zero impact on learning), respecting the
  homeostasis-only constraint. The seeded A/B harness that classifies agents
  deliberate-vs-incidental is **deferred to a follow-up plan**, authored against
  this plan's measured baseline (thresholds cannot be set defensibly before the
  baseline exists, and the baseline may show intent at chance).
- **Root cause:** Plan 0009 shipped a complete avoidance-intent measurement
  (`agent_avoidance_accumulate`, the `P_AVOIDANCE_*` slots, `Agent`/`AgentFitness`
  fields, `compute_avoidance_intent_fraction`, `behavior_metric` column, a passing
  discrimination test), but never added the symmetric approach-intent measurement
  for food, and the baseline intent distribution under pure homeostatic learning
  (post-Plan-0012) is unknown. Without the approach signal and the baseline, there
  is no empirical way to characterize whether food-seeking is deliberate or
  incidental.
- **Approach:** Mirror, don't invent. Copy the existing avoidance machinery at
  every layer for food — physics slots, accumulation function, dual death-path
  save/restore, state→cache copy, `AgentFitness` field, population fraction,
  `behavior_metric` column — then capture the across-agent baseline distribution of
  both intent fractions and document the shipped telemetry. Every task is
  observational only and respects the homeostasis-only constraint. The A/B
  validation harness is deferred to a follow-up plan authored against the baseline
  this plan measures.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Approach-Intent Telemetry Pipeline | `wire-approach-intent-counters` | ✅ Done |
| 0002 | Homeostasis-Only Intent Metrics | `populate-approach-intent-fraction` | ✅ Done |
| 0003 | Baseline Measurement & Documentation | `measure-baseline-intent-distribution`, `document-intent-measurement-framework` | ✅ Done |
