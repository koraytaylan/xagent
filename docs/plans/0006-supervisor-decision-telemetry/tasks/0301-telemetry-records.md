---
id: telemetry-records
title: "Pure telemetry records, series and config diff"
workstream: "0300"
kind: task
role: developer
depends_on:
  - boundary-detail
gated: false
status: done
merged_as: "5156034ca53c7f78f859050977d3674d3e762973"
touches:
  - crates/xagent-sandbox/src/supervisor_telemetry.rs
  - crates/xagent-sandbox/src/lib.rs
  - crates/xagent-sandbox/src/run_summary.rs
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --lib supervisor_telemetry::tests::"
  - name: summary-proof
    command: "cargo test -p xagent-sandbox --lib run_summary::tests::"
  - name: fmt
    command: "cargo fmt --all -- --check"
    fix: "cargo fmt --all"
  - name: clippy
    command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
---

# Pure telemetry records, series and config diff

## Outcome

A new pure module `crates/xagent-sandbox/src/supervisor_telemetry.rs` turns
data the supervisor core and run summary already expose into telemetry records:
slot snapshots, boundary records with the newborn's config difference, series
points, intake rows, late-window measures and progress records. It has no GPU,
SQLite, file I/O or wall-clock use. `run_summary.rs` gains one read-only getter.
No decision changes.

## Decisions

- `lib.rs`: `pub mod supervisor_telemetry;` after `supervisor_store`.
- `pub const TELEMETRY_CADENCE_SAMPLES: u64 = 10;`
- `ConfigFieldDiff { pub field: String, pub parent: serde_json::Value, pub newborn:
  serde_json::Value }`. `config_diff(parent: &BrainConfig, newborn: &BrainConfig)
  -> Vec<ConfigFieldDiff>` (`xagent_shared::config::BrainConfig`): serialize both
  with `serde_json::to_value`; if either is not an object, return empty. Else
  take the sorted union of keys (`BTreeSet<String>`) and push each key whose
  values differ (missing key is `Value::Null`).
- `BoundaryRecord { pub detail: BoundaryDetail, pub newborn_occupant: Option<u64>,
  pub config_diff: Vec<ConfigFieldDiff>, pub retired_lifespan_ticks: Option<u64> }`.
  Newborn fields are set only for a `Replaced` boundary with a replacement input.
  Lifespan is `tick.checked_sub(retired_born_tick)`.
- `Metric` (`Clone, Copy, Debug, PartialEq, Eq`) with `as_str` returning exactly
  `food_per_agent_tick`, `deaths_per_agent_tick`, `window_score_spread`,
  `retired_lifespan_ticks`, `archive_best_outcome`, `archive_median_outcome`,
  `late_toward_nourishing`, `late_away_harmful`, `late_nourishing_share`.
- `SeriesPoint { tick: u64, metric: Metric, value: f64 }`;
  `IntakeRow { occupant_id: u64, slot: usize, age_ticks: u64, meals: u32 }` (one
  per standing on a persisted frame);
  `ProgressRecord { tick, total_ticks: Option<u64>, agent_tick_budget: u64,
  ticks_to_boundary: u64, nourishing: Option<FoodKind>, next_flip_tick:
  Option<u64>, ticks_per_second: Option<f64> }` with `ticks_to_boundary =
  next_boundary_tick.saturating_sub(tick)`;
  `TelemetryFrame { tick, slots: Vec<SlotStanding>, boundary:
  Option<BoundaryRecord>, series, intake, progress, persist: bool }`.
- All records derive `Clone, Debug, PartialEq`. No serde derive here.
- Input: `SampleInput<'a> { tick, standings: &'a [SlotStanding], boundary:
  Option<&'a BoundaryDetail>, replacement: Option<NewbornInput<'a>>,
  archive_outcomes: &'a [f32], total_ticks: Option<u64>, agent_tick_budget: u64,
  next_boundary_tick: u64, nourishing: Option<FoodKind>, next_flip_tick:
  Option<u64>, ticks_per_second: Option<f64>, season_flipped: bool }` and
  `NewbornInput<'a> { newborn_occupant: u64, parent_config: &'a BrainConfig,
  newborn_config: &'a BrainConfig }`.
- `TelemetryBuilder::new(agent_count: usize)`, `on_sample(&mut self,
  SampleInput<'_>) -> TelemetryFrame`, `on_season_closed(&mut self, tick: u64,
  totals: &KindWindowTotals)`.
- Persist: `tick % (SUPERVISOR_SAMPLE_TICKS * TELEMETRY_CADENCE_SAMPLES) == 0 ||
  boundary.is_some() || season_flipped` (`xagent_shared::config::SUPERVISOR_SAMPLE_TICKS`).
  `slots`, `progress`, `persist`, and `boundary` (when input has one) are always
  set; `series` and `intake` only on persisted frames.
- Rates: keep per-slot previous meal and death counters and the previous
  persisted tick (zero initially, `Vec` of length `agent_count`). On each
  persisted frame, for standings with `slot < agent_count`, delta = saturating
  `current − previous`, except a slot whose occupant changed contributes its
  current counters in full. `food_per_agent_tick = sum / (agent_count ×
  (tick − prev_tick))`, deaths likewise; zero denominator omits. Then store the
  current counters, occupants and tick.
- `window_score_spread`: max − min of finite `Some` window scores; omitted under
  two. `retired_lifespan_ticks`: on replacement frames only.
- Archive: drop non-finite outcomes; best is the first finite (rank order);
  median is `crate::supervisor::median` of finite values. Omitted if none.
- Late window: keep previously seen closed totals (six numbers, zero initially).
  `on_season_closed` differences them and queues: `late_toward_nourishing = Δsum
  / Δsamples`, `late_away_harmful` likewise (omitted on zero sample delta),
  `late_nourishing_share = Δnourishing / (Δnourishing + Δharmful)` (omitted on
  zero sum), at `tick`. Queued points drain into the next `on_sample` frame.
  Callers call `on_season_closed` in the same tick, before the `season_flipped`
  `on_sample`, so the points land in a persisted frame.
- Memory O(`agent_count`). No panic on any input: no `unwrap`, `expect`, or
  unchecked indexing; non-finite values are omitted.
- `run_summary.rs`: `pub fn closed_totals(&self) -> Option<&KindWindowTotals>`
  returning `self.kinds_on.then_some(&self.totals)`. `totals` already holds
  closed-season sums, so nothing else changes.
- Plain-prose doc comments; no plan numbers, task slugs or workstream words.

## Non-goals

- No edits to `supervisor.rs`, `supervisor_run.rs`, `supervisor_store.rs`,
  `sim_runtime.rs`, `headless.rs` or other tests. No GPU, SQLite, file I/O or
  wall clock in the new module. No serde derives, store writes or driver calls.
- No decision, random draw, season schedule or replacement change.
- If `supervisor::median`, `SlotStanding`, `BoundaryDetail` or `FoodKind` differ
  from what is stated here, stop and report rather than editing `supervisor.rs`.

**Steps:**

1. Add `pub mod supervisor_telemetry;` to `lib.rs` after `supervisor_store`.
   Create `supervisor_telemetry.rs` with the constant, `Metric`, the record
   structs, `SampleInput`, `NewbornInput`, and the `TelemetryBuilder` struct
   with `new`. Names exactly as in Decisions. Derive `Clone, Debug, PartialEq` on
   `BoundaryRecord`, `TelemetryFrame` and all types they contain.
2. Implement `config_diff` as specified.
3. Implement `on_sample`: always-filled parts (`slots`, `boundary`, `progress`,
   `persist`), then persisted parts (`intake`, the two rates with the previous
   counter state, spread, lifespan, archive points), draining queued late points.
4. Implement `on_season_closed` with the stored totals and the three formulas,
   omitting zero-denominator points.
5. In `run_summary.rs`, add the `closed_totals` getter with a doc comment. In its
   `mod tests`, add `closed_totals_matches_after_begin_season`: with food kinds
   on, `begin_season(0, Some(FoodKind::First))`, record a physics sample, then
   `begin_season(…, Some(FoodKind::First))` to close it; assert `closed_totals()`
   equals the closed totals, and another sample in the new open season leaves
   it unchanged.
6. In `supervisor_telemetry.rs` `mod tests`, with directly built `SlotStanding`
   and `BoundaryDetail` values, add: `config_diff_lists_changed_fields_in_key_order`
   (clone `BrainConfig::default()`, change two public top-level fields, assert
   sorted field list and values, and that equal configs give empty);
   `rates_plain_and_replacement` (two persisted frames plus an occupant change,
   hand-computed rates); `spread_lifespan_archive` (NaN excluded, lifespan,
   archive best and median for odd and even counts, empty archive omitted);
   `persist_flags_cadence_boundary_and_season`; `late_window_points_from_closed_totals`;
   `omitted_points_on_zero_denominators_and_nan`.

- **Done when:** the focused-proof check passes with the new supervisor_telemetry::tests covering INV-5 (config difference listed in key order, empty for equal configs) and INV-6 (rates, spread, lifespan, archive, persist flags, late-window points, omitted points on zero denominators and NaN), the summary-proof check passes with closed_totals_matches_after_begin_season green, and the fmt, clippy and guard checks pass, proving INV-5 and INV-6.
