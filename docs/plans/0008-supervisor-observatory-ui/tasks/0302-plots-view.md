---
id: plots-view
title: Time-series charts with axis markers and the learning view
workstream: "0300"
kind: task
role: developer
depends_on:
  - observatory-model
gated: false
touches:
  - crates/xagent-sandbox/src/observatory/plots.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: cargo test -p xagent-sandbox --lib observatory::plots::tests
    timeout_ms: 3600000
  - name: sandbox-clippy
    command: cargo clippy -p xagent-sandbox --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: rustfmt --check --edition 2021 crates/xagent-sandbox/src/observatory/plots.rs
    fix: rustfmt --edition 2021 crates/xagent-sandbox/src/observatory/plots.rs
---

# Time-series charts with axis markers and the learning view

## Outcome

`crates/xagent-sandbox/src/observatory/plots.rs` (a stub created by
`observatory-model`) draws the four population charts over ticks with
replacement and season-flip markers, and the learning view: cumulative intake
against occupant age plus the late-window measures table.

## Decisions (binding)

- `pub struct ChartSeries { pub label: String, pub points: Vec<(f64, f64)> }`,
  `pub struct AxisMarker { pub tick: u64, pub kind: MarkerKind }` and
  `pub enum MarkerKind { Replacement, SeasonFlip }`.
- `pub fn axis_markers(&ObservatoryModel) -> Vec<AxisMarker>`: one `Replacement`
  per boundary with `picked.is_some()`, one `SeasonFlip` per timeline event of
  kind `Season`, none for idle boundaries, sorted by tick.
- `pub fn chart_series(&ObservatoryModel) -> [Vec<ChartSeries>; 4]`: chart 1 food
  and deaths per agent-tick; chart 2 window-score spread; chart 3 retired-occupant
  lifespan; chart 4 archive best and median. `None` and non-finite values are
  skipped.
- `pub fn line_chart(ui, title: &str, series: &[ChartSeries], markers:
  &[AxisMarker])`: painter-based like `ui.rs` `render_fitness_chart`
  (`allocate_exact_size` with `Sense::hover()`, `painter_at(rect)`, a `vline` per
  marker with a distinct colour per kind). Width is `ui.available_width()`
  clamped to a finite named minimum, because it can be infinite inside a scroll
  area (CLAUDE.md egui notes); the height is a named const.
- `pub fn intake_curves(&ObservatoryModel, selected: &[u64]) -> Vec<ChartSeries>`:
  one series per selected tracked occupant, x = age ticks, y = cumulative meals.
  `pub fn late_window_rows(&ObservatoryModel) -> Vec<[String; 4]>`: occupant,
  meals per alive tick, turn toward food, sighting meal share (`n/a` without a
  late window).
- Renderers `show_charts(ui, &ObservatoryModel)` and `show_learning(ui,
  &ObservatoryModel, &mut LearningViewState)` (`LearningViewState { selected:
  Vec<u64> }`, `Clone, Debug, Default`; a checkbox per tracked occupant, the
  intake chart without markers, then an `egui::Grid` of `late_window_rows`).
- No new dependency, no `ScrollArea`, no `GpuKernel`, `wgpu`, `rusqlite`,
  `Connection`, `store_read::` or `read_full_state_blocking` name, no planning
  references, `TODO` or `FIXME`; tick to `f64` conversion goes through one
  documented helper.

**Steps:**

1. Read `observatory/mod.rs` and `ui.rs` `render_fitness_chart`; report a missing
   model type as a blocker.
2. Write `axis_markers`, `chart_series`, `intake_curves` and `late_window_rows`.
3. Write `line_chart`, `show_charts` and `show_learning`.
4. Add `#[cfg(test)] mod tests` with `plots_markers_at_replacements_and_flips`
   (exactly the replacement and flip ticks, none for an idle boundary),
   `plots_chart_series_cover_four_charts` (each chart's labels and points, `None`
   and NaN skipped), `plots_intake_curves_use_age` (x is age, y cumulative meals,
   only selected occupants) and `plots_late_window_rows_list_measures`.
5. Make the three checks pass.

- **Done when:** `focused-proof` passes with `plots_markers_at_replacements_and_flips`, `plots_chart_series_cover_four_charts`, `plots_intake_curves_use_age` and `plots_late_window_rows_list_measures` reported ok, proving INV-7 and INV-8.
