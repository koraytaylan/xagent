---
id: free-life-metrics-and-log
title: Define the free-life per-tick sample, per-agent metrics and tick log
workstream: "0400"
kind: task
role: developer
depends_on: []
gated: false
touches:
  - crates/xagent-sandbox/src/lib.rs
  - crates/xagent-sandbox/src/free_life/mod.rs
  - crates/xagent-sandbox/src/free_life/metrics.rs
  - crates/xagent-sandbox/src/free_life/tick_log.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --lib free_life::"
    timeout_ms: 3600000
  - name: sandbox-guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
  - name: sandbox-clippy
    command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: "cargo fmt -p xagent-sandbox -- --check"
    fix: "cargo fmt -p xagent-sandbox"
---

# Define the free-life per-tick sample, per-agent metrics and tick log

## Outcome

The `xagent_sandbox::free_life` module exists with `metrics` (sample, config,
streaming accumulator, `AgentMetrics`, `WindowMetrics`, constants) and
`tick_log` (validating writer and reader). The code is pure CPU: no GPU, kernel,
physics, default-behaviour or other-crate change.

## Authority

Types, constants, tick log format and metric definitions are fixed by the
ARCHITECTURE.md sections "Free-life types (free-life-metrics-and-log)" and
"Metric definitions (free-life-metrics-and-log)". Implement them exactly.

## Scope

- `crates/xagent-sandbox/src/lib.rs`: add `pub mod free_life;` between
  `pub mod bench;` and `pub mod governor;`. No other line.
- `crates/xagent-sandbox/src/free_life/mod.rs`: `//!` docs, then `pub mod metrics;`
  and `pub mod tick_log;` only.
- `crates/xagent-sandbox/src/free_life/metrics.rs`: constants, types,
  accumulator, unit tests.
- `crates/xagent-sandbox/src/free_life/tick_log.rs`: header, columns, writer,
  reader, unit tests.

## Non-goals

- No runner, `run_free_life`, CLI, `headless.rs`, `main.rs` or `tests/free_life.rs`.
- No kernel, WGSL, buffer or governor change. No edit to `crates/xagent-brain`
  beyond the `ENCODED_DIMENSION` import.
- No plan, task or workstream name in any source, comment or string
  (`contributing_guard.rs`).

## Decisions

- `ENCODING_WIDTH = xagent_brain::buffers::ENCODED_DIMENSION`; `PATHWAY_INPUTS: usize = 10`
  is a documented literal.
- `brain_tick` is the 0-based index within the free life.
- `sign(0.0)` is `0.0`; do not use `f32::signum`.
- A death closes an open sighting; the first sample is the baseline (its deltas are 0).
- Late window: `late_length = round(count_as_f64(total) * LATE_WINDOW_FRACTION)`,
  converted back through one helper that clamps to `[0, total]`. A sample is late
  iff `brain_tick >= total - late_length`. This integer-boundary rule replaces the
  floating `(1 - 0.3) * total` wording in the architecture, which can misplace the
  boundary by one tick. Tests: total 10 gives late ticks 7, 8, 9; total 300,000
  gives a start of 210,000.
- Clippy pedantic: no `as` between integer types or from integer to float. The
  only allows are `#[allow(clippy::cast_precision_loss)]` on `count_as_f64`
  (reason: counts stay far below 2^53) and `#[allow(clippy::cast_possible_truncation,
  clippy::cast_sign_loss)]` on the late-length helper, each with its reason in the
  doc comment. Histogram bins compare against upper edges; no cast.
- Tick log: `MAX_EXACT_F32_INTEGER: u64 = 1 << 24`; `MAX_HEADER_BYTES: usize = 65_536`,
  the reader's header read cap.
- Keep each function under about 50 lines; split accumulator work into helpers.

## Required unit tests

Named exactly; each under `free_life::metrics::tests::` or
`free_life::tick_log::tests::`.

- metrics: `sign_zero_is_zero`, `pearson_positive_pair`,
  `pearson_zero_variance_is_zero`, `sighting_meal_inside_grace`,
  `sighting_expires_after_grace`, `hazard_death_counted`,
  `off_hazard_death_not_counted`, `sighting_open_at_finish`, `late_window_boundary`,
  `turn_toward_food_left_right`, `turn_away_from_hazard_off_ground`,
  `correlation_positive_sign`, `value_gap_skips_empty_bin`,
  `credit_two_tick_hand_computed`, `late_credit_excludes_early_sample`,
  `death_resets_traces`.
- tick_log: `column_count_matches_columns`, `round_trip_all_columns`,
  `writer_finish_short_write`, `reader_rejects_truncated_payload`,
  `reader_rejects_wrong_row_count`, `reader_rejects_wrong_column_list`,
  `reader_rejects_malformed_header`, `reader_rejects_header_without_newline`.

`round_trip_all_columns` uses two agents over two ticks and reads every column
back by name.

**Steps:**

1. Wire the module in `lib.rs` and `free_life/mod.rs`. In `metrics.rs`, define the
   constants, `ENCODING_WIDTH`, `PATHWAY_INPUTS`, the public types, `sign`,
   `count_as_f64`, the late-length helper and the Pearson helper, with doc comments.
   Add `sign_zero_is_zero`, `pearson_positive_pair` and `pearson_zero_variance_is_zero`.
2. Implement `AgentAccumulator::new`, `push` and `finish`: late-window split, meal
   and death deltas, hazard deaths and the sighting state machine. Add the
   sighting, hazard-death and `late_window_boundary` tests from the list above.
3. Implement the turn, correlation and value-gap metrics. Add
   `turn_toward_food_left_right`, `turn_away_from_hazard_off_ground`,
   `correlation_positive_sign` and `value_gap_skips_empty_bin`.
4. Implement the credit traces, reset at a death, with credit summed over window
   samples only. Add `credit_two_tick_hand_computed`,
   `late_credit_excludes_early_sample` and `death_resets_traces`.
5. Implement `tick_log.rs` writing: header struct, `columns()`, `COLUMN_COUNT`,
   `MAX_EXACT_F32_INTEGER`, `TickLogWriter` (`create`, `write_sample`, `finish`) and
   `TickLogError`. Add `column_count_matches_columns`, `round_trip_all_columns` and
   `writer_finish_short_write`.
6. Implement `read_tick_log`, `TickLog` and `column_index` under the reader rules
   of the architecture. Add the five `reader_rejects_*` tests. The host then runs
   the task checks; fix any findings in this task's files only.

- **Done when:** `cargo test -p xagent-sandbox --lib free_life::` passes with zero failures and lists every test named above, and these tests prove INV-6 (metrics follow the definitions exactly) and INV-7 (the tick log round-trips every column and the reader rejects any size, header or column list other than the specified one), each failing if the code under test were deleted or the `food_side` sign were flipped.
