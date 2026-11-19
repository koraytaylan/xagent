---
id: run-summary
title: "In-run summary with season-relative food-kind windows"
workstream: "0200"
kind: task
role: developer
depends_on: [evolution-mode-config]
gated: false
touches: [crates/xagent-sandbox/src/lib.rs, crates/xagent-sandbox/src/run_summary.rs]
status: todo
merged_as: ""
checks:
  - {name: focused-proof, command: "cargo test -p xagent-sandbox --lib run_summary::tests::"}
  - {name: sandbox-clippy, command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings", timeout_ms: 3600000}
  - {name: sandbox-fmt, command: "cargo fmt -p xagent-sandbox -- --check", fix: "cargo fmt -p xagent-sandbox"}
  - {name: sandbox-guard, command: "cargo test -p xagent-sandbox --test contributing_guard"}
---

# In-run summary with season-relative food-kind windows

## Outcome

Pure module `crates/xagent-sandbox/src/run_summary.rs`, declared `pub mod run_summary;` in `lib.rs`. It provides `RunSummary`, `KindWindowTotals`, `RunSummaryAccumulator` and `RunSummaryError` with the shapes in ARCHITECTURE.md "Interfaces (binding)" → "Run summary", and reads only copies of physics and food rows (INV-16).

## Scope

- `lib.rs`: add `pub mod run_summary;` alphabetically.
- `run_summary.rs` (new): `//!` doc; constants `WINDOW_FRACTION_NUMERATOR: usize = 3`, `WINDOW_FRACTION_DENOMINATOR: usize = 10`, `COUNTER_EXACT_LIMIT: f32 = 16_777_216.0`; the types; `run_summary::tests`. Every `pub` item has a `///` doc.

## Non-goals

- No GPU, kernel, database or file I/O; driver wiring is later work.
- No verdict logic, thresholds, `EXPECTED_*` constants or new physics or food constants.
- No edits outside the two touched files; no plan, task or workstream identifiers in source or tests.

## Prerequisites

- This plan's `evolution-mode-config`: `xagent_shared::{EvolutionMode, Inheritance}`.
- Plan 0004 (committed text, absent from baseline): `FoodKind` (`First`, `Second`); `P_HARMFUL_MEAL_COUNT` (slot 53); `nearest_food_of_kind(position: [f32; 3], facing: [f32; 2], food_state: &[f32], kind: FoodKind, range: f32) -> Option<NearestFood>` with public `NearestFood { distance, bearing }` (only `bearing` is read).
- Baseline: `SCENT_RANGE`; `PHYS_STRIDE`, `P_ALIVE`, `P_FOOD_COUNT`, `P_DEATH_COUNT`, `P_MOTOR_TURN_OUT`, `P_POS_X`, `P_POS_Z`, `P_FACING_X`, `P_FACING_Z`, `FOOD_STATE_STRIDE`.

## Decisions

- Import slots and constants by name; no literal indices.
- Errors: `RunSummaryError` variants `PhysicsLengthMismatch`, `FoodLengthMismatch` (length not a multiple of `FOOD_STATE_STRIDE`), `TickNotAfterPrevious`, `CounterDecreased`, `InvalidCounter` (non-finite, fractional, negative or at least `COUNTER_EXACT_LIMIT`), `AgentOutOfRange`, `MissingNourishingKind` (kinds on and either `begin_season(_, None)` or a sample with no open season), `SizeOverflow`. Derives `Debug, Clone, PartialEq, Eq`, with `Display` and `Error`.
- Counters `P_FOOD_COUNT`, `P_DEATH_COUNT`, and `P_HARMFUL_MEAL_COUNT` (kinds on only) go through one validating helper, then one `u64` conversion helper with the cast `#[allow]`. Deltas go to run totals and the open season's current interval; baselines then update.
- Turn sample per alive agent: position `[P_POS_X, 0.0, P_POS_Z]` (0004 reads only position indices 0 and 2, so y is `0.0`), facing `[P_FACING_X, P_FACING_Z]`. Nourishing bearing `b` adds `t * approach_sign_of(b)` to toward-nourishing. Harmful bearing `b` adds `-t * approach_sign_of(b)` to away-from-harmful. `t` is `P_MOTOR_TURN_OUT`; `None` adds nothing. The harmful kind comes from a private `other_kind` match.
- Sign helper: `fn approach_sign_of(bearing: f32) -> f32`, private to `run_summary.rs` (not `pub`). It returns `0.0` when `bearing` is `0.0` or NaN, and `-bearing.signum()` otherwise. The zero case must be explicit because `f32::signum(0.0)` is `1.0`. It re-states 0004's convention `approach_sign = -signum(bearing)` (0004's sign test `executed_turn × bearing < 0` means a turn toward food). It exists locally because 0004 exports no sign helper.
- Windows: a season with `n` intervals (its actual length) has `w = n * 3 / 10` (integer division). Early is `0..w`, late is `n - w..n`. Since `2w <= 0.6n < n`, they are disjoint, so no interval counts twice. A season is classified when it closes.
- Kinds off: `nourishing` is ignored, no interval is buffered, `seasons` still counts begun seasons, and `kinds` is `None`.
- Baselines start at 0; `reset_agent_baseline` and `reset_all_baselines` set them to 0.
- `finish(self, ticks)`: `agent_ticks = ticks.checked_mul(u64::try_from(agent_count))`, failures as `SizeOverflow`. `kinds` is `Some` iff kinds are on.

## Steps

**Steps:**
1. Add the module declaration; create `run_summary.rs` with the doc, constants, types, error enum and documented accumulator signatures (`new`, `begin_season`, `reset_agent_baseline`, `reset_all_baselines`, `record_physics_sample`, `finish`).
2. Implement the counter helpers, length, tick and counter checks, baselines and totals, with tests `run_summary_meal_and_death_deltas`, `run_summary_rejects_decreasing_counter_without_reset` and `run_summary_rejects_wrong_physics_length`.
3. Implement turn sampling, the private `approach_sign_of` helper, the interval buffer, `other_kind` and season-close windows, with tests `run_summary_season_windows_first_and_last_thirty_percent`, `run_summary_turn_judged_against_current_season_kind` and `run_summary_approach_sign_matches_food_kind_convention`.
4. Implement the kinds-off path and the `MissingNourishingKind` and `FoodLengthMismatch` checks, with test `run_summary_kinds_off_has_no_kind_totals`.
5. Implement checked `agent_ticks`, with tests `run_summary_agent_ticks_overflow_is_error` and `run_summary_serde_round_trip`.

Test expectations (all in `run_summary::tests`, values hand-computed):
- `run_summary_meal_and_death_deltas`: two agents, three samples, one `reset_agent_baseline` between two; totals equal hand sums.
- `run_summary_rejects_decreasing_counter_without_reset`: lower counter is `CounterDecreased`.
- `run_summary_rejects_wrong_physics_length`: short slice is `PhysicsLengthMismatch`.
- `run_summary_season_windows_first_and_last_thirty_percent`: 10 intervals give early 0-2, late 7-9; a truncated season of 4 gives 1 early and 1 late.
- `run_summary_turn_judged_against_current_season_kind`: agent at origin facing `[0.0, 1.0]`, `First` item at `(1.0, 0.0)`, turn `+1.0`, 10 intervals per season. Season one (`First` nourishing) has late toward sum `+3.0`; season two (`Second` nourishing) has late away-from-harmful sum `-3.0`.
- `run_summary_approach_sign_matches_food_kind_convention`: `approach_sign_of(-0.5) == 1.0`, `approach_sign_of(0.5) == -1.0`, `approach_sign_of(0.0) == 0.0`; agent at origin facing `[0.0, 1.0]` with a `First` item at `(1.0, 0.0)`: `nearest_food_of_kind` gives a bearing `b` (about `-π/2`), and `approach_sign_of(b) * b < 0.0`.
- `run_summary_kinds_off_has_no_kind_totals`: `kinds` is `None`; `seasons` counts begun seasons.
- `run_summary_agent_ticks_overflow_is_error`: `finish(u64::MAX)` with two agents is `SizeOverflow`.
- `run_summary_serde_round_trip`: serde_json round trip equal, for `kinds` `Some` and `None`.

- **Done when:** check focused-proof (`cargo test -p xagent-sandbox --lib run_summary::tests::`) reports the nine tests run_summary_meal_and_death_deltas, run_summary_rejects_decreasing_counter_without_reset, run_summary_rejects_wrong_physics_length, run_summary_season_windows_first_and_last_thirty_percent, run_summary_turn_judged_against_current_season_kind, run_summary_approach_sign_matches_food_kind_convention, run_summary_kinds_off_has_no_kind_totals, run_summary_serde_round_trip and run_summary_agent_ticks_overflow_is_error ok with zero failed, and sandbox-clippy and sandbox-guard pass, proving INV-16.

## Evidence and risks

- The worker has no shell; every proof is a host-run check named in the frontmatter.
- 0004 helpers and `P_HARMFUL_MEAL_COUNT` are absent at baseline, so the packet compiles only after 0004's food-kind-metrics task lands; preflight `pf-run-summary` checks the committed 0004 text for `pub fn nearest_food_of_kind(` and `pub struct NearestFood { pub distance: f32, pub bearing: f32 }`.
- If committed 0004 signatures differ from Prerequisites, report the mismatch with file and line; edit nothing outside the touched files.
