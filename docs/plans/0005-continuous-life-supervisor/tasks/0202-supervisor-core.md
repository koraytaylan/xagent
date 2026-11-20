---
id: supervisor-core
title: "Pure deterministic supervisor core"
workstream: "0200"
kind: task
role: developer
depends_on: [run-mode-column, run-summary]
gated: false
touches: [crates/xagent-sandbox/src/lib.rs, crates/xagent-sandbox/src/supervisor.rs]
status: done
merged_as: "209ec76f5906b2a50d524598dab73d453a7d34d3"
checks:
  - {name: focused-proof, command: "cargo test -p xagent-sandbox --lib supervisor::tests::"}
  - {name: sandbox-clippy, command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings", timeout_ms: 3600000}
  - {name: sandbox-fmt, command: "cargo fmt -p xagent-sandbox -- --check", fix: "cargo fmt -p xagent-sandbox"}
  - {name: sandbox-guard, command: "cargo test -p xagent-sandbox --test contributing_guard"}
---

# Pure deterministic supervisor core

## Outcome

`lib.rs` declares `pub mod supervisor;` in alphabetical order. The new
`crates/xagent-sandbox/src/supervisor.rs` is a pure CPU core with no GPU, database
or file I/O. Its unit tests run without a GPU.

## Scope

- `lib.rs`: add `pub mod supervisor;` in alphabetical position.
- `supervisor.rs` (new) with an inline `#[cfg(test)] mod tests`; every test is
  named `supervisor::tests::...`.
- Semantics are binding in ARCHITECTURE.md "Interfaces (binding)" → "Supervisor
  core". This packet fixes names and the refinements below.

## Non-goals

- No GPU, kernel, SQLite, file I/O, `WorldState`, run summary or store code.
- No edit to any other file. No new learning rule, reward term or stored score.

## Prerequisites

- `run-mode-column` makes `composite_fitness` and `exploration_grid_cells() ->
  f32` crate-visible. `evolution-mode-config` provides `SupervisorConfig` with
  `validate(agent_count)`, `mutation_strength` and `momentum_decay`.
- `run-summary` edits `lib.rs`; this task depends on it so the `pub mod` line lands
  after that edit.

## Decisions

- **Imports** use the paths `agent/mod.rs` uses (`crate::agent`, `crate::governor`,
  `crate::momentum::MutationMomentum`, `xagent_brain::buffers` `P_*` and
  `PHYS_STRIDE`, `xagent_shared` config types).
- **Streams.** Five `pub const` tags with `///` docs: `BOUNDARY_STREAM =
  0xB0A1_D5C4_0000_0001`, `SEASON_STREAM = 0x5EA5_0000_0000_0002`, `TIEBREAK_STREAM =
  0x71E8_0000_0000_0003`, `PARENT_STREAM = 0x9A4E_0000_0000_0004`, `INITIAL_STREAM =
  0x1A17_0000_0000_0005`. `stream_seed` is splitmix64 of `run_seed ^ tag` with
  named constants `0x9E37_79B9_7F4A_7C15`, `0xBF58_476D_1CE4_E5B9`,
  `0x94D0_49BB_1331_11EB`.
- **Schedule.** `TickSchedule::new(seed, mean_ticks)`: gaps of `SUPERVISOR_SAMPLE_TICKS
  * k`, `k` from `random_range(ceil(m/2)..=floor(3m/2))`, integer arithmetic only,
  `m = mean_ticks / SUPERVISOR_SAMPLE_TICKS`. Methods `next_tick()`, `advance()`.
- **Counters.** `SlotCounters { alive, deaths, food, distance, energy_spent: f32,
  cell: Option<u16> }`; `SlotCounters::from_physics(row, world_size) ->
  Result<_, SupervisorError>`. The `world_size` argument is the one deliberate
  refinement, needed for the cell. Length must equal `PHYS_STRIDE`. `alive = row[P_ALIVE]
  > ALIVE_THRESHOLD` (`0.5`). `cell` is `Some(heatmap_cell(..))` only when alive and
  `P_POS_X`, `P_POS_Z` are finite.
- **Cell and counts.** Private `heatmap_cell` returns `cz * HEATMAP_RES + cx` with
  the same float expressions and clamps as `Agent::record_heatmap`. Private
  `count_u32` is `None` unless finite, `>= 0`, integral and `<= COUNT_EXACT_LIMIT
  (16_777_216.0)`. A `None` among a score's inputs makes that score NaN. Counters are
  never clamped.
- **Ledger.** `SlotLedger` has `pub occupant_id`, `pub born_tick`, `pub birth_brain`,
  `pub config`, and private pending-death flag, last sample, window ring, and life
  cell set. `pending_death` is set when `deaths` rises; the first sample after birth
  is compared with zero (the zeroed spawn row).
- **Scores.** Window score over a full ring (exactly `W / SUPERVISOR_SAMPLE_TICKS + 1`
  samples): `last - first` for deaths, food, distance and energy (negative or `None`
  gives NaN); cells are the distinct `Some` cells of the newer `W /
  SUPERVISOR_SAMPLE_TICKS` samples, not the base sample. Life outcome uses the same
  `composite_fitness` with whole-life counters and `life_cells.len()`.
- **Eligibility and tie-break** follow ARCHITECTURE "Supervisor core". Candidates are
  the death-eligible slots in slot order, else the underperformers; exactly one
  `random_range(0..len)` draw when any exists.
- **Archive.** `ArchiveEntry` (all fields `pub`, `Clone`, `entry_id` monotone from 1).
  `Archive` methods: `insert(entry) -> Admission` where `pub enum Admission { Refused,
  Admitted { evicted: Option<u64> } }`; `entries()` in rank order; `get(entry_id)`;
  `len()`. Non-finite outcomes are refused.
- **Payloads carry the full entry.** `ReplacementOutcome { newborn_brain,
  newborn_config, occupant_id, archived: Option<ArchiveEntry>, evicted: Option<u64> }`
  and `ArchiveChange { slot, entry: ArchiveEntry, evicted: Option<u64> }`, so the
  store writes without re-reading the archive.
- **Parent** is drawn in `record_sample` only when a candidate exists, from the
  parent stream, as ARCHITECTURE states. Darwinian with a non-empty archive gives
  `ArchiveEntry`; an empty archive gives `PeerBirth`; WeightCopy gives `PeerLearned`.
- **Newborn** draws from the parent stream: `mutate_steering_weights`, then
  `mutate_sensory_genes` with `MutationMomentum::new(momentum_decay)`. Learned state
  is `Some` exactly for `PeerLearned`, else `LearnedStateRequired` or
  `LearnedStateUnexpected`.
- **Public API** names are binding: `Supervisor::new(governor, brain, run_seed,
  world_size, agent_count)` (rejects a non-finite or non-positive `world_size`),
  `slot`, `next_sample_tick`, `archive`, `record_sample(tick, counters) ->
  SampleDecision`, `complete_replacement(plan, tick, learned_parent)`, `finish(tick)
  -> Vec<ArchiveChange>`. `SupervisorError` has `Config`, `InvalidWorldSize`,
  `PhysicsRowLength`, `CounterLength`, `TickMismatch`, `UnknownSlot`,
  `LearnedStateRequired`, `LearnedStateUnexpected`, `UnknownArchiveEntry`, with
  `Debug`, `Display` and `Error`.
- Non-finite physics is never an error; it makes a score NaN or a cell `None`.
- Keep functions under about 50 lines; every private helper is used.

**Steps:**

1. Add `pub mod supervisor;` to `lib.rs`. Create `supervisor.rs` with the module doc,
   constants, `stream_seed`, `TickSchedule`, `SupervisorError`, `SlotCounters::
   from_physics`, `heatmap_cell` and `count_u32`.
2. Add `SlotLedger` (record, clear), window score, life outcome, median,
   `eligibility`, and the tie-break with its single draw.
3. Add `ArchiveEntry`, `Archive` (insert, entries, get), `Admission`,
   `ArchiveChange`, `ReplacementOutcome`, parent planning and newborn mutation, then
   `complete_replacement` and `finish`.
4. Add `Supervisor` with `new`, `slot`, `next_sample_tick`, `archive` and
   `record_sample` returning `SampleDecision`.
5. Add `#[cfg(test)] mod tests` with these eleven tests, one per line:
   - `schedule_ticks_on_sample_grid_within_bounds`
   - `schedule_same_seed_same_ticks`
   - `schedule_streams_are_independent`
   - `eligibility_pending_death_persists_until_replaced`
   - `eligibility_underperformer_strictly_below_peer_median`
   - `eligibility_requires_min_age_and_full_window`
   - `eligibility_ignores_archived_scores`
   - `tiebreak_death_first_and_one_slot`
   - `tiebreak_same_seed_same_choice`
   - `archive_bounded_rank_order_and_eviction`
   - `archive_rejects_non_finite_outcome`
6. Add these seven tests, one per line:
   - `parent_draw_from_archive_uses_birth_brain`
   - `parent_fallback_is_best_peer_birth_brain`
   - `newborn_is_mutation_of_parent_birth_brain`
   - `weight_copy_requires_and_uses_learned_state`
   - `window_score_matches_composite_fitness_over_window`
   - `heatmap_cell_matches_agent_record_heatmap`
   - `core_source_has_no_gpu_types`

   Build brains with `fresh_brain_state`, physics rows with `PHYS_STRIDE`, and
   forbidden names with `concat!`.

- **Done when:** the check focused-proof passes with all eighteen tests in Steps 5 and 6 reported ok, the check sandbox-clippy reports no warnings, the check sandbox-fmt passes, and the check sandbox-guard passes; together they prove INV-03, INV-04, INV-05, INV-06 and INV-07.

## Evidence and risks

- The worker has no shell; the host-run checks are the proof.
- The heatmap test relies on `Agent::new` and the public `heatmap` field. If it cannot
  set a position that way, the worker reports the file and line, not an edit outside
  touches.
- Counters are exact `f32` integers below `COUNT_EXACT_LIMIT`; above it a score is
  NaN, which is documented.
