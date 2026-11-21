---
id: boundary-detail
title: "Draw-free slot standings and boundary detail in the core"
workstream: "0200"
kind: task
role: developer
depends_on: []
gated: false
status: done
merged_as: "922cbaf2bb99dd11ba8b73deac3bb143f502c824"
touches:
  - crates/xagent-sandbox/src/supervisor.rs
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --lib supervisor::tests::"
  - name: sandbox-build
    command: "cargo test -p xagent-sandbox --no-run"
    timeout_ms: 3600000
  - name: fmt
    command: "cargo fmt --all -- --check"
    fix: "cargo fmt --all"
  - name: clippy
    command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
---

# Draw-free slot standings and boundary detail in the core

## Outcome

`supervisor.rs` (pure: no GPU, SQLite or I/O) exposes, without any random
draw, each slot's standing, the full detail of the last boundary, the next
boundary tick, each occupant's parent, and the parent's config on the
replacement outcome. Every random stream keeps its draw count and order.

## Decisions

- Public types as named in ARCHITECTURE.md "Core additions": `EligibilityGround`,
  `SlotStanding`, `TieBreak`, `ParentDraw`, `BoundaryDetail`, `IdleReason`. Each
  derives `Clone, Debug, PartialEq` (no `Eq`, because of `f32`). Doc comments
  are plain prose, with no plan number, task slug or workstream word.
- `SlotLedger` gains `pub parent: Option<ParentRef>`. `SlotLedger::born` keeps
  its signature and sets `None`, so the test helper `ledger()` is unchanged.
  `complete_replacement` sets `ledger.parent = Some(plan.parent)` on the newborn.
- `ReplacementOutcome` gains `pub parent_config: BrainConfig`, cloned from the
  config `resolve_parent` returned, before mutation.
- `median` becomes `pub(crate) fn median(...)`, otherwise unchanged.
- Grounds, in this order (identical to today's code): `pending_death` gives
  `Death`; no full window gives `NoFullWindow`; `age(tick) < min_age_ticks` gives
  `TooYoung`; no finite peer median gives `NoPeerMedian`; `own < median` gives
  `BelowMedian { score, median }`; otherwise `AtOrAboveMedian { score, median }`.
  A NaN own score falls through to `AtOrAboveMedian`, as today.
- A private `fn eligibility_pass(slots, tick, config) -> (Vec<usize>, Vec<usize>,
  Vec<SlotStanding>)` builds one standing per slot plus the `(deaths, under)`
  lists in slot order. `fn eligible_slots(` keeps its signature and delegates to
  it. `standings` returns the same pass, so the query and the decision read the
  same numbers.
- `SlotStanding`: `age_ticks = age(tick)`; `alive` and `meals`/`deaths` from the
  latest window sample (`alive` false and counters 0 before any sample; counters
  via `count_u32`, 0 when `None`); `window_score = full_window_score()`;
  `peer_median`; the ground; `parent` from the ledger.
- `choose_candidate` keeps its name, its `Option<(usize, ReplaceReason)>` return
  (the existing tests destructure and compare it as that pair) and its single
  `random_range` draw, and stays on the production decision path. The boundary
  detail records the drawn index into the chosen pool as the chosen slot's
  position in that pool, whose entries are distinct slots, so `slot ==
  pool[drawn_index]` and no extra draw is made. `choose_candidate` is not
  required to return the index.
- `plan_parent(&mut self, slot: usize) -> ParentRef {` keeps this exact line
  (the plan preflight greps it). Only the archive draw (`Darwinian` with a
  non-empty archive) stores `ParentDraw { pool_len, drawn_index }` in a new
  private `parent_draw` field; peer parents store nothing.
- A private `last_boundary: Option<BoundaryDetail>` is set at each boundary in
  `record_sample`, never at a `Recorded` sample. `Replaced` candidates are the
  standings of every slot grounded `Death` or `BelowMedian`. The tiebreak holds
  the chosen pool and `drawn_index`. `parent_draw` is taken right after
  `plan_parent`. The retiree's occupant and born tick are read before archiving.
  `Idle` holds all standings and counts the four grounds in `IdleReason`.
- `standings(&self, tick)` is read-only and draws nothing. `last_boundary`
  returns the stored detail. `next_boundary_tick` returns
  `self.boundary.next_tick()`.
- `AgentBrainState` has no `PartialEq`; tests compare its `Debug` string.

## Non-goals

- No change to replacement choice, parent draw, newborn mutation, any random
  stream, the season schedule, the GPU or SQLite. No edit to any file other than
  `supervisor.rs`, and no existing test body changes.
- If a caller outside `supervisor.rs` builds `SlotLedger` or
  `ReplacementOutcome` by struct literal, stop and report it instead of editing
  outside the touch. (Planning grep found none.)

**Steps:**

1. Add `pub parent` to `SlotLedger` (set by `born` to `None`, set on the newborn
   in `complete_replacement`). Add `pub parent_config` to `ReplacementOutcome`.
   Change `median` to `pub(crate)`. The existing tests still compile unchanged.
2. Add the six public types with doc comments. Add `eligibility_pass` in the
   ground order; make `eligible_slots` return the same two lists by delegating.
   Keep `choose_candidate`'s return type; record its drawn index as above.
3. Add the private `parent_draw` and `last_boundary` fields, initialised in
   `Supervisor::new`. Record the archive draw in `plan_parent`, keeping its
   signature line.
4. In `record_sample`, at each boundary, call `eligibility_pass` once, build the
   `Replaced` or `Idle` detail as in Decisions, and store it in `last_boundary`.
   Keep the draw order and the early `Recorded` return exactly as they are.
5. Add `pub fn standings(&self, tick: u64)`, `pub fn last_boundary(&self)` and
   `pub fn next_boundary_tick(&self)`, with doc comments.
6. In `mod tests`, add tests prefixed `boundary_detail_` or `standings_`:
   - INV-3: for `Darwinian` and `WeightCopy`, two identically seeded supervisors
     get the same counters for enough samples to see several replacements and an
     idle boundary. One calls the three queries after every sample and the other
     never does. Assert equal `SampleDecision`s, `ReplacementOutcome`s (brain by
     `Debug`, config, occupant id) and archive entries.
   - INV-4: each `Replaced` detail has every `BelowMedian` score below its median;
     `Death` candidates are exactly the slots that died; `slot ==
     tiebreak.pool[drawn_index]`; an archive parent equals
     `archive().entries()[drawn_index].entry_id`, read right after that
     `record_sample` returns. Each `Idle` detail's four counts sum to the slot count.
   - `next_boundary_tick` equals the tick of the next non-`Recorded` decision.
   - A newborn's `parent == Some(plan.parent)`, and `parent_config` equals the
     parent's config.

- **Done when:** the focused-proof check passes with the existing `supervisor::tests` unchanged, the new `boundary_detail_` and `standings_` test groups green, the INV-3 test showing that calling the queries leaves decisions, outcomes and archives identical for both inheritance modes, the INV-4 assertions holding for every recorded detail, and the sandbox-build, fmt, clippy and guard checks also passing, proving INV-3 and INV-4.
