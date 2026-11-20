---
id: return-fit-tool
title: "Offline return-fit tool"
workstream: "0300"
kind: task
role: developer
depends_on: []
gated: false
status: done
merged_as: "630d6113afd5931b63c1a7eb3040ef1074e15d2d"
touches:
  - crates/xagent-sandbox/src/bin/pathway_return_fit.rs
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --bin pathway_return_fit"
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: "cargo fmt -p xagent-sandbox -- --check"
    fix: "cargo fmt -p xagent-sandbox"
  - name: sandbox-clippy
    command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: contributing-guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
    timeout_ms: 3600000
---

# Offline return-fit tool

## Outcome

New binary `crates/xagent-sandbox/src/bin/pathway_return_fit.rs` (Cargo auto-discovery). No other file changes; no new dependency.

- `fit --log <path>... --out <json>`: reads per-brain-tick logs and writes the record for sets `pathway`, `encoding`, `encoding_and_pathway`, with the winner.
- `verify <json>`: re-derives constants, pooled values, SEs, counts, paired SE and winner from `per_agent`; on mismatch prints the field and both values and exits non-zero.

## Scope

1. Reuse via `xagent_sandbox::free_life`: tick log reader (`free_life/tick_log.rs`: JSON header naming columns, then little-endian `f32` rows) and predicates in `free_life/metrics.rs` (`FOOD_IN_VIEW_DISTANCE = 5.5`, `VIEW_HALF_ANGLE = π/4`, `CENTRED_BEARING = 0.15`, `OFF_AXIS_BEARING = 0.45`). If an item is not public, re-implement only that item here with the same format or values; never edit a 0001 file, and record the path per item in the module doc. The logged encoding is the centred `s_memory_key` (128 columns), used as logged.
2. Constants with doc comments (no bare literals except 0, 1, -1): `TD_DISCOUNT: f64 = 0.97` (mirrors `common.wgsl`); `TERMINAL_DEATH_TD_ERROR: f64 = -1.0`; `TRUNCATION_TICKS: usize = 300`; `BLOCK_TICKS: usize = 1000`; `HOLDOUT_MODULUS: u64 = 10`; `HOLDOUT_BELOW: u64 = 3`; `INNER_VALIDATION_MODULUS: u64 = 5`; `LAMBDA_GRID: [f64; 5] = [1e-3, 1e-2, 1e-1, 1.0, 10.0]`; `PATHWAY_WIDTH = 10`, `ENCODING_WIDTH = 128`, `ENCODING_AND_PATHWAY_WIDTH = 138`, each plus one intercept; splitmix increment `0x9E37_79B9_7F4A_7C15`, mixers `0xBF58_476D_1CE4_E5B9` and `0x94D0_49BB_1331_11EB`; `PATHWAY_INPUT_LAG_ROWS: usize = 1` (critic pathway row offset, D5/D8).
3. Method per D8 with these fixed choices:
   - Returns: backward `G_t = r'_t + TD_DISCOUNT * (1 - d_t) * G_{t+1}`, where `r'_t` is `TERMINAL_DEATH_TD_ERROR` when death marker `d_t` is set, else `r_t`; `G` past the log end is 0. Then drop the last `TRUNCATION_TICKS` ticks.
   - Alignment (after returns, before blocks and hold-out): kept row `t` takes its pathway columns from logged row `t − PATHWAY_INPUT_LAG_ROWS` of the same agent; encoding columns and target stay the row's own. Drop each agent's first row and every row `t` where `d_{t−1}` is set, from all three sets.
   - Blocks: `b = t / BLOCK_TICKS`, `t` the original row index; a trailing partial block is kept only if it has at least `BLOCK_TICKS` ticks.
   - Hold-out: `h = next(seed)`, `h = next(h ^ agent)`, `h = next(h ^ block)`; held out iff `h % HOLDOUT_MODULUS < HOLDOUT_BELOW`. `next` adds the increment to the state, mixes it, and returns the mixed value.
   - Ridge: `(XᵀX + λI) β = Xᵀy` in f64 by Cholesky; intercept unpenalised; a non-positive pivot is an error naming the file.
   - λ: blocks with `b % INNER_VALIDATION_MODULUS == 0` are inner validation, the rest inner fit. Pick the highest inner validation R² (ties to smaller λ), then refit on all training blocks.
   - Per-agent held-out scores: R², in-view gap, centred gap (0001 predicates). Empty held-out set or empty class is an error naming the file.
   - Pooled: mean over agents. SE is `sd / sqrt(n)` with sample sd; the paired R² advantage of `encoding_and_pathway` over `pathway` uses `sd(d) / sqrt(n)`. Positive-agent counts per set for both gaps. `n < 2` is an error naming the file.
   - Winner: candidates are sets with both pooled gaps positive; none gives `encoding`. Otherwise the highest pooled R²; if that is `encoding_and_pathway`, `pathway` is a candidate and wins when the paired advantage is not above its SE.
4. Record JSON: `schema_version`; constants (including `pathway_input_lag_rows`); `agents` (`seed`, `agent`, log file name); `sets` keyed `pathway`, `encoding`, `encoding_and_pathway`, each with `per_agent` (`agent`, `lambda`, `held_out_r2`, `in_view_gap`, `centred_gap`), `pooled_*`, their `_se` fields, and the two positive-agent counts; plus `paired_r2_advantage_se` and `winner` (`CriticInput` serde names). Unknown or missing fields and unknown set names are errors.
5. Verify: constants must equal this binary's; recompute pooled values, SEs, counts and paired SE from `per_agent`; real numbers within `1e-9` relative, counts exact; reapply the winner rule.
6. Errors: no `unwrap`/`expect` on I/O or parsing; every read, parse and solve returns `Result` naming the file. Reject non-finite values, wrong row widths, empty sets, fewer than two agents, singular systems. No truncating `as` casts.

## Non-goals

- No gate or route logic, no decision record, no change to D8 constants.
- No plan, task or decision-document names in source or messages (INV-12).

## Tests

Each test is named for what it falsifies and uses named constants.

- Returns (INV-8): rewards `[1.0, 2.0, 3.0]`, no death: `G_2 = 3`, `G_1 = 2 + TD_DISCOUNT * 3`, `G_0 = 1 + TD_DISCOUNT * G_1`; death on the last tick gives `G_2 = -1`; a second life does not leak into the first; `TRUNCATION_TICKS + 2` ticks keep 2.
- Splitmix (INV-8): from state 0, `0xE220A8397B1DCDAF` then `0x6E789E6AA1B965F4`; held-out share over 10,000 blocks is within `0.02` of `HOLDOUT_BELOW / HOLDOUT_MODULUS`; changing the agent changes the pattern.
- Ridge (INV-8): `y = 2 + 3·x1 − x2` on 200 rows recovers coefficients within `1e-3` at `λ = 1e-3`; held-out R² above `0.999` on exact data; below `0.05` on noise; a duplicate column at `λ = 0` errors naming the file.
- Split (INV-8): a trailing partial block shorter than `BLOCK_TICKS` is dropped; the chosen λ has the highest inner validation R², ties smaller.
- Alignment (INV-8), `pathway_columns_lag_one_row_and_drop_life_starts`: a three-row life, then a death, then a second life; each kept row's pathway features equal the previous row's logged whitened inputs, its encoding columns are the same row's, and the first row and first post-death row are absent from all three sets.
- Gaps (INV-8): in-view `1.0`, others `0.0`, gives gap `1.0`; swapped gives `-1.0`.
- Spread (INV-8): `[1.0, 2.0, 3.0, 4.0]` gives mean `2.5` and SE `sqrt(5/3) / 2`; positive count for `[1.0, -1.0, 0.0, 2.0]` is `2`.
- Winner rule (INV-8): all gaps negative gives `encoding`; advantage at or below one SE gives `pathway`; above gives `encoding_and_pathway`.
- Verify (INV-8), in memory: an unchanged `fit` record is accepted; an edited `winner`, a pooled R² changed beyond `1e-9`, a changed SE or count, or a missing `paired_r2_advantage_se` is rejected.
- Malformed input (INV-8): `NaN`, row width 9, too few ticks for a held-out set, and one agent each error naming the file; none panic.

## Steps

**Steps:**

1. Read the integrated 0001 reader and predicates through `free_life`; decide per item whether to import or re-implement, and record it in the module doc.
2. Create the binary with the module doc, constants, clap subcommands, log adapter, returns with the truncation drop, and the return and malformed-input tests.
3. Add the one-row pathway lag with the life-start drop, splitmix, hold-out, block and partial-block functions, Cholesky ridge and the inner-split λ choice, with the alignment, splitmix, ridge and split tests.
4. Add per-agent scoring, pooled values, SEs, counts, paired SE, the winner rule and the `fit` record writer, with the gap, spread and winner-rule tests.
5. Add `verify` and the `verify <json>` subcommand with a failing exit on mismatch, with the verify tests on an in-memory record.
6. Request host checks in order: `focused-proof`, `sandbox-fmt` (fix, then request again), `sandbox-clippy`, `contributing-guard`; fix findings only in the touched file.

- **Done when:** `cargo test -p xagent-sandbox --bin pathway_return_fit` passes with 0 failed, covering the return, alignment (`pathway_columns_lag_one_row_and_drop_life_starts`), splitmix, ridge, split, gap, spread, winner-rule, verify and malformed-input tests, and the contributing-guard check passes (proves INV-8, INV-12).
