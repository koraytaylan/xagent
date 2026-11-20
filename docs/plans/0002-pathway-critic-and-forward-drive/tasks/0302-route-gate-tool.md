---
id: route-gate-tool
title: "Route-gate tool"
workstream: "0300"
kind: task
role: developer
depends_on:
  - route-flags
gated: false
status: done
merged_as: "217b3de76502c383edd8dabcbefb3f71e7236b53"
touches:
  - crates/xagent-sandbox/src/bin/pathway_route_gate.rs
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --bin pathway_route_gate"
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

# Route-gate tool

## Outcome

A new binary `crates/xagent-sandbox/src/bin/pathway_route_gate.rs`, found by
Cargo auto-discovery (no `autobins` key; no `Cargo.toml` edit, no new
dependency). Gate method and thresholds follow ARCHITECTURE D9 and INV-9.
No file outside `touches` changes.

Prerequisite: plan 0001 is integrated and `route-flags` is done.

Subcommands:

- `free-life --route critic|forward --critic-input encoding|pathway|encoding_and_pathway --commit <rev> --control-metrics <path>... --route-metrics <path>... [--route-log <path>...] --out <json>`
  writes the record with per-agent and pooled numbers, conditional credit `C`,
  unconditional credit `U` (information only) and G1-G3. `--critic-input` is
  required for `--route critic`, rejected otherwise, and recorded, not inferred.
  `--route-log` is required for `--route forward` and rejected for `critic`.
- `not-applicable --route critic --critic-input encoding --winner-record <path> --commit <rev> --out <json>`
  reads the 0401 winner record (top-level string `winner`). It requires
  `winner == "encoding"`; any other winner, a missing field or an unreadable
  file is an error naming the file and writes nothing. Otherwise it writes
  `verdict` and G1-G4 all `"not_applicable"`, `failed_gate` null,
  `setting_under_test` `"encoding"`, empty `seeds`, `agents`, `per_seed`,
  `agents_per_seed` 0, and `ticks_per_seed`, `config`, `pooled` null.
- `generations --record <json> --control-db <path>... --route-db <path>...`
  adds G4 per seed and the final verdict, rewriting the record.
- `verify <json>` recomputes every number, gate and verdict, and checks the
  record's default against `DEFAULT_CRITIC_INPUT` and
  `DEFAULT_FORWARD_PATHWAY_ENABLED` (production passes the shipped constants).
  It exits non-zero with a message naming the field on any disagreement.

`--commit` is required for every record-writing subcommand; it is recorded
verbatim and only an empty string is rejected. The binary never runs an
external program.

## Decisions

- Conditional credit uses the logged TD error column (`td_error`, the kernel's
  clamped `s_pred_td[S_TD_ERROR]`). It is never rebuilt from reward and value,
  and there is no `MAX_TD_ERROR` constant.
- Credit conventions (D9 leaves them open): `e^-` is read before the tick's
  update and is zero at each life start; the trace is zeroed after a death
  tick; `g_k` is the life mean over all `T` ticks; normalisation is per sense
  (smell `k` in 0..2, vision `k` in 2..10). `free_life_forward_credit[10]` is
  copied verbatim (length 10, finite) and never recomputed.
- Named constants with doc comments (no bare literals beyond 0, 1, -1), as D9
  and `common.wgsl` fix them: `TD_DISCOUNT`, `TD_LAMBDA`, `TRACE_DECAY`, seed and
  agent counts, the G1-G4 thresholds, and the input widths.
- Plan 0001 readers and predicates: call them through `xagent_sandbox::` if
  public; otherwise re-implement the format parsing and predicates in this
  binary (no edits to 0001 files). The module doc records, without naming any
  plan, task or decision, which path was used and the exact header field names.
- Record structs use `serde(deny_unknown_fields)`. No `unwrap`, `expect` or
  panic on I/O, parsing or database access; no truncating `as` casts.
- A contradiction with the integrated 0001 formats is reported, not guessed.

## Tests

Named for what they falsify; use named constants for counts and seeds.

- Credit rebuild (INV-9): 1 agent, 4 ticks, `x = (1, 0)`, `s = 1.0`,
  `δ = [0.0, 0.685, -0.4, 0.3]`, `d = [0, 0, 1, 1]`. Expect `e^- = [0.0, 0.5, 0.9365, 0.0]`
  and `g_0 = -0.008025` within `1e-12`. A second test drops the tick-2 death flag
  and asserts `g_0` changes.
- Trace normalisation (INV-9): smell `x = (2, 0)`, `s = 1` gives increment
  `0.4` on tick 0; visual `(0,0,0,0,0,0,0,3)` gives `0.3` for input 9 and leaves
  the smell trace unchanged.
- Conditional credit sign (INV-9): `δ` positive exactly where food is ahead and
  centred, with positive trace, gives `C > 0`; reversed `δ` gives `C < 0`; both finite.
- Copied credit (INV-9): `free_life_forward_credit` equals the metrics values
  element for element; changing one metrics value changes only that field, not `c`.
- G1 boundary (INV-9): 20 of 40 positive with pooled mean `> 0` fails; 21 passes;
  pooled mean `≤ 0` fails even with 40 positive agents.
- G2 (INV-9): pooled share higher with 2 of 4 seeds higher fails; 3 of 4 passes;
  pooled share equal to control fails.
- G3 boundary (INV-9): meals difference exactly `−SE` (within `1e-9`) passes and
  `−SE − 1e-6` fails; deaths `+SE` passes and `+SE + 1e-6` fails.
- G4 (INV-9): in-memory SQLite with the `node(id, run_id, generation, ...)` and
  `agent_result(id, node_id, agent_index, ..., composite_fitness REAL)` columns
  from `governor.rs` `init_schema`, generations 60..=78 for both arms. Three seeds
  higher and one lower fails and names that seed; all four higher passes; a
  missing generation 71 is an error naming the file.
- Not-applicable (INV-9): winner `encoding` writes the empty-arm record; winner
  `pathway` and a file without `winner` are rejected, naming the file, writing
  nothing. The record verifies with `critic_input = encoding` and fails with `pathway`.
- Verify (INV-9): a record from the tool's own writer verifies. Then, on the same
  in-memory record: edited `verdict` rejected; `pooled` mean changed beyond `1e-9`
  relative rejected; wrong `failed_gate` rejected; forward `verdict = "pass"` rejected
  with `forward_pathway_enabled = false` and accepted with `true`; critic winner
  `pathway`, `verdict = "pass"` rejected with `encoding`, accepted with `pathway`;
  critic `"not_applicable"` rejected with `pathway`, accepted with `encoding`;
  empty `commit` rejected. Tests pass a `CodeDefaults` value to `verify_record`.
- Malformed input (INV-9): `NaN` metrics row, 3 seeds, 39 route agents, and a
  route log missing an agent each return an error naming the file, without panic.

## Steps

**Steps:**

1. Read the integrated 0001 metrics writer, tick-log reader and critic gap
   predicates, the header names for `td_error`, executed forward noise, death
   and ground-truth food, and `CriticInput`/`DEFAULT_*` from route-flags. Choose
   import or re-implement per Decisions and record it in the module doc.
2. Create the binary: module doc, constants, clap skeleton for the four
   subcommands, `deny_unknown_fields` record structs, per-agent credit rebuild
   (`g`, `m`, `C`, `U`, copied credit). Add the credit, normalisation, sign and
   copied-credit tests.
3. Add arm validation (seeds, counts, ticks, finite values), G1, G2, G3 with
   pooled and per-seed numbers, and `free-life`, which sets `verdict` to
   `"pending"` unless G1-G3 already fail. Add the G1, G2, G3 and malformed-input tests.
4. Add G4 over `&Connection` and the `generations` subcommand, which opens each
   database read-only with `busy_timeout` and finishes the record. Add the G4 tests.
5. Add `verify_record(record, CodeDefaults)` and the `verify` subcommand, plus
   `not-applicable` with its winner-record check. Add those tests on in-memory records.
6. Run the checks in order: `focused-proof`, `sandbox-fmt` (apply its fix, rerun),
   `sandbox-clippy`, `contributing-guard`. Fix findings only inside the touched file.

- **Done when:** `cargo test -p xagent-sandbox --bin pathway_route_gate` passes the credit-rebuild, G1-G4 boundary, not-applicable and verify tests, and the contributing-guard check passes (proves INV-9, INV-12).

