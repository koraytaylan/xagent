---
id: route-gate-decisions
title: "Route gate runs, decision records and adoption"
workstream: "0400"
kind: task
role: architect
depends_on:
  - kernel-routes
  - legacy-decode-and-birth-state
  - route-gate-tool
  - critic-input-decision
gated: false
status: todo
merged_as: ""
touches:
  - docs/decisions/pathway-critic-route.md
  - docs/decisions/pathway-critic-route.json
  - docs/decisions/pathway-forward-route.md
  - docs/decisions/pathway-forward-route.json
  - crates/xagent-shared/src/config.rs
  - crates/xagent-sandbox/tests/integration.rs
checks:
  - name: focused-proof
    command: "cargo run -p xagent-sandbox --bin pathway_route_gate -- verify docs/decisions/pathway-forward-route.json"
    timeout_ms: 3600000
  - name: critic-route-record
    command: "cargo run -p xagent-sandbox --bin pathway_route_gate -- verify docs/decisions/pathway-critic-route.json"
    timeout_ms: 3600000
  - name: route-state
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test pathway_route_state"
    timeout_ms: 3600000
  - name: pathway-routes
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test pathway_routes"
    timeout_ms: 3600000
  - name: routes-off-fingerprint
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test pathway_default_fingerprint"
    timeout_ms: 3600000
  - name: sandbox-integration
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration"
    timeout_ms: 7200000
  - name: shared-fmt
    command: "cargo fmt -p xagent-shared -p xagent-sandbox -- --check"
    fix: "cargo fmt -p xagent-shared -p xagent-sandbox"
---

# Route gate runs, decision records and adoption

## Outcome

Two committed decision records from paired GPU runs of the built binary, checked
by `pathway_route_gate verify`:

- `docs/decisions/pathway-forward-route.json` and `.md`: forward route
  (`forward_pathway_enabled`), gates G1 to G4, verdict.
- `docs/decisions/pathway-critic-route.json` and `.md`: critic route
  (`critic_input` = critic-input-decision winner), gates G2 to G4, verdict. If the
  winner is `encoding`, `not-applicable` writes verdict `not_applicable`; no runs.

A `pass` verdict flips its `DEFAULT_*` constant in
`crates/xagent-shared/src/config.rs` with a measured doc comment. Otherwise the
default stays off and config.rs is unchanged. A negative verdict with green
`verify` is correct. Thresholds, credit convention, seeds, agents and ticks are
fixed by D9 and route-gate-tool; do not tune them.

## Scope

- Binaries: `cargo run --release -p xagent-sandbox -- ...` (xagent) and
  `cargo run -p xagent-sandbox --bin pathway_route_gate -- ...` (gate tool).
- Free-life flags are copied verbatim from `crates/xagent-sandbox/src/main.rs`;
  on mismatch, stop and report it.
- `--commit "$(git rev-parse HEAD)"` is taken once in step 1 and reused. Never
  hand-edit tool-written JSON.
- Scratch is `target/pathway-route/` (git-ignored); never committed or cited.
- Background each long run, log to `$MAKINA_SHELL_JOB/output/`, poll
  `$MAKINA_SHELL_JOBS/*/state`; at most four GPU jobs; check `MemAvailable` first.
- 80-generation procedure: `docs/reviews/2026-09-25-learner-foraging-investigation.md`;
  G4 window is generations 60 to 78. `target/trunk-lesions/` is not a control.

## Steps

**Steps:**

1. Confirm the four dependencies are merged and
   `cargo run -p xagent-sandbox --bin pathway_return_fit -- verify docs/decisions/pathway-critic-inputs.json`
   passes. Read the critic `winner`. Record `git rev-parse HEAD` and the SHA-256
   of the `--dump-config` output. Dump the default config to
   `target/pathway-route/` and copy it into arm configs, editing only the two
   `brain` keys: control (`critic_input` = `encoding`, `forward_pathway_enabled`
   = `false`); forward (`encoding`, `true`); critic, only when the winner is not
   `encoding` (winner, `false`). Confirm with `diff` that each arm differs from
   control in its own key only.
2. Free lives. For seeds 5 to 8, run control and each arm as
   `cargo run --release -p xagent-sandbox -- --config <arm> --seed S --free-life --free-life-agents 10 --free-life-ticks 300000 --free-life-out target/pathway-route/free-life/<arm>-s<S>.json`,
   adding `--free-life-log target/pathway-route/free-life/<arm>-s<S>.log` for
   control and forward only. Each run must exit 0 and list 10 agents of 300,000
   ticks. Rerun a failed run once with the same seed; a second failure is a
   blocker. Then run `pathway_route_gate free-life` with the same `--commit`:
   forward (`--route forward --route-log <logs>`,
   `--out docs/decisions/pathway-forward-route.json`); critic when the winner is
   not `encoding` (`--route critic --critic-input <winner>`, no `--route-log`,
   `--out docs/decisions/pathway-critic-route.json`); critic when the winner is
   `encoding`, via `pathway_route_gate not-applicable --route critic --critic-input encoding --winner-record docs/decisions/pathway-critic-inputs.json`
   with the same commit and `--out docs/decisions/pathway-critic-route.json`.
3. Generations. For each route that passed G1 to G3 (G2 and G3 for critic),
   derive configs from the step 1 arms with `governor.tick_budget` 40000 in
   control and route alike. Run
   `cargo run --release -p xagent-sandbox -- --config <gen> --seed S --no-render --db target/pathway-route/gen/<arm>-s<S>.db --generations 80`
   for seeds 5 to 8 at the step 1 commit; control runs are shared and the step 2
   rerun rule applies. Then run `pathway_route_gate generations --record <json>`
   with `--control-db` and `--route-db` for seeds 5 to 8. Routes that failed
   G1 to G3 and the `not_applicable` critic get no generation runs; the tool
   records their G4.
4. Adoption. For each `pass` record, flip its constant in `config.rs`
   (`DEFAULT_FORWARD_PATHWAY_ENABLED = true` or
   `DEFAULT_CRITIC_INPUT = CriticInput::<winner>`) and set its doc comment to
   measured gate numbers only, with no plan, task or decision-document names.
   Leave other constants alone. If the flip changes a test in
   `crates/xagent-sandbox/tests/integration.rs` that pins the old default, pin
   that test's config to routes-off explicitly and keep its assertion; never
   weaken an assertion. `pathway_default_fingerprint.rs` must already pin both
   flags off, or stop and report a blocker. `tests/pathway_routes.rs` and
   `tests/pathway_route_state.rs` are outside `touches` and must not be edited;
   they set both flags explicitly (INV-14). If `pathway-routes` or `route-state`
   fails after a flip, INV-14 was not honoured: stop and report a blocker.
5. Records. Write `docs/decisions/pathway-forward-route.md` and, when a critic
   record exists, `docs/decisions/pathway-critic-route.md`; JSON stays
   tool-written. Each Markdown record holds: the question; setup (commit, config
   SHA-256 per arm, seeds, agents, ticks, exact commands); a per-gate table from
   JSON numbers (mean and SE; agents passing G1 of 40; seeds passing G2 of 4;
   meal and death differences with SE; per-agent values from the `agents` array);
   unconditional forward credit as information only; verdict; failing gate if
   negative; adoption decision. It states that the commit precedes the adoption
   edit, that the edit changes only the default constant, that logs and databases
   are not committed, and gives the regeneration commands.
6. Finish. Run `focused-proof`, `critic-route-record`, `route-state`,
   `pathway-routes`, `routes-off-fingerprint`, `sandbox-integration`,
   `shared-fmt` in order. On a `shared-fmt` failure apply its fix and rerun. In
   `routes-off-fingerprint` output, confirm a digest comparison against the
   recorded fixture ran and no skip naming both adapters was printed; a skip on
   this host is a blocker. Fix in-scope findings only in touched files; record
   other failures as pre-existing, with output.

- **Done when:** the focused-proof check exits 0 on the paired-run forward record, the critic-route-record check passes on the critic record, the routes-off-fingerprint check passes with a real digest comparison and both flags pinned off, and the pathway-routes check passes after any adoption, so each route default is on only if its gates all passed (proves INV-1, INV-11, INV-14).

## Non-goals

- Only the six `touches`; config.rs changes limited to the two `DEFAULT_*`
  constants and doc comments. No change to plan 0001 files, route-gate-tool,
  `Cargo.toml`, dependencies or tool-written JSON.
- No change to D9 thresholds, credit convention, seeds, agents, ticks or G4 window.
- No unconditional forward credit in any gate or verdict.
- No commit of logs, databases, config dumps or scratch outputs; no plan, task
  or decision-document names in source.
