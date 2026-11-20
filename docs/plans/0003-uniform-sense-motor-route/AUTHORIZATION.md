# Authorization record

## Corrected requirement: `headless-cli-help` action (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: the `study-route-witness` execution check action
  `headless-cli-help` ran `cargo run -p xagent-sandbox -- --help`.
- Contradiction: plan 0002 (landed in `e246a6e`) added the
  `pathway_return_fit` and `pathway_route_gate` binaries to `xagent-sandbox`,
  so `cargo run` without `--bin` exits 101 ("could not determine which binary
  to run").
- Correction: the action now runs
  `cargo run -p xagent-sandbox --bin xagent -- --help`, the same CLI the check
  always targeted (`[[bin]] name = "xagent"` in
  `crates/xagent-sandbox/Cargo.toml`). No bound, task, dependency or behavior
  changed.
- Witness: `cargo run -p xagent-sandbox --bin xagent -- --help` exits 0 and
  prints the CLI help; `validate_plan` reports ok.
