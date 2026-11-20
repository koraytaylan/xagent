# Critic route gate decision

## Question

Should the critic input route (`brain.critic_input`) default to the winner of
the critic input decision (`docs/decisions/pathway-critic-inputs.json`)? The
route would be tested on gates G2 to G4 against a control arm with both routes
off.

## Setup

- Commit: `fd7419c7c0c4cc4c4fba4ad7026e1fe0458eccbf`, the output of
  `git rev-parse HEAD` taken once and passed as `--commit`. The commit precedes
  any adoption edit; none was made.
- The critic input decision winner is `encoding`, which is the current default
  critic input. There is no route arm to run, so no config arm, seed, agent or
  tick count applies (the record holds empty `seeds` and `agents`, and null
  `config` and `pooled`). Default config dump SHA-256:
  `78c12cf9477828b8a5f357f140ac0ae4a23adaebe359fe631765aa7940f00b4a`.

Commands (repository root):

```sh
COMMIT=$(git rev-parse HEAD)
cargo run -p xagent-sandbox --bin pathway_return_fit -- verify docs/decisions/pathway-critic-inputs.json
cargo run -p xagent-sandbox --bin pathway_route_gate -- not-applicable --route critic \
  --critic-input encoding --winner-record docs/decisions/pathway-critic-inputs.json \
  --commit "$COMMIT" --out docs/decisions/pathway-critic-route.json
cargo run -p xagent-sandbox --bin pathway_route_gate -- verify docs/decisions/pathway-critic-route.json
```

These commands regenerate the record. The JSON record is tool-written and not
hand-edited; no logs, databases or config dumps exist for this route, and none
are committed.

## Gates

| Gate | Result |
|---|---|
| G1 | not_applicable |
| G2 | not_applicable |
| G3 | not_applicable |
| G4 | not_applicable |

No meal, death, sighting or generation numbers exist: no runs were made.

## Verdict

`not_applicable`. No gate failed; no gate was evaluated.

## Adoption

Not adopted. `DEFAULT_CRITIC_INPUT` stays `CriticInput::Encoding` and
`crates/xagent-shared/src/config.rs` is unchanged; an adoption edit would
change only that default constant and its doc comment.
