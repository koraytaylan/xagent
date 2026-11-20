# Sense route wiring

Route admission for the declared sense table `DECLARED_SENSES` in
`crates/xagent-brain/src/senses.rs`, derived from the pathway route records
and the shipped defaults. `senses::tests::declared_routes_match_route_wiring`
asserts the table below.

## Records

| Record | Verdict | Adopted |
|---|---|---|
| `docs/decisions/pathway-forward-route.md` / `.json` | `fail` (failed gate `G4`); `.md` and `.json` agree | no |
| `docs/decisions/pathway-critic-route.md` / `.json` | `not_applicable` (setting under test `encoding`); `.md` and `.json` agree | no |
| `docs/decisions/pathway-critic-inputs.md` / `.json` | winner `encoding`; `.md` and `.json` agree | n/a |

## Defaults (`crates/xagent-shared/src/config.rs`)

- `DEFAULT_FORWARD_PATHWAY_ENABLED` = `false`
- `DEFAULT_CRITIC_INPUT` = `CriticInput::Encoding`

## Admission

- Turn: admitted on both senses (always), `Mutated`.
- Forward: not admitted (verdict not adopted and default is `false`).
- Critic: not admitted (verdict not adopted, default is `Encoding`); the
  critic-inputs winner `encoding` equals the default, so there is no
  record/default disagreement and the gate is not blocked.

## Routes per sense

| Sense | Turn | Forward | Critic |
|---|---|---|---|
| smell | `Mutated` | not declared | not declared |
| vision | `Mutated` | not declared | not declared |
