# Inheritance default

Newborn inheritance now defaults to weight-copy. A supervisor configuration
that omits `inheritance` gives weight-copy, and `darwinian` (in JSON or via
`--supervisor-inheritance darwinian`) is still selectable.

This overrides the 0005 REJECT verdict recorded in `DECISION.md`, and the 0005
rule that weight-copy is never adopted. To be exact, the user overrides the 0005 REJECT verdict
and the rule that "weight-copy is never adopted" no longer holds.

The reason is direction, not data: the supervisor becomes the only mode, and
generation mode is removed by a later plan.

This is the user's decision, not a new measurement. No new runs back it.

`DECISION.md` and `results.json` are kept unchanged as history.
