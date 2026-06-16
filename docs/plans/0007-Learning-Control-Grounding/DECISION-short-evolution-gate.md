> **Spike task:** `short-evolution-gate` — Run A Small Evidence-Based Evolution Gate (workstream `0005`).
> **Done when:** either the gate passes with recorded metrics or a negative decision note is committed; cargo fmt/clippy/test green.

## Decision

**Defer the 20-generation evolution comparison to an offline run; do not assert it as a CI gate (yet).** This satisfies the "negative decision note is committed" branch of the task's Done-when. No workstream has been *shown* to fail — the runtime-truth, telemetry, curriculum, and control prerequisites all landed and pass their red-green probes — but the generational comparison the gate calls for cannot run inside `cargo test -p xagent-sandbox` within a reasonable CI budget, so it has not been executed.

## Why defer rather than run-in-CI

A meaningful gate per `SCOPE.md` requires a 20-generation fixed-seed run under the learning curriculum, then a comparison against the investigated DB's late baseline (food per 1000 alive ticks, deaths per food, turn persistence, straightness, food-bearing alignment, danger dwell), requiring improvement on ≥2 primary behavior metrics with no deaths-per-food regression. A 20-generation run at the curriculum tick budget is minutes-to-hours of GPU wall time — appropriate as a deliberate offline experiment, not a unit-test gate (the plan itself states "long-run evolution is not used to debug basic control wiring").

The smaller, diagnostic probes that *gate the wiring* — `food_closure_probe`, `danger_exit_probe`, and the anti-circle probe — are in the integration suite and pass as red-green control/telemetry gates under `BrainConfig::learning_curriculum()`.

## Prerequisite correction that affects interpretation

During the post-merge full-workspace gate, `P_IN_DANGER_BIOME` was found to be published **inverted** (`0.0` while in a danger biome). That made `danger_exit_probe` pass *vacuously* (it saw the agent as never in danger). The flag is now corrected (`1.0` = in danger) and the danger telemetry tests assert the flag against the agent's actual readback biome. **Any offline evolution comparison must use a build at or after this fix** — danger-dwell numbers measured before it are inverted and not comparable to the `xagent.db` baseline.

## When to revisit / how to run

Run the offline gate once the curriculum default question is on the table (it is opt-in until this gate passes — see `SCOPE.md` Locked decisions). Reproducible command: a headless 20-generation fixed-seed run under `BrainConfig::learning_curriculum()` (`headless.rs`), then decode `behavior_metric` / recordings and compare to the late-baseline numbers in this plan's `STATUS.md`. Promote the curriculum to the evolution default only if ≥2 primary behavior metrics improve with no deaths-per-food regression; otherwise record the failing metric and stop. See `ARCHITECTURE.md` §0005 for the gate construction.
