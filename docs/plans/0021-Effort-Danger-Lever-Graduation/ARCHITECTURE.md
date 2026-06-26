# Architecture — Plan 0021 (deltas)

> Edits in `crates/xagent-sandbox/src/headless.rs`,
> `crates/xagent-sandbox/src/main.rs`,
> `docs/plans/0021-Effort-Danger-Lever-Graduation/0001-EFFORT-FITNESS-DECISION.md`,
> `docs/plans/0021-Effort-Danger-Lever-Graduation/0002-DANGER-PERCEPT-DECISION.md`,
> `docs/plans/0013-Innate-Survival-Instincts/0013-INNATE-INSTINCT-DECISION.md`,
> and `docs/plans/STATUS.md`.
> Line numbers are hints; locate by symbol (grep for `validate_speed_decoupling`,
> `run_headless_with_flags`, `run_innate_instinct_ab`, `ValidationStats`,
> `print_validation_metrics`).

## 0001 — Effort-Fitness-Production-A-B

Today `validate_speed_decoupling` (`headless.rs:419-458`) constructs a paired,
seed-deterministic A/B: a baseline arm (all flags OFF) and an ON arm
(`effort_rebased_fitness=true`, `speed_cost_exponent=2.0`,
`danger_percept_enabled=false`), runs `num_generations`, and reports
`mean_ticks_alive`, `mean_fitness`, the Pearson correlation of evolved
`movement_speed` with fitness, and `danger_dwell_fraction` through
`print_validation_metrics` (print/markdown output only). The harness was exercised
once to validate Plan 0010 (`0010` STATUS: "speed-decoupling validation run
completed") but has no recorded production-scale run, no bootstrap precision, and
no machine-readable decision summary. Plan 0014 fixed the camper inversion (removed
the `ticks` factor, re-derived `FORAGING_ENERGY_TARGET`/`EXPLORATION_RATE_TARGET`,
added `competent > camper` guards) and the calibration replay now passes
(`governor.rs:135-151`), so the corrected lever is ready for a graduation decision.

Edits:

- **Parameterize population and generation count** so the harness runs at
  production scale instead of the local-test envelope. `validate_speed_decoupling`
  already threads `config.world.seed` and `config.governor.population_size` into
  `format_validation_markdown`; extend the signature to accept the production
  envelope (population 100, 50 generations as defaults) and leave the two
  `run_headless_with_flags` calls and the metric math untouched — instrument the
  envelope, not the A/B logic:

```rust
/// Run the paired baseline-vs-ON speed-decoupling A/B at a fixed envelope.
/// `population` and `num_generations` default to the production envelope
/// (100, 50); the core A/B (`run_headless_with_flags` baseline + ON) is
/// unchanged — only the world size and generation budget are now caller-set.
pub fn validate_speed_decoupling(config: FullConfig, population: u32, num_generations: u64);
```

- **Wrap the paired run in a bootstrap loop and emit a machine-readable summary.**
  Collect `N = 100` replicates — each a seeded but independent world at the fixed
  envelope — storing the four per-replicate metrics, then reduce each to a point
  estimate (replicate mean), a 95% CI (2.5th/97.5th percentile), and an effect size
  (`ON − baseline`). Emit JSON alongside the existing markdown so the decision doc
  can cite measured precision rather than a single point:

```rust
/// One metric's bootstrap summary over the N replicates: the point estimate,
/// its 95% CI bounds (2.5th/97.5th percentile), and the ON−baseline effect.
/// Recorded for ticks_alive, speed_correlation, fitness, and danger_dwell so the
/// flip/retire/defer rule is backed by precision, not point-estimate noise.
struct BootstrapMetric {
    point: f32,
    lower_ci: f32,
    upper_ci: f32,
    effect: f32,
}
```

- **Print the decision rule against the bootstrapped bounds** so the executor reads
  the verdict, not just the numbers. The exact thresholds (flip if baseline
  `speed_correlation` is strongly positive and ON is near zero, ticks_alive ON
  within band, danger_dwell ON retained; retire on any negative axis or a >10% ticks
  drop; defer otherwise) are the locked decision rule — see SCOPE (locked
  decisions). The harness only evaluates and prints them.

Properties that make this safe:
- `effort_rebased_fitness` is **default-off** in every preset; this workstream
  measures the gated lever and records a verdict — it flips no default and ships no
  new behavior. The actual flip, if the verdict is FLIP, is a follow-up config change
  out of this plan's scope.
- The A/B logic is the one Plan 0010 authored and used; only the envelope, the
  bootstrap wrapper, and the output are added, so the measured behavior is the same
  code path that hardening already exercised.
- The 95% CI requirement (locked decision: bootstrap `N = 100`) means the verdict is
  backed by measured precision, not a single-run point estimate that could be noise.

## 0002 — Danger-Percept-Production-A-B

Today the `danger_percept_enabled` flag gates three coupled mechanisms: (a) the
danger-feature encoding (two extra feature slots, `buffers.rs:274-277` and `:376`),
(b) avoidance-intent accumulation (`P_AVOIDANCE_*` counters in `kernel_tick.wgsl`),
and (c) avoidance potential-based reward shaping. `run_headless_with_flags`
(`headless.rs:628-664`) already accepts the flag and returns a `ValidationStats`
carrying `mean_ticks_alive`, `mean_fitness`, and the approach/avoidance intent
fractions. The hardened suite measures the percept in isolation —
`crates/xagent-brain/tests/intent_baseline_measurement.rs:24-43` records the avoidance distribution at default
config with danger ON, and `crates/xagent-brain/tests/danger_percept_ablation_ab.rs:11-96` runs a paired A/B
with/without the danger signal (both arms `danger_percept_enabled=true` so the
intent counters always run). What is missing is a full production A/B of
`danger_percept` OFF (baseline) vs ON, with bootstrap precision, and a recorded
verdict — the exact ask of the 2026-06-25 review F8 (`docs/reviews/2026-06-25-glm-52.md`).

Edits:

- **Wrap `run_headless_with_flags` in a bootstrap-CI A/B over the danger flag.**
  Call it twice per replicate — baseline (`danger_percept_enabled=false`, all other
  flags off) and ON (`danger_percept_enabled=true`, all other flags off) — across
  `N = 100` seeded-independent replicates at the production envelope (100 population,
  50 generations), reusing the `BootstrapMetric` reducer from 0001. The core A/B is
  unchanged; only the stats collection is wrapped.

- **Surface steering alignment in `ValidationStats` if absent.** The verdict needs
  the per-agent steering signal alongside survival and intent; if
  `mean_steering_alignment` is not already a `ValidationStats` field, add it and
  populate it in the readback loop from the same telemetry the intent fractions are
  derived from:

```rust
/// Mean per-agent steering alignment over the run: how well the chosen turn
/// tracks the danger/food gradient. Read back next to the intent fractions so
/// the danger A/B can report it as a deferral signal (the 0018 credit-path
/// bottleneck keeps it in the chance band [0.38, 0.62]).
mean_steering_alignment: f32,
```

- **Print the decision rule against the bootstrapped bounds.** Flip if avoidance
  intent ON clears baseline and survival holds; retire if intent shows no signal or
  survival regresses; defer if intent and survival pass but steering stays in the
  chance band. The exact thresholds are the locked decision rule — see SCOPE
  (locked decisions). Steering alignment is reported as a deferral signal, not a hard
  flip gate, honoring the 0018 finding that the credit path — not the percept — is
  the steering bottleneck.

Properties that make this safe:
- `danger_percept_enabled` is **default-off**; no shipped behavior changes until a
  FLIP verdict is recorded and a follow-up flips the preset. This workstream only
  measures and decides.
- The underlying percept, intent counters, and avoidance shaping were hardened by
  Plan 0010 and gate-tested by the 0015/0016 danger ablation; this adds only the
  production A/B envelope and the recorded verdict on top of code that already runs
  in the test suite.
- Steering alignment is captured as a "nice-to-have" deferral signal rather than a
  flip gate, so a chance-band steering result correctly reads as "intent measured,
  credit path blocks" — it cannot wrongly retire a lever whose intent and survival
  axes pass.

## 0003 — Innate-Instincts-Graduation-Status

Today `run_innate_instinct_ab` (`headless.rs:492-560`) implements the prove-or-kill
gate, and Plan 0013 recorded a FAIL verdict in
`0013-INNATE-INSTINCT-DECISION.md`: survival +0.19% (gate ≥ +10%), alignment 0.000
(gate ≥ 0.4, caveat: `danger_percept` OFF), food-per-death 0.37 (gate ≥ 2.0). The
decision doc lists revisit conditions (re-run with `danger_percept` ON, weaker
seeds, longer runs), but the decision's long-term status is unrecorded: is this a
permanent REJECT (the mechanism does not work), or is it deferred pending
credit-path improvement (a seeded prior cannot be acted upon through a chance-level
credit path)? The 0018 gradient-shaping result — 400× magnitude gain, no steering
change — argues for the latter: a static prior cannot overcome the credit bottleneck,
so revisiting the lever only makes sense once steering clears the chance band. This
workstream records which it is. It changes no code — only the decision doc and the
STATUS row.

Edits:

- **Mark the decision terminal (Variant A)** if the project commits to the
  blank-slate architecture and closes the seeded-prior avenue. Append a closing
  section to `0013-INNATE-INSTINCT-DECISION.md` stating the gate FAIL is permanent,
  the seeding code ships complete behind the default-off flag for future research
  reference, and future credit-path improvements do not automatically re-open the
  gate:

```markdown
## Terminal decision (recorded 2026-06-25, plan 0021)

The innate-instincts mechanism is complete, proven not to contribute under the
current credit path, and archived. The seeding code ships behind the default-off
`innate_instincts_enabled` flag for future research reference. The blank-slate
learning model is the committed baseline; this gate remains closed unless the
project explicitly decides to revisit seeded-prior research.
```

- **Mark the decision deferred on the credit path (Variant B)** if the project
  intends to revisit the lever once steering is above chance. Update the "When to
  revisit" section to name the explicit, falsifiable gate so a future planner can
  execute it cold:

```markdown
## Decision deferred (recorded 2026-06-25, plan 0021)

The mechanism is sound but the credit bottleneck prevents any prior from being
reliably acted upon. This gate re-opens when a subsequent plan achieves
steering_alignment >= 0.62 on production seeds with the default learning path; at
that point, re-run the A/B with `danger_percept_enabled` ON and
`innate_instincts_enabled` ON/OFF to measure the seeded-prior contribution to the
improved signal. Until the credit path clears the chance band, seeded priors
cannot demonstrate an independent effect.
```

- **Reconcile the root `docs/plans/STATUS.md` row 0013** with whichever variant
  lands: "Gate FAIL, flag stays default-off, decision TERMINAL. Seeding shipped
  complete." (Variant A) or "Gate FAIL, flag stays default-off, decision DEFERRED
  pending credit-path above 0.62." (Variant B), with the `Last updated` line bumped.

The choice between Variant A and Variant B is a research-direction call resolved in
SCOPE (locked decisions), not a code-quality judgment; this workstream records it
clearly enough that a future planner knows the precondition.

Properties that make this safe:
- `innate_instincts_enabled` is **default-off** and this workstream changes no code
  — only the decision-doc clarity and the STATUS marking; no runtime, shader, or
  harness path is touched.
- Whichever variant lands, the seeding mechanism remains shipped and the flag stays
  default-off, so the observable behavior is unchanged; only the recorded
  graduation status moves.

## Test strategy

Three measurement harnesses, no new unit tests; the A/Bs are measurement probes run
locally and recorded in decision docs, not red-green integration tests.

- **0001:** `validate_speed_decoupling` (`headless.rs:419-458`) wraps two
  `run_headless_with_flags` calls (baseline vs ON) in a bootstrap loop (`N = 100`
  replicates, 50 generations, population 100) and computes 95% CI on
  `mean_ticks_alive`, `speed_correlation`, `mean_fitness`, and
  `danger_dwell_fraction`, emitting JSON plus the printed flip/retire/defer rule.
  The verdict is recorded in `0001-EFFORT-FITNESS-DECISION.md`.
- **0002:** `run_headless_with_flags` (`headless.rs:628-664`), wrapped in the same
  bootstrap-CI loop with the `danger_percept_enabled` flag toggled, runs `N = 100`
  replicates and computes 95% CI on `mean_ticks_alive`, the approach/avoidance intent
  fractions, and `mean_steering_alignment`. The verdict is recorded in
  `0002-DANGER-PERCEPT-DECISION.md`, explicitly noting whether steering is in the
  chance band [0.38, 0.62].
- **0003:** documentation-only; no test, no measurement — a decision-doc and
  STATUS-row clarification.
- CI gate (every task): `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test -p xagent-sandbox` (GPU tests self-skip without an adapter; CI runs
  Mesa lavapipe). The A/B harnesses are measurement tools, not integration tests.

## Interaction with prior work

- **Honors the 0009/0010 speed-decoupling correctness.** The effort-fitness A/B uses
  the hardened `validate_speed_decoupling` harness 0010 authored and 0009 defined;
  the gate was scheduled for 0009 but explicitly deferred pending 0010 hardening, so
  running it to a decision is the natural follow-on.
- **Respects the 0014 math correctness.** The effort-fitness lever is now correct (no
  `ticks` factor, camper guards in place); this plan decides whether to ship it
  as-is or retire it, backed by production evidence — it does not re-derive the math.
- **Builds on the 0013 innate-instincts gate.** The prove-or-kill REJECT is recorded;
  this plan clarifies whether the decision is terminal or gated-on-credit-path,
  respecting the 2026-06-25 review's observation that a static prior cannot overcome
  the credit bottleneck.
- **Acknowledges the 0018 credit-path bottleneck.** The 0018 gradient-shaping result
  (400× magnitude gain, no steering change) means steering alignment stays in the
  chance band until the credit path is fixed. The danger-percept and effort-fitness
  A/Bs may show intent/survival wins, but steering remains a deferral signal here, not
  a flip gate.
