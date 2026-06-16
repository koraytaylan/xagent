# Architecture — Plan 0005 (deltas)

> Edits center on `crates/xagent-sandbox/src/bench.rs`,
> `crates/xagent-sandbox/src/main.rs`,
> `crates/xagent-shared/src/config.rs`,
> `crates/xagent-sandbox/src/governor.rs`,
> `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`, and the integration
> tests in `crates/xagent-sandbox/tests/integration.rs`. Line numbers are hints;
> locate by symbol.

## 0001 — Occupancy throughput

Today the evolution runs `population_size = 10` agents per generation
(`config.rs:265`, `governor.rs:1579`), dispatched as one 256-thread workgroup
each (`kernel_tick.wgsl:492`). The on-target sweep shows ticks/sec is flat to
N≈50 and the occupancy knee is N≈200, so at N=10 the GPU is ~90% idle and useful
throughput (agent-ticks/sec) is ~1/10th of what the same hardware sustains. The
governor turns `population_size` into `pop_size / eval_repeats` unique genomes
(`governor.rs:942-944`), so at the defaults only 5 unique genomes are evaluated
per generation.

### Occupancy sweep harness

Today `bench.rs` has `run_bench` (single N) and `run_profile`/`run_phase_ab`, but
no agent-count sweep; the knee was found by a shell loop. Make it a first-class,
reproducible tool.

Edits:

- **`run_agent_sweep`** (`bench.rs`, new): run a fixed `total_ticks` for each N in
  a default list, print tps, agent-ticks/sec (`tps × N`), and flag the N that
  maximizes agent-ticks/sec as the knee. Reuses `create_kernel` and the single
  fused `dispatch_batch(0, total_ticks)` path.

```rust
/// Sweep agent counts to locate the GPU occupancy knee. Prints, per N: tps and
/// agent-ticks/sec (tps × N — the useful-work metric for evolution), and marks
/// the N that maximizes agent-ticks/sec. Read-only measurement; ships no
/// behavior change.
pub fn run_agent_sweep(brain: BrainConfig, world: WorldConfig, ticks: u64, counts: &[usize]);
```

- **`--bench-agent-sweep` flag** (`main.rs`): mirror the `--bench-phase-ab`
  handler; default N list `[1, 4, 10, 50, 100, 200, 400, 1000]`, honoring
  `--bench-ticks`.

### Population default sized to the knee

Edits:

- **Raise the default `population_size`** (`config.rs:265` `GovernorConfig::default`,
  and the preset constructors `governor.rs:1579` / any `easy`/`normal`/`hard`
  preset) to the sweep-reported knee (≈200 on the reference GPU), with a
  doc-comment citing the sweep and a documented safe maximum. The kernel already
  rebuilds for the new agent count via `GpuKernel::new(next_agent_count, …)` on a
  population change (`sim_runtime.rs`, generation-handoff rebuild path), so no
  buffer-plumbing change is required — only confirm memory scales and record the
  practical ceiling (N=1000 runs; N=5000 did not complete).

```rust
/// Default population sized to the GPU occupancy knee (≈200 on the reference
/// GPU): below it the GPU is idle, above it agent-ticks/sec plateaus while each
/// generation's wall time grows. See docs/reviews/2026-06-15-brain-pass-latency-ceiling.md.
fn default_population_size() -> usize { 192 }
```

### Governor capacity allocation

Today `unique_count = pop_size / eval_repeats` (`governor.rs:944`). With the
population at ~200 and `eval_repeats = 2`, that is ~96 unique genomes — a large
exploration increase. The decision to lock here: **the extra capacity goes to
unique genomes, holding `eval_repeats` at its current default**, so a generation
evaluates ~10× more distinct configs (the throughput is spent on search breadth,
which is the evolutionary win). `eval_repeats` stays a separate, independently
tunable noise-reduction knob.

Edits:

- No formula change is required — leaving `eval_repeats` fixed while
  `population_size` grows already routes the capacity to `unique_count`. The task
  makes this explicit (doc-comment at `governor.rs:942-944`) and **validates it
  on a fixed seed**: a headless evolution run at the new default versus N=10 on
  the same seed, asserting the fitness trajectory / deaths-per-food is no worse
  (expected: better, from broader search), recorded in the baseline spec.

Properties that make this safe:
- `population_size` is already a first-class config consumed at one site
  (`governor.rs:942`) and drives kernel sizing through the existing rebuild path;
  raising the default exercises only already-tested code with a larger N.
- Governor unit tests construct `GovernorConfig { population_size: N, .. }`
  explicitly (`governor.rs` tests), so they are unaffected by the default change;
  the fixed-seed validation is the new guard.

## 0002 — Brain-pass cost profile

Today the only intra-kernel timing is whole-pass (`XAGENT_SKIP_GLOBAL` /
`XAGENT_SKIP_VISION`); there is no breakdown *within* `brain_tick_inner`
(`kernel_tick.wgsl:416-445`), and the active top-K path (subgroup vs bitonic
fallback) is unverified on target.

### Subgroup top-K verification

Edits:

- **Expose `has_subgroup`** (`gpu_kernel.rs` — the field already exists, "retained
  for runtime diagnostics"): log it once at kernel construction and add a
  `pub fn has_subgroup(&self) -> bool` accessor. Record the on-target value in
  the baseline spec. This is a one-fact deliverable: subgroup top-K active or the
  barrier-heavy fallback active.

### Per-cooperative-pass limit probe

Today `brain_tick_inner` runs all seven cooperative passes unconditionally (under
the `alive` guard). Add a measurement-only limit so cumulative cost can be
measured pass-by-pass.

Edits:

- **`kernel_pass_limit` knob** (`gpu_kernel.rs` `DispatchProbe`, env
  `XAGENT_KERNEL_PASS_LIMIT`, default `7` = all seven cooperative passes): plumb
  it to the shader through the **already-present, currently-unused second word of
  the kernel push constant** (`KernelPushConstants { start_tick, _pad }`,
  `kernel_tick.wgsl:486-490`, set up by Plan 0003). Rename `_pad → pass_limit`
  and set it in `dispatch_ticks` alongside `start_tick`
  (`set_push_constants(0, &[tick_cursor as u32, pass_limit])`) for both the full
  and remainder kernel batches. No uniform-slot or `WORLD_CONFIG_SIZE` change; the
  `physics_remainder` pass uses a different pipeline and is untouched.
- **Guard the passes** (`kernel_tick.wgsl` `brain_tick_inner`): gate each
  cooperative pass on a *workgroup-uniform* limit so barrier uniformity is
  preserved (the limit is a push constant — uniform across the workgroup; `alive`
  is already broadcast-uniform via `s_alive`, so `alive && (pass_index < limit)`
  is uniform):

```wgsl
// Measurement-only: run only the first `limit` cooperative passes so the
// cumulative GPU cost of brain_tick_inner can be profiled pass-by-pass. `limit`
// comes from the kernel push constant (uniform), so every gated pass + its
// barriers are reached uniformly by all 256 threads. Default 7 (all passes) =>
// byte-identical results to today (the determinism tests gate this).
let limit = kpc.pass_limit;
if (alive && 0u < limit) { coop_feature_extract(agent_id, tid); }
workgroupBarrier();
// … one guard per pass (pass_index < limit), barriers always executed …
```

Running `--bench-phase-ab`-style across `XAGENT_KERNEL_PASS_LIMIT = 0..7` yields
the cumulative cost curve; consecutive deltas are the per-pass costs. The
dominant pass (top-K is the prime suspect) is recorded in the baseline spec.

Properties that make this safe:
- Default `7` runs all passes exactly as today; with the knob unset the dispatch
  produces byte-identical results and all tests pass untouched (the guard is a
  uniform comparison, not a result change).
- The guard predicate is workgroup-uniform, so the barrier-uniformity invariant
  (`kernel_tick.wgsl` SAFETY block) is preserved; skipped passes skip *with* all
  threads, never some.
- Producing wrong results while set is acceptable and documented — it is never on
  in tests or release.

## 0003 — Brain-pass latency reductions (gated, ship-or-record)

**Gated** on the `0002` profile: only the cooperative pass measured as dominant
is touched. Each task below is one specific, correctness-justified, bit-identical
transformation; it ships if it beats a stated threshold on target, otherwise its
measured negative is recorded in `0003-BRAIN-LATENCY-DECISION.md`. The plan's
concrete value is already banked by `0001`/`0002`, so any negative here is a
recorded result, not a dead end.

Candidate transformations (the profile decides which execute):

- **Force the subgroup top-K path** (if `0002` shows the bitonic fallback active
  and `wgpu::Features::SUBGROUP` is available on target): ensure
  `apply_subgroup_markers` splices the subgroup builtins and the kernel requests
  SUBGROUP (`gpu_kernel.rs:463-466`). Top-K output is identical (same K elements,
  same tie-break) → bit-identical; if the subgroup sort tie-breaks differently,
  re-baseline the determinism tests with a recorded rationale.
- **Prune provably-redundant storage barriers** in `brain_tick_inner`
  (`kernel_tick.wgsl:416-445`) and the cooperative passes: audit each
  `storageBarrier(); workgroupBarrier();` (`:434`, `:440`, `:443`,
  `brain_passes.wgsl:461`, `:792`, `:820`) against the actual cross-thread
  read/write set; downgrade a `storageBarrier()` to a bare `workgroupBarrier()`
  only where no thread reads another thread's *storage-buffer* write across that
  point (workgroup-memory-only handoffs do not need a storage barrier). Each
  removal is justified in the task and gated bit-identical.
- **Reduce thread-0 serialization** (`kernel_tick.wgsl:516-519`, `:533-534`):
  audit whether the thread-0-only physics sub-tick loop and death/respawn admit a
  bit-identical multi-thread form. Physics sub-ticks are sequentially dependent,
  so this likely records a structural negative (concrete: *why* it is serial) —
  acceptable under the ship-or-record rule.

Multi-agent-per-workgroup packing (raising occupancy density at fixed N) is noted
as the larger redesign that would follow if barrier pruning alone is
insufficient; it is **not** built here (SCOPE, Out of scope).

## Test strategy

- **`deterministic_across_batch_sizes`** and **`fused_dispatch_matches_split`**
  (`integration.rs`) are the bit-identical gate for every `0002`/`0003` shader
  change; they must stay green (or be explicitly re-baselined for a justified
  reduction-order change in `0003`).
- **Fixed-seed evolution comparison** (`0001`): a headless run at the new default
  population versus N=10 on the same seed, deaths-per-food / fitness trajectory
  no worse, recorded in `docs/superpowers/specs/2026-06-10-learning-baseline.md`.
- **Probe defaults off**: `XAGENT_KERNEL_PASS_LIMIT` unset ⇒ byte-identical
  dispatch; existing tests pass untouched.
- CI gate that must stay green:

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p xagent-sandbox
```

## Interaction with prior work

- **Builds on the 2026-06-15 brain-pass-latency review.** This plan executes that
  review's two levers (occupancy, per-workgroup barrier depth) with the
  measure-before-build discipline it recommends.
- **Honors Plan 0003.** Reuses its probe tooling (`DispatchProbe`,
  `--bench-phase-ab`, `[BENCH-PROBE]`), keeps the fused single-submit path, and
  does not re-open the rejected `global`-pass parallelization (`0004`).
- **Honors Plan 0002.** The worker boundary and decoupled publication are
  untouched; the population change rides the existing generation-handoff kernel
  rebuild.
- **Honors Plan 0001.** `vision_stride` / `brain_tick_stride` / sensory-lag-100
  stay fixed; no `TD_DISCOUNT` recalibration is implied.
