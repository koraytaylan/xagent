# XAgent Plan 0005 — GPU Occupancy & Brain-Pass Latency

Bank the measured ~10× evolution-throughput win by running at the GPU occupancy
knee (sweep harness → raise the default population → spend the capacity on unique
genomes, fixed-seed-validated); produce a definitive on-target per-cooperative-
pass cost profile of the fused brain pass (subgroup-path verification + a
measurement-only pass-limit knob); then, gated strictly on that profile, apply
one specific, correctness-justified, bit-identical latency reduction to the
dominant pass that either ships a measured speedup or records a measured
negative. The plan's value is banked by `0001`/`0002` regardless of `0003`.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md) for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/{plan_slug}--{id}/`).
- **Depends on** lists *direct* prerequisites only (`—` means none).
- **Done when** is the verifiable acceptance check. Every task must keep
  `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, and
  `cargo test -p xagent-sandbox` green ("cargo fmt/clippy/test green") unless it
  explicitly says it is documentation/measurement-only.
- GPU tests self-skip without an adapter (`GpuKernel::is_available()`); CI runs
  Mesa lavapipe. Throughput *numbers* require the target discrete GPU; harness
  *wiring* is verified on lavapipe.
- Line numbers are hints; locate every site by the named symbol (grep).

---

## 0001 — Occupancy throughput (the guaranteed win)

### occupancy-sweep-harness — Locate the GPU occupancy knee with a first-class tool

The knee was found with a shell loop over `--bench`; make it a reproducible,
single-command tool so the shipped population default is measured, not guessed.
`bench.rs` already has `run_bench` / `run_profile` / `run_phase_ab` and a
`create_kernel` helper; mirror them.

**Steps:**

1. In `crates/xagent-sandbox/src/bench.rs`, add `run_agent_sweep`:

   ```rust
   /// Sweep agent counts to locate the GPU occupancy knee. For each N: print
   /// tps and agent-ticks/sec (tps × N, the useful-work metric for evolution),
   /// and mark the N that maximizes agent-ticks/sec as the knee. Read-only.
   pub fn run_agent_sweep(
       brain: BrainConfig,
       world_config: WorldConfig,
       total_ticks: u64,
       counts: &[usize],
   );
   ```

   Each N: `create_kernel(&brain, &world_config, n)`, time
   `dispatch_batch(0, total_ticks as u32)` + `read_full_state_blocking()`, print
   `tps`, `tps * n` (agent-ticks/sec), and after the loop print the knee N
   (max agent-ticks/sec).
2. In `crates/xagent-sandbox/src/main.rs`, add a `--bench-agent-sweep` flag and
   handler mirroring `--bench-phase-ab`; default N list
   `[1, 4, 10, 50, 100, 200, 400, 1000]`, honoring `--bench-ticks` /
   `--bench-agents` (agents ignored for the sweep) / `--world-size`.
3. Build release, run `./target/release/xagent --bench-agent-sweep
   --bench-ticks 200000` on the target GPU, and record the table + identified
   knee in a dated subsection of
   `docs/superpowers/specs/2026-06-10-learning-baseline.md`.

- **Depends on:** —
- **Done when:** `--bench-agent-sweep` prints per-N tps + agent-ticks/sec and the
  knee; the on-target table is recorded in the baseline spec; default behavior is
  unchanged (new subcommand only); cargo fmt/clippy/test green.

### population-default-to-occupancy-knee — Ship the population sized to the knee

The default `population_size` is 10 (`config.rs:265` `GovernorConfig::default`,
`governor.rs:1579` and any preset constructors), deep in the GPU-idle zone. Raise
it to the knee from `occupancy-sweep-harness`.

**Steps:**

1. In `crates/xagent-shared/src/config.rs`, add `default_population_size()`
   returning the knee value (≈192; the largest N before agent-ticks/sec plateaus,
   rounded to a multiple of `default_eval_repeats()`), wire it as the
   `population_size` default in `GovernorConfig::default` with a doc-comment
   citing the sweep, and a `// safe max: <N>` note (the largest N that completes;
   N=1000 runs, N=5000 did not — pick a conservative documented ceiling).
2. Update any in-tree preset/config that hardcodes `population_size: 10` for the
   *default* run path (not the unit-test literals, which stay as-is). Confirm the
   generation-handoff kernel rebuild (`sim_runtime.rs`,
   `GpuKernel::new(next_agent_count, …)`) sizes buffers for the new N with no
   other change.
3. Run `cargo test -p xagent-sandbox` and `-p xagent-brain`; run a short headless
   evolution (`--no-render --generations 2`) at the new default to confirm it
   starts, sizes, and steps without error.

- **Depends on:** `occupancy-sweep-harness`
- **Done when:** the default `population_size` is the measured knee with a
  documented safe max; headless + GUI start and run at that N; governor unit
  tests (which use explicit `population_size` literals) stay green; cargo
  fmt/clippy/test green.

### governor-capacity-allocation-validation — Spend the capacity on unique genomes, fixed-seed-validated

With the population at ~200 and `eval_repeats = 2` (`config.rs:246-248`), the
governor yields `pop_size / repeats` ≈ 96 unique genomes (`governor.rs:942-944`)
— ~10× more search breadth. Lock this allocation (capacity → unique genomes,
`eval_repeats` unchanged) and prove evolution does not regress.

**Steps:**

1. In `crates/xagent-sandbox/src/governor.rs` at the `unique_count` computation
   (`:942-944`), add a doc-comment recording the locked decision: extra
   `population_size` is spent on unique genomes (search breadth); `eval_repeats`
   stays an independent noise-reduction knob. No formula change.
2. Run a fixed-seed headless evolution comparison (same `seed`, `tick_budget`,
   generations) at the new default population versus `population_size = 10`,
   following the existing fixed-seed-comparison pattern in
   `docs/superpowers/specs/2026-06-10-learning-baseline.md`. Record deaths-per-food
   and the fitness/champion trajectory for both arms in a dated subsection.
3. State the verdict: the larger population must be **no worse** on
   deaths-per-food / best-fitness at equal wall-clock-per-generation budget
   (expected: better, from broader search). If it regresses, record the cause and
   the chosen mitigation (e.g. raise `eval_repeats` in lockstep).

- **Depends on:** `population-default-to-occupancy-knee`
- **Done when:** the locked allocation is documented and the fixed-seed
  before/after evolution table + verdict are recorded showing no regression;
  cargo fmt/clippy/test green. (Measurement task beyond the green gate.)

---

## 0002 — Brain-pass cost profile (the guaranteed artifact)

### subgroup-topk-verification — Record whether the fast top-K path is active on target

`coop_recall_topk` (`brain_passes.wgsl:237`) is a 7-stage bitonic sort with a
barrier per stage/step (`:250-275`) in the workgroup-memory fallback; a
subgroup-accelerated path exists, gated on `wgpu::Features::SUBGROUP`
(`gpu_kernel.rs:463-466`, `apply_subgroup_markers`, `bitonic_sort_subgroup.wgsl`).
The kernel already stores `has_subgroup` "for runtime diagnostics" but never
surfaces it.

**Steps:**

1. In `crates/xagent-brain/src/gpu_kernel.rs`, add `pub fn has_subgroup(&self) ->
   bool` and a `log::info!` at kernel construction stating whether SUBGROUP is
   active (and therefore which top-K path compiled).
2. Build release, run any `--bench*` mode on the target with `RUST_LOG=info`, and
   record in the baseline spec whether the target Metal device reports SUBGROUP
   active or the bitonic fallback.

- **Depends on:** —
- **Done when:** the binary logs subgroup/top-K-path status at kernel init, the
  on-target value is recorded in the baseline spec; default behavior unchanged;
  cargo fmt/clippy/test green.

### per-cooperative-pass-limit-probe — Profile `brain_tick_inner` pass-by-pass

There is no intra-`brain_tick_inner` (`kernel_tick.wgsl:416-445`) cost breakdown.
Add a measurement-only limit that runs only the first `k` cooperative passes so
the cumulative cost curve (and per-pass deltas) can be measured. Default = all
passes ⇒ byte-identical.

**Steps:**

1. In `crates/xagent-brain/src/gpu_kernel.rs`, extend `DispatchProbe` with
   `kernel_pass_limit: u32` read from `XAGENT_KERNEL_PASS_LIMIT` (default `7`).
2. Reuse the kernel push constant's unused second word: rename
   `KernelPushConstants { start_tick, _pad }` → `{ start_tick, pass_limit }`
   (`kernel_tick.wgsl:486-490`), and in `dispatch_ticks` set both kernel-batch
   push constants to `[tick_cursor as u32, self.probe.kernel_pass_limit]` instead
   of `[…, 0u32]`. No `WORLD_CONFIG_SIZE` / uniform-slot change; the
   `physics_remainder` pass (different pipeline) is untouched.
3. In `kernel_tick.wgsl` `brain_tick_inner`, read `let limit = kpc.pass_limit;`
   and gate each cooperative pass body on `alive && (pass_index < limit)` (a
   workgroup-uniform predicate, since `limit` is a push constant and `alive` is
   the broadcast `s_alive`). **Keep every `workgroupBarrier()` / `storageBarrier()`
   unconditional** so barrier uniformity is preserved (the SAFETY block in
   `kernel_tick.wgsl`).
4. Build release; sweep `XAGENT_KERNEL_PASS_LIMIT = 0,1,2,3,4,5,6,7` with
   `--bench-phase-ab` (or `--bench`) on the target; record the cumulative-cost
   table and the per-pass deltas in the baseline spec, naming the dominant pass.

- **Depends on:** —
- **Done when:** with the knob unset the dispatch is byte-identical
  (`deterministic_across_batch_sizes` + `fused_dispatch_matches_split` green); the
  on-target per-pass cost table identifying the dominant cooperative pass is
  recorded; cargo fmt/clippy/test green.

---

## 0003 — Brain-pass latency reduction (gated, ship-or-record)

### dominant-pass-latency-reduction — Cut the dominant cooperative pass's latency, bit-identically (GATED)

**Gate:** start only after `subgroup-topk-verification` and
`per-cooperative-pass-limit-probe` have recorded, on target, which cooperative
pass dominates `brain_tick_inner` and whether the subgroup top-K path is active.
Apply the *one* transformation that matches the profile; do not guess.

Against the profile, apply the matching candidate from ARCHITECTURE §0003, each
with its written correctness argument and the bit-identical gate:

- **Top-K dominant + bitonic fallback active + SUBGROUP available** → force the
  subgroup top-K path (`apply_subgroup_markers`, `gpu_kernel.rs:463-466`); top-K
  output is the same K elements ⇒ bit-identical (or re-baseline the determinism
  tests if tie-break order legitimately differs, with a recorded rationale).
- **A specific `storageBarrier()` is provably redundant** (no thread reads
  another thread's storage-buffer write across it; e.g. a workgroup-memory-only
  handoff) → downgrade it to `workgroupBarrier()`, justifying it from the
  read/write set (`kernel_tick.wgsl:434/440/443`, `brain_passes.wgsl:461/792/820`).
- **Thread-0 physics/respawn serialization** (`kernel_tick.wgsl:516-519`,
  `:533-534`) → implement a bit-identical multi-thread form if one exists, else
  record the structural reason it is inherently serial (sequential sub-tick
  dependency).

**Steps:**

1. Read the `0002` profile; pick the single matching transformation above.
2. Implement it; keep `deterministic_across_batch_sizes` and
   `fused_dispatch_matches_split` green (or re-baseline with a recorded rationale
   for a justified reduction-order change only).
3. Measure on target: tps at `--bench-agents 10` (latency-bound regime) and at
   the occupancy knee. **Ship** if tps improves ≥10% at N=10 without regressing
   the knee; **otherwise revert** and record the measured negative.
4. Resolve the outcome into `0003-BRAIN-LATENCY-DECISION.md` (numbered to this
   workstream): the decision up front, the transformation tried, the on-target
   before/after, and — if reverted — the reason and the gate that would reopen it.

- **Depends on:** `subgroup-topk-verification`, `per-cooperative-pass-limit-probe`
- **Done when:** **either** a bit-identical (or justified-re-baselined)
  transformation is merged with a recorded ≥10%-at-N=10 on-target improvement and
  no knee regression, **or** `0003-BRAIN-LATENCY-DECISION.md` records the measured
  negative and the working tree is reverted clean; in both cases cargo
  fmt/clippy/test green.

---

**End of plan 0005 TASKS.** When every "Done when" bullet is green (and `0003` is
resolved one way or the other), the plan's end state is reached: a shipped ~10×
agent-ticks/sec occupancy win, a recorded per-pass brain profile, and either a
shipped brain-pass speedup or a recorded measured negative — a concrete result in
every branch.
