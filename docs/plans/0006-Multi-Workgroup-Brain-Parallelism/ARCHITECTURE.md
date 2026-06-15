# Architecture - Plan 0006 (deltas)

> Edits center on `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-brain/src/buffers.rs`,
> `crates/xagent-brain/src/shaders/kernel/common.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`,
> new split-brain WGSL files under `crates/xagent-brain/src/shaders/kernel/`,
> `crates/xagent-sandbox/src/bench.rs`, and
> `crates/xagent-sandbox/tests/integration.rs`. Line numbers are hints; locate
> by symbol.

## Cost model - what must be eating the time

At the default 8x6 vision layout, `FEATURE_COUNT = 265`,
`ENCODED_DIMENSION = 128`, `PREDICTOR_DIMENSION = 128`, `MEMORY_CAP = 128`, and
`RECALL_K = 16` (`common.wgsl:28-32`, `buffers.rs:612-617`). One raw
kernel-batch is 100 simulation ticks: 10 brain cycles (`vision_stride`) x 10
physics ticks (`brain_tick_stride`). At N=10, the current fused kernel exposes
only 10 long-running workgroups per brain cycle (`dispatch_workgroups(agent_count,
1, 1)`, `gpu_kernel.rs:1568`), so the GPU sees little independent work while
each workgroup runs the following dense loops.

| Region | Current loop shape | Work per agent brain-cycle | Measured clue | Parallel axis |
|---|---|---:|---|---|
| `coop_encode` | 128 output threads, each loops 265 features (`brain_passes.wgsl:129-136`) | 33,920 MADs + 128 tanh | Clean +1.95s in Plan 0005 pass-limit sweep | Split output rows across tiles and reduce each row across input lanes |
| predictor train+predict | 128 output threads, each loops 128 inputs twice (`brain_passes.wgsl:316-332`) | 16,384 weight updates + 16,384 MADs | Inside `predict_and_act`, one of the dominant measured passes | Split predictor rows across tiles and reduce each row across input lanes |
| encoder-credit learning | 128 output threads, each loops 265 features when credit is active (`brain_passes.wgsl:753-761`) | up to 33,920 weight updates | Inside `learn_and_store`, the largest measured pass | Split encoded rows across tiles; each workgroup owns disjoint weights |
| memory reinforcement | 128 pattern threads, each loops 128 encoded dims (`brain_passes.wgsl:767-789`) | 16,384 MADs + norms | Inside `learn_and_store`, secondary | Optional pattern x dimension tiling if final gate misses |
| recall score/top-K | 128 pattern dot products + bitonic sort (`brain_passes.wgsl:202-294`) | about 32k dot/norm ops + 28 barriers | Only about 1.6% for top-K; not first | Leave alone unless later profile contradicts |
| thread-0 action tail | serial reductions/copies over 128 dims plus scalar policy (`brain_passes.wgsl:349-695`) | thousands, not tens of thousands | Confounded inside `predict_and_act` | Move per-dim copies/transforms to 128 lanes; keep scalar reductions ordered |
| fused food detect | each agent workgroup scans every food item every brain cycle (`kernel_tick.wgsl:201-277`) | `agent_count * food_count * vision_stride` distance checks per batch | Included in pass-limit-0 floor | Use 3x3 food-grid neighborhood, same as `phase_food_detect.wgsl:19-66` |

The first three rows are the breakthrough target. The next two rows close the
remaining brain-side tail if the first split misses. The food row is the
non-brain floor recovery path. Together, these are the concrete work items that
can own a 2.6x gap; the already-measured `global`/`vision` passes cannot.

The mechanical parallelization is simple: dispatch `agent_count x row_tile_count`
workgroups, where each tile owns a disjoint group of output rows and 16-32 lanes
cooperate on the row's inner dimension. This turns a serial `for feature/input in
...` loop inside one thread into a short lane-local loop plus a fixed reduction.

This is why the plan emphasizes storage-backed split phases: a second workgroup
cannot consume `s_encoded` or `s_prediction` from the first workgroup, so the
large intermediates must move from `var<workgroup>` scratch into
`brain_scratch`, with dispatch boundaries between producer and consumer phases.

## 0001 - Throughput budget and no-go map

Today the repo has the probes needed for the first decision:
`run_bench` dispatches a fixed tick count through `dispatch_batch`
(`bench.rs:26-48`), `XAGENT_KERNEL_PASS_LIMIT` limits cooperative brain passes
(`gpu_kernel.rs:210-218`, `kernel_tick.wgsl:427-460`), and Plan 0005 already
recorded a pass profile at N=200. The missing number is the N=10 no-brain floor:
whether physics+food+death+vision alone can comfortably exceed 60 k raw tps.

Edits:

- **Budget table only** (`docs/superpowers/specs/2026-06-10-learning-baseline.md`):
  record N=10 full, pass-limit 0, pass-limit 2, pass-limit 5, and pass-limit 7
  using the existing release binary. No code change is required for this task.

Properties that make this safe:
- It runs existing default-off knobs only. It cannot perturb shipping behavior.
- The stop/continue rule is numerical and final: pass-limit 0 below 90 k tps
  means this plan cannot hit 60 k by parallelizing brain math alone.

## 0002 - Same-dispatch cooperative tiling

Today `coop_encode` uses 128 active threads, one output dimension per thread,
and each thread serially loops over all features (`brain_passes.wgsl:129-136`).
`coop_predict_and_act` does the same row-per-thread pattern for predictor
training and prediction (`brain_passes.wgsl:316-332`) and then returns to
thread-0 serial loops for context blend, value/policy reductions, and
publishing (`brain_passes.wgsl:349-695`). `coop_learn_and_store` repeats the
feature loop for encoder-credit weight updates (`brain_passes.wgsl:753-761`).

### In-workgroup dense tiling

Edits:

- **Dense tiling constants** (`brain_passes.wgsl`): add named constants near
  `BRAIN_WORKGROUP_SIZE`.

```wgsl
/// Output rows handled per cooperative dense tile. 64 rows x 4 lanes = 256
/// invocations, using the whole workgroup while preserving one workgroup/agent.
const DENSE_OUTPUT_TILE: u32 = 64u;

/// Number of lanes that cooperatively reduce one dot product in the
/// same-dispatch tiling path.
const DENSE_INNER_LANES: u32 = 4u;
```

- **Dense partial scratch** (`brain_passes.wgsl`): add one 256-float workgroup
  array reused by encode and predictor tiling.

```wgsl
/// Reused cooperative dense-dot scratch. Indexed by local invocation id; each
/// group of DENSE_INNER_LANES entries reduces one output row.
var<workgroup> s_dense_partials: array<f32, BRAIN_WORKGROUP_SIZE>;
```

- **`coop_encode` tiling** (`brain_passes.wgsl:129-136`): replace the one-row
  serial loop with two row tiles. Each output row is reduced by four lanes, each
  lane iterating `f = lane, lane + DENSE_INNER_LANES, ...`. The lane-0 reduction
  sums lane partials in ascending lane order and writes `s_encoded[dim]`.
- **Predictor row tiling** (`brain_passes.wgsl:316-332`): apply the same tiling
  shape to predictor train+predict. Weight updates stay per weight and
  prediction dot products reduce four partials per row.
- **Exact thread-0 loop cleanup** (`brain_passes.wgsl:371-395`,
  `:666-683`): move per-dimension context blending, `fast_tanh`, and
  prediction/credit copies to `tid < ENCODED_DIMENSION`; keep reductions whose
  order affects policy/value behind the later multi-workgroup task unless their
  drift gate is added in the same patch.

Properties that make this safe:
- It keeps the one-dispatch fused kernel shape, so no inter-workgroup visibility
  problem is introduced.
- Reduction order changes are intentional. Acceptance uses deterministic
  same-mode tests and fixed-seed no-regression, not byte equality against fused
  serial.
- If the N=10 speedup is below 25%, the result is recorded as "same-dispatch
  tiling insufficient" and the plan proceeds to split multi-workgroup work.

## 0003 - Split-cycle execution scaffold

Today `dispatch_ticks` records one fused kernel dispatch per full
kernel-batch (`gpu_kernel.rs:1541-1569`), and that shader loops
`vision_stride` cycles internally (`kernel_tick.wgsl:527-578`). That internal
loop prevents another workgroup from participating in a single agent's dense
brain math, because inter-workgroup synchronization requires a dispatch
boundary.

### Execution mode

Edits:

- **Execution mode enum** (`gpu_kernel.rs`, near `DispatchProbe`): parse
  `XAGENT_BRAIN_EXECUTION_MODE` once.

```rust
/// GPU brain execution strategy. FusedSerial is the shipping path; the split
/// modes exist to measure and then unlock per-agent multi-workgroup brain math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BrainExecutionMode {
    FusedSerial,
    SplitSerial,
    ParallelTiled,
}
```

- **Dispatch branch** (`dispatch_ticks`, `gpu_kernel.rs:1492`): keep the current
  body as `dispatch_ticks_fused_serial`. Add `dispatch_ticks_split_serial` that
  records, for each configured brain cycle in a full batch, one single-cycle
  kernel dispatch with `WC_VISION_STRIDE = 1`, then records the existing
  `global` and `vision` passes once after the configured number of cycles. The
  tick sequence and sensory lag remain the same: only the internal loop is
  externalized.
- **Test hook** (`gpu_kernel.rs`): add a test-only constructor/setter so
  integration tests can run `BrainExecutionMode::SplitSerial` without mutating
  process-global env state.

Properties that make this safe:
- `SplitSerial` still uses the existing serial brain code and only changes where
  the cycle boundary is recorded. It must be byte-identical to `FusedSerial` for
  final physics/brain state at fixed seeds.
- The 15% overhead gate decides whether multi-workgroup dispatch sequencing is
  viable before any new math is written.

## 0004 - Multi-workgroup dense brain phases

Today all brain intermediates are workgroup-local (`s_features`, `s_encoded`,
`s_prediction`, `s_credit`, and related scalars; `brain_passes.wgsl:22-36`).
That is perfect for the fused serial shader, but it blocks per-agent
multi-workgroup work. The parallel path needs storage-backed intermediates with
explicit dispatch ordering.

### Scratch layout

Edits:

- **Scratch constants** (`common.wgsl`, after `DECISION_STRIDE`): define a
  per-agent scratch buffer. Rust mirrors these in `buffers.rs`.

```wgsl
override SCRATCH_FEATURES: u32 = 0u;
override SCRATCH_ENCODED: u32 = SCRATCH_FEATURES + FEATURES_STRIDE;
override SCRATCH_HABITUATED: u32 = SCRATCH_ENCODED + ENCODED_DIMENSION;
override SCRATCH_HOMEO: u32 = SCRATCH_HABITUATED + ENCODED_DIMENSION;
override SCRATCH_RECALL: u32 = SCRATCH_HOMEO + 6u;
override SCRATCH_RECALL_SIMILARITY: u32 = SCRATCH_RECALL + RECALL_IDX_STRIDE;
override SCRATCH_PREDICTION: u32 = SCRATCH_RECALL_SIMILARITY + RECALL_K;
override SCRATCH_CREDIT: u32 = SCRATCH_PREDICTION + PREDICTOR_DIMENSION;
override SCRATCH_SCALARS: u32 = SCRATCH_CREDIT + ENCODED_DIMENSION;
override BRAIN_SCRATCH_STRIDE: u32 = SCRATCH_SCALARS + 4u;
```

- **Scratch binding** (`common.wgsl`, binding 13): binding 13 is unused today.

```wgsl
@group(0) @binding(13) var<storage, read_write> brain_scratch: array<f32>;
```

- **Rust buffer** (`gpu_kernel.rs`, buffer creation and bind group): allocate
  `n * layout.brain_scratch_stride * 4` bytes with `STORAGE | COPY_SRC |
  COPY_DST` and bind it at slot 13 in both world-config bind groups.

### Split pipeline sequence

Edits:

- **Cycle prefix shader** (`phase_brain_cycle_prefix.wgsl`): one workgroup per
  agent; performs the current per-cycle physics loop, food detection, and
  death/respawn from `kernel_tick.wgsl:530-560`; writes alive and any direct
  physics state exactly as today.
- **Feature-to-scratch shader** (`phase_brain_features.wgsl`): copies the logic
  of `coop_feature_extract` (`brain_passes.wgsl:70-123`) into
  `brain_scratch[agent_base + SCRATCH_FEATURES + i]`.
- **Tiled encode shader** (`phase_brain_encode_tiled.wgsl`): dispatch
  `workgroups(agent_count, ceil(ENCODED_DIMENSION / 16), 1)`. Each workgroup owns
  16 encoded output rows; 16 lanes per row reduce features from scratch and write
  `SCRATCH_ENCODED + dim`.
- **Serial tail shader** (`phase_brain_tail_from_scratch.wgsl`): one workgroup
  per agent; loads scratch features/encoded into existing workgroup arrays, then
  runs habituation, recall, action, memory reinforcement/store/decay, and trace
  publication. Encoder-credit update is skipped here when `ParallelTiled` is on.
- **Tiled encoder-credit shader** (`phase_brain_encoder_credit_tiled.wgsl`):
  dispatch `workgroups(agent_count, ceil(ENCODED_DIMENSION / 16), 1)`, update
  `O_ENC_WEIGHTS` from scratch features and `decision_buffer[DECISION_CREDIT]`.
  This is safe after the tail because encoder weights are not read again until
  the next cycle's encode.
- **Tiled predictor shader** (`phase_brain_predictor_tiled.wgsl`): dispatch
  `workgroups(agent_count, ceil(PREDICTOR_DIMENSION / 16), 1)`, perform
  predictor train+predict, and write `SCRATCH_PREDICTION`. The action tail reads
  that scratch prediction instead of recomputing rows serially.

Properties that make this safe:
- Dispatch ordering provides the cross-workgroup visibility WGSL cannot provide
  inside one dispatch.
- Each tiled workgroup owns a disjoint output row range, so no atomics are
  required for encoded rows, predictor rows, or encoder-credit weight updates.
- Encoder-credit reordering is semantically safe: it only writes
  `O_ENC_WEIGHTS`, which the current cycle no longer reads after encode.
- Parallel reductions may drift from fused serial. The default fused path remains
  exact; `ParallelTiled` has deterministic same-mode tests plus bounded-drift and
  fixed-seed learning gates.

### Action-tail reductions

Today thread 0 in `coop_predict_and_act` serially reduces prediction error,
value, norm clamps, policy dot products, attenuation mean, and publish/copy loops
(`brain_passes.wgsl:349-695`). Some scalar decisions must remain ordered, but the
vector parts do not need one thread.

Edits:

- **Parallel value/policy reductions** (`phase_brain_tail_from_scratch.wgsl`):
  use `tid < ENCODED_DIMENSION` partials and a fixed ascending tree reduction for
  value dot, action forward/turn dot, value/action L2 norms, prediction error,
  and attenuation mean.
- **Parallel vector copies** (`brain_passes.wgsl` and split tail): write
  `O_PREV_PREDICTION`, `DECISION_PREDICTION`, and `DECISION_CREDIT` from
  `tid < ENCODED_DIMENSION` instead of thread 0 loops.

### Memory reinforcement tiling

Today memory reinforcement assigns one pattern per thread and each thread loops
all 128 encoded dimensions (`brain_passes.wgsl:767-789`). If the dense-row
phases and action-tail reductions still miss 60 k, split reinforcement into
pattern x dimension tiles.

Edits:

- **Tiled reinforcement shader** (`phase_brain_memory_reinforce_tiled.wgsl`):
  dispatch `workgroups(agent_count, MEMORY_CAP, 1)` or
  `workgroups(agent_count, ceil(MEMORY_CAP / 8), 1)` depending on measured
  occupancy. Each workgroup owns a pattern or pattern tile, reduces encoded
  dimensions, and writes only that pattern's reinforcement/valence fields.

## 0005 - Non-brain floor recovery

Today the fused kernel's `agent_food_detect` scans every food item for every
agent every brain cycle (`kernel_tick.wgsl:201-277`). At the default world this
means hundreds of food distance checks per agent cycle even though the project
already maintains a food grid and has a grid-neighborhood detector
(`phase_food_detect.wgsl:19-66`). Food positions are static between global
rebuild/respawn passes; consumed flags are atomic and already skipped, so the
grid is a valid candidate for the fused per-cycle eat check.

Edits:

- **Grid-backed fused food detect** (`kernel_tick.wgsl`): replace the brute-force
  food scan with a 3x3 food-grid neighborhood scan matching
  `phase_food_detect.wgsl`, keeping the cooperative reduction only if multiple
  threads scan slots in parallel. Preserve nearest-food semantics within the
  scanned neighborhood and the existing atomic claim.
- **Overflow guard** (`kernel_tick.wgsl`): if a cell's atomic count exceeds
  `FOOD_GRID_MAX_PER_CELL`, fall back to brute-force for that agent/cycle so the
  optimized path cannot silently miss food in overfull cells.

Properties that make this safe:
- It changes only the candidate enumeration path; the same food state,
  `food_flags`, distance check, nearest-food selection, and atomic claim decide
  consumption.
- The overflow fallback preserves correctness in dense cells while letting the
  common case avoid scanning all food.

## 0006 - 60 k closure

Today there is no final decision artifact for the 60 k target. Plan 0006 must
end with one result: the default execution mode changes only when the target run
is >=60 k. If the target is still missed after the exhaustive work list, the
decision doc names the remaining owner of the gap and the plan is incomplete for
the stated performance goal until a follow-up is authored against that owner.

Edits:

- **Decision doc** (`0006-60K-CLOSURE.md`): created by the final task. It must
  start with "SHIP - 60 k reached" or "TARGET MISSED - remaining owner is ...",
  then list N=10 tps, N=200 agent-ticks/sec, split overhead, drift metrics,
  fixed-seed evolution result, and the exact next blocker if the target is
  missed.
- **Default switch** (`gpu_kernel.rs`): only if the decision is SHIP, change the
  default execution mode from `FusedSerial` to `ParallelTiled`. Keep
  `XAGENT_BRAIN_EXECUTION_MODE=fused-serial` as an escape hatch for one release.

## Test strategy

- **Default path unchanged:** existing `deterministic_across_batch_sizes` and
  `fused_dispatch_matches_split` stay green for `FusedSerial`.
- **Split serial equivalence:** add an integration test that runs the same fixed
  seed/tick count in `FusedSerial` and `SplitSerial` and asserts byte-equal
  physics, decision, brain, and pattern state.
- **Parallel tiled determinism:** add an integration test that runs
  `ParallelTiled` across one large dispatch and many smaller dispatches and
  asserts byte-equal state within that mode.
- **Parallel tiled drift:** add a fixed-seed comparison against `FusedSerial`
  with thresholds recorded in `TASKS.md`: finite state, same alive/death counts
  over the short smoke run, motor outputs in [-1, 1], and max telemetry drift
  under the task's threshold.
- **On-target performance gate:**

```bash
cargo build --release -p xagent-sandbox
./target/release/xagent --bench --bench-ticks 1000000 --bench-agents 10
XAGENT_BRAIN_EXECUTION_MODE=parallel-tiled \
  ./target/release/xagent --bench --bench-ticks 1000000 --bench-agents 10
XAGENT_BRAIN_EXECUTION_MODE=parallel-tiled \
  ./target/release/xagent --bench-agent-sweep --bench-ticks 200000
```

- CI gate that must stay green:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p xagent-sandbox
```

## Interaction with prior work

- **Builds on Plan 0003.** Keeps fused submit batching, probe counters, and the
  rejection of `global` parallelization.
- **Builds on Plan 0005.** Uses the pass-limit profile's measured result:
  top-K is not the bottleneck; dense brain passes are.
- **Preserves Plan 0002.** The worker boundary, async readback, and 60 Hz
  publication model are not reopened.
- **Preserves Plan 0001.** Sensory lag and stride decisions are not changed for
  throughput.
