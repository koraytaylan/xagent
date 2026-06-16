# XAgent Plan 0006 - Multi-Workgroup Brain Parallelism

Drive to 60 k raw tps at 1000x by measuring the N=10 no-brain floor, applying
same-dispatch dense tiling first, proving split-cycle overhead before building on
it, then moving every dominant dense brain loop into storage-backed
multi-workgroup phases. If the non-brain floor is exposed, replace the fused
brute-force food scan with grid-backed detection. The plan cannot stop after
"right direction"; the final closure task ships only when the target run clears
60 k or names the one remaining measured owner of the gap.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md) for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/0006-Multi-Workgroup-Brain-Parallelism--{id}/`).
- **Depends on** lists direct prerequisites only (`-` means none).
- **Done when** is the verifiable acceptance check. Every task must keep
  `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, and
  `cargo test -p xagent-sandbox` green ("cargo fmt/clippy/test green") unless it
  explicitly says it is documentation/measurement-only.
- GPU tests self-skip without an adapter (`GpuKernel::is_available()`); CI runs
  Mesa lavapipe. Throughput numbers require the target macOS/Metal GPU.
- Line numbers are hints; locate every site by the named symbol (grep).

---

## 0001 - Throughput budget and no-go map

### n10-throughput-budget-baseline - Prove the 60 k target is reachable by brain work

The target is raw N=10 ticks/sec, not agent-ticks/sec. Plan 0005 measured the
N=200 profile, but the decisive first gate is N=10: if physics+food+death+vision
with the brain disabled cannot clear 90 k tps, then even a perfect brain rewrite
cannot safely land 60 k.

**Steps:**
1. Build the release binary on the target macOS/Metal machine:

   ```bash
   cargo build --release -p xagent-sandbox
   ```

2. Run the N=10 budget table with existing knobs:

   ```bash
   ./target/release/xagent --bench --bench-ticks 1000000 --bench-agents 10
   XAGENT_KERNEL_PASS_LIMIT=0 ./target/release/xagent --bench --bench-ticks 1000000 --bench-agents 10
   XAGENT_KERNEL_PASS_LIMIT=2 ./target/release/xagent --bench --bench-ticks 1000000 --bench-agents 10
   XAGENT_KERNEL_PASS_LIMIT=5 ./target/release/xagent --bench --bench-ticks 1000000 --bench-agents 10
   XAGENT_KERNEL_PASS_LIMIT=7 ./target/release/xagent --bench --bench-ticks 1000000 --bench-agents 10
   ```

3. Add a dated "Plan 0006 N=10 budget" subsection to
   `docs/superpowers/specs/2026-06-10-learning-baseline.md` with columns:
   `mode`, `tps`, `batches`, `submits`, `read`.
4. Apply this decision rule in that subsection:
   - If pass-limit 0 is >=90,000 tps, continue: the non-brain floor supports 60 k.
   - If pass-limit 0 is <90,000 tps, mark
     `fused-food-grid-detect-floor-recovery` as mandatory before the final
     closure run. Do not stop the plan at this gate.

- **Depends on:** -
- **Done when:** the N=10 table and continue/floor-recovery verdict are recorded; no code
  changed; documentation-only, cargo gate not required.

---

## 0002 - Same-dispatch cooperative tiling

### same-dispatch-dense-tiling - Use all 256 lanes for dense dot products before splitting dispatches

The current dense passes leave half the workgroup idle and keep long inner loops:
`coop_encode` loops all features per output row (`brain_passes.wgsl:129-136`)
and predictor train+predict loops 128 inputs per row twice
(`brain_passes.wgsl:316-332`). This task keeps the fused dispatch shape and
tilings only inside the existing workgroup. It is the cheapest possible
parallelism; it must be tried before the more invasive split path.

**Steps:**
1. In `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`, add the dense
   tiling constants and scratch exactly as named in `ARCHITECTURE.md`:

   ```wgsl
   const DENSE_OUTPUT_TILE: u32 = 64u;
   const DENSE_INNER_LANES: u32 = 4u;
   var<workgroup> s_dense_partials: array<f32, BRAIN_WORKGROUP_SIZE>;
   ```

2. Rewrite `coop_encode` so each tile uses all 256 invocations:
   `output_in_tile = tid / DENSE_INNER_LANES`, `lane = tid % DENSE_INNER_LANES`,
   `dim = tile + output_in_tile`, lane-local feature loop
   `for (var f = lane; f < FEATURE_COUNT; f += DENSE_INNER_LANES)`, lane-0
   ascending reduction over the four lane partials, then `fast_tanh(sum)`.
3. Rewrite the predictor row train+predict block in `coop_predict_and_act` with
   the same tile/lane mapping. Each weight entry is updated by exactly one lane;
   prediction dot products use the same four-lane ascending reduction.
4. Move the context-blend/tanh and prediction/credit copy loops that are
   per-dimension independent from thread 0 to `tid < ENCODED_DIMENSION`, keeping
   the `k` loop order inside each dimension unchanged.
5. Add or update integration tests:
   - Existing default determinism tests stay green.
   - Add a short fixed-seed smoke comparison that asserts finite values, motor
     outputs in [-1, 1], and no alive/death count divergence over the smoke
     horizon. Do not assert byte equality against the old fused serial path,
     because reduction order intentionally changes.
6. Measure on target:

   ```bash
   ./target/release/xagent --bench --bench-ticks 1000000 --bench-agents 10
   ./target/release/xagent --bench-agent-sweep --bench-ticks 200000
   ```

   Record before/after N=10 tps and N=200 agent-ticks/sec in the Plan 0006
   baseline subsection.

- **Depends on:** `n10-throughput-budget-baseline`
- **Done when:** same-dispatch tiling is merged if N=10 tps improves by >=25%
  without reducing N=200 agent-ticks/sec by more than 5%; otherwise the measured
  negative is recorded in the Plan 0006 baseline subsection and the split
  multi-workgroup tasks remain mandatory; cargo fmt/clippy/test green.

---

## 0003 - Split-cycle execution scaffold

### split-serial-cycle-scaffold - Prove cycle splitting overhead before multi-workgroup math

Multi-workgroup-per-agent brain math requires dispatch boundaries. Today the
kernel hides all `vision_stride` brain cycles inside one dispatch
(`kernel_tick.wgsl:527-578`), so this task adds a behavior-equivalent
`SplitSerial` mode that runs the same serial brain one cycle at a time. This
measures the dispatch/barrier tax before any new algorithm can mask it.

**Steps:**
1. In `crates/xagent-brain/src/gpu_kernel.rs`, add `BrainExecutionMode` exactly
   as specified in `ARCHITECTURE.md` and parse `XAGENT_BRAIN_EXECUTION_MODE`:
   `fused-serial` (default), `split-serial`, `parallel-tiled`.
2. Refactor the current `dispatch_ticks` body into
   `dispatch_ticks_fused_serial`.
3. Add `dispatch_ticks_split_serial`:
   - For each full kernel-batch, write world config with `cycles = 1` for each
     cycle and dispatch the existing `kernel_pipeline` once per cycle with the
     correct `start_tick`.
   - After `self.vision_stride` single-cycle kernel dispatches, run the existing
     `global` and `vision` passes once, matching the current sensory lag.
   - Keep remainder-cycle and physics-remainder behavior byte-identical.
4. Add a test-only setter or constructor so integration tests can select
   `SplitSerial` without setting process-global env vars.
5. Add an integration test:

   ```rust
   #[test]
   fn split_serial_matches_fused_serial() {
       if !xagent_brain::GpuKernel::is_available() {
           eprintln!("Skipping: no GPU/fallback adapter available");
           return;
       }
       // Run the same fixed seed and tick count through FusedSerial and
       // SplitSerial, including a non-multiple remainder. Assert byte-equal
       // physics, decision, brain_state, and pattern_buffer snapshots.
   }
   ```

6. Measure on target:

   ```bash
   ./target/release/xagent --bench --bench-ticks 1000000 --bench-agents 10
   XAGENT_BRAIN_EXECUTION_MODE=split-serial \
     ./target/release/xagent --bench --bench-ticks 1000000 --bench-agents 10
   ```

- **Depends on:** `n10-throughput-budget-baseline`
- **Done when:** `SplitSerial` is byte-identical to `FusedSerial`; split-serial
  N=10 tps is no more than 15% slower than fused serial. If overhead is >15%,
  keep the scaffold but prioritize `same-dispatch-dense-tiling`,
  `parallel-reduce-action-tail`, and `fused-food-grid-detect-floor-recovery`
  before any further split dispatches; cargo fmt/clippy/test green.

---

## 0004 - Multi-workgroup dense brain phases

### scratch-buffer-and-feature-phase - Add storage-backed brain intermediates

All current brain intermediates live in workgroup memory
(`brain_passes.wgsl:22-36`), which cannot cross workgroups. This task adds the
single storage scratch buffer and the first split phase, but keeps behavior
unchanged until `ParallelTiled` is selected.

**Steps:**
1. In `crates/xagent-brain/src/shaders/kernel/common.wgsl`, add the
   `SCRATCH_*` offsets, `BRAIN_SCRATCH_STRIDE`, and binding 13 exactly as shown
   in `ARCHITECTURE.md`.
2. Mirror the scratch layout in `crates/xagent-brain/src/buffers.rs` with a
   `BrainLayout::brain_scratch_stride` field and tests proving Rust/WGSL offsets
   match for the default 8x6 layout and a non-default 17x13 layout.
3. In `GpuKernel::new`, allocate `brain_scratch_buffer` as
   `n * layout.brain_scratch_stride * 4` bytes and bind it at slot 13 in both
   bind groups. Keep all existing binding numbers stable.
4. Add `phase_brain_features.wgsl`, copying `coop_feature_extract` semantics but
   writing `brain_scratch[agent_base + SCRATCH_FEATURES + i]`.
5. In `ParallelTiled` mode only, run `phase_brain_features` after each cycle
   prefix and before tiled encode. In `FusedSerial` and `SplitSerial`, do not
   dispatch it.

- **Depends on:** `split-serial-cycle-scaffold`
- **Done when:** default fused behavior is unchanged, scratch layout tests pass,
  and a `ParallelTiled` smoke run reaches feature extraction without validation
  errors; cargo fmt/clippy/test green.

### multi-workgroup-encode-and-credit - Split encode and encoder-credit over output tiles

`encode` is a measured clean cost (+1.95s at the Plan 0005 N=200 profile), and
encoder-credit learning repeats the same feature x encoded weight matrix update
inside `learn_and_store`. These two operations expose independent row/feature
work and do not need atomics when each workgroup owns a disjoint output tile.

**Steps:**
1. Add `phase_brain_cycle_prefix.wgsl` for one cycle of physics, food detection,
   and death/respawn, copied from `kernel_tick.wgsl:530-560`.
2. Add `phase_brain_encode_tiled.wgsl`:
   - Workgroup size 256.
   - Dispatch shape `agent_count x ceil(ENCODED_DIMENSION / 16) x 1`.
   - Each workgroup owns 16 encoded dimensions.
   - For each dimension, 16 lanes reduce `FEATURE_COUNT` values from
     `SCRATCH_FEATURES` and write `SCRATCH_ENCODED + dim`.
3. Add `phase_brain_tail_from_scratch.wgsl`:
   - One workgroup per agent.
   - Load `SCRATCH_FEATURES` into `s_features` and `SCRATCH_ENCODED` into
     `s_encoded`.
   - Run passes after encode using existing helper logic.
   - Skip the encoder-credit subsection of `coop_learn_and_store` when
     `ParallelTiled` is active.
4. Add `phase_brain_encoder_credit_tiled.wgsl`:
   - Workgroup size 256.
   - Dispatch shape `agent_count x ceil(ENCODED_DIMENSION / 16) x 1`.
   - Update only `O_ENC_WEIGHTS + feature * ENCODED_DIMENSION + dim` from
     `SCRATCH_FEATURES` and `decision_buffer[DECISION_CREDIT + dim]`.
   - Run after the tail and before the next cycle prefix.
5. Add integration tests:
   - `parallel_tiled_encode_mode_deterministic_across_batch_sizes`.
   - A bounded-drift smoke comparison against fused serial with a fixed seed.
6. Measure on target at N=10 and N=200, recording tps and agent-ticks/sec.

- **Depends on:** `scratch-buffer-and-feature-phase`
- **Done when:** `ParallelTiled` with encode+credit tiled is deterministic within
  mode, bounded-drift against fused serial, improves N=10 tps by >=40% over the
  pre-task baseline, and does not regress N=200 agent-ticks/sec by more than 10%;
  cargo fmt/clippy/test green.

### multi-workgroup-predictor-and-action - Split predictor train+predict and remaining dimension work

After encode+credit, the remaining large measured pass is `predict_and_act`.
The predictor rows are independent, and much of the action tail is per-dimension
work currently serialized through thread 0. This task moves predictor
train+predict into tiled workgroups and parallelizes the safe per-dimension tail
work, leaving scalar policy decisions in one workgroup.

**Steps:**
1. Add `phase_brain_predictor_tiled.wgsl`:
   - Workgroup size 256.
   - Dispatch shape `agent_count x ceil(PREDICTOR_DIMENSION / 16) x 1`.
   - Each workgroup owns 16 predictor rows.
   - Train the row weights using `O_PREV_PREDICTION`, `SCRATCH_ENCODED`, and
     `O_PREV_ENCODED`, then compute the row prediction and write
     `SCRATCH_PREDICTION + dim`.
2. Split the action tail so `phase_brain_tail_from_scratch.wgsl` reads
   `SCRATCH_PREDICTION` and skips the old predictor row loop.
3. Move these per-dimension operations out of thread 0 in the tail:
   - context blend contribution per prediction dimension, preserving `k` order;
   - `fast_tanh` over prediction dimensions;
   - prediction and credit copies into `decision_buffer`;
   - any vector writes whose element values do not depend on a cross-dimension
     reduction.
4. Keep scalar reductions that set policy/value decisions in the tail unless a
   bounded-drift reduction test is added in the same patch.
5. Measure on target:

   ```bash
   XAGENT_BRAIN_EXECUTION_MODE=parallel-tiled \
     ./target/release/xagent --bench --bench-ticks 1000000 --bench-agents 10
   XAGENT_BRAIN_EXECUTION_MODE=parallel-tiled \
     ./target/release/xagent --bench-agent-sweep --bench-ticks 200000
   ```

- **Depends on:** `multi-workgroup-encode-and-credit`
- **Done when:** predictor/action tiling is deterministic within mode,
  bounded-drift against fused serial, and the target run reaches >=60,000 tps at
  N=10 or the measured shortfall is recorded for the final decision; cargo
  fmt/clippy/test green.

### parallel-reduce-action-tail - Remove the thread-0 vector tail inside action selection

`coop_predict_and_act` still leaves vector work in thread 0 after predictor
tiling: prediction error, value dot, L2 norm clamps, action dot products,
attenuation mean, and vector copies (`brain_passes.wgsl:349-695`). This task
keeps scalar policy decisions ordered but moves every independent vector piece to
all 128 dimension lanes.

**Steps:**
1. In the fused path and `phase_brain_tail_from_scratch.wgsl`, rewrite these
   loops to use `tid < ENCODED_DIMENSION` partials plus fixed ascending
   reductions:
   - prediction error sum;
   - value dot;
   - forward/turn/value L2 norm sums;
   - forward/turn policy dot products;
   - attenuation sum.
2. Move vector copies from thread 0 loops to `tid < ENCODED_DIMENSION`:
   - `O_PREV_PREDICTION`;
   - `DECISION_PREDICTION`;
   - `DECISION_CREDIT`.
3. Keep scalar decisions in the same order on thread 0: TD error formation,
   bias updates, memory blend scalar accumulation, exploration-rate computation,
   noise generation, motor clamp, and telemetry scalar writes.
4. Measure N=10 tps and N=200 agent-ticks/sec before/after; record both in the
   Plan 0006 baseline subsection.

- **Depends on:** `multi-workgroup-predictor-and-action`
- **Done when:** action-tail reductions are deterministic within mode,
  fixed-seed smoke metrics remain within the bounded-drift thresholds, N=10 tps
  improves by >=10% or the measured negative is recorded; cargo fmt/clippy/test green.

### multi-workgroup-memory-reinforcement - Tile the remaining large memory loop if target is still short

After encode, predictor, encoder-credit, and action tail are parallelized, the
last obvious brain-side loop is memory reinforcement: 128 pattern threads, each
looping over 128 encoded dimensions (`brain_passes.wgsl:767-789`). This is not
the first target, but it must be handled before claiming there is no more
parallelism to extract.

**Steps:**
1. Add `phase_brain_memory_reinforce_tiled.wgsl`:
   - Workgroup size 256.
   - Dispatch either one workgroup per pattern or 8 patterns per workgroup;
     choose the faster of the two with a local target measurement.
   - For each pattern, reduce encoded dimensions in fixed ascending order and
     update only that pattern's reinforcement and motor-valence fields.
2. Remove the old memory-reinforcement subsection from
   `coop_learn_and_store` / the split tail when `ParallelTiled` is active.
3. Keep memory store, decay, min tracking, active-count tracking, and
   `O_PREV_ENCODED` publication in the tail unless profiling after this task
   proves one of them owns more than 5% of the remaining wall time.
4. Measure N=10 tps and N=200 agent-ticks/sec before/after; record both.

- **Depends on:** `parallel-reduce-action-tail`
- **Done when:** memory reinforcement tiling is deterministic within mode,
  fixed-seed smoke metrics remain within thresholds, and N=10 tps either reaches
  >=60,000 or the measured shortfall is handed to `fused-food-grid-detect-floor-recovery`;
  cargo fmt/clippy/test green.

---

## 0005 - Non-brain floor recovery

### fused-food-grid-detect-floor-recovery - Replace the per-cycle brute-force food scan

If the N=10 pass-limit-0 floor is low, or if all dense brain work is parallelized
and the target is still short, the next concrete owner is `agent_food_detect`.
It currently scans every food item for every agent every brain cycle
(`kernel_tick.wgsl:201-277`). The repo already has a grid-neighborhood detector
in `phase_food_detect.wgsl:19-66`; bring that logic into the fused kernel.

**Steps:**
1. In `kernel_tick.wgsl`, replace the brute-force loop inside
   `agent_food_detect` with a 3x3 food-grid neighborhood scan matching
   `phase_food_detect`.
2. Preserve semantics:
   - skip consumed food with `food_flags`;
   - choose the nearest food within `WC_FOOD_RADIUS`;
   - claim with `atomicCompareExchangeWeak`;
   - award energy and increment `P_FOOD_COUNT` exactly as today.
3. Add an overflow fallback: if any scanned food-grid cell reports a count above
   `FOOD_GRID_MAX_PER_CELL`, run the old brute-force scan for that agent/cycle so
   overfull cells cannot hide food.
4. Add a deterministic GPU test with a small hand-authored food layout:
   - one nearby food in the same cell;
   - one nearby food in a neighboring cell;
   - one consumed nearby food that must be skipped;
   - one overfull-cell case that exercises the fallback.
5. Measure pass-limit-0 N=10 tps before/after and record it.

- **Depends on:** `n10-throughput-budget-baseline`
- **Done when:** grid-backed fused food detection is behavior-equivalent on the
  hand-authored cases, pass-limit-0 N=10 tps improves by >=20% if the floor was
  below 90k, or the measured negative is recorded; cargo fmt/clippy/test green.

---

## 0006 - 60 k closure

### sixty-k-throughput-closure - Ship ParallelTiled only after the target is actually reached

The plan must end with the target run clearing 60 k before defaults change.
Unlike the earlier gates, this task cannot declare success by finding the right
direction. It either ships a measured target hit or states the single remaining
measured owner of the gap after every known heavy loop has been addressed.

**Steps:**
1. Create `docs/plans/0006-Multi-Workgroup-Brain-Parallelism/0006-60K-CLOSURE.md`.
   Its first line after the title must be either:

   ```markdown
   **Decision: SHIP - 60 k reached.**
   ```

   or:

   ```markdown
   **Decision: TARGET MISSED - remaining owner is <phase>.**
   ```

2. Run the final target matrix on macOS/Metal:

   ```bash
   ./target/release/xagent --bench --bench-ticks 1000000 --bench-agents 10
   XAGENT_BRAIN_EXECUTION_MODE=parallel-tiled \
     ./target/release/xagent --bench --bench-ticks 1000000 --bench-agents 10
   XAGENT_BRAIN_EXECUTION_MODE=parallel-tiled \
     ./target/release/xagent --bench-agent-sweep --bench-ticks 200000
   ```

3. Run a fixed-seed headless learning comparison using the same seed, tick
   budget, and generation count as the Plan 0005 fixed-seed table. Record
   deaths-per-food, food/1k, best fitness mean/peak, and wall/gen for
   `FusedSerial` and `ParallelTiled`.
4. If `ParallelTiled` reaches >=60,000 N=10 tps, N=200 agent-ticks/sec regresses
   by <=10%, and the fixed-seed learning table is no worse, change the default
   execution mode to `ParallelTiled` while keeping
   `XAGENT_BRAIN_EXECUTION_MODE=fused-serial` as an escape hatch.
5. If the target is missed, keep the default `FusedSerial` and record the
   remaining measured owner after confirming all prerequisite owner tasks ran or
   recorded measured negatives:
   `same-dispatch-dense-tiling`, `multi-workgroup-encode-and-credit`,
   `multi-workgroup-predictor-and-action`, `parallel-reduce-action-tail`,
   `multi-workgroup-memory-reinforcement`, and
   `fused-food-grid-detect-floor-recovery`.
6. Update this plan's `STATUS.md` and the root `docs/plans/STATUS.md` with the
   final task count and outcome.

- **Depends on:** `multi-workgroup-memory-reinforcement`, `fused-food-grid-detect-floor-recovery`
- **Done when:** `0006-60K-CLOSURE.md` records the target matrix; if it records
  SHIP, default mode is `ParallelTiled`; if it records TARGET MISSED, every known
  owner above has a measured result and the remaining owner is named; status
  boards are in sync; cargo fmt/clippy/test green.

---

**End of plan 0006 TASKS.** When every "Done when" bullet is green, the plan's
end state is reached: either the current hardware runs the small-population
simulation at >=60 k raw tps, or the repo contains a measured, non-speculative
answer for why that target cannot be reached by GPU brain parallelism alone.
