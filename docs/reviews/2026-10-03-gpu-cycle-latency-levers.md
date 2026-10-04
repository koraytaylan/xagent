# Investigation: where a simulation cycle's GPU time goes, and how to cut it without changing results

**Date:** 2026-10-03
**Reviewer:** Claude (Claude Code)
**Base:** `33a7afaa` on `develop`
**Trigger:** find ways to make the GPU computation several times faster without damaging accuracy.

**Follow-up, 2026-10-04:** the [latest production experiments](#suppressing-unchanged-packed-encoder-stores)
measure 3.246× whole-simulation acceleration with optional FP32 reassociation,
or 2.840× with matching public hashes, for 10 agents on a Raphael integrated
GPU; these are separate from the original M3 Max measurements below, and the
requested 10× whole-simulation target remains unmet.

## Summary

- The default configuration runs at **13,050 ticks/sec** with 10 agents: **766 µs per brain cycle** (10 physics ticks).
- The earlier throughput work (plans 0003, 0005, 0006) measured with `vision_stride = 10`. The default is now 1, which changed the profile: vision and the global pass are 54% of a cycle, the brain 33%, physics/food/danger 13%. The ray march alone is 40%.
- Eight changes that leave every result bit-identical bring it to **22,400 ticks/sec (1.72×)**. They were prototyped and verified by state hash.
- Running the vision workgroups in the same dispatch as the brain workgroups, instead of after them, reads **28,100 ticks/sec (2.15×)** as a timing proxy. Making that exact needs a reordering of the cycle, described in R2.
- About **3×** (38–43k) is the estimated ceiling for exact changes on the GPU. It needs the brain's dense loops and the ray steps spread over more workgroups (R3). That figure is an estimate, not a measurement.
- Nothing found here gives more than about 3× for a single 10-agent world on the GPU. The two routes beyond that are listed under "Outside the current rules".

## Method

Apple M3 Max, Metal, release build, default world and brain (10 agents, 8×6 vision, strides 10/1), `--bench --bench-ticks 200000`. Readings repeat within about 1%.

Costs were isolated with the existing knobs (`XAGENT_KERNEL_PASS_LIMIT`, `XAGENT_SKIP_GLOBAL`, `XAGENT_SKIP_VISION`) and with throwaway shader switches in a scratch worktree. None of that code is committed.

"Bit-identical" means: brains seeded with `reset_agents_seeded(42)`, world seed 42, and an FNV hash of the whole `physics_state` readback equal to the base commit's after 20,000 ticks. See F6 for why longer windows cannot be compared on the base commit.

## Findings

### F1. A cycle costs what its longest per-thread loops cost

| Agents | 1 | 5 | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|---|---|
| ticks/sec | 14,119 | 13,189 | 13,050 | 12,583 | 12,061 | 11,167 | 7,006 |

Eighty agents run almost as fast as one, so at 10 agents most of the GPU is idle and the cycle is bound by latency, not by throughput.

- A `workgroupBarrier()` costs about 0.115 µs (200 extra barriers per cycle added 23 µs). The roughly 200 barriers in a brain cycle are about 23 µs in total.
- A dispatch whose workgroups do nothing costs nothing measurable.
- Recording kernel, global and vision as three compute passes instead of one costs 25–38 µs per cycle.
- Zero-initializing workgroup memory costs at most 10 µs per dispatch and was not measurable inside the full stack.
- What costs is a loop running on one thread. Thread 0's scan of the 104 food items costs 28 µs. A 25-step ray still costs about 120 µs after the pruning in F4 and with a workgroup to itself.

So the cycle time is close to the sum, over barrier-separated segments, of the longest per-thread loop in each segment. Shortening those loops, or running independent segments side by side in different workgroups, is what helps. Fewer barriers, fewer submits and fewer passes are worth little.

### F2. Profile of the default cycle (10 agents, µs per brain cycle)

| Part | µs | Share |
|---|---|---|
| Vision rays | 307 | 40% |
| Vision senses (touch, smell) | 48 | 6% |
| Global pass (clear 21, food respawn 23, collisions 15) | 64 | 8% |
| Kernel before the brain (physics 13, food detect 46, danger detect 16, fixed 26) | 97 | 13% |
| Brain | 256 | 33% |
| **Total** | **766** | |

Brain passes, from the pass-limit sweep with global and vision skipped: encode 36, recall score 11, top-K 9.5, predict-and-act 56, learn-and-store 136. The last figure includes about 50 µs of recall work in the earlier passes that only happens once memories exist. Inside learn-and-store the encoder-credit update is 36 and the reinforcement dot 5.

### F3. The ray march does about nine times the grid work it needs

Each of the 25 steps of a ray reads a 3×3 block of food-grid cells and a 3×3 block of agent-grid cells, although a food item can only be hit from within 1.0 of the sample point and consecutive steps (1.2 apart) stay in the same 8-unit cell for several steps. Measured against the 307 µs:

- probing only the food cells that overlap the ±1.0 box around the sample point: −106 µs, bit-identical;
- skipping the agent-grid probe altogether (measurement only): −109 µs;
- reading the grids as plain `u32` instead of through `atomicLoad`: −77 µs, bit-identical.

### F4. Eight exact changes: 13,050 → 22,400 ticks/sec

| # | Change | Alone (µs) | Removed from the stack (µs) |
|---|---|---|---|
| 1 | One ray per workgroup instead of 48 rays in one 256-wide workgroup | −110 | +96 |
| 2 | Food test probes only the grid cells overlapping the ±1.0 box around the sample point | −106 | +50 |
| 3 | Grid clear zeroes only each cell's count; readers never look past the count | −28 | +32 |
| 4 | One compute pass per submit; vision dispatched directly, not indirectly | −25 to −38 | +28 |
| 5 | Nearest-food index carried through the existing parallel reduction as (distance, index), replacing thread 0's rescan of all food for the bearing | −28 | +26 |
| 6 | The 16 recalled similarities are read from the sorted `s_similarities` instead of being recomputed by `cosine_sim_pat_s` | −18 | +23 |
| 7 | Agent-grid 3×3 rescanned only when the sample point's centre cell changes, and skipped while it holds no other live agent | ≈0 | +12 |
| 8 | Vision module declares `food_flags`, `food_grid`, `agent_grid` as plain `array<u32>`; it only reads them | −77 | +3 |

"Alone" is the change applied to the base commit. "Removed from the stack" is the cost of taking it out of the full stack; the two differ because the changes overlap. Change 7 only pays once rays have their own workgroups, and change 8 adds little once change 2 is in.

All eight together: 446 µs per cycle, **22,400 ticks/sec**, state hash equal to the base commit's. What is left: brain 235, vision 126, kernel before the brain 60, global 25.

The gain shrinks with population, because one workgroup per ray saturates the GPU: 1.41× at 40 agents and 1.33× at 80 (measured with four rays per workgroup). The rays-per-workgroup count should be chosen from the population.

### F5. Vision does not have to wait for the brain

Within a cycle the brain reads the sensory buffer written by the previous cycle's vision pass, and vision reads nothing the brain writes (positions, grids, food, and the static field-of-view and smell genes). The two are independent, but they run one after the other.

Dispatching the vision workgroups in the same dispatch as the kernel workgroups read 356 µs per cycle, **28,100 ticks/sec**, on top of the F4 stack. In this proxy vision reads positions while physics is still writing them, so its results are wrong; only its timing is meaningful. With vision removed entirely the same build reads 319 µs, so the overlap recovers most of the vision cost.

### F6. The base commit is not reproducible from run to run beyond about 20,000 ticks

Six 40,000-tick runs of the base commit with identical seeds produced five different state hashes. Six 20,000-tick runs produced one.

`phase_food_grid` has 256 threads claim cell slots with `atomicAdd`, so the order of food items within a cell depends on thread timing. Touch reports the first four contacts in slot order, so the order reaches the brain. With insertion forced onto one thread, 7 of 8 runs of 40,000 ticks landed on one hash: three of the base commit, three of the base commit with change 3 of F4, and two of the full F4 stack. One base-commit run still differed, so at least one more race exists (candidates: the food-respawn insert, the agent-grid insert, the eat race between agents).

This limits how long a bit-comparison gate can run. It is not caused by any change proposed here.

### F7. Measured and not worth doing

- Fewer submits: 24, 96 and 240 batches per submit read the same.
- Fewer barriers: at most about 20 µs per cycle in total.
- Turning off workgroup zero-initialization: no measurable gain in the full stack, and it needs a read-before-write audit.
- The visual pathway's Jacobi refresh: 5 µs per cycle amortized.
- `XAGENT_BRAIN_EXECUTION_MODE=parallel-tiled` as it stands: 14,629 ticks/sec, +12% over fused. It was slower than fused when plan 0006 measured it at `vision_stride = 10`.

## Recommendations

### R1. Land the eight exact changes (measured 1.72×)

Check each one against the build before it by seeded state hash at 20,000 ticks, the method used here, alongside `deterministic_across_batch_sizes` and `fused_dispatch_matches_split`. Choose rays per workgroup from the population (F4).

Why change 2 is exact: a food item is hit only if its squared distance to the sample point is below 1.0, so its x and z each lie within 1.0 of the point, so the cell it is registered in overlaps the ±1.0 box. `floor` and division by the cell size are monotonic, so the float comparison cannot disagree with this. The agent grid cannot be narrowed the same way: collisions move agents after the grid is built, so an agent's registered cell can differ from its current one. Change 7 keeps the full 3×3 and only avoids repeating it.

### R2. Run vision beside the brain (proxy 2.15×)

Reorder the cycle from kernel → global → vision to:

1. kernel prefix: physics, food detect, danger detect, death and respawn;
2. global pass;
3. one dispatch holding the brain workgroups and the vision workgroups.

Two things are needed for this to be exact:

- The brain must keep reading the positions it reads today, which are the ones before collisions. The prefix has to copy them into two physics slots for the position ring in `coop_predict_and_act`. Everything else the brain reads from `physics_state` (energy, integrity, the danger slots) is untouched by the global pass.
- The sensory buffer must be doubled. The brain reads the previous cycle's buffer while vision writes this cycle's, and the two swap every cycle. Telemetry readback has to follow the swap.
 Expected: about 60 + 25 + max(235, 126) + dispatch overhead, close to the proxy.

### R3. Shorten the two remaining chains (estimated, towards 3×)

- **Ray steps in parallel.** The 25 steps of a ray are independent given its origin and direction. One thread per step, then a reduction to the first step that hits or leaves the scene, gives the same result as the serial march. This is sized for small populations (12,000 threads at 10 agents).
- **Dense brain loops over more workgroups.** Encode (36 µs), encoder credit (36), predictor (about 25) and the recall dots (about 25) each run 64–132 iterations per thread in one workgroup. Spread over several workgroups per agent, as `parallel-tiled` already does for three of them, they shorten in proportion. The staging cost that sank this in plan 0006 was per-cycle passes and submits; F1 shows dispatches inside one pass are nearly free.
- **Learn-and-store beside vision.** Its outputs are first read by the next cycle, so it can run in the same dispatch as vision once the brain's shared state it needs (`s_features`, `s_encoded`, `s_memory_key`, the scalar pair) is written to `brain_scratch`.
- Smaller exact items, each measured: recall score computes the query norm in each of 128 threads (7 µs), the visual pathway step runs on thread 0 (24 µs), the reinforcement dot repeats the recall dot (5 µs).

### R4. Remove the grid insertion race (F6)

Independent of speed. Without it a bit-comparison gate longer than about 20,000 ticks cannot pass even on unchanged code.

## Outside the current rules

- **CPU at small populations.** One agent's cycle is about 150,000 multiply-adds plus 48 short rays. Ten agents on the M3 Max's CPU cores would plausibly finish a cycle in 100–150 µs, which is 5–8× the GPU at this population. It contradicts "per-tick simulation logic belongs in WGSL", needs a second implementation kept in step with the shaders, and its floats would not match the GPU's bit for bit. Not measured.
- **Several independent worlds per dispatch.** The GPU runs 80 agents at the speed of 10, so eight 10-agent worlds would evaluate about eight times as many agent-ticks per second with unchanged per-world dynamics. It does not make one world faster, and the baseline spec records that search breadth is not what limits evolution.

## Implementation of R1 (2026-10-03)

All eight exact changes of F4 are on `develop`, one commit each:

| # | Commit subject |
|---|---|
| 4 | perf(kernel): record each fused chunk in one compute pass, vision direct |
| 2 | perf(vision): probe only the food cells a ray sample can reach |
| 8 | perf(vision): read the food flags and grids without atomic loads |
| 1 | perf(vision): spread an agent's rays over workgroups sized to the population |
| 7 | perf(vision): rescan a ray's agent block only when it enters a new cell |
| 3 | perf(global): empty grid cells by zeroing only their counts |
| 5 | perf(kernel): find the food bearing in the parallel food scan |
| 6 | perf(brain): read recalled similarities from the sorted recall scores |

Notes on how each was made exact:

- **Change 2:** the food probe box is widened by 1% beyond the 1.0 hit radius, so float rounding at its edges can never leave a reachable cell out. Probing a superset of the reachable cells gives the same hits.
- **Change 1:** rays per workgroup are 1 up to 16 agents, 4 up to 128 and 16 beyond. The count doubles as needed to stay within the dispatch dimension limit, and `XAGENT_VISION_RAYS_PER_WORKGROUP` overrides it.
- **Change 5:** each thread tracks the bearing candidate by the rescan's own 3-D distance expression, and the (distance, index) pairs merge nearer-first, then lower-index. The merge goes through the brain's argmin scratch, which is idle at that point of the cycle.

Each change was checked against the build before it on a second machine: AMD Raphael integrated GPU (2 compute units), Vulkan, release build. The check is the seeded state hash of this report's method (world and brain seed 42, FNV over `physics_state` and every agent's `brain_state`). It ran at 10 agents over 20,000 ticks and, to exercise agent sightings, at 40 agents over 5,000 ticks. At 40 agents the base commit does not repeat over 20,000 ticks (F6), but it does over 5,000. Every change matched both hashes, change 1 at 1, 4, 16 and 256 rays per workgroup. The batch-size, fused/split and parallel-tiled determinism tests pass.

The profile on that machine differs from the M3 Max's. Skipping the vision pass doubled throughput, but removing its internal work did not move it: no scent −3%, no agent probe 0%, half-length rays +7%. Its cost there was mostly per-pass overhead. Change 4 alone raised throughput from about 4,100 to 5,300 ticks/sec. With an interactive xagent session sharing the GPU throughout, four interleaved runs of each read:

| | ticks/sec, 10 agents |
|---|---|
| Base | 4,194 (4,074–4,279) |
| All eight changes | **5,752** (5,583–6,059) |

That is +37% there. R2 and R3 remain open.

## Implementation of R2 (2026-10-03)

R2 is on `develop` as the shape of the fused cycle whenever `vision_stride = 1` (`XAGENT_BRAIN_BESIDE_VISION=0` opts out). A cycle then runs four steps:

1. The kernel with a pass limit of 0: physics, food and danger detection, death/respawn, and no brain passes.
2. The global pass.
3. One dispatch holding the brain workgroups and the vision workgroups (`brain_vision_tick.wgsl`, vision workgroups 256 wide).
4. `sensory_publish`.

R2 needed two pieces for exactness. They differ slightly from the sketch above:

- **Positions.** The kernel saves each agent's position after death/respawn in two new physics slots, `P_BRAIN_POS_X/Z` (`PHYS_STRIDE` 50 → 52). The brain's staleness ring reads them on every path, so the code is the same in both shapes.
- **Sensory buffer.** It is not swapped every cycle. Vision writes a second buffer (`sensory_next`, binding 17), and `sensory_publish` copies it into `sensory_buffer` after the combined dispatch. So the sensory buffer holds what the serial cycle left there at every cycle boundary, and telemetry and the other execution paths need no change.

**Verification:**

- The seeded hash (over the first 50 physics slots, for comparison with builds before the two new slots) is the same in both shapes: at 10 agents over 20,000 ticks, and at 40 agents over 5,000 ticks.
- A new GPU test (`brain_beside_vision.rs`) runs both shapes for 3,000 ticks: six agents close enough to collide, food and hazard ground. It requires every physics and brain-state value to match.
- Reading the post-collision position instead of the saved one makes the test fail.
- The full integration suite passes.

**It is on by default; `XAGENT_BRAIN_BESIDE_VISION=0` opts out.** It first landed off by default, because on the AMD Raphael iGPU (2 compute units) it was slower than the serial cycle: at 10 agents over 100,000 ticks, 4,455 against 5,908 ticks/sec. Each vision workgroup in the combined dispatch reserves the brain's workgroup memory, and on two compute units that costs more than the overlap gains. Whether it pays on the M3 Max, as the F5 proxy suggests, has not been measured with this implementation. On a GPU like that iGPU, `XAGENT_BRAIN_BESIDE_VISION=0` restores the faster serial cycle.

## Implementation of R4 (2026-10-03)

R4 is on `develop` in two parts, and with both a seeded run reproduces exactly.

- **Grid order.** After the agent grid is built, each cell's food and agent entries are put in index order (`phase_grid_order.wgsl`, an insertion sort per cell on the global pass and the physics-only path). Touch, collisions and the food scan then see the same order whichever thread inserted first.
- **Food claims.** F6's eat race was real: each agent's workgroup took a food item with a compare-and-swap on its flag and ate at once, while other workgroups in the same dispatch were still scanning those flags. A brain cycle is now two kernel dispatches. `kernel_claim_tick` runs the physics and the food scan without writing any food flag, and each agent in reach records itself on the item's claim slot with `atomicMin`. `kernel_tick` settles the claims first, so the lowest-index claimant eats in the same cycle as before and the others go without. The claim slots are the second half of the `food_flags` buffer, because the trail ring already holds the last of the sixteen storage bindings the device is asked for; each agent keeps its claim in a new physics slot, `P_FOOD_CLAIM` (`PHYS_STRIDE` 52 → 53).

**Verification** (state hash over every physics and brain-state value; seeds fixed):

| Runs | Base | Grid order | Grid order + claims |
|---|---|---|---|
| 10 agents, 100,000 ticks | 4 of 4 differ (meals 529–575) | 2 of 3 agree | 3 of 3 agree (meals 538) |
| 40 agents, 20,000 ticks | 4 of 4 differ (meals 311–331) | 3 of 3 agree | 3 of 3 agree |

Two runs each at 10 agents over 500,000 ticks and at 40 agents over 100,000 ticks also agree, and the brain-beside-vision cycle lands on the same hash as the serial one. A new GPU test (`food_claims.rs`) puts two agents equally close to one food item and requires the lower index to eat it, in either order and when neither is agent 0.

On the AMD Raphael iGPU the extra dispatch did not slow the serial cycle: 10 agents over 100,000 ticks ran at about 8,250 ticks/sec with the claims, against 7,640 with grid order alone and 7,600 before either. The brain-beside-vision cycle, now on by default, ran at about 6,800 there; `XAGENT_BRAIN_BESIDE_VISION=0` keeps the faster serial cycle on such a GPU.

## Exact parallel samples and registered-agent masks (2026-10-04)

The follow-up starts from `e8b07c7`, which already contains R1, R2 and R4.
The requested target is at least **10× for the entire simulation** with
precision intact; the representative workloads measured so far do **not**
establish that target.
Comparisons against the original 13,050 ticks/sec M3 Max measurement would mix
hardware and revisions and are not used here.

Two opt-in mechanisms are implemented:

- `XAGENT_VISION_PARALLEL_STEPS=1` assigns 32 invocations to each ray's 25
  existing sample points. Each invocation returns the terminating event at
  its sample, and an integer minimum selects the first event. The sample
  position, field-of-view calculation, squared-distance expressions, terrain
  interpolation, colors and depth expression are unchanged. A sky early exit
  is an event too: it suppresses later hits while retaining sky depth 1.0.
  The event ordering is food, agent, terrain, sky within each sample, then the
  next sample. There is no floating-point reduction or analytic intersection.
- `XAGENT_VISION_AGENT_MASKS=1`, together with parallel samples and at most
  32 agents, replaces each repeated 3×3 agent-grid search with one candidate
  mask. An agent is added to neighboring center-cell masks only after its
  registration obtains a retained grid slot. Masks are cleared with grid
  counts, so they represent the current registered grid before collisions.
  Hit tests still use current post-collision positions. Outside-grid sample
  centers retain the full search because they can see an in-range edge cell.

For a center cell, the mask is exactly the union of retained IDs in the
original nine cells. Changing their iteration order cannot change a ray's
result because all agent hits at the same sample have the same color and
depth. The unchanged serial marcher remains the comparison implementation.
Both standalone vision and brain-beside-vision use the new path when enabled;
the latter still publishes through `sensory_next`. Dispatches that cannot fit
the cooperative workgroup count retain serial samples. Defaults remain serial
samples with candidate masks disabled.

### Measurement scope

Hardware: AMD Ryzen 9 7950X3D integrated GPU, RADV RAPHAEL_MENDOCINO, Vulkan,
Mesa 26.0.8-1ubuntu0.3; optimized Rust builds. These are measurements on this
adapter, not estimates for the review's M3 Max.

`gpu_kernel::vision_validation` compares complete sensory-buffer bits,
including depth and the nonvisual tail. Its GPU timestamps cover 128 repeated
dispatches per sample, seven rounds with variant order rotated, reporting
medians. Scenes distinguish dense flat ground, dense irregular terrain, sparse
flat ground and explicitly synthetic unobstructed long rays. Ray-only timings
exclude the nonvisual senses; full-vision timings include them. Static mask
construction is outside these frozen-scene timings; the advancing simulation
benchmark includes mask maintenance every cycle. Packed serial groups of 32
and 64 rays are also measured to expose gains obtainable without the new
algorithm.

The advancing `vision_performance` example uses a seeded synthetic world,
100 warmup ticks, 10,000 measured ticks and three pairs in alternating arm
order. With ten agents and default 8×6 vision, the default combined schedule
has median elapsed times **1.554348 s serial / 1.315549 s parallel + masks
(1.182×)**. With `XAGENT_BRAIN_BESIDE_VISION=0`, the medians are **1.250385 s /
1.169957 s (1.069×)**. The separate schedule remains faster on this GPU.
These whole-simulation timings use the existing default rays-per-workgroup
choice, whereas microbenchmarks explicitly sweep that choice.

All six public-state hash components match within every pair and across
repetitions: physics, complete brain state, patterns, available sensory data,
the exposed decision subset, and food positions/timers. The example explicitly
reports its coverage gaps: final depth, food flags/claims, complete decisions,
and private scratch. Complete depth is checked separately by the direct GPU
vision tests. No claim of cross-adapter bit identity is made.

### Reproduction

Run the GPU tests explicitly, on an available hardware adapter:

```sh
cargo test --release -p xagent-brain --lib vision_validation -- \
  --ignored --nocapture --test-threads=1
XAGENT_VISION_AGENT_MASKS=1 cargo run --release -p xagent-brain \
  --example vision_performance -- --ticks 10000 --repeats 3
XAGENT_BRAIN_BESIDE_VISION=0 XAGENT_VISION_AGENT_MASKS=1 \
  cargo run --release -p xagent-brain --example vision_performance -- \
  --ticks 10000 --repeats 3
```

The example launches separate serial/parallel processes, fixes the brain and
world seeds, waits for GPU completion, and fails on mismatching public-state
hashes. `--execution split|parallel-tiled`, `--vision-stride`, `--seed`,
`--agents`, `--width`, and `--height` expose additional equivalence cases.

### Remaining route to 10×

Reducing the old ray cost by ten cannot reduce the entire cycle by ten when
the brain and other phases remain unchanged. The original profile's 40% ray
share would cap even free rays at 1.67× whole-cycle speed. Further experiments
must remove other work or its serial dependencies as well.

Approximate sphere depths, reordered floating-point sums, lower resolution
and reduced vision frequency do not satisfy the precision requirement.

### Cached object queries and ordered scent

Two additional opt-in implementations retain the discrete reference results:

- `XAGENT_VISION_OBJECT_QUERIES=1` caches retained food and agent positions
  cooperatively, rejects impossible objects using sample endpoint bounds,
  and finds the first candidate sample with a monotone dominant-coordinate
  search. It evaluates the original squared-distance expression at candidate
  samples; agent membership is checked at every geometric hit, using the
  registered grid rather than a post-collision spatial reconstruction.
  Capacity is 32 agents and 256 foods; larger scenes use parallel samples.
- `XAGENT_VISION_PARALLEL_SCENT=1` computes independent food contributions in
  parallel, then accumulates them in the original ascending food-index order,
  preserving both conditional nostril additions and the final expression.
  Chunks of 256 support arbitrary food counts without unbounded shared memory.

Both flags imply parallel vision and use a separate vision dispatch to avoid
reserving their shared caches alongside every brain workgroup. Defaults are
unchanged. This schedule change contributes to the overall measured gain and
must not be attributed solely to the object-query algorithm.

The full seven-test GPU matrix passes against a pure serial pipeline composed
only from the unchanged ray/senses source and a serial entry point, excluding
all cooperative fragments and their shared storage from the baseline.
Coverage includes capped food-grid overflow, delayed agent-grid eligibility,
31/32/33-agent mask boundaries, sky ordering, dead observers, dynamic fields
of view and vision sizes, scent range boundaries, and 257/513-food chunks.

For ten agents, the fastest full-vision variant in each measured scene uses
four rays per group, candidate masks and ordered parallel scent; object
queries are exact but did not beat that variant on this adapter:

| Frozen scene | Pure serial vision + senses | Best measured vision + senses | Speedup |
|---|---:|---:|---:|
| Dense flat | 188.376 µs | 44.543 µs | 4.229× |
| Dense terrain | 124.290 µs | 45.075 µs | 2.757× |
| Sparse flat | 107.470 µs | 33.606 µs | 3.198× |
| Synthetic 25-sample open rays | 242.733 µs | 32.098 µs | 7.562× |

Packing the serial baseline into 32/64-ray groups improves some scenes;
relative to the fastest measured serial grouping, those gains are 3.896×,
2.757×, 3.198× and 3.736× respectively. The synthetic ray-only case reaches
12.220× against the default grouping but 5.447× against packed serial; neither
is a whole-simulation result.

With object queries, masks and ordered scent, three alternating pairs of
10,000 advancing ticks give **1.492636 s serial / 1.131062 s optimized
(1.320×)**, with all six public-state hashes matching across arms and runs.
A longer 100,000-tick pair gives **14.885979 s / 11.387694 s (1.307×)** with
matching hashes. The whole-simulation 10× target remains unmet.

```sh
XAGENT_VISION_OBJECT_QUERIES=1 XAGENT_VISION_PARALLEL_SCENT=1 \
  XAGENT_VISION_AGENT_MASKS=1 cargo run --release -p xagent-brain \
  --example vision_performance -- --ticks 100000 --repeats 1
```

The next measurements target real cycle dispatch boundaries, brain resource
use and the feasibility of a single-workgroup persistent cycle; those
experiments must preserve the same evolving state before their timing is
accepted.


### Whole-cycle bottleneck after exact vision changes

An in-pass timestamp harness replays the actual production dispatch sequence
and compares all 13 mutable buffers byte-for-byte against normal execution,
including full sensory depth, food flags/claims, decisions, grids, brain scratch
and trails. Each trial restores the same state after 256 warmup brain cycles;
five alternating timing pairs each advance one normal 24-cycle command chunk.
The test requests in-pass timestamps only on supporting adapters.

With cached objects, candidate masks and ordered scent, the measured stage
medians on Raphael are:

| Production dispatch | GPU time per cycle | Share |
|---|---:|---:|
| Physics and food claim | 34.30 µs | 3.04% |
| Main kernel including brain | 1,014.51 µs | 89.87% |
| Global grid/collision work | 38.15 µs | 3.38% |
| Vision and senses | 41.94 µs | 3.71% |
| Sum | 1,128.90 µs | 100% |

Uninstrumented wall time is 1,123.29 µs per cycle; timestamped wall time is
1,152.39 µs, so instrumentation overhead is reported rather than silently
included in the simulation speed claim. The default combined schedule takes
1,458.68 µs uninstrumented in this scene, while separate serial vision takes
1,184.14 µs. These are a seeded flat-world diagnostic, distinct from the
rolling-terrain advancing example above.

A separate diagnostic restores the identical warmed state before every
single-cycle partial-brain trial, rotates limits 0–7 over seven rounds, and
never feeds partial results into another cycle. Cumulative differences suggest
approximately 166 µs for encoding, 306 µs for prediction/action, and 319 µs for
learning/storage, compared with 17 µs for recall scoring and 38 µs for recall
sorting. These deltas locate work; they are not valid reduced-simulation
throughput measurements.

Two exact brain resource changes were also measured independently against
saved binaries. Disabled visual cortex now uses a one-float shared placeholder
and a creation-time guard; the enabled path is unchanged. This releases about
6.8 KiB per workgroup but produces no measurable whole-simulation gain
(1.0008× in three alternating pairs). Encoder and encoder-credit threads now
visit contiguous output weights while preserving every original four-lane
sum, bias placement and update expression; three paired runs give 1.017× with
matching public-state hashes. The two enabled/disabled cortex integration
tests also pass. A direct comparison against the original brain shader's
output-major thread mapping and full cortex scratch allocation matches all
13 mutable buffers for 8×6 and 13×9 fields with cortex both disabled and
enabled; encoder weights change in every case, so credit-update coverage is
active. Raw cases use ten agents and 48 comparison cycles, while cortex cases
use one agent and eight cycles, each submitted separately after an earlier
large cortex job lost its GPU context. Neither optimization approaches the
10× whole-simulation target.

```sh
cargo test --release -p xagent-brain --lib profile_production_cycle_dispatches -- \
  --ignored --nocapture --test-threads=1
cargo test --release -p xagent-brain --lib profile_brain_pass_cumulative_costs -- \
  --ignored --nocapture --test-threads=1
```


### Persistent-world feasibility result

A test-only single-workgroup implementation reuses the production helpers and
serially reuses one brain scratch allocation across agents, with storage and
workgroup barriers at every world-wide dependency. It preserves food-claim
arbitration, sorted grid registration, collision iteration order, sensory lag
and exact sample arithmetic. All 13 mutable buffers match through initial
death/respawn and multiple continuation chunks; warmed timing pairs also
require byte identity before reporting a result.

On Raphael, 1,000 measured ticks after 150 warmup brain cycles take median
**0.121440 s for production serial / 0.230003 s persistent (0.528×)** over
three alternating pairs. Serializing the agents outweighs removal of dispatch
boundaries; the prototype therefore remains test-only. This result provides
no evidence for a 10× whole-simulation improvement.

```sh
cargo test --release -p xagent-brain --lib persistent_validation -- \
  --ignored --nocapture --test-threads=1
```


### Whitening and predictor feasibility measurements

RADV shader diagnostics report 248 vector registers per thread, 8 KiB shared
memory, four subgroups per SIMD and no spills for the main brain pipelines.
Those allocations motivate experiments but do not by themselves prove which
source operation causes the cost. Diagnostics use
[`RADV_DEBUG=shaderstats,nocache`](https://docs.mesa3d.org/envvars.html#radv-driver-environment-variables);
debug runs are excluded from timing comparisons.

Three further test-only variants preserve every mutable buffer in the tested
boundary fixtures and every paired timing trial:

| Variant | Production serial | Variant | Whole-simulation ratio |
|---|---:|---:|---:|
| Whitening in a separate dispatch, 960 ticks | 0.113836 s | 0.104879 s | 1.085× |
| Whitening matrices in 512 B shared memory, 960 ticks | 0.113778 s | 0.107371 s | 1.060× |
| Predictor train/predict fusion, 1,000 ticks | 0.118543 s | 0.114858 s | 1.032× |

Each median uses five alternating pairs from the same 256-cycle warmed
checkpoint. Whitening tests cover scheduled refreshes, non-diagonal
covariance, forced deaths and inactive agents; predictor fusion extends
full-state comparison through 100 cycles. The predictor variant retains each
clamped weight for its same-lane prediction, preserving the original
stride-four accumulation and final reduction while removing a reread and a
barrier per tile.

Both whitening variants still compile the hot brain to 248 vector registers
per thread. Whitening therefore is not the only cause of that allocation;
the measured gains cannot support a 10× claim or be multiplied as independent
speedups. These variants remain test-only while per-phase allocation and
combined register-pressure experiments identify the remaining cause.


### Two independent register-allocation peaks

Compile-only stage diagnostics retain runtime inputs and observable outputs,
then label each pipeline creation around RADV's allocation report. Encoding,
features, homeostasis, sorting and learning individually need 16–32 vector
registers; recall scoring and prediction/action each need 256. Splitting
whitening alone leaves the recall peak in the combined shader.

Recall reads the same 128 shared key values first for its norm, then again for
its dot product. Interleaving those two ascending loops preserves each sum's
order and shortens the time those inputs need to remain live. This is an exact
source transformation; its compiler effect is measured separately.

Across five rotated three-arm rounds from the same warmed checkpoint:

| Variant, 960 ticks | Median time | Ratio to same serial schedule |
|---|---:|---:|
| Production serial | 0.114851 s | 1.000× |
| Interleaved recall only | 0.114702 s | 1.001× |
| Interleaved recall + shared whitening | 0.088539 s | 1.297× |

Every arm matches all 13 mutable buffers. Driver diagnostics show the combined
change reduces the main kernel from 248 to 120 vector registers, increasing
the reported subgroups per SIMD from four to eight, with no spills. Either
change alone leaves the other allocation peak. This is evidence for a joint
compiler-resource effect, not for multiplying isolated benchmark ratios;
it still does not establish 10× whole-simulation acceleration.

### Cooperative whitening and exact dense tiles

The Jacobi rotations retain their original serial operation order while
64 invocations independently reconstruct the 64 whitening cells, preserving
each cell's ascending eight-term sum. With interleaved recall, the test-only
prototype compiles to 56 vector registers, 18 reported subgroups per SIMD and
no spills; all 13 mutable buffers match through the same death, inactive-agent
and refresh boundaries. Five paired 960-tick runs give **0.119816 s serial /
0.083537 s cooperative (1.434×)**. Isolating whitening into another dispatch
while interleaving recall gives a similar **1.429×**, at the cost of two extra
dispatches per cycle.

A separate exact tiled experiment retains the canonical feature, cortex,
adaptation and alive-agent handling, the original four partial sums and bias
placement, and both feature and encoded scratch inputs for the tail. It uses
a separate transient buffer, allowing all 13 production buffers to remain
byte-identical. This differs from the existing `ParallelTiled` mode, whose
16-partial arithmetic and feature handling do not match the serial oracle.

Both exact tile widths match through 100 cycles, including forced deaths and
refresh boundaries, and every timing pair compares all mutable state:

| Outputs per tile, 1,000 ticks | Serial median | Tiled median | Ratio |
|---|---:|---:|---:|
| 16 (64 invocations) | 0.118764 s | 0.110002 s | 1.080× |
| 32 (128 invocations) | 0.120334 s | 0.106947 s | 1.125× |

These five-pair measurements favor cooperative whitening over the nine-dispatch
exact tiled schedule on this adapter. They compare the same separate-vision
schedule within each trial, and must not be multiplied by the earlier vision
ratios to infer a combined speedup. The whole-simulation 10× target remains
unmet.

### Combined production opt-ins

The production cooperative path borrows 128 entries of existing learning
scratch for whitening; learning overwrites all 256 entries before reading
them. This avoids adding a shared-memory allocation or Metal resource slot.
The untouched serial shader remains the disabled path and the independent
test oracle. Cortex-enabled and odd 13×9 layouts match all 13 mutable buffers;
raw death/refresh/inactive fixtures match through 41 cycles. Combined with
the fused predictor, the brain path matches through 100 cycles and gives
1.500× against the same separate-vision schedule in five paired trials.

The full production configuration combines cooperative whitening/interleaved
recall, fused inline prediction, cached vision objects, registered candidate
masks and ordered parallel scent. In the rolling-terrain seeded example,
three alternating 10,000-tick pairs give **1.519204 s default / 0.756920 s
optimized (2.007×)**, with all six public-state hashes matching. A 100,000-tick
pair gives **14.922522 s / 7.337975 s (2.034×)** with matching hashes. The default
arm uses combined brain/vision; the optimized arm uses separate vision as
required by its caches. These are measured whole-configuration ratios, not
products of separate speedups. Public hashes retain their documented coverage
limits; construction and readback remain outside the timed interval.

```sh
XAGENT_BRAIN_COOPERATIVE_WHITENING=1 XAGENT_BRAIN_FUSED_PREDICTOR=1 \
XAGENT_VISION_OBJECT_QUERIES=1 XAGENT_VISION_PARALLEL_SCENT=1 \
XAGENT_VISION_AGENT_MASKS=1 cargo run --release -p xagent-brain \
  --example vision_performance -- --ticks 100000 --repeats 1
```

A fresh in-pass profile of the optimized flat-world fixture measures 34.25 µs
claim, 624.67 µs main/brain, 37.68 µs global and 40.93 µs vision per cycle;
uninstrumented wall time is 728.05 µs, instrumented 758.69 µs. The brain remains
84.70% of GPU stage time. Adding cooperative whitening to the exact 32-output
tiled experiment improves it to 1.329× against its separate serial reference,
but is slower than the production cooperative/fused-predictor combination.

**The requested 10× entire-simulation target has not been achieved.** Results
above are local to the two-compute-unit Raphael integrated GPU and do not
establish speedups on the original review's M3 Max. All production additions
are opt-in; their applicability is preserved without changing the defaults.
The objective is GPU-independent efficiency: these transformations retain
portable WGSL arithmetic, workgroup synchronization and workload-based
fallbacks, with no device-name or vendor-specific selection. Local register
counts explain an observed effect rather than define the intended hardware.

The complete combination also passes direct 13-buffer comparison against
independently compiled serial brain and pure serial vision for 8×6 and 9×7
fields over 100 cycles, including death and whitening-refresh boundaries.
Both arms use the separate-vision schedule and identical grid/mask generation
so the auxiliary buffer contents remain directly comparable. Another test
compares original and optimized brain sources through combined brain/vision,
split serial and standalone masked dispatches, each with native subgroups
and the workgroup fallback: all six cases match all 13 buffers over 41 cycles.

The required sandbox suite passes all 284 tests when serialized. An initial
parallel run timed out waiting for a generation-budget worker event; that
test passed in isolation and in the full serialized rerun.
Formatting, workspace-wide Clippy with all targets and warnings denied, and
the 79 normal brain-library unit tests also pass; the hardware-only checks
above were run explicitly in addition to that ordinary unit-test command.

### Sharing the ordered recall norm

A further portable experiment assigns an otherwise unused invocation to
compute the identical ascending query norm once while the pattern invocations
compute their unchanged dots. A barrier publishes the norm through existing
scratch before similarities are calculated. At 128 active patterns this
removes 127 repeated 128-term norm calculations and square roots per agent,
while retaining the distinct tree-reduction norm used later by learning.

All 13 buffers match through 100 cycles and every paired timing trial.
Five warmed 1,000-tick pairs give **0.077538 s current optimized brain /
0.076794 s shared norm (1.010×)**. This small observed difference is not
enough to establish a broadly useful timing gain, so the variant remains
test-only and is excluded from the reported 2.034× production configuration.

### Locating the remaining cost

A single compiled optimized main shader now supports runtime section stops,
with all partial executions restored from the same checkpoint and never used
as input to another cycle. The complete guarded shader matches all 13 buffers;
its control timing is within 0.2% of the uninstrumented shader. Five rotated
trials after 256 warmup cycles attribute approximately 145 µs to predictor
weight training and its dot products, 59 µs to recalled-context blending,
149 µs to encoder credit and context adaptation, and 36 µs to reinforcement.
These cumulative differences are diagnostic estimates and include timing
noise. A whitening refresh adds about 352 µs every 20 cycles, approximately
18 µs amortized; it is no longer the largest recurring cost.

The next exact experiments use the cooperative/fused-predictor baseline,
256 warmup cycles and five rotated 1,000-tick trials. Every timing arm also
compares all 13 buffers, and separate 100-cycle fixtures cover inactive agents,
death and whitening refreshes:

| Test-only variant | Baseline seconds | Candidate seconds | Ratio |
|---|---:|---:|---:|
| Four-item dense prefetch | 0.077376 | 0.072104 | 1.073× |
| Eight-item dense prefetch | 0.077376 | 0.071366 | 1.084× |
| Stable detection reduction trees | 0.077519 | 0.075638 | 1.025× |
| Shared context normalization | 0.077316 | 0.077611 | 0.996× |
| Shared previous predictor inputs | 0.077316 | 0.077026 | 1.004× |
| Both shared prediction caches | 0.077316 | 0.077051 | 1.003× |
| Trusted main shader without software bounds checks | 0.077618 | 0.076933 | 1.009× |

Prefetch retains the same stride-four accumulation and weight clamps, loading
independent inputs before consuming them. Missing tail terms execute no
addition. Detection trees retain original tie ordering; a dedicated fixture
with 320 foods proves that merged scratch-slot priority chooses food 288 over
the smaller food index 64. Bounds-check removal is a controlled, test-only
diagnostic; production checks remain enabled, and Vulkan hardware robustness
is unchanged. Cache and bounds-check timings show no compelling benefit.

Pointwise encoder credit, one weight per invocation, gives **0.089579 s /
0.084801 s (1.056×)** against the separate exact tiled schedule, with all 13
buffers identical. That baseline is slower than the optimized monolithic
brain, so this ratio is not an additional production improvement.

These experiments have not yet been incorporated into the reported 2.034×
whole production configuration, and their ratios must not be multiplied to
infer a combined result. The user subsequently authorized reordered FP32
sums with validated rounding differences; subsequent experiments may use
that allowance, but must report numerical error and behavioral changes
explicitly rather than treating divergent long trajectories as bitwise parity.

### Production dense prefetch and optional wider FP32 prediction

The following measurements use 10 agents on the same two-compute-unit
Raphael integrated GPU with RADV Vulkan; other GPU performance is unmeasured.

`XAGENT_BRAIN_DENSE_PREFETCH=1` enables the eight-item encoder/predictor
prefetch and implies fused inline prediction. With the earlier production
vision and cooperative-whitening options, three alternating 10,000-tick
pairs give **1.473513 s default / 0.652942 s optimized (2.257×)**; a
100,000-tick pair gives **14.530228 s / 6.606974 s (2.199×)**. All six public
hashes match. Direct 13-buffer checks also pass for 8×6 and 9×7 fields through
100 cycles, and for combined, split and masked brain dispatches with native
subgroups and the workgroup fallback through death/refresh boundaries.

With the user's authorization for reordered FP32 sums, the inline predictor
can instead use 8, 16 or 32 lanes per output row, selected by
`XAGENT_BRAIN_PREDICTOR_LANES`; the default remains four. The 256-thread
workgroup, per-weight training, clamps and number of dispatches are unchanged.
Wider rows read more adjacent weights and use a balanced final addition tree.
The option implies fused prediction and uses no vendor/device-name selection.

Five rotated 1,000-tick trials from the same warm checkpoint give:

| Predictor lanes | With scalar dense inputs | With production prefetch8 |
|---|---:|---:|
| 4, each column's baseline | 0.077352 s | 0.071136 s |
| 8 | 0.071831 s / 1.077× | 0.067097 s / 1.060× |
| 16 | 0.069284 s / 1.116× | 0.065264 s / 1.090× |
| 32 | 0.069684 s / 1.110× | 0.065867 s / 1.080× |

The complete 16-lane/prefetch/vision configuration measures **1.475807 s /
0.594244 s (2.484×)** in three alternating 10,000-tick pairs and
**14.540775 s / 6.012051 s (2.419×)** in a 100,000-tick pair. Public hashes
differ across variants; each arm repeats its own hashes exactly in the
three-pair run. These are seeded, evolving-trajectory timings, with possible
workload differences after rounding changes trajectories; the restored-state
measurements above show a benefit when both arms start from the same warmed
checkpoint, but also include 100 evolving cycles and do not fully isolate
dispatch cost from subsequent workload differences.

```sh
XAGENT_BRAIN_COOPERATIVE_WHITENING=1 XAGENT_BRAIN_DENSE_PREFETCH=1 \
XAGENT_BRAIN_PREDICTOR_LANES=16 XAGENT_VISION_OBJECT_QUERIES=1 \
XAGENT_VISION_PARALLEL_SCENT=1 XAGENT_VISION_AGENT_MASKS=1 \
cargo run --release -p xagent-brain --example vision_performance -- \
  --ticks 100000 --repeats 1 --precision fp32
```

Omit the predictor-lane option and use the default exact precision mode to
reproduce the hash-matching prefetch configuration. The benchmark's explicit
`--precision fp32` mode reports differing hashes and enforces repeatability
within each arm across multiple repetitions; a one-pair run does not check
repeatability, and a hash is not a numerical-error measurement.

The numerical harness reports max absolute error, RMS and symmetric normalized
L2 for named floating-point state regions, plus an aggregate count of discrete
storage differences, while
requiring valid layouts, finite values, actual clamp/cursor/grid invariants
and unchanged inactive brain/pattern state. A mature single-cycle comparison
of wider prediction changes only predictions, by at most 5.97e-8 in the
measured fixture. A separate raw-dot probe executes the actual predictor
prefix before context/tanh, compares with f64 sums under a conservative FP32
forward-error bound including explicit subnormal allowances, and checks
weight preservation and complete output writes. All **10,240 dots** pass:
1,280 rows for each of four lane widths, with and without production prefetch,
covering seeded inputs, positive sums, cancellation, exponent sweeps, tiny
normal/subnormal values and sparse rows. The 16-lane composition also passes
state-invariant checks through all three brain dispatch routes with native
subgroups and their fallback; the inactive brain and pattern state remains
byte-identical in every case.

Long behavior checks use three brain seeds, one flat-world geometry,
1,000 cycles, forced deaths and refreshes, with exact 13-buffer repeatability
within both arms. The reported means average population totals across seeds:
food eaten changes from 52.0 to 51.67, energy from 609.91 to 602.99, and summed
prediction error from 0.381 to 0.616; alive and death counts match. Hazard
entries and danger-path distance remain zero in this fixture, so these
samples do not exercise hazard behavior. They demonstrate observable
trajectory differences and **do not establish statistical behavioral
equivalence**. A local dot error bound is not a bound on a nonlinear
simulation's long-term state. Wider prediction therefore remains opt-in.

Other reordered FP32 experiments remain test-only: 16×16 multi-workgroup
dense tiles give 1.033× against cooperative/fused monolithic brain, while
8×32 tiles regress to 0.866×. Replacing thirteen action/learning reductions
with subgroup sums removes up to 65 workgroup barriers but measures only
1.003×; its three-seed trajectories also diverge. Prefetching encoder-credit
updates measures 1.019×/1.027× for four/eight items against dense-prefetch8,
but fails cross-variant bitwise parity and is excluded from production.

**For this 10-agent local benchmark, the best measured long whole-simulation
result is now 2.419× with FP32 reassociation, or 2.199× with matching public
hashes; 10× remains unachieved.**

Final validation passes formatting and workspace Clippy with all targets and
warnings denied, 82 normal brain-library tests, and all 284 sandbox tests with
cooperative whitening, dense prefetch and the 16-lane predictor enabled; the
118-test sandbox integration group completed in 549.72 seconds when serialized.
The hardware diagnostics above were run explicitly in addition to these suites.

### Profiling the composed candidate and reusing memory work

The complete production configuration with prefetch8, predictor16, cooperative
whitening, cached object queries, ordered parallel scent and registered agent
masks measures 594.293 µs per cycle without timestamp instrumentation in the
flat-world profiling fixture. Instrumented wall time is 618.430 µs, and the
599.937 µs GPU timestamp sum divides into claim 34.190 µs (5.70%), main brain
and physics 491.602 µs (81.94%), global update 33.523 µs (5.59%), and vision
40.622 µs (6.77%). Timestamp replay matches all thirteen buffers.

`brain_sections::benchmark_frozen_predictor_width_cycles` also isolates the
earlier predictor-width benefit from evolving trajectories: each arm executes
one cycle from identical restored state, with seven alternating timing pairs,
prefetch8 in both arms, and serial vision in this fixture. After 256 warmup
cycles, width4/width16 main time is 526.520/475.280 µs (1.108×), and the full
cycle is 704.040/657.320 µs (1.071×); at the refresh checkpoint after cycle 260,
main time is 862.000/800.880 µs (1.076×), and full-cycle time is
1,044.520/982.920 µs (1.063×). Each arm's timestamp replay matches its own
production dispatch state exactly; cross-arm rounding differences are
reported separately. Serial vision here takes approximately 109 µs, so these
full-cycle times are not the cached-vision profile above.

With `XAGENT_SECTIONS_PREFETCH=1 XAGENT_SECTIONS_PREDICTOR_LANES=16`, cumulative
section differences identify encoder credit plus context-weight adaptation
at approximately 133–146 µs, recalled context plus tanh at 55–57 µs,
predictor training plus dot products at 45–47 µs, and memory reinforcement
at 34–36 µs. These are differences between guarded prefixes and include
measurement noise; the full guarded path matches the unguarded thirteen
buffers. The periodic whitening refresh adds approximately 349 µs in the
measured refresh cycle.

Two further architectures remain test-only:

* `cached_combined_validation` overlaps cached object vision and ordered scent
  with the optimized brain in one dispatch, aliasing scratch arrays whose
  lifetimes do not overlap. The 8×6 and 9×7 cases use approximately 12.0–12.3 KiB
  of workgroup memory without another workgroup resource slot. Through death
  and refresh boundaries, all twelve persistent outputs match the standalone
  reference; the combined path additionally publishes `sensory_next`, whose
  standalone counterpart stays unchanged. Each candidate repetition matches
  all thirteen buffers. Five alternating 100-cycle pairs yield
  0.058544/0.057241 s, only **1.023×**.
* `recall_reuse_validation` caches unsorted recall cosines in existing argmin
  scratch for later reinforcement, since the query and pattern vectors remain
  unchanged between these phases. Five rotated 100-cycle trials give
  0.065467 s for the optimized reference, 0.062292 s for serial recall reuse
  (**1.051×**), and 0.063523 s for a cooperative variant (**1.031×**). The latter
  moves learning's original norm tree and two-lane dots into recall. Both
  change FP32 association, pass state-invariant and inactive-agent checks
  across 100 cycles including death/refresh, and repeat all thirteen buffers
  exactly within each arm. These timings include evolving trajectories and
  use the serial-vision fixture; they are not additional measured gains on
  the long whole-simulation benchmark.

None of these isolated ratios establishes a combined speedup or the 10× goal.

The raw recall probe separately checks 2,560 pattern cases across both reuse
variants against f64 dot/norm references and propagated FP32 error intervals.
Cancellation, zero and inactive patterns, tiny norms, the 1e-8 norm guard, and
the 0.3 reinforcement threshold are covered. Every case passes; no guard or
reinforcement-threshold flips occurred in these fixtures. This validates local
arithmetic, not statistical equivalence of long simulation trajectories.

### Production encoder credit beside the world update

Encoder credit only needs the adapted features and decision-credit vector
already computed by the brain. Its weight writes have no consumer until the
next brain cycle, while the global world update neither reads those weights
nor changes alive flags. `XAGENT_BRAIN_GLOBAL_CREDIT=1` publishes the actual
adapted features into a private buffer, replaces the inline credit loop with
one-weight-per-invocation workgroups, and schedules them beside the existing
global workgroup. No global spin barrier or additional dispatch is needed:
the same dispatch boundary completes both branches before the next cycle.

The default ten-agent, 267-feature case uses 1,341 workgroups for this combined
dispatch, including the original world-update group. Feature publication is
10,680 bytes per cycle. Each weight retains its original threshold, scale,
multiply/add, and clamp. The option selects a separate brain in fused serial
mode with vision stride 1; split/tiled/masked schedules, longer vision strides,
partial-brain probes, skipped global work and excessive workgroup counts retain
their original implementation. It selects no GPU vendor or device name and is
disabled by default.

The prototype's five alternating 100-cycle pairs measure
0.058598/0.052343 s (**1.119×**) against the optimized cached-vision configuration,
with all thirteen buffers identical. The production dispatch independently
passes all-thirteen-buffer comparisons, repetition, death and refresh checks
for 8×6, 9×7, 1×1, and cortex-enabled 8×6 fields through 100 cycles; six fallback
routes also match their original schedules, including physics remainders.
Warm lifecycle comparisons additionally cover split/tiled transitions and
reactivation, full agent-state replacement, seeded reset with agent upload,
and skipped vision. All thirteen buffers match after each stage, both with
the option alone and with the full optimized configuration, while private
feature storage is deliberately left unrestored.

The option-alone cortex reference lost its RADV context twice during long
multi-cycle submissions, before offloaded execution began. The driver marked
the context innocent; kernel reset attribution was unavailable, so the cause
is not established. Repeating all 100 cortex cycles with one cycle per
submission passes exact state and repetition checks for both arms. The earlier
optimized-brain cortex comparison also passed its larger submissions. The
parity harness now uses bounded cortex submissions; this does **not** validate
long bare-cortex submissions on this driver, and production scheduling has
not been changed to work around that limitation.

With prefetch8, predictor16, cooperative whitening and all the cached vision
options, three alternating 10,000-tick whole-simulation pairs measure
**1.472825/0.530179 s (2.778×)**, with exact per-arm hash repeatability. The
100,000-tick pair measures **14.525987/5.396931 s (2.692×)**. All six candidate
hashes in that long run match the earlier predictor16 configuration, so the
new scheduling change adds no observed trajectory difference to that result.
With the original four-lane predictor, a 100,000-tick pair measures
**14.545071/5.969116 s (2.437×)** and matches all six original public hashes.
Single-pair long runs do not independently establish repetition. These remain
measurements of the ten-agent synthetic rolling-terrain scene on the local
Raphael GPU; performance elsewhere is unmeasured, and **10× remains unmet**.

```sh
XAGENT_BRAIN_GLOBAL_CREDIT=1 XAGENT_BRAIN_COOPERATIVE_WHITENING=1 \
XAGENT_BRAIN_DENSE_PREFETCH=1 XAGENT_BRAIN_PREDICTOR_LANES=16 \
XAGENT_VISION_OBJECT_QUERIES=1 XAGENT_VISION_PARALLEL_SCENT=1 \
XAGENT_VISION_AGENT_MASKS=1 \
cargo run --release -p xagent-brain --example vision_performance -- \
  --ticks 100000 --repeats 1 --precision fp32
```

Omit the predictor-lane option and `--precision fp32` for the hash-matching
configuration. The benchmark's serial child explicitly disables global credit
along with the other optional transforms.

The updated flat-world profile measures 531.914 µs uninstrumented wall time
and 538.240 µs summed GPU stages per cycle: claim 34.200 µs (6.35%), main brain
and physics 341.485 µs (63.44%), global world update plus encoder credit
119.378 µs (22.18%), and vision 43.177 µs (8.02%). Timestamp replay matches all
thirteen buffers. Moving credit reduces main-stage latency but moves work into
the global stage; the measured complete-cycle gain includes both effects.

Validation passes formatting, workspace Clippy with all targets and warnings
denied, 83 normal brain-library tests, and all 284 sandbox tests with global
credit, cooperative whitening, prefetch8 and predictor16 enabled; the sandbox
integration group completed in 590.03 seconds when serialized. Focused GPU
checks above ran in addition to those suites.

### Dense-update observations without changing the shader

Inline counters changed a few rounding results, so that instrumentation was
discarded. `brain_update_diagnostics` instead captures the unmodified
production pipeline's before/after state and requires exact thirteen-buffer
replay for every observed cycle. Host-side diagnostic arithmetic bounds each
observed update and classifies clamps as absent, definite, or unresolved; this
analysis is outside production simulation execution.

Across three brain seeds and sixteen-cycle windows after cycles 256 and 1,000,
the 864 active-agent observations contain 28,593,831 attempted encoder updates.
15,608,383 (**54.59%**) leave the weight bits unchanged, including 13,658,025
whose update is definitely nonzero. The unchanged fraction grows from 21.98%
in the earlier windows to **87.31%** in the later windows. The average active
encoder-credit dimension count is 123.95/128, so the existing credit threshold
skips only about 3.16% of dimensions. By contrast, only 761 of 14,155,776
predictor updates (**0.00538%**) leave their weight bits unchanged.

No gradient or weight clamp, nor unresolved clamp classification, occurs in
these sampled updates; all sampled 1/8/16-cycle absolute-sum envelopes fit the
clamp bounds. This does not prove that deferred updates preserve the computation:
batching tiny nonzero updates can accumulate changes that per-tick FP32 rounding
currently discards. Nor do individual unchanged weights establish that entire
groups can be skipped cheaply. The windows contain no respawns and do not
establish behavior in every world; an alternative representation still needs
its own rounding proof, lifecycle checks, and measured benefit.

A stricter snapshot-only gate compares the largest possible update for a
feature with half the smallest adjacent-float gap among its 128 weights,
considering both neighbors and only finite, normal, in-range weights. It
certifies 46,001 of 230,688 feature rows, with zero observed false positives
against all 128 resulting weight words. Those rows cover 6.38% of attempted
updates in the earlier windows and 32.01% in the mature windows, or 19.17%
overall—far less than the individual no-op rate. No GPU certificate cache is
implemented or timed. Maintaining gap metadata, reducing the scale bounds,
and invalidating cached bounds after host writes and alternate schedules would
all cost work; these observations alone do not establish a speedup.

### Further layout and update-representation experiments

The same snapshot analysis now evaluates contiguous blocks of 32, 64 and 128
output weights separately, using each block's own maximum update and minimum
adjacent-float gap; every certificate hit still asserts unchanged weight bits.
The 32-weight blocks certify 40.15%, 66.44% and 47.00% of attempted updates in
the three mature seed windows, compared with 23.01%, 45.19% and 28.02% for
whole 128-weight feature rows; early-window coverage remains near 6.5%.
There are no observed false positives. The half-gap argument assumes
round-to-nearest, which [WGSL's floating-point rules](https://www.w3.org/TR/WGSL/#floating-point-accuracy) do not universally require,
so observed exactness is a compilation-specific validation result, not a
portable guarantee of bitwise equivalence.

An implemented 32-weight GPU gap cache then passes all-thirteen-buffer
comparisons through 100 cycles in both field sizes, including cold metadata,
death, refresh and replay. Eight explicit harness invalidations cover host
agent/batch writes, zero/subnormal/clamp-edge weights, reset, split/tiled
transitions, partial brain execution and skipped global work. Cache lower
bounds are checked against the actual matrix, and nonzero skip counts are
required. Its finite-input guards reject unsafe factor magnitudes before
multiplication; rejected cases execute the original update.

This cache is slower: five alternating 100-cycle pairs after a 1,000-cycle
warmup measure 0.052254/0.060551 s (**0.863×**) against production global
credit. It adds 85,440 private metadata bytes, including diagnostic skip
counts, plus update-bound checks and block reductions. The measured cost
outweighs skipped weight accesses on this GPU. Production APIs do not maintain
its metadata or invalidation, and the experiment is not promoted.

Three further experiments use the complete optimized global-credit cycle,
including cached vision, cooperative whitening, prefetch8 and predictor16:

* Context gathers prefetched in groups of 4/8/16 preserve all thirteen buffers
  across 8×6 and 9×7 fields with death and refresh boundaries. Five rotated
  100-cycle trials measure 0.997×/1.001×/0.998× against the scalar context loop;
  this does not establish a speedup.
* Encoder layouts with 2/8/16 lanes per output measure
  1.044×/0.962×/0.878× against four lanes after the same 256-cycle checkpoint.
  All arms pass state invariants and repeat all thirteen buffers over five
  trials. These layouts change dot-product association; reported trajectory
  differences are not a numerical error bound or evidence of long-run
  behavioral equivalence, and none is enabled in production.
* Transposed predictor storage preserves all thirteen buffers after decoding
  the matrix layout for comparison. Matching 4/8/16/32-lane pairs measure
  1.089×/0.968×/0.892×/0.807× respectively, but the best transposed arm takes
  0.053804 s versus 0.052334 s for the current row-major sixteen-lane arm.
  Initial upload and final readback conversions are outside these timings;
  no host API or alternative dispatch integration is implemented.

These are test-only measurements on the local GPU and do not improve the
reported 2.692× long whole-simulation result or establish the 10× target.

The deferred encoder primitive stores a base matrix plus 8 or 16 FP32
feature/scale factors, with the original per-output update gate. Its raw
queries add low-rank corrections to the base matvec. Materialization replays
every per-weight update and clamp chronologically in registers, loading and
storing each weight once per window; it reduces matrix traffic, not update
arithmetic. A GPU maximum-magnitude certificate bounds every intermediate
weight, and failure materializes the accepted prefix before applying the
uncertified update densely. This is an isolated primitive, with no simulation
or host-lifecycle integration.

Seven adversarial fixtures across both windows and 33 prefixes pass exact
materialized-matrix comparisons against sequential GPU updates, independent
raw-dot error bounds, clamp-fallback checks and repetition. Tiny updates that
sequential FP32 arithmetic discards still affect deferred queries: the largest
observed raw-dot difference is 3.242493e-5 for the sixteen-step tiny-update
fixture. The numerical oracle explicitly includes that discrepancy rather
than comparing only with ideal real-valued rank-one algebra; its seven CPU
tests also reject omitted factors and one-shot materialization. The shader
with snapshot capture disabled additionally matches the bounded shader's raw
dots, certificate metadata and final matrix for every numerical fixture.

Five rotated primitive timing trials with ten matrices and eight 33-step GPU
dispatches measure **1.126×** for the eight-step window and **1.044×** for
sixteen. Snapshot capture is disabled while timing, and the measured path
includes certification and chronological flushing. Reported logical matrix
traffic excludes cache effects and is not a bandwidth measurement. The result
does not demonstrate a gain over production global-credit scheduling or
behavioral equivalence of an evolving simulation. Export would need to
materialize a copy without changing the pending representation, so readback
frequency cannot alter subsequent query rounding.

### Vision beside the following claim step

A further test-only schedule snapshots all 53 physics words per agent after
the global collision barrier, then runs vision beside the following cycle's
physics and food-claim workgroups. Vision reads the private snapshot; claim
writes live physics and separate atomic food-claim slots. Food positions,
consumed flags, grids and the brain genes read by vision remain unchanged
until the following dispatch. Every submission still starts with its first
claim and ends with vision alone, so public API boundaries retain their
original state and no claim is left pending. No extra shader binding is added;
the snapshot privately replaces the unused sensory-next binding in these
experimental groups without modifying that public buffer.

Both 8×6 and 9×7 fixtures pass all-thirteen-buffer comparisons and repetition
through 100 cycles including death, refresh and singleton submissions. Five
alternating 100-cycle trials use 305 dispatches instead of 400, with the same
five submissions, but measure 0.052329/0.052947 s (**0.988×**). These timings
include the 2,120-byte physics snapshot and changed shader composition;
fewer dispatches do not produce a measured benefit on this GPU, and the
schedule is not promoted.

The completed experiments pass 90 normal brain-library tests, their focused
GPU checks, formatting and workspace Clippy with all targets and warnings
denied; none changes production simulation behavior or establishes 10×.

### Production ordered context gathering

The original context blend assigns one invocation to each output dimension.
Each invocation reads sixteen recalled pattern values from dimension-major
storage. The new gather assigns eight invocations to an output: each loads
two raw values into existing `s_dense_partials` and `s_reinf_dot` storage, and
invocation zero performs the original ascending normalization and weighted
accumulation before tanh. Four output tiles require eight unconditional
barriers. No shared allocation, shader binding or dispatch is added. The
scratch is released before cooperative whitening initializes its matrix
cells, and reinforcement later overwrites all entries before reading them.

Both eight- and sixteen-invocation variants match all thirteen mutable buffers
through 100 cycles in 8×6 and 9×7 fixtures, including forced death, whitening
refresh and two candidate replays. Five rotated 100-cycle measurements after
256 warmup cycles, with ten agents and the existing optimized global-credit
configuration, give 0.053548 s for scalar context, 0.051299 s for sixteen lanes
(**1.044×**) and 0.050529 s for eight lanes (**1.060×**). These are complete
cycle timings from `context-gather-widths-validation.log` on the local GPU.

`XAGENT_BRAIN_CONTEXT_GATHER=1` enables eight lanes; the default remains off.
Constructor-created pipelines also pass independent comparisons against an
ungathered control across fused, overlap-requested, split, tiled, vision-stride
two, full masked and brain-only masked dispatches. Each route compares its
own schedule. Raw 9×7 fixtures run 41 cycles and cortex 8×6 fixtures run four
cycles with short submissions; this does not extend the earlier driver's
long-cortex-submission validation. The checks run both with the other brain
options disabled and with the optimized global-credit configuration.

With the sixteen-lane predictor and the other production optimizations below,
three alternating 100,000-tick pairs measure median **14.598242/5.033922 s
(2.900×)**. Pair ratios are 2.879×, 2.900× and 2.908×. All six public hashes
repeat within each arm, while the candidate differs from the original serial
arm because this configuration permits FP32 reassociation. Context gathering
itself matches the ungathered optimized state in the focused comparisons;
these checks do not establish long-run behavioral equivalence of the wider
predictor. The long public benchmark still omits final depth, consumed-food
and claim flags, the complete decision buffer, and private scratch.

Keeping the original four-lane predictor, one 100,000-tick pair measures
**14.587345/5.648706 s (2.582×)** and matches all six public hashes. That single
pair does not independently establish repetition. Both results use the
ten-agent synthetic rolling-terrain scene on RADV Raphael with Mesa 26.0.8;
performance on other GPUs or populations is unmeasured. **10× remains unmet.**

```sh
XAGENT_BRAIN_GLOBAL_CREDIT=1 XAGENT_BRAIN_COOPERATIVE_WHITENING=1 \
XAGENT_BRAIN_DENSE_PREFETCH=1 XAGENT_BRAIN_PREDICTOR_LANES=16 \
XAGENT_BRAIN_CONTEXT_GATHER=1 XAGENT_VISION_OBJECT_QUERIES=1 \
XAGENT_VISION_PARALLEL_SCENT=1 XAGENT_VISION_AGENT_MASKS=1 \
cargo run --release -p xagent-brain --example vision_performance -- \
  --ticks 100000 --repeats 3 --precision fp32
```

Omit the predictor-lane option and `--precision fp32` for the hash-matching
configuration. The serial child explicitly clears context gathering and the
other optional optimizations. The long FP32 result is recorded in
`context-gather-production-long-benchmark.log`.

### Packed encoder prototype measurements

The initial packed encoder experiment keeps four adjacent output weights in each
`vec4<f32>` without transposing their feature-major byte order. One invocation
accumulates four outputs; 128 invocations cover the four original feature
lanes and all 128 outputs in one tile. Every invocation reaches the barriers.
The existing dense scratch grows from 256 to 512 scalars, and vector credit
assigns each complete vector to one invocation, reducing credit workgroups
from 134 to 34 per agent. Each component retains its original credit threshold,
multiply/add and clamp. Different invocations never write components of the
same vector.

The private binding-13 allocation contains scalar scratch followed by packed
weights and occupies 1,400,000 bytes for the default ten-agent fixture. The
unmirrored variant imports and exports matrices explicitly at test boundaries;
these copies are outside timing. Both 8×6 and 9×7 raw vision grids pass
all-thirteen-buffer comparisons after decoding, including death, refresh, inactive agents and
replay. Separate credit fixtures check mixed threshold decisions, clamp edges,
signed zero and inactive values outside the clamp range. Across ten raw-dot
fixtures, both raw vision grids and both implementations, 5,120 GPU output
values pass independent FP32 forward-error bounds against the FP64 oracle. The bounds
use round-to-nearest unit roundoff plus an allowance for subnormal flushing;
passing characterizes this compilation, not every WGSL-permitted rounding
choice. Matching scalar and packed outputs are bit-identical, including
the cancellation, subnormal and bias-only cases.

An initial five-pair whole-cycle comparison measures 0.052265/0.046414 s
(**1.126×**) against scalar global credit. A subsequent four-arm experiment
also tests writing every updated vector back to the ordinary scalar brain
buffer, so public state remains current without a decoded export. Five
rotated 100-cycle trials after the same 256-cycle warmup report:

| Arm | Seconds | Ratio to scalar reference |
|---|---:|---:|
| Scalar global credit, context gather off | 0.053989 | 1.000× |
| Packed encoder, boundary exports | 0.047386 | 1.139× |
| Packed encoder, scalar mirror each cycle | 0.050248 | 1.074× |
| Packed encoder, scalar mirror and context gather8 | 0.047847 | 1.128× |

All timed arms match the thirteen-buffer reference at the final checkpoint;
the mirrored arms require no decoded export for that comparison. Initial
imports remain outside timing. These numbers are from
`packed-encoder-mirror-timing.log` and use the existing production dispatch
recorder. They do not isolate an incremental packed gain over context gathering
alone, which is absent from this four-arm comparison. These prototype timings
exclude the public reset, host-write and fallback-schedule lifecycle. The
production integration below retains scalar mirroring and adds that lifecycle;
packed storage was absent from the preceding 2.900× context-gathering result.

### Production packed encoder cache

`XAGENT_BRAIN_PACKED_ENCODER=1` enables the packed encoder and global-credit
schedule; the default remains off. Each vector-credit invocation writes its
updated components into both the private vector allocation and the ordinary
scalar brain matrix. The scalar matrix therefore remains authoritative for
serialization, blocking and queued agent-state reads, and fallback pipelines.
No export or readback-layout change is needed.

The private cache starts invalid. Before the first packed compute pass, the
command encoder records a GPU copy of each scalar encoder matrix into its
packed range. Full resets, single-agent and batch brain-state writes invalidate
the cache. Dispatch through split, tiled, masked, longer-vision-stride or
incomplete brain/global routes also invalidates it; returning to the packed
route imports the current scalar matrix before use. Death preserves learned
weights and needs no separate import. The validity flag is atomic, retaining
shared-reference write APIs without adding per-tick CPU simulation logic.

Checked layout arithmetic rejects buffer, shader-index, dispatch and workgroup
storage overflows. Existing scratch grows by 1 KiB without another workgroup
resource slot. Conservatively rounding every shared allocation to sixteen
bytes gives 9,200 bytes for raw 8×6 vision and 15,296 bytes for the 24×24 cortex
retina, below the requested 16,384-byte limit. Larger unsupported layouts
retain scalar global credit. A CPU source-declaration audit guards the shared
storage estimate against shader changes.
The constructor fallback also passes a 20×21 raw-vision fixture with 2,127
features: its packed bound is 16,640 bytes versus 15,616 for scalar storage.
On the 16,384-byte device limit, the cache is absent, scalar global credit
remains active, and three cycles match all thirteen captured buffers.

The raw encoder probe now extracts the production `packed_passes` result and
removes only its final tanh; the credit probe uses the production credit
fragment. The unmirrored timing variant derives from that same fragment by
removing only the scalar-store tail. Thus the numerical fixtures exercise
the canonical arithmetic rather than a duplicated encoder or credit shader.
All 5,120 raw encoder outputs pass the FP32 bounds and match the scalar dots
bitwise. Constructor lifecycle checks also pass all thirteen captured
simulation buffers at five checkpoints per scenario: thirteen raw-vision
scenarios and five cortical scenarios, repeated with optimized and otherwise
bare brain options. They assert cache validity and that a warm cache records
no import, and cover cold cache states, host writes,
resets, fallback execution and resumed packed execution, plus request-time
queued reads and current blocking reads. These comparisons never export the
private cache into the scalar matrix to make the check pass. Warmup includes
death, whitening refresh and an inactive agent; cortical submissions remain
limited to one brain cycle.

With gathered context, the sixteen-lane predictor and the other production
options below, three alternating 100,000-tick pairs measure median
**14.549147/4.678881 s (3.110×)**. Individual pair ratios are 3.103×, 3.107× and
3.115×. Each arm repeats all six public hashes, and the candidate hashes match
the earlier context-only optimized candidate at the same tick count. They
differ from the original serial arm, as expected from the permitted predictor
reassociation. This is a complete production-cycle measurement with scalar
mirroring enabled. Initial cache import occurs during the shared 100-tick
warmup, so these timings do not measure frequent host mutation or fallback
transitions.

Keeping the original four-lane predictor, one 100,000-tick pair measures
**14.511148/5.289362 s (2.743×)** and matches all six public hashes. This single
pair does not establish repetition; its log is
`packed-encoder-exact-long-benchmark.log`. Separately, all six candidate hashes
are identical across the three context-only and three packed sixteen-lane
long runs. That agreement checks the readback coverage of packed storage,
not equivalence between the sixteen- and four-lane predictor trajectories.

The result is recorded in `packed-encoder-production-long-benchmark.log` and
uses ten agents in the synthetic rolling-terrain scene on RADV Raphael with
Mesa 26.0.8. The six-hash readback coverage and FP32 behavioral limitations
described above still apply. **The 10× whole-simulation target remains unmet.**

```sh
XAGENT_BRAIN_PACKED_ENCODER=1 XAGENT_BRAIN_COOPERATIVE_WHITENING=1 \
XAGENT_BRAIN_DENSE_PREFETCH=1 XAGENT_BRAIN_PREDICTOR_LANES=16 \
XAGENT_BRAIN_CONTEXT_GATHER=1 XAGENT_VISION_OBJECT_QUERIES=1 \
XAGENT_VISION_PARALLEL_SCENT=1 XAGENT_VISION_AGENT_MASKS=1 \
cargo run --release -p xagent-brain --example vision_performance -- \
  --ticks 100000 --repeats 3 --precision fp32
```

The serial child explicitly clears the packed flag along with the other
optional optimizations. Omit `XAGENT_BRAIN_PREDICTOR_LANES=16` and
`--precision fp32` to reproduce the hash-matching configuration.

### Packed row-major predictor

A separate test-only predictor groups four adjacent input weights in each
`vec4<f32>`, retaining row-major storage and the current sixteen-lane FP32
reduction tree. Each invocation owns a complete vector, applies the original
gradient and weight clamps, and accumulates four original lane sequences.
With 256 active invocations, 64 outputs share each tile and predictor barriers
fall from 48 to four. The 128-active variant covers 32 outputs per tile and
uses eight barriers; all 256 workgroup invocations still reach every barrier.

Both variants match all thirteen decoded buffers through 100 cycles across
8×6 and 9×7 fields, including death, whitening refresh, inactive agents and
replay. An independent probe uses the actual scalar and packed shader bodies:
all 7,680 raw outputs satisfy FP32 forward-error bounds, all corresponding raw
outputs match bitwise, and both packed trained matrices match scalar GPU
updates bitwise. This checks the current sixteen-lane reference, whose
association already differs from the original four-lane predictor.

Five rotated three-arm, 100-cycle trials after 256 warmup cycles measure
0.052307147 s for scalar global credit, 0.051685686 s for packed256
(**1.012×**) and 0.051968795 s for packed128 (**1.007×**). Matrix import and
decoded export are outside timing; no public host-lifecycle integration or
scalar mirroring is included. The small gains do not justify promotion, so
both variants remain test-only.

The variants enlarge existing dense scratch from 256 scalars to 1,024 or 512;
they add no workgroup resource slot. At the default 24×24 cortex retina, total
declared main-kernel workgroup storage is 17,140 bytes for packed256 and
15,092 bytes for packed128. The former exceeds the device's currently
requested 16,384-byte limit; even conservative per-variable alignment keeps
packed128 below that limit. The measured fixtures use raw vision, not cortex.

### Latest production profile and verification

After 256 warmup cycles, the 24-cycle production profile measures 474.658 µs
per cycle in the ordinary dispatch recorder and 501.044 µs with timestamps.
Timestamp stage measurements total 480.178 µs: claim 34.282 µs (7.14%), main
kernel and brain 294.100 µs (61.25%), global update with encoder credit
108.868 µs (22.67%), and vision 42.928 µs (8.94%). These are separately
measured wall and GPU timings, not interchangeable totals. Checkpoint restore
invalidates the cache, so wall timing includes the cold import; stage queries
start after the copy. The instrumented replay matches all thirteen captured
buffers. `cycle-profile-packed-context.log` records this ten-agent local-GPU
profile; main brain work and global encoder credit remain the largest costs.

Final verification passes 99 normal brain tests, 284 sandbox tests, the focused
GPU checks described above, `cargo fmt --all -- --check`, and workspace Clippy
with all targets and warnings denied. The sandbox total includes 141 library,
21 binary, two guard, 118 integration and two sensory tests. The resource-limit
fallback is additionally checked through the actual constructor. These checks
support the stated local results; they establish neither universal GPU
performance nor long-run behavioral equivalence of FP32 reassociation.


### Suppressing unchanged packed encoder stores

`XAGENT_BRAIN_SKIP_UNCHANGED_ENCODER_STORES=1` optionally checks the four
updated words after the original credit gates, FP32 arithmetic and clamps.
When their `u32` bit representations all equal the loaded vector, the shader
omits both the private vector store and the four public scalar stores.
Integer comparison distinguishes signed zeros; there is no rounding-gap
estimate or metadata cache. The existing import and mirrored-write lifecycle
maintains equality of both destinations before each update. This option is
default-off, only affects an already-active packed encoder, and does not
select hardware or enable packing by itself. Scalar fallback remains available.

Snapshot diagnostics observe three independently seeded brain trajectories
at cycles 256 and 1,000, with sixteen consecutive cycles per window. They read
the actual GPU matrices and the after-cycle decision credits used by global
credit, exclude inactive/death transitions, and check private/public matrix
agreement plus exact thirteen-buffer replay. No shader counters enter timing.
Of attempted vectors, **10.803%** retain all four words in the earlier window
and **70.748%** in the later window; individual later seeds range from 65.980%
to 75.665%. Credit-disabled vectors are counted separately. These are measured
opportunities, not a prediction of bandwidth or speedup.

The canonical production helper passes the mixed threshold/clamp fixture,
signed zeros, subnormal operands, tiny active updates, inactive agents and
odd-sized raw grids with partial credit workgroup tiles. Complete evolving
runs across 8×6 and 9×7 fields match all
thirteen captured buffers and the private matrix mirror through 100 cycles,
including death and whitening refresh, in two replays. Five alternating
100-cycle timing pairs with the same production recorder measure:

| Warmup cycles | Ordinary stores | Unchanged stores skipped | Whole-cycle ratio |
|---|---:|---:|---:|
| 256 | 0.046634604 s | 0.046738188 s | 0.997784× |
| 1,000 | 0.046165098 s | 0.043390527 s | 1.063944× |

Both arms include their checkpoint-triggered cold import in timing; state
readbacks are outside it. The logs are `packed-store-canonical-validation.log`
and `packed-store-production-lifecycle.log`. The actual constructor also
passes the existing eighteen raw/cortical lifecycle scenarios and resource
fallback test with suppression enabled, against an independent scalar-credit
control; cortical checks use one-cycle submissions.

Three alternating 100,000-tick pairs against the original serial configuration
measure **14.893836464/4.588355608 s (3.246×)**, with all six hashes repeatable
within each arm. All six candidate hashes also match all three earlier packed
long runs with suppression disabled. The log is
`packed-store-production-long-benchmark.log`; add the store-suppression flag
to the packed benchmark command above to reproduce it. These whole-simulation
measurements retain the earlier synthetic-scene and readback-coverage limits.
The ratio to the original serial arm is not an isolated estimate of the new
store optimization, because timings can vary between measurement sessions.

A separate three-pair, alternating on/off comparison keeps every other option
identical and uses the example's `--arm parallel --ticks 100000` mode. It
measures **4.709516207/4.495535372 s (1.047599×)**, with all six hashes equal
across all six runs (`packed-store-toggle-long-benchmark.log`). This directly
measures the store optimization within the existing optimized simulation.
The option is off for the baseline and on for the candidate; both retain
packed weights, context gathering, cooperative whitening, prefetch eight,
sixteen predictor lanes, cached object queries, agent masks and parallel scent.

Keeping the original four-lane predictor, one 100,000-tick pair measures
**14.698164957/5.175427695 s (2.840×)** and matches all six original public
hashes (`packed-store-exact-long-benchmark.log`). This is one pair and does not
establish repetition. No isolated ratio is multiplied into a whole-simulation
claim.

Verification with suppression enabled passes 99 normal brain tests in debug
mode, all 284 sandbox tests, the seven focused release GPU tests described
above, formatting checks and workspace Clippy with all targets and warnings
denied. The ordinary brain suite uses debug mode because its existing
`zero_expected_panics_in_debug` test requires debug assertions; an initial
release run of that suite failed only that expected-panic test before the
debug rerun passed. No test assertion was weakened. Sandbox verification is
recorded in `sandbox-packed-store-tests.log`, including all 118 integration
tests and both sensory tests.

### Packed main-kernel section profile

`XAGENT_SECTIONS_PACKED_CONTEXT=1` makes the section diagnostic compose the
actual packed encoder, gathered context and global-credit main, including the
same vision-mask overrides in its global pipeline. Checkpoint restoration
invalidates the private cache; imports precede timestamps. Claim and main use
the production private bind group. Runtime-uniform prefix guards allow all
stops to use one compiled shader, and every prefix starts from a restored
checkpoint without feeding an incomplete state into a later cycle.

With prefetch eight and sixteen predictor lanes, after 256 cycles the
unguarded main takes **276.080 µs**, versus **279.480 µs** with all guards
inactive (1.0123× overhead); their complete cycles match all thirteen buffers.
The larger consecutive prefix differences are encoder 34.60 µs, predictor
training/dot 40.68 µs, recalled context 30.36 µs, motor/sensory/telemetry
32.20 µs, and memory reinforcement 32.76 µs. Recall scoring and selection add
15.12 and 14.16 µs. These are single-checkpoint medians, not independently
additive steady-state measurements: small negative differences reveal timing
noise. Encoder credit itself runs in global and is excluded from these main
sections.

At cycle 260, scheduled whitening refresh contributes approximately 343.60 µs;
unguarded/guarded full main times are 623.760/623.320 µs and again match all
thirteen buffers. That refresh cost is intermittent and must not be charged
to every cycle. `brain-sections-packed-context.log` records five rotated
measurements per prefix at both checkpoints. The earlier 24-cycle whole-stage
profile averages over multiple cycles and remains a separate measurement.

### Further structural opportunity and its limits

Raw vision has a small color palette and constant alpha, but sensory
adaptation subtracts an independent running mean from each visual feature.
Identical current colors therefore do not imply identical adapted inputs.
The eight-channel whitening calculation belongs to a separate action pathway
after encoding and cannot factor the full dense encoder. Even aggregating
48 identical alpha histories could remove at most 47 of the 267 default
encoder terms before accounting for weight maintenance; arbitrary host-written
means also invalidate that shared-history assumption.

A different candidate would cache the visual projection and update it only
from changed raw inputs. In real arithmetic, if `v` is the adapted visual
vector, `delta_r` is the change in raw vision, and `beta = 0.99`, then
`v_next = beta*v + delta_r`. With unclamped rank-one encoder credit
`W_next = W + v*c^T`, a cached visual projection obeys
`a_next = beta*a + beta*c*dot(v,v) + transpose(W_next)*delta_r`.
Combining sparse raw changes with deferred weight updates could avoid the
dense base-matrix query retained by the earlier deferred encoder experiment.

This identity is not yet an FP32 implementation: actual EMA rounding adds a
generally dense residual, while per-weight rounding and clamping invalidate
the ideal rank-one update. A useful rejection test would capture actual
published features and weight matrices across early/mature, stationary/moving
windows, then measure raw-change sparsity and projection residuals against the
existing raw-dot error bounds over one, eight and sixteen cycles. It must also
establish a viable rebase frequency and snapshot-only public materialization.
No acceleration is claimed for this unimplemented candidate.

The measured encoder section is only about 35 µs, and even eliminating the
entire earlier 294.10 µs main stage leaves 186.08 µs of claim/global/vision.
Consequently an encoder-only redesign cannot establish 10× whole-simulation
acceleration on this workload. Independent global work, such as food-grid
rebuilds when food availability and positions are unchanged, needs its own
opportunity and timing measurements. **The requested 10× target remains unmet.**


The remaining serial Jacobi rotations in whitening are another bounded
candidate: eight lanes could update independent rows in the column phase,
then independent columns in the row phase, preserving the original rotation
and sweep order. This differs from the current cooperative output
reconstruction. It would require up to 1,008 rotation barriers per refresh,
so the 343.6 µs refresh measurement does not establish a gain; even eliminating
that surcharge completely would save only about 17.2 µs per cycle under
steady once-per-twenty-cycle refreshes.

Food-grid reuse has a stricter exactness condition than unchanged positions:
availability and grid-cell membership must also be unchanged, and no cell
may exceed its sixteen retained slots. Overflow must retain the ordinary
rebuild because atomic insertion can choose different retained IDs even for
identical inputs. An eligible cached path must preserve unused slot bytes,
claim resets, respawn timer/RNG/insertion order, agent grids and collisions;
host world uploads and alternate execution routes need invalidation. A
snapshot-only opportunity measurement should verify full grid-byte equality
before a conditional-rebuild prototype, whose savings could overlap work
already hidden by concurrent encoder credit.
