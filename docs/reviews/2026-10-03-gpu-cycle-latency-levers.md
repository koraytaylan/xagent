# Investigation: where a simulation cycle's GPU time goes, and how to cut it without changing results

**Date:** 2026-10-03
**Reviewer:** Claude (Claude Code)
**Base:** `33a7afaa` on `develop`
**Trigger:** find ways to make the GPU computation several times faster without damaging accuracy.

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
