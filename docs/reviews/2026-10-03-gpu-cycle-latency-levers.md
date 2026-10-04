# Investigation: where a simulation cycle's GPU time goes, and how to cut it without changing results

**Date:** 2026-10-03
**Reviewer:** Claude (Claude Code)
**Base:** `33a7afaa` on `develop`
**Trigger:** find ways to make the GPU computation several times faster without damaging accuracy.

**Follow-up, 2026-10-04:** the [latest production experiments](#reducing-the-main-workgroup-to-128-threads)
measure 3.887× whole-simulation acceleration with optional FP32 reassociation,
or an earlier 2.840× with matching original public hashes, for 10 agents on a Raphael integrated
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

### Global-world attribution beside packed credit

`global-world-profile-final.log` isolates the original world workgroup within
the production combined global shader. Seven alternating trials per arm each
restore the same warmed checkpoint, import the private encoder cache, and run
one claim/main/global/vision cycle. GPU timestamps surround only global.
The full arm dispatches world plus packed-credit groups; the measurement-only
arm dispatches workgroup zero of the same compiled shader.

| Warmup cycles | Full global | World group only | Difference |
|---|---:|---:|---:|
| 256 | 108.200 µs | 33.680 µs | 74.520 µs |
| 1,000 | 91.560 µs | 33.480 µs | 58.080 µs |

The full recorder matches all thirteen production buffers. For the world-only
arm, the expected state replaces only the deliberately omitted encoder
matrices with their initial values; every other word remains subject to exact
comparison, and the private/scalar mirror also matches. No omitted-credit
state is fed into a subsequent cycle. The differences are **non-additive
attribution**, not isolated credit costs or achievable speedups: removing
credit changes how the remaining work is scheduled. They show that the
ordinary world work alone still takes about 33.5 µs on this adapter.

### Cooperative Jacobi rotation prototype

The remaining serial whitening rotations now have a test-only cooperative
implementation. Eight lanes update independent covariance rows, then columns,
with barriers between dependent phases; the eigenvector column update shares
the first phase. Sweep and rotation order, scalar angle calculation, early
exits and ordered output reconstruction are retained. Uniform refresh and
sweep decisions keep ordinary and cortical cycles out of the rotation-loop
barriers. Existing scratch is reused without another binding or dispatch.

`cooperative-jacobi-validation.log` records four passing GPU tests. The 8×6
and 9×7 raw fields match all thirteen buffers and the private encoder mirror
through 100 cycles, death/refresh boundaries and two replays. Ten covariance
fixtures exercise scheduled and skipped refreshes, and four bounded cortical
cycles verify that raw whitening remains skipped. A separate CPU composition
test passes in `jacobi-composition.log`, checking that other helper functions
and scratch declarations survive the source substitution.

Five alternating pairs after 256 warmup cycles measure:

| Case | Current production source | Cooperative rotations | Ratio |
|---|---:|---:|---:|
| 100 evolving cycles | 0.046495265 s | 0.045858546 s | 1.013884× |
| One forced refresh cycle | 0.000887237 s | 0.000742767 s | 1.194502× |
| One ordinary cycle | 0.000612313 s | 0.000608566 s | 1.006157× |

These are complete-cycle wall times, including the restored checkpoint's cold
import and submission overhead; the single-cycle rows do not measure the
whitening function alone. Every timed result matches all thirteen buffers.
The intermittent refresh improvement yields only a small evolving-cycle gain,
so this prototype remains unpromoted.

### Food-grid reuse prototype

The test-only food-grid cache records available, in-bounds cell membership for
104 items in 108 words of unused private prediction scratch. It skips only
food-count clearing, food insertion and food sorting when that membership is
unchanged and the previous grid did not overflow. Agent grids, claim resets,
collisions and the original respawn routine always execute. Timer expiry
forces rebuilding both that cycle and the next, preserving the original
pre-respawn build followed by respawn insertion. Overflow always retains the
ordinary rebuild; its atomic winner selection remains an existing limitation
on portable exactness claims.

The three GPU tests in `food-grid-cache-validation.log` pass all-thirteen-buffer
comparisons, including unused stale grid slots and the private matrix mirror.
Both raw fields run 100 cycles with death/refresh and replay. Twelve explicit
transition steps cover consumption, timer initialization/decrement/expiry,
same-cell and cross-cell host moves, an actual food claim, seventeen-item
overflow and recovery. The test asserts when reuse must and must not occur;
checkpoint restoration and arm changes invalidate the private metadata.

After 1,000 warmup cycles, five alternating 100-cycle pairs measure
**0.043342417/0.042987244 s (1.008262×)** with all thirteen buffers equal.
Both arms include cold imports. This small local gain is unpromoted, and the
prototype does not establish a complete production invalidation lifecycle
for arbitrary host world uploads or alternate dispatch routes. Its gain
cannot be multiplied by the world-only attribution above.

### Raw visual events and accumulated projection residuals

Raw vision has a small color palette and constant alpha, but sensory
adaptation subtracts an independent running mean from each visual feature.
Identical current colors therefore do not imply identical adapted inputs.
The eight-channel whitening calculation belongs to a separate action pathway
after encoding and cannot factor the full dense encoder. Even aggregating
48 identical alpha histories could remove at most 47 of the 267 default
encoder terms before accounting for weight maintenance; arbitrary host-written
means also invalidate that shared-history assumption.

Two snapshot-only diagnostics now measure the proposed visual-projection
recurrence on actual moving production trajectories. They use brain seeds
42, 314 and 2026, ten agents, and sixteen consecutive transitions after 256
and 1,000 warmup cycles. Raw sensory input is captured **before** each cycle;
actual adapted features are read from private published storage **after** it.
Matrices used by encoding come from the before-cycle state, prior to that
cycle's global credit. Every sampled cycle passes exact thirteen-buffer replay
and private/scalar matrix checks. Death/reset or inactive transitions are
excluded explicitly; none occurs in these sampled windows. No stationary
fixture, changed simulation arithmetic or timing counters are included.

`visual-event-diagnostics.log` reports:

| Warmup cycles | Raw RGBA/depth rays unchanged | Adapted RGBA/depth rays unchanged | Largest one-step absolute projection envelope / fresh-dot budget |
|---|---:|---:|---:|
| 256 | 22,918 / 23,040 (99.470%) | 0 / 23,040 | 1.068% |
| 1,000 | 22,960 / 23,040 (99.653%) | 0 / 23,040 | 6.711% |

Thus sparse raw events coexist with changing adapted rays. Neither signed
one-step projection errors nor their absolute term envelopes exceed the
existing fresh-dot forward budget across 122,880 output rows. This first
screen uses the current matrix and excludes weight-update reuse.

The second diagnostic, `raw-event-recurrence.log`, includes stored weight
updates and resets its cached FP64 projection every one, eight or sixteen
cycles. Its corrected clamp-classification rerun in
`raw-event-recurrence-final.log` passes with the same measured results. With visual features `v`, raw change `delta_r`, actual adaptation rate
`rho` and `beta = 1 - rho`, the ideal update is
`a_next = beta*(a + c*dot(v,v)) + transpose(W_next)*delta_r`.
Here the norm contains **visual rows only**; nonvisual terms and bias remain
fresh. `W_next` is the actual matrix before the next cycle's credit. The
learning rate is read from the actual GPU uniform, and `c` is the FP64 product
of that rate, the previous cycle's stored credit and the source-derived credit
scale, with the original credit threshold applied.

The diagnostic measures adaptation residual
`epsilon = v_next - (beta*v + delta_r)` and weight residual
`R = W_next - W - v*c^T`, including coefficient rounding, per-weight rounding,
discarded increments and ideal clamp residuals. The accumulated difference between fresh and cached projections decomposes as
`e_next = beta*e + beta*transpose(R)*v + transpose(W_next)*epsilon`.
This uses actual published features and stored matrices at each step; it is
not an evolving simulation driven by the proposed cached result.

For each window length, 122,880 output-row prefixes are checked across both
warmup ages. No measured error or absolute residual envelope exceeds the
existing full-feature-plus-bias fresh-dot budget. Maxima over all prefixes
are:

| Warmup cycles | Rebase window | FP64 cached error / budget | Absolute residual envelope / budget |
|---|---:|---:|---:|
| 256 | 1 cycle | 0.277% | 1.133% |
| 256 | 8 cycles | 0.753% | 6.003% |
| 256 | 16 cycles | 0.900% | 11.136% |
| 1,000 | 1 cycle | 1.154% | 6.711% |
| 1,000 | 8 cycles | 3.975% | 46.834% |
| 1,000 | 16 cycles | 4.228% | 79.627% |

The largest sixteen-cycle FP64 discrepancy is 1.548e-7. Per-weight residuals
reach 1.863e-9. Of nonzero ideal visual-weight increments, 16.235% in the early
window and 81.470% in the mature window leave the stored word unchanged.
The diagnostic distinguishes these from below-threshold credit outputs
(1,628 and 1,336 respectively). **No clamp events are observed:** both ideal
clamp-crossing counts and observed stored-at-limit counts are zero. These
trajectories therefore provide no empirical saturation-boundary coverage.

This is a numerical opportunity screen, **not GPU accuracy acceptance**.
It omits candidate FP32 recurrence, norm, coefficient and projection arithmetic,
uses the existing conservative round-to-nearest/FTZ dot budget only as a
comparison scale, and measures no altered decisions or long-run behavior.
The mature sixteen-cycle envelope already consumes much more of that scale
than the one-cycle envelope. A GPU candidate still needs raw arithmetic
probes, adversarial saturation/reset cases, behavioral validation, a viable
rebase policy and a measured complete-cycle benefit. No acceleration is
claimed for this ideal multi-cycle recurrence.

The encoder section remains only about 35 µs, so an encoder-only redesign
cannot establish 10× whole-simulation acceleration on this workload. The
cooperative Jacobi and food-grid experiments are test-only, and neither is
included in the production headline. **The measured whole-simulation result
at this stage was 3.246×; the requested 10× target remains unmet.**


### Fresh next-input projection: raw numerical validation

A separate test-only candidate computes fresh visual projection partials beside
packed encoder credit, using the actual updated, clamped weights. It predicts
the next adapted visual input from current raw sensory data and the stored
mean. The next encoder uses those partials only when every predicted visual
word matches the actual input bit for bit and the cached death generation
still matches. Otherwise it executes the original full packed encoder.
Nonvisual features and bias remain fresh. This does not use the ideal
multi-cycle EMA or rank-one recurrence described above.

The independent hardware probe in `fresh-projection-raw-dots-final.log` invokes the
candidate's actual credit and encoder helper bodies. Both encoder routes
publish their pre-tanh value in the timing source itself; the probe uses that
publication rather than duplicating the arithmetic or inverting tanh. The
original scalar credit shader independently checks trained matrices, while
the private packed matrix must equal its public scalar mirror. Each candidate
raw output and its fresh-encoder reference are checked against the existing
FP64 full-feature-plus-bias oracle and unchanged round-to-nearest/FTZ forward
budget.

Twelve fixtures across 8×6 and 9×7 raw fields cover 267 and 342 features,
including partial eight-feature tiles, seeded values, cancellation, wide
exponents, tiny normals, subnormals, sparse tails, bias-only and signed-zero
inputs. Mixed credits exercise values below, at and above the update
threshold, both clamp endpoints, and disabled components whose stored weights
lie outside the clamp. A changed-input case drives a moderate predicted
feature nearly to zero and must use the full fresh encoder.

The passing test records the following candidate evaluations; the fresh
reference is independently bounded for each of these same inputs:

| Route | Raw outputs checked | Largest error / fresh-dot bound | Bit differences from fresh encoder |
|---|---:|---:|---:|
| Cached projection | 5,888 | 1.5574% | 3,085 |
| Full-encoder fallback | 12,288 | 2.6387% | 0 |
| **Total** | **18,176** | **2.6387%** | **3,085** |

The cached count comprises 3,072 matching-input outputs and 2,816 live-agent
outputs with an inactive agent present. The fallback count comprises 3,072
outputs each for changed input, changed death generation, invalid cache and
host reset followed by import. These are repeated evaluations of the fixture
families, not 18,176 independently sampled trajectories. All trained matrices
match the scalar-credit shader bitwise, including the inactive agent; its
256 omitted encoder outputs across the two fields retain their sentinel bits
and are excluded from numerical counts.

The route checks are falsifiable. Deliberately corrupting a cached partial
changes a matching-input result, proving that the fast path consumes it.
Changed input, death generation and invalid-cache cases then reject the same
corrupt partials and match the fresh encoder bitwise. A real seeded host reset
starts with a freshly repopulated projection: every agent must report valid
metadata and actual cached use before its partial is corrupted. The test then
explicitly clears the detached cache's metadata header and invalidates/imports
its matrices after reset; matching input must take the fresh fallback. This
checks the prototype's manual reset protocol, not a production constructor
API. The strengthened rerun retains exactly the original 18,176 numerical
results and aggregate bounds. Deliberately corrupted results and the repeated
warm-cache preparation are excluded from that table.

These results establish bounded raw arithmetic for the observed helper
compilation and fixtures. Cached reductions do change FP32 bits; the test does
not establish complete-trajectory equivalence, cache hit frequency, a
production host-mutation lifecycle, or a whole-simulation speedup. Timing and
evolving-state validation are separate from this numerical gate.


### Fresh next-input projection: cycle coverage and timing

`fresh-projection-trajectory-final.log` passes 100-cycle checks for both raw
fields, with death/refresh boundaries, an inactive agent, private/scalar
matrix agreement and exact thirteen-buffer replay within each arm. Comparisons
between the original and cached arms report FP32 differences and enforce
finite/range/state invariants; they do not require bit equality. After 100
cycles, maximum absolute differences across reported numeric fields are
1.228e-5 for 8×6 and 3.254e-5 for 9×7, with zero reported discrete-field
changes. These two fixtures use one seed and do not establish a behavioral
distribution or a pipeline-wide error bound.

To make cached-path coverage reliable, both arms receive the same held sky
visual input before cycles 40 and 41. All world and vision dispatches still
execute. The fixture records 11 hits/889 misses for 8×6 and 9 hits/891 misses
for 9×7; it explicitly requires both routes. These deliberately held-input
checks are distinct from the unmodified moving scene used for timing.

In that mature scene, after 1,000 warmup cycles, five alternating 100-cycle
pairs measure **0.045574866/0.044523232 s (1.023620×)**. The candidate records
842 hits and 158 misses, an 84.2% hit rate over those 1,000 agent cycles.
Each arm repeats all thirteen buffers exactly; cross-arm FP32 drift remains
reported separately. The timing preflight observes zero discrete changes
and a maximum numeric-field difference of 6.104e-5. State readbacks are outside
timing, while cold cache import, raw-value publication and hit/miss counters
are included. The prototype appends 159,040 bytes of private storage for this
ten-agent field without adding a binding or dispatch. The log is
`fresh-projection-timing.log`.

This small local whole-cycle gain does not justify production promotion.
Neither the held-input fixtures nor the single mature timing scene establishes
long-run behavioral equivalence or a general cache-hit rate. The prototype
remains test-only, and the production headline at this stage was **3.246×**, with the
**10× whole-simulation target unmet**.

### Verification of the world and visual-reuse diagnostics

The complete change passes 102 normal brain tests in debug mode, fifteen
focused release GPU tests, all 284 sandbox tests run serially, formatting,
and workspace Clippy with all targets and warnings denied. The existing
packed encoder and predictor raw-dot probes also pass after extracting their
unchanged FP64 reference/bound calculation for the snapshot diagnostics.
Hardware-dependent experiments remain ignored in the normal test suite and
are run explicitly. Logs include `brain-fresh-projection-debug.log`,
`clippy-fresh-projection.log`, `sandbox-visual-events-serial-tests.log`,
`packed-encoder-dot-bound-refactor.log` and
`packed-predictor-dot-bound-refactor.log`.

The initial parallel sandbox run failed two worker startup/event deadlines;
the serial rerun passes the same assertions, including all 118 integration
tests in 535.90 seconds. The initial fresh-projection trajectory test also
found no cache hits in the moving 9×7 startup fixture and failed its route
coverage assertion. The held-input case above makes that coverage deterministic
without removing the assertion or weakening numerical checks. These changes
add diagnostics and test-only candidates; no new runtime option or default
is promoted by this verification.

### Reducing the main workgroup to 128 threads

`XAGENT_BRAIN_MAIN_THREADS=128` now selects a smaller main workgroup when
packed encoding, the sixteen-lane predictor and eight-lane context gather
are present. It defaults off, enables no other option, and does not inspect
GPU vendor or device names. Unsupported source compositions and packed-cache
resource fallbacks retain the 256-thread main. Existing execution-mode,
vision-stride, partial-brain and skipped-global fallbacks are unchanged.

The packed encoder already performs its arithmetic on 128 invocations, so
its weight ownership and four-part reduction stay intact. The predictor
processes eight outputs per tile instead of sixteen, keeping each output's
sixteen-lane tree. Context tiles shrink similarly. Each reinforcement thread
computes both original ascending even/odd chains before their existing sum.
Cortex normalization initializes all 256 logical partials and keeps its
original tree, including the leading positive-zero addition. A cortex output
count above 128 components rejects the transformation. Claim, global and vision
keep 256 threads, shared allocations are unchanged, and there is no additional
dispatch, binding or per-cycle host work.

The production constructor passes exact thirteen-buffer comparisons for raw
8×6 and 9×7 fields through 100 cycles, death and refresh boundaries, an inactive
agent, two candidate replays and the private/scalar encoder mirror. A bounded
three-agent cortex case passes three individually submitted cycles with two
deaths and two replays. A predictor4/no-context case explicitly verifies the
256-thread fallback. CPU composition tests cover missing/duplicate source
contracts, repeated application, predictor4/8/32, context absent/16, and
supported sources with scalar or cooperative whitening.

The existing production lifecycle suite also passes with the new option
actually active: eighteen raw/cortex cases cover host writes, reset, queued
readback snapshots and departures from/resumption of packed execution. Its
workgroup-resource case explicitly verifies scalar credit and a 256-thread
main when packing is unavailable. Logs are
`main128-memory-production-validation.log` and
`main128-production-lifecycle.log`.

Five alternating complete 100-cycle pairs after 1,000 warmup cycles measure
**0.045049295/0.036604195 s (1.230714×)** against the previous optimized stack,
with all thirteen buffers and the private mirror equal. Cold cache import
is included; state readbacks are outside timing.

Three alternating **100,000-tick** production pairs against the original
default configuration (serial vision and default brain/vision scheduling)
measure **15.035975847/3.868469985 s (3.887×)**. Each arm's six
public hashes repeat exactly; cross-arm hashes differ because the candidate
includes the already validated predictor16 reassociation. A separate direct
256/128 toggle comparison, also three alternating 100,000-tick pairs, measures
**4.672434687/3.859867571 s (1.210517×)**, with all six public hashes identical
across all six runs. These are measured combined results, not products of
isolated speedups. Logs are `main128-production-long-benchmark.log` and
`main128-toggle-long-benchmark.log`.

Both long experiments use the unchanged synthetic seed42, ten-agent, 8×6
scene with terrain, hazards, food and collisions. Pipeline construction and
final readback are excluded; all simulation dispatches and GPU completion are
timed. These public hashes do not include every internal buffer, which is why
the separate thirteen-buffer gates remain necessary. The implementation uses
portable WGSL, but performance on other GPUs and populations is unmeasured.
**The requested 10× whole-simulation target remains unmet.**

Reproduce the combined long comparison with:

```sh
XAGENT_BRAIN_PACKED_ENCODER=1 \
XAGENT_BRAIN_SKIP_UNCHANGED_ENCODER_STORES=1 \
XAGENT_BRAIN_CONTEXT_GATHER=1 \
XAGENT_BRAIN_COOPERATIVE_WHITENING=1 \
XAGENT_BRAIN_DENSE_PREFETCH=1 \
XAGENT_BRAIN_PREDICTOR_LANES=16 \
XAGENT_BRAIN_MAIN_THREADS=128 \
XAGENT_VISION_AGENT_MASKS=1 \
XAGENT_VISION_OBJECT_QUERIES=1 \
XAGENT_VISION_PARALLEL_SCENT=1 \
cargo run --release -p xagent-brain --example vision_performance -- \
  --ticks 100000 --agents 10 --seed 42 --width 8 --height 6 \
  --vision-stride 1 --execution fused --repeats 3 --precision fp32
```

### Structural alternatives checked beside main128

Three other candidates remain test-only and were measured independently
against the preceding 256-thread optimized stack:

| Candidate | Five-pair reference / candidate time | Complete-cycle speedup |
|---|---:|---:|
| Memory maintenance beside world and encoder credit | 0.044454795 / 0.044467549 s | 0.999713× |
| Grouped per-agent physics followed by cooperative claims | 0.043383995 / 0.043275682 s | 1.002503× |
| Parallel visual whitening rows and covariance cells | 0.048718767 / 0.047911550 s | 1.016848× |

Each sample advances 100 complete cycles; memory and physics use 1,000 warmup
cycles, while visual pathway uses 256. These small or absent gains do not
establish a combined improvement with main128 and are excluded from its
headline.

Memory maintenance retains main's exact norm, slot store and critic replay,
then executes reinforcement, decay, active count and eviction selection in
one additional global workgroup per agent. It reads the exact newly stored
key/norm and skips reinforcement of that fresh slot because main has already
overwritten the old slot's reinforcement and valence. No dispatch is added,
but every global group reserves 2,576 bytes of shared memory. Exact full-state
and mirror checks pass for both raw layouts; direct fixtures verify overwrite,
another reinforced slot, episodic credit, deactivation and first-index eviction
ties. The initial host resource guard incorrectly matched a shared-memory
token in a comment; the corrected guard checks declarations. The passing log
is `main128-memory-production-validation.log`.

Grouped physics gives one invocation to each agent in a 32-thread workgroup,
preserving that agent's original sequence of sub-ticks. A separate dispatch
then runs the original 256-thread food claim. Both timed arms share the same
chunking and cache-import protocol, and the extra dispatch is timed. The
custom recorder is independently compared with production across all thirteen
buffers, two raw layouts, death/refresh, inactive state and two replays.
This fixture has ten agents and one brain stride; it does not validate multiple
physics workgroups or a general production lifecycle. The log is
`packed-physics-validation.log`.

The visual candidate retains scalar hemifield pooling, then computes eight
ordered whitening rows and 64 independent covariance cells cooperatively in
released scratch. An earlier redistribution of the hemifield divisions failed
exact comparisons: the diagnostic located differences in the mean before
whitening, with the whitening matrix itself unchanged. Restoring the original
hemifield function makes all strict checks pass without relaxing assertions:
both raw layouts, 100-cycle trajectories, death/refresh, two replays, direct
old-mean/old-covariance probes at refresh/nonrefresh ticks, and bounded cortex
execution. Logs are `visual-pathway-phase-diagnostic.log`,
`visual-pathway-phase-repair.log` and `visual-pathway-final-validation.log`.

### Remaining cost with the 128-thread production main

The production dispatch profiler uses the actual constructor-selected pipeline
and checks its instrumented recorder against all thirteen mutable buffers.
After 256 warmup cycles, its 24-cycle sample measures 397.149 µs per cycle of
uninstrumented wall time, 415.473 µs instrumented wall time and 393.710 µs
summed GPU stage medians:

| Stage | GPU time | Share |
|---|---:|---:|
| Physics and food claim | 34.173 µs | 8.68% |
| Remaining kernel and brain | 210.357 µs | 53.43% |
| World update and encoder credit | 106.282 µs | 26.99% |
| Vision and other senses | 42.898 µs | 10.90% |

The log is `main128-cycle-profile.log`. This short restored-state profile is
distinct from the long throughput experiment. The earlier internal section
timings above used a 256-thread pipeline and are not attribution of the new
main; the updated 128-thread section profile appears below.

At unchanged costs for the other stages, their measured times sum to
183.353 µs, above the roughly 150 µs cycle budget derived from the long-run
original baseline. This is conditional attribution, not a lower bound after
rescheduling. Further gains need investigation across dispatches as well as
inside the brain.

A source audit identifies a possible four-dispatch ordering: physics/claim;
food settlement, danger and death prefix; brain alongside world; then vision
alongside encoder credit. Brain reads the saved pre-collision position and
physics fields disjoint from the world's position writes; vision reads FOV
and smell genes disjoint from encoder weights. The test-only implementation
and measurements are recorded below; its combined vision/credit pipeline
reserves about 7.9 KB of vision scratch in every credit workgroup.

### Verification of the 128-thread production option

The final change passes all 108 normal brain tests, fifteen focused release
GPU tests, the production dispatch-profile parity check, all 284 sandbox tests
run serially, formatting and workspace Clippy across all targets with warnings
denied. Long benchmarks additionally verify per-arm public-hash repeatability
and exact equality across the direct 256/128 toggle. Hardware experiments remain
ignored in the ordinary suite and were run explicitly. Final routine-check logs
are `main128-final-brain-debug.log`, `main128-clippy.log` and
`main128-sandbox-serial-tests.log`; the GPU and timing logs are identified above.
Only the guarded 128-thread main gains a production option in this follow-up;
memory offload, grouped physics and cooperative visual-pathway updates remain
test-only.

### Physics-local state and packed-credit row batching

Two further test-only candidates use the actual 128-thread production main
as their reference, retaining its private encoder cache and production
recorder. Five alternating pairs each time 100 complete cycles after 1,000
warmup cycles, including cache import and GPU completion but excluding state
readbacks:

| Candidate | Reference / candidate median | Complete-cycle speedup |
|---|---:|---:|
| Function-local physics state across sub-ticks | 0.035875523 / 0.034510394 s | 1.039557× |
| Two encoder-credit rows per invocation | 0.035574912 / 0.035798660 s | 0.993750× |
| Four encoder-credit rows per invocation | 0.035598646 / 0.035957045 s | 0.990033× |
| Eight encoder-credit rows per invocation | 0.035472460 / 0.036078502 s | 0.983202× |

The physics candidate redirects the canonical per-tick field accesses into
function storage, loads the referenced fields before the sub-tick loop, and
publishes only fields the canonical function writes before the existing claim
barriers. Initially inactive agents perform no writes; a death during the loop
keeps the original alive gate and death tick. It adds no persistent cache,
shared allocation or dispatch. Exact thirteen-buffer, inactive-agent and
private-mirror checks pass for 8×6 and 9×7 raw fields, brain strides 1 and 10,
100 cycles through death/refresh and two candidate replays. Keeping the
expressions unchanged alone does not guarantee identical backend rounding;
the strict comparisons establish parity on the tested GPU. The small timing
gain is preliminary and has no long-run or other-device confirmation. Logs
are `physics-cache-composition.log` and `physics-cache-validation.log`.

Row batching retains each vec4 weight's update arithmetic, gates, clamps and
private/public stores, but reuses its credit values across 2, 4 or 8 feature
rows. The unchanged-store early return exits a single-row helper, so an
unchanged first row cannot skip a later update. For ten agents with 267
features, global workgroups fall from 341 to 171, 91 or 51 without changing
shared storage or dispatch count. All three widths pass exact thirteen-buffer
and private-mirror checks across both raw layouts, death/refresh and two
replays; direct fixtures cover thresholds, clamps, inactive agents, partial
tails, disabled vectors and an unchanged row followed by a changed row.
Despite fewer workgroups, every measured width is slightly slower, so none
is promoted. Logs are `packed-credit-batch-composition.log` and
`packed-credit-batch-validation.log`.

These independent comparisons are excluded from the 3.887× production
headline and do not establish a combined speedup.

### Smaller main and overlapped four-dispatch schedule

The 64-thread main processes the original 128 logical owners in two halves,
completing both halves of each independent block before its original
barrier. Packed encoder and reinforcement scratch retain their complete
logical ownership and arithmetic chains. Predictor and context tiles shrink,
while reductions and recall tie breakers keep their original logical order.
The candidate adds neither dispatches nor shared storage and explicitly
rejects cortex layouts. CPU checks validate both subgroup and fallback
compositions; GPU checks exercise the native subgroup path, not a forced
fallback path.

The four-dispatch candidate retains the original claim, then runs a separate
food-settlement/danger/death prefix, brain alongside a 128-thread world
update, and finally vision alongside packed encoder credit. Shared storage
is 9,200 bytes for brain/world at 8×6 and 9,504 bytes at 9×7, and 7,872 bytes
for vision/credit in both layouts. World loops retain their coverage and
phase order with 128 threads. Untimed preflight checks both grid capacities
after every cycle; parity is not claimed for overflowing grids. The custom
reference recorder is independently compared with production, and both
timed schedules use identical chunking and import behavior.

Both candidates pass exact thirteen-buffer, private-mirror and inactive-agent
comparisons for two raw layouts through 100 cycles, two forced deaths,
refresh boundaries and two replays. Five alternating 100-cycle timing pairs
after 1,000 warmup cycles, with GPU completion and cache import included,
give:

| Candidate | Reference / candidate median | Complete-cycle speedup |
|---|---:|---:|
| 64-thread main versus production128 | 0.035501955 / 0.042057227 s | 0.844134× |
| Brain/world followed by vision/credit overlap | 0.035583398 / 0.034192100 s | 1.040691× |

The smaller main is slower and remains test-only. The overlap gain is small
and does not yet cover general populations, cortex, overflowing grids or
production lifecycle transitions, so it also remains test-only. A second
physics-cache comparison measures **0.035945073/0.034585835 s (1.039300×)**
and passes stronger source-change and forced-death assertions. All six GPU
checks in this batch pass; the log is
`scheduling-candidates-gpu-validation.log`. No product of these ratios is
reported as a measured combined result, and the production headline remains
3.887×.

### Section attribution of the actual 128-thread main

`XAGENT_SECTIONS_MAIN_THREADS=128` now makes the section diagnostic use the
production compact-main transformation for warmup, its unguarded control and
its instrumented shader. It requires packed context and sixteen predictor
lanes, rejects unsupported configurations before constructing a GPU kernel,
and reports the actual width. The default and frozen predictor-width
comparison retain 256 threads.

Five restored single-cycle samples per prefix after 256 warmup cycles give
the following consecutive prefix differences:

| Section | Approximate GPU time |
|---|---:|
| Main before brain | 22.360 µs |
| Encoder | 27.560 µs |
| Recall scoring | 18.040 µs |
| Predictor training and dot product | 34.400 µs |
| Recall context and tanh | 28.520 µs |
| Memory reinforcement | 20.960 µs |

The complete unguarded main takes 198.080 µs and the guarded main 195.720 µs,
with all thirteen buffers equal. At the scheduled-refresh checkpoint after
260 cycles, they take 383.600 and 385.400 µs, again with all thirteen buffers
equal. Prefix differences are diagnostic estimates rather than independent
stage measurements; small negative differences at short sections expose
their timing noise. Cache import happens before the timestamped main, and
global store suppression remains off in this diagnostic, as before, without
changing main arithmetic. The log is `main128-brain-sections.log`.

### Rejected exact reuse of recall pattern loads

A separate prototype computed recall's ascending dot and reinforcement's
even/odd partial sums during one traversal, retaining those partials in
released argmin scratch until reinforcement. It added no allocation or
dispatch. The isolated probe passed bitwise cosine and active-partial
comparisons plus 3,040 FP64 dot-error checks per raw layout, including active
flags immediately below, at and above 0.5.

The full-simulation gate nevertheless failed, including after restoring the
canonical one-dimension-at-a-time recall loop. A diagnostic replayed each
candidate cycle from the exact same baseline prefix and found the first
difference at cycle three: nine prediction words for agent nine and their
nine saved previous-prediction copies differed; the other eleven buffers
matched. One reported prediction differed by about 2.98e-8. Source review
found no intervening mutation of the cached inputs or omitted barrier, but
compiler-dependent contraction remains a hypothesis, not an established
cause. The isolated probe therefore does not certify the complete shader.

No speedup or accepted precision result is reported for this candidate; it
is excluded from the committed implementation. Its source and unchanged
strict tests are preserved locally in the task cache for further numerical
investigation. Logs are `recall-dual-sum-validation.log`,
`recall-dual-sum-repair-validation.log` and
`recall-dual-sum-first-difference.log`.

### Audited next option: delayed publication of encoder weights

The private packed encoder could become authoritative between public-state
boundaries, removing the four scalar mirror stores from each updated vec4.
This needs three explicit states: scalar-authoritative, synchronized and
packed-dirty. Scalar fallback dispatches must export first; single-agent host
writes must export all untouched dirty agents before submitting the incoming
write and invalidating the cache. A successful full-population replacement
may discard old packed state, while a failed `try_reset_agents` must preserve
it. Full brain readbacks need an export before their staging copy in queue
order, including request-time asynchronous snapshots; physics snapshots and
telemetry fields outside the encoder matrix need no export.

The old unmirrored experiment excluded boundary copies and used an older
stack, so it establishes neither current end-to-end speed nor these lifecycle
semantics. A new comparison must time final export completion and frequent
snapshots as well as steady simulation. This remains a source-audited option,
not an implemented production change or evidence of reaching 10×.

### Verification of the scheduling experiments

The accepted follow-up contains four test-only candidates and the updated
main-width profiler, with no new production option. All 115 normal brain
tests, nine focused release GPU checks, the section-profiler parity check,
284 serial sandbox tests, formatting and workspace Clippy across all targets
with warnings denied pass. Logs are `scheduling-accepted-brain-tests.log`,
`scheduling-accepted-clippy.log` and
`scheduling-candidates-sandbox-tests.log`, together with the GPU logs named
above. The rejected dual-sum prototype is excluded from these passing
counts and from the committed source. The 10× whole-simulation goal remains
unmet.

### Same-WGSL CPU backend comparison

The existing optimized example was also run through Mesa's CPU Vulkan
implementation, preserving the complete WGSL simulation and its production
128-thread main. `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json` and
`LIBGL_ALWAYS_SOFTWARE=1` select this diagnostic; the logged adapter confirms
`device_type: Cpu` and `backend: Vulkan`. Its minimum subgroup width is eight,
so the existing portable shared-memory recall fallback is used. The normal
GPU run uses its supported subgroup path; no production adapter selection or
device-specific tuning was added.

Three alternating pairs of the same optimized arm, ten agents, seed 42,
8×6 raw vision, 100 warmup ticks and 10,000 timed ticks measure median
**0.387712229 s on the GPU versus 0.834973237 s on the CPU**: the CPU backend
is 2.154× slower on this host. Compilation and readback remain outside the
interval, while simulation completion is included. Each backend repeats its
own six public hashes exactly; cross-backend hashes differ and have not
passed a numerical-accuracy gate. This is a local backend diagnostic, not
evidence about other CPUs or GPUs or validation of interchangeable
trajectories. The log is `lavapipe-main128-paired.log`.

### Packed authority with timed publication boundaries

A test-only candidate removes exactly the four scalar mirror stores from the
canonical packed credit helper; private weight arithmetic and stores, the
production main128 and dispatch recorder remain unchanged. The private
matrix is compared with the reference before publication, then explicit
buffer copies publish it and all thirteen public buffers must match.
Checks pass for both raw layouts through death, refresh, inactive agents
and two replays, on both the GPU and CPU Vulkan implementations.

Five alternating pairs time 100 evolving cycles after 1,000 warmup cycles.
Both arms use identical dispatch-call and completion boundaries; the
candidate's extra export submissions and final completion are inside the
timer. Cold import and state readback are outside both timers. Each export
copies 1,367,040 bytes:

| Export interval | GPU reference / candidate median | GPU speedup | CPU speedup |
|---|---:|---:|---:|
| Every cycle | 0.042525910 / 0.047459992 s | 0.896037× | 0.952956× |
| Every 24 cycles | 0.036163839 / 0.035066540 s | 1.031292× | 1.013739× |
| Final cycle only | 0.035368865 / 0.033975683 s | 1.041005× | 0.996829× |

These small, boundary-dependent gains do not justify promoting this
prototype without the public lifecycle work described above; frequent
publication regresses on both tested backends. The test deliberately permits
only restored/cold arm transitions, not host writes or scalar fallback while
private authority is live. Logs are `authoritative-packed-gpu.log` and
`authoritative_packed_validation-lavapipe.log`.

### Cached coefficients for gathered context

The production main128 gathers sixteen prediction dimensions per tile and
recomputes the ordered similarity total and sixteen context coefficients for
each output. Another test-only candidate calculates these once, retaining
the original blend order, guard conditions and tanh. Seventeen released
packed-encoder scratch words hold the values, with one additional barrier
and no additional allocation or dispatch.

Exact thirteen-buffer, private-mirror and two-replay checks pass for both
raw layouts through death/refresh on both tested backends. An independent
actual-helper probe compares all 128 outputs for ten fixtures, covering
empty/partial/full recall, negative and zero totals, values immediately
below/at/above the 1e-8 guard, zero coefficients and signed-zero inputs;
all public buffers remain unchanged by the probe.

Five alternating 100-cycle pairs after 1,000 warmup cycles measure
**0.035623620/0.035024993 s, or 1.017091×**, on the GPU and **1.014766×**
on the CPU backend, with completion and cold import included. It remains
test-only because the measured whole-simulation gain is small. These
separate backend comparisons establish within-backend candidate/reference
parity, not cross-backend trajectory equality. Logs are
`gathered-context-cache-gpu.log` and
`gathered_context_cache_validation-lavapipe.log`.

### Population and raw-field sensitivity

The existing production configuration was compared with the example's
unoptimized serial arm across additional workloads on the same GPU, using
three alternating seeded pairs, 100 warmup ticks and 10,000 timed ticks:

| Agents | Raw field | Serial median | Optimized median | Whole-simulation speedup |
|---:|---|---:|---:|---:|
| 1 | 8×6 | 0.249748728 s | 0.207184307 s | 1.205× |
| 32 | 8×6 | 6.033873659 s | 2.176847922 s | 2.772× |
| 33 | 8×6 | 6.231415203 s | 2.214188569 s | 2.814× |
| 100 | 8×6 | 20.698095546 s | 7.526145591 s | 2.750× |
| 10 | 12×8 | 1.898621979 s | 0.521320697 s | 3.642× |

Each arm repeats its own six public hashes; cross-arm hashes differ under
the FP32 option. These timings do not independently certify long-run
trajectory equivalence or every private buffer. They show why the ten-agent
3.887× result must not be extrapolated to every workload. Object-based
vision and retained masks disable above 32 agents, while packed credit and
main128 remain selected; there is no measured regression across that
eligibility boundary. Logs are `main128-workload-<agents>-<width>x<height>.log`.

The matrix deliberately stays below 256 agents: the current global
agent-grid, collision and trail phases assign one agent per invocation in a
single 256-thread group. The sandbox's existing 400/1,000-agent sweep rows
therefore cannot establish complete-population simulation performance until
that separate coverage limit is addressed. Default cortex dimensions fit
the production shared-memory guard, but the available cortex checks remain
short correctness fixtures rather than this seeded timing comparison.

### One workgroup per agent for updated visual projections

A third test-only candidate combines encoder credit and the next predicted
visual projection in one 256-thread workgroup per agent, using 128 arithmetic
owners. Each owner updates its original vec4 weight with the canonical
thresholds, FP32 arithmetic, clamp, unchanged-store suppression and scalar
mirror, then accumulates that actual updated weight into its original
stride-four visual sum. Lane zero starts with the original bias. Four sums
per output are cached separately; the next main continues each with fresh
nonvisual inputs before the original left-associated final reduction.
Bitwise equality of every visual input and the death generation gate reuse;
all other cases execute the original packed encoder.

This reduces global workgroups from 341/431 to eleven for the two raw
layouts, without adding a dispatch. Staging each predicted visual input once
uses 960/1,264 bytes of shared memory, and the private cache grows by
25,920 bytes for ten agents. Unconditional storage and workgroup barriers
finish all old-feature reads before those features are replaced with their
predicted values. Main shared memory is unchanged.

The actual shader helpers pass **21,120 raw output checks per backend**
against fresh-dot bits and the unchanged FP64-derived error bound, whose
largest observed error ratio is 0.026388. Independent scalar credit and
private/public matrix comparisons pass. Fixtures include cancellation,
wide exponents, subnormals, update thresholds, clamps, unchanged components,
poisoned partials, changed visual inputs, nonvisual-only changes, death,
inactive agents and explicitly invalidated detached-cache resets. The
nonvisual-only case requires both reuse and an observable output change,
including the 315-feature visual boundary of the odd-sized field.

Both backends also pass exact thirteen-buffer comparisons and repeatability
through 100 cycles of both raw layouts, forced death/refresh and held visual
inputs, plus exact checks of the published predicted inputs. These are
test-managed cache transitions; no production lifecycle integration is
claimed. Five alternating 100-cycle pairs after 1,000 warmup cycles give:

| Backend | Reference / candidate median | Whole-simulation speedup | Reuse hits / misses |
|---|---:|---:|---:|
| GPU | 0.036103516 / 0.035917099 s | 1.005190× | 842 / 158 |
| CPU Vulkan | 0.079506603 / 0.088673932 s | 0.896618× | 921 / 79 |

Completion, cold import and diagnostic raw-value publication are timed;
readbacks are excluded. Reducing dispatch workgroups and a later matrix
read therefore does not establish a useful overall gain for this candidate,
which remains test-only. Logs are `agent-projection-raw.log`,
`agent-projection-state.log`, `agent-projection-timing.log`,
`agent-projection-raw-lavapipe.log` and
`agent-projection-full-lavapipe.log`.

### Remaining architectural limits and verification

A possible smaller follow-up is a 128-thread version of the new global
projection dispatch: all arithmetic owners already fit in 128 lanes, but
both predicted-input copy strides and all six world-loop strides must also
change, with an agent-count guard and grid-capacity checks. Longer world
loops could offset any occupancy gain; it has not been implemented or timed.

An algebraic recurrence could avoid more visual-matrix work, but FP32
weight rounding, clamping and adaptation rounding leave dense residuals
even when raw inputs are unchanged. Any delayed matrix representation must
replay each original update and clamp chronologically when materialized;
adding all pending increments and clamping once changes the result. A useful
prototype needs a cheap conservative error bound and rebase condition that
do not themselves traverse the dense matrix. The existing sixteen-cycle
diagnostic envelope already consumes about 79.6% of its fresh-dot error
budget before candidate GPU arithmetic, so observed cancellation is
insufficient evidence. Predictor updates additionally contain a per-element
gradient clamp, and their nonlinear next input lacks the visual recurrence.
Even eliminating the approximate encoder and credit attribution entirely
would suggest only about 1.33× additional whole-cycle gain on this profile;
those separate attribution measurements are not additive exact costs.

All **119 normal debug brain tests**, **nine focused GPU checks**, their
**nine CPU Vulkan counterparts**, formatting and workspace Clippy across
all targets with warnings denied pass. Logs for the normal checks are
`projection-authority-brain-debug.log` and
`projection-authority-clippy.log`; experiment logs are identified above.
An earlier unfiltered release-unit run hit the pre-existing
`zero_expected_panics_in_debug` test, which expects a debug assertion;
the normal debug suite passes and the focused release checks all pass.

This follow-up adds only test modules, test shader fragments and review
documentation, with no production behavior or dependencies changed. The
284-test serial sandbox result recorded in the preceding section covers
the unchanged production code and was not repeated. None of the three
candidates changes the 3.887× committed production headline, and the 10×
whole-simulation goal remains unmet.
