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
