# 0008 — Visual cortex throughput baseline

Decision artifact for plan 0008 task `visual-cortex-throughput-baseline`
("Measure The Cost", SCOPE finding 7). This task **only measures and records**
the throughput cost of the new `coop_visual_cortex` pass. It does **not** assert
a tps target and does **not** flip `visual_cortex_enabled`'s default — that is
the next task, `visual-encoder-default-gate`, which weighs these numbers against
its budget.

## What was measured

Full-pipeline ticks-per-second (tps) with `visual_cortex_enabled` **OFF** vs.
**ON**, at the default population and the curriculum retina default, via the
focused sibling bench added here:

```
xagent --bench-visual-cortex --bench-agents 10 --bench-ticks 20000
```

The bench (`crates/xagent-sandbox/src/bench.rs::run_visual_cortex_ab`, dispatched
from `--bench-visual-cortex` in `crates/xagent-sandbox/src/main.rs`) builds a
fresh `GpuKernel` per arm, toggling only `BrainConfig::visual_cortex_enabled`.
The flag and the retina dimensions are compile-time WGSL overrides baked at
`GpuKernel::new`, so each arm rebuilds the pipeline with the correct width and
pass set. Wall time is taken from the first dispatch to the GPU completing the
last tick (a `Maintain::Wait` drain), so the number reflects GPU execution
(including the new pass when ON), not just submit-return.

The dispatch is chunked into `kernel_batch_size() × MAX_FUSED_BATCHES`-tick
windows, each drained with `Maintain::Wait`, **for both arms identically**. This
is required because the cortex-ON arm is ~400× slower: submitting all 20 000
ticks in one fused batch would queue minutes of GPU work and trip wgpu's
submission watchdog (`panic_on_timeout`). Chunking caps each drain to one
window; the per-chunk wait is real GPU execution and stays inside the timed
region, so the tps stays faithful. (The chunked OFF arm reads ~34 k tps vs.
~35.3 k tps for the legacy single-submit `--bench` — the per-chunk sync overhead
is small, confirming the OFF arm is a faithful 0006-fused baseline.)

## Configuration

| Parameter | Value |
|---|---|
| Retina resolution | **32 × 32** (1024 luminance pixels) — curriculum default |
| Gabor bank | 4 orientations × 2 scales × 2 phases (quadrature) |
| Complex-cell output | 4 orientations × 2 scales × 4 × 4 MAX-pool grid = **128 features** (`VISUAL_FEATURE_COUNT`) |
| Population (agents) | 10 (`default_population_size`) |
| Ticks per arm | 20 000 |
| OFF feature width | `8 × 6 × 4 + 8 × 6 + 25` raw-vision features (byte-identical to pre-plan build) |
| ON feature width | `VISUAL_FEATURE_COUNT (128) + NON_VISUAL_FEATURE_COUNT (25) = 153` |

## Measured throughput

Measured on the development host (no Mesa lavapipe; real GPU adapter present), so
the measurement ran rather than self-skipping:

- **Host:** Apple M3 Max
- **wgpu backend:** Metal (Apple GPU)
- **Build:** `--release`
- **Date:** 2026-06-17

| Arm | tps | Retained vs OFF |
|---|---:|---|
| `visual_cortex_enabled` **OFF** (0006-fused baseline) | **≈ 34 000** | — |
| `visual_cortex_enabled` **ON** (retina 32×32, 128-feature bank) | **≈ 81** | **≈ 0.24 %** (≈ 420× slower) |

Representative raw runs (stable across repeats, OFF 33.5–34.3 k, ON 81–82):

```
[visual-cortex-ab] 10 agents, 20000 ticks, retina 32x32 — visual cortex on/off tps
                     arm             tps                  vs off
   cortex OFF (baseline)           34275                       —
               cortex ON              81  0.24% kept (422x slower)
```

## Reading

At retina 32×32 with the 4×2×2 Gabor bank, turning the cortex pass on costs
~99.76 % of throughput (~420× slowdown), retaining only ~0.24 % of the fused
baseline. The cost is dominated by the dense convolution work the single
256-thread workgroup does per agent: 1024-pixel DoG, then 16 Gabor kernels
(4 orientations × 2 scales × 2 phases) convolved over the retina, then
quadrature-energy + MAX pooling — all serialized within one fused dispatch.

This number is recorded here as the input to `visual-encoder-default-gate`. That
task defines the budget (the TASKS.md sketch suggests ≥ 50 % of prior tps
retained) and decides flip-or-hold; this task makes no such decision. As a
factual observation for that task: 0.24 % retained is far below any plausible
≥ 50 % budget at 32×32, so the gate is expected to **hold** the default at
`false` (or require a much smaller retina / cheaper pooling) rather than flip it.
No default was flipped and no target was asserted here.

## Reproduce

```bash
# Default population (10), curriculum retina 32×32:
cargo run --release -p xagent-sandbox --bin xagent -- \
  --bench-visual-cortex --bench-agents 10 --bench-ticks 20000

# Self-skips with a clear note only if no GPU/fallback adapter is available
# (GpuKernel::is_available() == false). On a host with a real adapter — as here —
# it runs and prints the on/off table above.
```

---

# Default-flip gate decision (`visual-encoder-default-gate`)

This section is the resolution of plan 0008's gated task `visual-encoder-default-gate`
("Flip The Default"). It applies the SCOPE gate to the numbers measured above and
the red-green probes below.

## The gate (from SCOPE / TASKS 0005)

> The front end becomes the default encoder input **only if** the
> orientation-selectivity and phase/position-invariance probes pass **and** the
> measured throughput regression is within budget. **Budget:** retained
> throughput ≥ **50 %** of the prior 0006 fused baseline (the conjunction of both
> conditions is required — either one failing holds the default at `false`).

## Measured inputs

### (1) Red-green probes — **PASS** (mechanical, all clear with margin)

Run on the development host (Apple M3 Max, Metal, real adapter present, so the
GPU-gated probes ran rather than self-skipping), 32×32 retina, seeded
4 orientations × 2 scales × 2 phases Gabor bank. The probes execute the same
DoG → Gabor → quadrature-energy → MAX-pool math the GPU `coop_visual_cortex` runs
(byte-mirrored, drift-guarded by the `wgsl_*_constants_match_rust` unit tests).

| Probe | Threshold | Measured | Margin | Result |
|---|---|---:|---|---|
| `vertical_bar_excites_vertical_simple_cell` — orientation selectivity (preferred ÷ orthogonal energy) | ≥ 3.0× | **6.97×** | 2.3× the bar | PASS |
| `vertical_bar_excites_vertical_simple_cell` — swept tuning curve | unimodal, peak at θ = vertical | peak at vertical, monotone in lobe, tail < peak/3 | — | PASS |
| `vertical_bar_excites_vertical_simple_cell` — isotropic control | ratio ≪ 3.0× | 1.0 (not selective) | — | PASS (discriminating) |
| `complex_cell_phase_invariance` — energy change under half-λ carrier shift | < 10 % | **≈ 0.00002 %** (1.6e-7) | ~5 orders | PASS |
| `complex_cell_phase_invariance` — single-phase control under quarter-λ shift | ≥ 10 % (must break) | breaks the bar | — | PASS (discriminating) |
| `complex_cell_position_tolerance` — vector change under 1 px shift | < 15 % | **≈ 7.8 %** | ~half the bar | PASS |
| `complex_cell_position_tolerance` — 8 px far-shift control | > 15 % (must break) | **≈ 39.8 %** | — | PASS (discriminating) |

Also green (stage well-formedness, same run): `dog_kernel_sums_to_zero`,
`gabor_kernels_are_dc_balanced`, `complex_pool_output_is_nonnegative_and_normalized`,
`visual_cortex_passthrough_is_byte_identical`. The filters fire correctly: a
vertical bar drives the vertical-tuned cell ~7× the orthogonal cell with a
unimodal tuning curve, complex cells are phase-invariant to ~5 significant
figures, and the MAX pool absorbs a one-pixel shift while still discriminating a
real translation. This is exactly the mechanical claim SCOPE makes ("the filters
fire correctly … proven by the 0005 probes"), and it is fully met.

### (2) Throughput — **FAIL** (far below budget)

From the `## Measured throughput` table above, at the curriculum retina default
(32×32) and default population (10 agents) on the same host:

| Quantity | Value |
|---|---:|
| `visual_cortex_enabled` OFF (0006-fused baseline) | ≈ 34 000 tps |
| `visual_cortex_enabled` ON (retina 32×32, 128-feature bank) | ≈ 81 tps |
| Retained throughput | **≈ 0.24 %** |
| Budget required | ≥ 50 % |
| Verdict | **FAIL — ~210× under budget** |

The same verdict holds against the plan-0006 full-pipeline N=10 baseline
(≈ 23 230 tps): 81 ÷ 23 230 ≈ 0.35 %, still ~140× under the 50 % budget.
The conclusion is insensitive to which 0006 baseline figure is used.

## Decision — **HOLD** (`visual_cortex_enabled` default stays `false`)

The gate is a conjunction: probes **AND** throughput. The probes pass with wide
margin, but throughput retains only ~0.24 % of the fused baseline against a ≥ 50 %
budget — failing by more than two orders of magnitude. **One condition failing
holds the default**, so the default is **not** flipped.

`visual_cortex_enabled` is left at its `#[serde(default)]` value `false`
(`crates/xagent-shared/src/config.rs` `BrainConfig` field, `Default`, `tiny`, and
`large` presets). No curriculum retina default is changed; with the cortex off
the run remains byte-identical to the pre-plan build
(`visual_cortex_passthrough_is_byte_identical`). The visual cortex ships complete,
proven-correct, and behind the gate flag — available to turn on per-batch for
experiments, but not the default encoder input.

### Why throughput is this far under budget

The cost is structural, not a hot-loop inefficiency. The whole DoG → Gabor →
energy → MAX-pool pipeline runs serially inside a **single 256-thread workgroup
per agent**: a 1024-pixel zero-sum DoG convolution, then 16 Gabor kernels
(4 orientations × 2 scales × 2 phases) each convolved over the 1024-pixel retina,
then quadrature energy and MAX pooling — all between `coop_feature_extract` and
`coop_encode` in the fused dispatch. That is ~21× the ray work and a dense
convolution stack the prior pipeline never had. No amount of constant-factor
tuning closes a ~420× gap to the no-cortex arm; the cost has to be *designed*
down, which is out of this task's "flip the default" scope.

## Follow-up (the smaller task SCOPE/TASKS asks for on a hold)

The hold is provisional on throughput, not on correctness. The follow-up is a
cost-reduction workstream that re-runs **this exact gate** once a cheaper pass
clears the budget — the probes are already green and need not be re-derived, only
re-confirmed at the new resolution/bank. Cost levers, cheapest-first:

1. **Reduce the retina (curriculum gate).** SCOPE makes resolution explicitly
   curriculum-gated on throughput. Re-bench at 16×16 (256 px, ~16× less
   convolution work) and 24×24, and re-run the orientation probe at each (it
   requires ≥ ~3–4 px/cycle, so 16×16 with λ ≈ 5 still resolves the seed bank).
   16×16 alone is ~16× cheaper on the dominant convolution term — necessary but,
   on its own, still well short of 50 %.
2. **Cheaper pooling / smaller bank.** Drop scales 2 → 1 or pool more
   aggressively (fewer `POOL_ROWS × POOL_COLS`), halving the Gabor-convolution and
   energy work. Re-run `complex_cell_position_tolerance` to confirm the coarser
   pool still tolerates a 1 px shift.
3. **Separable / FFT-domain convolution.** The Gaussian/Gabor envelopes are
   separable; an x-then-y pass turns the O(k²) per-pixel convolution into O(2k),
   the largest single structural win without changing the biology. This is a
   kernel-architecture change, not a default flip — hence a follow-up.

Per the SCOPE "mechanical, not an evolutionary fitness gain" decision, the
**per-filter genome** widening remains separately gated on a *demonstrated
evolutionary signal* on the four bank genes and is **not** unblocked by this hold;
this follow-up is purely about the throughput cost of the existing global-bank
pass.

## Verification (this task)

- The three gate probes (`vertical_bar_excites_vertical_simple_cell`,
  `complex_cell_phase_invariance`, `complex_cell_position_tolerance`) plus the
  stage probes ran on the M3 Max adapter and **passed**; the measured numbers
  above were read from a one-off instrumented run and the instrumentation was
  reverted (no test-body change landed).
- Gates green with the default left `false`:
  `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test -p xagent-sandbox`.
- No code default changed — `visual_cortex_enabled` stays `false` — so the
  flag-off path remains byte-identical to the pre-plan build.
