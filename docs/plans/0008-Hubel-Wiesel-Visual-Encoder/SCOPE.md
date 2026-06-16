# Scope - Plan 0008

> Give agents a scientifically grounded early-visual-cortex front end — a dense
> luminance retina, retinal/LGN center-surround cells, orientation-selective V1
> simple cells, and position-and-phase-invariant V1 complex cells — seeded to
> biology, evolvable, and proven by red-green orientation/invariance probes
> before it becomes the encoder's visual input.

## Why this plan

Today the brain has no visual front end. The fused kernel hands the raw raycast
output straight into a single dense linear encoder, so there is no
center-surround stage, no oriented edge detector, and no position-invariant
feature — none of the Hubel & Wiesel hierarchy that the early mammalian visual
system is built from. This plan adds that hierarchy as a cooperative GPU pass,
makes its receptive fields heritable from biologically-correct seeds, and gates
the switch-over on probes that mechanically demonstrate orientation selectivity
and phase/position invariance. The load-bearing observations are:

1. **The brain has no eyes — vision is fed to a generic dense encoder.**
   `coop_feature_extract` copies the raw vision slice into `s_features` and
   `coop_encode` (`brain_passes.wgsl:119-172`, `brain_passes.wgsl:178-207`)
   projects it with one `FEATURE_COUNT × ENCODED_DIMENSION` weight matrix. There
   is no oriented-edge or center-surround stage; the encoder sees a flat pixel
   list with no visual structure imposed (see the brain crate README,
   "the brain has no eyes").
2. **Vision is a category-color raycast field, not luminance.** Each of the
   `VISION_W × VISION_H = 8 × 6 = 48` rays returns an RGBA hit color plus a
   normalized depth (`phase_vision.wgsl`; `common.wgsl:13-14`,
   `common.wgsl:220-228`). The Hubel & Wiesel chain operates on a luminance
   image of light/dark contrast, so a single-channel retinotopic luminance field
   must be derived before any cell can fire.
3. **`visual_encoding_size` is a legacy gene with no consumer.** It is declared
   legacy (`config.rs:37-42`), is never mutated (`agent/mod.rs:378`), is pinned
   to parent A in crossover (`agent/mod.rs:503`), and is shown "(legacy)" in five
   UI surfaces (`ui.rs:21-26`, `ui.rs:1386`, …). Issue #106 left it as
   "wire it to a real encoder or remove it." This plan resolves #106 by
   superseding it with structured visual-cortex configuration.
4. **Receptive fields should be biologically seeded yet evolvable.** The project
   ethos makes structural parameters heritable rather than hand-frozen. The
   quantitative models are settled — center-surround as a Difference-of-Gaussians
   (Rodieck 1965) with a σ_surround : σ_center ≈ 1.6 : 1 edge-operator ratio
   (Marr & Hildreth 1980), simple cells as 2-D Gabor filters validated to
   noise-level residual on real cat cells (Jones & Palmer 1987) — so biology is
   the seed and evolution tunes from there.
5. **Heritable genes are not reliably applied in the interactive runtime.**
   Plan 0007 finding: the live worker never re-applies per-agent heritable
   brain-state after inheritance (`sim_runtime.rs:317-331`,
   `sim_runtime.rs:433-445`), while the headless path does
   (`headless.rs:159-164`). A *vector-sized* visual genome inherits this hazard
   and would silently collapse to defaults in interactive evolution.
6. **Emergence must be proven, not assumed.** Plan 0007's locked stance — no
   behavior claim without red-green probes — applies doubly here: orientation
   selectivity (a vertical bar must excite the vertical-tuned cell and not the
   horizontal one) and complex-cell phase invariance (Adelson & Bergen 1985) are
   directly, mechanically testable on the GPU output.
7. **Retina density costs throughput and grows the encoder.** A faithful retina
   needs ≥ ~16×16 and ideally 32×32–64×64 pixels to resolve oriented bars
   without aliasing. Densifying 8×6 → 32×32 is 1024 rays × 25 raymarch steps
   (~21× today's vision cost), and naively widening the encoder input would grow
   the `O_ENC_WEIGHTS` matrix. The visual front end must *reduce* the encoder
   input (compact complex-cell vector) and the resolution must be curriculum-
   gated on measured throughput, exactly as plan 0007 gates control rate.
8. **A new pass must respect the fused-kernel barrier discipline.** The seven
   cooperative passes are barrier-synchronized inside one 256-thread workgroup
   (`kernel_tick.wgsl:475-493`). A new visual pass must place
   `workgroupBarrier()` between stages, keep control flow uniform before each
   barrier, and derive every buffer offset from `BrainLayout` (CONTRIBUTING:
   GPU/WGSL safety).

Together these say: the agents can see pixels but cannot see *structure*; the
substrate to impose biological visual structure exists (configurable rays, a
cooperative-pass slot, a heritable tail, a legacy field waiting for a job); and
the scientific models for each stage are settled enough to seed and to test.

## In scope

- **0001 — Retinal luminance front end.** Make retina resolution configurable
  and densifiable, derive a single-channel linear luminance field from the
  raycast hit color, and set a curriculum default resolution gated on throughput.
- **0002 — V1 cortical pass.** A new cooperative GPU pass implementing
  Difference-of-Gaussians center-surround (ON/OFF), an oriented Gabor simple-cell
  bank (quadrature phase pairs), and complex cells (quadrature energy + MAX
  pooling), whose compact output replaces the raw vision slice feeding the
  encoder.
- **0003 — Heritable visual genome.** Add biologically-seeded, evolvable
  Gabor-bank genes to `BrainConfig` end-to-end (defaults, presets, mutation,
  crossover, momentum, provenance, UI, brain-state tail), enforce the
  post-mutation invariants, and retire `visual_encoding_size` (closing #106).
- **0004 — Runtime genome authority for visual genes.** Ensure the interactive
  worker applies the per-agent visual genome after inheritance (depends on
  plan 0007's `effective-agent-config-upload`).
- **0005 — Red-green probes and emergence gate.** Mechanical tests for
  zero-sum center-surround, orientation selectivity / tuning curves, and
  complex-cell phase + position invariance; the visual front end becomes the
  default encoder input only after these pass and throughput stays in budget.

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| Brain has no visual front end (1) | `0001`, `0002` |
| Vision is color, not luminance (2) | `0001` |
| `visual_encoding_size` legacy gene (3) | `0003` |
| Receptive fields seeded-but-evolvable (4) | `0003` |
| Heritable genes unapplied in runtime (5) | `0004` |
| Emergence must be proven (6) | `0005` |
| Retina density cost (7) | `0001`, `0005` |
| New-pass barrier discipline (8) | `0002` |

## Locked decisions

- **Dense raycast → luminance, not a rendered retina.** The retina is built by
  densifying the existing raycast grid and deriving linear luminance from the
  per-ray hit color. A separate render-to-texture POV path is out of scope. The
  curriculum default resolution is **32×32**; it becomes denser only if measured
  throughput allows (gate in `0005`).
- **Receptive fields are biologically seeded and heritable.** Seeds: DoG with
  σ_surround / σ_center = 1.6 (Marr & Hildreth 1980); Gabor envelope σ = 0.56·λ
  (≈ 1-octave V1 bandwidth) and aspect ratio γ ≈ 0.5; orientations tiled evenly
  over [0, π). Evolution tunes **global bank genes** (spatial frequency λ,
  bandwidth, DoG surround ratio, aspect ratio, a bank orientation offset) — a
  handful of scalars that reshape the whole bank coherently.
  **Independent per-filter orientation/phase genomes are deferred** for two
  reasons: (a) a vector genome does not fit the scalar momentum/provenance
  machinery, and (b) — the load-bearing reason — a ~48-dimensional per-filter
  genome is a far larger search space, and plan 0007 measured the outer
  evolutionary loop as *not converging at all* (root still best, 249 failed
  descendants; learning at chance). Widening the genome before the small one is
  shown to move would enlarge the haystack before a needle is known to exist.
  This is a v1 narrowing, not a permanent ceiling; the graduation condition is
  explicit (see the "mechanical, not an evolutionary fitness gain" decision below).
- **Invariants enforced after every mutation.** DoG kernels stay zero-sum; Gabor
  kernels stay DC-balanced (∑ = 0); the quadrature pair stays exactly 90° apart
  (ψ_odd = ψ_even + π/2). Mutation that breaks these is renormalized in the same
  scope, never deferred.
- **Depth stops at V1 complex cells.** No V2/V4/IT, no cortical feedback or
  recurrence, no temporal/motion energy, no color or magnocellular/parvocellular
  split, single monocular channel. The spatial case of the Adelson & Bergen
  (1985) energy model only.
- **Visual features replace the vision slice into the encoder.** The complex-cell
  vector becomes the encoder's visual input; the non-visual feature tail
  (proprioception, interoception, touch) is unchanged. `FEATURE_COUNT` is
  redefined accordingly from a single canonical constant shared by Rust and WGSL.
- **No emergence claim without probes.** The front end becomes the default
  encoder input only after the `0005` orientation-selectivity and
  phase/position-invariance probes pass and the throughput regression is within
  the gate budget. Until then it ships behind a config flag, defaulted off.
- **This plan's success is mechanical, not an evolutionary fitness gain.** The
  claim is that the filters fire correctly — a vertical bar excites the
  vertical-tuned cell, complex cells are phase/position invariant — proven by the
  `0005` probes. It is explicitly *not* a claim that visual evolution improves
  food-seeking, because the outer loop is currently stalled (plan 0007). A
  fitness gain depends on the learning/credit path that plan 0007 owns; no such
  gain is asserted here, and the global-bank genes may move little until that
  path works. The graduation to a per-filter genome is **gated on a demonstrated
  evolutionary signal**: only widen the genome once the four bank genes are shown
  to be tuned in a measured direction by selection (a non-trivial
  `record_mutations` trend), not merely once the probes pass.
- **Depends on plan 0007 runtime genome authority.** The heritable visual genome
  is only trustworthy in interactive runs once
  `0007 effective-agent-config-upload` lands; `0004` builds on it.

## Out of scope

- **Independent per-filter visual genome (vector genes).** Per-filter θ and ψ
  evolving independently would be maximally expressive but does not fit the
  string-keyed scalar momentum (`momentum.rs`) and per-name mutation provenance
  (`governor.rs record_mutations`), and — more importantly — multiplies the
  search space ~10× for an outer loop plan 0007 measured as non-converging.
  Deferred to a follow-up **gated on a demonstrated evolutionary signal** on the
  global bank genes (see Locked decisions); v1 evolves global bank parameters
  with seeded even-orientation tiling.
- **A rendered POV retina.** Rasterizing the agent's field of view to a texture
  would be the most faithful "eye" but is a substantial new GPU render path;
  rejected in favor of dense raycasts.
- **Higher visual areas and recurrence.** No V2/V4/IT, no feedback, no motion,
  no color, no binocularity (see Locked decisions).
- **A new brain encoder or memory architecture.** `coop_encode`,
  `ENCODED_DIMENSION`, memory, and TD(λ) credit assignment are unchanged except
  for the redefined visual input width.
- **Independent per-genome arenas.** Same boundary as plan 0007: this plan makes
  the shared-world runtime see structure, not a per-repeat arena.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
