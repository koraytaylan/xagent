# Architecture - Plan 0008 (deltas)

> Edits in `crates/xagent-brain/src/shaders/kernel/common.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl`,
> `crates/xagent-brain/src/buffers.rs`,
> `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-shared/src/config.rs`,
> `crates/xagent-sandbox/src/agent/mod.rs`,
> `crates/xagent-sandbox/src/momentum.rs`,
> `crates/xagent-sandbox/src/governor.rs`,
> `crates/xagent-sandbox/src/sim_runtime.rs`,
> `crates/xagent-sandbox/src/ui.rs`.
> Line numbers are hints; locate by symbol.

The visual cortex is a new cooperative pass inserted between
`coop_feature_extract` and `coop_encode`. It reads the dense luminance retina
out of `s_features`, runs three biological stages entirely in workgroup shared
memory, and writes a compact complex-cell vector back to the head of
`s_features` so the existing encoder consumes oriented features instead of raw
pixels. All per-filter parameters live in the heritable brain-state tail and are
seeded to biology.

Pipeline (one 256-thread workgroup per agent, all intermediates in workgroup
memory):

```
retina L(x,y)  →  DoG center-surround (ON/OFF)  →  Gabor simple cells (θ,λ,ψ)
              →  complex cells (quadrature energy + MAX pool)  →  feature vector
```

## 0001 — Retinal luminance front end

Today the vision pass writes an RGBA hit color (`VISION_COLOR_COUNT`) plus a
normalized depth (`VISION_DEPTH_COUNT`) per ray into `sensory_buffer`
(`phase_vision.wgsl`), the grid is `VISION_W × VISION_H = 8 × 6`
(`common.wgsl:13-14`), and nothing computes luminance. A Gabor needs the carrier
sampled at ≥ ~3–4 px/cycle, so 8×6 cannot resolve more than one or two
orientations.

### Retina resolution

The grid dimensions are already `override` constants that cascade into the
sensory layout (`common.wgsl:13-22`) and `BrainLayout::new` (`buffers.rs:191-237`).
This workstream exposes them as curriculum configuration rather than fixed
defaults.

- **Retina-resolution config** (`config.rs`, new fields next to `vision_width` /
  `vision_height`): a dedicated retina grid so the cortex resolution is tuned
  independently of the legacy 8×6 sensory contract.

```rust
/// Retinotopic luminance grid the visual cortex operates on. Locked per batch
/// (compile-time override into the kernel, like `vision_width`). Curriculum
/// default 32×32; raised only when throughput stays in budget (gate 0005).
#[serde(default = "default_retina_width")]
pub retina_width: usize,
#[serde(default = "default_retina_height")]
pub retina_height: usize,
```

Properties that make this safe:
- The pixel count is validated with `checked_mul` against an upper bound before
  any buffer is sized (mirrors `buffers.rs:194-195`); an oversized retina is a
  configuration error, not a silent overflow.
- Resolution is a locked per-batch constant, not a heritable gene, so the
  brain-state stride is uniform across the population.

### Luminance derivation

Stage 0 is a single-channel linear field `L(x, y) ∈ [0, 1]`. It is derived from
the per-ray hit color the vision pass already produces — no new ray work beyond
the density increase.

- **Luminance helper** (`common.wgsl`, used at the head of the visual pass):

```wgsl
// Linear (Rec. 709) luminance of a raycast hit color. Convolution kernels are
// linear operators, so they act on linear-light luminance, not gamma-encoded
// RGB. Hit colors are authored in linear space, so no inverse-gamma is applied.
fn retina_luminance(color: vec3<f32>) -> f32 {
    return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
}
```

Properties that make this safe:
- DC (mean luminance) is rejected downstream by the zero-sum DoG and the
  DC-balanced Gabor, so no separate brightness-normalization pass is required for
  orientation tuning.
- Optional per-frame local-contrast normalization (divide by `max(rms, EPSILON)`)
  is applied *after* Stage 1, guarded against divide-by-zero.

## 0002 — V1 cortical pass

Today the pass sequence is seven cooperative functions separated by barriers
(`kernel_tick.wgsl:475-493`), and `coop_encode` reads `FEATURE_COUNT` features
where the leading `VISION_COLOR_COUNT + VISION_DEPTH_COUNT` are raw vision
(`brain_passes.wgsl:178-207`, `buffers.rs` feature-count). This workstream
inserts `coop_visual_cortex` as the pass after `coop_feature_extract` and
redefines the encoder's visual input.

### Pass insertion and barriers

- **Sequence edit** (`kernel_tick.wgsl:475-493`): run the new pass after feature
  extraction and before encode, each separated by a `workgroupBarrier()`.

```wgsl
if (alive) { coop_feature_extract(agent_id, tid); }
workgroupBarrier();
if (alive) { coop_visual_cortex(agent_id, tid); }   // NEW: DoG → Gabor → complex
workgroupBarrier();
if (alive) { coop_encode(agent_id, tid); }
workgroupBarrier();
```

- **Workgroup scratch** (`brain_passes.wgsl`, top-of-file `var<workgroup>`): the
  retina, the signed DoG map, the simple-cell maps, and the complex-cell output.
  Sized from the retina override constants and the bank size.

```wgsl
var<workgroup> s_retina: array<f32, RETINA_PIXEL_COUNT>;         // Stage 0 luminance
var<workgroup> s_center_surround: array<f32, RETINA_PIXEL_COUNT>; // Stage 1 signed DoG
var<workgroup> s_complex: array<f32, VISUAL_FEATURE_COUNT>;       // Stage 3 output
```

Properties that make this safe:
- The new pass uses only `workgroupBarrier()` (all intermediates are workgroup
  memory; no `var<storage>` writes are read back in-pass, so `storageBarrier()`
  is not required here).
- The `if (alive)` guard is uniform across the workgroup and precedes every
  barrier inside the pass, satisfying the WGSL uniform-control-flow rule.
- Every offset derives from `BrainLayout`; no stride is hardcoded.

### Stage 1 — center-surround (Difference of Gaussians)

A zero-sum concentric kernel that reports local contrast, split into rectified
ON-center and OFF-center channels (Rodieck 1965; Hubel & Wiesel 1962).

```
G(x,y;σ) = (1 / (2π σ²)) · exp( −(x² + y²) / (2σ²) )
DoG(x,y) = G(x,y;σ_center) − G(x,y;σ_surround)        with σ_surround = 1.6·σ_center
r_on  = max(0,  DoG * L)
r_off = max(0, −(DoG * L))
```

- Unit-volume Gaussians and equal weights make `∑ DoG = 0` exactly; the kernel
  half-width is `ceil(3·σ_surround)` (truncate at 3σ of the larger Gaussian).
- `σ_center` and the surround ratio are seeded (1.0–1.5 px, 1.6) and the ratio is
  the heritable `dog_surround_ratio` gene (0003). The zero-sum constraint is
  re-imposed after mutation.

### Stage 2 — V1 simple cells (oriented Gabor bank)

The 2-D Gabor is the validated quantitative model of a simple-cell receptive
field (Jones & Palmer 1987); the elongated alternating ON/OFF lobes are Hubel &
Wiesel's (1962) "aligned row of LGN inputs".

```
x' =  x·cosθ + y·sinθ ,   y' = −x·sinθ + y·cosθ
Gabor(x,y) = exp( −(x'² + γ²·y'²) / (2σ²) ) · cos( 2π·x'/λ + ψ )
s_{θ,λ,ψ} = max(0, Gabor_{θ,λ,ψ} * L)
```

- **Bank seeds:** orientations tiled evenly over [0, π) — start 4 (0, 45, 90,
  135°), 8 once resolution ≥ 48 px (the HMAX S1 choice, Riesenhuber & Poggio
  1999); scales 2–3 (λ ≈ 3, 5, 8 px) with σ = 0.56·λ; phases a single quadrature
  pair (ψ = 0 even, ψ = π/2 odd).
- **DC balance:** subtract the kernel mean so `∑ Gabor = 0`, re-imposed after
  mutation.
- Per-filter kernels are computed once per pass into a small workgroup buffer
  (not recomputed per pixel); all divisions by λ and σ use `max(·, EPSILON)`.

### Stage 3 — V1 complex cells (energy + MAX pooling)

Phase invariance from the squared quadrature pair (Adelson & Bergen 1985);
position/scale tolerance from the MAX over a local neighborhood (HMAX C1,
Riesenhuber & Poggio 1999).

```
E_{θ,λ}(x,y) = sqrt( even_{θ,λ}(x,y)² + odd_{θ,λ}(x,y)² )       // phase invariance
C_θ(row,col) = MAX over (Δx,Δy)∈pool, over scale band  E_{θ,λ}(x+Δx, y+Δy)
```

- Energy uses the **linear** (un-rectified) Gabor outputs — squaring supplies
  non-negativity; rectifying first would double-count.
- Pool over **position and scale only**, never over the quadrature phases (the
  energy step already handles phase). Pool side ≈ `ceil(2·σ)` with ~50% overlap.
- Output length:
  `VISUAL_FEATURE_COUNT = orientations × scales × pool_rows × pool_cols`
  (e.g. 4 × 2 × 4 × 4 = 128). Each output is non-negative; the vector is
  L2-normalized per frame (divide by `max(norm, EPSILON)`) to mirror V1 response
  normalization.

### Wiring into the encoder

- **Feature-count redefinition** (`buffers.rs` `BrainLayout::new`, plus the
  matching `common.wgsl` override): the encoder's visual input becomes
  `VISUAL_FEATURE_COUNT` complex-cell features instead of
  `VISION_COLOR_COUNT + VISION_DEPTH_COUNT` raw pixels.

```rust
/// Encoder input width: compact complex-cell vector + the unchanged non-visual
/// tail (proprioception, interoception, touch). Replaces the raw-vision width.
let feature_count = visual_feature_count + NON_VISUAL_FEATURE_COUNT;
```

Properties that make this safe:
- `VISUAL_FEATURE_COUNT` and `NON_VISUAL_FEATURE_COUNT` are single canonical
  constants shared by Rust and WGSL (CONTRIBUTING: one source per shared
  constant); the encoder weight region `O_ENC_WEIGHTS` resizes from them.
- The visual cortex writes `s_features[0 .. VISUAL_FEATURE_COUNT)`; the non-visual
  tail keeps its relative order, shifted to start at `VISUAL_FEATURE_COUNT`.
- Behind a config flag defaulted off until the 0005 gate (Locked decision):
  when off, `coop_feature_extract` fills the legacy raw-vision features and the
  cortex pass is skipped, so the change is byte-identical to today.

## 0003 — Heritable visual genome

Today the heritable tail holds four contiguous scalars —
`habituation_sensitivity`, `max_curiosity_bonus`, `fatigue_floor`,
`movement_speed` (`buffers.rs:80-83`) — written by `write_agent_heritable_config`
(`gpu_kernel.rs:2479-2513`). This workstream appends the Gabor-bank genes and
retires `visual_encoding_size`.

### Visual-bank genes

- **New `BrainConfig` fields** (`config.rs`), seeded to biology, each with the
  full add-a-gene wiring (default fn, `tiny`/`large` presets, `mutate` clamp,
  crossover, momentum tuple, `record_mutations` provenance, UI edit + display,
  brain-state tail offset):

```rust
/// V1 Gabor carrier wavelength λ in retina pixels. Seed 5.0; envelope σ tied as
/// 0.56·λ (≈1-octave V1 bandwidth). Heritable; clamped to [2.0, 12.0].
#[serde(default = "default_gabor_wavelength")]
pub gabor_wavelength: f32,
/// Gabor envelope aspect ratio γ (long axis / short axis). Seed 0.5. Heritable;
/// clamped to [0.25, 1.0].
#[serde(default = "default_gabor_aspect_ratio")]
pub gabor_aspect_ratio: f32,
/// DoG surround:center sigma ratio. Seed 1.6 (Marr & Hildreth edge operator).
/// Heritable; clamped to [1.2, 3.0].
#[serde(default = "default_dog_surround_ratio")]
pub dog_surround_ratio: f32,
/// Whole-bank orientation offset in radians, added to the even [0,π) tiling.
/// Seed 0.0. Heritable; wrapped to [0, π).
#[serde(default = "default_orientation_offset")]
pub orientation_offset: f32,
```

- **Tail offsets** (`buffers.rs`, after `O_MOVEMENT_SPEED`): one slot per new
  scalar, with `FIXED_TAIL_SIZE` grown to match and the `write_agent_heritable_config`
  contiguity `debug_assert_eq!`s extended.

```rust
pub const O_GABOR_WAVELENGTH: usize = O_MOVEMENT_SPEED + 1;
pub const O_GABOR_ASPECT_RATIO: usize = O_GABOR_WAVELENGTH + 1;
pub const O_DOG_SURROUND_RATIO: usize = O_GABOR_ASPECT_RATIO + 1;
pub const O_ORIENTATION_OFFSET: usize = O_DOG_SURROUND_RATIO + 1;
```

Properties that make this safe:
- These are scalar genes, so the string-keyed momentum (`momentum.rs:121-169`)
  and per-name provenance (`governor.rs:1682-1747`) machinery applies unchanged
  — no vector-genome refactor (per-filter genes are out of scope).
- Invariants are enforced in the shader *after* reading the genes: DoG zero-sum,
  Gabor DC-balance, quadrature 90° offset. A mutated `dog_surround_ratio` that
  approaches 1.0 is clamped to ≥ 1.2 so the kernel cannot degenerate into a
  blur (non-edge) operator.
- New tail slots are initialized in `init_brain_state_for` (`buffers.rs:534-594`)
  using the same fixed-delta pattern as `movement_speed`, so headless and
  interactive seeds match.

### Retiring `visual_encoding_size`

- **Supersede, do not break deserialization** (`config.rs`, `ui.rs`,
  `agent/mod.rs`): keep the field with its `#[serde(default)]` so existing JSON
  and DB configs still load, drop it from the brain-config UI editor and from
  breeding pass-through, and update its doc comment to point at the structured
  visual config. This resolves issue #106 by replacement.

## 0004 — Runtime genome authority for visual genes

Today the interactive worker uploads physics and resets agents but never
re-applies the heritable tail after inheritance (`sim_runtime.rs:317-331`,
`sim_runtime.rs:433-445`), unlike headless (`headless.rs:159-164`). With four
scalar genes today this already drops mutations; with the larger visual genome
it would silently revert every agent's vision to defaults.

- **Build on plan 0007** (`sim_runtime.rs`): once
  `0007 effective-agent-config-upload` adds `patch_agent_configs`, the visual
  genes ride along automatically because they are written by the same extended
  `write_agent_heritable_config`. This workstream's task is the *test* that locks
  the guarantee for the visual slots specifically, plus any wiring 0007 left out.

Properties that make this safe:
- `write_agent_heritable_config` already writes a contiguous tail block; the new
  genes extend that block, so a single `queue.write_buffer` per agent still
  covers them.
- The dependency is explicit: `0004` is blocked until `0007` lands (SCOPE locked
  decision), and the test fails loudly if the worker path regresses.

## Test Strategy

Each stage ships a falsifiable GPU probe (self-skipping without an adapter). The
probes are the scientific acceptance criteria, not smoke tests.

- `dog_kernel_sums_to_zero` — the seeded DoG kernel weights sum to ≈ 0; a uniform
  retina yields ≈ 0 response, a contrast edge yields a non-zero response.
- `gabor_kernels_are_dc_balanced` — each seeded Gabor kernel sums to ≈ 0.
- `vertical_bar_excites_vertical_simple_cell` — a vertical bar rendered into the
  retina drives the vertical-tuned simple cell far above the horizontal-tuned one
  (the core Hubel & Wiesel result); a swept orientation produces a unimodal
  tuning curve peaked at the preferred orientation.
- `complex_cell_phase_invariance` — flipping the bar's phase by half a wavelength
  leaves the complex-cell energy ≈ constant while the simple-cell response
  changes sign (Adelson & Bergen).
- `complex_cell_position_tolerance` — a small spatial shift of the bar within the
  receptive field leaves the MAX-pooled complex response ≈ constant.
- `visual_features_replace_vision_byte_identical_when_flag_off` — with the cortex
  flag off, `FEATURE_COUNT` and the encoder output match the current build.
- `worker_reset_applies_visual_genome_after_inheritance` — two agents with
  distinct `gabor_wavelength` keep distinct tail slots across the worker reset
  path (0004).

Canonical CI gate for every task:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p xagent-sandbox
```

## Interaction With Prior Work

- **Plan 0007 (Learning Control Grounding) — depended upon.**
  `0004` requires `0007 effective-agent-config-upload`; the heritable-genome
  authority fix is reused rather than duplicated. The 0005 emergence gate mirrors
  0007's red-green / curriculum-gate discipline.
- **Plan 0006 (Multi-Workgroup Brain Parallelism) — respected.** The new pass
  follows the same-dispatch cooperative-tiling pattern that won in 0006 and stays
  inside the fused `FusedSerial` default; it adds work per agent, so its cost is
  tracked against the 0006/0005 throughput baselines and gated in 0005.
- **Issue #106 (`visual_encoding_size`) — resolved.** Closed by superseding the
  legacy field with structured visual-cortex configuration (0003).
- **`representation_dimension` precedent — followed.** The retina/feature
  constants are echoed-and-validated against the compile-time kernel constants
  exactly as `representation_dimension` echoes `ENCODED_DIMENSION` (`config.rs:43-52`).
