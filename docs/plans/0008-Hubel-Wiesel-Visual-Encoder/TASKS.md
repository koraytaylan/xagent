# XAgent Plan 0008 — Hubel-Wiesel Visual Encoder

This plan builds a biologically-grounded early-visual-cortex front end for the
brain: it makes the retina a configurable dense luminance grid, derives linear
luminance from raycast hit color, inserts a cooperative GPU pass that runs
Difference-of-Gaussians center-surround → oriented Gabor simple cells →
quadrature-energy + MAX-pooled complex cells, replaces the raw-vision slice
feeding `coop_encode` with the compact complex-cell vector, makes the Gabor-bank
receptive fields biologically-seeded heritable genes (retiring the legacy
`visual_encoding_size`), ensures the interactive worker applies that genome after
inheritance, and gates the switch-over on red-green orientation-selectivity and
phase/position-invariance probes.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md) for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/0008-hubel-wiesel-visual-encoder--{id}/`).
- **Depends on** lists *direct* prerequisites only (`—` means none).
- **Done when** is the verifiable acceptance criterion; every task must keep
  `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo test -p xagent-sandbox` green (state as "cargo fmt/clippy/test green").
- GPU tests self-skip without an adapter:

```rust
if !xagent_brain::GpuKernel::is_available() {
    eprintln!("Skipping: no GPU/fallback adapter available");
    return;
}
```

- All buffer offsets derive from `BrainLayout` / kernel constants — never
  hardcode a stride. Shared Rust↔WGSL constants have a single canonical source.
- Numeric safety: `try_into()` for lossy casts, `checked_mul` before sizing,
  `max(denominator, EPSILON)` before every division (Rust and WGSL).

---

## 0001 — Retinal luminance front end

### retina-resolution-config — Make Retina Resolution Configurable

Today the visual grid is fixed at `VISION_W × VISION_H = 8 × 6`
(`common.wgsl:13-14`) and threaded through `BrainLayout::new`
(`buffers.rs:191-237`). 8×6 is far too coarse to resolve oriented bars. Add a
dedicated retina grid so cortex resolution is tuned independently of the legacy
sensory raycast contract, with a curriculum default of 32×32.

**Steps:**
1. Add `retina_width: usize` and `retina_height: usize` to `BrainConfig`
   (`config.rs`), each with `#[serde(default = "default_retina_*")]`. Define
   `default_retina_width` / `default_retina_height` returning `32`. Document them
   as locked-per-batch (not heritable), mirroring `vision_width`'s doc.
2. Add `retina_width: 32, retina_height: 32` to `Default`, `tiny` (use `16`), and
   `large` (use `48`) presets (`config.rs:305-429`).
3. In `BrainLayout::new` (`buffers.rs:191-237`) compute
   `let retina_pixel_count = retina_width.checked_mul(retina_height).expect("retina pixel count overflow");`
   and store it on the layout. Add a `RETINA_PIXEL_COUNT` override into the
   kernel constants (`common.wgsl`) via the same path as `VISION_W`
   (`gpu_kernel.rs` vision override constants).
4. Add a unit test `retina_pixel_count_matches_config` (no GPU) asserting
   `BrainLayout::new` with `retina 32×32` yields `retina_pixel_count == 1024`.

- **Depends on:** —
- **Done when:** the test passes; a `--dump-config` shows `retina_width`/`retina_height`;
  cargo fmt/clippy/test green.

### retina-luminance-derivation — Derive Linear Luminance From Hit Color

Today each ray stores an RGBA hit color (`phase_vision.wgsl`); nothing computes
luminance. Add the linear-luminance helper the cortex pass uses to build the
Stage-0 retina `L(x,y)`.

**Steps:**
1. Add `retina_luminance(color: vec3<f32>) -> f32` to `common.wgsl` exactly as in
   ARCHITECTURE 0001 (Rec. 709 weights, linear-light, no inverse gamma).
2. Add a WGSL-level unit assertion via a Rust test
   `luminance_weights_sum_to_one` that recomputes `0.2126 + 0.7152 + 0.0722` in
   Rust and asserts it equals `1.0` within `1e-6` (guards against a typo drifting
   the weights; the WGSL constant and the test constant share one source comment).
3. Document in the helper that DC is rejected downstream (zero-sum DoG / DC-
   balanced Gabor), so no brightness-normalization pass precedes Stage 1.

- **Depends on:** —
- **Done when:** the weights test passes; `retina_luminance` compiles into the
  kernel; cargo fmt/clippy/test green.

---

## 0002 — V1 cortical pass

### visual-cortex-pass-skeleton — Insert The Cooperative Pass (Passthrough)

Today the seven passes run in sequence with barriers (`kernel_tick.wgsl:475-493`).
Insert `coop_visual_cortex` between `coop_feature_extract` and `coop_encode` as a
no-op passthrough first, behind a config flag defaulted off, so the change is
byte-identical until the stages land.

**Steps:**
1. Add `visual_cortex_enabled: bool` to `BrainConfig` (`config.rs`) with
   `#[serde(default)]` (defaults `false`) and a world-config uniform bit
   (`buffers.rs build_config_for`, `common.wgsl`). Document it as the 0005 gate
   flag.
2. Add `fn coop_visual_cortex(agent_id: u32, tid: u32)` to `brain_passes.wgsl`.
   In the passthrough version it returns immediately. Declare the workgroup
   scratch arrays from ARCHITECTURE 0002 sized by `RETINA_PIXEL_COUNT` /
   `VISUAL_FEATURE_COUNT`.
3. Insert the call + `workgroupBarrier()` into `kernel_tick.wgsl:475-493` exactly
   as in ARCHITECTURE 0002, guarded by the uniform `if (alive)` before the
   barrier.
4. Add GPU test `visual_cortex_passthrough_is_byte_identical`:

```rust
#[test]
fn visual_cortex_passthrough_is_byte_identical() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    // With visual_cortex_enabled = false, the encoded brain state after N ticks
    // must byte-match a build with the pass call removed. Run a fixed-seed world
    // for a fixed tick budget twice (flag off vs. baseline) and assert the
    // read-back encoded vectors are bit-identical.
    // (Mirror the comparison harness in split_serial_matches_fused_serial.)
}
```

- **Depends on:** —
- **Done when:** with the flag off the encoded state is bit-identical to the
  pre-task build (test passes); cargo fmt/clippy/test green.

### center-surround-dog — Stage 1 Difference-of-Gaussians (ON/OFF)

Implement the zero-sum center-surround stage that turns the retina into a local-
contrast map (Rodieck 1965; Marr & Hildreth 1980).

**Steps:**
1. In `coop_visual_cortex` (`brain_passes.wgsl`), Stage 0: all 256 threads write
   `s_retina[i] = retina_luminance(...)` for their strided pixels from the vision
   slice in `s_features`; `workgroupBarrier()`.
2. Stage 1: compute the seeded DoG kernel (σ_center from a `DOG_SIGMA_CENTER`
   constant, σ_surround = `dog_surround_ratio · σ_center` read from the heritable
   tail once 0003 lands; until then read the seed constant `1.6`). Convolve into
   `s_center_surround` (signed). Guard the kernel normalization divisor with
   `max(·, EPSILON)`. Re-impose `∑ kernel = 0` after reading the ratio.
3. Keep ON/OFF as the rectified split at the point of consumption in Stage 2
   (`r_on = max(0, v)`, `r_off = max(0, -v)`), so only the signed map is stored.
4. Add GPU test `dog_kernel_sums_to_zero` (embed the self-skip guard): build the
   seeded DoG kernel, assert `kernel.iter().sum::<f32>().abs() < 1e-5`; feed a
   uniform retina and assert the response magnitude `< 1e-4`; feed a half-bright
   / half-dark split and assert a response `> 0.1` at the boundary.

- **Depends on:** visual-cortex-pass-skeleton, retina-luminance-derivation
- **Done when:** `dog_kernel_sums_to_zero` fails if the kernel is replaced by a
  plain Gaussian (non-zero-sum) and passes with the DoG; cargo fmt/clippy/test green.

### gabor-simple-cells — Stage 2 Oriented Gabor Bank

Implement the orientation-selective simple-cell bank (Jones & Palmer 1987;
Hubel & Wiesel 1962).

**Steps:**
1. Define seed constants in `common.wgsl`: `GABOR_ORIENTATIONS` (4),
   `GABOR_SCALES` (2), `GABOR_PHASES` (2, quadrature). Derive per-filter θ as
   `i · π / GABOR_ORIENTATIONS + orientation_offset` (offset seed 0.0 until 0003),
   λ from `gabor_wavelength` (seed 5.0), σ = `0.56 · λ`, γ from
   `gabor_aspect_ratio` (seed 0.5), ψ ∈ {0, π/2}.
2. Compute each DC-balanced Gabor kernel once per pass into a small workgroup
   buffer (subtract kernel mean; re-impose after reading genes). Convolve the
   signed DoG map: `s_simple[θ,λ,ψ] = Gabor * center_surround`. Use `max(λ, EPSILON)`
   and `max(σ, EPSILON)` in the divisions.
3. Store both the linear (for Stage 3 energy) and `max(0, ·)` rectified outputs
   as needed; document which Stage 3 consumes (linear).
4. Add GPU test `gabor_kernels_are_dc_balanced`: assert every seeded Gabor kernel
   sums to `< 1e-5`. Add the orientation probe in `orientation-selectivity-probe`
   (0005); this task only asserts the kernels are well-formed.

- **Depends on:** center-surround-dog
- **Done when:** `gabor_kernels_are_dc_balanced` passes and fails if the mean-
  subtraction is removed; cargo fmt/clippy/test green.

### complex-cell-energy-pool — Stage 3 Energy + MAX Pooling

Implement phase invariance (quadrature energy, Adelson & Bergen 1985) and
position/scale tolerance (MAX pool, Riesenhuber & Poggio 1999).

**Steps:**
1. For each (θ, λ): `E = sqrt(even² + odd² )` using the **linear** Gabor outputs,
   guarded as `sqrt(max(even*even + odd*odd, 0.0))`.
2. MAX-pool `E` over a spatial neighborhood (side `ceil(2·σ)`, ~50% overlap) and
   over the scale band of the same orientation, writing
   `s_complex[orientation][scale_band][row][col]`. Pool over position and scale
   only — never over the two phases.
3. L2-normalize `s_complex` per frame: divide by `max(norm, EPSILON)`.
4. Define `VISUAL_FEATURE_COUNT = GABOR_ORIENTATIONS × GABOR_SCALES × POOL_ROWS × POOL_COLS`
   as a single canonical constant (`common.wgsl` + echoed in `buffers.rs`).
5. Add GPU test `complex_pool_output_is_nonnegative_and_normalized`: assert all
   `s_complex` values ≥ 0 and the L2 norm ≈ 1 (or 0 for a blank retina).

- **Depends on:** gabor-simple-cells
- **Done when:** the output test passes; cargo fmt/clippy/test green.

### wire-visual-features-into-encoder — Redefine The Encoder Visual Input

Today `coop_encode` reads `VISION_COLOR_COUNT + VISION_DEPTH_COUNT` raw pixels as
its leading features (`brain_passes.wgsl:178-207`). Replace that slice with the
complex-cell vector when the flag is on.

**Steps:**
1. In `BrainLayout::new` (`buffers.rs`), when `visual_cortex_enabled`, set
   `feature_count = visual_feature_count + NON_VISUAL_FEATURE_COUNT`; otherwise
   keep the legacy `color + depth + NON_VISUAL_FEATURE_COUNT`. Define
   `NON_VISUAL_FEATURE_COUNT` as the canonical constant for the 25-feature tail.
2. In `coop_visual_cortex`, after Stage 3, write
   `s_features[0 .. VISUAL_FEATURE_COUNT)` from `s_complex` and shift the non-
   visual tail to start at `VISUAL_FEATURE_COUNT`; ensure `coop_feature_extract`
   places the non-visual features at the matching offset when the flag is on.
3. Confirm `O_ENC_WEIGHTS` resizes from the new `feature_count` (it already
   derives from `FEATURE_COUNT`); grep all shaders that concatenate `common.wgsl`
   for dangling references to the old vision-width feature offsets.
4. Extend `visual_cortex_passthrough_is_byte_identical` (skeleton task) to also
   assert that with the flag **on**, `feature_count == VISUAL_FEATURE_COUNT + 25`
   and the encoder accepts the new width without a wgpu validation error.

- **Depends on:** complex-cell-energy-pool
- **Done when:** flag-off path stays byte-identical and flag-on path runs without
  validation errors at the new `feature_count`; cargo fmt/clippy/test green.

---

## 0003 — Heritable visual genome

### visual-genome-config — Add Biologically-Seeded Heritable Gabor Genes

Today the heritable tail is four scalars (`buffers.rs:80-83`,
`gpu_kernel.rs:2479-2513`). Add the four global bank genes from ARCHITECTURE 0003
with full end-to-end wiring.

**Steps:**
1. Add `gabor_wavelength`, `gabor_aspect_ratio`, `dog_surround_ratio`,
   `orientation_offset` to `BrainConfig` (`config.rs`) with `#[serde(default)]`
   fns and the seeds/clamps from ARCHITECTURE 0003. Add to `Default`, `tiny`,
   `large`.
2. Mutation (`agent/mod.rs:354-424`): add a `momentum.biased_perturb_f(...).clamp(min, max)`
   line per gene with the documented clamps. Wrap `orientation_offset` into
   `[0, π)` after perturb.
3. Crossover (`agent/mod.rs:489-547`): add ternary inheritance per gene.
4. Momentum (`momentum.rs:121-169`): add each as a `("gene_name", parent.gene)`
   tuple and the matching match arm.
5. Provenance (`governor.rs:1682-1747`): add each to `params_to_check` so
   mutation direction is recorded.
6. Tail offsets (`buffers.rs`): add `O_GABOR_WAVELENGTH`, `O_GABOR_ASPECT_RATIO`,
   `O_DOG_SURROUND_RATIO`, `O_ORIENTATION_OFFSET` after `O_MOVEMENT_SPEED`; grow
   `FIXED_TAIL_SIZE`; extend the contiguity `debug_assert_eq!`s and the `values`
   array in `write_agent_heritable_config` (`gpu_kernel.rs:2501-2512`); initialize
   in `init_brain_state_for` (`buffers.rs:534-594`) with the fixed-delta pattern.
7. UI (`ui.rs`): add a `DragValue` editor row (clamps matching mutation) and the
   four read-only display rows (mirror `movement_speed`'s surfaces).
8. Shader (`brain_passes.wgsl`): read the four genes from the tail in
   `coop_visual_cortex` and re-impose the invariants (DoG zero-sum, Gabor DC
   balance) after reading.
9. Add GPU test `heritable_visual_genes_round_trip`: write a config with a non-
   default `gabor_wavelength`, read back the agent's tail slot, assert equality.
   Add a CPU test `mutate_config_respects_visual_gene_bounds` (50 iterations from
   extreme parents) asserting each gene stays in its clamp.

- **Depends on:** complex-cell-energy-pool
- **Done when:** the round-trip and bounds tests pass and fail if any wiring site
  is omitted; cargo fmt/clippy/test green.

### retire-visual-encoding-size — Supersede The Legacy Field (Closes #106)

`visual_encoding_size` is legacy and unused (`config.rs:37-42`, five UI surfaces).
Now that structured visual config exists, retire it without breaking
deserialization.

**Steps:**
1. Keep the field in `BrainConfig` with its `#[serde(default)]` so existing JSON
   and `xagent.db` configs still load; update its doc comment to "superseded by
   plan 0008 visual-cortex config (`retina_*`, `gabor_*`); retained only for
   deserialization back-compat (issue #106)."
2. Remove its editable `DragValue` (`ui.rs:1386-1391`); leave or drop the read-
   only displays per house preference (drop them to avoid showing a dead field).
3. Remove it from breeding pass-through (`agent/mod.rs:378`, `:503`) — it no
   longer needs to be carried; the serde default supplies it on load.
4. Update the README table row and the brain README "no eyes" section to point at
   the new visual cortex.
5. Add a CPU test `legacy_config_without_visual_cortex_fields_still_loads`:
   deserialize a JSON blob that has `visual_encoding_size` but none of the new
   `retina_*`/`gabor_*` fields and assert it loads with the defaults applied.

- **Depends on:** visual-genome-config
- **Done when:** the legacy-load test passes; `visual_encoding_size` is gone from
  the UI editor; #106 referenced as resolved; cargo fmt/clippy/test green.

---

## 0004 — Runtime genome authority for visual genes

### visual-genome-runtime-authority — Apply The Visual Genome After Inheritance (GATED)

**Gate:** depends on plan 0007 `effective-agent-config-upload` landing the
`patch_agent_configs` worker path. Until then this task is blocked.

Today the interactive worker never re-applies the heritable tail after
inheritance (`sim_runtime.rs:317-331`, `sim_runtime.rs:433-445`); the headless
path does (`headless.rs:159-164`). The extended `write_agent_heritable_config`
now also carries the four visual genes, so the worker must invoke the patch path
or every agent's vision silently reverts to defaults.

**Steps:**
1. Confirm `patch_agent_configs` (from plan 0007) calls the extended
   `write_agent_heritable_config`; if 0007 wrote a fixed 4-slot block, widen it to
   the full tail block so the visual genes are included.
2. Verify both call sites (`Worker::new` after `upload_agents`,
   `reset_population` after inherited state is written).
3. Add GPU test `worker_reset_applies_visual_genome_after_inheritance`:

```rust
#[test]
fn worker_reset_applies_visual_genome_after_inheritance() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    // Two agents with distinct gabor_wavelength (e.g. 3.0 and 9.0). Force
    // inherited champion brain state into both (which overwrites the tail), run
    // the worker reset / patch path, read back both tail slots, and assert each
    // agent's O_GABOR_WAVELENGTH matches its own config — not the champion's.
}
```

- **Depends on:** visual-genome-config, `0007 effective-agent-config-upload`
- **Done when:** the test fails on the un-patched worker and passes after the
  patch covers the visual slots; cargo fmt/clippy/test green.

---

## 0005 — Red-green probes and emergence gate

### orientation-selectivity-probe — Vertical Bar Excites The Vertical Cell

This is the core Hubel & Wiesel result and the scientific crux of the plan. A
vertical luminance bar must drive the vertical-tuned simple cell far above the
horizontal-tuned one, and a swept orientation must yield a unimodal tuning curve.

**Steps:**
1. Add a test helper that writes a synthetic oriented bar into the retina at a
   given orientation (bypassing raycasts) and runs `coop_visual_cortex` for one
   agent.
2. Add GPU test `vertical_bar_excites_vertical_simple_cell`:

```rust
#[test]
fn vertical_bar_excites_vertical_simple_cell() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    // Render a vertical bar into the 32x32 retina. Read back the simple-cell
    // responses. Assert the response of the cell whose preferred orientation is
    // vertical exceeds the horizontal-tuned cell's response by a wide margin
    // (e.g. >= 3x). Then sweep the bar 0..pi in steps and assert the vertical
    // cell's response is maximal at vertical and falls monotonically away from it
    // (unimodal tuning curve).
}
```

3. Define the selectivity threshold (≥ 3× preferred-vs-orthogonal) as a named
   constant with a comment citing Hubel & Wiesel 1962.

- **Depends on:** wire-visual-features-into-encoder
- **Done when:** the test fails for an unoriented (isotropic) filter and passes
  for the seeded Gabor bank; cargo fmt/clippy/test green.

### complex-invariance-probe — Phase And Position Invariance

Complex cells must be invariant to bar phase (energy model) and tolerant to small
position shifts (MAX pool).

**Steps:**
1. Add GPU test `complex_cell_phase_invariance`: render a bar at the preferred
   orientation, read the complex response; shift the bar by half a wavelength
   (phase flip), read again. Assert the **complex** response changes by `< 10%`
   while the **simple** response changes sign (asserting the energy step is doing
   the work, not luck).
2. Add GPU test `complex_cell_position_tolerance`: shift the bar by one pixel
   within the receptive field and assert the MAX-pooled complex response changes
   by `< 15%`.
3. Cite Adelson & Bergen 1985 (energy) and Riesenhuber & Poggio 1999 (MAX) in the
   test doc comments.

- **Depends on:** complex-cell-energy-pool
- **Done when:** both invariance tests pass and `complex_cell_phase_invariance`
  fails if energy is replaced by a single-phase response; cargo fmt/clippy/test green.

### visual-cortex-throughput-baseline — Measure The Cost

Densifying the retina and adding the pass costs throughput (SCOPE finding 7). The
default-flip gate needs a number.

**Steps:**
1. Extend the agent-sweep bench (`--bench-agent-sweep`, from plan 0005) to report
   ticks-per-second with `visual_cortex_enabled` on vs. off at the default
   population and retina 32×32.
2. Record the measured on/off tps in `0008-VISUAL-CORTEX-BASELINE.md` in this
   folder (the decision artifact), including the retina resolution and bank size.

- **Depends on:** wire-visual-features-into-encoder
- **Done when:** the baseline doc records on/off tps at 32×32; cargo fmt/clippy/test green.

### visual-encoder-default-gate — Flip The Default (GATED)

**Gate:** the orientation-selectivity and complex-invariance probes pass, and the
measured throughput regression from `visual-cortex-throughput-baseline` is within
budget (define the budget as a fraction of the 0006 fused baseline, e.g. ≥ 50% of
prior tps retained, in the decision doc).

**Steps:**
1. If the gate holds, change `visual_cortex_enabled`'s default to `true` and set a
   curriculum retina default that meets the budget; record the decision and the
   retained-tps number in `0008-VISUAL-CORTEX-BASELINE.md`.
2. If the gate fails (probes fail or tps below budget), leave the default `false`,
   record why, and file the smaller follow-up (per-filter genome, lower-cost
   pooling, or reduced retina) — do not flip the default.

- **Depends on:** orientation-selectivity-probe, complex-invariance-probe,
  visual-cortex-throughput-baseline, visual-genome-config
- **Done when:** the decision doc resolves to flip-or-hold with the measured
  numbers; if flipped, the default-on path keeps cargo fmt/clippy/test green.

---

**End of plan 0008 TASKS.** When every "Done when" bullet is green, agents see
oriented, position-invariant visual structure through a biologically-seeded,
evolvable early visual cortex — proven by red-green probes, not assumed.
