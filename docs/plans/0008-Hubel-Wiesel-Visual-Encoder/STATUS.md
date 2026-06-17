# Plan 0008 - Hubel-Wiesel Visual Encoder - status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete (mechanical success; encoder default **HELD** on
throughput). All 14 tasks landed and squash-merged to `develop`. The visual
cortex is implemented and **probe-proven correct**, shipped behind
`visual_cortex_enabled` (default `false`) because the throughput gate failed.
_Last updated: 2026-06-17, against `develop` (plan 0008 squash-merged)._

- **Goal:** Give the brain a biologically-grounded early-visual-cortex front end
  — dense luminance retina → Difference-of-Gaussians center-surround → oriented
  Gabor simple cells → quadrature-energy + MAX-pooled complex cells — with
  heritable, biologically-seeded receptive fields, retiring the legacy
  `visual_encoding_size` and gating the switch-over on red-green
  orientation/invariance probes.
- **Measured baseline (current code):** vision is an `8 × 6 = 48`-ray category-
  color + depth raycast field (`phase_vision.wgsl`, `common.wgsl:13-14`), fed
  directly into the single dense encoder `coop_encode` (`brain_passes.wgsl:178-207`)
  with no center-surround, orientation, or invariance stage. `visual_encoding_size`
  is a legacy unused gene (`config.rs:37-42`, issue #106). The heritable tail
  holds four scalars (`buffers.rs:80-83`) and the interactive worker does not
  re-apply it after inheritance (plan 0007 finding: `sim_runtime.rs:317-331`,
  `sim_runtime.rs:433-445`).
- **Scientific anchors (web-verified):** DoG center-surround (Rodieck 1965) at
  σ_surround:σ_center ≈ 1.6 (Marr & Hildreth 1980); Gabor simple cells validated
  to noise-level residual (Jones & Palmer 1987); complex-cell phase invariance via
  squared quadrature energy (Adelson & Bergen 1985) and position invariance via
  MAX pooling (HMAX C1, Riesenhuber & Poggio 1999); simple→complex hierarchy and
  orientation columns (Hubel & Wiesel 1959, 1962, 1968).
- **Outcome:** **All 14 tasks landed.** A new cooperative `coop_visual_cortex`
  pass (DoG → 4×2×2 oriented Gabor bank → quadrature-energy + 4×4 MAX-pool → 128
  complex features) runs on a configurable dense luminance retina (curriculum
  default 32×32) between `coop_feature_extract` and `coop_encode`; four heritable
  global-bank genes (`gabor_wavelength`, `gabor_aspect_ratio`, `dog_surround_ratio`,
  `orientation_offset`) were wired end-to-end; `visual_encoding_size` is retired
  (closes #106); the live worker re-applies the genome via plan 0007's
  `patch_agent_configs`. **The filters fire correctly — proven, not assumed:** a
  vertical bar drives the vertical-tuned simple cell **6.97×** the orthogonal cell
  (threshold 3×) with a unimodal tuning curve (isotropic control = 1.0×); complex
  cells are phase-invariant to **~1.6e-7** under a half-λ shift (threshold <10%)
  and position-tolerant to **~7.8%** under a 1px shift (threshold <15%, 8px
  control breaks at 39.8%). **But the encoder-input default was HELD:** at retina
  32×32 / 128-feature bank the cortex-ON arm retains only **~0.24%** of the 0006
  fused baseline (≈81 vs ≈34,000 tps, ≈420× slower) — far below the ≥50% budget —
  so `visual_cortex_enabled` stays **default `false`** (the gate is a conjunction;
  one failing condition holds). The cortex ships complete and behind the flag,
  available per-batch for experiments. This is the SCOPE-declared **mechanical**
  success (filters correct), explicitly *not* an evolutionary fitness gain.
  Follow-up (cheapest-first): shrink the retina, cheaper pooling / smaller bank,
  separable convolution — each re-runs this same gate. Per-filter genome widening
  remains separately gated on a demonstrated evolutionary signal. Decision
  artifact: [`0008-VISUAL-CORTEX-BASELINE.md`](0008-VISUAL-CORTEX-BASELINE.md).

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Retinal luminance front end | `retina-resolution-config`, `retina-luminance-derivation` | ✅ Done |
| 0002 | V1 cortical pass | `visual-cortex-pass-skeleton`, `center-surround-dog`, `gabor-simple-cells`, `complex-cell-energy-pool`, `wire-visual-features-into-encoder` | ✅ Done |
| 0003 | Heritable visual genome | `visual-genome-config`, `retire-visual-encoding-size` | ✅ Done |
| 0004 | Runtime genome authority for visual genes | `visual-genome-runtime-authority` (gate met: 0007 `patch_agent_configs` on `develop`) | ✅ Done |
| 0005 | Red-green probes and emergence gate | `orientation-selectivity-probe` ✅, `complex-invariance-probe` ✅, `visual-cortex-throughput-baseline` ✅, `visual-encoder-default-gate` ✅ (resolved: **HOLD**) | ✅ Done |

## Verification

Full-workspace gates green on the squashed source (isolated target dir, real
Metal adapter — GPU probes ran, no self-skip): `cargo fmt --all -- --check`
clean; `cargo clippy --workspace --all-targets -- -D warnings` clean; `cargo
test --workspace --no-fail-fast` = 68 brain + 88 sandbox-lib + 14 bin + 83
integration + 7 doctest, **0 failed**. The orientation-selectivity,
phase-invariance, and position-tolerance probes (plus the stage-well-formedness
probes `dog_kernel_sums_to_zero`, `gabor_kernels_are_dc_balanced`,
`complex_pool_output_is_nonnegative_and_normalized`,
`visual_cortex_passthrough_is_byte_identical`) all pass with margin.
