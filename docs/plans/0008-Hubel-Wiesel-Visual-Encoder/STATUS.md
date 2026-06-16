# Plan 0008 - Hubel-Wiesel Visual Encoder - status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** Planned. Authored 2026-06-16 from a code audit of the current vision
pipeline plus a web-verified survey of the early-visual-cortex literature; no
implementation tasks have landed yet.
_Last updated: 2026-06-16, against `develop`._

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
- **Outcome:** Planned; no code outcome yet.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Retinal luminance front end | `retina-resolution-config`, `retina-luminance-derivation` | Planned |
| 0002 | V1 cortical pass | `visual-cortex-pass-skeleton`, `center-surround-dog`, `gabor-simple-cells`, `complex-cell-energy-pool`, `wire-visual-features-into-encoder` | Planned |
| 0003 | Heritable visual genome | `visual-genome-config`, `retire-visual-encoding-size` | Planned |
| 0004 | Runtime genome authority for visual genes | `visual-genome-runtime-authority` (blocked on plan 0007 `effective-agent-config-upload`) | Planned |
| 0005 | Red-green probes and emergence gate | `orientation-selectivity-probe`, `complex-invariance-probe`, `visual-cortex-throughput-baseline`, `visual-encoder-default-gate` | Planned |

## Verification

Documentation-only authoring so far. Code gates have not been run for this plan
yet.
