//! Stage 2 of the Hubel-Wiesel visual cortex (plan 0008): the orientation-
//! selective V1 simple-cell bank (oriented Gabor filters).
//!
//! This is the canonical Rust reference for the seeded Gabor bank. It mirrors,
//! literal-for-literal, the WGSL Stage 2 in
//! `shaders/kernel/brain_passes.wgsl::coop_visual_cortex` and the seed constants
//! in `shaders/kernel/common.wgsl`. The WGSL is the production path that runs on
//! the GPU; this module exists so the `gabor_kernels_are_dc_balanced` probe can
//! assert each kernel's mathematical invariant (DC balance, i.e. `∑ Gabor = 0`)
//! on the CPU without a workgroup-memory readback. The two share one set of
//! literals — when the heritable `gabor_wavelength` / `gabor_aspect_ratio` /
//! `orientation_offset` genes land (plan 0003) they override only the
//! corresponding argument here; the construction is identical.
//!
//! The 2-D Gabor is the validated quantitative model of a V1 simple-cell
//! receptive field (Jones & Palmer 1987); the elongated alternating ON/OFF lobes
//! are Hubel & Wiesel's (1962) "aligned row of LGN inputs":
//!
//! ```text
//! x' =  x·cosθ + y·sinθ ,   y' = −x·sinθ + y·cosθ
//! Gabor(x,y) = exp( −(x'² + γ²·y'²) / (2σ²) ) · cos( 2π·x'/λ + ψ )
//! ```
//!
//! Bank seeds (single canonical source, mirrored in `common.wgsl`):
//! - orientations tiled evenly over `[0, π)`: `θ_i = i·π/N + offset` (HMAX S1,
//!   Riesenhuber & Poggio 1999), `N = GABOR_ORIENTATIONS = 4` (0, 45, 90, 135°);
//! - scales `GABOR_SCALES = 2` derived from the carrier wavelength `λ`;
//! - phases `GABOR_PHASES = 2`, a single quadrature pair `ψ ∈ {0, π/2}` (even,
//!   odd);
//! - `σ = 0.56·λ` (≈ 1-octave V1 bandwidth).
//!
//! DC balance: the raw cosine-windowed kernel carries a sizeable DC offset for
//! `ψ = 0`; subtracting the kernel mean makes `∑ Gabor = 0` so the bank responds
//! to oriented contrast, not absolute brightness — the invariant the
//! `gabor_kernels_are_dc_balanced` probe pins. Removing the mean subtraction is
//! the falsification control: the even-phase kernels are then DC-imbalanced and
//! the probe fails.
//!
//! L2 normalization: after the mean is removed each kernel is divided by its L2
//! norm so every filter has unit energy. This is the standard Gabor receptive-
//! field convention (it makes the even/odd quadrature responses commensurable for
//! the Stage-3 energy step) and, because it scales the kernel down by its norm
//! (≈ 5–10 here), it shrinks the residual f32 DC of the larger kernels well under
//! the `1e-5` balance threshold. Scaling a zero-sum vector keeps it zero-sum, so
//! normalization cannot reintroduce DC.

use std::f32::consts::PI;

/// Number of evenly-tiled orientations over `[0, π)`. Single canonical source
/// mirrored by `GABOR_ORIENTATIONS` in `common.wgsl`. Start 4 (0, 45, 90, 135°),
/// the HMAX S1 choice (Riesenhuber & Poggio 1999).
pub const GABOR_ORIENTATIONS: usize = 4;

/// Number of carrier scales (wavelength bands). Single canonical source mirrored
/// by `GABOR_SCALES` in `common.wgsl`.
pub const GABOR_SCALES: usize = 2;

/// Number of carrier phases — a single quadrature pair (even `ψ = 0`, odd
/// `ψ = π/2`). Single canonical source mirrored by `GABOR_PHASES` in
/// `common.wgsl`. Pooled over by the complex-cell energy step, never reduced.
pub const GABOR_PHASES: usize = 2;

/// Seed carrier wavelength λ in retina pixels (`gabor_wavelength` gene seed,
/// plan 0003). Single canonical source mirrored by `GABOR_WAVELENGTH_SEED` in
/// `common.wgsl`. Heritable; clamped to `[GABOR_WAVELENGTH_MIN, ..MAX]`.
pub const GABOR_WAVELENGTH_SEED: f32 = 5.0;

/// Seed envelope aspect ratio γ (long axis / short axis), `gabor_aspect_ratio`
/// gene seed (plan 0003). Single canonical source mirrored by
/// `GABOR_ASPECT_RATIO_SEED` in `common.wgsl`. Heritable; clamped to
/// `[GABOR_ASPECT_RATIO_MIN, ..MAX]`.
pub const GABOR_ASPECT_RATIO_SEED: f32 = 0.5;

/// Seed whole-bank orientation offset in radians, added to the even `[0, π)`
/// tiling (`orientation_offset` gene seed, plan 0003). Single canonical source
/// mirrored by `GABOR_ORIENTATION_OFFSET_SEED` in `common.wgsl`. Heritable;
/// wrapped to `[0, π)`.
pub const GABOR_ORIENTATION_OFFSET_SEED: f32 = 0.0;

/// Envelope sigma σ as a fraction of the carrier wavelength λ (σ = ratio·λ).
/// 0.56 gives the ≈ 1-octave spatial-frequency bandwidth measured in V1. Single
/// canonical source mirrored by `GABOR_SIGMA_LAMBDA_RATIO` in `common.wgsl`.
pub const GABOR_SIGMA_LAMBDA_RATIO: f32 = 0.56;

/// Ratio between successive scale bands' carrier wavelengths (each band's λ is
/// `GABOR_SCALE_STEP · λ_prev`). One octave per band — the standard HMAX S1
/// spacing. Single canonical source mirrored by `GABOR_SCALE_STEP` in
/// `common.wgsl`.
pub const GABOR_SCALE_STEP: f32 = 2.0;

/// Kernel truncation radius in sigmas (3σ of the envelope). Mirrors
/// `GABOR_SUPPORT_SIGMAS` in `common.wgsl`.
pub const GABOR_SUPPORT_SIGMAS: f32 = 3.0;

/// Lower clamp on the carrier wavelength λ (plan-0003 gene clamp). Mirrors the
/// clamp in `coop_visual_cortex`. Below ~2 px/cycle the carrier is undersampled
/// on the retina grid.
pub const GABOR_WAVELENGTH_MIN: f32 = 2.0;

/// Upper clamp on the carrier wavelength λ (plan-0003 gene clamp and the support-
/// bounding ceiling). Mirrors `GABOR_WAVELENGTH_MAX` in `common.wgsl`.
pub const GABOR_WAVELENGTH_MAX: f32 = 12.0;

/// Lower clamp on the envelope aspect ratio γ (plan-0003 gene clamp). Mirrors the
/// clamp in `coop_visual_cortex`.
pub const GABOR_ASPECT_RATIO_MIN: f32 = 0.25;

/// Upper clamp on the envelope aspect ratio γ (plan-0003 gene clamp). At 1.0 the
/// envelope is isotropic. Mirrors the clamp in `coop_visual_cortex`.
pub const GABOR_ASPECT_RATIO_MAX: f32 = 1.0;

/// Floor for divisions (mirrors `EPSILON` in `common.wgsl`).
const EPSILON: f32 = 1e-6;

/// A square, row-major Gabor convolution kernel of odd side `2·radius + 1`,
/// tagged with the parameters it was built from.
#[derive(Clone, Debug)]
pub struct GaborKernel {
    /// Half-width in pixels; the kernel side is `2·radius + 1`.
    pub radius: usize,
    /// Preferred orientation θ in radians.
    pub theta: f32,
    /// Carrier wavelength λ in pixels.
    pub wavelength: f32,
    /// Carrier phase ψ in radians (`0` even, `π/2` odd).
    pub phase: f32,
    /// Row-major weights, `(2·radius + 1)²` entries; DC-balanced (`∑ = 0`).
    pub weights: Vec<f32>,
}

impl GaborKernel {
    /// Kernel side length (`2·radius + 1`).
    pub fn side(&self) -> usize {
        2 * self.radius + 1
    }

    /// Weight at kernel offset `(kx, ky)` with `kx, ky ∈ [-radius, radius]`.
    pub fn at(&self, kx: i32, ky: i32) -> f32 {
        let side = self.side() as i32;
        let r = self.radius as i32;
        let col = kx + r;
        let row = ky + r;
        debug_assert!(col >= 0 && col < side && row >= 0 && row < side);
        self.weights[(row * side + col) as usize]
    }
}

/// Preferred orientation θ for bank index `i ∈ [0, GABOR_ORIENTATIONS)`:
/// `i·π/N + offset`, wrapped into `[0, π)`. Mirrors `gabor_theta` in
/// `brain_passes.wgsl`.
pub fn gabor_theta(orientation_index: usize, orientation_offset: f32) -> f32 {
    let n = (GABOR_ORIENTATIONS as f32).max(EPSILON);
    let raw = (orientation_index as f32) * PI / n + orientation_offset;
    // Wrap into [0, π): rem_euclid keeps it non-negative even for a negative
    // (mutated) offset, mirroring the WGSL wrap.
    raw.rem_euclid(PI)
}

/// Carrier wavelength λ for scale band `s ∈ [0, GABOR_SCALES)`: the base
/// wavelength scaled by one octave per band, clamped to the gene bounds. Mirrors
/// `gabor_wavelength_for_scale` in `brain_passes.wgsl`.
pub fn gabor_wavelength_for_scale(base_wavelength: f32, scale_band: usize) -> f32 {
    let base = base_wavelength.clamp(GABOR_WAVELENGTH_MIN, GABOR_WAVELENGTH_MAX);
    let lambda = base * GABOR_SCALE_STEP.powi(scale_band as i32);
    // Clamp again so the largest band cannot grow the support past the bound.
    lambda.clamp(GABOR_WAVELENGTH_MIN, GABOR_WAVELENGTH_MAX)
}

/// Carrier phase ψ for phase index `p ∈ [0, GABOR_PHASES)`: the quadrature pair
/// `{0, π/2}` (even, odd). Mirrors `gabor_phase` in `brain_passes.wgsl`.
pub fn gabor_phase(phase_index: usize) -> f32 {
    (phase_index as f32) * (PI / 2.0)
}

/// Envelope sigma σ for a carrier wavelength λ (`σ = ratio·λ`, floored). Mirrors
/// `gabor_sigma` in `brain_passes.wgsl`.
fn gabor_sigma(wavelength: f32) -> f32 {
    (GABOR_SIGMA_LAMBDA_RATIO * wavelength).max(EPSILON)
}

/// Kernel half-width in pixels for a carrier wavelength λ: 3σ of the envelope.
/// Mirrors `gabor_kernel_radius` in `brain_passes.wgsl`.
pub fn gabor_kernel_radius(wavelength: f32) -> usize {
    let sigma = gabor_sigma(wavelength);
    (GABOR_SUPPORT_SIGMAS * sigma).ceil() as usize
}

/// Raw (un-balanced) Gabor tap at integer offset `(kx, ky)` for orientation θ,
/// carrier wavelength λ, aspect ratio γ, and phase ψ. Mirrors `gabor_raw` in
/// `brain_passes.wgsl`.
fn gabor_raw(kx: i32, ky: i32, theta: f32, wavelength: f32, aspect_ratio: f32, phase: f32) -> f32 {
    let sigma = gabor_sigma(wavelength);
    let sigma_sq = (sigma * sigma).max(EPSILON);
    let gamma = aspect_ratio.clamp(GABOR_ASPECT_RATIO_MIN, GABOR_ASPECT_RATIO_MAX);
    let lambda = wavelength.max(EPSILON);
    let x = kx as f32;
    let y = ky as f32;
    let (sin_t, cos_t) = theta.sin_cos();
    let x_rot = x * cos_t + y * sin_t;
    let y_rot = -x * sin_t + y * cos_t;
    let envelope = (-(x_rot * x_rot + gamma * gamma * y_rot * y_rot) / (2.0 * sigma_sq)).exp();
    let carrier = (2.0 * PI * x_rot / lambda + phase).cos();
    envelope * carrier
}

/// Build a single DC-balanced, unit-energy Gabor kernel for the given
/// parameters. The center sigma derives from `wavelength`; the aspect ratio and
/// orientation are clamped/wrapped exactly as the WGSL does after reading the
/// (eventual) heritable genes. The kernel mean is subtracted so `∑ weights = 0`
/// (the DC-balance invariant), then the weights are L2-normalized to unit energy.
/// Mirrors the per-filter kernel build in `coop_visual_cortex`.
pub fn build_gabor_kernel(
    theta: f32,
    wavelength: f32,
    aspect_ratio: f32,
    phase: f32,
) -> GaborKernel {
    let lambda = wavelength.clamp(GABOR_WAVELENGTH_MIN, GABOR_WAVELENGTH_MAX);
    let radius = gabor_kernel_radius(lambda);
    let side = 2 * radius + 1;
    let entries = side * side;

    let mut weights = vec![0.0_f32; entries];
    let mut raw_sum = 0.0_f32;
    for (k, w) in weights.iter_mut().enumerate() {
        let kx = (k % side) as i32 - radius as i32;
        let ky = (k / side) as i32 - radius as i32;
        let tap = gabor_raw(kx, ky, theta, lambda, aspect_ratio, phase);
        *w = tap;
        raw_sum += tap;
    }
    // DC balance: subtract the kernel mean so ∑ weights = 0. Removing this loop is
    // the falsification control for `gabor_kernels_are_dc_balanced`.
    let mean = raw_sum / (entries as f32).max(EPSILON);
    for w in weights.iter_mut() {
        *w -= mean;
    }
    // L2-normalize to unit energy (standard Gabor convention; keeps even/odd
    // responses commensurable for the Stage-3 quadrature energy and shrinks the
    // residual f32 DC of the larger kernels below the 1e-5 balance threshold).
    // Scaling a zero-sum vector stays zero-sum, so this cannot reintroduce DC.
    let norm = weights
        .iter()
        .map(|w| w * w)
        .sum::<f32>()
        .sqrt()
        .max(EPSILON);
    for w in weights.iter_mut() {
        *w /= norm;
    }
    GaborKernel {
        radius,
        theta,
        wavelength: lambda,
        phase,
        weights,
    }
}

/// The full seeded Gabor bank: every `(orientation, scale, phase)` filter built
/// from the seed constants. Length is
/// `GABOR_ORIENTATIONS × GABOR_SCALES × GABOR_PHASES`. Filters are ordered
/// orientation-major, then scale, then phase — the same nesting the WGSL Stage 2
/// loop walks.
pub fn seeded_gabor_bank() -> Vec<GaborKernel> {
    let mut bank = Vec::with_capacity(GABOR_ORIENTATIONS * GABOR_SCALES * GABOR_PHASES);
    for orientation_index in 0..GABOR_ORIENTATIONS {
        let theta = gabor_theta(orientation_index, GABOR_ORIENTATION_OFFSET_SEED);
        for scale_band in 0..GABOR_SCALES {
            let wavelength = gabor_wavelength_for_scale(GABOR_WAVELENGTH_SEED, scale_band);
            for phase_index in 0..GABOR_PHASES {
                let phase = gabor_phase(phase_index);
                bank.push(build_gabor_kernel(
                    theta,
                    wavelength,
                    GABOR_ASPECT_RATIO_SEED,
                    phase,
                ));
            }
        }
    }
    bank
}

/// Valid-region (zero-padded border) convolution of a `width × height` signed
/// field (the Stage-1 DoG map) with a [`GaborKernel`], returning the linear
/// simple-cell response. Mirrors the WGSL convolution loop in
/// `coop_visual_cortex` Stage 2.
pub fn convolve(kernel: &GaborKernel, field: &[f32], width: usize, height: usize) -> Vec<f32> {
    assert_eq!(field.len(), width * height, "field size mismatch");
    let radius = kernel.radius as i32;
    let mut out = vec![0.0_f32; width * height];
    for prow in 0..height as i32 {
        for pcol in 0..width as i32 {
            let mut acc = 0.0_f32;
            for ky in -radius..=radius {
                let sr = prow + ky;
                if sr < 0 || sr >= height as i32 {
                    continue;
                }
                for kx in -radius..=radius {
                    let sc = pcol + kx;
                    if sc < 0 || sc >= width as i32 {
                        continue;
                    }
                    let pidx = (sr as usize) * width + sc as usize;
                    acc += kernel.at(kx, ky) * field[pidx];
                }
            }
            out[(prow as usize) * width + pcol as usize] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_bank_has_expected_filter_count() {
        let bank = seeded_gabor_bank();
        assert_eq!(bank.len(), GABOR_ORIENTATIONS * GABOR_SCALES * GABOR_PHASES);
    }

    #[test]
    fn every_seeded_kernel_is_dc_balanced() {
        for kernel in seeded_gabor_bank() {
            let sum: f32 = kernel.weights.iter().sum();
            assert!(
                sum.abs() < 1e-5,
                "seeded Gabor kernel (θ={}, λ={}, ψ={}) must be DC-balanced, got {sum}",
                kernel.theta,
                kernel.wavelength,
                kernel.phase
            );
        }
    }

    #[test]
    fn orientations_tile_evenly_over_half_circle() {
        // 0, π/4, π/2, 3π/4 for the seeded offset 0.
        let expected = [0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0];
        for (i, &want) in expected.iter().enumerate() {
            let got = gabor_theta(i, GABOR_ORIENTATION_OFFSET_SEED);
            assert!(
                (got - want).abs() < 1e-5,
                "orientation {i}: expected {want}, got {got}"
            );
        }
    }

    #[test]
    fn scale_bands_step_by_one_octave() {
        let lo = gabor_wavelength_for_scale(GABOR_WAVELENGTH_SEED, 0);
        let hi = gabor_wavelength_for_scale(GABOR_WAVELENGTH_SEED, 1);
        assert!((lo - GABOR_WAVELENGTH_SEED).abs() < 1e-5);
        // Second band is one octave up, still inside the clamp (5 → 10 ≤ 12).
        assert!((hi - GABOR_WAVELENGTH_SEED * GABOR_SCALE_STEP).abs() < 1e-5);
    }

    #[test]
    fn quadrature_phases_are_zero_and_half_pi() {
        assert!((gabor_phase(0) - 0.0).abs() < 1e-6);
        assert!((gabor_phase(1) - PI / 2.0).abs() < 1e-6);
    }

    #[test]
    fn wgsl_gabor_constants_match_rust() {
        // Drift guard: the WGSL seed constants in common.wgsl are the production
        // copy; these Rust constants mirror them for the CPU probe. common.wgsl
        // is concatenated verbatim into every pipeline (gpu_kernel.rs), so a
        // matching literal here == the same value compiled into the GPU pass.
        // Mirrors `dog::tests::wgsl_dog_constants_match_rust`.
        let common_src = include_str!("shaders/kernel/common.wgsl");
        let checks: &[(&str, &str)] = &[
            ("const GABOR_ORIENTATIONS: u32 = 4u;", "GABOR_ORIENTATIONS"),
            ("const GABOR_SCALES: u32 = 2u;", "GABOR_SCALES"),
            ("const GABOR_PHASES: u32 = 2u;", "GABOR_PHASES"),
            (
                "const GABOR_WAVELENGTH_SEED: f32 = 5.0;",
                "GABOR_WAVELENGTH_SEED",
            ),
            (
                "const GABOR_ASPECT_RATIO_SEED: f32 = 0.5;",
                "GABOR_ASPECT_RATIO_SEED",
            ),
            (
                "const GABOR_ORIENTATION_OFFSET_SEED: f32 = 0.0;",
                "GABOR_ORIENTATION_OFFSET_SEED",
            ),
            (
                "const GABOR_SIGMA_LAMBDA_RATIO: f32 = 0.56;",
                "GABOR_SIGMA_LAMBDA_RATIO",
            ),
            ("const GABOR_SCALE_STEP: f32 = 2.0;", "GABOR_SCALE_STEP"),
            (
                "const GABOR_SUPPORT_SIGMAS: f32 = 3.0;",
                "GABOR_SUPPORT_SIGMAS",
            ),
        ];
        for (needle, name) in checks {
            assert!(
                common_src.contains(needle),
                "common.wgsl must contain `{needle}` so WGSL {name} matches Rust"
            );
        }
        assert_eq!(GABOR_ORIENTATIONS, 4);
        assert_eq!(GABOR_SCALES, 2);
        assert_eq!(GABOR_PHASES, 2);
        assert_eq!(GABOR_WAVELENGTH_SEED, 5.0);
        assert_eq!(GABOR_ASPECT_RATIO_SEED, 0.5);
        assert_eq!(GABOR_ORIENTATION_OFFSET_SEED, 0.0);
        assert_eq!(GABOR_SIGMA_LAMBDA_RATIO, 0.56);
        assert_eq!(GABOR_SCALE_STEP, 2.0);
        assert_eq!(GABOR_SUPPORT_SIGMAS, 3.0);
    }
}
