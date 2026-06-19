//! Stage 3 of the Hubel-Wiesel visual cortex: the position- and
//! phase-invariant V1 complex cells (quadrature energy + MAX pooling).
//!
//! This is the canonical Rust reference for the complex-cell stage. It mirrors,
//! literal-for-literal, the WGSL Stage 3 in
//! `shaders/kernel/brain_passes.wgsl::coop_visual_cortex` and the bank/pool
//! constants in `shaders/kernel/common.wgsl`. The WGSL is the production path
//! that runs on the GPU; this module exists so the
//! `complex_pool_output_is_nonnegative_and_normalized` probe (and the 0005
//! invariance probes) can assert the stage's mathematical invariants on the CPU
//! without a workgroup-memory readback.
//!
//! Two biological mechanisms compose here:
//!
//! 1. **Phase invariance — quadrature energy** (Adelson & Bergen 1985). The even
//!    (`ψ = 0`) and odd (`ψ = π/2`) Gabor simple-cell responses to the signed
//!    DoG map form a quadrature pair; their squared sum
//!    `E_{θ,λ}(x,y) = sqrt(even² + odd²)` is invariant to the carrier phase, so a
//!    light/dark bar and its phase-flipped (dark/light) counterpart drive the
//!    same energy. Squaring supplies non-negativity, so the linear (un-rectified)
//!    Gabor outputs are used — rectifying first would double-count.
//!
//! 2. **Position/scale tolerance — MAX pooling** (HMAX C1, Riesenhuber & Poggio
//!    1999). `E` is MAX-pooled over a coarse `POOL_ROWS × POOL_COLS` spatial grid
//!    with ~50% overlapping cells, so an oriented feature shifted by a pixel
//!    within a cell leaves the pooled response ≈ constant. Pooling is over
//!    **position only** here — the two phases are already collapsed by the energy
//!    step (never pool over phase), and each scale keeps its own output slot.
//!
//! ### Scale banding
//!
//! ARCHITECTURE 0002 phrases the pool as "MAX over (Δx,Δy)∈pool, over scale
//! band", while the output length is `orientations × scales × pool_rows ×
//! pool_cols` — scale is an output dimension, not pooled away. With
//! `GABOR_SCALES = 2` the faithful reconciliation (and the one the
//! `VISUAL_FEATURE_COUNT` constant pins) is one scale band per scale: each
//! output `[orientation][scale][row][col]` is the spatial MAX of the energy
//! `E_{θ,λ}` for that single (orientation, scale). A scale band wider than one
//! scale is a future widening of `GABOR_SCALES`; the per-scale slot stays the
//! canonical output shape.
//!
//! ### Output normalization
//!
//! Every pooled value is `≥ 0` (a square root of a sum of squares). The whole
//! `VISUAL_FEATURE_COUNT`-vector is then L2-normalized per frame (divide by
//! `max(‖·‖, EPSILON)`), mirroring V1 response normalization. A blank retina
//! produces an all-zero vector whose norm is `< EPSILON`, so the guarded divide
//! leaves it all-zero (not NaN). This is the invariant the
//! `complex_pool_output_is_nonnegative_and_normalized` probe pins.

use crate::gabor::{
    self, gabor_phase, gabor_theta, gabor_wavelength_for_scale, GaborKernel, GABOR_ORIENTATIONS,
    GABOR_ORIENTATION_OFFSET_SEED, GABOR_SCALES, GABOR_WAVELENGTH_SEED,
};

/// Number of pooled rows in the complex-cell output grid. Single canonical
/// source mirrored by `POOL_ROWS` in `common.wgsl`. The standard HMAX C1 grid.
pub const POOL_ROWS: usize = 4;

/// Number of pooled columns in the complex-cell output grid. Single canonical
/// source mirrored by `POOL_COLS` in `common.wgsl`.
pub const POOL_COLS: usize = 4;

/// Length of the complex-cell feature vector the visual cortex emits:
/// `GABOR_ORIENTATIONS × GABOR_SCALES × POOL_ROWS × POOL_COLS`. Single canonical
/// source mirrored by `VISUAL_FEATURE_COUNT` in `common.wgsl` (derived there from
/// the same factors, not a bare literal) and echoed/validated against
/// `BrainLayout`.
pub const VISUAL_FEATURE_COUNT: usize = GABOR_ORIENTATIONS * GABOR_SCALES * POOL_ROWS * POOL_COLS;

/// Index of the even (`ψ = 0`) member of each quadrature phase pair, matching the
/// `phase_index` ordering in `seeded_gabor_bank` / the WGSL Stage 2 loop.
const PHASE_EVEN: usize = 0;
/// Index of the odd (`ψ = π/2`) member of each quadrature phase pair.
const PHASE_ODD: usize = 1;

/// Floor for divisions (mirrors `EPSILON` in `common.wgsl`).
const EPSILON: f32 = 1e-6;

/// Inclusive `[lo, hi]` pixel bounds (along one axis) of pool cell `cell` of
/// `cells` over a retina dimension of `extent` pixels. Each nominal block is
/// `extent / cells` wide; the bounds are widened by a half-block margin on each
/// side so adjacent cells overlap ~50% (HMAX C1 overlapping pooling), then
/// clamped to `[0, extent)`. Mirrors `pool_bounds` in `brain_passes.wgsl`.
///
/// Returns `(lo, hi)` with `lo <= hi`, both valid indices into `0..extent`.
fn pool_bounds(cell: usize, cells: usize, extent: usize) -> (usize, usize) {
    // Float block size (guarded). `cells >= 1` by construction (POOL_* are 4).
    let block = (extent as f32) / (cells as f32).max(EPSILON);
    let margin = block * 0.5; // half-block ⇒ ~50% overlap with each neighbour.
    let start = (cell as f32) * block - margin;
    let end = ((cell + 1) as f32) * block + margin;
    // Clamp into [0, extent-1] in the SAME order as the WGSL `pool_bounds` so the
    // two paths are literal mirrors (`extent >= 1` for any real retina).
    let last = extent.max(1) - 1;
    let lo = (start.floor().max(0.0) as usize).min(last);
    let hi = ((end.ceil().max(0.0) as usize).max(lo)).min(last);
    (lo, hi)
}

/// Per-pixel quadrature energy `E_{θ,λ}(x,y) = sqrt(max(even² + odd², 0))` for one
/// (orientation, scale), built from the even/odd linear Gabor convolutions of the
/// signed DoG map. Mirrors the on-the-fly even/odd convolution + energy in the
/// WGSL Stage 3. `even` and `odd` are the linear (un-rectified) responses of
/// `convolve` over the `width × height` DoG `field`.
pub fn quadrature_energy(even: &[f32], odd: &[f32], width: usize, height: usize) -> Vec<f32> {
    assert_eq!(even.len(), width * height, "even map size mismatch");
    assert_eq!(odd.len(), width * height, "odd map size mismatch");
    let mut energy = vec![0.0_f32; width * height];
    for (e, (&ev, &od)) in energy.iter_mut().zip(even.iter().zip(odd.iter())) {
        // sqrt(max(·, 0)) guards the f32 sum-of-squares against a negative
        // rounding residual (it is mathematically ≥ 0).
        *e = (ev * ev + od * od).max(0.0).sqrt();
    }
    energy
}

/// MAX-pool one (orientation, scale) energy map `E` over the `POOL_ROWS ×
/// POOL_COLS` overlapping spatial grid, appending the `POOL_ROWS × POOL_COLS`
/// pooled values (row-major) to `out`. Each pooled value is the maximum energy
/// over the cell's (overlapping) pixel block. Mirrors the pooling loop in the
/// WGSL Stage 3. Pools over **position only** — the energy step already
/// collapsed phase, and each scale keeps its own slots.
fn max_pool_into(energy: &[f32], width: usize, height: usize, out: &mut Vec<f32>) {
    assert_eq!(energy.len(), width * height, "energy size mismatch");
    for pool_row in 0..POOL_ROWS {
        let (r_lo, r_hi) = pool_bounds(pool_row, POOL_ROWS, height);
        for pool_col in 0..POOL_COLS {
            let (c_lo, c_hi) = pool_bounds(pool_col, POOL_COLS, width);
            let mut peak = 0.0_f32; // energy is ≥ 0, so 0 is a valid identity.
            for row in r_lo..=r_hi {
                for col in c_lo..=c_hi {
                    peak = peak.max(energy[row * width + col]);
                }
            }
            out.push(peak);
        }
    }
}

/// L2-normalize a complex-cell feature vector in place by `max(‖·‖, EPSILON)`.
/// A blank (all-zero) vector has norm `< EPSILON`, so the guarded divide leaves
/// it all-zero rather than producing NaN. Mirrors the per-frame normalization in
/// the WGSL Stage 3.
pub fn l2_normalize(features: &mut [f32]) {
    let norm = features.iter().map(|v| v * v).sum::<f32>().sqrt();
    let denom = norm.max(EPSILON);
    for v in features.iter_mut() {
        *v /= denom;
    }
}

/// Run the full Stage 3 complex-cell pipeline over a signed DoG `field`
/// (`width × height`, the Stage-1 center-surround map), returning the
/// L2-normalized `VISUAL_FEATURE_COUNT`-vector. For each (orientation, scale) it
/// convolves the even and odd seeded Gabor kernels, forms the quadrature energy,
/// MAX-pools it over the `POOL_ROWS × POOL_COLS` overlapping grid, and writes the
/// `[orientation][scale][row][col]` block; the whole vector is then L2-normalized.
/// Mirrors `coop_visual_cortex` Stage 3.
///
/// Output ordering is orientation-major → scale → pool-row → pool-col, matching
/// the WGSL filter/output nesting and the `seeded_gabor_bank` order.
pub fn complex_features(field: &[f32], width: usize, height: usize) -> Vec<f32> {
    assert_eq!(field.len(), width * height, "DoG field size mismatch");
    let mut out: Vec<f32> = Vec::with_capacity(VISUAL_FEATURE_COUNT);
    for orientation_index in 0..GABOR_ORIENTATIONS {
        let theta = gabor_theta(orientation_index, GABOR_ORIENTATION_OFFSET_SEED);
        for scale_band in 0..GABOR_SCALES {
            let wavelength = gabor_wavelength_for_scale(GABOR_WAVELENGTH_SEED, scale_band);
            let even = even_odd_kernel(theta, wavelength, PHASE_EVEN);
            let odd = even_odd_kernel(theta, wavelength, PHASE_ODD);
            let even_map = gabor::convolve(&even, field, width, height);
            let odd_map = gabor::convolve(&odd, field, width, height);
            let energy = quadrature_energy(&even_map, &odd_map, width, height);
            max_pool_into(&energy, width, height, &mut out);
        }
    }
    debug_assert_eq!(out.len(), VISUAL_FEATURE_COUNT);
    l2_normalize(&mut out);
    out
}

/// Build the even (`phase_index = 0`) or odd (`phase_index = 1`) seeded Gabor
/// kernel for one (orientation θ, wavelength λ) using the seed aspect ratio.
/// Mirrors the per-filter kernel build in the WGSL Stage 3.
fn even_odd_kernel(theta: f32, wavelength: f32, phase_index: usize) -> GaborKernel {
    gabor::build_gabor_kernel(
        theta,
        wavelength,
        gabor::GABOR_ASPECT_RATIO_SEED,
        gabor_phase(phase_index),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The output length is exactly the product of the bank and pool factors —
    /// the canonical `VISUAL_FEATURE_COUNT` the WGSL scratch and (eventually) the
    /// encoder width derive from.
    #[test]
    fn feature_count_is_product_of_bank_and_pool() {
        assert_eq!(
            VISUAL_FEATURE_COUNT,
            GABOR_ORIENTATIONS * GABOR_SCALES * POOL_ROWS * POOL_COLS
        );
        assert_eq!(VISUAL_FEATURE_COUNT, 128);
    }

    /// Quadrature energy is non-negative and phase-invariant: an even/odd pair
    /// and its sign-flipped (phase-flipped) counterpart produce identical energy.
    #[test]
    fn energy_is_nonnegative_and_phase_invariant() {
        let even = vec![0.6_f32, -0.4, 0.0, 1.0];
        let odd = vec![-0.8_f32, 0.3, 0.0, -0.5];
        let e = quadrature_energy(&even, &odd, 2, 2);
        for v in &e {
            assert!(*v >= 0.0, "energy must be non-negative, got {v}");
        }
        // Phase flip: negate both members (a half-wavelength carrier shift flips
        // the sign of both linear responses). Energy is unchanged.
        let even_flip: Vec<f32> = even.iter().map(|v| -v).collect();
        let odd_flip: Vec<f32> = odd.iter().map(|v| -v).collect();
        let e_flip = quadrature_energy(&even_flip, &odd_flip, 2, 2);
        for (a, b) in e.iter().zip(e_flip.iter()) {
            assert!((a - b).abs() < 1e-6, "energy must be phase-invariant");
        }
    }

    /// L2 normalization makes a non-blank vector unit-norm and leaves a blank
    /// (all-zero) vector all-zero (guarded divide — no NaN).
    #[test]
    fn l2_normalize_unit_norm_and_blank_safe() {
        let mut v = vec![3.0_f32, 0.0, 4.0]; // ‖·‖ = 5
        l2_normalize(&mut v);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "non-blank vector must be unit norm"
        );

        let mut blank = vec![0.0_f32; 8];
        l2_normalize(&mut blank);
        for x in &blank {
            assert_eq!(*x, 0.0, "blank vector stays all-zero (no NaN)");
        }
    }

    /// Pool bounds stay inside the retina, are non-empty, and overlap their
    /// neighbours (the ~50% overlap that gives position tolerance).
    #[test]
    fn pool_bounds_are_valid_and_overlapping() {
        let extent = 32;
        let mut prev_hi: Option<usize> = None;
        for cell in 0..POOL_COLS {
            let (lo, hi) = pool_bounds(cell, POOL_COLS, extent);
            assert!(lo <= hi, "cell {cell}: lo {lo} must be <= hi {hi}");
            assert!(
                hi < extent,
                "cell {cell}: hi {hi} must be < extent {extent}"
            );
            if let Some(p) = prev_hi {
                // Overlapping cells: this cell starts at or before the previous
                // cell's last pixel.
                assert!(lo <= p, "cell {cell}: lo {lo} must overlap prev hi {p}");
            }
            prev_hi = Some(hi);
        }
    }

    /// A blank DoG field yields the all-zero complex vector (norm 0); a non-blank
    /// field yields a unit-norm, non-negative vector — the
    /// `complex_pool_output_is_nonnegative_and_normalized` invariant, exercised on
    /// the CPU mirror.
    #[test]
    fn complex_features_blank_and_nonblank() {
        let (w, h) = (32, 32);

        let blank = vec![0.0_f32; w * h];
        let blank_out = complex_features(&blank, w, h);
        assert_eq!(blank_out.len(), VISUAL_FEATURE_COUNT);
        for v in &blank_out {
            assert_eq!(*v, 0.0, "blank field must give all-zero complex output");
        }

        // Vertical contrast edge in the DoG map (odd-symmetric lobes about the
        // mid column) — a real oriented structure.
        let mut field = vec![0.0_f32; w * h];
        for row in 0..h {
            for col in 0..w {
                field[row * w + col] = if col >= w / 2 { 1.0 } else { -1.0 };
            }
        }
        let out = complex_features(&field, w, h);
        assert_eq!(out.len(), VISUAL_FEATURE_COUNT);
        for v in &out {
            assert!(*v >= 0.0, "complex output must be non-negative, got {v}");
        }
        let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "non-blank complex output must be L2-normalized, got norm {norm}"
        );
    }

    /// Drift guard: the WGSL pool/feature constants in common.wgsl are the
    /// production copy; these Rust constants mirror them. Mirrors
    /// `gabor::tests::wgsl_gabor_constants_match_rust`.
    #[test]
    fn wgsl_complex_constants_match_rust() {
        let common_src = include_str!("shaders/kernel/common.wgsl");
        assert!(
            common_src.contains("const POOL_ROWS: u32 = 4u;"),
            "common.wgsl POOL_ROWS must match Rust POOL_ROWS ({POOL_ROWS})"
        );
        assert!(
            common_src.contains("const POOL_COLS: u32 = 4u;"),
            "common.wgsl POOL_COLS must match Rust POOL_COLS ({POOL_COLS})"
        );
        // VISUAL_FEATURE_COUNT is derived from the bank/pool factors in both
        // places (not a bare literal), so assert the derivation expression is
        // present rather than a hardcoded number.
        assert!(
            common_src.contains("GABOR_ORIENTATIONS * GABOR_SCALES * POOL_ROWS * POOL_COLS"),
            "common.wgsl VISUAL_FEATURE_COUNT must derive from the bank/pool factors"
        );
        assert_eq!(POOL_ROWS, 4);
        assert_eq!(POOL_COLS, 4);
        assert_eq!(VISUAL_FEATURE_COUNT, 128);
    }
}
