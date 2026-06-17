//! Stage 1 of the Hubel-Wiesel visual cortex (plan 0008): the center-surround
//! Difference-of-Gaussians (DoG) kernel.
//!
//! This is the canonical Rust reference for the seeded DoG kernel. It mirrors,
//! literal-for-literal, the WGSL Stage 1 in
//! `shaders/kernel/brain_passes.wgsl::coop_visual_cortex` and the seed constants
//! in `shaders/kernel/common.wgsl`. The WGSL is the production path that runs on
//! the GPU; this module exists so the `dog_kernel_sums_to_zero` probe can assert
//! the kernel's mathematical invariants (zero-sum, edge response) on the CPU
//! without a workgroup-memory readback. The two share one set of literals — when
//! the heritable `dog_surround_ratio` gene lands (plan 0003) it overrides only
//! the surround ratio argument here; the construction is identical.
//!
//! A DoG is a zero-sum concentric kernel that reports local contrast (an edge
//! operator), Rodieck 1965; Marr & Hildreth 1980:
//!
//! ```text
//! G(x,y;σ) = (1 / (2π σ²)) · exp( −(x² + y²) / (2σ²) )
//! DoG(x,y) = G(x,y;σ_center) − G(x,y;σ_surround),   σ_surround = ratio·σ_center
//! ```
//!
//! Unit-volume Gaussians with equal weights make `∑ DoG ≈ 0`; the mean is
//! subtracted after building so `∑ kernel = 0` exactly (independent of
//! truncation error). A plain Gaussian (no surround subtraction) is non-zero-sum
//! and is rejected by `dog_kernel_sums_to_zero`.

/// Seed center Gaussian sigma in retina pixels. Single canonical source mirrored
/// by `DOG_SIGMA_CENTER` in `common.wgsl`.
pub const DOG_SIGMA_CENTER: f32 = 1.0;

/// Seed surround:center sigma ratio (Marr & Hildreth 1980). Single canonical
/// source mirrored by `DOG_SURROUND_RATIO_SEED` in `common.wgsl`. Becomes the
/// heritable `dog_surround_ratio` gene in plan 0003.
pub const DOG_SURROUND_RATIO_SEED: f32 = 1.6;

/// Kernel truncation radius in sigmas (3σ of the larger Gaussian). Mirrors
/// `DOG_SUPPORT_SIGMAS` in `common.wgsl`.
pub const DOG_SUPPORT_SIGMAS: f32 = 3.0;

/// Lower clamp on the surround ratio (matches the plan-0003 gene clamp): below
/// this the kernel degenerates into a blur rather than an edge operator. Mirrors
/// the clamp in `coop_visual_cortex`.
pub const DOG_SURROUND_RATIO_MIN: f32 = 1.2;

/// Upper clamp on the surround ratio (matches the plan-0003 gene clamp and the
/// `DOG_SURROUND_RATIO_MAX` support-bounding constant in `common.wgsl`).
pub const DOG_SURROUND_RATIO_MAX: f32 = 3.0;

/// Floor for divisions (mirrors `EPSILON` in `common.wgsl`).
const EPSILON: f32 = 1e-6;

/// A square, row-major separated convolution kernel of odd side `2·radius + 1`.
#[derive(Clone, Debug)]
pub struct DogKernel {
    /// Half-width in pixels; the kernel side is `2·radius + 1`.
    pub radius: usize,
    /// Row-major weights, `(2·radius + 1)²` entries.
    pub weights: Vec<f32>,
}

impl DogKernel {
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

/// Unit-volume 2-D isotropic Gaussian at radius² `r2` (pixels²). Mirrors
/// `gaussian_2d` in `brain_passes.wgsl`.
fn gaussian_2d(r2: f32, sigma: f32) -> f32 {
    let sigma_sq = (sigma * sigma).max(EPSILON);
    (-r2 / (2.0 * sigma_sq)).exp() / (2.0 * std::f32::consts::PI * sigma_sq)
}

/// Build the seeded zero-sum Difference-of-Gaussians kernel for the given
/// surround ratio. The center sigma is the fixed `DOG_SIGMA_CENTER` seed; the
/// ratio is clamped to `[DOG_SURROUND_RATIO_MIN, DOG_SURROUND_RATIO_MAX]` exactly
/// as the WGSL does after reading the (eventual) heritable gene. The mean is
/// subtracted so `∑ weights = 0` exactly.
pub fn build_dog_kernel(surround_ratio: f32) -> DogKernel {
    let sigma_center = DOG_SIGMA_CENTER.max(EPSILON);
    let ratio = surround_ratio.clamp(DOG_SURROUND_RATIO_MIN, DOG_SURROUND_RATIO_MAX);
    let sigma_surround = (ratio * sigma_center).max(EPSILON);
    let radius = (DOG_SUPPORT_SIGMAS * sigma_surround).ceil() as usize;
    let side = 2 * radius + 1;
    let entries = side * side;

    let mut weights = vec![0.0_f32; entries];
    let mut raw_sum = 0.0_f32;
    for (k, w) in weights.iter_mut().enumerate() {
        let kx = (k % side) as i32 - radius as i32;
        let ky = (k / side) as i32 - radius as i32;
        let r2 = (kx * kx + ky * ky) as f32;
        let dog = gaussian_2d(r2, sigma_center) - gaussian_2d(r2, sigma_surround);
        *w = dog;
        raw_sum += dog;
    }
    let mean = raw_sum / (entries as f32).max(EPSILON);
    for w in weights.iter_mut() {
        *w -= mean;
    }
    DogKernel { radius, weights }
}

/// The seeded DoG kernel using the default surround-ratio seed.
pub fn seeded_dog_kernel() -> DogKernel {
    build_dog_kernel(DOG_SURROUND_RATIO_SEED)
}

/// Valid-region (zero-padded border) convolution of a `width × height` luminance
/// field with a [`DogKernel`], returning the signed center-surround response.
/// Mirrors the WGSL convolution loop in `coop_visual_cortex` Stage 1.
pub fn convolve(kernel: &DogKernel, luminance: &[f32], width: usize, height: usize) -> Vec<f32> {
    assert_eq!(luminance.len(), width * height, "luminance size mismatch");
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
                    acc += kernel.at(kx, ky) * luminance[pidx];
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
    fn seeded_kernel_is_zero_sum() {
        let kernel = seeded_dog_kernel();
        let sum: f32 = kernel.weights.iter().sum();
        assert!(
            sum.abs() < 1e-5,
            "seeded DoG kernel must sum to zero, got {sum}"
        );
    }

    #[test]
    fn center_weight_is_positive_surround_negative() {
        // ON-center DoG: positive at the origin, negative in the surround ring.
        let kernel = seeded_dog_kernel();
        assert!(kernel.at(0, 0) > 0.0, "center tap must be positive (ON)");
        let r = kernel.radius as i32;
        assert!(
            kernel.at(r, 0) < 0.0,
            "edge-of-support tap must be negative (surround)"
        );
    }

    #[test]
    fn wgsl_dog_constants_match_rust() {
        // Drift guard: the WGSL seed constants in common.wgsl are the production
        // copy; these Rust constants mirror them for the CPU probe. common.wgsl
        // is concatenated verbatim into every pipeline (gpu_kernel.rs), so a
        // matching literal here == the same value compiled into the GPU pass.
        // Mirrors `buffers::tests::luminance_weights_sum_to_one`.
        let common_src = include_str!("shaders/kernel/common.wgsl");
        assert!(
            common_src.contains("const DOG_SIGMA_CENTER: f32 = 1.0;"),
            "common.wgsl DOG_SIGMA_CENTER must match Rust DOG_SIGMA_CENTER ({DOG_SIGMA_CENTER})"
        );
        assert_eq!(DOG_SIGMA_CENTER, 1.0);
        assert!(
            common_src.contains("const DOG_SURROUND_RATIO_SEED: f32 = 1.6;"),
            "common.wgsl DOG_SURROUND_RATIO_SEED must match Rust ({DOG_SURROUND_RATIO_SEED})"
        );
        assert_eq!(DOG_SURROUND_RATIO_SEED, 1.6);
        assert!(
            common_src.contains("const DOG_SUPPORT_SIGMAS: f32 = 3.0;"),
            "common.wgsl DOG_SUPPORT_SIGMAS must match Rust ({DOG_SUPPORT_SIGMAS})"
        );
        assert_eq!(DOG_SUPPORT_SIGMAS, 3.0);
        assert!(
            common_src.contains("const DOG_SURROUND_RATIO_MAX: f32 = 3.0;"),
            "common.wgsl DOG_SURROUND_RATIO_MAX must match Rust ({DOG_SURROUND_RATIO_MAX})"
        );
        assert_eq!(DOG_SURROUND_RATIO_MAX, 3.0);
    }
}
