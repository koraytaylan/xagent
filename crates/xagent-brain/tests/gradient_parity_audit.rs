//! Parity audit: all kernel paths that compute the learning signal are
//! identified and verified to compute identically (pure homeostatic gradients).
//!
//! ## Findings
//!
//! This document records the comprehensive audit of all gradient computation paths
//! in the kernel shaders to verify parity after removing reward-based potential
//! shaping terms.
//!
//! ### Audit Methodology
//!
//! Searched all kernel shader files for:
//! - `raw_gradient` variable declaration or assignment
//! - `energy_delta * ENERGY_WEIGHT` expressions
//! - `integrity_delta * INTEGRITY_WEIGHT` expressions
//!
//! ### Gradient Assembly Paths Identified
//!
//! #### Path 1: Fused Kernel (THE SOLE ASSEMBLY)
//!
//! **Location:** `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl:798-801`
//! **Function:** `coop_habituate_homeo` (Pass 3: Habituate + Homeostasis)
//! **Code:**
//! ```wgsl
//! let raw_gradient = energy_delta * ENERGY_WEIGHT
//!     + integrity_delta * INTEGRITY_WEIGHT
//!     + shaping                    // = 0.0 (removed)
//!     + danger_shaping;            // = 0.0 (removed)
//! ```
//! **Status:** Computes pure homeostatic gradient post-removal
//!
//! #### Verified Non-Paths (Confirmed No Gradient Assembly)
//!
//! Searched all 15 phase shader files (phase_*.wgsl):
//! - phase_agent_grid.wgsl — NO gradient assembly
//! - phase_brain_encode_tiled.wgsl — NO gradient assembly
//! - phase_brain_encoder_credit_tiled.wgsl — NO gradient assembly
//! - phase_brain_features.wgsl — NO gradient assembly
//! - phase_brain_predictor_tiled.wgsl — NO gradient assembly
//! - phase_brain_tail_from_scratch.wgsl — NO gradient assembly
//! - phase_clear.wgsl — NO gradient assembly
//! - phase_collision.wgsl — NO gradient assembly
//! - phase_death.wgsl — NO gradient assembly
//! - phase_food_detect.wgsl — NO gradient assembly (writes P_NEAREST_FOOD_DISTANCE only)
//! - phase_food_grid.wgsl — NO gradient assembly
//! - phase_food_respawn.wgsl — NO gradient assembly
//! - phase_physics.wgsl — NO gradient assembly (snapshots P_PREV_ENERGY/P_PREV_INTEGRITY only)
//! - phase_prepare_dispatch.wgsl — NO gradient assembly
//! - phase_vision.wgsl — NO gradient assembly
//!
//! Searched other kernel shaders (non-phase):
//! - bitonic_sort_subgroup.wgsl — NO gradient assembly
//! - brain_tick.wgsl — NO gradient assembly
//! - global_tick.wgsl — NO gradient assembly
//! - kernel_tick.wgsl — NO gradient assembly (dispatches phases; no gradient)
//! - physics_tick.wgsl — NO gradient assembly
//! - vision_tick.wgsl — NO gradient assembly
//! - common.wgsl — contains const definitions only (P_RAW_GRADIENT_OUT, ENERGY_WEIGHT, etc.)
//!
//! ### Gradient Consumption & Amplification
//!
//! After `raw_gradient` is assembled in `brain_passes.wgsl:796-799`, it is:
//! 1. Blended into three timescale EMA states (fast/medium/slow) via
//!    `brain_state[brain_base + O_HOMEO + {0,1,2}]` at lines 800-805
//! 2. Amplified by urgency and stored to workgroup-shared `s_homeo[1u]`
//!    at line 819 for downstream credit passes
//! 3. Stored un-amplified to workgroup-shared `s_homeo[6u]` at line 820
//!    for CPU readback (P_RAW_GRADIENT_OUT physics slot)
//! 4. Used as the delta term in temporal-difference value updates
//!    (Pass 4 & 5 read it via `s_homeo[1u]`)
//! 5. Stored to motor output `pattern_buffer` at line 1616 for motor credit
//!
//! None of these are *secondary assembly* sites — they consume the
//! single `raw_gradient` value computed in `coop_habituate_homeo`.
//!
//! ### Conclusion
//!
//! **Parity Verified:** Only ONE gradient assembly path exists.
//! The fused `coop_habituate_homeo` function in `brain_passes.wgsl` is
//! the sole computation site for `raw_gradient`. It computes:
//!
//! ```
//! raw_gradient = energy_delta * ENERGY_WEIGHT + integrity_delta * INTEGRITY_WEIGHT
//! ```
//!
//! with both shaping terms (`shaping`, `danger_shaping`) set to 0.0.
//! No split-kernel or secondary assembly path exists. All uses of the
//! raw gradient are consumers, not re-assemblies, so there is no parity
//! divergence to resolve.
//!
//! This is the **canonical** and **sole** gradient assembly path. Any
//! future edit that computes a gradient must route it through this
//! function or this audit must be updated to reflect the new path.

// Embed shader sources at compile time so assertions below are falsifiable:
// if the file is deleted or the gradient expression changes, the test fails.
const BRAIN_PASSES_SRC: &str = include_str!("../src/shaders/kernel/brain_passes.wgsl");
const PHASE_AGENT_GRID_SRC: &str = include_str!("../src/shaders/kernel/phase_agent_grid.wgsl");
const PHASE_BRAIN_ENCODE_TILED_SRC: &str =
    include_str!("../src/shaders/kernel/phase_brain_encode_tiled.wgsl");
const PHASE_BRAIN_ENCODER_CREDIT_TILED_SRC: &str =
    include_str!("../src/shaders/kernel/phase_brain_encoder_credit_tiled.wgsl");
const PHASE_BRAIN_FEATURES_SRC: &str =
    include_str!("../src/shaders/kernel/phase_brain_features.wgsl");
const PHASE_BRAIN_PREDICTOR_TILED_SRC: &str =
    include_str!("../src/shaders/kernel/phase_brain_predictor_tiled.wgsl");
const PHASE_BRAIN_TAIL_FROM_SCRATCH_SRC: &str =
    include_str!("../src/shaders/kernel/phase_brain_tail_from_scratch.wgsl");
const PHASE_CLEAR_SRC: &str = include_str!("../src/shaders/kernel/phase_clear.wgsl");
const PHASE_COLLISION_SRC: &str = include_str!("../src/shaders/kernel/phase_collision.wgsl");
const PHASE_DEATH_SRC: &str = include_str!("../src/shaders/kernel/phase_death.wgsl");
const PHASE_FOOD_DETECT_SRC: &str = include_str!("../src/shaders/kernel/phase_food_detect.wgsl");
const PHASE_FOOD_GRID_SRC: &str = include_str!("../src/shaders/kernel/phase_food_grid.wgsl");
const PHASE_FOOD_RESPAWN_SRC: &str = include_str!("../src/shaders/kernel/phase_food_respawn.wgsl");
const PHASE_PHYSICS_SRC: &str = include_str!("../src/shaders/kernel/phase_physics.wgsl");
const PHASE_PREPARE_DISPATCH_SRC: &str =
    include_str!("../src/shaders/kernel/phase_prepare_dispatch.wgsl");
const PHASE_VISION_SRC: &str = include_str!("../src/shaders/kernel/phase_vision.wgsl");

/// Verifies that the sole gradient assembly path in `brain_passes.wgsl` is
/// `coop_habituate_homeo`, computes pure homeostatic gradients (no shaping terms),
/// and that no phase shader independently assembles `raw_gradient`.
///
/// This test is falsifiable: deleting `coop_habituate_homeo`, removing the
/// `energy_delta * ENERGY_WEIGHT` assembly expression, or reintroducing a
/// non-zero `shaping` or `danger_shaping` binding all cause an assertion failure.
#[test]
fn gradient_parity_audit_documents_sole_assembly() {
    // Assert the gradient assembly function exists in brain_passes.wgsl.
    // Falsified by: deleting coop_habituate_homeo.
    assert!(
        BRAIN_PASSES_SRC.contains("fn coop_habituate_homeo("),
        "brain_passes.wgsl must contain fn coop_habituate_homeo"
    );

    // Assert the pure homeostatic assembly expression is present.
    // Falsified by: removing or renaming the energy delta term.
    assert!(
        BRAIN_PASSES_SRC.contains("energy_delta * ENERGY_WEIGHT"),
        "brain_passes.wgsl must assemble raw_gradient from energy_delta * ENERGY_WEIGHT"
    );

    // Falsified by: removing or renaming the integrity delta term.
    assert!(
        BRAIN_PASSES_SRC.contains("integrity_delta * INTEGRITY_WEIGHT"),
        "brain_passes.wgsl must assemble raw_gradient from integrity_delta * INTEGRITY_WEIGHT"
    );

    // Assert the approach-shaping term is the zero constant, not a live computation.
    // Falsified by: reintroducing a non-zero shaping expression.
    assert!(
        BRAIN_PASSES_SRC.contains("let shaping: f32 = 0.0;"),
        "brain_passes.wgsl shaping term must be the zero constant (approach-PBRS removed)"
    );

    // Assert the avoidance-shaping term is the zero constant, not a live computation.
    // Falsified by: reintroducing a non-zero danger_shaping expression.
    assert!(
        BRAIN_PASSES_SRC.contains("let danger_shaping: f32 = 0.0;"),
        "brain_passes.wgsl danger_shaping term must be the zero constant (avoidance-PBRS removed)"
    );

    // Assert that no phase shader assembles raw_gradient independently.
    // Falsified by: adding a gradient assembly to any phase shader.
    let phase_shaders: &[(&str, &str)] = &[
        ("phase_agent_grid.wgsl", PHASE_AGENT_GRID_SRC),
        (
            "phase_brain_encode_tiled.wgsl",
            PHASE_BRAIN_ENCODE_TILED_SRC,
        ),
        (
            "phase_brain_encoder_credit_tiled.wgsl",
            PHASE_BRAIN_ENCODER_CREDIT_TILED_SRC,
        ),
        ("phase_brain_features.wgsl", PHASE_BRAIN_FEATURES_SRC),
        (
            "phase_brain_predictor_tiled.wgsl",
            PHASE_BRAIN_PREDICTOR_TILED_SRC,
        ),
        (
            "phase_brain_tail_from_scratch.wgsl",
            PHASE_BRAIN_TAIL_FROM_SCRATCH_SRC,
        ),
        ("phase_clear.wgsl", PHASE_CLEAR_SRC),
        ("phase_collision.wgsl", PHASE_COLLISION_SRC),
        ("phase_death.wgsl", PHASE_DEATH_SRC),
        ("phase_food_detect.wgsl", PHASE_FOOD_DETECT_SRC),
        ("phase_food_grid.wgsl", PHASE_FOOD_GRID_SRC),
        ("phase_food_respawn.wgsl", PHASE_FOOD_RESPAWN_SRC),
        ("phase_physics.wgsl", PHASE_PHYSICS_SRC),
        ("phase_prepare_dispatch.wgsl", PHASE_PREPARE_DISPATCH_SRC),
        ("phase_vision.wgsl", PHASE_VISION_SRC),
    ];

    for (name, src) in phase_shaders {
        assert!(
            !src.contains("energy_delta * ENERGY_WEIGHT"),
            "{name} must not assemble raw_gradient (energy_delta * ENERGY_WEIGHT found)"
        );
        assert!(
            !src.contains("integrity_delta * INTEGRITY_WEIGHT"),
            "{name} must not assemble raw_gradient (integrity_delta * INTEGRITY_WEIGHT found)"
        );
    }
}
