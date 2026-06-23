// ── Kernel common definitions ──────────────────────────────────────────
// Shared constants, buffer bindings, and helper functions for all phases.
// Values MUST match buffers.rs and any WGSL `override` constants supplied
// by the Rust host at pipeline creation via
// `PipelineCompilationOptions::constants`.

// ── Vision grid (pipeline-overridable constants) ──────────────────────────
// VISION_W and VISION_H are supplied by the Rust host at pipeline creation
// time via `PipelineCompilationOptions::constants`. Defaults below keep the
// shader standalone-compilable (LSP tooling, WGSL validators) at the 8×6 grid.
// See `gpu_kernel.rs::vision_override_constants()` for the canonical override map.

override VISION_W: u32 = 8u;
override VISION_H: u32 = 6u;

// ── Retina grid (pipeline-overridable constants) ───────────────────────────
// RETINA_WIDTH / RETINA_HEIGHT are the retinotopic luminance grid dimensions
// (config `retina_width` / `retina_height`, plan 0008). They are locked per
// batch and independent of the legacy VISION_W × VISION_H sensory grid.
// Supplied by the Rust host at pipeline creation via `vision_override_constants()`;
// the defaults below keep the shader standalone-compilable at the 32×32 retina.
// RETINA_PIXEL_COUNT is derived as RETINA_WIDTH × RETINA_HEIGHT — it is the
// single canonical source for sizing the cortex workgroup scratch, and the
// visual-cortex stages reconstruct 2-D pixel coordinates from the width so no
// stride is hardcoded.
override RETINA_WIDTH: u32 = 32u;
override RETINA_HEIGHT: u32 = 32u;
override RETINA_PIXEL_COUNT: u32 = RETINA_WIDTH * RETINA_HEIGHT;

// ── DoG center-surround seed constants (plan 0008 center-surround-dog) ───────
// Stage 1 of the visual cortex is a zero-sum Difference-of-Gaussians:
//   DoG(x,y) = G(x,y; σ_center) − G(x,y; σ_surround),  σ_surround = ratio·σ_center
// with unit-volume Gaussians so ∑ DoG = 0 exactly (a local-contrast / edge
// operator, Rodieck 1965; Marr & Hildreth 1980). σ_center is a fixed per-batch
// seed; the surround ratio seed is 1.6 (Marr & Hildreth) and becomes the
// heritable `dog_surround_ratio` gene in plan 0003 — until then the seed is read
// here. These literals are the single canonical source, mirrored by the Rust
// `dog` module constants and exercised by the `dog_kernel_sums_to_zero` probe.
const DOG_SIGMA_CENTER: f32 = 1.0;
const DOG_SURROUND_RATIO_SEED: f32 = 1.6;
// Kernel truncation radius: 3σ of the larger (surround) Gaussian, the standard
// edge-operator support. Half-width in pixels = ceil(3·σ_surround).
const DOG_SUPPORT_SIGMAS: f32 = 3.0;
// Surround-ratio clamp (the plan-0003 `dog_surround_ratio` gene bounds, re-imposed
// in `coop_visual_cortex` after reading the gene). The MIN keeps the kernel an
// edge operator (below ~1.2 the DoG degenerates into a blur); the MAX bounds the
// kernel support so the convolution loop stays finite even when the heritable gene
// grows the surround sigma. Mirror the Rust `dog::DOG_SURROUND_RATIO_{MIN,MAX}`.
const DOG_SURROUND_RATIO_MIN: f32 = 1.2;
const DOG_SURROUND_RATIO_MAX: f32 = 3.0;
// Worst-case kernel half-width (radius), ≥ the seeded kernel. It clamps the
// convolution radius so the per-tap loop is bounded:
//   radius = ceil(DOG_SUPPORT_SIGMAS · DOG_SURROUND_RATIO_MAX · DOG_SIGMA_CENTER)
//          = ceil(3·3·1) = 9  (side 19, ≤ 361 taps).
const DOG_KERNEL_MAX_RADIUS: u32 = 9u;

// ── Gabor simple-cell bank seed constants (plan 0008 gabor-simple-cells) ─────
// Stage 2 of the visual cortex is an orientation-selective Gabor bank — the
// validated quantitative model of a V1 simple-cell receptive field (Jones &
// Palmer 1987); the elongated alternating ON/OFF lobes are Hubel & Wiesel's
// (1962) "aligned row of LGN inputs". Each filter is a DC-balanced 2-D Gabor:
//   x' =  x·cosθ + y·sinθ ,   y' = −x·sinθ + y·cosθ
//   Gabor(x,y) = exp( −(x'² + γ²·y'²)/(2σ²) ) · cos( 2π·x'/λ + ψ )
// computed analytically per tap and mean-subtracted so ∑ Gabor = 0 (the DC
// balance the `gabor_kernels_are_dc_balanced` probe pins). These literals are
// the single canonical source, mirrored by the Rust `gabor` module and read in
// `brain_passes.wgsl::coop_visual_cortex` Stage 2.
//
// Bank dimensions: orientations tiled over [0, π) (HMAX S1, Riesenhuber & Poggio
// 1999), scales one octave apart, and a single quadrature phase pair (even ψ=0,
// odd ψ=π/2). The complex-cell energy step pools over scale and position but
// NEVER over the two phases, so GABOR_PHASES stays 2.
const GABOR_ORIENTATIONS: u32 = 4u;   // 0, 45, 90, 135°
const GABOR_SCALES: u32 = 2u;
const GABOR_PHASES: u32 = 2u;          // quadrature pair {0, π/2}
// Carrier wavelength λ seed in retina pixels (heritable `gabor_wavelength` gene,
// plan 0003; seed read here until then). Envelope σ = GABOR_SIGMA_LAMBDA_RATIO·λ.
const GABOR_WAVELENGTH_SEED: f32 = 5.0;
// Envelope aspect ratio γ seed (long axis / short axis; heritable
// `gabor_aspect_ratio` gene, plan 0003).
const GABOR_ASPECT_RATIO_SEED: f32 = 0.5;
// Whole-bank orientation offset seed in radians, added to the even [0,π) tiling
// (heritable `orientation_offset` gene, plan 0003).
const GABOR_ORIENTATION_OFFSET_SEED: f32 = 0.0;
// σ as a fraction of λ — 0.56 gives the ≈1-octave V1 spatial-frequency bandwidth.
const GABOR_SIGMA_LAMBDA_RATIO: f32 = 0.56;
// Wavelength ratio between successive scale bands (one octave per band).
const GABOR_SCALE_STEP: f32 = 2.0;
// Kernel truncation radius in sigmas (3σ of the envelope).
const GABOR_SUPPORT_SIGMAS: f32 = 3.0;
// Gene clamps (plan-0003 `gabor_wavelength` / `gabor_aspect_ratio` bounds),
// re-imposed in the shader after reading the genes so a mutated value can never
// undersample the carrier or grow the kernel support past the worst-case radius.
const GABOR_WAVELENGTH_MIN: f32 = 2.0;
const GABOR_WAVELENGTH_MAX: f32 = 12.0;
const GABOR_ASPECT_RATIO_MIN: f32 = 0.25;
const GABOR_ASPECT_RATIO_MAX: f32 = 1.0;
// Worst-case kernel half-width (radius), ≥ the largest seeded kernel. Bounds the
// per-tap convolution loop:
//   λ_max band = min(GABOR_WAVELENGTH_MAX · GABOR_SCALE_STEP, GABOR_WAVELENGTH_MAX)
//              = 12 (clamped); σ = 0.56·12 = 6.72; radius = ceil(3·6.72) = 21
// (side 43). The seeded bank's largest band is λ = 10 → σ = 5.6 → radius 17.
const GABOR_KERNEL_MAX_RADIUS: u32 = 21u;

// ── Complex-cell MAX-pool grid (plan 0008 complex-cell-energy-pool) ─────────
// Stage 3 MAX-pools the per-pixel quadrature energy E_{θ,λ}(x,y) over a coarse
// POOL_ROWS × POOL_COLS spatial grid (HMAX C1 position tolerance, Riesenhuber &
// Poggio 1999). Each pool cell covers a contiguous block of the retina and the
// blocks overlap ~50% (the half-block-margin in `pool_bounds`), so a small
// position shift of an oriented bar stays inside the same cell — the
// `complex_cell_position_tolerance` probe (0005) pins this. 4 × 4 = 16 cells
// per (orientation, scale) is the standard HMAX C1 grid. Single canonical source
// mirrored by `POOL_ROWS` / `POOL_COLS` in the Rust `complex` module.
const POOL_ROWS: u32 = 4u;
const POOL_COLS: u32 = 4u;

// ── Visual cortex feature vector size (plan 0008) ──────────────────────────
// VISUAL_FEATURE_COUNT is the length of the complex-cell output the visual
// cortex pass writes back to the head of `s_features`:
// GABOR_ORIENTATIONS × GABOR_SCALES × POOL_ROWS × POOL_COLS (4 × 2 × 4 × 4
// = 128). It is the canonical source for sizing the cortex workgroup scratch and
// is derived from the bank/pool constants (no longer a bare literal) so the bank
// or pool grid can grow without a stale length drifting from the math. Mirrored
// by `VISUAL_FEATURE_COUNT` in the Rust `complex` module and echoed-and-validated
// against `BrainLayout` (the `representation_dimension` precedent).
const VISUAL_FEATURE_COUNT: u32 =
    GABOR_ORIENTATIONS * GABOR_SCALES * POOL_ROWS * POOL_COLS;

// ── Derived vision / sensory constants ─────────────────────────────────────
// Expressions that read `override` inputs must themselves be `override` —
// they are evaluated at pipeline creation time, not shader-module creation.

override VISION_RAYS: u32 = VISION_W * VISION_H;
override VISION_COLOR_COUNT: u32 = VISION_RAYS * 4u;
override VISION_DEPTH_COUNT: u32 = VISION_RAYS;
const MAX_TOUCH_CONTACTS: u32 = 4u;
override SENSORY_STRIDE: u32 = VISION_COLOR_COUNT + VISION_DEPTH_COUNT + 27u;

// ── Brain dimensions ────────────────────────────────────────────────────────

const ENCODED_DIMENSION: u32 = 128u;
const PREDICTOR_DIMENSION: u32 = ENCODED_DIMENSION;

// Visual-cortex encoder-input selector (plan 0008 wire-visual-features-into-encoder).
// Supplied by the Rust host at pipeline creation via `vision_override_constants()`
// from `BrainConfig::visual_cortex_enabled` (1u = on, 0u = off). It is the SAME
// boolean as the runtime `CFG_VISUAL_CORTEX_ENABLED` uniform, but the encoder
// input WIDTH (`FEATURE_COUNT`, which sizes the encoder weight matrix and every
// brain buffer) must be fixed at pipeline-creation time, not read per tick — so
// the width is selected by this pipeline override while the per-pass behavior
// (where `coop_feature_extract` places the non-visual tail, whether the cortex
// runs) is gated on the matching uniform. Both come from the one config field, so
// they agree by construction. Locked per batch.
override VISUAL_CORTEX_FEATURES_ACTIVE: u32 = 0u;

// Danger percept encoder-input selector (plan 0009 danger-percept-sense).
// Supplied by the Rust host at pipeline creation via the world-config bit
// `WC_DANGER_PERCEPT_ENABLED` (1u = on, 0u = off). When on, the non-visual
// feature tail width grows to include danger bearing + distance (25 -> 27);
// the encoder input width `FEATURE_COUNT` thus grows by 2 to match. Locked
// per batch, independent of the visual-cortex flag.
override DANGER_PERCEPT_FEATURES_ACTIVE: u32 = 0u;

// Non-visual feature tail (plan 0008 wire-visual-features-into-encoder): the
// proprioception / interoception / touch features `coop_feature_extract` writes
// after the visual block — velocity magnitude(1) + facing(3) + angular(1) +
// energy ratio(1) + integrity ratio(1) + energy delta(1) + integrity delta(1) +
// touch(16) = 25 base. When danger_percept is enabled (plan 0009), add danger
// bearing(1) + distance(1) = 27 total. Single canonical source, mirrored by
// `NON_VISUAL_FEATURE_COUNT` in `buffers.rs`. The base is constant but the
// tail width changes with the danger-percept flag; the leading visual block
// changes independently with the cortex flag.
override NON_VISUAL_FEATURE_COUNT: u32 = 25u + 2u * DANGER_PERCEPT_FEATURES_ACTIVE;
// Encoder input width. Flag off: the legacy raw-vision slice
// (VISION_COLOR_COUNT + VISION_DEPTH_COUNT) + the non-visual tail — byte-identical
// to the pre-cortex build. Flag on: the compact complex-cell vector
// (VISUAL_FEATURE_COUNT) + the same non-visual tail. Selected by arithmetic on the
// pipeline override (no runtime branch); `BrainLayout::with_retina_flagged`
// mirrors this exact formula so the Rust buffer sizing and the WGSL offsets agree.
// `active` is 0u or 1u, so exactly one term survives.
override FEATURE_COUNT: u32 =
    (1u - VISUAL_CORTEX_FEATURES_ACTIVE) * (VISION_COLOR_COUNT + VISION_DEPTH_COUNT)
    + VISUAL_CORTEX_FEATURES_ACTIVE * VISUAL_FEATURE_COUNT
    + NON_VISUAL_FEATURE_COUNT;
const MEMORY_CAP: u32 = 128u;
const RECALL_K: u32 = 16u;
const ERROR_HISTORY_LEN: u32 = 128u;

// ── Brain state offsets (derived from FEATURE_COUNT) ────────────────────────
// These must be `override` because they transitively reference FEATURE_COUNT.

const O_ENC_WEIGHTS: u32 = 0u;
override O_ENC_BIASES: u32 = FEATURE_COUNT * ENCODED_DIMENSION;
override O_PREDICTOR_WEIGHTS: u32 = O_ENC_BIASES + ENCODED_DIMENSION;
override O_PREDICTOR_CONTEXT_WEIGHT: u32 = O_PREDICTOR_WEIGHTS + PREDICTOR_DIMENSION * ENCODED_DIMENSION;
override O_PREDICTION_ERROR_RING: u32 = O_PREDICTOR_CONTEXT_WEIGHT + 1u;
override O_PREDICTION_ERROR_CURSOR: u32 = O_PREDICTION_ERROR_RING + ERROR_HISTORY_LEN;
override O_PREDICTION_ERROR_COUNT: u32 = O_PREDICTION_ERROR_CURSOR + 1u;
override O_HAB_EMA: u32 = O_PREDICTION_ERROR_COUNT + 1u;
override O_HAB_ATTEN: u32 = O_HAB_EMA + ENCODED_DIMENSION;
override O_PREV_ENCODED: u32 = O_HAB_ATTEN + ENCODED_DIMENSION;
override O_HOMEO: u32 = O_PREV_ENCODED + ENCODED_DIMENSION;
override O_ACTION_FORWARD_WEIGHTS: u32 = O_HOMEO + 6u;
override O_ACTION_TURN_WEIGHTS: u32 = O_ACTION_FORWARD_WEIGHTS + ENCODED_DIMENSION;
override O_ACT_BIASES: u32 = O_ACTION_TURN_WEIGHTS + ENCODED_DIMENSION;
override O_EXPLORATION_RATE: u32 = O_ACT_BIASES + 2u;
const POS_RING_LEN: u32 = 16u;
override O_POS_RING_X: u32 = O_EXPLORATION_RATE + 1u;
override O_POS_RING_Z: u32 = O_POS_RING_X + POS_RING_LEN;
override O_POS_RING_CURSOR: u32 = O_POS_RING_Z + POS_RING_LEN;
override O_POS_RING_LEN: u32 = O_POS_RING_CURSOR + 1u;
override O_ACCUM_FWD: u32 = O_POS_RING_LEN + 1u;
override O_FATIGUE_FACTOR: u32 = O_ACCUM_FWD + 1u;
override O_PREV_PREDICTION: u32 = O_FATIGUE_FACTOR + 1u;
override O_TICK_COUNT: u32 = O_PREV_PREDICTION + PREDICTOR_DIMENSION;
override O_HAB_SENSITIVITY: u32 = O_TICK_COUNT + 1u;
override O_HAB_MAX_CURIOSITY: u32 = O_HAB_SENSITIVITY + 1u;
override O_FATIGUE_FLOOR: u32 = O_HAB_MAX_CURIOSITY + 1u;
override O_MOVEMENT_SPEED: u32 = O_FATIGUE_FLOOR + 1u;

// ── Visual-genome tail (plan 0008 visual-genome-config) ─────────────────────
// Four heritable Gabor/DoG bank genes, contiguous right after O_MOVEMENT_SPEED.
// `coop_visual_cortex` reads them from brain_state and re-imposes the gene clamps
// + the DoG zero-sum / Gabor DC-balance invariants. Mirrors the O_GABOR_* /
// O_DOG_SURROUND_RATIO / O_ORIENTATION_OFFSET constants in `buffers.rs`.
override O_GABOR_WAVELENGTH: u32 = O_MOVEMENT_SPEED + 1u;
override O_GABOR_ASPECT_RATIO: u32 = O_GABOR_WAVELENGTH + 1u;
override O_DOG_SURROUND_RATIO: u32 = O_GABOR_ASPECT_RATIO + 1u;
override O_ORIENTATION_OFFSET: u32 = O_DOG_SURROUND_RATIO + 1u;

// ── TD(λ) critic state ──────────────────────────────────────────────────────
// Value head (learned, inherited) plus eligibility traces (episodic,
// zeroed on death). Trace biases pack three scalars:
// [critic_bias, forward_bias, turn_bias].

override O_VALUE_WEIGHTS: u32 = O_ORIENTATION_OFFSET + 1u;
override O_VALUE_BIAS: u32 = O_VALUE_WEIGHTS + ENCODED_DIMENSION;
override O_PREV_VALUE: u32 = O_VALUE_BIAS + 1u;
override O_TRACE_CRITIC: u32 = O_PREV_VALUE + 1u;
override O_TRACE_FWD: u32 = O_TRACE_CRITIC + ENCODED_DIMENSION;
override O_TRACE_TURN: u32 = O_TRACE_FWD + ENCODED_DIMENSION;
override O_TRACE_BIASES: u32 = O_TRACE_TURN + ENCODED_DIMENSION;

// ── Per-agent buffer strides ────────────────────────────────────────────────

override BRAIN_STRIDE: u32 = O_TRACE_BIASES + 3u;
const PATTERN_STRIDE: u32 = O_LAST_STORED_IDX + 1u;
override FEATURES_STRIDE: u32 = FEATURE_COUNT;
const DECISION_PREDICTION: u32 = 0u;
const DECISION_CREDIT: u32 = ENCODED_DIMENSION;
const DECISION_MOTOR: u32 = ENCODED_DIMENSION + ENCODED_DIMENSION;
const DECISION_STRIDE: u32 = DECISION_MOTOR + 4u;
const HOMEO_OUT_STRIDE: u32 = 6u;
const RECALL_IDX_STRIDE: u32 = 17u;    // 16 indices + 1 count

// ── Per-agent brain scratch (binding 13) offsets ────────────────────────────
// Storage-backed intermediates so multi-workgroup brain phases (plan 0006) can
// cooperate per agent across dispatch boundaries. Layout mirrors the fused
// var<workgroup> arrays 1:1. override (transitively references FEATURES_STRIDE).
override SCRATCH_FEATURES: u32 = 0u;
override SCRATCH_ENCODED: u32 = SCRATCH_FEATURES + FEATURES_STRIDE;
override SCRATCH_HABITUATED: u32 = SCRATCH_ENCODED + ENCODED_DIMENSION;
override SCRATCH_HOMEO: u32 = SCRATCH_HABITUATED + ENCODED_DIMENSION;
override SCRATCH_RECALL: u32 = SCRATCH_HOMEO + 6u;
override SCRATCH_RECALL_SIMILARITY: u32 = SCRATCH_RECALL + RECALL_IDX_STRIDE;
override SCRATCH_PREDICTION: u32 = SCRATCH_RECALL_SIMILARITY + RECALL_K;
override SCRATCH_CREDIT: u32 = SCRATCH_PREDICTION + PREDICTOR_DIMENSION;
override SCRATCH_SCALARS: u32 = SCRATCH_CREDIT + ENCODED_DIMENSION;
override BRAIN_SCRATCH_STRIDE: u32 = SCRATCH_SCALARS + 4u;

// ── Pattern memory offsets ──────────────────────────────────────────────────
// O_PAT_STATES uses SoA (Structure-of-Arrays) layout: [dim][pattern]
// Index as: pattern_base + d * MEMORY_CAP + pattern_idx
// This gives coalesced reads when 128 threads each read one pattern.
// Other regions (norms, reinf, motor, meta, active) remain AoS.

const O_PAT_STATES: u32 = 0u;
const O_PAT_NORMS: u32 = MEMORY_CAP * ENCODED_DIMENSION;
const O_PAT_REINF: u32 = O_PAT_NORMS + MEMORY_CAP;
const O_PAT_MOTOR: u32 = O_PAT_REINF + MEMORY_CAP;
const O_PAT_META: u32 = O_PAT_MOTOR + MEMORY_CAP * 3u;
const O_PAT_ACTIVE: u32 = O_PAT_META + MEMORY_CAP * 3u;
const O_ACTIVE_COUNT: u32 = O_PAT_ACTIVE + MEMORY_CAP;
const O_MIN_REINF_IDX: u32 = O_ACTIVE_COUNT + 1u;
const O_LAST_STORED_IDX: u32 = O_MIN_REINF_IDX + 1u;

// ── Config buffer offsets ───────────────────────────────────────────────────

const CFG_LEARNING_RATE: u32 = 4u;
const CFG_DECAY_RATE: u32 = 5u;
const CFG_DISTRESS_EXP: u32 = 6u;
const CFG_METABOLIC_RATE: u32 = 7u;
const CFG_INTEGRITY_SCALE: u32 = 8u;
// Visual-cortex gate flag (plan 0008): 1.0 runs the Hubel-Wiesel cortex pass,
// 0.0 is a no-op passthrough. Mirrors `CFG_VISUAL_CORTEX_ENABLED` in buffers.rs.
const CFG_VISUAL_CORTEX_ENABLED: u32 = 9u;
const CFG_DANGER_PERCEPT_ENABLED: u32 = 10u;

// ── Agent physics buffer layout (P_*) ───────────────────────────────────────

const PHYS_STRIDE: u32 = 47u;
const P_POS_X: u32 = 0u;
const P_POS_Y: u32 = 1u;
const P_POS_Z: u32 = 2u;
const P_VEL_X: u32 = 3u;
const P_VEL_Y: u32 = 4u;
const P_VEL_Z: u32 = 5u;
const P_FACING_X: u32 = 6u;
const P_FACING_Y: u32 = 7u;
const P_FACING_Z: u32 = 8u;
const P_YAW: u32 = 9u;
const P_ANGULAR_VEL: u32 = 10u;
const P_ENERGY: u32 = 11u;
const P_MAX_ENERGY: u32 = 12u;
const P_INTEGRITY: u32 = 13u;
const P_MAX_INTEGRITY: u32 = 14u;
const P_PREV_ENERGY: u32 = 15u;
const P_PREV_INTEGRITY: u32 = 16u;
const P_ALIVE: u32 = 17u;
const P_FOOD_COUNT: u32 = 18u;
const P_TICKS_ALIVE: u32 = 19u;
const P_DIED_FLAG: u32 = 20u;
const P_MEMORY_CAP: u32 = 21u;
const P_PROCESSING_SLOTS: u32 = 22u;
const P_DEATH_COUNT: u32 = 23u;
const P_PREDICTION_ERROR: u32 = 24u;
const P_EXPLORATION_RATE_OUT: u32 = 25u;
const P_FATIGUE_FACTOR_OUT: u32 = 26u;
const P_MOTOR_FWD_OUT: u32 = 27u;
const P_MOTOR_TURN_OUT: u32 = 28u;
const P_GRADIENT_OUT: u32 = 29u;
const P_URGENCY_OUT: u32 = 30u;
const P_LAST_DEATH_TICK: u32 = 31u;
const P_NEAREST_FOOD_DISTANCE: u32 = 32u;
const P_PREV_POTENTIAL: u32 = 33u;
const P_NEAREST_FOOD_BEARING: u32 = 34u;
const P_IN_DANGER_BIOME: u32 = 35u;
const P_DISTANCE_TRAVELED: u32 = 36u;
const P_ENERGY_SPENT: u32 = 37u;
const P_DANGER_PATH_LENGTH: u32 = 38u;
/// Distance to the nearest in-range danger cell. Sentinel value is DANGER_SENSE_RADIUS
/// when no danger is in range.
const P_NEAREST_DANGER_DISTANCE: u32 = 39u;
/// Signed bearing (radians) from current facing direction to the nearest in-range
/// danger biome cell. Sentinel value is 0.0 when no danger is in range.
const P_NEAREST_DANGER_BEARING: u32 = 40u;
/// Reserved: previous avoidance potential (no longer written after reward-shaping
/// removal). Retained for layout parity and test compatibility; reset on respawn.
const P_PREV_DANGER_POTENTIAL: u32 = 41u;
/// Cumulative count of ticks where danger was in sense range.
const P_AVOIDANCE_SENSE_RANGE_TICKS: u32 = 42u;
/// Cumulative count of ticks where danger was in sense range AND motor turn opposed bearing.
const P_AVOIDANCE_TURNS_OPPOSING: u32 = 43u;
/// Pre-amplification homeostatic learning signal raw_gradient
/// (pure homeostatic: energy_delta*ENERGY_WEIGHT + integrity_delta*INTEGRITY_WEIGHT),
/// written by coop_habituate_homeo for CPU readback. Per-agent live state,
/// never serialized.
const P_RAW_GRADIENT_OUT: u32 = 44u;
/// Cumulative count of ticks where food was in sense range.
const P_APPROACH_SENSE_RANGE_TICKS: u32 = 45u;
/// Cumulative count of ticks where food was in sense range AND motor turn rotated toward bearing.
const P_APPROACH_TURNS_TOWARD: u32 = 46u;

// ── Food buffer layout ─────────────────────────────────────────────────────

const FOOD_STATE_STRIDE: u32 = 4u;
const FOOD_POSITION_X: u32 = 0u;
const FOOD_POSITION_Y: u32 = 1u;
const FOOD_POSITION_Z: u32 = 2u;
const FOOD_RESPAWN_TIMER: u32 = 3u;

// ── Math constants ─────────────────────────────────────────────────────────

const PI: f32 = 3.14159265;
const TWO_PI: f32 = 6.28318530;
// Small positive floor for divisions (CONTRIBUTING numeric safety:
// `max(denominator, EPSILON)` before every division). Used by the visual-cortex
// stages and any other pass guarding a divide-by-zero.
const EPSILON: f32 = 1e-6;
/// Square root of 2; used as the motor-magnitude clamp (maximum L∞ norm of
/// motor commands before drag). Bounds the movement-cost accumulator by clamping
/// the combined forward+strafe magnitude to √2 ≈ 1.414, so a diagonally-maxed
/// agent burns energy at the same rate as a max-forward agent.
const SQRT_2: f32 = 1.41421356237;

// ── Physics constants ───────────────────────────────────────────────────────

const GRAVITY: f32 = 20.0;
/// Baseline locomotion speed; movement energy and the path-length-hazard
/// reference step are both normalized by it. One name, one source of truth.
const DEFAULT_MOVE_SPEED: f32 = 20.0;
const TURN_SPEED: f32 = 3.0;
const AGENT_HALF_HEIGHT: f32 = 1.0;
const METABOLIC_BASE_COST: f32 = 0.0001;
const METABOLIC_MEMORY_COST: f32 = 0.00003;
const METABOLIC_PROCESSING_COST: f32 = 0.0001;
// FOOD_CONSUME_RADIUS_SQ removed — read from wconfig via wc_f32(WC_FOOD_RADIUS)
const JUMP_VELOCITY: f32 = 8.0;
const COLLISION_MIN_DIST: f32 = 2.0;
const COLLISION_MIN_DIST_SQ: f32 = 4.0;
const COLLISION_FIXED_SCALE: f32 = 1024.0;

// ── Grid constants ──────────────────────────────────────────────────────────
// NOTE: GRID_WIDTH, GRID_OFFSET, TERRAIN_* are world-size-dependent.
// They are read from wconfig at runtime via wc_f32() / wc_u32().

const CELL_SIZE: f32 = 8.0;
const FOOD_GRID_MAX_PER_CELL: u32 = 16u;
const FOOD_GRID_CELL_STRIDE: u32 = 17u;   // 1 + 16
const AGENT_GRID_MAX_PER_CELL: u32 = 32u;
const AGENT_GRID_CELL_STRIDE: u32 = 33u;  // 1 + 32

// ── Terrain constants ───────────────────────────────────────────────────────
// Static terrain grid properties (129 vertices per side).

const TERRAIN_VPS: u32 = 129u;
const BIOME_GRID_RES: u32 = 256u;
/// Biome grid is 256×256; the last valid index is 255. Naming it removes the
/// silent coupling between the clamp literal and the grid resolution.
const BIOME_GRID_MAX_INDEX: u32 = 255u;

// ── Food respawn constants ──────────────────────────────────────────────────

const FOOD_RESPAWN_TIME: f32 = 10.0;
const FOOD_HEIGHT_OFFSET: f32 = 0.35;
const FOOD_RESPAWN_ATTEMPTS: u32 = 64u;

// ── Vision constants ────────────────────────────────────────────────────────

const VISION_FOV_HALF: f32 = PI / 4.0;   // PI/4 = 45 degrees half-FOV
const VISION_MAX_DIST: f32 = 30.0;
// World-units radius of the nearest-food sense scan in agent_food_detect.
// Set to VISION_MAX_DIST to match the visual field range.
const FOOD_SENSE_RADIUS: f32 = 30.0;
// World-units radius within which danger (biome cells) are sensed for avoidance
// bearing calculation. Symmetric to FOOD_SENSE_RADIUS.
const DANGER_SENSE_RADIUS: f32 = 30.0;
const VISION_STEP_SIZE: f32 = 1.2;
const VISION_NUM_STEPS: u32 = 25u;
const FOOD_RAY_RADIUS_SQ: f32 = 1.0;
const AGENT_RAY_RADIUS_SQ: f32 = 2.25;

// ── Touch constants ─────────────────────────────────────────────────────────

const TOUCH_FOOD: u32 = 1u;
const TOUCH_TERRAIN_EDGE: u32 = 2u;
const TOUCH_HAZARD: u32 = 3u;
const TOUCH_AGENT: u32 = 4u;
const TOUCH_FOOD_RANGE: f32 = 3.0;
const TOUCH_AGENT_RANGE: f32 = 5.0;
const TOUCH_EDGE_RANGE: f32 = 3.0;

// Hazard contacts have no meaningful planar direction (the hazard is
// the terrain underfoot), so they carry a fixed mid-scale intensity
// instead of a closeness value. Matches the CPU reference in
// agent/senses.rs.
const TOUCH_HAZARD_INTENSITY: f32 = 0.5;

// ── Biome type values ───────────────────────────────────────────────────────

const BIOME_FOOD_RICH: u32 = 0u;
const BIOME_BARREN: u32 = 1u;
const BIOME_DANGER: u32 = 2u;

// ── World config indices (WC_*) ─────────────────────────────────────────────

const WC_WORLD_SIZE: u32 = 0u;
const WC_DT: u32 = 1u;
const WC_ENERGY_DEPLETION: u32 = 2u;
const WC_MOVEMENT_COST: u32 = 3u;
const WC_HAZARD_DAMAGE: u32 = 4u;
const WC_INTEGRITY_REGEN: u32 = 5u;
const WC_FOOD_ENERGY: u32 = 6u;
const WC_FOOD_RADIUS: u32 = 7u;
const WC_TERRAIN_VPS: u32 = 8u;
const WC_TERRAIN_INV_STEP: u32 = 9u;
const WC_TERRAIN_HALF: u32 = 10u;
const WC_BIOME_INV_CELL: u32 = 11u;
const WC_FOOD_COUNT: u32 = 12u;
const WC_AGENT_COUNT: u32 = 13u;
const WC_TICK: u32 = 14u;
const WC_RNG_SEED: u32 = 15u;
const WC_WORLD_HALF_BOUND: u32 = 16u;
const WC_BIOME_GRID_RES: u32 = 17u;
const WC_GRID_WIDTH: u32 = 18u;
const WC_GRID_OFFSET: u32 = 19u;
const WC_TICKS_TO_RUN: u32 = 20u;
const WC_PHASE_MASK: u32 = 21u;  // bit0=physics, bit1=vision, bit2=brain
const WC_VISION_STRIDE: u32 = 22u;
const WC_BRAIN_TICK_STRIDE: u32 = 23u;
const WC_SPEED_COST_EXPONENT: u32 = 24u;
const WC_DANGER_PERCEPT_ENABLED: u32 = 25u;
// Danger-percept ablation mask (measurement-only): 1u = blind the brain by packing
// the "no danger in range" sentinel into the danger encoder features while the
// geometry-gated avoidance counters keep counting; 0u = pack the true detected
// values. Only meaningful when WC_DANGER_PERCEPT_ENABLED == 1u.
const WC_DANGER_PERCEPT_BLINDED: u32 = 26u;

// ── Habituation / homeostasis constants ─────────────────────────────────────

const HAB_EMA_ALPHA: f32 = 0.02;
const ATTEN_FLOOR: f32 = 0.1;
const MAX_HOMEOSTATIC_DELTA: f32 = 0.3;
const ENERGY_WEIGHT: f32 = 0.6;
const INTEGRITY_WEIGHT: f32 = 0.4;
const GRADIENT_FAST_BLEND: f32 = 0.6;
const GRADIENT_MEDIUM_BLEND: f32 = 0.04;
const GRADIENT_SLOW_BLEND: f32 = 0.004;
const DISTRESS_SCALE: f32 = 10.0;
const MAX_DISTRESS: f32 = 10.0;
const GRADIENT_WEIGHT_FAST: f32 = 0.5;
const GRADIENT_WEIGHT_MEDIUM: f32 = 0.35;
const GRADIENT_WEIGHT_SLOW: f32 = 0.15;

// ── Predict-and-act constants ───────────────────────────────────────────────

const ACTION_WEIGHT_LEARNING_RATE: f32 = 0.10;
const MAX_WEIGHT_NORM: f32 = 2.0;
const ENCODER_CREDIT_SCALE: f32 = 0.1;
const CREDIT_EPSILON: f32 = 1e-6;
const KLINOTAXIS_SENSITIVITY: f32 = 500.0;
const MEMORY_BLEND_STRENGTH: f32 = 0.4;

// ── TD(λ) credit constants ──────────────────────────────────────────────────

// Per-brain-tick discount. Horizon 1/(1−γ) ≈ 33 brain ticks ≈ 11 s of
// real time at the default strides (brain tick every 10 physics ticks
// at 30 Hz) — several food approaches long. A vision-edge approach
// itself is ~45 physics ticks ≈ 4.5 brain ticks at default speed; the
// horizon is intentionally longer so the critic bridges sparse
// encounters. At brain_tick_stride = 1 the same constant gives a 1.1 s
// horizon — if the default stride changes, recalibrate γ to keep the
// real-time horizon (γ ≈ 1 − stride/330).
const TD_DISCOUNT: f32 = 0.97;
// Eligibility trace decay. Combined per-tick trace retention is
// TD_DISCOUNT × TD_LAMBDA ≈ 0.87; the critic's bootstrapping propagates
// credit beyond the raw trace span across repeated experiences.
const TD_LAMBDA: f32 = 0.9;
// Critic learns 10× slower than the actor: the value estimate must be
// stabler than the policy it evaluates.
const CRITIC_LEARNING_RATE: f32 = 0.01;
// Per-dimension trace updates scale inversely with the feature dimension:
// the aggregate step (a sum of ENCODED_DIMENSION products of trace ×
// feature, each O(1)) would otherwise grow with dimensionality and push
// the bootstrapped critic past the linear-TD stability limit.
const TD_VECTOR_SCALE: f32 = 1.0 / f32(ENCODED_DIMENSION);
// Actor (forward/turn) weight-step scale, separate from the critic's
// TD_VECTOR_SCALE. The 1/ENCODED_DIMENSION factor is a critic-bootstrapping
// stability bound; applied to the actor it throttled the policy step to
// 0.10/128 ≈ 8e-4. The actor only needs to stay inside the MAX_WEIGHT_NORM L2
// ball (enforced every tick), so it can latch onto a sign-correct δ at a usable
// rate. 1/16 lifts the step ~8× while staying well within that bound.
const ACTOR_VECTOR_SCALE: f32 = 1.0 / 16.0;
// Bound on the TD error. No single transition is allowed to teach more
// than this; protects against respawn/clamp artifacts (mirrors the intent
// of MAX_HOMEOSTATIC_DELTA on the reward side).
const MAX_TD_ERROR: f32 = 1.0;

// Terminal TD error applied through the dying episode's eligibility
// traces at the moment of death, before they are cleared for the next
// life. Death must be the single worst lesson the learner can receive,
// but never stronger than the per-transition bound that protects
// against artifacts.
const TERMINAL_DEATH_TD_ERROR: f32 = -MAX_TD_ERROR;

// ═══════════════════════════════════════════════════════════════════════════
// Buffer bindings — 14 storage + 2 uniform, single bind group
// (binding 13 is now brain_scratch; the numbering of the remaining
// bindings is stable so the bind-group layout in gpu_kernel.rs stays aligned)
// ═══════════════════════════════════════════════════════════════════════════

@group(0) @binding(0)  var<storage, read_write> physics_state:        array<f32>;
@group(0) @binding(1)  var<storage, read_write> decision_buffer:      array<f32>;
@group(0) @binding(2)  var<storage, read>       heightmap:         array<f32>;
@group(0) @binding(3)  var<storage, read>       biome_grid:        array<u32>;
@group(0) @binding(4)  var<uniform>             wconfig:           array<vec4<f32>, 7>;
@group(0) @binding(5)  var<storage, read_write> food_state:        array<f32>;
@group(0) @binding(6)  var<storage, read_write> food_flags:        array<atomic<u32>>;
@group(0) @binding(7)  var<storage, read_write> food_grid:         array<atomic<u32>>;
@group(0) @binding(8)  var<storage, read_write> agent_grid:        array<atomic<u32>>;
@group(0) @binding(9)  var<storage, read_write> collision_scratch: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> sensory_buffer:       array<f32>;
@group(0) @binding(11) var<storage, read_write> brain_state:       array<f32>;
@group(0) @binding(12) var<storage, read_write> pattern_buffer:       array<f32>;
@group(0) @binding(13) var<storage, read_write> brain_scratch:       array<f32>;
@group(0) @binding(14) var<uniform>             brain_config:      array<vec4<f32>, 3>;
@group(0) @binding(15) var<storage, read_write> dispatch_args:     array<u32, 6>;

// ═══════════════════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════════════════

// ── World config accessors ──────────────────────────────────────────────────

fn wc_f32(idx: u32) -> f32 {
    return wconfig[idx / 4u][idx % 4u];
}

fn wc_u32(idx: u32) -> u32 {
    return u32(wconfig[idx / 4u][idx % 4u]);
}

fn bc_f32(idx: u32) -> f32 {
    return brain_config[idx / 4u][idx % 4u];
}

// ── RNG (PCG hash) ──────────────────────────────────────────────────────────

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_to_float(h: u32) -> f32 {
    return f32(h) / 4294967295.0;
}

// ── Grid coordinate helpers ─────────────────────────────────────────────────

fn cell_coord(v: f32) -> i32 {
    return i32(floor(v / CELL_SIZE));
}

fn cell_index(cx: u32, cz: u32) -> u32 {
    return cx * wc_u32(WC_GRID_WIDTH) + cz;
}

// ── Terrain height sampling (bilinear) ──────────────────────────────────────

fn sample_height(x: f32, z: f32) -> f32 {
    let terrain_half = wc_f32(WC_TERRAIN_HALF);
    let terrain_inv_step = wc_f32(WC_TERRAIN_INV_STEP);
    let terrain_vps = wc_u32(WC_TERRAIN_VPS);
    let terrain_max_idx = terrain_vps - 2u;
    let terrain_max_coord = f32(terrain_vps - 1u);

    let gx = clamp((x + terrain_half) * terrain_inv_step, 0.0, terrain_max_coord);
    let gz = clamp((z + terrain_half) * terrain_inv_step, 0.0, terrain_max_coord);
    let ix = min(u32(gx), terrain_max_idx);
    let iz = min(u32(gz), terrain_max_idx);
    let fx = gx - f32(ix);
    let fz = gz - f32(iz);
    let h00 = heightmap[iz * terrain_vps + ix];
    let h10 = heightmap[iz * terrain_vps + ix + 1u];
    let h01 = heightmap[(iz + 1u) * terrain_vps + ix];
    let h11 = heightmap[(iz + 1u) * terrain_vps + ix + 1u];
    return mix(mix(h00, h10, fx), mix(h01, h11, fx), fz);
}

// ── Biome grid lookup ───────────────────────────────────────────────────────

fn sample_biome(x: f32, z: f32) -> u32 {
    let biome_half = wc_f32(WC_TERRAIN_HALF);
    let biome_inv = wc_f32(WC_BIOME_INV_CELL);
    let col = min(u32((x + biome_half) * biome_inv), BIOME_GRID_MAX_INDEX);
    let row = min(u32((z + biome_half) * biome_inv), BIOME_GRID_MAX_INDEX);
    return biome_grid[row * BIOME_GRID_RES + col];
}

// ── Activation function ─────────────────────────────────────────────────────

fn fast_tanh(x: f32) -> f32 {
    if (abs(x) > 4.5) { return sign(x); }
    let x2 = x * x;
    return x * (27.0 + x2) / (27.0 + 9.0 * x2);
}

// ── NaN guard ───────────────────────────────────────────────────────────────

fn is_finite(v: f32) -> bool {
    return v == v && abs(v) < 3.4e38;
}

// ── Retina luminance (plan 0008, Stage 0) ───────────────────────────────────
// Linear (Rec. 709) luminance of a raycast hit color, the single-channel field
// L(x,y) the visual cortex operates on. Convolution kernels are linear
// operators, so they act on linear-light luminance, not gamma-encoded RGB. Hit
// colors are authored in linear space, so no inverse-gamma is applied.
//
// The weights MUST sum to 1.0; the Rust test `luminance_weights_sum_to_one`
// recomputes the same three literals and guards against a typo drifting them.
//
// No brightness-normalization pass precedes Stage 1: DC (mean luminance) is
// rejected downstream by the zero-sum DoG and the DC-balanced Gabor bank, so
// only local contrast — not absolute brightness — survives into the cortex.
//
// Rec. 709 luminance weights (single canonical source; mirrored in the Rust
// test): 0.2126 (R) + 0.7152 (G) + 0.0722 (B) = 1.0.
fn retina_luminance(color: vec3<f32>) -> f32 {
    return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
}
