//! GPU buffer layout constants, AgentBrainState, and packing utilities.
//!
//! All persistent brain state is stored in two flat f32 arrays on GPU:
//! - brain_state_buf: encoder weights, predictor, habituation, homeostasis,
//!   action, staleness, TD value head + eligibility traces
//! - pattern_buf: pattern memory (states, norms, reinforcement, motor, meta, active)
//!
//! Integer values (cursors, counts, ticks) are stored as f32 and cast
//! via bitcast in WGSL. Safe for exact integers up to 2^24 = 16,777,216.

use std::sync::atomic::AtomicBool;

use xagent_shared::{BrainConfig, SensoryFrame, TouchContact};

use crate::complex::VISUAL_FEATURE_COUNT;

/// First-hit guard for the `representation_dimension` vs `ENCODED_DIMENSION`
/// mismatch warning in `build_config_for`. `build_config_for` runs on every
/// `GpuKernel::reset_agents*` and on each evolution-spawned config, so without
/// this guard a single bad saved config would spam the log across a long run.
static REPR_DIM_MISMATCH_WARNED: AtomicBool = AtomicBool::new(false);

// ── Dimensions ───────────────────────────────────────────────────────

pub const ENCODED_DIMENSION: usize = 128;
pub const PREDICTOR_DIMENSION: usize = ENCODED_DIMENSION;
/// Feature count for the reference 8×6 vision layout that anchors the
/// static `O_*` offset constants below. The runtime default layout is
/// `BrainLayout::default()` (which follows `BrainConfig::default()`); use
/// `BrainLayout::feature_count` for any live layout. The static constants
/// remain valid as tail *deltas* (`O_X - O_PREDICTOR_CONTEXT_WEIGHT`) for
/// every layout, because the tail is vision-independent.
const FEATURE_COUNT: usize = 8 * 6 * 4 + 8 * 6 + NON_VISUAL_FEATURE_COUNT;
/// Non-visual feature tail width (plan 0008 wire-visual-features-into-encoder):
/// the proprioception / interoception / touch features `coop_feature_extract`
/// writes after the visual block — velocity magnitude(1) + facing(3) +
/// angular(1) + energy ratio(1) + integrity ratio(1) + energy delta(1) +
/// integrity delta(1) + touch(16) = 25. It is the same in both encoder layouts;
/// only the leading visual block changes width with the cortex flag. Single
/// canonical source mirrored by `NON_VISUAL_FEATURE_COUNT` in `common.wgsl`. Note
/// this differs from `NON_VISUAL_COUNT` (27, the *sensory upload* width) by the
/// two energy/integrity delta slots the brain re-reads same-cycle from physics
/// rather than from the batch-lagged sensory buffer.
pub const NON_VISUAL_FEATURE_COUNT: usize = 25;
pub const MEMORY_CAP: usize = 128;
pub const RECALL_K: usize = 16;
pub const INITIAL_FORWARD_BIAS: f32 = 0.3;
pub const ERROR_HISTORY_LEN: usize = 128;
pub const MAX_TOUCH_CONTACTS: usize = 4;
pub const TOUCH_FEATURES: usize = 4; // dir_x, dir_z, intensity, tag/4

// ── Sensory input layout (CPU → GPU upload) ───────────────────────────

/// Non-visual sensory channels (vision-independent).
/// velocity(3) + facing(3) + angular_vel(1) + energy(1) + integrity(1) + e_delta(1) + i_delta(1) + touch(16)
pub const NON_VISUAL_COUNT: usize = 3 + 3 + 1 + 1 + 1 + 1 + 1 + MAX_TOUCH_CONTACTS * TOUCH_FEATURES;

// ── Brain state buffer offsets (reference 8×6 layout) ─────────────────
// Absolute values are valid only for an 8×6 vision grid; for live layouts
// use `BrainLayout` to compute dynamic offsets.
// The *tail* offsets (from `O_PREDICTOR_CONTEXT_WEIGHT` onward) are vision-independent:
// `O_FOO - O_PREDICTOR_CONTEXT_WEIGHT` is the same regardless of vision size.

pub const O_ENC_WEIGHTS: usize = 0;
pub const O_ENC_BIASES: usize = O_ENC_WEIGHTS + FEATURE_COUNT * ENCODED_DIMENSION;
pub const O_PREDICTOR_WEIGHTS: usize = O_ENC_BIASES + ENCODED_DIMENSION;
pub const O_PREDICTOR_CONTEXT_WEIGHT: usize =
    O_PREDICTOR_WEIGHTS + PREDICTOR_DIMENSION * ENCODED_DIMENSION;
pub const O_PREDICTION_ERROR_RING: usize = O_PREDICTOR_CONTEXT_WEIGHT + 1;
pub const O_PREDICTION_ERROR_CURSOR: usize = O_PREDICTION_ERROR_RING + ERROR_HISTORY_LEN;
pub const O_PREDICTION_ERROR_COUNT: usize = O_PREDICTION_ERROR_CURSOR + 1;
pub const O_HAB_EMA: usize = O_PREDICTION_ERROR_COUNT + 1;
pub const O_HAB_ATTEN: usize = O_HAB_EMA + ENCODED_DIMENSION;
pub const O_PREV_ENCODED: usize = O_HAB_ATTEN + ENCODED_DIMENSION;

// homeo: [grad_fast, grad_med, grad_slow, urgency, prev_energy, prev_integrity]
pub const O_HOMEO: usize = O_PREV_ENCODED + ENCODED_DIMENSION;
pub const O_ACTION_FORWARD_WEIGHTS: usize = O_HOMEO + 6;
pub const O_ACTION_TURN_WEIGHTS: usize = O_ACTION_FORWARD_WEIGHTS + ENCODED_DIMENSION;

// act_biases: [fwd_bias, turn_bias]
pub const O_ACT_BIASES: usize = O_ACTION_TURN_WEIGHTS + ENCODED_DIMENSION;
pub const O_EXPLORATION_RATE: usize = O_ACT_BIASES + 2;
pub const POS_RING_LEN: usize = 16;
pub const O_POS_RING_X: usize = O_EXPLORATION_RATE + 1;
pub const O_POS_RING_Z: usize = O_POS_RING_X + POS_RING_LEN;
pub const O_POS_RING_CURSOR: usize = O_POS_RING_Z + POS_RING_LEN;
pub const O_POS_RING_LEN: usize = O_POS_RING_CURSOR + 1;
pub const O_ACCUM_FWD: usize = O_POS_RING_LEN + 1;
pub const O_FATIGUE_FACTOR: usize = O_ACCUM_FWD + 1;
pub const O_PREV_PREDICTION: usize = O_FATIGUE_FACTOR + 1;
pub const O_TICK_COUNT: usize = O_PREV_PREDICTION + PREDICTOR_DIMENSION;
pub const O_HAB_SENSITIVITY: usize = O_TICK_COUNT + 1;
pub const O_HAB_MAX_CURIOSITY: usize = O_HAB_SENSITIVITY + 1;
pub const O_FATIGUE_FLOOR: usize = O_HAB_MAX_CURIOSITY + 1;
pub const O_MOVEMENT_SPEED: usize = O_FATIGUE_FLOOR + 1;

// ── Visual-genome tail (plan 0008 visual-genome-config) ───────────────
// Four heritable Gabor/DoG bank genes, contiguous right after
// `O_MOVEMENT_SPEED`, written per agent by `write_agent_heritable_config`
// and read by the `coop_visual_cortex` shader pass. Mirrored by the
// `O_GABOR_*` / `O_DOG_SURROUND_RATIO` / `O_ORIENTATION_OFFSET` overrides in
// `common.wgsl`. Order matches `write_agent_heritable_config`'s `values` array.
pub const O_GABOR_WAVELENGTH: usize = O_MOVEMENT_SPEED + 1;
pub const O_GABOR_ASPECT_RATIO: usize = O_GABOR_WAVELENGTH + 1;
pub const O_DOG_SURROUND_RATIO: usize = O_GABOR_ASPECT_RATIO + 1;
pub const O_ORIENTATION_OFFSET: usize = O_DOG_SURROUND_RATIO + 1;

// ── TD(λ) critic state ────────────────────────────────────────────────
// Value head (learned, inherited) plus eligibility traces (episodic,
// zeroed on death). Trace biases pack three scalars:
// [critic_bias, forward_bias, turn_bias].

pub const O_VALUE_WEIGHTS: usize = O_ORIENTATION_OFFSET + 1;
pub const O_VALUE_BIAS: usize = O_VALUE_WEIGHTS + ENCODED_DIMENSION;
pub const O_PREV_VALUE: usize = O_VALUE_BIAS + 1;
pub const O_TRACE_CRITIC: usize = O_PREV_VALUE + 1;
pub const O_TRACE_FWD: usize = O_TRACE_CRITIC + ENCODED_DIMENSION;
pub const O_TRACE_TURN: usize = O_TRACE_FWD + ENCODED_DIMENSION;
pub const O_TRACE_BIASES: usize = O_TRACE_TURN + ENCODED_DIMENSION;
pub const BRAIN_STRIDE: usize = O_TRACE_BIASES + 3;

/// Number of elements in `brain_state` from `O_PREDICTOR_CONTEXT_WEIGHT` (inclusive)
/// to `BRAIN_STRIDE` (exclusive). This tail is layout-independent: it
/// doesn't change with feature_count / vision dimensions.
pub const FIXED_TAIL_SIZE: usize = BRAIN_STRIDE - O_PREDICTOR_CONTEXT_WEIGHT;

// ── Pattern memory buffer offsets (per agent) ─────────────────────────

pub const O_PAT_STATES: usize = 0;
pub const O_PAT_NORMS: usize = O_PAT_STATES + MEMORY_CAP * ENCODED_DIMENSION;
pub const O_PAT_REINF: usize = O_PAT_NORMS + MEMORY_CAP;
pub const O_PAT_MOTOR: usize = O_PAT_REINF + MEMORY_CAP;
// motor: [forward, turn, outcome_valence] × cap
pub const O_PAT_META: usize = O_PAT_MOTOR + MEMORY_CAP * 3;
// meta: [created_at, last_accessed, activation_count] × cap
pub const O_PAT_ACTIVE: usize = O_PAT_META + MEMORY_CAP * 3;
pub const O_ACTIVE_COUNT: usize = O_PAT_ACTIVE + MEMORY_CAP;
pub const O_MIN_REINF_IDX: usize = O_ACTIVE_COUNT + 1;
pub const O_LAST_STORED_IDX: usize = O_MIN_REINF_IDX + 1;
pub const PATTERN_STRIDE: usize = O_LAST_STORED_IDX + 1;

// ── Agent physics buffer layout (per agent, GPU-resident) ─────────────

pub const P_POS_X: usize = 0;
pub const P_POS_Y: usize = 1;
pub const P_POS_Z: usize = 2;
pub const P_VEL_X: usize = 3;
pub const P_VEL_Y: usize = 4;
pub const P_VEL_Z: usize = 5;
pub const P_FACING_X: usize = 6;
pub const P_FACING_Y: usize = 7;
pub const P_FACING_Z: usize = 8;
pub const P_YAW: usize = 9;
pub const P_ANGULAR_VEL: usize = 10;
pub const P_ENERGY: usize = 11;
pub const P_MAX_ENERGY: usize = 12;
pub const P_INTEGRITY: usize = 13;
pub const P_MAX_INTEGRITY: usize = 14;
pub const P_PREV_ENERGY: usize = 15;
pub const P_PREV_INTEGRITY: usize = 16;
pub const P_ALIVE: usize = 17;
pub const P_FOOD_COUNT: usize = 18;
pub const P_TICKS_ALIVE: usize = 19;
pub const P_DIED_FLAG: usize = 20;
pub const P_MEMORY_CAP: usize = 21;
pub const P_PROCESSING_SLOTS: usize = 22;
pub const P_DEATH_COUNT: usize = 23;
pub const P_PREDICTION_ERROR: usize = 24;
pub const P_EXPLORATION_RATE_OUT: usize = 25;
pub const P_FATIGUE_FACTOR_OUT: usize = 26;
pub const P_MOTOR_FWD_OUT: usize = 27;
pub const P_MOTOR_TURN_OUT: usize = 28;
pub const P_GRADIENT_OUT: usize = 29;
pub const P_URGENCY_OUT: usize = 30;
/// Tick at which this agent most recently died, stored as f32 via numeric
/// conversion (`f32(tick)` in WGSL, `as u64` on readback).  Exact for integer
/// ticks up to 2^24 — matches the precision guarantees of `P_TICKS_ALIVE`.
/// Written by the physics phase when energy/integrity reaches zero, preserved
/// across the death/respawn reset so CPU readback can attribute the death to
/// an exact tick instead of the end-of-batch upper bound.
pub const P_LAST_DEATH_TICK: usize = 31;
/// Planar distance (world units) to the nearest food within `FOOD_SENSE_RADIUS`,
/// written by the food-detect pass each cycle; the `FOOD_SENSE_RADIUS` sentinel
/// when no food is in range. Per-agent live state, regenerated each run —
/// never serialized or inherited.
pub const P_NEAREST_FOOD_DISTANCE: usize = 32;
/// Reserved: previous brain tick's approach potential (no longer written
/// after reward-shaping removal). Retained for layout parity and test
/// compatibility; reset on respawn. Per-agent live state, never serialized.
pub const P_PREV_POTENTIAL: usize = 33;
/// Signed bearing (radians) from current facing direction to the nearest
/// in-range food, computed in agent_food_detect. Sentinel value is 0.0 when no
/// food is in range. Zero-mean: positive = food to the right, negative = food
/// to the left. Written by the food-detect pass each cycle; per-agent live state.
pub const P_NEAREST_FOOD_BEARING: usize = 34;
/// Whether the agent is currently in a danger biome. Written by agent_physics
/// when sampling biome type. 1.0 for danger, 0.0 for safe.
pub const P_IN_DANGER_BIOME: usize = 35;
/// Cumulative distance traveled over the generation. Σ planar |Δpos| since spawn.
pub const P_DISTANCE_TRAVELED: usize = 36;
/// Cumulative energy spent over the generation. Σ movement + metabolic drain since spawn.
pub const P_ENERGY_SPENT: usize = 37;
/// Cumulative distance traveled while in danger. Σ planar |Δpos| while in_danger.
pub const P_DANGER_PATH_LENGTH: usize = 38;
/// Distance to the nearest in-range danger cell. Sentinel value is DANGER_SENSE_RADIUS
/// when no danger is in range.
pub const P_NEAREST_DANGER_DISTANCE: usize = 39;
/// Signed bearing (radians) from current facing direction to the nearest in-range
/// danger biome cell. Computed like P_NEAREST_FOOD_BEARING. Sentinel value is 0.0
/// when no danger is in range.
pub const P_NEAREST_DANGER_BEARING: usize = 40;
/// Reserved: previous avoidance potential (no longer written after reward-shaping
/// removal). Retained for layout parity and test compatibility; reset on respawn.
/// Per-agent live state, never serialized.
pub const P_PREV_DANGER_POTENTIAL: usize = 41;
/// Cumulative count of ticks where danger was in sense range. Used to compute
/// the fraction of ticks where avoidance decision was made.
pub const P_AVOIDANCE_SENSE_RANGE_TICKS: usize = 42;
/// Cumulative count of ticks where danger was in sense range AND the motor turn
/// opposed the danger bearing (deliberate turn-away). Used to compute avoidance intent.
pub const P_AVOIDANCE_TURNS_OPPOSING: usize = 43;
/// Pre-amplification homeostatic learning signal raw_gradient
/// (pure homeostatic: energy_delta*ENERGY_WEIGHT + integrity_delta*INTEGRITY_WEIGHT),
/// written by coop_habituate_homeo for CPU readback. Per-agent live state,
/// never serialized.
pub const P_RAW_GRADIENT_OUT: usize = 44;
/// Cumulative count of ticks where food was in sense range. Used to compute
/// the fraction of ticks where approach decision was made.
pub const P_APPROACH_SENSE_RANGE_TICKS: usize = 45;
/// Cumulative count of ticks where food was in sense range AND the motor turn
/// rotated toward the nearest food bearing (deliberate turn-toward). Used to compute approach intent.
pub const P_APPROACH_TURNS_TOWARD: usize = 46;
pub const PHYS_STRIDE: usize = 47;
/// Brain runs once every N physics ticks. Must match the cycle logic in dispatch_batch.
pub const BRAIN_TICK_STRIDE: u32 = 4;

// ── Sensory constants ───────────────────────────────────────────────────────

/// World-units radius within which danger biome cells are sensed for avoidance
/// bearing calculation. Symmetric to FOOD_SENSE_RADIUS (food).
pub const DANGER_SENSE_RADIUS: f32 = 30.0;

// ── Runtime layout for configurable vision dimensions ─────────────────

/// Runtime-computed buffer layout that depends on vision_width × vision_height.
/// All strides and offsets cascade from the pixel count.
#[derive(Clone, Debug)]
pub struct BrainLayout {
    pub vision_width: u32,
    pub vision_height: u32,
    pub vision_color_count: usize,
    pub vision_depth_count: usize,
    /// Retinotopic luminance grid width the visual cortex operates on. Locked
    /// per batch (config `retina_width`), independent of the legacy sensory
    /// vision grid. See `xagent_shared::BrainConfig::retina_width`.
    pub retina_width: usize,
    /// Retinotopic luminance grid height; see `retina_width`.
    pub retina_height: usize,
    /// `retina_width * retina_height` — number of pixels in the cortex retina.
    /// Validated with `checked_mul` so an oversized retina is a configuration
    /// error, not a silent overflow.
    pub retina_pixel_count: usize,
    /// Whether this layout sizes the encoder for the Hubel-Wiesel visual cortex
    /// (plan 0008). When `true`, `feature_count` is the compact complex-cell
    /// vector + non-visual tail (`VISUAL_FEATURE_COUNT + NON_VISUAL_FEATURE_COUNT`);
    /// when `false` it is the legacy raw-vision slice + non-visual tail. Must
    /// match `BrainConfig::visual_cortex_enabled` and the WGSL FEATURE_COUNT
    /// pipeline override (`vision_override_constants`). Locked per batch.
    pub visual_cortex_enabled: bool,
    /// Whether this layout sizes the sensory/feature buffers to include the
    /// dedicated danger percept (plan 0009). When `true`, non-visual feature
    /// count grows by 2 (danger bearing + distance). Must match
    /// `BrainConfig::danger_percept_enabled` and the WGSL
    /// `DANGER_PERCEPT_FEATURES_ACTIVE` pipeline override (`vision_override_constants`).
    /// Locked per batch.
    pub danger_percept_enabled: bool,
    pub feature_count: usize,
    pub sensory_stride: usize,
    pub brain_stride: usize,
    pub brain_scratch_stride: usize,
}

impl BrainLayout {
    /// Build a layout for the given vision grid, deriving the retina grid from
    /// `BrainConfig::default()` (curriculum default 32×32). Use
    /// [`BrainLayout::with_retina`] to set an explicit retina resolution.
    pub fn new(vision_width: u32, vision_height: u32) -> Self {
        let defaults = BrainConfig::default();
        Self::with_retina(
            vision_width,
            vision_height,
            defaults.retina_width,
            defaults.retina_height,
        )
    }

    /// Build a layout from a full [`BrainConfig`], threading the vision grid,
    /// the configured retina resolution, AND the visual-cortex flag — so the
    /// encoder width matches `BrainConfig::visual_cortex_enabled`. This is the
    /// constructor `GpuKernel::new` uses; the flag must agree with the WGSL
    /// FEATURE_COUNT pipeline override (`vision_override_constants`).
    pub fn from_config(config: &BrainConfig) -> Self {
        Self::with_retina_flagged(
            config.vision_width,
            config.vision_height,
            config.retina_width,
            config.retina_height,
            config.visual_cortex_enabled,
            config.danger_percept_enabled,
        )
    }

    /// Build a layout with an explicit retina grid and the legacy raw-vision
    /// encoder width (visual cortex off). For the cortex-on width use
    /// [`BrainLayout::from_config`] (or [`BrainLayout::with_retina_flagged`]).
    pub fn with_retina(
        vision_width: u32,
        vision_height: u32,
        retina_width: usize,
        retina_height: usize,
    ) -> Self {
        Self::with_retina_flagged(
            vision_width,
            vision_height,
            retina_width,
            retina_height,
            false,
            false,
        )
    }

    /// Build a layout with an explicit retina grid and visual-cortex flag.
    /// `retina_pixel_count` is validated with `checked_mul`; an oversized retina
    /// is a configuration error, not a silent overflow.
    ///
    /// `visual_cortex_enabled` selects the encoder input width (the leading
    /// visual block of `feature_count`), mirroring the WGSL FEATURE_COUNT override
    /// in `common.wgsl` exactly:
    ///   off — legacy raw-vision slice (`color_count + depth_count`)
    ///   on  — compact complex-cell vector (`VISUAL_FEATURE_COUNT`)
    /// plus the unchanged `NON_VISUAL_FEATURE_COUNT` tail in both cases.
    pub fn with_retina_flagged(
        vision_width: u32,
        vision_height: u32,
        retina_width: usize,
        retina_height: usize,
        visual_cortex_enabled: bool,
        danger_percept_enabled: bool,
    ) -> Self {
        let retina_pixel_count = retina_width
            .checked_mul(retina_height)
            .expect("retina pixel count overflow");
        let pixel_count = (vision_width as usize)
            .checked_mul(vision_height as usize)
            .expect("vision dimensions overflow pixel count");
        let color_count = pixel_count
            .checked_mul(4)
            .expect("vision dimensions overflow color count");
        let depth_count = pixel_count;
        // Encoder visual-block width: the complex-cell vector when the cortex is
        // on, else the legacy raw-vision slice. Mirrors the WGSL FEATURE_COUNT
        // override (single canonical formula across Rust ↔ WGSL).
        let visual_block = if visual_cortex_enabled {
            VISUAL_FEATURE_COUNT
        } else {
            color_count
                .checked_add(depth_count)
                .expect("vision dimensions overflow visual block")
        };
        // Danger features (bearing + distance) are read same-cycle from
        // `physics_state` by `coop_feature_extract` (mirroring the interoception
        // reads), so they do NOT travel through the sensory buffer.  Only the
        // encoder input width (`feature_count`) grows; `sensory_stride` (the
        // per-agent sensory buffer slot count) is unchanged regardless of the flag.
        let danger_feature_width = if danger_percept_enabled { 2 } else { 0 };
        let feature_count = visual_block
            .checked_add(NON_VISUAL_FEATURE_COUNT)
            .and_then(|v| v.checked_add(danger_feature_width))
            .expect("vision dimensions overflow feature count");
        let sensory_stride = color_count
            .checked_add(depth_count)
            .and_then(|v| v.checked_add(NON_VISUAL_COUNT))
            .expect("vision dimensions overflow sensory stride");
        // brain_stride = feature_count * ENCODED_DIMENSION + ENCODED_DIMENSION + ENCODED_DIMENSION*ENCODED_DIMENSION + FIXED_TAIL_SIZE
        // (FIXED_TAIL_SIZE = fixed fields starting at O_PREDICTOR_CONTEXT_WEIGHT
        // through the TD critic state, incl. position ring and traces)
        let brain_stride = feature_count
            .checked_mul(ENCODED_DIMENSION)
            .and_then(|v| v.checked_add(ENCODED_DIMENSION))
            .and_then(|v| v.checked_add(PREDICTOR_DIMENSION * ENCODED_DIMENSION))
            .and_then(|v| v.checked_add(FIXED_TAIL_SIZE))
            .expect("vision dimensions overflow brain stride");
        let brain_scratch_stride = feature_count
            .checked_add(ENCODED_DIMENSION) // encoded
            .and_then(|v| v.checked_add(ENCODED_DIMENSION)) // habituated
            .and_then(|v| v.checked_add(HOMEO_OUT_STRIDE)) // homeo (6)
            .and_then(|v| v.checked_add(RECALL_IDX_STRIDE)) // recall (17)
            .and_then(|v| v.checked_add(RECALL_K)) // recall similarity (16)
            .and_then(|v| v.checked_add(PREDICTOR_DIMENSION)) // prediction
            .and_then(|v| v.checked_add(ENCODED_DIMENSION)) // credit
            .and_then(|v| v.checked_add(4)) // scalars
            .expect("vision dimensions overflow brain scratch stride");
        Self {
            vision_width,
            vision_height,
            vision_color_count: color_count,
            vision_depth_count: depth_count,
            retina_width,
            retina_height,
            retina_pixel_count,
            visual_cortex_enabled,
            danger_percept_enabled,
            feature_count,
            sensory_stride,
            brain_stride,
            brain_scratch_stride,
        }
    }
}

impl Default for BrainLayout {
    /// Follows `BrainConfig::default()` so the two defaults can never drift.
    fn default() -> Self {
        Self::from_config(&BrainConfig::default())
    }
}

// ── Food buffer layout (per food item) ───────────────────────────────

pub const FOOD_POSITION_X: usize = 0;
pub const FOOD_POSITION_Y: usize = 1;
pub const FOOD_POSITION_Z: usize = 2;
pub const FOOD_RESPAWN_TIMER: usize = 3;
pub const FOOD_STATE_STRIDE: usize = 4;

// ── Grid constants ───────────────────────────────────────────────────

pub const GRID_CELL_SIZE: f32 = 8.0;
pub const FOOD_GRID_MAX_PER_CELL: usize = 16;
pub const FOOD_GRID_CELL_STRIDE: usize = 1 + FOOD_GRID_MAX_PER_CELL;
pub const AGENT_GRID_MAX_PER_CELL: usize = 32;
pub const AGENT_GRID_CELL_STRIDE: usize = 1 + AGENT_GRID_MAX_PER_CELL;

/// Compute grid width (cells per axis) for a given world size.
pub fn grid_width(world_size: f32) -> usize {
    (world_size / GRID_CELL_SIZE).ceil() as usize + 4
}

// ── World config uniform layout ──────────────────────────────────────

pub const WC_WORLD_SIZE: usize = 0;
pub const WC_DT: usize = 1;
pub const WC_ENERGY_DEPLETION: usize = 2;
pub const WC_MOVEMENT_COST: usize = 3;
pub const WC_HAZARD_DAMAGE: usize = 4;
pub const WC_INTEGRITY_REGEN: usize = 5;
pub const WC_FOOD_ENERGY: usize = 6;
pub const WC_FOOD_RADIUS: usize = 7;
pub const WC_TERRAIN_VPS: usize = 8;
pub const WC_TERRAIN_INV_STEP: usize = 9;
pub const WC_TERRAIN_HALF: usize = 10;
pub const WC_BIOME_INV_CELL: usize = 11;
pub const WC_FOOD_COUNT: usize = 12;
pub const WC_AGENT_COUNT: usize = 13;
pub const WC_TICK: usize = 14;
pub const WC_RNG_SEED: usize = 15;
pub const WC_WORLD_HALF_BOUND: usize = 16;
pub const WC_BIOME_GRID_RES: usize = 17;
pub const WC_GRID_WIDTH: usize = 18;
pub const WC_GRID_OFFSET: usize = 19;
pub const WC_TICKS_TO_RUN: usize = 20;
pub const WC_PHASE_MASK: usize = 21; // bit0=physics, bit1=vision, bit2=brain
pub const WC_VISION_STRIDE: usize = 22;
pub const WC_BRAIN_TICK_STRIDE: usize = 23;
pub const WC_SPEED_COST_EXPONENT: usize = 24;
/// Danger percept gate flag (plan 0009). `1.0` = pack nearest-danger bearing and
/// distance into non-visual features, `0.0` = no-op (feature count unchanged).
/// Uses a previously-unused padding slot, so `WORLD_CONFIG_SIZE` is unchanged.
/// Mirrored by `WC_DANGER_PERCEPT_ENABLED` in `common.wgsl`.
pub const WC_DANGER_PERCEPT_ENABLED: usize = 25;
/// Danger-percept ablation mask (measurement-only). `1.0` = force the
/// danger distance/bearing packed into the encoder feature vector to the
/// "no danger in range" sentinel (distance 1.0, bearing 0.0), blinding the brain
/// while the geometry-gated avoidance-intent counters keep counting. `0.0` =
/// pack the true detected values. Only meaningful when `WC_DANGER_PERCEPT_ENABLED`
/// is `1.0`. Uses a previously-unused padding slot, so `WORLD_CONFIG_SIZE` is
/// unchanged. Mirrored by `WC_DANGER_PERCEPT_BLINDED` in `common.wgsl`.
pub const WC_DANGER_PERCEPT_BLINDED: usize = 26;
pub const WORLD_CONFIG_SIZE: usize = 28; // padded to 7 × vec4

// ── Transient buffer sizes (per agent) ────────────────────────────────

pub const FEATURES_STRIDE: usize = FEATURE_COUNT;
pub const ENCODED_STRIDE: usize = ENCODED_DIMENSION;
pub const HABITUATED_STRIDE: usize = ENCODED_DIMENSION;
pub const HOMEO_OUT_STRIDE: usize = 6; // grad, raw_grad, urgency, grad_fast, grad_med, grad_slow
pub const SIMILARITIES_STRIDE: usize = MEMORY_CAP;
pub const RECALL_IDX_STRIDE: usize = RECALL_K + 1; // 16 indices + 1 count
pub const DECISION_PREDICTION: usize = 0;
pub const DECISION_CREDIT: usize = ENCODED_DIMENSION;
pub const DECISION_MOTOR: usize = ENCODED_DIMENSION + ENCODED_DIMENSION;
pub const DECISION_STRIDE: usize = DECISION_MOTOR + 4;

// ── Brain scratch buffer offsets (8×6 layout anchors) ──────────────────────────
// Mirrors SCRATCH_* WGSL overrides for the default 8×6 layout; used for
// testing offset consistency. Runtime layouts compute brain_scratch_stride
// dynamically via BrainLayout::new.
pub const SCRATCH_FEATURES: usize = 0;
pub const SCRATCH_ENCODED: usize = SCRATCH_FEATURES + FEATURES_STRIDE;
pub const SCRATCH_HABITUATED: usize = SCRATCH_ENCODED + ENCODED_DIMENSION;
pub const SCRATCH_HOMEO: usize = SCRATCH_HABITUATED + ENCODED_DIMENSION;
pub const SCRATCH_RECALL: usize = SCRATCH_HOMEO + HOMEO_OUT_STRIDE;
pub const SCRATCH_RECALL_SIMILARITY: usize = SCRATCH_RECALL + RECALL_IDX_STRIDE;
pub const SCRATCH_PREDICTION: usize = SCRATCH_RECALL_SIMILARITY + RECALL_K;
pub const SCRATCH_CREDIT: usize = SCRATCH_PREDICTION + PREDICTOR_DIMENSION;
pub const SCRATCH_SCALARS: usize = SCRATCH_CREDIT + ENCODED_DIMENSION;
pub const BRAIN_SCRATCH_STRIDE: usize = SCRATCH_SCALARS + 4;

// ── Config buffer layout ──────────────────────────────────────────────

pub const CFG_REPR_DIM: usize = 0;
pub const CFG_FEATURE_COUNT: usize = 1;
pub const CFG_MEMORY_CAP: usize = 2;
pub const CFG_RECALL_K: usize = 3;
pub const CFG_LEARNING_RATE: usize = 4;
pub const CFG_DECAY_RATE: usize = 5;
pub const CFG_DISTRESS_EXP: usize = 6;
pub const CFG_METABOLIC_RATE: usize = 7;
pub const CFG_INTEGRITY_SCALE: usize = 8;
/// Visual-cortex gate flag (plan 0008). `1.0` = run the Hubel-Wiesel cortex
/// pass, `0.0` = no-op passthrough + legacy raw-vision encoder input. Uses a
/// previously-unused padding slot, so `CONFIG_SIZE` is unchanged. Mirrored by
/// `CFG_VISUAL_CORTEX_ENABLED` in `common.wgsl`.
pub const CFG_VISUAL_CORTEX_ENABLED: usize = 9;
/// Danger-percept sensory feature gate flag. `1.0` = pack danger distance and bearing
/// into the encoder input (two additional non-visual features via `DANGER_PERCEPT_FEATURES_ACTIVE`),
/// `0.0` = no-op (danger features omitted from sensory input). The avoidance-PBRS shaping
/// term that formerly read this flag has been removed; this flag now controls only sensory
/// feature availability, not any reward shaping. Uses a previously-unused padding slot, so
/// `CONFIG_SIZE` is unchanged. Mirrored by `CFG_DANGER_PERCEPT_ENABLED` in `common.wgsl`.
pub const CFG_DANGER_PERCEPT_ENABLED: usize = 10;
pub const CONFIG_SIZE: usize = 12; // padded to 12 for uniform vec4 alignment (3 × vec4)

// ── AgentBrainState (CPU-side snapshot for evolution) ──────────────────

/// CPU-side snapshot of one agent's full brain state.
/// Used for cross-generation inheritance and mutation.
#[derive(Clone, Debug)]
pub struct AgentBrainState {
    pub brain_state: Vec<f32>, // BRAIN_STRIDE f32s
    pub patterns: Vec<f32>,    // PATTERN_STRIDE f32s
}

impl AgentBrainState {
    pub fn new_blank() -> Self {
        Self::new_for(BRAIN_STRIDE)
    }

    pub fn new_for(brain_stride: usize) -> Self {
        Self {
            brain_state: vec![0.0; brain_stride],
            patterns: vec![0.0; PATTERN_STRIDE],
        }
    }
}

// ── Sensory frame packing ─────────────────────────────────────────────

/// Pack a SensoryFrame into a flat f32 slice for GPU upload.
/// Selects up to 4 highest-intensity touch contacts, zero-pads the rest.
pub fn pack_sensory_frame(frame: &SensoryFrame, layout: &BrainLayout, out: &mut [f32]) {
    assert!(
        out.len() >= layout.sensory_stride,
        "output buffer too short: {} < {}",
        out.len(),
        layout.sensory_stride,
    );

    let mut offset = 0;

    // Vision color (RGBA per pixel)
    let color_len = frame.vision.color.len().min(layout.vision_color_count);
    out[offset..offset + color_len].copy_from_slice(&frame.vision.color[..color_len]);
    for i in color_len..layout.vision_color_count {
        out[offset + i] = 0.0;
    }
    offset += layout.vision_color_count;

    // Vision depth (one per pixel)
    let depth_len = frame.vision.depth.len().min(layout.vision_depth_count);
    out[offset..offset + depth_len].copy_from_slice(&frame.vision.depth[..depth_len]);
    for i in depth_len..layout.vision_depth_count {
        out[offset + i] = 0.0;
    }
    offset += layout.vision_depth_count;

    // Velocity (3)
    out[offset] = frame.velocity.x;
    out[offset + 1] = frame.velocity.y;
    out[offset + 2] = frame.velocity.z;
    offset += 3;

    // Facing (3)
    out[offset] = frame.facing.x;
    out[offset + 1] = frame.facing.y;
    out[offset + 2] = frame.facing.z;
    offset += 3;

    // Angular velocity (1)
    out[offset] = frame.angular_velocity;
    offset += 1;

    // Energy signal (1)
    out[offset] = frame.energy_signal;
    offset += 1;

    // Integrity signal (1)
    out[offset] = frame.integrity_signal;
    offset += 1;

    // Energy delta (1)
    out[offset] = frame.energy_delta;
    offset += 1;

    // Integrity delta (1)
    out[offset] = frame.integrity_delta;
    offset += 1;

    // Touch contacts: pick top 4 by intensity, zero-pad
    let mut sorted_contacts: Vec<&TouchContact> = frame.touch_contacts.iter().collect();
    sorted_contacts.sort_by(|a, b| {
        b.intensity
            .partial_cmp(&a.intensity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for slot in 0..MAX_TOUCH_CONTACTS {
        if slot < sorted_contacts.len() {
            let c = sorted_contacts[slot];
            out[offset] = c.direction.x;
            out[offset + 1] = c.direction.z;
            out[offset + 2] = c.intensity;
            out[offset + 3] = c.surface_tag as f32 / 4.0;
        } else {
            out[offset] = 0.0;
            out[offset + 1] = 0.0;
            out[offset + 2] = 0.0;
            out[offset + 3] = 0.0;
        }
        offset += TOUCH_FEATURES;
    }
}

/// Fill a caller-owned world-config buffer in place.
///
/// Writes the first WORLD_CONFIG_SIZE slots into `out` without allocating, so the
/// per-batch GPU dispatch path can reuse one scratch array instead of heap-
/// allocating a fresh `Vec` every batch. `out` is fully zeroed first so the
/// result is byte-identical to [`build_world_config`]'s `vec![0.0; N]` init —
/// the padding slots the caller sets afterward stay `0.0`.
#[allow(clippy::too_many_arguments)]
pub fn fill_world_config(
    out: &mut [f32; WORLD_CONFIG_SIZE],
    config: &xagent_shared::WorldConfig,
    food_count: usize,
    agent_count: usize,
    tick: u64,
    ticks_to_run: u32,
    vision_stride: u32,
    brain_tick_stride: u32,
    speed_cost_exponent: f32,
    danger_percept_enabled: bool,
    danger_percept_blinded: bool,
) {
    let gw = grid_width(config.world_size);
    let go = gw / 2;
    let terrain_step = config.world_size / 128.0;
    out.fill(0.0);
    out[WC_WORLD_SIZE] = config.world_size;
    out[WC_DT] = 1.0 / config.tick_rate;
    out[WC_ENERGY_DEPLETION] = config.energy_depletion_rate;
    out[WC_MOVEMENT_COST] = config.movement_energy_cost;
    out[WC_HAZARD_DAMAGE] = config.hazard_damage_rate;
    out[WC_INTEGRITY_REGEN] = config.integrity_regen_rate;
    out[WC_FOOD_ENERGY] = config.food_energy_value;
    out[WC_FOOD_RADIUS] = 2.0;
    out[WC_TERRAIN_VPS] = 129.0;
    out[WC_TERRAIN_INV_STEP] = 1.0 / terrain_step;
    out[WC_TERRAIN_HALF] = config.world_size / 2.0;
    out[WC_BIOME_INV_CELL] = 256.0 / config.world_size;
    out[WC_FOOD_COUNT] = food_count as f32;
    out[WC_AGENT_COUNT] = agent_count as f32;
    out[WC_TICK] = tick as f32;
    out[WC_RNG_SEED] = config.seed as f32;
    out[WC_WORLD_HALF_BOUND] = config.world_size / 2.0 - 1.0;
    out[WC_BIOME_GRID_RES] = 256.0;
    out[WC_GRID_WIDTH] = gw as f32;
    out[WC_GRID_OFFSET] = go as f32;
    out[WC_TICKS_TO_RUN] = ticks_to_run as f32;
    out[WC_VISION_STRIDE] = vision_stride as f32;
    out[WC_BRAIN_TICK_STRIDE] = brain_tick_stride as f32;
    out[WC_SPEED_COST_EXPONENT] = speed_cost_exponent;
    out[WC_DANGER_PERCEPT_ENABLED] = if danger_percept_enabled { 1.0 } else { 0.0 };
    out[WC_DANGER_PERCEPT_BLINDED] = if danger_percept_blinded { 1.0 } else { 0.0 };
}

/// Build the world config uniform data.
///
/// Thin wrapper over [`fill_world_config`] for callers that want an owned
/// `Vec` (tests, one-off uploads). The hot per-batch dispatch path uses
/// `fill_world_config` against a reused scratch array instead.
pub fn build_world_config(
    config: &xagent_shared::WorldConfig,
    food_count: usize,
    agent_count: usize,
    tick: u64,
    ticks_to_run: u32,
    vision_stride: u32,
    brain_tick_stride: u32,
    speed_cost_exponent: f32,
    danger_percept_enabled: bool,
    danger_percept_blinded: bool,
) -> Vec<f32> {
    let mut wc = [0.0f32; WORLD_CONFIG_SIZE];
    fill_world_config(
        &mut wc,
        config,
        food_count,
        agent_count,
        tick,
        ticks_to_run,
        vision_stride,
        brain_tick_stride,
        speed_cost_exponent,
        danger_percept_enabled,
        danger_percept_blinded,
    );
    wc.to_vec()
}

/// Initialize brain_state buffer data for one agent from BrainConfig.
/// Xavier init for encoder, small random for predictor, zero for rest.
/// Layout is derived from config's `vision_width`/`vision_height`.
pub fn init_brain_state(config: &BrainConfig, rng: &mut impl rand::Rng) -> Vec<f32> {
    init_brain_state_for(
        config,
        &BrainLayout::new(config.vision_width, config.vision_height),
        rng,
    )
}

/// Layout-aware version for configurable vision dimensions.
pub fn init_brain_state_for(
    config: &BrainConfig,
    layout: &BrainLayout,
    rng: &mut impl rand::Rng,
) -> Vec<f32> {
    let fc = layout.feature_count;
    let mut state = vec![0.0_f32; layout.brain_stride];

    // Encoder weights: Xavier init (scale = 1/sqrt(feature_count))
    let scale = 1.0 / (fc as f32).sqrt();
    for i in 0..(fc * ENCODED_DIMENSION) {
        state[i] = (rng.random::<f32>() * 2.0 - 1.0) * scale;
    }

    // Compute dynamic offsets: O_PREDICTOR_CONTEXT_WEIGHT = fc*ENCODED_DIMENSION + ENCODED_DIMENSION + PREDICTOR_DIMENSION*ENCODED_DIMENSION
    let o_pred_weights = fc * ENCODED_DIMENSION + ENCODED_DIMENSION;
    let o_pred_ctx_wt = o_pred_weights + PREDICTOR_DIMENSION * ENCODED_DIMENSION;

    // Predictor weights: small random
    for i in 0..(PREDICTOR_DIMENSION * ENCODED_DIMENSION) {
        state[o_pred_weights + i] = (rng.random::<f32>() * 2.0 - 1.0) * 0.1;
    }

    // Predictor context weight: 0.15
    state[o_pred_ctx_wt] = 0.15;

    // Fixed deltas from O_PREDICTOR_CONTEXT_WEIGHT (same regardless of feature_count)
    let delta_hab_atten = O_HAB_ATTEN - O_PREDICTOR_CONTEXT_WEIGHT;
    let delta_exploration = O_EXPLORATION_RATE - O_PREDICTOR_CONTEXT_WEIGHT;
    let delta_fatigue_factor = O_FATIGUE_FACTOR - O_PREDICTOR_CONTEXT_WEIGHT;
    let delta_hab_sens = O_HAB_SENSITIVITY - O_PREDICTOR_CONTEXT_WEIGHT;
    let delta_hab_curiosity = O_HAB_MAX_CURIOSITY - O_PREDICTOR_CONTEXT_WEIGHT;
    let delta_fatigue_floor = O_FATIGUE_FLOOR - O_PREDICTOR_CONTEXT_WEIGHT;
    let delta_movement_speed = O_MOVEMENT_SPEED - O_PREDICTOR_CONTEXT_WEIGHT;
    // Plan 0008 visual-genome genes (contiguous after movement_speed).
    let delta_gabor_wavelength = O_GABOR_WAVELENGTH - O_PREDICTOR_CONTEXT_WEIGHT;
    let delta_gabor_aspect_ratio = O_GABOR_ASPECT_RATIO - O_PREDICTOR_CONTEXT_WEIGHT;
    let delta_dog_surround_ratio = O_DOG_SURROUND_RATIO - O_PREDICTOR_CONTEXT_WEIGHT;
    let delta_orientation_offset = O_ORIENTATION_OFFSET - O_PREDICTOR_CONTEXT_WEIGHT;

    // Habituation attenuation: 1.0 (no attenuation initially)
    for i in 0..ENCODED_DIMENSION {
        state[o_pred_ctx_wt + delta_hab_atten + i] = 1.0;
    }

    // Forward bias: agents default to moving forward so exploration
    // covers ground instead of random-walking in place.  Learning
    // adjusts this via credit assignment.
    let delta_act_biases = O_ACT_BIASES - O_PREDICTOR_CONTEXT_WEIGHT;
    state[o_pred_ctx_wt + delta_act_biases] = INITIAL_FORWARD_BIAS;

    // Exploration rate: 0.5 (balanced start)
    state[o_pred_ctx_wt + delta_exploration] = 0.5;

    // Fatigue factor: 1.0 (no fatigue)
    state[o_pred_ctx_wt + delta_fatigue_factor] = 1.0;

    // Heritable config values (stored per-agent for GPU access and kept in
    // sync by `write_agent_heritable_config()`, including `movement_speed`).
    state[o_pred_ctx_wt + delta_hab_sens] = config.habituation_sensitivity;
    state[o_pred_ctx_wt + delta_hab_curiosity] = config.max_curiosity_bonus;
    state[o_pred_ctx_wt + delta_fatigue_floor] = config.fatigue_floor;
    state[o_pred_ctx_wt + delta_movement_speed] = config.movement_speed;
    state[o_pred_ctx_wt + delta_gabor_wavelength] = config.gabor_wavelength;
    state[o_pred_ctx_wt + delta_gabor_aspect_ratio] = config.gabor_aspect_ratio;
    state[o_pred_ctx_wt + delta_dog_surround_ratio] = config.dog_surround_ratio;
    state[o_pred_ctx_wt + delta_orientation_offset] = config.orientation_offset;

    state
}

/// Initialize pattern memory buffer for one agent: all slots empty.
pub fn init_pattern_memory() -> Vec<f32> {
    vec![0.0_f32; PATTERN_STRIDE]
}

/// Seed two innate instinct patterns into the pattern buffer for one agent.
/// Instinct 0 (slot 0): danger context — high negative valence paired with
/// avoidance motor priors (backward, evasive turn).
/// Instinct 1 (slot 1): energy-gain context — high positive valence paired with
/// approach motor priors (forward, no turn bias).
/// Both are subject to normal recall, decay, and reinforcement like learned patterns.
/// Returns a vector of length PATTERN_STRIDE with the two priors pre-filled and
/// remaining slots zero-initialized.
pub fn seed_instinct_patterns(danger_strength: f32, food_strength: f32) -> Vec<f32> {
    let mut patterns = vec![0.0_f32; PATTERN_STRIDE];

    // ─ Instinct 0: danger (slot 0) ─
    // Encoded state: fixed signature for "danger detected" (e.g., all -0.5).
    for d in 0..ENCODED_DIMENSION {
        patterns[O_PAT_STATES + d * MEMORY_CAP] = -0.5;
    }
    patterns[O_PAT_NORMS] = 1.0;
    patterns[O_PAT_REINF] = 1.0;
    // Motor: [forward, turn, valence]
    patterns[O_PAT_MOTOR] = -0.7; // backward (escape behavior).
    patterns[O_PAT_MOTOR + 1] = 0.5; // evasive turn.
    patterns[O_PAT_MOTOR + 2] = -danger_strength; // strong negative valence.
                                                  // Active and metadata
    patterns[O_PAT_ACTIVE] = 1.0;
    patterns[O_ACTIVE_COUNT] = 1.0;
    patterns[O_LAST_STORED_IDX] = 1.0;

    // ─ Instinct 1: energy gain / food (slot 1) ─
    // Encoded state: fixed signature for "food/energy detected" (e.g., all +0.5).
    for d in 0..ENCODED_DIMENSION {
        patterns[O_PAT_STATES + d * MEMORY_CAP + 1] = 0.5;
    }
    patterns[O_PAT_NORMS + 1] = 1.0;
    patterns[O_PAT_REINF + 1] = 1.0;
    // Motor: [forward, turn, valence]
    patterns[O_PAT_MOTOR + 3] = 0.7; // forward (approach behavior).
    patterns[O_PAT_MOTOR + 4] = 0.0; // no turn bias.
    patterns[O_PAT_MOTOR + 5] = food_strength; // strong positive valence.
                                               // Active and metadata
    patterns[O_PAT_ACTIVE + 1] = 1.0;
    patterns[O_ACTIVE_COUNT] = 2.0;
    patterns[O_LAST_STORED_IDX] = 2.0;

    patterns
}

/// Build config buffer values from BrainConfig.
/// Layout is derived from config's `vision_width`/`vision_height`.
pub fn build_config(config: &BrainConfig) -> Vec<f32> {
    build_config_for(
        config,
        &BrainLayout::new(config.vision_width, config.vision_height),
    )
}

/// Build config buffer values with explicit layout.
///
/// `representation_dimension` is locked to the compile-time `ENCODED_DIMENSION`
/// constant (WGSL workgroup arrays cannot be runtime-sized). If the provided
/// `BrainConfig` disagrees, a warning is logged once per process (de-duped via
/// `REPR_DIM_MISMATCH_WARNED` so `reset_agents*` and evolution breeding don't
/// spam the log) and `ENCODED_DIMENSION` wins. See issue #106 for the field
/// taxonomy.
pub fn build_config_for(config: &BrainConfig, layout: &BrainLayout) -> Vec<f32> {
    if config.representation_dimension != ENCODED_DIMENSION
        && !REPR_DIM_MISMATCH_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed)
    {
        log::warn!(
            "BrainConfig.representation_dimension={} does not match compile-time \
             ENCODED_DIMENSION={}; the kernel uses ENCODED_DIMENSION and ignores the \
             config value (see issue #106). Further mismatches this run will be silent.",
            config.representation_dimension,
            ENCODED_DIMENSION,
        );
    }
    let mut cfg = vec![0.0_f32; CONFIG_SIZE];
    cfg[CFG_REPR_DIM] = ENCODED_DIMENSION as f32;
    cfg[CFG_FEATURE_COUNT] = layout.feature_count as f32;
    cfg[CFG_MEMORY_CAP] = MEMORY_CAP as f32;
    cfg[CFG_RECALL_K] = RECALL_K as f32;
    cfg[CFG_LEARNING_RATE] = config.learning_rate;
    cfg[CFG_DECAY_RATE] = config.decay_rate;
    cfg[CFG_DISTRESS_EXP] = config.distress_exponent;
    cfg[CFG_METABOLIC_RATE] = config.metabolic_rate;
    cfg[CFG_INTEGRITY_SCALE] = config.integrity_scale;
    cfg[CFG_VISUAL_CORTEX_ENABLED] = if config.visual_cortex_enabled {
        1.0
    } else {
        0.0
    };
    cfg[CFG_DANGER_PERCEPT_ENABLED] = if config.danger_percept_enabled {
        1.0
    } else {
        0.0
    };
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;
    use xagent_shared::SensoryFrame;

    #[test]
    fn default_layout_sensory_and_feature_counts() {
        let layout = BrainLayout::default();
        // Default 8×6: 192 color + 48 depth + 27 non-visual = 267 sensory
        assert_eq!(layout.sensory_stride, 267);
        // Feature count excludes 2 non-visual fields (energy_delta, integrity_delta)
        assert_eq!(layout.feature_count, 265);
        assert!(layout.sensory_stride >= layout.feature_count);
        // The static offset constants anchor to the default layout.
        assert_eq!(layout.brain_stride, BRAIN_STRIDE);
        // Brain scratch stride at 8×6: 265 + 128 + 128 + 6 + 17 + 16 + 128 + 128 + 4 = 820
        assert_eq!(layout.brain_scratch_stride, 820);
    }

    #[test]
    fn odd_grid_layout_counts() {
        // 17×13 (the range-visibility grid): 884 color + 221 depth +
        // 27 non-visual = 1132 sensory; feature_count drops the 2 deltas.
        let layout = BrainLayout::new(17, 13);
        assert_eq!(layout.sensory_stride, 1132);
        assert_eq!(layout.feature_count, 1130);
        // Brain scratch stride at 17×13: 1130 + 128 + 128 + 6 + 17 + 16 + 128 + 128 + 4 = 1685
        assert_eq!(layout.brain_scratch_stride, 1685);
    }

    #[test]
    fn brain_stride_is_consistent() {
        // The four heritable visual-genome genes (plan 0008) are contiguous right
        // after `O_MOVEMENT_SPEED`; the TD critic state follows them as the last
        // region of the brain layout: value head, prev value, three trace
        // vectors, three trace biases.
        assert_eq!(O_GABOR_WAVELENGTH, O_MOVEMENT_SPEED + 1);
        assert_eq!(O_GABOR_ASPECT_RATIO, O_GABOR_WAVELENGTH + 1);
        assert_eq!(O_DOG_SURROUND_RATIO, O_GABOR_ASPECT_RATIO + 1);
        assert_eq!(O_ORIENTATION_OFFSET, O_DOG_SURROUND_RATIO + 1);
        assert_eq!(O_VALUE_WEIGHTS, O_ORIENTATION_OFFSET + 1);
        assert_eq!(O_TRACE_BIASES, O_VALUE_WEIGHTS + 4 * ENCODED_DIMENSION + 2);
        assert_eq!(BRAIN_STRIDE, O_TRACE_BIASES + 3);
    }

    #[test]
    fn pattern_stride_is_consistent() {
        assert_eq!(PATTERN_STRIDE, O_LAST_STORED_IDX + 1);
    }

    #[test]
    fn brain_scratch_stride_offset_chain() {
        // Verify that BRAIN_SCRATCH_STRIDE equals the sum of all scratch offsets
        // for the 8×6 default layout.
        assert_eq!(SCRATCH_FEATURES, 0);
        assert_eq!(SCRATCH_ENCODED, SCRATCH_FEATURES + FEATURES_STRIDE);
        assert_eq!(SCRATCH_HABITUATED, SCRATCH_ENCODED + ENCODED_DIMENSION);
        assert_eq!(SCRATCH_HOMEO, SCRATCH_HABITUATED + ENCODED_DIMENSION);
        assert_eq!(SCRATCH_RECALL, SCRATCH_HOMEO + HOMEO_OUT_STRIDE);
        assert_eq!(
            SCRATCH_RECALL_SIMILARITY,
            SCRATCH_RECALL + RECALL_IDX_STRIDE
        );
        assert_eq!(SCRATCH_PREDICTION, SCRATCH_RECALL_SIMILARITY + RECALL_K);
        assert_eq!(SCRATCH_CREDIT, SCRATCH_PREDICTION + PREDICTOR_DIMENSION);
        assert_eq!(SCRATCH_SCALARS, SCRATCH_CREDIT + ENCODED_DIMENSION);
        assert_eq!(BRAIN_SCRATCH_STRIDE, SCRATCH_SCALARS + 4);
    }

    #[test]
    fn pack_sensory_frame_fills_buffer() {
        let layout = BrainLayout::default();
        let frame = SensoryFrame::new_blank(layout.vision_width, layout.vision_height);
        let mut buf = vec![0.0_f32; layout.sensory_stride];
        pack_sensory_frame(&frame, &layout, &mut buf);
        assert!(buf.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn pack_sensory_frame_dynamic_layout() {
        let layout = BrainLayout::new(12, 8);
        let frame = SensoryFrame::new_blank(layout.vision_width, layout.vision_height);
        let mut buf = vec![0.0_f32; layout.sensory_stride];
        pack_sensory_frame(&frame, &layout, &mut buf);
        assert!(buf.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn init_brain_state_has_correct_length() {
        let config = BrainConfig::default();
        let layout = BrainLayout::new(config.vision_width, config.vision_height);
        let mut rng = rand::rng();
        let state = init_brain_state(&config, &mut rng);
        assert_eq!(state.len(), layout.brain_stride);
    }

    #[test]
    fn brain_layout_default_values() {
        let layout = BrainLayout::default();
        let config = xagent_shared::BrainConfig::default();
        assert_eq!(layout.vision_width, config.vision_width);
        assert_eq!(layout.vision_height, config.vision_height);
        let pixels = (config.vision_width * config.vision_height) as usize;
        assert_eq!(layout.vision_color_count, pixels * 4);
        assert_eq!(layout.vision_depth_count, pixels);
        assert_eq!(layout.sensory_stride, pixels * 5 + NON_VISUAL_COUNT);
        assert_eq!(layout.feature_count, pixels * 5 + 25);
    }

    #[test]
    fn brain_layout_dynamic_is_consistent() {
        for (w, h) in [(4, 4), (6, 4), (8, 4), (8, 6), (8, 8), (12, 8), (32, 24)] {
            let layout = BrainLayout::new(w, h);
            assert_eq!(layout.vision_width, w);
            assert_eq!(layout.vision_height, h);
            let pixels = (w * h) as usize;
            assert_eq!(layout.vision_color_count, pixels * 4);
            assert_eq!(
                layout.feature_count,
                layout.vision_color_count + layout.vision_depth_count + 25
            );
            // Verify brain_stride matches the offset chain
            let fc = layout.feature_count;
            let o_pred_ctx_wt = fc * ENCODED_DIMENSION
                + ENCODED_DIMENSION
                + PREDICTOR_DIMENSION * ENCODED_DIMENSION;
            assert_eq!(layout.brain_stride, o_pred_ctx_wt + FIXED_TAIL_SIZE);
        }
    }

    #[test]
    fn retina_pixel_count_matches_config() {
        // The retina grid is locked per batch (plan 0008) and independent of the
        // legacy vision grid. A 32×32 retina must yield exactly 1024 pixels.
        let layout = BrainLayout::with_retina(8, 6, 32, 32);
        assert_eq!(layout.retina_width, 32);
        assert_eq!(layout.retina_height, 32);
        assert_eq!(layout.retina_pixel_count, 1024);

        // `from_config` threads the configured retina; the default config is
        // 32×32, so it must also produce 1024.
        let layout = BrainLayout::from_config(&BrainConfig::default());
        assert_eq!(layout.retina_pixel_count, 1024);
        // `new` derives the retina from the default config for back-compat.
        assert_eq!(BrainLayout::new(8, 6).retina_pixel_count, 1024);
    }

    #[test]
    fn luminance_weights_sum_to_one() {
        // Rec. 709 luminance weights (plan 0008, Stage 0). These three literals
        // are the single canonical source mirrored by the WGSL `retina_luminance`
        // helper in common.wgsl. A linear-light luminance must preserve a flat
        // field: the weights MUST sum to exactly 1.0, otherwise a uniform retina
        // would be scaled and the DC the cortex relies on rejecting would shift.
        // This guards against a typo drifting either copy.
        const LUMINANCE_R: f32 = 0.2126;
        const LUMINANCE_G: f32 = 0.7152;
        const LUMINANCE_B: f32 = 0.0722;
        let sum = LUMINANCE_R + LUMINANCE_G + LUMINANCE_B;
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Rec. 709 luminance weights must sum to 1.0, got {sum}"
        );

        // The helper that consumes these weights must be present in common.wgsl
        // so it concatenates into every pipeline (the fused kernel included) and
        // is available to the visual-cortex pass. common.wgsl is shared verbatim
        // by all pipelines (see gpu_kernel.rs), so presence here == compiled in.
        let common_src = include_str!("shaders/kernel/common.wgsl");
        assert!(
            common_src.contains("fn retina_luminance(color: vec3<f32>) -> f32"),
            "retina_luminance must be declared in common.wgsl"
        );
        // The WGSL copy must carry the same three weights as the test above.
        assert!(
            common_src.contains("0.2126 * color.r")
                && common_src.contains("0.7152 * color.g")
                && common_src.contains("0.0722 * color.b"),
            "common.wgsl retina_luminance weights must match the Rec. 709 weights"
        );
    }

    #[test]
    fn brain_layout_large_vision() {
        let layout = BrainLayout::new(32, 24);
        assert_eq!(layout.vision_color_count, 32 * 24 * 4);
        assert_eq!(layout.vision_depth_count, 32 * 24);
        assert_eq!(layout.sensory_stride, 32 * 24 * 5 + NON_VISUAL_COUNT);
        let expected = layout.feature_count * ENCODED_DIMENSION
            + ENCODED_DIMENSION
            + PREDICTOR_DIMENSION * ENCODED_DIMENSION
            + FIXED_TAIL_SIZE;
        assert_eq!(layout.brain_stride, expected);
    }

    #[test]
    fn init_brain_state_for_dynamic_layout() {
        let config = BrainConfig::default();
        let layout = BrainLayout::new(6, 4);
        let mut rng = rand::rng();
        let state = init_brain_state_for(&config, &layout, &mut rng);
        assert_eq!(state.len(), layout.brain_stride);
    }

    // ── Config buffer invariants ─────────────────────────────────────────

    #[test]
    fn config_size_fits_vec4_alignment() {
        assert_eq!(
            CONFIG_SIZE % 4,
            0,
            "CONFIG_SIZE must be a multiple of 4 (vec4 alignment)"
        );
    }

    #[test]
    fn config_indices_within_bounds() {
        assert!(CFG_REPR_DIM < CONFIG_SIZE);
        assert!(CFG_FEATURE_COUNT < CONFIG_SIZE);
        assert!(CFG_MEMORY_CAP < CONFIG_SIZE);
        assert!(CFG_RECALL_K < CONFIG_SIZE);
        assert!(CFG_LEARNING_RATE < CONFIG_SIZE);
        assert!(CFG_DECAY_RATE < CONFIG_SIZE);
        assert!(CFG_DISTRESS_EXP < CONFIG_SIZE);
        assert!(CFG_METABOLIC_RATE < CONFIG_SIZE);
        assert!(CFG_INTEGRITY_SCALE < CONFIG_SIZE);
    }

    #[test]
    fn build_config_packs_metabolic_rate_and_integrity_scale() {
        let mut config = BrainConfig::default();
        config.metabolic_rate = 0.01;
        config.integrity_scale = 0.02;
        let packed = build_config(&config);
        assert_eq!(packed.len(), CONFIG_SIZE);
        assert!(
            (packed[CFG_METABOLIC_RATE] - 0.01).abs() < 1e-7,
            "metabolic_rate not packed at index {}: got {}",
            CFG_METABOLIC_RATE,
            packed[CFG_METABOLIC_RATE]
        );
        assert!(
            (packed[CFG_INTEGRITY_SCALE] - 0.02).abs() < 1e-7,
            "integrity_scale not packed at index {}: got {}",
            CFG_INTEGRITY_SCALE,
            packed[CFG_INTEGRITY_SCALE]
        );
    }

    #[test]
    fn build_config_locks_repr_dim_to_encoded_dimension() {
        // representation_dimension is a locked (compile-time) field: the GPU
        // config slot always reflects ENCODED_DIMENSION, no matter what the
        // BrainConfig says. See issue #106.
        let mut config = BrainConfig::default();
        config.representation_dimension = ENCODED_DIMENSION + 64;
        let packed = build_config(&config);
        assert!((packed[CFG_REPR_DIM] - ENCODED_DIMENSION as f32).abs() < f32::EPSILON);

        config.representation_dimension = 1;
        let packed = build_config(&config);
        assert!((packed[CFG_REPR_DIM] - ENCODED_DIMENSION as f32).abs() < f32::EPSILON);
    }

    #[test]
    fn build_config_non_default_values_survive_roundtrip() {
        let mut config = BrainConfig::default();
        config.learning_rate = 0.123;
        config.decay_rate = 0.456;
        config.distress_exponent = 3.5;
        config.metabolic_rate = 0.05;
        config.integrity_scale = 0.03;
        let packed = build_config(&config);
        assert!((packed[CFG_LEARNING_RATE] - 0.123).abs() < 1e-6);
        assert!((packed[CFG_DECAY_RATE] - 0.456).abs() < 1e-6);
        assert!((packed[CFG_DISTRESS_EXP] - 3.5).abs() < 1e-6);
        assert!((packed[CFG_METABOLIC_RATE] - 0.05).abs() < 1e-7);
        assert!((packed[CFG_INTEGRITY_SCALE] - 0.03).abs() < 1e-7);
    }

    // ── Shader↔Rust constant sync ────────────────────────────────────────

    /// Parse `const NAME: TY = VALu;` lines from the shader source
    /// and return a map of name→value for all integer constants.
    fn parse_wgsl_u32_constants(src: &str) -> std::collections::HashMap<String, u32> {
        let mut map = std::collections::HashMap::new();
        for line in src.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("const ") {
                // e.g. "P_ENERGY: u32 = 11u;" or "WC_PHASE_MASK: u32 = 21u;  // comment"
                if let Some((name, tail)) = rest.split_once(':') {
                    if let Some(val_part) = tail.split('=').nth(1) {
                        // Strip inline comment if present
                        let val_part = if let Some(comment_start) = val_part.find("//") {
                            &val_part[..comment_start]
                        } else {
                            val_part
                        };
                        let val_str = val_part
                            .trim()
                            .trim_end_matches(';')
                            .trim_end_matches('u')
                            .trim();
                        if let Ok(v) = val_str.parse::<u32>() {
                            map.insert(name.trim().to_string(), v);
                        }
                    }
                }
            }
        }
        map
    }

    #[test]
    fn soa_pattern_index_covers_same_region_as_aos() {
        let mut aos_offsets = std::collections::HashSet::new();
        let mut soa_offsets = std::collections::HashSet::new();
        for pat in 0..MEMORY_CAP {
            for d in 0..ENCODED_DIMENSION {
                aos_offsets.insert(O_PAT_STATES + pat * ENCODED_DIMENSION + d);
                soa_offsets.insert(O_PAT_STATES + d * MEMORY_CAP + pat);
            }
        }
        assert_eq!(aos_offsets.len(), MEMORY_CAP * ENCODED_DIMENSION);
        assert_eq!(soa_offsets.len(), MEMORY_CAP * ENCODED_DIMENSION);
        assert_eq!(
            aos_offsets, soa_offsets,
            "SoA and AoS must cover identical offsets"
        );
        assert_eq!(*aos_offsets.iter().min().unwrap(), 0);
        assert_eq!(
            *aos_offsets.iter().max().unwrap(),
            MEMORY_CAP * ENCODED_DIMENSION - 1
        );
    }

    #[test]
    fn shader_phys_constants_match_rust() {
        let src = include_str!("shaders/kernel/common.wgsl");
        let wgsl = parse_wgsl_u32_constants(src);

        assert_eq!(wgsl["PHYS_STRIDE"], PHYS_STRIDE as u32);
        assert_eq!(wgsl["P_POS_X"], P_POS_X as u32);
        assert_eq!(wgsl["P_POS_Y"], P_POS_Y as u32);
        assert_eq!(wgsl["P_POS_Z"], P_POS_Z as u32);
        assert_eq!(wgsl["P_VEL_X"], P_VEL_X as u32);
        assert_eq!(wgsl["P_VEL_Y"], P_VEL_Y as u32);
        assert_eq!(wgsl["P_VEL_Z"], P_VEL_Z as u32);
        assert_eq!(wgsl["P_ENERGY"], P_ENERGY as u32);
        assert_eq!(wgsl["P_MAX_ENERGY"], P_MAX_ENERGY as u32);
        assert_eq!(wgsl["P_INTEGRITY"], P_INTEGRITY as u32);
        assert_eq!(wgsl["P_MAX_INTEGRITY"], P_MAX_INTEGRITY as u32);
        assert_eq!(wgsl["P_PREV_ENERGY"], P_PREV_ENERGY as u32);
        assert_eq!(wgsl["P_PREV_INTEGRITY"], P_PREV_INTEGRITY as u32);
        assert_eq!(wgsl["P_ALIVE"], P_ALIVE as u32);
        assert_eq!(wgsl["P_FOOD_COUNT"], P_FOOD_COUNT as u32);
        assert_eq!(wgsl["P_TICKS_ALIVE"], P_TICKS_ALIVE as u32);
        assert_eq!(wgsl["P_DIED_FLAG"], P_DIED_FLAG as u32);
        assert_eq!(wgsl["P_MEMORY_CAP"], P_MEMORY_CAP as u32);
        assert_eq!(wgsl["P_PROCESSING_SLOTS"], P_PROCESSING_SLOTS as u32);
        assert_eq!(wgsl["P_DEATH_COUNT"], P_DEATH_COUNT as u32);
        assert_eq!(wgsl["P_PREDICTION_ERROR"], P_PREDICTION_ERROR as u32);
        assert_eq!(
            wgsl["P_EXPLORATION_RATE_OUT"],
            P_EXPLORATION_RATE_OUT as u32
        );
        assert_eq!(wgsl["P_FATIGUE_FACTOR_OUT"], P_FATIGUE_FACTOR_OUT as u32);
        assert_eq!(wgsl["P_LAST_DEATH_TICK"], P_LAST_DEATH_TICK as u32);
        assert_eq!(
            wgsl["P_NEAREST_FOOD_DISTANCE"],
            P_NEAREST_FOOD_DISTANCE as u32
        );
        assert_eq!(wgsl["P_PREV_POTENTIAL"], P_PREV_POTENTIAL as u32);
        assert_eq!(
            wgsl["P_NEAREST_FOOD_BEARING"],
            P_NEAREST_FOOD_BEARING as u32
        );
        assert_eq!(wgsl["P_IN_DANGER_BIOME"], P_IN_DANGER_BIOME as u32);
        assert_eq!(wgsl["P_DISTANCE_TRAVELED"], P_DISTANCE_TRAVELED as u32);
        assert_eq!(wgsl["P_ENERGY_SPENT"], P_ENERGY_SPENT as u32);
        assert_eq!(wgsl["P_DANGER_PATH_LENGTH"], P_DANGER_PATH_LENGTH as u32);
        assert_eq!(
            wgsl["P_NEAREST_DANGER_DISTANCE"],
            P_NEAREST_DANGER_DISTANCE as u32
        );
        assert_eq!(
            wgsl["P_NEAREST_DANGER_BEARING"],
            P_NEAREST_DANGER_BEARING as u32
        );
        assert_eq!(
            wgsl["P_PREV_DANGER_POTENTIAL"],
            P_PREV_DANGER_POTENTIAL as u32
        );
        assert_eq!(
            wgsl["P_AVOIDANCE_SENSE_RANGE_TICKS"],
            P_AVOIDANCE_SENSE_RANGE_TICKS as u32
        );
        assert_eq!(
            wgsl["P_AVOIDANCE_TURNS_OPPOSING"],
            P_AVOIDANCE_TURNS_OPPOSING as u32
        );
        assert_eq!(
            wgsl["P_APPROACH_SENSE_RANGE_TICKS"],
            P_APPROACH_SENSE_RANGE_TICKS as u32
        );
        assert_eq!(
            wgsl["P_APPROACH_TURNS_TOWARD"],
            P_APPROACH_TURNS_TOWARD as u32
        );
    }

    #[test]
    fn shader_config_constants_match_rust() {
        let src = include_str!("shaders/kernel/common.wgsl");
        let wgsl = parse_wgsl_u32_constants(src);

        assert_eq!(wgsl["CFG_LEARNING_RATE"], CFG_LEARNING_RATE as u32);
        assert_eq!(wgsl["CFG_DECAY_RATE"], CFG_DECAY_RATE as u32);
        assert_eq!(wgsl["CFG_DISTRESS_EXP"], CFG_DISTRESS_EXP as u32);
        assert_eq!(wgsl["CFG_METABOLIC_RATE"], CFG_METABOLIC_RATE as u32);
        assert_eq!(wgsl["CFG_INTEGRITY_SCALE"], CFG_INTEGRITY_SCALE as u32);
        assert_eq!(
            wgsl["CFG_VISUAL_CORTEX_ENABLED"],
            CFG_VISUAL_CORTEX_ENABLED as u32
        );
        assert_eq!(
            wgsl["CFG_DANGER_PERCEPT_ENABLED"],
            CFG_DANGER_PERCEPT_ENABLED as u32
        );
    }

    #[test]
    fn shader_wc_constants_match_rust() {
        let src = include_str!("shaders/kernel/common.wgsl");
        let wgsl = parse_wgsl_u32_constants(src);

        assert_eq!(wgsl["WC_WORLD_SIZE"], WC_WORLD_SIZE as u32);
        assert_eq!(wgsl["WC_DT"], WC_DT as u32);
        assert_eq!(wgsl["WC_ENERGY_DEPLETION"], WC_ENERGY_DEPLETION as u32);
        assert_eq!(wgsl["WC_MOVEMENT_COST"], WC_MOVEMENT_COST as u32);
        assert_eq!(wgsl["WC_HAZARD_DAMAGE"], WC_HAZARD_DAMAGE as u32);
        assert_eq!(wgsl["WC_INTEGRITY_REGEN"], WC_INTEGRITY_REGEN as u32);
        assert_eq!(wgsl["WC_FOOD_ENERGY"], WC_FOOD_ENERGY as u32);
        assert_eq!(wgsl["WC_FOOD_RADIUS"], WC_FOOD_RADIUS as u32);
        assert_eq!(wgsl["WC_TERRAIN_VPS"], WC_TERRAIN_VPS as u32);
        assert_eq!(wgsl["WC_TERRAIN_INV_STEP"], WC_TERRAIN_INV_STEP as u32);
        assert_eq!(wgsl["WC_TERRAIN_HALF"], WC_TERRAIN_HALF as u32);
        assert_eq!(wgsl["WC_BIOME_INV_CELL"], WC_BIOME_INV_CELL as u32);
        assert_eq!(wgsl["WC_FOOD_COUNT"], WC_FOOD_COUNT as u32);
        assert_eq!(wgsl["WC_AGENT_COUNT"], WC_AGENT_COUNT as u32);
        assert_eq!(wgsl["WC_TICK"], WC_TICK as u32);
        assert_eq!(wgsl["WC_RNG_SEED"], WC_RNG_SEED as u32);
        assert_eq!(wgsl["WC_WORLD_HALF_BOUND"], WC_WORLD_HALF_BOUND as u32);
        assert_eq!(wgsl["WC_BIOME_GRID_RES"], WC_BIOME_GRID_RES as u32);
        assert_eq!(wgsl["WC_GRID_WIDTH"], WC_GRID_WIDTH as u32);
        assert_eq!(wgsl["WC_GRID_OFFSET"], WC_GRID_OFFSET as u32);
        assert_eq!(wgsl["WC_TICKS_TO_RUN"], WC_TICKS_TO_RUN as u32);
        assert_eq!(wgsl["WC_PHASE_MASK"], WC_PHASE_MASK as u32);
        assert_eq!(wgsl["WC_VISION_STRIDE"], WC_VISION_STRIDE as u32);
        assert_eq!(wgsl["WC_BRAIN_TICK_STRIDE"], WC_BRAIN_TICK_STRIDE as u32);
        assert_eq!(
            wgsl["WC_SPEED_COST_EXPONENT"],
            WC_SPEED_COST_EXPONENT as u32
        );
        assert_eq!(
            wgsl["WC_DANGER_PERCEPT_ENABLED"],
            WC_DANGER_PERCEPT_ENABLED as u32
        );
        assert_eq!(
            wgsl["WC_DANGER_PERCEPT_BLINDED"],
            WC_DANGER_PERCEPT_BLINDED as u32
        );
    }

    #[test]
    fn shader_brain_config_array_size_matches_config_size() {
        let src = include_str!("shaders/kernel/common.wgsl");
        // Find: brain_config: array<vec4<f32>, N>
        let needle = "brain_config:";
        let line = src
            .lines()
            .find(|l| l.contains(needle))
            .expect("brain_config binding not found in shader");
        // Extract N from array<vec4<f32>, N>
        let after_comma = line
            .split("array<vec4<f32>,")
            .nth(1)
            .expect("unexpected brain_config type format");
        let n_str = after_comma
            .trim()
            .trim_end_matches(';')
            .trim_end_matches('>')
            .trim();
        let shader_vec4_count: usize = n_str
            .parse()
            .expect("failed to parse brain_config array size");
        assert_eq!(
            shader_vec4_count * 4,
            CONFIG_SIZE,
            "shader brain_config has {} vec4s (={} floats) but CONFIG_SIZE={}",
            shader_vec4_count,
            shader_vec4_count * 4,
            CONFIG_SIZE
        );
    }

    #[test]
    fn bitonic_sort_top_k_matches_greedy_scan() {
        fn greedy_top_k(sims: &[f32], k: usize) -> Vec<usize> {
            let mut s = sims.to_vec();
            let mut result = Vec::new();
            for _ in 0..k {
                let mut best_idx = 0;
                let mut best_val = -3.0_f32;
                for (j, &v) in s.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best_idx = j;
                    }
                }
                if best_val <= -1.5 {
                    break;
                }
                result.push(best_idx);
                s[best_idx] = -3.0;
            }
            result
        }

        fn bitonic_top_k(sims: &[f32], k: usize) -> Vec<usize> {
            let n = sims.len();
            let mut vals = sims.to_vec();
            let mut idxs: Vec<u32> = (0..n as u32).collect();

            for stage in 0..7u32 {
                for step in 0..=stage {
                    let block_size = 1u32 << (stage + 1 - step);
                    let half = block_size >> 1;
                    for tid in 0..(n as u32 / 2) {
                        let group = tid / half;
                        let local = tid % half;
                        let i = (group * block_size + local) as usize;
                        let j = i + half as usize;
                        let descending = ((i >> (stage as usize + 1)) & 1) == 0;
                        let should_swap =
                            (descending && vals[i] < vals[j]) || (!descending && vals[i] > vals[j]);
                        if should_swap {
                            vals.swap(i, j);
                            idxs.swap(i, j);
                        }
                    }
                }
            }

            let mut result = Vec::new();
            for i in 0..k {
                if vals[i] <= -1.5 {
                    break;
                }
                result.push(idxs[i] as usize);
            }
            result
        }

        let mut sims = vec![-2.0_f32; 128];
        sims[5] = 0.95;
        sims[10] = 0.80;
        sims[0] = 0.75;
        sims[127] = 0.70;
        sims[64] = 0.65;
        sims[33] = 0.60;
        sims[99] = 0.55;
        sims[17] = 0.50;
        sims[42] = 0.45;
        sims[88] = 0.40;
        sims[3] = 0.35;
        sims[111] = 0.30;
        sims[50] = 0.25;
        sims[77] = 0.20;
        sims[22] = 0.15;
        sims[61] = 0.10;
        sims[100] = 0.05;

        let greedy = greedy_top_k(&sims, 16);
        let bitonic = bitonic_top_k(&sims, 16);

        assert_eq!(
            greedy.len(),
            bitonic.len(),
            "top-K count mismatch: greedy={}, bitonic={}",
            greedy.len(),
            bitonic.len()
        );
        for idx in &greedy {
            assert!(
                bitonic.contains(idx),
                "greedy selected index {} (sim={}) but bitonic did not. bitonic={:?}",
                idx,
                sims[*idx],
                bitonic
            );
        }
    }

    #[test]
    fn phys_stride_covers_all_fields() {
        // Highest P_* offset + 1 must equal PHYS_STRIDE
        let max_field = *[
            P_POS_X,
            P_POS_Y,
            P_POS_Z,
            P_VEL_X,
            P_VEL_Y,
            P_VEL_Z,
            P_FACING_X,
            P_FACING_Y,
            P_FACING_Z,
            P_YAW,
            P_ANGULAR_VEL,
            P_ENERGY,
            P_MAX_ENERGY,
            P_INTEGRITY,
            P_MAX_INTEGRITY,
            P_PREV_ENERGY,
            P_PREV_INTEGRITY,
            P_ALIVE,
            P_FOOD_COUNT,
            P_TICKS_ALIVE,
            P_DIED_FLAG,
            P_MEMORY_CAP,
            P_PROCESSING_SLOTS,
            P_DEATH_COUNT,
            P_PREDICTION_ERROR,
            P_EXPLORATION_RATE_OUT,
            P_FATIGUE_FACTOR_OUT,
            P_MOTOR_FWD_OUT,
            P_MOTOR_TURN_OUT,
            P_GRADIENT_OUT,
            P_URGENCY_OUT,
            P_LAST_DEATH_TICK,
            P_NEAREST_FOOD_DISTANCE,
            P_PREV_POTENTIAL,
            P_NEAREST_FOOD_BEARING,
            P_IN_DANGER_BIOME,
            P_DISTANCE_TRAVELED,
            P_ENERGY_SPENT,
            P_DANGER_PATH_LENGTH,
            P_NEAREST_DANGER_DISTANCE,
            P_NEAREST_DANGER_BEARING,
            P_PREV_DANGER_POTENTIAL,
            P_AVOIDANCE_SENSE_RANGE_TICKS,
            P_AVOIDANCE_TURNS_OPPOSING,
            P_RAW_GRADIENT_OUT,
            P_APPROACH_SENSE_RANGE_TICKS,
            P_APPROACH_TURNS_TOWARD,
        ]
        .iter()
        .max()
        .unwrap();
        assert_eq!(
            PHYS_STRIDE,
            max_field + 1,
            "PHYS_STRIDE ({}) should be highest field offset ({}) + 1",
            PHYS_STRIDE,
            max_field
        );
    }

    #[test]
    fn shader_has_16_bindings() {
        let src = include_str!("shaders/kernel/common.wgsl");
        let binding_count = src
            .lines()
            .filter(|l| l.trim().starts_with("@group(0) @binding("))
            .count();
        // Bindings 0-15: binding 13 is now brain_scratch; the numbering of
        // the remaining bindings is stable so the bind-group layout in
        // gpu_kernel.rs stays aligned.
        assert_eq!(
            binding_count, 16,
            "Expected 16 bindings (0-15: binding 13 is now brain_scratch), found {binding_count}"
        );
    }

    // ── World config buffer invariants ───────────────────────────────────

    #[test]
    fn world_config_size_fits_vec4_alignment() {
        assert_eq!(
            WORLD_CONFIG_SIZE % 4,
            0,
            "WORLD_CONFIG_SIZE must be a multiple of 4 (vec4 alignment)"
        );
        assert_eq!(
            WORLD_CONFIG_SIZE, 28,
            "WORLD_CONFIG_SIZE should be 28 (7 × vec4)"
        );
    }

    #[test]
    fn world_config_indices_within_bounds() {
        assert!(WC_WORLD_SIZE < WORLD_CONFIG_SIZE);
        assert!(WC_DT < WORLD_CONFIG_SIZE);
        assert!(WC_ENERGY_DEPLETION < WORLD_CONFIG_SIZE);
        assert!(WC_MOVEMENT_COST < WORLD_CONFIG_SIZE);
        assert!(WC_HAZARD_DAMAGE < WORLD_CONFIG_SIZE);
        assert!(WC_INTEGRITY_REGEN < WORLD_CONFIG_SIZE);
        assert!(WC_FOOD_ENERGY < WORLD_CONFIG_SIZE);
        assert!(WC_FOOD_RADIUS < WORLD_CONFIG_SIZE);
        assert!(WC_TERRAIN_VPS < WORLD_CONFIG_SIZE);
        assert!(WC_TERRAIN_INV_STEP < WORLD_CONFIG_SIZE);
        assert!(WC_TERRAIN_HALF < WORLD_CONFIG_SIZE);
        assert!(WC_BIOME_INV_CELL < WORLD_CONFIG_SIZE);
        assert!(WC_FOOD_COUNT < WORLD_CONFIG_SIZE);
        assert!(WC_AGENT_COUNT < WORLD_CONFIG_SIZE);
        assert!(WC_TICK < WORLD_CONFIG_SIZE);
        assert!(WC_RNG_SEED < WORLD_CONFIG_SIZE);
        assert!(WC_WORLD_HALF_BOUND < WORLD_CONFIG_SIZE);
        assert!(WC_BIOME_GRID_RES < WORLD_CONFIG_SIZE);
        assert!(WC_GRID_WIDTH < WORLD_CONFIG_SIZE);
        assert!(WC_GRID_OFFSET < WORLD_CONFIG_SIZE);
        assert!(WC_TICKS_TO_RUN < WORLD_CONFIG_SIZE);
        assert!(WC_PHASE_MASK < WORLD_CONFIG_SIZE);
        assert!(WC_VISION_STRIDE < WORLD_CONFIG_SIZE);
        assert!(WC_BRAIN_TICK_STRIDE < WORLD_CONFIG_SIZE);
        assert!(WC_SPEED_COST_EXPONENT < WORLD_CONFIG_SIZE);
        assert!(WC_DANGER_PERCEPT_ENABLED < WORLD_CONFIG_SIZE);
        assert!(WC_DANGER_PERCEPT_BLINDED < WORLD_CONFIG_SIZE);
    }
}
