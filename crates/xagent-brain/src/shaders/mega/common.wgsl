// ── Mega-kernel common definitions ──────────────────────────────────────────
// Shared constants, buffer bindings, and helper functions for all phases.
// Values MUST match buffers.rs and the per-shader constants injected by
// wgsl_constants() / wgsl_physics_constants().

// ── Vision grid (string-replaced at shader compile time) ───────────────────

const VISION_W: u32 = 8u;
const VISION_H: u32 = 6u;

// ── Derived vision / sensory constants ─────────────────────────────────────

const VISION_RAYS: u32 = VISION_W * VISION_H;
const VISION_COLOR_COUNT: u32 = VISION_RAYS * 4u;
const VISION_DEPTH_COUNT: u32 = VISION_RAYS;
const MAX_TOUCH_CONTACTS: u32 = 4u;
const SENSORY_STRIDE: u32 = VISION_COLOR_COUNT + VISION_DEPTH_COUNT + 27u;

// ── Brain dimensions ────────────────────────────────────────────────────────

const DIM: u32 = 32u;
const FEATURE_COUNT: u32 = VISION_COLOR_COUNT + 25u;
const MEMORY_CAP: u32 = 128u;
const RECALL_K: u32 = 16u;
const ACTION_HISTORY_LEN: u32 = 64u;
const ERROR_HISTORY_LEN: u32 = 128u;

// ── Brain state offsets (derived from FEATURE_COUNT) ────────────────────────

const O_ENC_WEIGHTS: u32 = 0u;
const O_ENC_BIASES: u32 = FEATURE_COUNT * DIM;
const O_PRED_WEIGHTS: u32 = O_ENC_BIASES + DIM;
const O_PRED_CTX_WT: u32 = O_PRED_WEIGHTS + DIM * DIM;
const O_PRED_ERR_RING: u32 = O_PRED_CTX_WT + 1u;
const O_PRED_ERR_CURSOR: u32 = O_PRED_ERR_RING + ERROR_HISTORY_LEN;
const O_PRED_ERR_COUNT: u32 = O_PRED_ERR_CURSOR + 1u;
const O_HAB_EMA: u32 = O_PRED_ERR_COUNT + 1u;
const O_HAB_ATTEN: u32 = O_HAB_EMA + DIM;
const O_PREV_ENCODED: u32 = O_HAB_ATTEN + DIM;
const O_HOMEO: u32 = O_PREV_ENCODED + DIM;
const O_ACT_FWD_WTS: u32 = O_HOMEO + 6u;
const O_ACT_TURN_WTS: u32 = O_ACT_FWD_WTS + DIM;
const O_ACT_BIASES: u32 = O_ACT_TURN_WTS + DIM;
const O_EXPLORATION_RATE: u32 = O_ACT_BIASES + 2u;
const O_FATIGUE_FWD_RING: u32 = O_EXPLORATION_RATE + 1u;
const O_FATIGUE_TURN_RING: u32 = O_FATIGUE_FWD_RING + ACTION_HISTORY_LEN;
const O_FATIGUE_CURSOR: u32 = O_FATIGUE_TURN_RING + ACTION_HISTORY_LEN;
const O_FATIGUE_FACTOR: u32 = O_FATIGUE_CURSOR + 1u;
const O_FATIGUE_LEN: u32 = O_FATIGUE_FACTOR + 1u;
const O_PREV_PREDICTION: u32 = O_FATIGUE_LEN + 1u;
const O_TICK_COUNT: u32 = O_PREV_PREDICTION + DIM;
const O_HAB_SENSITIVITY: u32 = O_TICK_COUNT + 1u;
const O_HAB_MAX_CURIOSITY: u32 = O_HAB_SENSITIVITY + 1u;
const O_FATIGUE_RECOVERY: u32 = O_HAB_MAX_CURIOSITY + 1u;
const O_FATIGUE_FLOOR: u32 = O_FATIGUE_RECOVERY + 1u;

// ── Per-agent buffer strides ────────────────────────────────────────────────

const BRAIN_STRIDE: u32 = O_FATIGUE_FLOOR + 1u;
const PATTERN_STRIDE: u32 = 5251u;
const HISTORY_STRIDE: u32 = 2370u;
const FEATURES_STRIDE: u32 = FEATURE_COUNT;
const DECISION_STRIDE: u32 = 68u;      // prediction(32) + credit(32) + motor(4)
const HOMEO_OUT_STRIDE: u32 = 6u;
const RECALL_IDX_STRIDE: u32 = 17u;    // 16 indices + 1 count

// ── Pattern memory offsets ──────────────────────────────────────────────────
// O_PAT_STATES uses SoA (Structure-of-Arrays) layout: [dim][pattern]
// Index as: p_base + d * MEMORY_CAP + pattern_idx
// This gives coalesced reads when 128 threads each read one pattern.
// Other regions (norms, reinf, motor, meta, active) remain AoS.

const O_PAT_STATES: u32 = 0u;
const O_PAT_NORMS: u32 = 4096u;
const O_PAT_REINF: u32 = 4224u;
const O_PAT_MOTOR: u32 = 4352u;
const O_PAT_META: u32 = 4736u;
const O_PAT_ACTIVE: u32 = 5120u;
const O_ACTIVE_COUNT: u32 = 5248u;
const O_MIN_REINF_IDX: u32 = 5249u;
const O_LAST_STORED_IDX: u32 = 5250u;

// ── Action history offsets ──────────────────────────────────────────────────

const O_MOTOR_RING: u32 = 0u;
const O_STATE_RING: u32 = 320u;
const O_HIST_CURSOR: u32 = 2368u;
const O_HIST_LEN: u32 = 2369u;

// ── Config buffer offsets ───────────────────────────────────────────────────

const CFG_LEARNING_RATE: u32 = 4u;
const CFG_DECAY_RATE: u32 = 5u;
const CFG_DISTRESS_EXP: u32 = 6u;
const CFG_METABOLIC_RATE: u32 = 7u;
const CFG_INTEGRITY_SCALE: u32 = 8u;

// ── Agent physics buffer layout (P_*) ───────────────────────────────────────

const PHYS_STRIDE: u32 = 31u;
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

// ── Food buffer layout ─────────────────────────────────────────────────────

const FOOD_STATE_STRIDE: u32 = 4u;
const F_POS_X: u32 = 0u;
const F_POS_Y: u32 = 1u;
const F_POS_Z: u32 = 2u;
const F_RESPAWN_TIMER: u32 = 3u;

// ── Physics constants ───────────────────────────────────────────────────────

const GRAVITY: f32 = 20.0;
const MOVE_SPEED: f32 = 8.0;
const TURN_SPEED: f32 = 3.0;
const AGENT_HALF_HEIGHT: f32 = 1.0;
const METABOLIC_BASE_COST: f32 = 0.0001;
const METABOLIC_MEMORY_COST: f32 = 0.00003;
const METABOLIC_PROCESSING_COST: f32 = 0.0001;
const FOOD_CONSUME_RADIUS_SQ: f32 = 6.25;
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

// ── Food respawn constants ──────────────────────────────────────────────────

const FOOD_RESPAWN_TIME: f32 = 10.0;
const FOOD_HEIGHT_OFFSET: f32 = 0.35;
const FOOD_RESPAWN_ATTEMPTS: u32 = 64u;

// ── Vision constants ────────────────────────────────────────────────────────

const VISION_FOV_HALF: f32 = 0.7853982;   // PI/4 = 45 degrees half-FOV
const VISION_MAX_DIST: f32 = 30.0;
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

// ── Habituation / homeostasis constants ─────────────────────────────────────

const HAB_EMA_ALPHA: f32 = 0.02;
const ATTEN_FLOOR: f32 = 0.1;
const ENERGY_WEIGHT: f32 = 0.6;
const INTEGRITY_WEIGHT: f32 = 0.4;
const FAST_ALPHA: f32 = 0.6;
const MED_ALPHA: f32 = 0.04;
const SLOW_ALPHA: f32 = 0.004;
const DISTRESS_SCALE: f32 = 10.0;
const MAX_DISTRESS: f32 = 10.0;
const GRAD_BLEND_FAST: f32 = 0.5;
const GRAD_BLEND_MED: f32 = 0.35;
const GRAD_BLEND_SLOW: f32 = 0.15;

// ── Predict-and-act constants ───────────────────────────────────────────────

const CREDIT_DECAY: f32 = 0.04;
const WEIGHT_LR: f32 = 0.10;
const PAIN_AMP: f32 = 3.0;
const DEADZONE: f32 = 0.01;
const MAX_WEIGHT_NORM: f32 = 2.0;
const ANTICIPATION_WEIGHT: f32 = 0.5;
const TONIC_CREDIT_SCALE: f32 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════
// Buffer bindings — 15 storage + 2 uniform, single bind group
// ═══════════════════════════════════════════════════════════════════════════

@group(0) @binding(0)  var<storage, read_write> agent_phys:        array<f32>;
@group(0) @binding(1)  var<storage, read_write> decision_buf:      array<f32>;
@group(0) @binding(2)  var<storage, read>       heightmap:         array<f32>;
@group(0) @binding(3)  var<storage, read>       biome_grid:        array<u32>;
@group(0) @binding(4)  var<uniform>             wconfig:           array<vec4<f32>, 6>;
@group(0) @binding(5)  var<storage, read_write> food_state:        array<f32>;
@group(0) @binding(6)  var<storage, read_write> food_flags:        array<atomic<u32>>;
@group(0) @binding(7)  var<storage, read_write> food_grid:         array<atomic<u32>>;
@group(0) @binding(8)  var<storage, read_write> agent_grid:        array<atomic<u32>>;
@group(0) @binding(9)  var<storage, read_write> collision_scratch: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> sensory_buf:       array<f32>;
@group(0) @binding(11) var<storage, read_write> brain_state:       array<f32>;
@group(0) @binding(12) var<storage, read_write> pattern_buf:       array<f32>;
@group(0) @binding(13) var<storage, read_write> history_buf:       array<f32>;
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
    let col = min(u32((x + biome_half) * biome_inv), 255u);
    let row = min(u32((z + biome_half) * biome_inv), 255u);
    return biome_grid[row * 256u + col];
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
