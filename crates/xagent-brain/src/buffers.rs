//! GPU buffer layout constants, AgentBrainState, and packing utilities.
//!
//! All persistent brain state is stored in three flat f32 arrays on GPU:
//! - brain_state_buf: encoder weights, predictor, habituation, homeostasis, action, fatigue
//! - pattern_buf: pattern memory (states, norms, reinforcement, motor, meta, active)
//! - history_buf: action history ring buffer
//!
//! Integer values (cursors, counts, ticks) are stored as f32 and cast
//! via bitcast in WGSL. Safe for exact integers up to 2^24 = 16,777,216.

use xagent_shared::{BrainConfig, SensoryFrame, TouchContact};

// ── Dimensions (fixed for default config) ─────────────────────────────

pub const DIM: usize = 32;
pub const FEATURE_COUNT: usize = 217;
pub const MEMORY_CAP: usize = 128;
pub const RECALL_K: usize = 16;
pub const ACTION_HISTORY_LEN: usize = 64;
pub const ERROR_HISTORY_LEN: usize = 128;
pub const MAX_TOUCH_CONTACTS: usize = 4;
pub const TOUCH_FEATURES: usize = 4; // dir_x, dir_z, intensity, tag/4

// ── Sensory input layout (CPU → GPU upload) ───────────────────────────

pub const VISION_W: usize = 8;
pub const VISION_H: usize = 6;
pub const VISION_COLOR_COUNT: usize = VISION_W * VISION_H * 4; // 192 RGBA
pub const VISION_DEPTH_COUNT: usize = VISION_W * VISION_H;     // 48
pub const NON_VISUAL_COUNT: usize = 3 + 3 + 1 + 1 + 1 + 1 + 1 + MAX_TOUCH_CONTACTS * TOUCH_FEATURES;
// velocity(3) + facing(3) + angular_vel(1) + energy(1) + integrity(1) + e_delta(1) + i_delta(1) + touch(16)
pub const SENSORY_STRIDE: usize = VISION_COLOR_COUNT + VISION_DEPTH_COUNT + NON_VISUAL_COUNT;
// 192 + 48 + 27 = 267

// ── Brain state buffer offsets (per agent) ────────────────────────────

pub const O_ENC_WEIGHTS: usize = 0;
pub const O_ENC_BIASES: usize = O_ENC_WEIGHTS + FEATURE_COUNT * DIM; // 6944
pub const O_PRED_WEIGHTS: usize = O_ENC_BIASES + DIM;                // 6976
pub const O_PRED_CTX_WT: usize = O_PRED_WEIGHTS + DIM * DIM;        // 8000
pub const O_PRED_ERR_RING: usize = O_PRED_CTX_WT + 1;               // 8001
pub const O_PRED_ERR_CURSOR: usize = O_PRED_ERR_RING + ERROR_HISTORY_LEN; // 8129
pub const O_PRED_ERR_COUNT: usize = O_PRED_ERR_CURSOR + 1;          // 8130
pub const O_HAB_EMA: usize = O_PRED_ERR_COUNT + 1;                  // 8131
pub const O_HAB_ATTEN: usize = O_HAB_EMA + DIM;                     // 8163
pub const O_PREV_ENCODED: usize = O_HAB_ATTEN + DIM;                // 8195
pub const O_HOMEO: usize = O_PREV_ENCODED + DIM;                    // 8227
// homeo: [grad_fast, grad_med, grad_slow, urgency, prev_energy, prev_integrity]
pub const O_ACT_FWD_WTS: usize = O_HOMEO + 6;                      // 8233
pub const O_ACT_TURN_WTS: usize = O_ACT_FWD_WTS + DIM;             // 8265
pub const O_ACT_BIASES: usize = O_ACT_TURN_WTS + DIM;              // 8297
// act_biases: [fwd_bias, turn_bias]
pub const O_EXPLORATION_RATE: usize = O_ACT_BIASES + 2;            // 8299
pub const O_FATIGUE_FWD_RING: usize = O_EXPLORATION_RATE + 1;      // 8300
pub const O_FATIGUE_TURN_RING: usize = O_FATIGUE_FWD_RING + ACTION_HISTORY_LEN; // 8364
pub const O_FATIGUE_CURSOR: usize = O_FATIGUE_TURN_RING + ACTION_HISTORY_LEN;   // 8428
pub const O_FATIGUE_FACTOR: usize = O_FATIGUE_CURSOR + 1;          // 8429
pub const O_FATIGUE_LEN: usize = O_FATIGUE_FACTOR + 1;             // 8430
pub const O_PREV_PREDICTION: usize = O_FATIGUE_LEN + 1;            // 8431
pub const O_TICK_COUNT: usize = O_PREV_PREDICTION + DIM;           // 8463
pub const O_HAB_SENSITIVITY: usize = O_TICK_COUNT + 1;             // 8464
pub const O_HAB_MAX_CURIOSITY: usize = O_HAB_SENSITIVITY + 1;      // 8465
pub const O_FATIGUE_RECOVERY: usize = O_HAB_MAX_CURIOSITY + 1;     // 8466
pub const O_FATIGUE_FLOOR: usize = O_FATIGUE_RECOVERY + 1;         // 8467
pub const BRAIN_STRIDE: usize = O_FATIGUE_FLOOR + 1;               // 8468

/// Fixed tail size: fields from O_PRED_CTX_WT through O_FATIGUE_FLOOR + 1.
/// This is layout-independent — the tail doesn't change when vision dimensions vary.
pub const FIXED_TAIL_SIZE: usize = BRAIN_STRIDE - O_PRED_CTX_WT;   // 468

// ── Pattern memory buffer offsets (per agent) ─────────────────────────

pub const O_PAT_STATES: usize = 0;
pub const O_PAT_NORMS: usize = O_PAT_STATES + MEMORY_CAP * DIM;    // 4096
pub const O_PAT_REINF: usize = O_PAT_NORMS + MEMORY_CAP;           // 4224
pub const O_PAT_MOTOR: usize = O_PAT_REINF + MEMORY_CAP;           // 4352
// motor: [forward, turn, outcome_valence] × cap
pub const O_PAT_META: usize = O_PAT_MOTOR + MEMORY_CAP * 3;        // 4736
// meta: [created_at, last_accessed, activation_count] × cap
pub const O_PAT_ACTIVE: usize = O_PAT_META + MEMORY_CAP * 3;       // 5120
pub const O_ACTIVE_COUNT: usize = O_PAT_ACTIVE + MEMORY_CAP;       // 5248
pub const O_MIN_REINF_IDX: usize = O_ACTIVE_COUNT + 1;             // 5249
pub const O_LAST_STORED_IDX: usize = O_MIN_REINF_IDX + 1;          // 5250
pub const PATTERN_STRIDE: usize = O_LAST_STORED_IDX + 1;           // 5251

// ── Action history buffer offsets (per agent) ─────────────────────────

pub const O_MOTOR_RING: usize = 0;
// motor_ring: [forward, turn, tick, gradient, _pad] × ACTION_HISTORY_LEN
pub const O_STATE_RING: usize = O_MOTOR_RING + ACTION_HISTORY_LEN * 5; // 320
pub const O_HIST_CURSOR: usize = O_STATE_RING + ACTION_HISTORY_LEN * DIM; // 2368
pub const O_HIST_LEN: usize = O_HIST_CURSOR + 1;                   // 2369
pub const HISTORY_STRIDE: usize = O_HIST_LEN + 1;                  // 2370

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
pub const PHYS_STRIDE: usize = 27;
/// Brain runs once every N physics ticks. Must match the cycle logic in dispatch_batch.
pub const BRAIN_TICK_STRIDE: u32 = 4;

// ── Runtime layout for configurable vision dimensions ─────────────────

/// Runtime-computed buffer layout that depends on vision_w × vision_h.
/// All strides and offsets cascade from the pixel count.
#[derive(Clone, Debug)]
pub struct BrainLayout {
    pub vision_w: u32,
    pub vision_h: u32,
    pub vision_color_count: usize,
    pub vision_depth_count: usize,
    pub feature_count: usize,
    pub sensory_stride: usize,
    pub brain_stride: usize,
}

impl BrainLayout {
    pub fn new(vision_w: u32, vision_h: u32) -> Self {
        let pixel_count = (vision_w as usize)
            .checked_mul(vision_h as usize)
            .expect("vision dimensions overflow pixel count");
        let color_count = pixel_count
            .checked_mul(4)
            .expect("vision dimensions overflow color count");
        let depth_count = pixel_count;
        let feature_count = color_count
            .checked_add(25)
            .expect("vision dimensions overflow feature count");
        let sensory_stride = color_count
            .checked_add(depth_count)
            .and_then(|v| v.checked_add(NON_VISUAL_COUNT))
            .expect("vision dimensions overflow sensory stride");
        // brain_stride = feature_count * DIM + DIM + DIM*DIM + FIXED_TAIL_SIZE
        let brain_stride = feature_count
            .checked_mul(DIM)
            .and_then(|v| v.checked_add(DIM))
            .and_then(|v| v.checked_add(DIM * DIM))
            .and_then(|v| v.checked_add(FIXED_TAIL_SIZE))
            .expect("vision dimensions overflow brain stride");
        Self {
            vision_w,
            vision_h,
            vision_color_count: color_count,
            vision_depth_count: depth_count,
            feature_count,
            sensory_stride,
            brain_stride,
        }
    }
}

impl Default for BrainLayout {
    fn default() -> Self {
        Self::new(8, 6)
    }
}

// ── Food buffer layout (per food item) ───────────────────────────────

pub const F_POS_X: usize = 0;
pub const F_POS_Y: usize = 1;
pub const F_POS_Z: usize = 2;
pub const F_RESPAWN_TIMER: usize = 3;
pub const FOOD_STATE_STRIDE: usize = 4;

// ── Grid constants ───────────────────────────────────────────────────

pub const GRID_CELL_SIZE: f32 = 8.0;
pub const FOOD_GRID_MAX_PER_CELL: usize = 16;
pub const FOOD_GRID_CELL_STRIDE: usize = 1 + FOOD_GRID_MAX_PER_CELL; // 17
pub const AGENT_GRID_MAX_PER_CELL: usize = 32;
pub const AGENT_GRID_CELL_STRIDE: usize = 1 + AGENT_GRID_MAX_PER_CELL; // 33

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
pub const WC_PHASE_MASK: usize = 21;  // bit0=physics, bit1=vision, bit2=brain
pub const WC_VISION_STRIDE: usize = 22;
pub const WC_BRAIN_TICK_STRIDE: usize = 23;
pub const WORLD_CONFIG_SIZE: usize = 24; // padded to 6 × vec4

// ── Transient buffer sizes (per agent) ────────────────────────────────

pub const FEATURES_STRIDE: usize = FEATURE_COUNT;     // 217
pub const ENCODED_STRIDE: usize = DIM;                 // 32
pub const HABITUATED_STRIDE: usize = DIM;              // 32
pub const HOMEO_OUT_STRIDE: usize = 6;                 // grad, raw_grad, urgency, grad_fast, grad_med, grad_slow
pub const SIMILARITIES_STRIDE: usize = MEMORY_CAP;     // 128
pub const RECALL_IDX_STRIDE: usize = RECALL_K + 1;     // 16 indices + 1 count
pub const DECISION_STRIDE: usize = DIM + DIM + 4;      // prediction(32) + credit(32) + motor(4) = 68

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
pub const CONFIG_SIZE: usize = 12; // padded to 12 for uniform vec4 alignment (3 × vec4)

// ── AgentBrainState (CPU-side snapshot for evolution) ──────────────────

/// Serializable snapshot of one agent's full brain state.
/// Used for cross-generation inheritance, mutation, and DB persistence.
#[derive(Clone, Debug)]
pub struct AgentBrainState {
    pub brain_state: Vec<f32>,   // BRAIN_STRIDE f32s
    pub patterns: Vec<f32>,      // PATTERN_STRIDE f32s
    pub history: Vec<f32>,       // HISTORY_STRIDE f32s
}

impl AgentBrainState {
    pub fn new_blank() -> Self {
        Self::new_for(BRAIN_STRIDE)
    }

    pub fn new_for(brain_stride: usize) -> Self {
        Self {
            brain_state: vec![0.0; brain_stride],
            patterns: vec![0.0; PATTERN_STRIDE],
            history: vec![0.0; HISTORY_STRIDE],
        }
    }
}

// ── Sensory frame packing ─────────────────────────────────────────────

/// Pack a SensoryFrame into a flat f32 slice for GPU upload.
/// Selects up to 4 highest-intensity touch contacts, zero-pads the rest.
pub fn pack_sensory_frame(frame: &SensoryFrame, out: &mut [f32]) {
    debug_assert!(out.len() >= SENSORY_STRIDE);

    let mut offset = 0;

    // Vision color (192 f32)
    let color_len = frame.vision.color.len().min(VISION_COLOR_COUNT);
    out[offset..offset + color_len].copy_from_slice(&frame.vision.color[..color_len]);
    for i in color_len..VISION_COLOR_COUNT {
        out[offset + i] = 0.0;
    }
    offset += VISION_COLOR_COUNT;

    // Vision depth (48 f32)
    let depth_len = frame.vision.depth.len().min(VISION_DEPTH_COUNT);
    out[offset..offset + depth_len].copy_from_slice(&frame.vision.depth[..depth_len]);
    for i in depth_len..VISION_DEPTH_COUNT {
        out[offset + i] = 0.0;
    }
    offset += VISION_DEPTH_COUNT;

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
    sorted_contacts.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap_or(std::cmp::Ordering::Equal));

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

/// Generate WGSL constants header shared by all shaders.
pub fn wgsl_constants() -> String {
    format!(
        "// Auto-generated constants — do not edit manually
const DIM: u32 = {DIM}u;
const FEATURE_COUNT: u32 = {FEATURE_COUNT}u;
const MEMORY_CAP: u32 = {MEMORY_CAP}u;
const RECALL_K: u32 = {RECALL_K}u;
const ACTION_HISTORY_LEN: u32 = {ACTION_HISTORY_LEN}u;
const ERROR_HISTORY_LEN: u32 = {ERROR_HISTORY_LEN}u;
const SENSORY_STRIDE: u32 = {SENSORY_STRIDE}u;
const BRAIN_STRIDE: u32 = {BRAIN_STRIDE}u;
const PATTERN_STRIDE: u32 = {PATTERN_STRIDE}u;
const HISTORY_STRIDE: u32 = {HISTORY_STRIDE}u;
const FEATURES_STRIDE: u32 = {FEATURES_STRIDE}u;
const DECISION_STRIDE: u32 = {DECISION_STRIDE}u;
const HOMEO_OUT_STRIDE: u32 = {HOMEO_OUT_STRIDE}u;
const RECALL_IDX_STRIDE: u32 = {RECALL_IDX_STRIDE}u;

// Brain state offsets
const O_ENC_WEIGHTS: u32 = {O_ENC_WEIGHTS}u;
const O_ENC_BIASES: u32 = {O_ENC_BIASES}u;
const O_PRED_WEIGHTS: u32 = {O_PRED_WEIGHTS}u;
const O_PRED_CTX_WT: u32 = {O_PRED_CTX_WT}u;
const O_PRED_ERR_RING: u32 = {O_PRED_ERR_RING}u;
const O_PRED_ERR_CURSOR: u32 = {O_PRED_ERR_CURSOR}u;
const O_PRED_ERR_COUNT: u32 = {O_PRED_ERR_COUNT}u;
const O_HAB_EMA: u32 = {O_HAB_EMA}u;
const O_HAB_ATTEN: u32 = {O_HAB_ATTEN}u;
const O_PREV_ENCODED: u32 = {O_PREV_ENCODED}u;
const O_HOMEO: u32 = {O_HOMEO}u;
const O_ACT_FWD_WTS: u32 = {O_ACT_FWD_WTS}u;
const O_ACT_TURN_WTS: u32 = {O_ACT_TURN_WTS}u;
const O_ACT_BIASES: u32 = {O_ACT_BIASES}u;
const O_EXPLORATION_RATE: u32 = {O_EXPLORATION_RATE}u;
const O_FATIGUE_FWD_RING: u32 = {O_FATIGUE_FWD_RING}u;
const O_FATIGUE_TURN_RING: u32 = {O_FATIGUE_TURN_RING}u;
const O_FATIGUE_CURSOR: u32 = {O_FATIGUE_CURSOR}u;
const O_FATIGUE_FACTOR: u32 = {O_FATIGUE_FACTOR}u;
const O_FATIGUE_LEN: u32 = {O_FATIGUE_LEN}u;
const O_PREV_PREDICTION: u32 = {O_PREV_PREDICTION}u;
const O_TICK_COUNT: u32 = {O_TICK_COUNT}u;
const O_HAB_SENSITIVITY: u32 = {O_HAB_SENSITIVITY}u;
const O_HAB_MAX_CURIOSITY: u32 = {O_HAB_MAX_CURIOSITY}u;
const O_FATIGUE_RECOVERY: u32 = {O_FATIGUE_RECOVERY}u;
const O_FATIGUE_FLOOR: u32 = {O_FATIGUE_FLOOR}u;

// Pattern memory offsets
const O_PAT_STATES: u32 = {O_PAT_STATES}u;
const O_PAT_NORMS: u32 = {O_PAT_NORMS}u;
const O_PAT_REINF: u32 = {O_PAT_REINF}u;
const O_PAT_MOTOR: u32 = {O_PAT_MOTOR}u;
const O_PAT_META: u32 = {O_PAT_META}u;
const O_PAT_ACTIVE: u32 = {O_PAT_ACTIVE}u;
const O_ACTIVE_COUNT: u32 = {O_ACTIVE_COUNT}u;
const O_MIN_REINF_IDX: u32 = {O_MIN_REINF_IDX}u;
const O_LAST_STORED_IDX: u32 = {O_LAST_STORED_IDX}u;

// Action history offsets
const O_MOTOR_RING: u32 = {O_MOTOR_RING}u;
const O_STATE_RING: u32 = {O_STATE_RING}u;
const O_HIST_CURSOR: u32 = {O_HIST_CURSOR}u;
const O_HIST_LEN: u32 = {O_HIST_LEN}u;

// Config offsets
const CFG_LEARNING_RATE: u32 = {CFG_LEARNING_RATE}u;
const CFG_DECAY_RATE: u32 = {CFG_DECAY_RATE}u;
const CFG_DISTRESS_EXP: u32 = {CFG_DISTRESS_EXP}u;

fn fast_tanh(x: f32) -> f32 {{
    if (abs(x) > 4.5) {{ return sign(x); }}
    let x2 = x * x;
    return x * (27.0 + x2) / (27.0 + 9.0 * x2);
}}

fn pcg_hash(input: u32) -> u32 {{
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}}

fn rand_f32(seed: u32) -> f32 {{
    return f32(pcg_hash(seed)) / 4294967295.0;
}}

fn rand_normal(seed1: u32, seed2: u32) -> f32 {{
    let u1 = max(rand_f32(seed1), 0.0001);
    let u2 = rand_f32(seed2);
    return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}}
",
    )
}

/// Generate WGSL constants for physics/vision shaders.
/// These are separate from brain constants to avoid name collisions.
pub fn wgsl_physics_constants(
    world_size: f32,
    food_count: usize,
    agent_count: usize,
) -> String {
    let gw = grid_width(world_size);
    let go = gw / 2;
    let terrain_vps: u32 = 129;
    let terrain_step = world_size / 128.0;
    format!(
r#"// Auto-generated physics constants — do not edit manually
const PHYS_STRIDE: u32 = {PHYS_STRIDE}u;
const P_POS_X: u32 = {P_POS_X}u;
const P_POS_Y: u32 = {P_POS_Y}u;
const P_POS_Z: u32 = {P_POS_Z}u;
const P_VEL_X: u32 = {P_VEL_X}u;
const P_VEL_Y: u32 = {P_VEL_Y}u;
const P_VEL_Z: u32 = {P_VEL_Z}u;
const P_FACING_X: u32 = {P_FACING_X}u;
const P_FACING_Y: u32 = {P_FACING_Y}u;
const P_FACING_Z: u32 = {P_FACING_Z}u;
const P_YAW: u32 = {P_YAW}u;
const P_ANGULAR_VEL: u32 = {P_ANGULAR_VEL}u;
const P_ENERGY: u32 = {P_ENERGY}u;
const P_MAX_ENERGY: u32 = {P_MAX_ENERGY}u;
const P_INTEGRITY: u32 = {P_INTEGRITY}u;
const P_MAX_INTEGRITY: u32 = {P_MAX_INTEGRITY}u;
const P_PREV_ENERGY: u32 = {P_PREV_ENERGY}u;
const P_PREV_INTEGRITY: u32 = {P_PREV_INTEGRITY}u;
const P_ALIVE: u32 = {P_ALIVE}u;
const P_FOOD_COUNT: u32 = {P_FOOD_COUNT}u;
const P_TICKS_ALIVE: u32 = {P_TICKS_ALIVE}u;
const P_DIED_FLAG: u32 = {P_DIED_FLAG}u;
const P_MEMORY_CAP: u32 = {P_MEMORY_CAP}u;
const P_PROCESSING_SLOTS: u32 = {P_PROCESSING_SLOTS}u;

const FOOD_STATE_STRIDE: u32 = {FOOD_STATE_STRIDE}u;
const F_POS_X: u32 = {F_POS_X}u;
const F_POS_Y: u32 = {F_POS_Y}u;
const F_POS_Z: u32 = {F_POS_Z}u;
const F_RESPAWN_TIMER: u32 = {F_RESPAWN_TIMER}u;

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

const CELL_SIZE: f32 = {GRID_CELL_SIZE};
const GRID_WIDTH: u32 = {gw}u;
const GRID_OFFSET: i32 = {go};
const FOOD_GRID_MAX_PER_CELL: u32 = {FOOD_GRID_MAX_PER_CELL}u;
const FOOD_GRID_CELL_STRIDE: u32 = {FOOD_GRID_CELL_STRIDE}u;
const AGENT_GRID_MAX_PER_CELL: u32 = {AGENT_GRID_MAX_PER_CELL}u;
const AGENT_GRID_CELL_STRIDE: u32 = {AGENT_GRID_CELL_STRIDE}u;

const TERRAIN_VPS: u32 = {terrain_vps}u;
const TERRAIN_MAX_IDX: u32 = {max_idx}u;
const TERRAIN_INV_STEP: f32 = {inv_step};
const TERRAIN_HALF: f32 = {half};
const TERRAIN_MAX_COORD: f32 = {max_coord};
const BIOME_GRID_RES: u32 = 256u;

const FOOD_COUNT_VAL: u32 = {food_count}u;
const AGENT_COUNT_VAL: u32 = {agent_count}u;

const FOOD_RESPAWN_TIME: f32 = 10.0;
const FOOD_HEIGHT_OFFSET: f32 = 0.35;
const FOOD_RESPAWN_ATTEMPTS: u32 = 64u;

const VISION_FOV_HALF: f32 = 0.7853982;  // PI/4 = 45 degrees (half of 90)
const VISION_MAX_DIST: f32 = 30.0;
const VISION_STEP_SIZE: f32 = 1.2;
const VISION_NUM_STEPS: u32 = 25u;
const VISION_RAYS: u32 = 48u;
const VISION_W: u32 = 8u;
const VISION_H: u32 = 6u;
const VISION_COLOR_COUNT: u32 = 192u;  // 48 rays × 4 RGBA
const VISION_DEPTH_COUNT: u32 = 48u;
const MAX_TOUCH_CONTACTS: u32 = 4u;
const FOOD_RAY_RADIUS_SQ: f32 = 1.0;
const AGENT_RAY_RADIUS_SQ: f32 = 2.25;

const TOUCH_FOOD: u32 = 1u;
const TOUCH_TERRAIN_EDGE: u32 = 2u;
const TOUCH_HAZARD: u32 = 3u;
const TOUCH_AGENT: u32 = 4u;
const TOUCH_FOOD_RANGE: f32 = 3.0;
const TOUCH_AGENT_RANGE: f32 = 5.0;
const TOUCH_EDGE_RANGE: f32 = 3.0;

fn cell_coord(v: f32) -> i32 {{
    return i32(floor(v / CELL_SIZE));
}}
"#,
        gw = gw,
        go = go,
        half = world_size / 2.0,
        inv_step = 1.0 / terrain_step,
        max_idx = terrain_vps - 2,
        max_coord = (terrain_vps - 1) as f32,
    )
}

/// Build the world config uniform data.
pub fn build_world_config(
    config: &xagent_shared::WorldConfig,
    food_count: usize,
    agent_count: usize,
    tick: u64,
    ticks_to_run: u32,
    vision_stride: u32,
    brain_tick_stride: u32,
) -> Vec<f32> {
    let gw = grid_width(config.world_size);
    let go = gw / 2;
    let terrain_step = config.world_size / 128.0;
    let mut wc = vec![0.0f32; WORLD_CONFIG_SIZE];
    wc[WC_WORLD_SIZE] = config.world_size;
    wc[WC_DT] = 1.0 / config.tick_rate;
    wc[WC_ENERGY_DEPLETION] = config.energy_depletion_rate;
    wc[WC_MOVEMENT_COST] = config.movement_energy_cost;
    wc[WC_HAZARD_DAMAGE] = config.hazard_damage_rate;
    wc[WC_INTEGRITY_REGEN] = config.integrity_regen_rate;
    wc[WC_FOOD_ENERGY] = config.food_energy_value;
    wc[WC_FOOD_RADIUS] = 2.5;
    wc[WC_TERRAIN_VPS] = 129.0;
    wc[WC_TERRAIN_INV_STEP] = 1.0 / terrain_step;
    wc[WC_TERRAIN_HALF] = config.world_size / 2.0;
    wc[WC_BIOME_INV_CELL] = 256.0 / config.world_size;
    wc[WC_FOOD_COUNT] = food_count as f32;
    wc[WC_AGENT_COUNT] = agent_count as f32;
    wc[WC_TICK] = tick as f32;
    wc[WC_RNG_SEED] = config.seed as f32;
    wc[WC_WORLD_HALF_BOUND] = config.world_size / 2.0 - 1.0;
    wc[WC_BIOME_GRID_RES] = 256.0;
    wc[WC_GRID_WIDTH] = gw as f32;
    wc[WC_GRID_OFFSET] = go as f32;
    wc[WC_TICKS_TO_RUN] = ticks_to_run as f32;
    wc[WC_VISION_STRIDE] = vision_stride as f32;
    wc[WC_BRAIN_TICK_STRIDE] = brain_tick_stride as f32;
    wc
}

/// Initialize brain_state buffer data for one agent from BrainConfig.
/// Xavier init for encoder, small random for predictor, zero for rest.
pub fn init_brain_state(config: &BrainConfig, rng: &mut impl rand::Rng) -> Vec<f32> {
    init_brain_state_for(config, &BrainLayout::default(), rng)
}

/// Layout-aware version for configurable vision dimensions.
pub fn init_brain_state_for(config: &BrainConfig, layout: &BrainLayout, rng: &mut impl rand::Rng) -> Vec<f32> {
    let fc = layout.feature_count;
    let mut state = vec![0.0_f32; layout.brain_stride];

    // Encoder weights: Xavier init (scale = 1/sqrt(feature_count))
    let scale = 1.0 / (fc as f32).sqrt();
    for i in 0..(fc * DIM) {
        state[i] = (rng.random::<f32>() * 2.0 - 1.0) * scale;
    }

    // Compute dynamic offsets: O_PRED_CTX_WT = fc*DIM + DIM + DIM*DIM
    let o_pred_weights = fc * DIM + DIM;
    let o_pred_ctx_wt = o_pred_weights + DIM * DIM;

    // Predictor weights: small random
    for i in 0..(DIM * DIM) {
        state[o_pred_weights + i] = (rng.random::<f32>() * 2.0 - 1.0) * 0.1;
    }

    // Predictor context weight: 0.15
    state[o_pred_ctx_wt] = 0.15;

    // Fixed deltas from O_PRED_CTX_WT (same regardless of feature_count)
    let delta_hab_atten = O_HAB_ATTEN - O_PRED_CTX_WT;
    let delta_exploration = O_EXPLORATION_RATE - O_PRED_CTX_WT;
    let delta_fatigue_factor = O_FATIGUE_FACTOR - O_PRED_CTX_WT;
    let delta_hab_sens = O_HAB_SENSITIVITY - O_PRED_CTX_WT;
    let delta_hab_curiosity = O_HAB_MAX_CURIOSITY - O_PRED_CTX_WT;
    let delta_fatigue_rec = O_FATIGUE_RECOVERY - O_PRED_CTX_WT;
    let delta_fatigue_floor = O_FATIGUE_FLOOR - O_PRED_CTX_WT;

    // Habituation attenuation: 1.0 (no attenuation initially)
    for i in 0..DIM {
        state[o_pred_ctx_wt + delta_hab_atten + i] = 1.0;
    }

    // Exploration rate: 0.5 (balanced start)
    state[o_pred_ctx_wt + delta_exploration] = 0.5;

    // Fatigue factor: 1.0 (no fatigue)
    state[o_pred_ctx_wt + delta_fatigue_factor] = 1.0;

    // Heritable config values (stored per-agent for GPU access)
    state[o_pred_ctx_wt + delta_hab_sens] = config.habituation_sensitivity;
    state[o_pred_ctx_wt + delta_hab_curiosity] = config.max_curiosity_bonus;
    state[o_pred_ctx_wt + delta_fatigue_rec] = config.fatigue_recovery_sensitivity;
    state[o_pred_ctx_wt + delta_fatigue_floor] = config.fatigue_floor;

    state
}

/// Initialize pattern memory buffer for one agent: all slots empty.
pub fn init_pattern_memory() -> Vec<f32> {
    vec![0.0_f32; PATTERN_STRIDE]
}

/// Initialize action history buffer for one agent: empty ring.
pub fn init_action_history() -> Vec<f32> {
    vec![0.0_f32; HISTORY_STRIDE]
}

/// Build config buffer values from BrainConfig (default layout).
pub fn build_config(config: &BrainConfig) -> Vec<f32> {
    build_config_for(config, &BrainLayout::default())
}

/// Build config buffer values with explicit layout.
pub fn build_config_for(config: &BrainConfig, layout: &BrainLayout) -> Vec<f32> {
    let mut cfg = vec![0.0_f32; CONFIG_SIZE];
    cfg[CFG_REPR_DIM] = DIM as f32;
    cfg[CFG_FEATURE_COUNT] = layout.feature_count as f32;
    cfg[CFG_MEMORY_CAP] = MEMORY_CAP as f32;
    cfg[CFG_RECALL_K] = RECALL_K as f32;
    cfg[CFG_LEARNING_RATE] = config.learning_rate;
    cfg[CFG_DECAY_RATE] = config.decay_rate;
    cfg[CFG_DISTRESS_EXP] = config.distress_exponent;
    cfg[CFG_METABOLIC_RATE] = config.metabolic_rate;
    cfg[CFG_INTEGRITY_SCALE] = config.integrity_scale;
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;
    use xagent_shared::SensoryFrame;

    #[test]
    fn sensory_stride_matches_feature_count() {
        assert_eq!(SENSORY_STRIDE, 267);
        assert_eq!(FEATURE_COUNT, 217);
        assert!(SENSORY_STRIDE >= FEATURE_COUNT);
    }

    #[test]
    fn brain_stride_is_consistent() {
        assert_eq!(BRAIN_STRIDE, O_FATIGUE_FLOOR + 1);
    }

    #[test]
    fn pattern_stride_is_consistent() {
        assert_eq!(PATTERN_STRIDE, O_LAST_STORED_IDX + 1);
    }

    #[test]
    fn history_stride_is_consistent() {
        assert_eq!(HISTORY_STRIDE, O_HIST_LEN + 1);
    }

    #[test]
    fn pack_sensory_frame_fills_buffer() {
        let frame = SensoryFrame::new_blank(VISION_W as u32, VISION_H as u32);
        let mut buf = vec![0.0_f32; SENSORY_STRIDE];
        pack_sensory_frame(&frame, &mut buf);
        assert!(buf.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn init_brain_state_has_correct_length() {
        let config = BrainConfig::default();
        let mut rng = rand::rng();
        let state = init_brain_state(&config, &mut rng);
        assert_eq!(state.len(), BRAIN_STRIDE);
    }

    #[test]
    fn brain_layout_default_matches_constants() {
        let layout = BrainLayout::default();
        assert_eq!(layout.vision_w, 8);
        assert_eq!(layout.vision_h, 6);
        assert_eq!(layout.feature_count, FEATURE_COUNT);
        assert_eq!(layout.sensory_stride, SENSORY_STRIDE);
        assert_eq!(layout.brain_stride, BRAIN_STRIDE);
    }

    #[test]
    fn brain_layout_dynamic_is_consistent() {
        for (w, h) in [(4, 4), (6, 4), (8, 4), (8, 6), (8, 8), (12, 8)] {
            let layout = BrainLayout::new(w, h);
            assert_eq!(layout.vision_w, w);
            assert_eq!(layout.vision_h, h);
            let pixels = (w * h) as usize;
            assert_eq!(layout.vision_color_count, pixels * 4);
            assert_eq!(layout.feature_count, layout.vision_color_count + 25);
            // Verify brain_stride matches the offset chain
            let fc = layout.feature_count;
            let o_pred_ctx_wt = fc * DIM + DIM + DIM * DIM;
            assert_eq!(layout.brain_stride, o_pred_ctx_wt + FIXED_TAIL_SIZE);
        }
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
        assert_eq!(CONFIG_SIZE % 4, 0, "CONFIG_SIZE must be a multiple of 4 (vec4 alignment)");
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
        assert!((packed[CFG_METABOLIC_RATE] - 0.01).abs() < 1e-7,
            "metabolic_rate not packed at index {}: got {}",
            CFG_METABOLIC_RATE, packed[CFG_METABOLIC_RATE]);
        assert!((packed[CFG_INTEGRITY_SCALE] - 0.02).abs() < 1e-7,
            "integrity_scale not packed at index {}: got {}",
            CFG_INTEGRITY_SCALE, packed[CFG_INTEGRITY_SCALE]);
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
                // e.g. "P_ENERGY: u32 = 11u;"
                if let Some((name, tail)) = rest.split_once(':') {
                    if let Some(val_part) = tail.split('=').nth(1) {
                        let val_str = val_part.trim().trim_end_matches(';').trim_end_matches('u').trim();
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
            for d in 0..DIM {
                aos_offsets.insert(O_PAT_STATES + pat * DIM + d);
                soa_offsets.insert(O_PAT_STATES + d * MEMORY_CAP + pat);
            }
        }
        assert_eq!(aos_offsets.len(), MEMORY_CAP * DIM);
        assert_eq!(soa_offsets.len(), MEMORY_CAP * DIM);
        assert_eq!(aos_offsets, soa_offsets, "SoA and AoS must cover identical offsets");
        assert_eq!(*aos_offsets.iter().min().unwrap(), 0);
        assert_eq!(*aos_offsets.iter().max().unwrap(), MEMORY_CAP * DIM - 1);
    }

    #[test]
    fn shader_phys_constants_match_rust() {
        let src = include_str!("shaders/mega/common.wgsl");
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
        assert_eq!(wgsl["P_EXPLORATION_RATE_OUT"], P_EXPLORATION_RATE_OUT as u32);
        assert_eq!(wgsl["P_FATIGUE_FACTOR_OUT"], P_FATIGUE_FACTOR_OUT as u32);
    }

    #[test]
    fn shader_config_constants_match_rust() {
        let src = include_str!("shaders/mega/common.wgsl");
        let wgsl = parse_wgsl_u32_constants(src);

        assert_eq!(wgsl["CFG_LEARNING_RATE"], CFG_LEARNING_RATE as u32);
        assert_eq!(wgsl["CFG_DECAY_RATE"], CFG_DECAY_RATE as u32);
        assert_eq!(wgsl["CFG_DISTRESS_EXP"], CFG_DISTRESS_EXP as u32);
        assert_eq!(wgsl["CFG_METABOLIC_RATE"], CFG_METABOLIC_RATE as u32);
        assert_eq!(wgsl["CFG_INTEGRITY_SCALE"], CFG_INTEGRITY_SCALE as u32);
    }

    #[test]
    fn shader_brain_config_array_size_matches_config_size() {
        let src = include_str!("shaders/mega/common.wgsl");
        // Find: brain_config: array<vec4<f32>, N>
        let needle = "brain_config:";
        let line = src.lines().find(|l| l.contains(needle))
            .expect("brain_config binding not found in shader");
        // Extract N from array<vec4<f32>, N>
        let after_comma = line.split("array<vec4<f32>,").nth(1)
            .expect("unexpected brain_config type format");
        let n_str = after_comma.trim().trim_end_matches(';').trim_end_matches('>')
            .trim();
        let shader_vec4_count: usize = n_str.parse()
            .expect("failed to parse brain_config array size");
        assert_eq!(shader_vec4_count * 4, CONFIG_SIZE,
            "shader brain_config has {} vec4s (={} floats) but CONFIG_SIZE={}",
            shader_vec4_count, shader_vec4_count * 4, CONFIG_SIZE);
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
                    if v > best_val { best_val = v; best_idx = j; }
                }
                if best_val <= -1.5 { break; }
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
                        let should_swap = (descending && vals[i] < vals[j])
                            || (!descending && vals[i] > vals[j]);
                        if should_swap {
                            vals.swap(i, j);
                            idxs.swap(i, j);
                        }
                    }
                }
            }

            let mut result = Vec::new();
            for i in 0..k {
                if vals[i] <= -1.5 { break; }
                result.push(idxs[i] as usize);
            }
            result
        }

        let mut sims = vec![-2.0_f32; 128];
        sims[5] = 0.95; sims[10] = 0.80; sims[0] = 0.75; sims[127] = 0.70;
        sims[64] = 0.65; sims[33] = 0.60; sims[99] = 0.55; sims[17] = 0.50;
        sims[42] = 0.45; sims[88] = 0.40; sims[3] = 0.35; sims[111] = 0.30;
        sims[50] = 0.25; sims[77] = 0.20; sims[22] = 0.15; sims[61] = 0.10;
        sims[100] = 0.05;

        let greedy = greedy_top_k(&sims, 16);
        let bitonic = bitonic_top_k(&sims, 16);

        assert_eq!(greedy.len(), bitonic.len(),
            "top-K count mismatch: greedy={}, bitonic={}", greedy.len(), bitonic.len());
        for idx in &greedy {
            assert!(bitonic.contains(idx),
                "greedy selected index {} (sim={}) but bitonic did not. bitonic={:?}",
                idx, sims[*idx], bitonic);
        }
    }

    #[test]
    fn phys_stride_covers_all_fields() {
        // Highest P_* offset + 1 must equal PHYS_STRIDE
        let max_field = *[
            P_POS_X, P_POS_Y, P_POS_Z,
            P_VEL_X, P_VEL_Y, P_VEL_Z,
            P_FACING_X, P_FACING_Y, P_FACING_Z,
            P_YAW, P_ANGULAR_VEL,
            P_ENERGY, P_MAX_ENERGY, P_INTEGRITY, P_MAX_INTEGRITY,
            P_PREV_ENERGY, P_PREV_INTEGRITY,
            P_ALIVE, P_FOOD_COUNT, P_TICKS_ALIVE, P_DIED_FLAG,
            P_MEMORY_CAP, P_PROCESSING_SLOTS, P_DEATH_COUNT,
            P_PREDICTION_ERROR, P_EXPLORATION_RATE_OUT, P_FATIGUE_FACTOR_OUT,
        ].iter().max().unwrap();
        assert_eq!(PHYS_STRIDE, max_field + 1,
            "PHYS_STRIDE ({}) should be highest field offset ({}) + 1",
            PHYS_STRIDE, max_field);
    }

    #[test]
    fn shader_has_16_bindings() {
        let src = include_str!("shaders/mega/common.wgsl");
        let binding_count = src.lines()
            .filter(|l| l.trim().starts_with("@group(0) @binding("))
            .count();
        assert_eq!(binding_count, 16,
            "Expected 16 bindings (0-14 existing + 15 dispatch_args), found {}", binding_count);
    }
}
