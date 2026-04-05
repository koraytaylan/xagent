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
pub const CONFIG_SIZE: usize = 7;

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
        Self {
            brain_state: vec![0.0; BRAIN_STRIDE],
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

/// Initialize brain_state buffer data for one agent from BrainConfig.
/// Xavier init for encoder, small random for predictor, zero for rest.
pub fn init_brain_state(config: &BrainConfig, rng: &mut impl rand::Rng) -> Vec<f32> {
    let mut state = vec![0.0_f32; BRAIN_STRIDE];

    // Encoder weights: Xavier init (scale = 1/sqrt(feature_count))
    let scale = 1.0 / (FEATURE_COUNT as f32).sqrt();
    for i in 0..(FEATURE_COUNT * DIM) {
        state[O_ENC_WEIGHTS + i] = (rng.random::<f32>() * 2.0 - 1.0) * scale;
    }

    // Encoder biases: zero (already)

    // Predictor weights: small random
    for i in 0..(DIM * DIM) {
        state[O_PRED_WEIGHTS + i] = (rng.random::<f32>() * 2.0 - 1.0) * 0.1;
    }

    // Predictor context weight: 0.15
    state[O_PRED_CTX_WT] = 0.15;

    // Habituation attenuation: 1.0 (no attenuation initially)
    for i in 0..DIM {
        state[O_HAB_ATTEN + i] = 1.0;
    }

    // Exploration rate: 0.5 (balanced start)
    state[O_EXPLORATION_RATE] = 0.5;

    // Fatigue factor: 1.0 (no fatigue)
    state[O_FATIGUE_FACTOR] = 1.0;

    // Heritable config values (stored per-agent for GPU access)
    state[O_HAB_SENSITIVITY] = config.habituation_sensitivity;
    state[O_HAB_MAX_CURIOSITY] = config.max_curiosity_bonus;
    state[O_FATIGUE_RECOVERY] = config.fatigue_recovery_sensitivity;
    state[O_FATIGUE_FLOOR] = config.fatigue_floor;

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

/// Build config buffer values from BrainConfig.
pub fn build_config(config: &BrainConfig) -> Vec<f32> {
    let mut cfg = vec![0.0_f32; CONFIG_SIZE];
    cfg[CFG_REPR_DIM] = DIM as f32;
    cfg[CFG_FEATURE_COUNT] = FEATURE_COUNT as f32;
    cfg[CFG_MEMORY_CAP] = MEMORY_CAP as f32;
    cfg[CFG_RECALL_K] = RECALL_K as f32;
    cfg[CFG_LEARNING_RATE] = config.learning_rate;
    cfg[CFG_DECAY_RATE] = config.decay_rate;
    cfg[CFG_DISTRESS_EXP] = config.distress_exponent;
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
}
