//! The per-tick agent record used by supervisor tick-window replay.

/// Per-tick per-agent state snapshot for replay.
#[derive(Clone, Debug)]
pub struct TickRecord {
    pub position: [f32; 3],
    pub yaw: f32,
    pub alive: bool,
    pub energy: f32,
    pub integrity: f32,
    pub motor_forward: f32,
    pub motor_turn: f32,
    pub exploration_rate: f32,
    pub prediction_error: f32,
    pub gradient: f32,
    pub raw_gradient: f32,
    pub urgency: f32,
    pub credit_magnitude: f32,
    pub patterns_recalled: u16,
    pub phase: u8, // 0=RANDOM, 1=EXPLORING, 2=LEARNING, 3=ADAPTED
    /// Sensory habituation: mean attenuation [0.1, 1.0].
    pub mean_attenuation: f32,
    /// Curiosity bonus from sensory monotony.
    pub curiosity_bonus: f32,
    /// Motor fatigue factor [fatigue_floor, 1.0]. Low = fatigued.
    pub fatigue_factor: f32,
    /// Position-based staleness [0.0, 1.0]. High = stuck in place.
    pub staleness: f32,
    /// Vision color data (vision_width * vision_height * 4 f32 values). Only stored at keyframes
    /// to save memory.
    pub vision_color: Option<Vec<f32>>,
}
