//! The cognitive architecture for xagent — a GPU-resident predictive processing brain.
//!
//! All brain computation runs on GPU via 7 WGSL compute shaders.
//! No behavior is hardcoded. Fear, curiosity, habit, and attention emerge
//! from the interaction of capacity constraints, prediction error, and
//! homeostatic pressure.

pub mod buffers;
pub mod gpu_brain;
pub mod gpu_physics;

pub use buffers::AgentBrainState;
pub use gpu_brain::GpuBrain;

/// Padé approximant for tanh, accurate to ~1e-4 for |x| < 4.5.
#[inline(always)]
pub fn fast_tanh(x: f32) -> f32 {
    if x.abs() > 4.5 { return x.signum(); }
    let x2 = x * x;
    x * (27.0 + x2) / (27.0 + 9.0 * x2)
}

/// Per-tick telemetry snapshot for external observation.
///
/// Populated with stub/zero values when the GPU brain doesn't expose
/// per-tick readback; kept for recording and UI compatibility.
#[derive(Clone, Debug, Default)]
pub struct BrainTelemetry {
    /// Current tick number.
    pub tick: u64,
    /// Scalar prediction error this tick (0 = perfect, higher = more surprised).
    pub prediction_error: f32,
    /// Memory utilization [0.0, 1.0].
    pub memory_utilization: f32,
    /// Number of active patterns in memory.
    pub memory_active_count: usize,
    /// Action entropy (high = uniform/exploratory, low = committed).
    pub action_entropy: f32,
    /// Current exploration rate.
    pub exploration_rate: f32,
    /// Homeostatic gradient (positive = improving, negative = worsening).
    pub homeostatic_gradient: f32,
    /// Homeostatic urgency (higher = closer to critical levels).
    pub homeostatic_urgency: f32,
    /// Recall budget allocated this tick.
    pub recall_budget: usize,
    /// Average prediction error over a recent window.
    pub avg_prediction_error: f32,
    /// Fraction of actions that were exploitative (informed) [0.0, 1.0].
    pub exploitation_ratio: f32,
    /// Composite decision quality score [0.0, 1.0].
    pub decision_quality: f32,
    /// Sensory habituation: mean attenuation across encoded dimensions [0.1, 1.0].
    pub mean_attenuation: f32,
    /// Curiosity bonus from sensory monotony.
    pub curiosity_bonus: f32,
    /// Motor fatigue factor [fatigue_floor, 1.0]. Low = fatigued.
    pub fatigue_factor: f32,
    /// Recent motor output variance (higher = more diverse).
    pub motor_variance: f32,
}

impl BrainTelemetry {
    /// Behavior phase label based on exploitation ratio.
    pub fn behavior_phase(&self) -> &'static str {
        let stability = 1.0 - self.homeostatic_urgency.clamp(0.0, 1.0);
        let score = self.exploitation_ratio
            * (1.0 - self.prediction_error.clamp(0.0, 1.0))
            * stability;
        if score < 0.02 {
            "RANDOM"
        } else if score < 0.08 {
            "EXPLORING"
        } else if score < 0.20 {
            "LEARNING"
        } else {
            "ADAPTED"
        }
    }
}
