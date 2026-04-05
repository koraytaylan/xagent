//! The cognitive architecture for xagent — a predictive processing brain.
//!
//! This crate implements the complete cognitive loop: encode sensory input,
//! recall relevant patterns from memory, predict the next state, compare
//! prediction to reality, learn from the error, and select an action.
//!
//! No behavior is hardcoded. Fear, curiosity, habit, and attention emerge
//! from the interaction of capacity constraints, prediction error, and
//! homeostatic pressure.

pub mod brain;
pub mod buffers;
pub mod capacity;
pub mod encoder;
pub mod homeostasis;
pub mod memory;
pub mod predictor;
pub mod action;
pub mod habituation;
pub mod motor_fatigue;

/// Padé approximant for tanh, accurate to ~1e-4 for |x| < 4.5.
#[inline(always)]
pub(crate) fn fast_tanh(x: f32) -> f32 {
    if x.abs() > 4.5 { return x.signum(); }
    let x2 = x * x;
    x * (27.0 + x2) / (27.0 + 9.0 * x2)
}

pub use brain::{Brain, BrainTelemetry, DecisionSnapshot, LearnedState};
pub use habituation::SensoryHabituation;
pub use memory::RecalledPattern;
pub use motor_fatigue::MotorFatigue;
