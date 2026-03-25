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
pub mod capacity;
pub mod encoder;
pub mod homeostasis;
pub mod memory;
pub mod predictor;
pub mod action;

pub use brain::{Brain, BrainTelemetry};
