//! The core trait for swappable cognitive architectures.
//!
//! Any struct implementing [`CognitiveArchitecture`] can serve as the agent's
//! "brain". This enables experimentation with different cognitive models while
//! keeping the sandbox and shared types unchanged.

use crate::{MotorCommand, SensoryFrame};

/// Trait for cognitive architectures that process sensory input and produce motor output.
pub trait CognitiveArchitecture {
    /// Process one tick of sensory input and produce a motor command.
    fn tick(&mut self, frame: &SensoryFrame) -> MotorCommand;
}
