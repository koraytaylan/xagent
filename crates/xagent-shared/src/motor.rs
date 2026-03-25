//! Motor output types emitted by the brain each tick.
//!
//! The brain expresses its intentions through continuous locomotion controls
//! and optional discrete actions. The sandbox physics engine interprets these.

use serde::{Deserialize, Serialize};

/// A motor command emitted by the brain each tick.
///
/// Contains continuous-valued controls for locomotion and
/// discrete action triggers for interactions.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MotorCommand {
    /// Forward/backward thrust, range [-1.0, 1.0].
    pub forward: f32,
    /// Left/right strafe, range [-1.0, 1.0].
    pub strafe: f32,
    /// Turn rate, range [-1.0, 1.0] (negative = left, positive = right).
    pub turn: f32,
    /// Discrete action to perform this tick, if any.
    pub action: Option<MotorAction>,
}

/// Discrete actions the agent can perform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MotorAction {
    /// Attempt to consume whatever is in front of the agent.
    Consume,
    /// Push/interact with object in front.
    Push,
    /// Jump.
    Jump,
}

impl MotorCommand {
    /// A command that does nothing.
    pub fn idle() -> Self {
        Self::default()
    }
}
