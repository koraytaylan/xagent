//! Physical body state types for agents in the simulation.
//!
//! The body is the interface between the cognitive architecture and the
//! physical world. Internal physiological variables (energy, integrity)
//! produce interoceptive signals that the brain receives but must learn
//! to interpret.

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// The physical state of an agent body in the world.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyState {
    /// World-space position.
    pub position: Vec3,
    /// Forward-facing direction (unit vector).
    pub facing: Vec3,
    /// Current velocity.
    pub velocity: Vec3,
    /// Internal physiological state.
    pub internal: InternalState,
    /// Whether the agent is alive.
    pub alive: bool,
}

/// Internal physiological variables.
///
/// These produce interoceptive signals in `SensoryFrame`, but the agent
/// has no "understanding" of them — it must discover their significance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InternalState {
    /// Energy, range [0.0, max_energy]. Depletes with time and movement.
    pub energy: f32,
    /// Maximum energy capacity.
    pub max_energy: f32,
    /// Physical integrity, range [0.0, max_integrity]. Damaged by hazards.
    pub integrity: f32,
    /// Maximum integrity.
    pub max_integrity: f32,
}

impl InternalState {
    /// Create a new internal state at full health (energy and integrity at max).
    pub fn new(max_energy: f32, max_integrity: f32) -> Self {
        Self {
            energy: max_energy,
            max_energy,
            integrity: max_integrity,
            max_integrity,
        }
    }

    /// Normalized energy signal [0.0, 1.0].
    pub fn energy_signal(&self) -> f32 {
        self.energy / self.max_energy
    }

    /// Normalized integrity signal [0.0, 1.0].
    pub fn integrity_signal(&self) -> f32 {
        self.integrity / self.max_integrity
    }

    /// Returns true if the agent should be dead.
    pub fn is_dead(&self) -> bool {
        self.energy <= 0.0 || self.integrity <= 0.0
    }
}

impl BodyState {
    /// Create a new body at the given position, facing +Z, at rest.
    pub fn new(position: Vec3, internal: InternalState) -> Self {
        Self {
            position,
            facing: Vec3::Z,
            velocity: Vec3::ZERO,
            internal,
            alive: true,
        }
    }
}
