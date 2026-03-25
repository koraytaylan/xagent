//! Sensory input types delivered to the brain each tick.
//!
//! The brain receives a [`SensoryFrame`] as an opaque stream of signals.
//! It has no built-in understanding of what these values mean — it must
//! discover their significance through experience and prediction error.

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Raw visual data from the agent's point of view.
/// A low-resolution grid of color+depth samples within the agent's field of view.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisualField {
    /// Width of the visual grid.
    pub width: u32,
    /// Height of the visual grid.
    pub height: u32,
    /// Flattened RGBA color values, row-major, length = width * height * 4.
    pub color: Vec<f32>,
    /// Depth values per pixel, length = width * height.
    pub depth: Vec<f32>,
}

/// Contact information from touch sense.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TouchContact {
    /// Direction of contact relative to agent body.
    pub direction: Vec3,
    /// Intensity of contact (0.0 = none, 1.0 = hard impact).
    pub intensity: f32,
    /// Surface type tag (terrain, food, hazard, agent, etc.).
    pub surface_tag: u32,
}

/// A single frame of sensory input delivered to the brain each tick.
///
/// Contains everything the agent can perceive: vision, body awareness,
/// internal physiological signals, and touch. The brain receives this
/// as an opaque stream — it must learn what each signal means.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensoryFrame {
    /// What the agent sees from its viewpoint.
    pub vision: VisualField,

    // -- Proprioception (body awareness) --
    /// Agent's velocity in world space.
    pub velocity: Vec3,
    /// Agent's forward direction.
    pub facing: Vec3,
    /// Agent's angular velocity (how fast it's turning).
    pub angular_velocity: f32,

    // -- Interoception (internal signals) --
    /// Energy level signal, normalized to [0.0, 1.0].
    pub energy_signal: f32,
    /// Physical integrity signal, normalized to [0.0, 1.0].
    pub integrity_signal: f32,
    /// Rate of energy change (positive = gaining, negative = losing).
    pub energy_delta: f32,
    /// Rate of integrity change.
    pub integrity_delta: f32,

    // -- Touch --
    /// Active touch contacts this tick.
    pub touch_contacts: Vec<TouchContact>,

    /// Current simulation tick.
    pub tick: u64,
}

impl VisualField {
    /// Create a blank visual field with the given resolution.
    /// Colors initialize to black (0.0), depths to far plane (1.0).
    pub fn new(width: u32, height: u32) -> Self {
        let pixel_count = (width * height) as usize;
        Self {
            width,
            height,
            color: vec![0.0; pixel_count * 4],
            depth: vec![1.0; pixel_count],
        }
    }
}
