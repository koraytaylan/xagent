//! Sensory input types delivered to the brain each tick.
//!
//! The brain receives a [`SensoryFrame`] as an opaque stream of signals.
//! It has no built-in understanding of what these values mean — it must
//! discover their significance through experience and prediction error.

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Distance over which a food item's odour concentration falls by a factor
/// of e. Each food item contributes `exp(−d / SCENT_DECAY_LENGTH)` to the
/// concentration at a nostril `d` units away, so odour reaches well past the
/// distance at which a single item is still resolvable by eye. Mirrors
/// `SCENT_DECAY_LENGTH` in the brain crate's `common.wgsl`.
pub const SCENT_DECAY_LENGTH: f32 = 10.0;
/// Food farther than this from a nostril adds nothing: three decay lengths,
/// beyond which an item contributes under 5% of its value at the nose.
/// Mirrors `SCENT_RANGE` in `common.wgsl`.
pub const SCENT_RANGE: f32 = 3.0 * SCENT_DECAY_LENGTH;
/// The nostrils sit this far ahead of the body's centre, along the facing
/// direction. Mirrors `NOSTRIL_FORWARD_OFFSET` in `common.wgsl`.
pub const NOSTRIL_FORWARD_OFFSET: f32 = 0.5;
/// Each nostril sits this far to the side of the facing line, so the two
/// sample the odour field at points 2 units apart. Mirrors
/// `NOSTRIL_SIDE_OFFSET` in `common.wgsl`.
pub const NOSTRIL_SIDE_OFFSET: f32 = 1.0;

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

    // -- Smell --
    /// Perceived food odour at the left and right nostrils, each in
    /// `[0, 1)`: `1 − exp(−smell_strength · C)`, where `C` is the odour
    /// concentration at the nostril (see [`SCENT_DECAY_LENGTH`]).
    pub scent: [f32; 2],

    /// Current simulation tick.
    pub tick: u64,
}

impl SensoryFrame {
    /// Create a blank sensory frame with pre-allocated vision buffers.
    pub fn new_blank(vision_width: u32, vision_height: u32) -> Self {
        Self {
            vision: VisualField::new(vision_width, vision_height),
            velocity: Vec3::ZERO,
            facing: Vec3::Z,
            angular_velocity: 0.0,
            energy_signal: 0.0,
            integrity_signal: 0.0,
            energy_delta: 0.0,
            integrity_delta: 0.0,
            touch_contacts: Vec::with_capacity(8),
            scent: [0.0; 2],
            tick: 0,
        }
    }
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

    /// Reset to blank state without reallocating. Colors → black, depths → far.
    pub fn clear(&mut self) {
        self.color.fill(0.0);
        self.depth.fill(1.0);
    }
}
