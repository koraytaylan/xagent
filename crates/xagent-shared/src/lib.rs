//! Shared types and configuration for the xagent cognitive agent platform.
//!
//! This crate defines the interface contract between the brain and sandbox crates.
//! It contains no logic — only data structures and configuration with presets.

pub mod body;
pub mod config;
pub mod motor;
pub mod sensory;

pub use body::{BodyState, InternalState};
pub use config::{
    AgentDescriptor, BrainConfig, FullConfig, GovernorConfig, WorldConfig, DOG_SURROUND_RATIO_MAX,
    DOG_SURROUND_RATIO_MIN, GABOR_ASPECT_RATIO_MAX, GABOR_ASPECT_RATIO_MIN, GABOR_WAVELENGTH_MAX,
    GABOR_WAVELENGTH_MIN, HORIZONTAL_FOV_MAX, HORIZONTAL_FOV_MIN, INSTINCT_DANGER_STRENGTH_MAX,
    INSTINCT_DANGER_STRENGTH_MIN, INSTINCT_FOOD_STRENGTH_MAX, INSTINCT_FOOD_STRENGTH_MIN,
    ORIENTATION_OFFSET_PERIOD, SMELL_STRENGTH_MAX, SMELL_STRENGTH_MIN, VERTICAL_FOV_MAX,
    VERTICAL_FOV_MIN, VISION_PLASTICITY_MAX, VISION_PLASTICITY_MIN,
};
pub use motor::{MotorAction, MotorCommand};
pub use sensory::{
    SensoryFrame, TouchContact, VisualField, NOSTRIL_FORWARD_OFFSET, NOSTRIL_SIDE_OFFSET,
    SCENT_DECAY_LENGTH, SCENT_RANGE,
};
