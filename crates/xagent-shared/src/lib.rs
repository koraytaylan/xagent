//! Shared types, traits, and configuration for the xagent cognitive agent platform.
//!
//! This crate defines the interface contract between the brain and sandbox crates.
//! It contains no logic — only data structures, the [`CognitiveArchitecture`] trait,
//! and configuration with presets.

pub mod body;
pub mod config;
pub mod motor;
pub mod sensory;
pub mod traits;

pub use body::{BodyState, InternalState};
pub use config::{AgentDescriptor, BrainConfig, FullConfig, GovernorConfig, WorldConfig};
pub use motor::{MotorAction, MotorCommand};
pub use sensory::{SensoryFrame, TouchContact, VisualField};
pub use traits::CognitiveArchitecture;
