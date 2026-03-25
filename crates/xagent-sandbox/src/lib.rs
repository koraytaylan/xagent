//! The 3D simulation sandbox for xagent cognitive agents.
//!
//! This crate provides the physical world (procedural terrain, biomes, food, hazards),
//! the wgpu-based renderer, physics simulation, sensory extraction, and the main
//! simulation loop that connects agents to their environment.

pub mod agent;
pub mod physics;
pub mod recording;
pub mod renderer;
pub mod ui;
pub mod world;
