//! The 3D simulation sandbox for xagent cognitive agents.
//!
//! This crate provides the physical world (procedural terrain, biomes, food, hazards),
//! the wgpu-based renderer, physics simulation, sensory extraction, and the main
//! simulation loop that connects agents to their environment.

pub mod agent;
pub mod bench;
pub mod food_kind_metrics;
pub mod free_life;
pub mod governor;
pub mod headless;
pub mod momentum;
pub mod overlay;
pub mod physics;
pub mod pooling_study;
pub mod renderer;
pub mod replay;
pub mod run_summary;
pub mod supervisor;
pub mod supervisor_run;
pub mod supervisor_store;
pub mod ui;
pub mod world;
