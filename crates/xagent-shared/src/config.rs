//! Configuration types and presets for brain capacity and world parameters.
//!
//! All parameters that affect the cognitive architecture's capacity constraints
//! or the world's difficulty are defined here, with named presets for common
//! experimental configurations.

use serde::{Deserialize, Serialize};

/// Configuration for the brain's capacity constraints.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainConfig {
    /// Maximum number of patterns the memory can hold.
    pub memory_capacity: usize,
    /// Maximum number of patterns that can be recalled/compared per tick.
    pub processing_slots: usize,
    /// Resolution of the visual encoder (downsampled from raw vision).
    pub visual_encoding_size: usize,
    /// Length of the internal representation vector.
    pub representation_dim: usize,
    /// Base learning rate for association updates.
    pub learning_rate: f32,
    /// Decay rate for unreinforced patterns per tick.
    pub decay_rate: f32,
}

/// Configuration for the world simulation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldConfig {
    /// World size in units (square terrain side length).
    pub world_size: f32,
    /// Energy depletion rate per tick (base metabolic cost).
    pub energy_depletion_rate: f32,
    /// Energy cost per unit of movement.
    pub movement_energy_cost: f32,
    /// Damage per tick in hazard zones.
    pub hazard_damage_rate: f32,
    /// Integrity regeneration per tick (when energy > 50%).
    pub integrity_regen_rate: f32,
    /// Energy restored per food item consumed.
    pub food_energy_value: f32,
    /// Density of food items in food-rich biomes (items per unit²).
    pub food_density: f32,
    /// Simulation ticks per second.
    pub tick_rate: f32,
    /// Random seed for world generation.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_seed() -> u64 {
    42
}

/// Describes an agent to be spawned into the world.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentDescriptor {
    /// Human-readable name for this agent.
    pub name: String,
    /// Brain configuration.
    pub brain: BrainConfig,
    /// Maximum energy.
    pub max_energy: f32,
    /// Maximum integrity.
    pub max_integrity: f32,
    /// Visual field resolution (width x height).
    pub visual_resolution: (u32, u32),
    /// Field of view in degrees.
    pub fov_degrees: f32,
}

/// Combined configuration for brain + world, used for JSON serialization.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FullConfig {
    #[serde(default)]
    pub brain: BrainConfig,
    #[serde(default)]
    pub world: WorldConfig,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            memory_capacity: 128,
            processing_slots: 16,
            visual_encoding_size: 64,
            representation_dim: 32,
            learning_rate: 0.05,
            decay_rate: 0.001,
        }
    }
}

impl BrainConfig {
    /// Minimal capacity — interesting for observing constraints.
    pub fn tiny() -> Self {
        Self {
            memory_capacity: 24,
            processing_slots: 8,
            visual_encoding_size: 32,
            representation_dim: 16,
            learning_rate: 0.08,
            decay_rate: 0.002,
        }
    }

    /// More capacity — slower emergence but richer behavior.
    pub fn large() -> Self {
        Self {
            memory_capacity: 512,
            processing_slots: 32,
            visual_encoding_size: 128,
            representation_dim: 64,
            learning_rate: 0.03,
            decay_rate: 0.0005,
        }
    }
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            world_size: 256.0,
            energy_depletion_rate: 0.01,
            movement_energy_cost: 0.005,
            hazard_damage_rate: 1.0,
            integrity_regen_rate: 0.005,
            food_energy_value: 20.0,
            food_density: 0.005,
            tick_rate: 30.0,
            seed: 42,
        }
    }
}

impl WorldConfig {
    /// Lots of food, slow energy drain, mild hazards.
    pub fn easy() -> Self {
        Self {
            energy_depletion_rate: 0.005,
            movement_energy_cost: 0.002,
            hazard_damage_rate: 0.5,
            food_density: 0.005,
            food_energy_value: 30.0,
            ..Self::default()
        }
    }

    /// Scarce food, fast energy drain, deadly hazards.
    pub fn hard() -> Self {
        Self {
            energy_depletion_rate: 0.02,
            movement_energy_cost: 0.01,
            hazard_damage_rate: 2.0,
            food_density: 0.001,
            food_energy_value: 15.0,
            ..Self::default()
        }
    }
}

impl Default for AgentDescriptor {
    fn default() -> Self {
        Self {
            name: "Agent-0".into(),
            brain: BrainConfig::default(),
            max_energy: 100.0,
            max_integrity: 100.0,
            visual_resolution: (16, 12),
            fov_degrees: 90.0,
        }
    }
}
