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
    /// Exponent for the homeostatic distress curve. Higher = calm longer, panic harder.
    /// Heritable: mutated during breeding, clamped to [1.5, 5.0]. Default 2.0.
    #[serde(default = "default_distress_exponent")]
    pub distress_exponent: f32,
    /// Scales per-dimension variance into attenuation range. Higher = faster boredom.
    /// Heritable: mutated during breeding, clamped to [5.0, 50.0]. Default 20.0.
    #[serde(default = "default_habituation_sensitivity")]
    pub habituation_sensitivity: f32,
    /// Maximum curiosity bonus from sensory monotony. Higher = stronger exploration drive.
    /// Heritable: mutated during breeding, clamped to [0.1, 1.0]. Default 0.6.
    #[serde(default = "default_max_curiosity_bonus")]
    pub max_curiosity_bonus: f32,
    /// Scales motor variance into fatigue relief. Higher = easier recovery from fatigue.
    /// Heritable: mutated during breeding, clamped to [2.0, 20.0]. Default 8.0.
    #[serde(default = "default_fatigue_recovery_sensitivity")]
    pub fatigue_recovery_sensitivity: f32,
    /// Minimum motor output under fatigue. Lower = harsher dampening.
    /// Heritable: mutated during breeding, clamped to [0.05, 0.4]. Default 0.1.
    #[serde(default = "default_fatigue_floor")]
    pub fatigue_floor: f32,
    /// Number of vision rays (W × H). Affects sensory buffer size and feature count.
    /// Lower = faster vision dispatch. Default 48 (8×6).
    #[serde(default = "default_vision_rays")]
    pub vision_rays: u32,
    /// Physics ticks per brain+vision cycle. Higher = faster but less responsive.
    /// Default 4.
    #[serde(default = "default_brain_tick_stride")]
    pub brain_tick_stride: u32,
    /// Brain cycles between global passes (grid rebuild, collisions, vision).
    /// Higher = more brain throughput, less frequent vision updates.
    /// Default 10.
    #[serde(default = "default_vision_stride")]
    pub vision_stride: u32,
    /// Multiplier for all energy costs (metabolic + movement). Default 1.0.
    /// Lower = agents survive longer. Higher = harsher energy pressure.
    #[serde(default = "default_metabolic_rate")]
    pub metabolic_rate: f32,
    /// Multiplier for integrity damage and regen. Default 1.0.
    /// Lower = agents take less damage. Higher = hazard zones are deadlier.
    #[serde(default = "default_integrity_scale")]
    pub integrity_scale: f32,
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

fn default_distress_exponent() -> f32 {
    2.0
}

fn default_habituation_sensitivity() -> f32 {
    20.0
}

fn default_max_curiosity_bonus() -> f32 {
    0.6
}

fn default_fatigue_recovery_sensitivity() -> f32 {
    8.0
}

fn default_fatigue_floor() -> f32 {
    0.1
}

fn default_seed() -> u64 {
    42
}

fn default_vision_rays() -> u32 {
    48
}

fn default_brain_tick_stride() -> u32 {
    4
}

fn default_vision_stride() -> u32 {
    10
}

fn default_metabolic_rate() -> f32 {
    1.0
}

fn default_integrity_scale() -> f32 {
    1.0
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
    #[serde(default)]
    pub governor: GovernorConfig,
}

/// Configuration for the evolution governor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GovernorConfig {
    /// Number of agents per generation.
    pub population_size: usize,
    /// Simulation ticks per generation before evaluation.
    pub tick_budget: u64,
    /// Number of top agents whose configs survive to the next generation.
    pub elitism_count: usize,
    /// Maximum number of generations to run (0 = unlimited).
    pub max_generations: u64,
    /// Consecutive generations of fitness regression before backtracking.
    pub patience: u32,
    /// Base mutation strength (0.1 = ±10%). Scales up with failed attempts.
    #[serde(default = "default_mutation_strength")]
    pub mutation_strength: f32,
    /// How many times each unique config is evaluated per generation (noise reduction).
    #[serde(default = "default_eval_repeats")]
    pub eval_repeats: usize,
    /// Number of independent evolutionary lineages (island model).
    #[serde(default = "default_num_islands")]
    pub num_islands: usize,
    /// Generations between best-config migration across islands.
    #[serde(default = "default_migration_interval")]
    pub migration_interval: u32,
    /// Decay factor for per-island mutation momentum (0.0–1.0).
    /// Higher = longer memory of winning mutation directions.
    #[serde(default = "default_momentum_decay")]
    pub momentum_decay: f32,
}

fn default_mutation_strength() -> f32 {
    0.1
}

fn default_eval_repeats() -> usize {
    2
}

fn default_num_islands() -> usize {
    3
}

fn default_migration_interval() -> u32 {
    5
}

fn default_momentum_decay() -> f32 {
    0.9
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            population_size: 10,
            tick_budget: 50_000,
            elitism_count: 3,
            max_generations: 0,
            patience: 5,
            mutation_strength: 0.1,
            eval_repeats: 2,
            num_islands: 3,
            migration_interval: 5,
            momentum_decay: 0.9,
        }
    }
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
            distress_exponent: 2.0,
            habituation_sensitivity: 20.0,
            max_curiosity_bonus: 0.6,
            fatigue_recovery_sensitivity: 8.0,
            fatigue_floor: 0.1,
            vision_rays: 48,
            brain_tick_stride: 4,
            vision_stride: 10,
            metabolic_rate: 1.0,
            integrity_scale: 1.0,
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
            distress_exponent: 2.0,
            habituation_sensitivity: 20.0,
            max_curiosity_bonus: 0.6,
            fatigue_recovery_sensitivity: 8.0,
            fatigue_floor: 0.1,
            vision_rays: 24,
            brain_tick_stride: 4,
            vision_stride: 10,
            metabolic_rate: 1.0,
            integrity_scale: 1.0,
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
            distress_exponent: 2.0,
            habituation_sensitivity: 20.0,
            max_curiosity_bonus: 0.6,
            fatigue_recovery_sensitivity: 8.0,
            fatigue_floor: 0.1,
            vision_rays: 96,
            brain_tick_stride: 4,
            vision_stride: 10,
            metabolic_rate: 1.0,
            integrity_scale: 1.0,
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
