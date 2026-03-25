//! Noise-driven biome classification for world regions.
//!
//! Biome type is determined by Perlin noise at frequency 0.03. The noise value
//! is thresholded: >0.2 = FoodRich, <-0.2 = Danger, middle = Barren.
//! This creates natural-looking region boundaries with smooth transitions.

use noise::{NoiseFn, Perlin};

/// Biome classification for world regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiomeType {
    /// Lush area where food spawns — green tones.
    FoodRich,
    /// Sparse area — tan/brown tones.
    Barren,
    /// Hazardous area that damages agents — red tones.
    Danger,
}

/// Noise-driven biome map that can be queried at any (x, z) position.
pub struct BiomeMap {
    perlin: Perlin,
    frequency: f64,
}

impl BiomeMap {
    /// Create a biome map from a seed. Uses a different seed than terrain to ensure
    /// biome boundaries don't align with terrain features.
    pub fn new(seed: u32) -> Self {
        Self {
            perlin: Perlin::new(seed),
            frequency: 0.03,
        }
    }

    /// Return the biome type at world-space (x, z).
    pub fn biome_at(&self, x: f32, z: f32) -> BiomeType {
        let val = self.perlin.get([x as f64 * self.frequency, z as f64 * self.frequency]);
        if val > 0.2 {
            BiomeType::FoodRich
        } else if val < -0.2 {
            BiomeType::Danger
        } else {
            BiomeType::Barren
        }
    }
}
