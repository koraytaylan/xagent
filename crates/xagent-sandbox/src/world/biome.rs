//! Noise-driven biome classification for world regions.
//!
//! Biome type is determined by Perlin noise at frequency 0.03. The noise value
//! is thresholded: >0.2 = FoodRich, <-0.2 = Danger, middle = Barren.
//!
//! For performance, biome values are pre-baked into a grid at construction time.
//! Ray marching queries hit the grid (O(1) lookup) instead of evaluating Perlin
//! noise on every call.

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

/// Grid resolution for the baked biome map.
const BIOME_GRID_RES: usize = 256;

/// Pre-baked biome map: classifies world regions via a lookup grid.
pub struct BiomeMap {
    grid: Vec<BiomeType>,
    world_size: f32,
    inv_cell: f32,
}

impl BiomeMap {
    /// Create a biome map from a seed. Bakes a BIOME_GRID_RES² lookup grid.
    pub fn new(seed: u32, world_size: f32) -> Self {
        let perlin = Perlin::new(seed);
        let frequency = 0.03;
        let res = BIOME_GRID_RES;
        let half = world_size / 2.0;
        let cell = world_size / res as f32;
        let mut grid = vec![BiomeType::Barren; res * res];

        for row in 0..res {
            let z = -half + (row as f32 + 0.5) * cell;
            for col in 0..res {
                let x = -half + (col as f32 + 0.5) * cell;
                let val = perlin.get([x as f64 * frequency, z as f64 * frequency]);
                grid[row * res + col] = if val > 0.2 {
                    BiomeType::FoodRich
                } else if val < -0.2 {
                    BiomeType::Danger
                } else {
                    BiomeType::Barren
                };
            }
        }

        Self {
            grid,
            world_size,
            inv_cell: res as f32 / world_size,
        }
    }

    /// Return the biome type at world-space (x, z) via grid lookup.
    #[inline]
    pub fn biome_at(&self, x: f32, z: f32) -> BiomeType {
        let half = self.world_size / 2.0;
        let col = ((x + half) * self.inv_cell) as usize;
        let row = ((z + half) * self.inv_cell) as usize;
        let res = BIOME_GRID_RES;
        let col = col.min(res - 1);
        let row = row.min(res - 1);
        self.grid[row * res + col]
    }
}
