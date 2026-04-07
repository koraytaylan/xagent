//! Procedural terrain generation using multi-octave Perlin noise.
//!
//! The terrain is a heightmap grid with bilinear interpolation for smooth
//! height queries at arbitrary world positions. Three noise octaves create
//! natural-looking rolling hills with fine detail.

use noise::{NoiseFn, Perlin};

use super::biome::{BiomeMap, BiomeType};
use super::Mesh;
use crate::renderer::Vertex;

/// Heightmap terrain stored as a flat grid of height values.
/// World coordinates range from `[-size/2, size/2]` on both X and Z axes.
pub struct TerrainData {
    pub size: f32,
    pub subdivisions: u32,
    /// Flat array of heights: heights[z * verts_per_side + x]
    pub heights: Vec<f32>,
    // Pre-cached constants for height_at (avoids recomputation per call).
    half: f32,
    inv_step: f32,
    vps: u32,
    max_coord: f32, // (vps - 1) as f32
    max_idx: u32,   // vps - 2
}

impl TerrainData {
    /// Generate terrain using multi-octave Perlin noise.
    ///
    /// Three octaves at frequencies 0.02, 0.05, 0.10 with amplitudes 4.0, 2.0, 0.5
    /// produce terrain heights in roughly ±5 units. Lower frequencies create broad hills;
    /// higher frequencies add fine surface detail.
    pub fn generate(size: f32, subdivisions: u32, seed: u32) -> Self {
        assert!(subdivisions >= 2, "Terrain subdivisions must be >= 2");
        let perlin = Perlin::new(seed);
        let vps = subdivisions + 1;
        let half = size / 2.0;
        let step = size / subdivisions as f32;

        let mut heights = Vec::with_capacity((vps * vps) as usize);

        for z in 0..vps {
            for x in 0..vps {
                let wx = (-half + x as f32 * step) as f64;
                let wz = (-half + z as f32 * step) as f64;

                // Multi-octave noise for natural-looking terrain (±5 units)
                let h = perlin.get([wx * 0.02, wz * 0.02]) * 4.0
                    + perlin.get([wx * 0.05, wz * 0.05]) * 2.0
                    + perlin.get([wx * 0.10, wz * 0.10]) * 0.5;

                heights.push(h as f32);
            }
        }

        let step = size / subdivisions as f32;
        Self {
            size,
            subdivisions,
            heights,
            half,
            inv_step: 1.0 / step,
            vps,
            max_coord: (vps - 1) as f32,
            max_idx: vps - 2,
        }
    }

    /// Bilinear-interpolated height at any world-space (x, z).
    #[inline(always)]
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let gx = ((x + self.half) * self.inv_step).clamp(0.0, self.max_coord);
        let gz = ((z + self.half) * self.inv_step).clamp(0.0, self.max_coord);

        let ix = (gx.floor() as u32).min(self.max_idx);
        let iz = (gz.floor() as u32).min(self.max_idx);
        let fx = gx - ix as f32;
        let fz = gz - iz as f32;

        let vps = self.vps;
        let h00 = self.heights[(iz * vps + ix) as usize];
        let h10 = self.heights[(iz * vps + ix + 1) as usize];
        let h01 = self.heights[((iz + 1) * vps + ix) as usize];
        let h11 = self.heights[((iz + 1) * vps + ix + 1) as usize];

        let h0 = h00 + (h10 - h00) * fx;
        let h1 = h01 + (h11 - h01) * fx;
        h0 + (h1 - h0) * fz
    }

    /// Build the renderable terrain mesh, coloring vertices by height and biome.
    pub fn build_mesh(&self, biome_map: &BiomeMap) -> Mesh {
        let half = self.size / 2.0;
        let step = self.size / self.subdivisions as f32;
        let vps = self.subdivisions + 1;

        let mut vertices = Vec::with_capacity((vps * vps) as usize);
        let mut indices = Vec::with_capacity((self.subdivisions * self.subdivisions * 6) as usize);

        for z in 0..vps {
            for x in 0..vps {
                let px = -half + x as f32 * step;
                let pz = -half + z as f32 * step;
                let py = self.heights[(z * vps + x) as usize];
                let color = terrain_color(px, pz, py, biome_map);

                vertices.push(Vertex {
                    position: [px, py, pz],
                    color,
                });
            }
        }

        for z in 0..self.subdivisions {
            for x in 0..self.subdivisions {
                let tl = z * vps + x;
                let tr = tl + 1;
                let bl = (z + 1) * vps + x;
                let br = bl + 1;

                indices.push(tl);
                indices.push(bl);
                indices.push(tr);
                indices.push(tr);
                indices.push(bl);
                indices.push(br);
            }
        }

        Mesh { vertices, indices }
    }
}

/// Blend height-based tint with biome color.
fn terrain_color(x: f32, z: f32, height: f32, biome_map: &BiomeMap) -> [f32; 3] {
    // Normalize height to roughly [0, 1]
    let t = ((height + 5.0) / 10.0).clamp(0.0, 1.0);

    let biome = biome_map.biome_at(x, z);
    match biome {
        BiomeType::FoodRich => [0.12 + t * 0.10, 0.40 + t * 0.05, 0.08 + t * 0.04],
        BiomeType::Barren => [0.40 + t * 0.20, 0.32 + t * 0.05, 0.15 + t * 0.05],
        BiomeType::Danger => [0.45 + t * 0.20, 0.15 + t * 0.05, 0.10 + t * 0.03],
    }
}
