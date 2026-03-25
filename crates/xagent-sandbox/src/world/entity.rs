//! Food entities: spawning, consumption, and respawn mechanics.
//!
//! Food items are scattered across FoodRich biomes at world generation time.
//! When consumed, they start a respawn timer and relocate to a new random
//! food-rich position, forcing agents to forage rather than camp.

use glam::Vec3;
use rand::Rng;

use super::biome::{BiomeMap, BiomeType};
use super::terrain::TerrainData;
use super::Mesh;
use crate::renderer::Vertex;

/// A food item in the world.
pub struct FoodItem {
    pub position: Vec3,
    pub consumed: bool,
    /// Seconds remaining until respawn (counts down while consumed).
    pub respawn_timer: f32,
}

impl FoodItem {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            consumed: false,
            respawn_timer: 0.0,
        }
    }
}

/// Scatter food items across food-rich biomes.
pub fn spawn_food(terrain: &TerrainData, biome_map: &BiomeMap, density: f32) -> Vec<FoodItem> {
    let half = terrain.size / 2.0;
    let mut rng = rand::rng();
    let mut items = Vec::new();

    let step = 4.0; // sample grid spacing
    let area = step * step;
    let expected = density * area;

    let mut x = -half;
    while x < half {
        let mut z = -half;
        while z < half {
            if biome_map.biome_at(x, z) == BiomeType::FoodRich && rng.random::<f32>() < expected {
                let fx = (x + rng.random::<f32>() * step).clamp(-half, half);
                let fz = (z + rng.random::<f32>() * step).clamp(-half, half);
                let fy = terrain.height_at(fx, fz) + 0.35;
                items.push(FoodItem::new(Vec3::new(fx, fy, fz)));
            }
            z += step;
        }
        x += step;
    }

    items
}

/// Build a combined mesh for all non-consumed food items (small green cubes).
pub fn generate_food_mesh(items: &[FoodItem]) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for item in items {
        if item.consumed {
            continue;
        }
        append_cube(
            &mut vertices,
            &mut indices,
            item.position,
            0.6,
            [0.1, 0.8, 0.2],
        );
    }

    Mesh { vertices, indices }
}

/// Tick respawn timers for consumed food. When a food item respawns it
/// relocates to a new random position in a food-rich biome, forcing agents
/// to forage rather than camp a single spot.
pub fn update_food(
    items: &mut [FoodItem],
    dt: f32,
    terrain: &TerrainData,
    biome_map: &BiomeMap,
) -> bool {
    let mut any_respawned = false;
    let mut rng = rand::rng();
    let half = terrain.size / 2.0;

    for item in items.iter_mut() {
        if item.consumed {
            item.respawn_timer -= dt;
            if item.respawn_timer <= 0.0 {
                item.consumed = false;
                any_respawned = true;

                // Relocate to a new random food-rich position
                for _ in 0..64 {
                    let x = rng.random_range(-half..half);
                    let z = rng.random_range(-half..half);
                    if biome_map.biome_at(x, z) == BiomeType::FoodRich {
                        let y = terrain.height_at(x, z) + 0.35;
                        item.position = Vec3::new(x, y, z);
                        break;
                    }
                }
            }
        }
    }
    any_respawned
}

// ── helper ──────────────────────────────────────────────────────────────

fn append_cube(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    pos: Vec3,
    size: f32,
    color: [f32; 3],
) {
    let h = size / 2.0;
    let base = vertices.len() as u32;

    #[rustfmt::skip]
    let p: [[f32; 3]; 8] = [
        [pos.x - h, pos.y - h, pos.z + h],
        [pos.x + h, pos.y - h, pos.z + h],
        [pos.x + h, pos.y + h, pos.z + h],
        [pos.x - h, pos.y + h, pos.z + h],
        [pos.x - h, pos.y - h, pos.z - h],
        [pos.x + h, pos.y - h, pos.z - h],
        [pos.x + h, pos.y + h, pos.z - h],
        [pos.x - h, pos.y + h, pos.z - h],
    ];

    let shades = [1.0_f32, 0.9, 0.85, 0.7, 0.8, 0.75];
    #[rustfmt::skip]
    let faces: [(usize, usize, usize, usize); 6] = [
        (0, 1, 2, 3), (5, 4, 7, 6), (3, 2, 6, 7),
        (4, 5, 1, 0), (4, 0, 3, 7), (1, 5, 6, 2),
    ];

    for (fi, &(a, b, c, d)) in faces.iter().enumerate() {
        let s = shades[fi];
        let col = [color[0] * s, color[1] * s, color[2] * s];
        let fb = base + (fi as u32) * 4;

        vertices.push(Vertex { position: p[a], color: col });
        vertices.push(Vertex { position: p[b], color: col });
        vertices.push(Vertex { position: p[c], color: col });
        vertices.push(Vertex { position: p[d], color: col });

        indices.extend_from_slice(&[fb, fb + 1, fb + 2, fb, fb + 2, fb + 3]);
    }
}
