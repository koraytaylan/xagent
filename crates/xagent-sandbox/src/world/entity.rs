//! Food entities: spawning, consumption, and respawn mechanics.
//!
//! Food items are scattered across FoodRich biomes at world generation time.
//! When consumed, they start a respawn timer and relocate to a new random
//! food-rich position, forcing agents to forage rather than camp.

use glam::Vec3;
use rand::Rng;
use xagent_brain::buffers::{FOOD_STATE_STRIDE, F_POS_X, F_POS_Y, F_POS_Z, F_RESPAWN_TIMER};

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

/// Scatter food items across food-rich biomes using the thread-local RNG.
///
/// Prefer [`spawn_food_seeded`] for deterministic world generation.
pub fn spawn_food(terrain: &TerrainData, biome_map: &BiomeMap, density: f32) -> Vec<FoodItem> {
    spawn_food_seeded(terrain, biome_map, density, rand::rng())
}

/// Scatter food items across food-rich biomes using a caller-supplied RNG.
///
/// Passing a seeded `SmallRng` produces deterministic worlds for a given seed.
pub fn spawn_food_seeded(
    terrain: &TerrainData,
    biome_map: &BiomeMap,
    density: f32,
    mut rng: impl Rng,
) -> Vec<FoodItem> {
    let half = terrain.size / 2.0;
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

/// Build the food mesh from the authoritative GPU food state
/// (`GpuKernel::cached_food_state`: `[pos_x, pos_y, pos_z, respawn_timer]`
/// per item). Items awaiting respawn (`respawn_timer > 0`) are skipped, so the
/// main viewport shows exactly the food the simulation currently has — the same
/// source and filter the mini-map already uses in `update_world_snapshot`.
///
/// This replaces meshing the CPU-side `food_items`, which is uploaded once at
/// startup and never synced back from the GPU during a live run, so it always
/// rendered the original food layout (eaten food never disappeared, respawned
/// food never relocated). Reuses the already-downloaded `food_cache` readback —
/// no new GPU work, no new readback, the same per-frame cube build as before.
pub fn generate_food_mesh_from_state(food_state: &[f32]) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for item in food_state.chunks_exact(FOOD_STATE_STRIDE) {
        if item[F_RESPAWN_TIMER] > 0.0 {
            continue;
        }
        append_cube(
            &mut vertices,
            &mut indices,
            Vec3::new(item[F_POS_X], item[F_POS_Y], item[F_POS_Z]),
            0.6,
            [0.1, 0.8, 0.2],
        );
    }

    Mesh { vertices, indices }
}

/// Tick respawn timers for consumed food. When a food item respawns it
/// relocates to a new random position in a food-rich biome, forcing agents
/// to forage rather than camp a single spot.
/// Tick food respawn timers. Returns indices of food items that respawned.
///
/// The `rng` parameter should be the world's seeded RNG for deterministic
/// food respawn positions.
pub fn update_food(
    items: &mut [FoodItem],
    dt: f32,
    terrain: &TerrainData,
    biome_map: &BiomeMap,
    respawned_indices: &mut Vec<usize>,
    rng: &mut impl Rng,
) {
    respawned_indices.clear();
    let half = terrain.size / 2.0;

    for (i, item) in items.iter_mut().enumerate() {
        if item.consumed {
            item.respawn_timer -= dt;
            if item.respawn_timer <= 0.0 {
                item.consumed = false;
                respawned_indices.push(i);

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
}

// ── helper ──────────────────────────────────────────────────────────────

fn append_cube(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    position: Vec3,
    size: f32,
    color: [f32; 3],
) {
    let h = size / 2.0;
    let base = vertices.len() as u32;

    #[rustfmt::skip]
    let p: [[f32; 3]; 8] = [
        [position.x - h, position.y - h, position.z + h],
        [position.x + h, position.y - h, position.z + h],
        [position.x + h, position.y + h, position.z + h],
        [position.x - h, position.y + h, position.z + h],
        [position.x - h, position.y - h, position.z - h],
        [position.x + h, position.y - h, position.z - h],
        [position.x + h, position.y + h, position.z - h],
        [position.x - h, position.y + h, position.z - h],
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

        vertices.push(Vertex {
            position: p[a],
            color: col,
        });
        vertices.push(Vertex {
            position: p[b],
            color: col,
        });
        vertices.push(Vertex {
            position: p[c],
            color: col,
        });
        vertices.push(Vertex {
            position: p[d],
            color: col,
        });

        indices.extend_from_slice(&[fb, fb + 1, fb + 2, fb, fb + 2, fb + 3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xagent_brain::buffers::FOOD_STATE_STRIDE;

    /// One cube is `6 faces × 4 verts = 24` vertices and `6 × 6 = 36`
    /// indices (see `append_cube`).
    const VERTS_PER_CUBE: usize = 24;
    const INDICES_PER_CUBE: usize = 36;

    /// The authoritative GPU food state meshes only food that is currently
    /// available — items with `respawn_timer > 0` (eaten, awaiting respawn)
    /// must be skipped, matching the mini-map's `update_world_snapshot`
    /// filter so the viewport stops showing eaten food.
    #[test]
    fn food_mesh_from_state_skips_respawning_items() {
        // Two records of [pos_x, pos_y, pos_z, respawn_timer]: one available
        // at (3, 1, -2), one awaiting respawn at (7, 1, 4).
        let state = [
            3.0_f32, 1.0, -2.0, 0.0, // available → one cube
            7.0, 1.0, 4.0, 5.0, // respawning → skipped
        ];
        assert_eq!(state.len(), 2 * FOOD_STATE_STRIDE);

        let mesh = generate_food_mesh_from_state(&state);

        assert_eq!(
            mesh.vertices.len(),
            VERTS_PER_CUBE,
            "exactly one cube (the available item) should be meshed"
        );
        assert_eq!(mesh.indices.len(), INDICES_PER_CUBE);

        // Every vertex must lie within ±half-size (0.3) of the available
        // item at (x=3, z=-2) and none near the respawning item at (7, 4).
        for v in &mesh.vertices {
            assert!(
                (v.position[0] - 3.0).abs() <= 0.3 + 1e-6,
                "vertex x {} not near available item x=3.0",
                v.position[0]
            );
            assert!(
                (v.position[2] - (-2.0)).abs() <= 0.3 + 1e-6,
                "vertex z {} not near available item z=-2.0",
                v.position[2]
            );
        }
    }

    /// When every item is awaiting respawn, the viewport mesh is empty —
    /// no stale food cubes linger.
    #[test]
    fn food_mesh_from_state_empty_when_all_respawning() {
        let state = [0.0_f32, 1.0, 0.0, 2.0, 5.0, 1.0, 5.0, 9.0];
        let mesh = generate_food_mesh_from_state(&state);
        assert!(mesh.vertices.is_empty());
        assert!(mesh.indices.is_empty());
    }
}
