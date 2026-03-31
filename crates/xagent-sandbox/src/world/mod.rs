//! World simulation: terrain, biomes, food entities, and combined world state.

pub mod biome;
pub mod entity;
pub mod spatial;
pub mod terrain;

use xagent_shared::WorldConfig;

use self::biome::BiomeMap;
use self::entity::FoodItem;
use self::spatial::FoodGrid;
use self::terrain::TerrainData;
use crate::renderer::Vertex;

/// A renderable mesh: vertices and triangle indices for GPU upload.
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// Complete world simulation state.
pub struct WorldState {
    pub terrain: TerrainData,
    pub biome_map: BiomeMap,
    pub food_items: Vec<FoodItem>,
    pub food_grid: FoodGrid,
    pub config: WorldConfig,
    /// Scratch buffer for respawned food indices (avoids allocation per tick).
    respawned_scratch: Vec<usize>,
}

impl WorldState {
    /// Generate a complete world from configuration: terrain heightmap, biome map,
    /// and scattered food items. Uses deterministic seeds derived from `config.seed`.
    pub fn new(config: WorldConfig) -> Self {
        let terrain_seed = config.seed as u32;
        let biome_seed = (config.seed.wrapping_add(95)) as u32;
        let terrain = TerrainData::generate(config.world_size, 128, terrain_seed);
        let biome_map = BiomeMap::new(biome_seed, config.world_size);
        let food_items = entity::spawn_food(&terrain, &biome_map, config.food_density);
        let food_grid = FoodGrid::from_items(&food_items);

        Self {
            terrain,
            biome_map,
            food_items,
            food_grid,
            config,
            respawned_scratch: Vec::new(),
        }
    }

    /// Static terrain mesh (build once).
    pub fn terrain_mesh(&self) -> Mesh {
        self.terrain.build_mesh(&self.biome_map)
    }

    /// Combined mesh of all visible food items.
    pub fn food_mesh(&self) -> Mesh {
        entity::generate_food_mesh(&self.food_items)
    }

    /// Pick a random position that is NOT in a danger biome.
    /// Tries up to 50 candidates; falls back to the last one if all are danger
    /// (extremely unlikely with typical biome distributions).
    pub fn safe_spawn_position(&self) -> glam::Vec3 {
        use rand::Rng;
        let mut rng = rand::rng();
        let half = self.config.world_size / 2.0 - 5.0;
        let mut x = 0.0f32;
        let mut z = 0.0f32;
        for _ in 0..50 {
            x = rng.random_range(-half..half);
            z = rng.random_range(-half..half);
            if self.biome_map.biome_at(x, z) != biome::BiomeType::Danger {
                break;
            }
        }
        let y = self.terrain.height_at(x, z) + 1.0;
        glam::Vec3::new(x, y, z)
    }

    /// Tick food respawn timers. Returns `true` if any food respawned.
    /// Uses incremental grid inserts instead of full rebuild.
    pub fn update(&mut self, dt: f32) -> bool {
        entity::update_food(
            &mut self.food_items,
            dt,
            &self.terrain,
            &self.biome_map,
            &mut self.respawned_scratch,
        );
        for &idx in &self.respawned_scratch {
            let pos = self.food_items[idx].position;
            self.food_grid.insert(idx, pos.x, pos.z);
        }
        !self.respawned_scratch.is_empty()
    }
}
