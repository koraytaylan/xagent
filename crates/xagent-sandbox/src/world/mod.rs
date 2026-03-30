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
}

impl WorldState {
    /// Generate a complete world from configuration: terrain heightmap, biome map,
    /// and scattered food items. Uses deterministic seeds derived from `config.seed`.
    pub fn new(config: WorldConfig) -> Self {
        let terrain_seed = config.seed as u32;
        let biome_seed = (config.seed.wrapping_add(95)) as u32;
        let terrain = TerrainData::generate(config.world_size, 128, terrain_seed);
        let biome_map = BiomeMap::new(biome_seed);
        let food_items = entity::spawn_food(&terrain, &biome_map, config.food_density);
        let food_grid = FoodGrid::from_items(&food_items);

        Self {
            terrain,
            biome_map,
            food_items,
            food_grid,
            config,
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

    /// Tick food respawn timers. Returns `true` if any food respawned.
    /// Rebuilds the spatial grid when food positions change.
    pub fn update(&mut self, dt: f32) -> bool {
        let changed = entity::update_food(&mut self.food_items, dt, &self.terrain, &self.biome_map);
        if changed {
            self.food_grid.rebuild(&self.food_items);
        }
        changed
    }
}
