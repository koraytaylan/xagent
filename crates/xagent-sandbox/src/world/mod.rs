//! World simulation: terrain, biomes, food entities, and combined world state.

pub mod biome;
pub mod entity;
pub mod spatial;
pub mod terrain;

use std::sync::Mutex;

use rand::SeedableRng;
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
    /// Seeded RNG for deterministic spawn positions and food respawn.
    spawn_rng: Mutex<rand::rngs::SmallRng>,
}

impl WorldState {
    /// Generate a complete world from configuration: terrain heightmap, biome map,
    /// and scattered food items. Uses deterministic seeds derived from `config.seed`.
    pub fn new(config: WorldConfig) -> Self {
        let terrain_seed = config.seed as u32;
        let biome_seed = (config.seed.wrapping_add(95)) as u32;
        let food_seed = config.seed.wrapping_add(137);
        let spawn_seed = config.seed.wrapping_add(211);
        let terrain = TerrainData::generate(config.world_size, 128, terrain_seed);
        let biome_map = BiomeMap::new(biome_seed, config.world_size);
        let food_rng = rand::rngs::SmallRng::seed_from_u64(food_seed);
        let food_items = entity::spawn_food_seeded(&terrain, &biome_map, config.food_density, food_rng);
        let food_grid = FoodGrid::from_items(&food_items);
        let spawn_rng = Mutex::new(rand::rngs::SmallRng::seed_from_u64(spawn_seed));

        Self {
            terrain,
            biome_map,
            food_items,
            food_grid,
            config,
            respawned_scratch: Vec::new(),
            spawn_rng,
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
    ///
    /// Uses the world's seeded RNG so results are deterministic for a given
    /// `WorldConfig::seed`.
    ///
    /// NOTE: Takes `&self` but acquires an internal `Mutex` lock on the RNG.
    /// Concurrent callers will serialize on this lock. If respawn is ever
    /// parallelized, this becomes a contention point.
    pub fn safe_spawn_position(&self) -> glam::Vec3 {
        use rand::Rng;
        let mut rng = self.spawn_rng.lock().expect("spawn_rng mutex poisoned");
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

    /// Return a reference to the last respawned food indices (from the most recent update() call).
    pub fn last_respawned_indices(&self) -> &[usize] {
        &self.respawned_scratch
    }

    /// Tick food respawn timers. Returns `true` if any food respawned.
    /// Uses incremental grid inserts instead of full rebuild.
    pub fn update(&mut self, dt: f32) -> bool {
        let mut rng = self.spawn_rng.lock().expect("spawn_rng mutex poisoned");
        entity::update_food(
            &mut self.food_items,
            dt,
            &self.terrain,
            &self.biome_map,
            &mut self.respawned_scratch,
            &mut *rng,
        );
        drop(rng);
        for &idx in &self.respawned_scratch {
            let pos = self.food_items[idx].position;
            self.food_grid.insert(idx, pos.x, pos.z);
        }
        !self.respawned_scratch.is_empty()
    }
}
