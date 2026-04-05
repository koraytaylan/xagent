//! Grid-based spatial index for O(1) proximity queries on food items and agents.
//!
//! The world is divided into cells of `CELL_SIZE` units. Each cell stores
//! indices into the source Vec. Proximity queries check only the 3×3
//! neighborhood around the query point, replacing O(N) linear scans with O(1).
//!
//! Uses a flat `Vec<Vec<usize>>` instead of `HashMap` for cell storage.
//! Vision raycasting queries the grid ~2,400 times per agent per tick
//! (48 rays × 50 steps), so eliminating hash overhead is critical.

use glam::Vec3;

/// Cell size in world units. Must be >= the largest query radius used by
/// proximity checks (currently 5.0 for agent touch) so that a 3×3
/// neighborhood always covers the search area.
const CELL_SIZE: f32 = 8.0;

/// Convert world coordinate to cell index along one axis.
#[inline(always)]
fn cell_coord(v: f32) -> i32 {
    (v / CELL_SIZE).floor() as i32
}

// ── FoodGrid ──────────────────────────────────────────────────────────────

/// Spatial grid mapping cell coordinates to food item indices.
/// Backed by a flat Vec for O(1) lookup without hashing.
pub struct FoodGrid {
    cells: Vec<Vec<usize>>,
    width: i32,
    offset: i32,
}

impl FoodGrid {
    /// Build the grid from current food items.
    /// `world_size` determines the grid dimensions.
    pub fn new(world_size: f32) -> Self {
        let width = grid_width(world_size);
        let offset = width / 2;
        FoodGrid {
            cells: vec![Vec::new(); (width * width) as usize],
            width,
            offset,
        }
    }

    /// Build the grid from current food items.
    pub fn from_items(items: &[super::entity::FoodItem], world_size: f32) -> Self {
        let mut grid = Self::new(world_size);
        for (i, item) in items.iter().enumerate() {
            if item.consumed {
                continue;
            }
            if let Some(idx) = grid.flat_index(item.position.x, item.position.z) {
                grid.cells[idx].push(i);
            }
        }
        grid
    }

    /// Rebuild from current food items (reuses Vec allocations within cells).
    pub fn rebuild(&mut self, items: &[super::entity::FoodItem]) {
        for cell in self.cells.iter_mut() {
            cell.clear();
        }
        for (i, item) in items.iter().enumerate() {
            if item.consumed {
                continue;
            }
            if let Some(idx) = self.flat_index(item.position.x, item.position.z) {
                self.cells[idx].push(i);
            }
        }
    }

    /// Return indices of food items in the 3×3 cell neighborhood of `(x, z)`.
    #[inline]
    pub fn query_nearby(&self, x: f32, z: f32) -> FlatNearbyIter<'_> {
        let cx = cell_coord(x);
        let cz = cell_coord(z);
        FlatNearbyIter {
            cells: &self.cells,
            width: self.width,
            offset: self.offset,
            cx,
            cz,
            dx: -1,
            dz: -1,
            inner_idx: 0,
        }
    }

    /// Mark a food item as consumed (remove from its cell).
    pub fn remove(&mut self, idx: usize, x: f32, z: f32) {
        if let Some(flat) = self.flat_index(x, z) {
            self.cells[flat].retain(|&i| i != idx);
        }
    }

    /// Insert a food item into the grid at position (x, z).
    pub fn insert(&mut self, idx: usize, x: f32, z: f32) {
        if let Some(flat) = self.flat_index(x, z) {
            self.cells[flat].push(idx);
        }
    }

    #[inline(always)]
    fn flat_index(&self, x: f32, z: f32) -> Option<usize> {
        let cx = cell_coord(x) + self.offset;
        let cz = cell_coord(z) + self.offset;
        if cx >= 0 && cx < self.width && cz >= 0 && cz < self.width {
            Some((cx * self.width + cz) as usize)
        } else {
            None
        }
    }
}

// ── AgentGrid ─────────────────────────────────────────────────────────────

/// Spatial grid mapping cell coordinates to agent indices.
/// Same cell size and neighborhood pattern as `FoodGrid`.
pub struct AgentGrid {
    cells: Vec<Vec<usize>>,
    width: i32,
    offset: i32,
}

impl AgentGrid {
    /// Build the grid from agent positions. Dead agents (alive==false) are skipped.
    pub fn from_positions(positions: &[(Vec3, bool)], world_size: f32) -> Self {
        let width = grid_width(world_size);
        let offset = width / 2;
        let mut cells = vec![Vec::new(); (width * width) as usize];
        for (i, (pos, alive)) in positions.iter().enumerate() {
            if !*alive {
                continue;
            }
            let cx = cell_coord(pos.x) + offset;
            let cz = cell_coord(pos.z) + offset;
            if cx >= 0 && cx < width && cz >= 0 && cz < width {
                cells[(cx * width + cz) as usize].push(i);
            }
        }
        AgentGrid { cells, width, offset }
    }

    /// Clear and rebuild from current positions (reuses Vec allocations within cells).
    pub fn rebuild(&mut self, positions: &[(Vec3, bool)]) {
        for cell in self.cells.iter_mut() {
            cell.clear();
        }
        for (i, (pos, alive)) in positions.iter().enumerate() {
            if !*alive {
                continue;
            }
            let cx = cell_coord(pos.x) + self.offset;
            let cz = cell_coord(pos.z) + self.offset;
            if cx >= 0 && cx < self.width && cz >= 0 && cz < self.width {
                self.cells[(cx * self.width + cz) as usize].push(i);
            }
        }
    }

    /// Return indices of agents in the 3×3 cell neighborhood of `(x, z)`.
    #[inline]
    pub fn query_nearby(&self, x: f32, z: f32) -> FlatNearbyIter<'_> {
        let cx = cell_coord(x);
        let cz = cell_coord(z);
        FlatNearbyIter {
            cells: &self.cells,
            width: self.width,
            offset: self.offset,
            cx,
            cz,
            dx: -1,
            dz: -1,
            inner_idx: 0,
        }
    }
}

// ── Shared flat iterator ──────────────────────────────────────────────────

/// Iterator over indices in the 3×3 neighborhood of a query cell.
/// Shared by both `FoodGrid` and `AgentGrid`.
pub struct FlatNearbyIter<'a> {
    cells: &'a [Vec<usize>],
    width: i32,
    offset: i32,
    cx: i32,
    cz: i32,
    dx: i32,
    dz: i32,
    inner_idx: usize,
}

impl<'a> Iterator for FlatNearbyIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            let rx = self.cx + self.dx + self.offset;
            let rz = self.cz + self.dz + self.offset;
            if rx >= 0 && rx < self.width && rz >= 0 && rz < self.width {
                let flat = (rx * self.width + rz) as usize;
                let cell = &self.cells[flat];
                if self.inner_idx < cell.len() {
                    let val = cell[self.inner_idx];
                    self.inner_idx += 1;
                    return Some(val);
                }
            }
            // Advance to next cell in 3×3 neighborhood
            self.inner_idx = 0;
            self.dx += 1;
            if self.dx > 1 {
                self.dx = -1;
                self.dz += 1;
                if self.dz > 1 {
                    return None;
                }
            }
        }
    }
}

/// Compute grid width (cells per axis) for a given world size.
/// Adds 2 extra cells on each side to handle entities near world edges
/// and vision rays that extend slightly beyond bounds.
fn grid_width(world_size: f32) -> i32 {
    (world_size / CELL_SIZE).ceil() as i32 + 4
}

// ── Legacy type aliases for backward compatibility ────────────────────────

/// Alias so callers using `NearbyIter` still compile.
pub type NearbyIter<'a> = FlatNearbyIter<'a>;

/// Alias so callers using `AgentNearbyIter` still compile.
pub type AgentNearbyIter<'a> = FlatNearbyIter<'a>;
