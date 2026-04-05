//! Grid-based spatial index for O(1) proximity queries on food items and agents.
//!
//! The world is divided into cells of `CELL_SIZE` units. Each cell stores
//! indices into the source Vec. Proximity queries check only the 3×3
//! neighborhood around the query point, replacing O(N) linear scans with O(1).

use std::collections::HashMap;

use glam::Vec3;

/// Cell size in world units. Must be >= the largest query radius used by
/// proximity checks (currently 5.0 for agent touch) so that a 3×3
/// neighborhood always covers the search area.
const CELL_SIZE: f32 = 8.0;

/// Spatial grid mapping cell coordinates to food item indices.
pub struct FoodGrid {
    cells: HashMap<(i32, i32), Vec<usize>>,
}

impl FoodGrid {
    /// Build the grid from current food items.
    pub fn from_items(items: &[super::entity::FoodItem]) -> Self {
        let mut cells: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (i, item) in items.iter().enumerate() {
            if item.consumed {
                continue;
            }
            let key = Self::cell_key(item.position.x, item.position.z);
            cells.entry(key).or_default().push(i);
        }
        FoodGrid { cells }
    }

    /// Rebuild from current food items (reuses Vec allocations within cells).
    pub fn rebuild(&mut self, items: &[super::entity::FoodItem]) {
        for cell in self.cells.values_mut() {
            cell.clear();
        }
        for (i, item) in items.iter().enumerate() {
            if item.consumed {
                continue;
            }
            let key = Self::cell_key(item.position.x, item.position.z);
            self.cells.entry(key).or_default().push(i);
        }
        self.cells.retain(|_, v| !v.is_empty());
    }

    /// Return indices of food items within `radius` of `(x, z)`.
    /// Only checks the 3×3 cell neighborhood — O(1) amortized.
    pub fn query_nearby(&self, x: f32, z: f32) -> NearbyIter<'_> {
        let (cx, cz) = Self::cell_key(x, z);
        NearbyIter {
            grid: self,
            cx,
            cz,
            dx: -1,
            dz: -1,
            inner_idx: 0,
        }
    }

    /// Mark a food item as consumed (remove from its cell).
    pub fn remove(&mut self, idx: usize, x: f32, z: f32) {
        let key = Self::cell_key(x, z);
        if let Some(cell) = self.cells.get_mut(&key) {
            cell.retain(|&i| i != idx);
        }
    }

    /// Insert a food item into the grid at position (x, z).
    pub fn insert(&mut self, idx: usize, x: f32, z: f32) {
        let key = Self::cell_key(x, z);
        self.cells.entry(key).or_default().push(idx);
    }

    fn cell_key(x: f32, z: f32) -> (i32, i32) {
        (
            (x / CELL_SIZE).floor() as i32,
            (z / CELL_SIZE).floor() as i32,
        )
    }
}

/// Iterator over food indices in the 3×3 neighborhood of a query cell.
pub struct NearbyIter<'a> {
    grid: &'a FoodGrid,
    cx: i32,
    cz: i32,
    dx: i32,
    dz: i32,
    inner_idx: usize,
}

impl<'a> Iterator for NearbyIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            let key = (self.cx + self.dx, self.cz + self.dz);
            if let Some(cell) = self.grid.cells.get(&key) {
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

// ── Agent spatial grid ─────────────────────────────────────────────────

/// Spatial grid mapping cell coordinates to agent indices.
/// Same cell size and neighborhood pattern as `FoodGrid`.
pub struct AgentGrid {
    cells: HashMap<(i32, i32), Vec<usize>>,
}

impl AgentGrid {
    /// Build the grid from agent positions. Dead agents (alive==false) are skipped.
    pub fn from_positions(positions: &[(Vec3, bool)]) -> Self {
        let mut cells: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (i, (pos, alive)) in positions.iter().enumerate() {
            if !alive {
                continue;
            }
            let key = Self::cell_key(pos.x, pos.z);
            cells.entry(key).or_default().push(i);
        }
        AgentGrid { cells }
    }

    /// Clear and rebuild from current positions (reuses Vec allocations within cells).
    pub fn rebuild(&mut self, positions: &[(Vec3, bool)]) {
        for cell in self.cells.values_mut() {
            cell.clear();
        }
        for (i, (pos, alive)) in positions.iter().enumerate() {
            if !alive {
                continue;
            }
            let key = Self::cell_key(pos.x, pos.z);
            self.cells.entry(key).or_default().push(i);
        }
        self.cells.retain(|_, v| !v.is_empty());
    }

    /// Return indices of agents in the 3×3 cell neighborhood of `(x, z)`.
    pub fn query_nearby(&self, x: f32, z: f32) -> AgentNearbyIter<'_> {
        let (cx, cz) = Self::cell_key(x, z);
        AgentNearbyIter {
            grid: self,
            cx,
            cz,
            dx: -1,
            dz: -1,
            inner_idx: 0,
        }
    }

    fn cell_key(x: f32, z: f32) -> (i32, i32) {
        (
            (x / CELL_SIZE).floor() as i32,
            (z / CELL_SIZE).floor() as i32,
        )
    }
}

/// Iterator over agent indices in the 3×3 neighborhood of a query cell.
pub struct AgentNearbyIter<'a> {
    grid: &'a AgentGrid,
    cx: i32,
    cz: i32,
    dx: i32,
    dz: i32,
    inner_idx: usize,
}

impl<'a> Iterator for AgentNearbyIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            let key = (self.cx + self.dx, self.cz + self.dz);
            if let Some(cell) = self.grid.cells.get(&key) {
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
