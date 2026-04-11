use glam::Vec3;
use xagent_shared::{SensoryFrame, TouchContact, VisualField};

use super::AgentBody;
use crate::world::biome::BiomeType;
use crate::world::WorldState;

/// Surface tag constants for touch contacts. The brain receives these as opaque
/// integers — it must learn what each tag correlates with through experience.
const TOUCH_FOOD: u32 = 1;
const TOUCH_TERRAIN_EDGE: u32 = 2;
const TOUCH_HAZARD: u32 = 3;
const TOUCH_AGENT: u32 = 4;

/// Positions of other agents for inter-agent perception.
pub struct OtherAgent {
    pub position: Vec3,
    pub alive: bool,
}

/// Produce a full sensory frame from current agent & world state.
/// Writes into `frame` to avoid heap allocation (reuses existing buffers).
pub fn extract_senses(agent: &AgentBody, world: &WorldState, tick: u64, frame: &mut SensoryFrame) {
    extract_senses_with_others(agent, world, tick, &[], frame)
}

/// Produce a sensory frame with awareness of other agents given as a
/// shared positions slice. Skips agent at `self_index` automatically.
/// Uses the spatial `AgentGrid` for O(1) proximity queries in vision and touch.
pub fn extract_senses_with_positions(
    agent: &AgentBody,
    world: &WorldState,
    tick: u64,
    all_positions: &[(Vec3, bool)],
    self_index: usize,
    agent_grid: &crate::world::spatial::AgentGrid,
    frame: &mut SensoryFrame,
) {
    sample_vision_positions(
        agent,
        world,
        all_positions,
        self_index,
        agent_grid,
        &mut frame.vision,
    );
    fill_frame_non_vision(
        agent,
        world,
        tick,
        all_positions,
        self_index,
        agent_grid,
        frame,
    );
}

/// Fill non-vision parts of a sensory frame (proprioception, interoception, touch).
///
/// Used by the GPU bench path: vision is filled from GPU output, then this
/// function fills the remaining fields from CPU-accessible agent state.
pub fn fill_frame_non_vision(
    agent: &AgentBody,
    world: &WorldState,
    tick: u64,
    all_positions: &[(Vec3, bool)],
    self_index: usize,
    agent_grid: &crate::world::spatial::AgentGrid,
    frame: &mut SensoryFrame,
) {
    frame.velocity = agent.body.velocity;
    frame.facing = agent.body.facing;
    frame.angular_velocity = agent.angular_velocity;
    frame.energy_signal = agent.body.internal.energy_signal();
    frame.integrity_signal = agent.body.internal.integrity_signal();
    frame.energy_delta = agent.energy_delta();
    frame.integrity_delta = agent.integrity_delta();
    frame.touch_contacts.clear();
    detect_touch_positions(
        agent,
        world,
        all_positions,
        self_index,
        agent_grid,
        &mut frame.touch_contacts,
    );
    frame.tick = tick;
}

/// Produce a sensory frame with awareness of other agents.
/// Writes into `frame` to avoid heap allocation (reuses existing buffers).
pub fn extract_senses_with_others(
    agent: &AgentBody,
    world: &WorldState,
    tick: u64,
    others: &[OtherAgent],
    frame: &mut SensoryFrame,
) {
    sample_vision(agent, world, others, &mut frame.vision);
    frame.velocity = agent.body.velocity;
    frame.facing = agent.body.facing;
    frame.angular_velocity = agent.angular_velocity;
    frame.energy_signal = agent.body.internal.energy_signal();
    frame.integrity_signal = agent.body.internal.integrity_signal();
    frame.energy_delta = agent.energy_delta();
    frame.integrity_delta = agent.integrity_delta();
    frame.touch_contacts.clear();
    detect_touch_with_others(agent, world, others, &mut frame.touch_contacts);
    frame.tick = tick;
}

// ── vision ──────────────────────────────────────────────────────────────

/// Low-resolution raycast sampling of terrain colors in front of the agent.
/// Resolution is driven by `vf.width × vf.height` with step size 1.0 for efficient marching.
/// Also detects other agents along each ray. Writes into existing `vf` buffer.
fn sample_vision(
    agent: &AgentBody,
    world: &WorldState,
    others: &[OtherAgent],
    vf: &mut VisualField,
) {
    let w = vf.width;
    let h = vf.height;
    vf.clear();

    let half_fov = (90.0_f32 / 2.0).to_radians();
    let tan_hf = half_fov.tan();

    let pos = agent.body.position;
    let fwd = agent.body.facing;
    let right = Vec3::new(fwd.z, 0.0, -fwd.x).normalize_or_zero();

    let max_dist: f32 = 50.0;
    let step: f32 = 1.0;

    for row in 0..h {
        for col in 0..w {
            let u = (col as f32 / (w - 1) as f32) * 2.0 - 1.0;
            let v = (row as f32 / (h - 1) as f32) * 2.0 - 1.0;

            let ray = (fwd + right * u * tan_hf + Vec3::Y * (-v) * tan_hf).normalize_or_zero();

            let (color, depth) =
                march_ray_unified(pos, ray, world, max_dist, step, AgentSlice::Others(others));

            let idx = (row * w + col) as usize;
            let ci = idx * 4;
            vf.color[ci] = color[0];
            vf.color[ci + 1] = color[1];
            vf.color[ci + 2] = color[2];
            vf.color[ci + 3] = color[3];
            vf.depth[idx] = depth / max_dist;
        }
    }
}

/// Discriminated union for the agent-list representations.
/// Avoids duplicating the ray-march loop.
enum AgentSlice<'a> {
    Others(&'a [OtherAgent]),
    Grid {
        grid: &'a crate::world::spatial::AgentGrid,
        all: &'a [(Vec3, bool)],
        self_index: usize,
    },
}

/// Fixed-step ray marching for terrain, food, and agent intersection.
///
/// Advances a ray from `origin` in direction `direction` in increments of `step` units,
/// checking for food items, other agents, and terrain at each step. Returns the
/// hit color and distance traveled. If no hit within `max_dist`, returns sky color.
///
/// Food items are rendered as bright lime-green, distinct from biome colors.
/// This makes food *physically visible* — the brain must still learn that this
/// color correlates with positive gradient. Without food visibility, agents have
/// zero directional signal for food.
fn march_ray_unified(
    origin: Vec3,
    direction: Vec3,
    world: &WorldState,
    max_dist: f32,
    step: f32,
    agents: AgentSlice,
) -> ([f32; 4], f32) {
    let sky: [f32; 4] = [0.53, 0.81, 0.92, 1.0];
    let agent_color: [f32; 4] = [0.9, 0.2, 0.6, 1.0];
    let food_color: [f32; 4] = [0.70, 0.95, 0.20, 1.0];
    let agent_radius_sq: f32 = 1.5 * 1.5;
    let food_radius_sq: f32 = 1.0 * 1.0;

    // Early-out: upward-pointing rays above terrain are almost certainly sky.
    if direction.y > 0.3 {
        let origin_h = world.terrain.height_at(origin.x, origin.z);
        if origin.y > origin_h {
            return (sky, max_dist);
        }
    }

    let num_steps = (max_dist / step) as u32;
    let dir_step = direction * step;
    let mut p = origin;
    for s in 0..num_steps {
        // Check food items via spatial grid (O(1) per step)
        for idx in world.food_grid.query_nearby(p.x, p.z) {
            let food = &world.food_items[idx];
            if food.consumed {
                continue;
            }
            let diff = p - food.position;
            if diff.length_squared() < food_radius_sq {
                return (food_color, s as f32 * step);
            }
        }

        // Check agent hits — dispatch on slice variant
        match &agents {
            AgentSlice::Others(others) => {
                for other in *others {
                    if !other.alive {
                        continue;
                    }
                    let diff = p - other.position;
                    if diff.length_squared() < agent_radius_sq {
                        return (agent_color, s as f32 * step);
                    }
                }
            }
            AgentSlice::Grid {
                grid,
                all,
                self_index,
            } => {
                for j in grid.query_nearby(p.x, p.z) {
                    if j == *self_index {
                        continue;
                    }
                    let (other_pos, alive) = all[j];
                    if !alive {
                        continue;
                    }
                    let diff = p - other_pos;
                    if diff.length_squared() < agent_radius_sq {
                        return (agent_color, s as f32 * step);
                    }
                }
            }
        }

        // Check terrain hit
        let gh = world.terrain.height_at(p.x, p.z);
        if p.y <= gh {
            let c = match world.biome_map.biome_at(p.x, p.z) {
                BiomeType::FoodRich => [0.15, 0.50, 0.10, 1.0],
                BiomeType::Barren => [0.50, 0.40, 0.20, 1.0],
                BiomeType::Danger => [0.60, 0.20, 0.10, 1.0],
            };
            return (c, s as f32 * step);
        }
        p += dir_step;
    }
    (sky, max_dist)
}

// ── touch ───────────────────────────────────────────────────────────────

/// Detect all touch contacts including other agents.
/// Agent-to-agent touch range is 5.0 units with intensity inversely proportional
/// to distance. Appends to the provided contacts buffer.
fn detect_touch_with_others(
    agent: &AgentBody,
    world: &WorldState,
    others: &[OtherAgent],
    contacts: &mut Vec<TouchContact>,
) {
    detect_touch(agent, world, contacts);

    // Other agents as touch contacts
    let agent_touch_range = 5.0;
    let pos = agent.body.position;
    for other in others {
        if !other.alive {
            continue;
        }
        let diff = other.position - pos;
        let dist = diff.length();
        if dist < agent_touch_range && dist > 0.01 {
            contacts.push(TouchContact {
                direction: diff.normalize_or_zero(),
                intensity: 1.0 - (dist / agent_touch_range),
                surface_tag: TOUCH_AGENT,
            });
        }
    }
}

/// Detect touch contacts from environmental features (food, terrain edges, hazards).
/// Uses the spatial grid for O(1) food proximity lookup.
/// Touch range for food is 3.0 units, terrain edge detection starts at 3.0 units
/// from world boundary. Hazard zones produce a constant downward contact.
fn detect_touch(agent: &AgentBody, world: &WorldState, contacts: &mut Vec<TouchContact>) {
    let pos = agent.body.position;

    // Nearby food via spatial grid
    for idx in world.food_grid.query_nearby(pos.x, pos.z) {
        let food = &world.food_items[idx];
        if food.consumed {
            continue;
        }
        let diff = food.position - pos;
        let dist = diff.length();
        if dist < 3.0 {
            contacts.push(TouchContact {
                direction: diff.normalize_or_zero(),
                intensity: 1.0 - (dist / 3.0),
                surface_tag: TOUCH_FOOD,
            });
        }
    }

    // Terrain edges
    let half = world.config.world_size / 2.0;
    let edge = 3.0;
    for (axis, sign, coord) in [
        (Vec3::X, 1.0_f32, pos.x + half),
        (Vec3::NEG_X, -1.0, half - pos.x),
        (Vec3::Z, 1.0, pos.z + half),
        (Vec3::NEG_Z, -1.0, half - pos.z),
    ] {
        if coord < edge {
            contacts.push(TouchContact {
                direction: axis * sign,
                intensity: 1.0 - (coord / edge).max(0.0),
                surface_tag: TOUCH_TERRAIN_EDGE,
            });
        }
    }

    // Hazard zone contact
    if world.biome_map.biome_at(pos.x, pos.z) == BiomeType::Danger {
        contacts.push(TouchContact {
            direction: Vec3::NEG_Y,
            intensity: 0.5,
            surface_tag: TOUCH_HAZARD,
        });
    }
}

// ── position-slice variants (C2: avoid Vec<OtherAgent> allocation) ──────

/// Vision sampling using a shared positions slice and spatial agent grid.
fn sample_vision_positions(
    agent: &AgentBody,
    world: &WorldState,
    all_positions: &[(Vec3, bool)],
    self_index: usize,
    agent_grid: &crate::world::spatial::AgentGrid,
    vf: &mut VisualField,
) {
    let w = vf.width;
    let h = vf.height;
    vf.clear();

    let half_fov = (90.0_f32 / 2.0).to_radians();
    let tan_hf = half_fov.tan();

    let pos = agent.body.position;
    let fwd = agent.body.facing;
    let right = Vec3::new(fwd.z, 0.0, -fwd.x).normalize_or_zero();

    let max_dist: f32 = 50.0;
    let step: f32 = 1.0;

    for row in 0..h {
        for col in 0..w {
            let u = (col as f32 / (w - 1) as f32) * 2.0 - 1.0;
            let v = (row as f32 / (h - 1) as f32) * 2.0 - 1.0;

            let ray = (fwd + right * u * tan_hf + Vec3::Y * (-v) * tan_hf).normalize_or_zero();

            let (color, depth) = march_ray_unified(
                pos,
                ray,
                world,
                max_dist,
                step,
                AgentSlice::Grid {
                    grid: agent_grid,
                    all: all_positions,
                    self_index,
                },
            );

            let idx = (row * w + col) as usize;
            let ci = idx * 4;
            vf.color[ci] = color[0];
            vf.color[ci + 1] = color[1];
            vf.color[ci + 2] = color[2];
            vf.color[ci + 3] = color[3];
            vf.depth[idx] = depth / max_dist;
        }
    }
}

/// Touch detection using shared positions slice and spatial agent grid.
fn detect_touch_positions(
    agent: &AgentBody,
    world: &WorldState,
    all_positions: &[(Vec3, bool)],
    self_index: usize,
    agent_grid: &crate::world::spatial::AgentGrid,
    contacts: &mut Vec<TouchContact>,
) {
    detect_touch(agent, world, contacts);

    let agent_touch_range = 5.0;
    let pos = agent.body.position;
    for j in agent_grid.query_nearby(pos.x, pos.z) {
        if j == self_index {
            continue;
        }
        let (other_pos, alive) = all_positions[j];
        if !alive {
            continue;
        }
        let diff = other_pos - pos;
        let dist = diff.length();
        if dist < agent_touch_range && dist > 0.01 {
            contacts.push(TouchContact {
                direction: diff.normalize_or_zero(),
                intensity: 1.0 - (dist / agent_touch_range),
                surface_tag: TOUCH_AGENT,
            });
        }
    }
}
