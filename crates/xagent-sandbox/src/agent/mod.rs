pub mod senses;

use crate::momentum::MutationMomentum;
use crate::renderer::Vertex;
use crate::world::Mesh;
use glam::Vec3;
use rand::Rng;

/// Upper bound for memory_capacity to prevent GPU buffer overflow.
/// Large preset uses 512; 2048 gives ~4x evolutionary headroom.
const MAX_MEMORY_CAPACITY: usize = 2048;

/// Upper bound for processing_slots to keep recall cost bounded.
/// Large preset uses 32; 128 gives ~4x evolutionary headroom.
const MAX_PROCESSING_SLOTS: usize = 128;
use xagent_brain::buffers::{
    AgentBrainState, ENCODED_DIMENSION, FIXED_TAIL_SIZE, O_ACTION_FORWARD_WEIGHTS,
    O_ACTION_TURN_WEIGHTS, O_PREDICTOR_CONTEXT_WEIGHT, PREDICTOR_DIMENSION,
};
use xagent_shared::{BodyState, BrainConfig, InternalState, SensoryFrame};

/// Heatmap grid resolution (cells per axis). Covers the world in a
/// `HEATMAP_RES × HEATMAP_RES` grid. Each cell tracks how many ticks
/// an agent spent in that region.
pub const HEATMAP_RES: usize = 64;

/// Maximum trail control points per agent. With distance-based sampling
/// (MIN_TRAIL_DIST ≈ 3 units) this covers extremely long lives.
pub const MAX_TRAIL_POINTS: usize = 4000;

/// Minimum distance (squared) between consecutive trail samples.
/// Only record a new point when the agent has moved at least this far.
pub(crate) const MIN_TRAIL_DIST_SQ: f32 = 9.0; // 3.0²

/// Runtime agent body extending shared [`BodyState`] with simulation bookkeeping.
///
/// Tracks yaw angle for smooth turning, angular velocity for proprioception,
/// and previous energy/integrity values for computing per-tick interoceptive deltas.
pub struct AgentBody {
    pub body: BodyState,
    pub yaw: f32,
    pub angular_velocity: f32,
    prev_energy: f32,
    prev_integrity: f32,
}

impl AgentBody {
    /// Create a new agent body at the given position with default health (100/100).
    /// Starts facing +Z with zero velocity.
    pub fn new(position: Vec3) -> Self {
        let internal = InternalState::new(100.0, 100.0);
        let e = internal.energy;
        let i = internal.integrity;
        Self {
            body: BodyState::new(position, internal),
            yaw: 0.0,
            angular_velocity: 0.0,
            prev_energy: e,
            prev_integrity: i,
        }
    }

    /// Snapshot current energy/integrity so deltas can be computed later.
    pub fn snapshot_internals(&mut self) {
        self.prev_energy = self.body.internal.energy;
        self.prev_integrity = self.body.internal.integrity;
    }

    /// Per-tick change in energy since last snapshot. Positive = gained energy.
    pub fn energy_delta(&self) -> f32 {
        self.body.internal.energy - self.prev_energy
    }

    /// Per-tick change in integrity since last snapshot. Positive = healing.
    pub fn integrity_delta(&self) -> f32 {
        self.body.internal.integrity - self.prev_integrity
    }
}

// ── Color generation for multi-agent rendering ──────────────────────────

/// Golden angle in degrees (~137.508°). Successive multiples of this angle
/// produce maximally spread hues for any population size, avoiding the
/// clustering that a fixed-size palette causes when `agent_count > palette_len`.
const GOLDEN_ANGLE_DEG: f32 = 137.508;

/// Gray color applied to dead agent cubes.
const DEAD_COLOR: [f32; 3] = [0.3, 0.3, 0.3];

/// Convert an HSV color (h in degrees, s/v in 0..1) to sRGB [r, g, b] in 0..1.
/// Inputs are normalized: h is wrapped to 0..360, s and v are clamped to 0..1.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = h.rem_euclid(360.0);
    let s = s.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match h_prime as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r1 + m, g1 + m, b1 + m]
}

/// Deterministic, visually distinct color for the agent at `index`.
/// Uses golden-angle hue stepping so any population size gets well-separated colors.
pub fn agent_color(index: usize) -> [f32; 3] {
    let hue = (index as f32 * GOLDEN_ANGLE_DEG) % 360.0;
    hsv_to_rgb(hue, 0.8, 0.9)
}

/// Maximum number of agents for performance.
pub const MAX_AGENTS: usize = 100;

/// Ticks an agent must survive before it can reproduce.
pub const REPRODUCTION_THRESHOLD: u64 = 5000;

// ── Agent: bundles body + brain + metadata ─────────────────────────────

/// A complete agent with body, brain index, and metadata.
/// The brain itself lives on GPU via GpuKernel; Agent only stores its index
/// and a copy of BrainConfig for metabolic drain and evolution.
pub struct Agent {
    pub id: u32,
    pub body: AgentBody,
    pub brain_idx: u32,
    pub brain_config: BrainConfig,
    pub color: [f32; 3],
    pub birth_tick: u64,
    pub death_count: u32,
    pub generation: u32,
    pub life_start_tick: u64,
    pub longest_life: u64,
    pub respawn_cooldown: u32,
    /// Whether this agent has already reproduced in its current life.
    pub has_reproduced: bool,
    /// Total food items consumed across all lives (cumulative for evolution scoring).
    pub food_consumed: u32,
    /// Total ticks spent alive across all lives (cumulative for evolution scoring).
    pub total_ticks_alive: u64,
    /// Position visit counts for heatmap visualization.
    pub heatmap: Vec<u32>,
    /// Distance-sampled control points for trail visualization (current life only).
    pub trail: Vec<[f32; 3]>,
    /// Set when a new trail point is added; cleared after the mesh is rebuilt.
    pub trail_dirty: bool,
    /// Cached motor command for ticks where the brain is decimated.
    pub cached_motor: xagent_shared::MotorCommand,
    /// Brain telemetry from GPU physics readback.
    pub cached_prediction_error: f32,
    pub cached_exploration_rate: f32,
    pub cached_fatigue_factor: f32,
    /// Rolling history for sidebar sparklines (capped length).
    pub prediction_error_history: std::collections::VecDeque<f32>,
    pub exploration_rate_history: std::collections::VecDeque<f32>,
    pub energy_history: std::collections::VecDeque<f32>,
    pub integrity_history: std::collections::VecDeque<f32>,
    pub fatigue_history: std::collections::VecDeque<f32>,
    /// Pre-allocated sensory frame buffer, reused each tick to avoid heap churn.
    pub cached_frame: SensoryFrame,
    /// Cached brain telemetry from GPU readback (updated periodically for selected agent).
    pub cached_urgency: f32,
    pub cached_gradient: f32,
    pub cached_mean_attenuation: f32,
    pub cached_curiosity_bonus: f32,
    pub cached_staleness: f32,
}

impl Agent {
    /// Create a new agent with an assigned GPU brain index, color, and zero death count.
    pub fn new(id: u32, position: Vec3, brain_idx: u32, config: BrainConfig, tick: u64) -> Self {
        let vision_width = config.vision_width;
        let vision_height = config.vision_height;
        Self {
            id,
            body: AgentBody::new(position),
            brain_idx,
            brain_config: config,
            color: agent_color(id as usize),
            birth_tick: tick,
            death_count: 0,
            generation: 0,
            life_start_tick: tick,
            longest_life: 0,
            respawn_cooldown: 0,
            has_reproduced: false,
            food_consumed: 0,
            total_ticks_alive: 0,
            heatmap: vec![0u32; HEATMAP_RES * HEATMAP_RES],
            trail: Vec::with_capacity(256),
            trail_dirty: false,
            cached_motor: xagent_shared::MotorCommand::default(),
            cached_prediction_error: 0.0,
            cached_exploration_rate: 0.0,
            cached_fatigue_factor: 1.0,
            prediction_error_history: std::collections::VecDeque::with_capacity(128),
            exploration_rate_history: std::collections::VecDeque::with_capacity(128),
            energy_history: std::collections::VecDeque::with_capacity(128),
            integrity_history: std::collections::VecDeque::with_capacity(128),
            fatigue_history: std::collections::VecDeque::with_capacity(128),
            cached_frame: SensoryFrame::new_blank(vision_width, vision_height),
            cached_urgency: 0.0,
            cached_gradient: 0.0,
            cached_mean_attenuation: 0.0,
            cached_curiosity_bonus: 0.0,
            cached_staleness: 0.0,
        }
    }

    /// Age in ticks since last respawn.
    pub fn age(&self, current_tick: u64) -> u64 {
        current_tick.saturating_sub(self.life_start_tick)
    }

    /// Whether the agent has survived long enough to reproduce.
    pub fn can_reproduce(&self, current_tick: u64) -> bool {
        self.body.body.alive && self.age(current_tick) >= REPRODUCTION_THRESHOLD
    }

    /// Record current position into the heatmap grid.
    pub fn record_heatmap(&mut self, world_size: f32) {
        let half = world_size / 2.0;
        let cell = world_size / HEATMAP_RES as f32;
        let cx = ((self.body.body.position.x + half) / cell) as usize;
        let cz = ((self.body.body.position.z + half) / cell) as usize;
        let cx = cx.min(HEATMAP_RES - 1);
        let cz = cz.min(HEATMAP_RES - 1);
        self.heatmap[cz * HEATMAP_RES + cx] = self.heatmap[cz * HEATMAP_RES + cx].saturating_add(1);
    }

    /// Record current position if the agent has moved far enough from the
    /// last sample. Distance-based sampling keeps the trail compact for long lives.
    pub fn record_trail(&mut self) {
        let p = self.body.body.position;
        let pos = [p.x, p.y, p.z];

        if let Some(last) = self.trail.last() {
            let dx = pos[0] - last[0];
            let dy = pos[1] - last[1];
            let dz = pos[2] - last[2];
            if dx * dx + dy * dy + dz * dz < MIN_TRAIL_DIST_SQ {
                return;
            }
        }

        if self.trail.len() < MAX_TRAIL_POINTS {
            self.trail.push(pos);
            self.trail_dirty = true;
        }
    }

    /// Clear trail data (called on death/respawn for a fresh life).
    pub fn reset_trail(&mut self) {
        self.trail.clear();
        self.trail_dirty = true;
    }

    /// Record the life that just ended and start a new life from `current_tick`.
    pub fn record_death_and_restart_life(&mut self, current_tick: u64) {
        // P_TICKS_ALIVE does not increment on the death tick, so exclude that
        // terminal tick in CPU-side lifetime accounting as well.
        let life_duration = current_tick
            .saturating_sub(self.life_start_tick)
            .saturating_sub(1);
        self.longest_life = self.longest_life.max(life_duration);
        self.life_start_tick = current_tick;
        self.reset_trail();
    }

    /// Number of unique heatmap cells visited (non-zero entries).
    pub fn unique_cells_explored(&self) -> u32 {
        self.heatmap.iter().filter(|&&c| c > 0).count() as u32
    }

    /// Reset per-generation stats for evolution. Keeps BrainConfig (the "genome")
    /// but zeroes cumulative fitness counters and resets the body.
    /// Note: brain state reset must be done externally via GpuKernel.
    pub fn reset_for_new_life(&mut self, position: Vec3, tick: u64) {
        self.body = AgentBody::new(position);
        self.death_count = 0;
        self.generation = 0;
        self.life_start_tick = tick;
        self.longest_life = 0;
        self.respawn_cooldown = 0;
        self.has_reproduced = false;
        self.food_consumed = 0;
        self.total_ticks_alive = 0;
        self.heatmap.fill(0);
        self.trail.clear();
        self.trail_dirty = true;
        self.prediction_error_history.clear();
        self.exploration_rate_history.clear();
        self.energy_history.clear();
        self.integrity_history.clear();
        self.fatigue_history.clear();
    }
}

/// Create a mutated BrainConfig from a parent config (±10% per param).
pub fn mutate_config(parent: &BrainConfig) -> BrainConfig {
    mutate_config_with_strength(parent, 0.1, &MutationMomentum::new(0.9))
}

/// Create a mutated BrainConfig with a configurable mutation strength.
/// `strength` controls the perturbation range: e.g. 0.1 → ±10%, 0.3 → ±30%.
pub fn mutate_config_with_strength(
    parent: &BrainConfig,
    strength: f32,
    momentum: &MutationMomentum,
) -> BrainConfig {
    let mut rng = rand::rng();

    BrainConfig {
        memory_capacity: momentum
            .biased_perturb_u(
                &mut rng,
                parent.memory_capacity,
                "memory_capacity",
                strength,
            )
            .min(MAX_MEMORY_CAPACITY),
        processing_slots: momentum
            .biased_perturb_u(
                &mut rng,
                parent.processing_slots,
                "processing_slots",
                strength,
            )
            .min(MAX_PROCESSING_SLOTS),
        visual_encoding_size: parent.visual_encoding_size,
        representation_dimension: parent.representation_dimension,
        learning_rate: momentum.biased_perturb_f(
            &mut rng,
            parent.learning_rate,
            "learning_rate",
            strength,
        ),
        decay_rate: momentum.biased_perturb_f(&mut rng, parent.decay_rate, "decay_rate", strength),
        distress_exponent: momentum
            .biased_perturb_f(
                &mut rng,
                parent.distress_exponent,
                "distress_exponent",
                strength,
            )
            .clamp(1.5, 5.0),
        habituation_sensitivity: momentum
            .biased_perturb_f(
                &mut rng,
                parent.habituation_sensitivity,
                "habituation_sensitivity",
                strength,
            )
            .clamp(5.0, 50.0),
        max_curiosity_bonus: momentum
            .biased_perturb_f(
                &mut rng,
                parent.max_curiosity_bonus,
                "max_curiosity_bonus",
                strength,
            )
            .clamp(0.1, 1.0),
        fatigue_floor: momentum
            .biased_perturb_f(&mut rng, parent.fatigue_floor, "fatigue_floor", strength)
            .clamp(0.05, 0.4),
        vision_width: parent.vision_width,
        vision_height: parent.vision_height,
        brain_tick_stride: parent.brain_tick_stride,
        vision_stride: parent.vision_stride,
        metabolic_rate: parent.metabolic_rate,
        integrity_scale: parent.integrity_scale,
        movement_speed: momentum
            .biased_perturb_f(&mut rng, parent.movement_speed, "movement_speed", strength)
            .clamp(20.0, 100.0),
    }
}

/// Perturb inherited GPU brain state for neuroevolution.
/// Mutates encoder weights (10%), action weights (20%), and predictor weights (5%)
/// with configurable strength. This lets evolution explore behavioral
/// variations that within-lifetime learning might miss.
pub fn mutate_brain_state(state: &AgentBrainState, strength: f32) -> AgentBrainState {
    let mut rng = rand::rng();
    let mut mutated = state.clone();

    // Derive layout from actual state length (supports dynamic vision_width × vision_height).
    // brain_stride = fc * ENCODED_DIMENSION + ENCODED_DIMENSION + ENCODED_DIMENSION*ENCODED_DIMENSION + FIXED_TAIL_SIZE
    let variable_part = state
        .brain_state
        .len()
        .checked_sub(ENCODED_DIMENSION + PREDICTOR_DIMENSION * ENCODED_DIMENSION + FIXED_TAIL_SIZE)
        .expect("brain_state too short for layout");
    assert!(
        variable_part % ENCODED_DIMENSION == 0,
        "brain_state length not aligned to ENCODED_DIMENSION"
    );
    let fc = variable_part / ENCODED_DIMENSION;

    let predictor_weights_offset = fc * ENCODED_DIMENSION + ENCODED_DIMENSION;
    let predictor_context_offset =
        predictor_weights_offset + PREDICTOR_DIMENSION * ENCODED_DIMENSION;
    // Fixed deltas from O_PREDICTOR_CONTEXT_WEIGHT are layout-independent
    let act_fwd =
        predictor_context_offset + (O_ACTION_FORWARD_WEIGHTS - O_PREDICTOR_CONTEXT_WEIGHT);
    let act_turn = predictor_context_offset + (O_ACTION_TURN_WEIGHTS - O_PREDICTOR_CONTEXT_WEIGHT);

    assert!(
        act_turn + ENCODED_DIMENSION <= state.brain_state.len(),
        "computed offsets exceed brain_state bounds"
    );

    // Mutate encoder weights (small perturbation)
    for i in 0..(fc * ENCODED_DIMENSION) {
        if rng.random::<f32>() < 0.1 {
            mutated.brain_state[i] += (rng.random::<f32>() * 2.0 - 1.0) * strength * 0.1;
            mutated.brain_state[i] = mutated.brain_state[i].clamp(-2.0, 2.0);
        }
    }

    // Mutate action weights
    for i in 0..ENCODED_DIMENSION {
        if rng.random::<f32>() < 0.2 {
            mutated.brain_state[act_fwd + i] += (rng.random::<f32>() * 2.0 - 1.0) * strength * 0.2;
            mutated.brain_state[act_turn + i] += (rng.random::<f32>() * 2.0 - 1.0) * strength * 0.2;
        }
    }

    // Mutate predictor weights
    for i in 0..(PREDICTOR_DIMENSION * ENCODED_DIMENSION) {
        if rng.random::<f32>() < 0.05 {
            mutated.brain_state[predictor_weights_offset + i] +=
                (rng.random::<f32>() * 2.0 - 1.0) * strength * 0.1;
            mutated.brain_state[predictor_weights_offset + i] =
                mutated.brain_state[predictor_weights_offset + i].clamp(-3.0, 3.0);
        }
    }

    mutated
}

/// Uniform crossover: randomly pick each parameter from parent A or B.
pub fn crossover_config(a: &BrainConfig, b: &BrainConfig) -> BrainConfig {
    let mut rng = rand::rng();
    BrainConfig {
        memory_capacity: if rng.random::<f32>() < 0.5 {
            a.memory_capacity
        } else {
            b.memory_capacity
        },
        processing_slots: if rng.random::<f32>() < 0.5 {
            a.processing_slots
        } else {
            b.processing_slots
        },
        visual_encoding_size: a.visual_encoding_size,
        representation_dimension: a.representation_dimension,
        learning_rate: if rng.random::<f32>() < 0.5 {
            a.learning_rate
        } else {
            b.learning_rate
        },
        decay_rate: if rng.random::<f32>() < 0.5 {
            a.decay_rate
        } else {
            b.decay_rate
        },
        distress_exponent: if rng.random::<f32>() < 0.5 {
            a.distress_exponent
        } else {
            b.distress_exponent
        },
        habituation_sensitivity: if rng.random::<f32>() < 0.5 {
            a.habituation_sensitivity
        } else {
            b.habituation_sensitivity
        },
        max_curiosity_bonus: if rng.random::<f32>() < 0.5 {
            a.max_curiosity_bonus
        } else {
            b.max_curiosity_bonus
        },
        fatigue_floor: if rng.random::<f32>() < 0.5 {
            a.fatigue_floor
        } else {
            b.fatigue_floor
        },
        vision_width: a.vision_width,
        vision_height: a.vision_height,
        brain_tick_stride: a.brain_tick_stride,
        vision_stride: a.vision_stride,
        metabolic_rate: a.metabolic_rate,
        integrity_scale: a.integrity_scale,
        movement_speed: if rng.random::<f32>() < 0.5 {
            a.movement_speed
        } else {
            b.movement_speed
        },
    }
}

/// Convert a single sRGB channel value to linear space.
/// This ensures colors rendered through an sRGB framebuffer match egui's sRGB display.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Generate a cube mesh at the given position with the given color.
/// Color is expected in sRGB space and is converted to linear for the GPU pipeline.
pub fn generate_agent_mesh(position: Vec3, size: f32, color: [f32; 3]) -> Mesh {
    let linear = [
        srgb_to_linear(color[0]),
        srgb_to_linear(color[1]),
        srgb_to_linear(color[2]),
    ];
    let h = size / 2.0;
    let p = position;

    #[rustfmt::skip]
    let positions: [[f32; 3]; 8] = [
        [p.x - h, p.y - h, p.z + h],
        [p.x + h, p.y - h, p.z + h],
        [p.x + h, p.y + h, p.z + h],
        [p.x - h, p.y + h, p.z + h],
        [p.x - h, p.y - h, p.z - h],
        [p.x + h, p.y - h, p.z - h],
        [p.x + h, p.y + h, p.z - h],
        [p.x - h, p.y + h, p.z - h],
    ];

    let face_colors: [[f32; 3]; 6] = [
        linear,
        [linear[0] * 0.9, linear[1] * 0.9, linear[2] * 0.9],
        [linear[0] * 0.8, linear[1] * 0.8, linear[2] * 0.8],
        [linear[0] * 0.7, linear[1] * 0.7, linear[2] * 0.7],
        [linear[0] * 0.85, linear[1] * 0.85, linear[2] * 0.85],
        [linear[0] * 0.75, linear[1] * 0.75, linear[2] * 0.75],
    ];

    #[rustfmt::skip]
    let face_verts: [(usize, usize, usize, usize); 6] = [
        (0, 1, 2, 3),
        (5, 4, 7, 6),
        (3, 2, 6, 7),
        (4, 5, 1, 0),
        (4, 0, 3, 7),
        (1, 5, 6, 2),
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    for (face_idx, (a, b, c, d)) in face_verts.iter().enumerate() {
        let base = (face_idx * 4) as u32;
        let col = face_colors[face_idx];

        vertices.push(Vertex {
            position: positions[*a],
            color: col,
        });
        vertices.push(Vertex {
            position: positions[*b],
            color: col,
        });
        vertices.push(Vertex {
            position: positions[*c],
            color: col,
        });
        vertices.push(Vertex {
            position: positions[*d],
            color: col,
        });

        indices.push(base);
        indices.push(base + 1);
        indices.push(base + 2);
        indices.push(base);
        indices.push(base + 2);
        indices.push(base + 3);
    }

    Mesh { vertices, indices }
}

/// Generate a combined mesh for all agents (one vertex buffer).
pub fn generate_all_agents_mesh(agents: &[Agent]) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for agent in agents {
        let color = if agent.body.body.alive {
            agent.color
        } else {
            DEAD_COLOR
        };
        let sub = generate_agent_mesh(agent.body.body.position, 2.0, color);
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&sub.vertices);
        for idx in &sub.indices {
            indices.push(base + idx);
        }
    }

    Mesh { vertices, indices }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xagent_brain::buffers::O_ACTION_TURN_WEIGHTS;

    #[test]
    fn reset_for_new_life_clears_fatigue_history() {
        let mut agent = Agent::new(0, Vec3::ZERO, 0, BrainConfig::default(), 0);
        // Simulate accumulated fatigue history from a previous life.
        for i in 0..50 {
            agent.fatigue_history.push_back(1.0 - i as f32 * 0.01);
        }
        assert!(!agent.fatigue_history.is_empty());

        agent.reset_for_new_life(Vec3::new(5.0, 0.0, 5.0), 1000);

        assert!(
            agent.fatigue_history.is_empty(),
            "reset_for_new_life must clear fatigue_history"
        );
    }

    #[test]
    fn mutate_brain_state_perturbs_weights() {
        let mut state = AgentBrainState::new_blank();
        // Set some non-zero action weights
        for i in 0..ENCODED_DIMENSION {
            state.brain_state[O_ACTION_FORWARD_WEIGHTS + i] = 0.5;
            state.brain_state[O_ACTION_TURN_WEIGHTS + i] = 0.5;
        }
        let mutated = mutate_brain_state(&state, 0.1);
        // Action weights should differ (20% mutation rate per weight)
        let fwd_same = (0..ENCODED_DIMENSION).all(|i| {
            mutated.brain_state[O_ACTION_FORWARD_WEIGHTS + i]
                == state.brain_state[O_ACTION_FORWARD_WEIGHTS + i]
        });
        assert!(
            !fwd_same,
            "mutate_brain_state should perturb action weights"
        );
    }

    #[test]
    fn mutate_config_respects_fatigue_bounds() {
        let momentum = MutationMomentum::new(0.9);
        // Push initial config to extreme values to verify the clamps rein them in.
        let parent = BrainConfig {
            fatigue_floor: 0.4,
            ..BrainConfig::default()
        };
        for _ in 0..50 {
            let child = mutate_config_with_strength(&parent, 0.3, &momentum);
            assert!(
                child.fatigue_floor <= 0.4,
                "fatigue_floor must be <= 0.4, got {}",
                child.fatigue_floor,
            );
            assert!(
                child.fatigue_floor >= 0.05,
                "fatigue_floor must be >= 0.05, got {}",
                child.fatigue_floor,
            );
        }
    }

    #[test]
    fn mutate_brain_state_works_with_dynamic_layout() {
        use xagent_brain::BrainLayout;
        // Use non-default vision dimensions to exercise dynamic offset logic
        let layout = BrainLayout::new(6, 4);
        let mut state = AgentBrainState::new_for(layout.brain_stride);
        let dyn_act_fwd = layout.feature_count * ENCODED_DIMENSION
            + ENCODED_DIMENSION
            + PREDICTOR_DIMENSION * ENCODED_DIMENSION
            + (O_ACTION_FORWARD_WEIGHTS - O_PREDICTOR_CONTEXT_WEIGHT);
        for i in 0..ENCODED_DIMENSION {
            state.brain_state[dyn_act_fwd + i] = 0.5;
        }
        // Should not panic — validates checked arithmetic with non-default layout
        let mutated = mutate_brain_state(&state, 0.1);
        assert_eq!(mutated.brain_state.len(), layout.brain_stride);
    }

    #[test]
    #[should_panic(expected = "brain_state too short for layout")]
    fn mutate_brain_state_rejects_short_buffer() {
        let state = AgentBrainState::new_for(10); // way too small
        let _ = mutate_brain_state(&state, 0.1);
    }

    #[test]
    fn mutate_config_respects_movement_speed_bounds() {
        let momentum = MutationMomentum::new(0.9);
        // Push initial config to extreme values to verify clamps.
        let too_fast = BrainConfig {
            movement_speed: 200.0,
            ..BrainConfig::default()
        };
        let too_slow = BrainConfig {
            movement_speed: 1.0,
            ..BrainConfig::default()
        };
        for _ in 0..50 {
            let child = mutate_config_with_strength(&too_fast, 0.3, &momentum);
            assert!(
                child.movement_speed <= 100.0,
                "movement_speed must be <= 100.0, got {}",
                child.movement_speed,
            );
            assert!(
                child.movement_speed >= 20.0,
                "movement_speed must be >= 20.0, got {}",
                child.movement_speed,
            );

            let child = mutate_config_with_strength(&too_slow, 0.3, &momentum);
            assert!(
                child.movement_speed <= 100.0,
                "movement_speed must be <= 100.0, got {}",
                child.movement_speed,
            );
            assert!(
                child.movement_speed >= 20.0,
                "movement_speed must be >= 20.0, got {}",
                child.movement_speed,
            );
        }
    }

    #[test]
    fn record_death_and_restart_life_updates_longest_life() {
        let mut agent = Agent::new(0, Vec3::ZERO, 0, BrainConfig::default(), 10);
        let initial_life_start_tick = agent.life_start_tick;
        agent.trail.push([1.0, 0.0, 1.0]);
        agent.trail_dirty = false;

        agent.record_death_and_restart_life(42);
        let first_life = 42u64
            .saturating_sub(initial_life_start_tick)
            .saturating_sub(1);
        assert_eq!(agent.longest_life, first_life);
        assert_eq!(agent.life_start_tick, 42);
        assert!(agent.trail.is_empty());
        assert!(agent.trail_dirty);

        agent.trail_dirty = false;
        agent.record_death_and_restart_life(55);
        assert_eq!(
            agent.longest_life, first_life,
            "shorter life should not reduce longest_life"
        );
        assert_eq!(agent.life_start_tick, 55);
        assert!(agent.trail_dirty);
    }

    #[test]
    fn hsv_to_rgb_outputs_in_unit_range() {
        let hues = [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0, 359.999];
        for &h in &hues {
            let [r, g, b] = hsv_to_rgb(h, 0.8, 0.9);
            assert!(
                (0.0..=1.0).contains(&r) && (0.0..=1.0).contains(&g) && (0.0..=1.0).contains(&b),
                "RGB out of range for hue {h}: [{r}, {g}, {b}]",
            );
        }
    }

    #[test]
    fn hsv_to_rgb_normalizes_inputs() {
        // Negative hue wraps into 0..360
        let neg = hsv_to_rgb(-90.0, 0.5, 0.5);
        let pos = hsv_to_rgb(270.0, 0.5, 0.5);
        assert_eq!(neg, pos, "negative hue should wrap to equivalent positive");

        // Hue > 360 wraps
        let over = hsv_to_rgb(420.0, 0.5, 0.5);
        let wrapped = hsv_to_rgb(60.0, 0.5, 0.5);
        assert_eq!(over, wrapped, "hue > 360 should wrap");

        // Out-of-range s/v are clamped — result must still be in 0..1
        let [r, g, b] = hsv_to_rgb(180.0, 2.0, -0.5);
        assert!(
            (0.0..=1.0).contains(&r) && (0.0..=1.0).contains(&g) && (0.0..=1.0).contains(&b),
            "clamped s/v should produce valid RGB: [{r}, {g}, {b}]",
        );
    }

    #[test]
    fn first_16_agent_colors_are_distinct() {
        let colors: Vec<[f32; 3]> = (0..16).map(agent_color).collect();
        let eps = 1e-4;
        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                let diff = colors[i]
                    .iter()
                    .zip(colors[j].iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    diff > eps,
                    "agent_color({i}) and agent_color({j}) are too similar: {:?} vs {:?}",
                    colors[i],
                    colors[j],
                );
            }
        }
    }
}
