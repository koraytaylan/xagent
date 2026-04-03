pub mod senses;

use crate::momentum::MutationMomentum;
use crate::renderer::Vertex;
use crate::world::Mesh;
use glam::Vec3;
use rand::Rng;
use xagent_brain::Brain;

/// Upper bound for memory_capacity to prevent GPU buffer overflow.
/// Large preset uses 512; 2048 gives ~4x evolutionary headroom.
const MAX_MEMORY_CAPACITY: usize = 2048;

/// Upper bound for processing_slots to keep recall cost bounded.
/// Large preset uses 32; 128 gives ~4x evolutionary headroom.
const MAX_PROCESSING_SLOTS: usize = 128;
use xagent_brain::LearnedState;
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
const MIN_TRAIL_DIST_SQ: f32 = 9.0; // 3.0²

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

// ── Color palette for multi-agent rendering ────────────────────────────

/// Eight-color palette for distinguishing agents in multi-agent mode.
/// Colors are chosen for maximum visual contrast on the terrain.
const AGENT_COLORS: [[f32; 3]; 8] = [
    [0.85, 0.20, 0.20], // red
    [0.20, 0.40, 0.90], // blue
    [0.20, 0.80, 0.25], // green
    [0.90, 0.85, 0.15], // yellow
    [0.70, 0.25, 0.85], // purple
    [0.95, 0.55, 0.10], // orange
    [0.10, 0.80, 0.80], // cyan
    [0.90, 0.40, 0.70], // pink
];

/// Gray color applied to dead agent cubes.
const DEAD_COLOR: [f32; 3] = [0.3, 0.3, 0.3];

/// Get the display color for agent at the given index (wraps around the 8-color palette).
pub fn agent_color(index: usize) -> [f32; 3] {
    AGENT_COLORS[index % AGENT_COLORS.len()]
}

/// Maximum number of agents for performance.
pub const MAX_AGENTS: usize = 100;

/// Ticks an agent must survive before it can reproduce.
pub const REPRODUCTION_THRESHOLD: u64 = 5000;

// ── Agent: bundles body + brain + metadata ─────────────────────────────

/// A complete agent with body, brain, and metadata.
pub struct Agent {
    pub id: u32,
    pub body: AgentBody,
    pub brain: Brain,
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
    /// Rolling history for sidebar sparklines (capped length).
    pub prediction_error_history: std::collections::VecDeque<f32>,
    pub exploration_rate_history: std::collections::VecDeque<f32>,
    pub energy_history: std::collections::VecDeque<f32>,
    pub integrity_history: std::collections::VecDeque<f32>,
    pub fatigue_history: std::collections::VecDeque<f32>,
    /// Ring buffer of recent brain decisions for the decision stream UI.
    pub decision_log: std::collections::VecDeque<xagent_brain::DecisionSnapshot>,
    /// Pre-allocated sensory frame buffer, reused each tick to avoid heap churn.
    pub cached_frame: SensoryFrame,
}

impl Agent {
    /// Create a new agent with a fresh brain, assigned color, and zero death count.
    pub fn new(id: u32, position: Vec3, config: BrainConfig, tick: u64) -> Self {
        Self {
            id,
            body: AgentBody::new(position),
            brain: Brain::new(config),
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
            prediction_error_history: std::collections::VecDeque::with_capacity(128),
            exploration_rate_history: std::collections::VecDeque::with_capacity(128),
            energy_history: std::collections::VecDeque::with_capacity(128),
            integrity_history: std::collections::VecDeque::with_capacity(128),
            fatigue_history: std::collections::VecDeque::with_capacity(128),
            decision_log: std::collections::VecDeque::with_capacity(256),
            cached_frame: SensoryFrame::new_blank(8, 6),
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

    /// Number of unique heatmap cells visited (non-zero entries).
    pub fn unique_cells_explored(&self) -> u32 {
        self.heatmap.iter().filter(|&&c| c > 0).count() as u32
    }

    /// Reset per-generation stats for evolution. Keeps BrainConfig (the "genome")
    /// but zeroes cumulative fitness counters and resets the body/brain.
    pub fn reset_for_new_life(&mut self, position: Vec3, tick: u64) {
        self.body = AgentBody::new(position);
        self.brain = Brain::new(self.brain.config.clone());
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
        self.decision_log.clear();
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
        memory_capacity: momentum.biased_perturb_u(&mut rng, parent.memory_capacity, "memory_capacity", strength).min(MAX_MEMORY_CAPACITY),
        processing_slots: momentum.biased_perturb_u(&mut rng, parent.processing_slots, "processing_slots", strength).min(MAX_PROCESSING_SLOTS),
        visual_encoding_size: parent.visual_encoding_size,
        representation_dim: parent.representation_dim,
        learning_rate: momentum.biased_perturb_f(&mut rng, parent.learning_rate, "learning_rate", strength),
        decay_rate: momentum.biased_perturb_f(&mut rng, parent.decay_rate, "decay_rate", strength),
        distress_exponent: momentum.biased_perturb_f(&mut rng, parent.distress_exponent, "distress_exponent", strength).clamp(1.5, 5.0),
        habituation_sensitivity: momentum.biased_perturb_f(&mut rng, parent.habituation_sensitivity, "habituation_sensitivity", strength).clamp(5.0, 50.0),
        max_curiosity_bonus: momentum.biased_perturb_f(&mut rng, parent.max_curiosity_bonus, "max_curiosity_bonus", strength).clamp(0.1, 1.0),
        fatigue_recovery_sensitivity: momentum.biased_perturb_f(&mut rng, parent.fatigue_recovery_sensitivity, "fatigue_recovery_sensitivity", strength).clamp(2.0, 20.0),
        fatigue_floor: momentum.biased_perturb_f(&mut rng, parent.fatigue_floor, "fatigue_floor", strength).clamp(0.05, 0.4),
    }
}

/// Perturb inherited weights for neuroevolution.
/// Mutates a random 10% of action weights by +/-strength, and 1% of encoder
/// weights by +/-(strength * 0.1). This lets evolution explore behavioral
/// variations that within-lifetime learning might miss.
pub fn mutate_learned_state(state: &LearnedState, strength: f32) -> LearnedState {
    let mut rng = rand::rng();
    let mut result = state.clone();

    // Perturb action weights (forward + turn + biases)
    for w in result.action_weights.iter_mut() {
        if rng.random::<f32>() < 0.1 {
            *w += rng.random_range(-strength..strength);
        }
    }

    // Perturb encoder weights more conservatively
    for w in result.encoder_weights.iter_mut() {
        if rng.random::<f32>() < 0.01 {
            *w += rng.random_range(-strength * 0.1..strength * 0.1);
        }
    }

    result
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
        representation_dim: a.representation_dim,
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
        fatigue_recovery_sensitivity: if rng.random::<f32>() < 0.5 {
            a.fatigue_recovery_sensitivity
        } else {
            b.fatigue_recovery_sensitivity
        },
        fatigue_floor: if rng.random::<f32>() < 0.5 {
            a.fatigue_floor
        } else {
            b.fatigue_floor
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

        vertices.push(Vertex { position: positions[*a], color: col });
        vertices.push(Vertex { position: positions[*b], color: col });
        vertices.push(Vertex { position: positions[*c], color: col });
        vertices.push(Vertex { position: positions[*d], color: col });

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

    #[test]
    fn reset_for_new_life_clears_fatigue_history() {
        let mut agent = Agent::new(0, Vec3::ZERO, BrainConfig::default(), 0);
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
    fn mutate_learned_state_perturbs_weights() {
        let state = LearnedState {
            encoder_weights: vec![1.0; 100],
            encoder_biases: vec![0.0; 10],
            action_weights: vec![0.5; 66],
            predictor_weights: vec![0.0; 100],
            predictor_context_weight: 0.2,
        };
        let mutated = mutate_learned_state(&state, 0.1);
        // Action weights should differ (10% mutation rate)
        assert_ne!(mutated.action_weights, state.action_weights);
        // Predictor weights should be unchanged
        assert_eq!(mutated.predictor_weights, state.predictor_weights);
        // Predictor context weight should be unchanged
        assert_eq!(mutated.predictor_context_weight, state.predictor_context_weight);
        // Encoder biases should be unchanged
        assert_eq!(mutated.encoder_biases, state.encoder_biases);
    }
}
