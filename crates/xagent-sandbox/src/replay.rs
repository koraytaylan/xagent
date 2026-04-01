//! Per-generation replay recording and playback.
//!
//! Records minimal per-tick state (position, rotation, vitals, motor, decision data)
//! during simulation. After a generation completes, the recording can be played back
//! through the agent detail UI.

/// Per-tick per-agent state snapshot for replay.
#[derive(Clone, Debug)]
pub struct TickRecord {
    pub position: [f32; 3],
    pub yaw: f32,
    pub alive: bool,
    pub energy: f32,
    pub integrity: f32,
    pub motor_forward: f32,
    pub motor_turn: f32,
    pub exploration_rate: f32,
    pub prediction_error: f32,
    pub gradient: f32,
    pub raw_gradient: f32,
    pub urgency: f32,
    pub credit_magnitude: f32,
    pub patterns_recalled: u16,
    pub phase: u8, // 0=RANDOM, 1=EXPLORING, 2=LEARNING, 3=ADAPTED
    /// Sensory habituation: mean attenuation [0.1, 1.0].
    pub mean_attenuation: f32,
    /// Curiosity bonus from sensory monotony.
    pub curiosity_bonus: f32,
    /// Motor fatigue factor [fatigue_floor, 1.0]. Low = fatigued.
    pub fatigue_factor: f32,
    /// Motor output variance.
    pub motor_variance: f32,
    /// Vision color data (8*6*4 = 192 f32 values). Only stored at keyframes
    /// (every VISION_KEYFRAME_INTERVAL ticks) to save memory.
    pub vision_color: Option<Vec<f32>>,
}

/// Food state event (sparse, only when food is consumed or respawned).
#[derive(Clone, Debug)]
pub struct FoodEvent {
    pub tick: u64,
    pub food_index: usize,
    pub consumed: bool,
    /// New position after respawn (only set when consumed=false, i.e., respawn).
    pub new_position: Option<[f32; 3]>,
}

/// Interval (in ticks) at which vision data is stored.
pub const VISION_KEYFRAME_INTERVAL: u64 = 30;

/// Complete recording of one generation's simulation.
pub struct GenerationRecording {
    pub generation: u32,
    pub total_ticks: u64,
    pub agent_count: usize,
    /// (agent_id, color) for each agent slot.
    pub agent_info: Vec<(u32, [f32; 3])>,
    /// Flat array indexed as `tick_records[tick * agent_count + agent_idx]`.
    pub tick_records: Vec<TickRecord>,
    /// Sparse food events (consumption + respawn).
    pub food_events: Vec<FoodEvent>,
    /// Food positions at tick 0 (for reconstruction).
    pub initial_food_positions: Vec<[f32; 3]>,
}

impl GenerationRecording {
    /// Create a new recording for the given agent population.
    pub fn new(
        generation: u32,
        agents: &[(u32, [f32; 3])],
        initial_food: &[[f32; 3]],
        estimated_ticks: usize,
    ) -> Self {
        let agent_count = agents.len();
        Self {
            generation,
            total_ticks: 0,
            agent_count,
            agent_info: agents.to_vec(),
            tick_records: Vec::with_capacity(estimated_ticks * agent_count),
            food_events: Vec::new(),
            initial_food_positions: initial_food.to_vec(),
        }
    }

    /// Record one tick of data for all agents.
    pub fn record_tick(&mut self, tick: u64, records: &[TickRecord]) {
        debug_assert_eq!(records.len(), self.agent_count);
        self.tick_records.extend_from_slice(records);
        self.total_ticks = tick + 1;
    }

    /// Record a food event.
    pub fn record_food_event(&mut self, event: FoodEvent) {
        self.food_events.push(event);
    }

    /// Get the TickRecord for a specific agent at a specific tick.
    pub fn get(&self, tick: u64, agent_idx: usize) -> Option<&TickRecord> {
        if tick >= self.total_ticks || agent_idx >= self.agent_count {
            return None;
        }
        let idx = tick as usize * self.agent_count + agent_idx;
        self.tick_records.get(idx)
    }

    /// Get all agents' records at a specific tick.
    pub fn get_tick(&self, tick: u64) -> Option<&[TickRecord]> {
        if tick >= self.total_ticks {
            return None;
        }
        let start = tick as usize * self.agent_count;
        let end = start + self.agent_count;
        if end <= self.tick_records.len() {
            Some(&self.tick_records[start..end])
        } else {
            None
        }
    }

    /// Reconstruct food positions at a given tick by replaying food events.
    pub fn food_at_tick(&self, tick: u64) -> Vec<([f32; 3], bool)> {
        let mut food: Vec<([f32; 3], bool)> = self
            .initial_food_positions
            .iter()
            .map(|&p| (p, false))
            .collect();

        for event in &self.food_events {
            if event.tick > tick {
                break;
            }
            if event.food_index < food.len() {
                food[event.food_index].1 = event.consumed;
                if let Some(pos) = event.new_position {
                    food[event.food_index].0 = pos;
                }
            }
        }

        food
    }

    /// Convert phase u8 back to label.
    pub fn phase_label(phase: u8) -> &'static str {
        match phase {
            0 => "RANDOM",
            1 => "EXPLORING",
            2 => "LEARNING",
            3 => "ADAPTED",
            _ => "?",
        }
    }

    /// Convert phase label to u8 for storage.
    pub fn phase_to_u8(phase: &str) -> u8 {
        match phase {
            "RANDOM" => 0,
            "EXPLORING" => 1,
            "LEARNING" => 2,
            "ADAPTED" => 3,
            _ => 0,
        }
    }
}
