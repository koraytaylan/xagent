use xagent_shared::{BrainConfig, MotorCommand, SensoryFrame};

use crate::action::ActionSelector;
use crate::capacity::CapacityManager;
use crate::encoder::SensoryEncoder;
use crate::homeostasis::{HomeostaticMonitor, HomeostaticState};
use crate::memory::PatternMemory;
use crate::predictor::Predictor;

/// Maximum lookahead steps for multi-step prediction rollout.
/// At 30 ticks/sec, 10 steps ≈ 1/3 second of foresight — enough to detect
/// trajectory changes (approaching danger) while preserving ~26% of state
/// information. Longer rollouts (30+) contract to a fixed point, producing
/// a constant bias that overrides the reactive policy.
const MAX_LOOK_AHEAD: usize = 10;

/// Per-tick telemetry snapshot for external observation.
#[derive(Clone, Debug, Default)]
pub struct BrainTelemetry {
    /// Current tick number.
    pub tick: u64,
    /// Scalar prediction error this tick (0 = perfect, higher = more surprised).
    pub prediction_error: f32,
    /// Memory utilization [0.0, 1.0].
    pub memory_utilization: f32,
    /// Number of active patterns in memory.
    pub memory_active_count: usize,
    /// Action entropy (high = uniform/exploratory, low = committed).
    pub action_entropy: f32,
    /// Current exploration rate.
    pub exploration_rate: f32,
    /// Homeostatic gradient (positive = improving, negative = worsening).
    pub homeostatic_gradient: f32,
    /// Homeostatic urgency (higher = closer to critical levels).
    pub homeostatic_urgency: f32,
    /// Recall budget allocated this tick.
    pub recall_budget: usize,
    /// Average prediction error over a recent window.
    pub avg_prediction_error: f32,
    /// Fraction of actions that were exploitative (informed) [0.0, 1.0].
    pub exploitation_ratio: f32,
    /// Composite decision quality score [0.0, 1.0].
    pub decision_quality: f32,
}

impl BrainTelemetry {
    /// Behavior phase label based on exploitation ratio.
    pub fn behavior_phase(&self) -> &'static str {
        // Composite score: exploitation alone isn't enough — an agent that
        // avoids danger but starves isn't truly adapted. Factor in urgency
        // (how close to critical homeostatic levels) to penalize agents that
        // haven't solved all survival pressures.
        let stability = 1.0 - self.homeostatic_urgency.clamp(0.0, 1.0);
        let score = self.exploitation_ratio
            * (1.0 - self.prediction_error.clamp(0.0, 1.0))
            * stability;
        if score < 0.02 {
            "RANDOM"
        } else if score < 0.08 {
            "EXPLORING"
        } else if score < 0.20 {
            "LEARNING"
        } else {
            "ADAPTED"
        }
    }
}

/// The cognitive core. Receives sensory input, produces motor output.
///
/// Each tick: encode → recall → predict → compare → learn → select action.
pub struct Brain {
    pub config: BrainConfig,
    pub encoder: SensoryEncoder,
    pub memory: PatternMemory,
    pub predictor: Predictor,
    pub action_selector: ActionSelector,
    pub homeostasis: HomeostaticMonitor,
    pub capacity: CapacityManager,
    tick_count: u64,
    /// Latest telemetry snapshot.
    last_telemetry: BrainTelemetry,
}

impl Brain {
    /// Create a new brain with the given capacity configuration.
    ///
    /// Initializes all subsystems (encoder, memory, predictor, action selector,
    /// homeostasis, capacity manager) with parameters derived from the config.
    pub fn new(config: BrainConfig) -> Self {
        let encoder = SensoryEncoder::new(config.representation_dim, config.visual_encoding_size);
        let memory = PatternMemory::new(config.memory_capacity, config.representation_dim);
        let predictor = Predictor::new(config.representation_dim);
        let action_selector = ActionSelector::new(config.representation_dim);
        let homeostasis = HomeostaticMonitor::new();
        let capacity = CapacityManager::new(config.processing_slots);

        Self {
            config,
            encoder,
            memory,
            predictor,
            action_selector,
            homeostasis,
            capacity,
            tick_count: 0,
            last_telemetry: BrainTelemetry::default(),
        }
    }

    /// Process one tick: sensory input → motor output.
    pub fn tick(&mut self, frame: &SensoryFrame) -> MotorCommand {
        self.tick_count += 1;

        // 1. Encode sensory input into internal representation
        let encoded = self.encoder.encode(frame);

        // 2. Compute homeostatic gradient from interoceptive signals
        let homeo_state: HomeostaticState =
            self.homeostasis.update(frame.energy_signal, frame.integrity_signal);

        // 3. Compute prediction error from previous tick's prediction
        let mut scalar_error = 0.0_f32;

        if let Some(prev_prediction) = self.predictor.last_prediction() {
            scalar_error = self.predictor.prediction_error(&prev_prediction, &encoded);
            let error_vec = Predictor::prediction_error_vec(&prev_prediction, &encoded);

            let modulated_lr = self.config.learning_rate * (1.0 + homeo_state.gradient.abs());

            // Learn: update memory reinforcements
            self.memory.learn(&encoded, scalar_error, modulated_lr);

            // Learn: update predictor weights via gradient descent
            self.predictor.learn(&error_vec, &prev_prediction.data, modulated_lr);

            // Learn: adapt encoder weights
            self.encoder.adapt(scalar_error, modulated_lr);

            self.predictor.record_error(scalar_error);
        }

        // 4. Adaptive recall budget based on prediction error
        let (recall_budget, _surprise_budget) =
            self.capacity.allocate_recall_budget_adaptive(scalar_error);

        // 5. Recall relevant patterns from memory (within capacity budget)
        let recalled = self.memory.recall(&encoded, recall_budget);
        self.capacity.report_usage(recalled.len());

        // 6. Predict next state from current + recalled patterns
        let prediction = self.predictor.predict(&encoded, &recalled);

        // 6b. Multi-step rollout for prospective evaluation.
        // Instead of giving the action selector a 1-tick prediction (barely
        // different from current state), we simulate the agent's trajectory
        // N steps into the future. This turns prediction_error=0 from a
        // useless metric into genuine foresight: the agent can "see" that
        // its current trajectory leads into a danger zone 30 ticks from now.
        let confidence = 1.0 - scalar_error.clamp(0.0, 1.0);
        let look_ahead = (confidence * MAX_LOOK_AHEAD as f32) as usize;
        let prospection_prediction = if look_ahead > 1 {
            self.predictor.rollout(&prediction, look_ahead - 1)
        } else {
            prediction.clone()
        };

        // 7. Store current state as a pattern
        self.memory.store(encoded.clone());

        // 8. Decay old patterns
        self.memory.decay(self.config.decay_rate);

        // 9. Select action based on predictions, homeostatic state, and prediction error
        // Pass the multi-step rollout prediction for prospection (not the 1-step)
        let command = self.action_selector.select(
            &encoded,
            &prospection_prediction,
            &recalled,
            homeo_state.gradient,
            scalar_error,
            homeo_state.urgency,
            &mut self.memory,
        );

        // 10. Record prediction for next tick's error computation
        self.predictor.record_prediction(prediction);

        // 11. Compute behavior quality metrics
        let exploitation_ratio = self.action_selector.exploitation_ratio();
        let exploration_rate = self.action_selector.exploration_rate();
        let decision_quality = (1.0 - scalar_error.clamp(0.0, 1.0))
            * (1.0 - exploration_rate)
            * (1.0 + homeo_state.gradient).clamp(0.0, 2.0)
            / 2.0;

        // 12. Update telemetry
        self.last_telemetry = BrainTelemetry {
            tick: self.tick_count,
            prediction_error: scalar_error,
            memory_utilization: self.memory.utilization(),
            memory_active_count: self.memory.active_count(),
            action_entropy: self.action_selector.action_entropy(),
            exploration_rate,
            homeostatic_gradient: homeo_state.gradient,
            homeostatic_urgency: homeo_state.urgency,
            recall_budget,
            avg_prediction_error: self.predictor.recent_avg_error(32),
            exploitation_ratio,
            decision_quality,
        };

        command
    }

    /// Get the latest telemetry snapshot.
    pub fn telemetry(&self) -> &BrainTelemetry {
        &self.last_telemetry
    }

    /// Current tick count.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Apply death trauma: reduces all memory reinforcements by the given
    /// fraction, modelling the cognitive cost of a catastrophic discontinuity.
    pub fn trauma(&mut self, fraction: f32) {
        self.memory.trauma(fraction);
    }

    /// Signal that the agent has died. Sends a massive negative credit event
    /// to the action selector so it retroactively punishes the actions that
    /// led to death. This is the brain's strongest learning signal.
    pub fn death_signal(&mut self) {
        self.action_selector.death_signal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use xagent_shared::{SensoryFrame, VisualField};

    fn make_frame(energy: f32, integrity: f32) -> SensoryFrame {
        let w = 4;
        let h = 3;
        let pixels = (w * h) as usize;
        SensoryFrame {
            vision: VisualField {
                width: w,
                height: h,
                color: vec![0.5; pixels * 4],
                depth: vec![1.0; pixels],
            },
            velocity: Vec3::new(0.1, 0.0, 0.0),
            facing: Vec3::Z,
            angular_velocity: 0.0,
            energy_signal: energy,
            integrity_signal: integrity,
            energy_delta: 0.0,
            integrity_delta: 0.0,
            touch_contacts: vec![],
            tick: 0,
        }
    }

    #[test]
    fn brain_produces_motor_command() {
        let mut brain = Brain::new(BrainConfig::default());
        let frame = make_frame(0.8, 0.9);
        let cmd = brain.tick(&frame);
        let _ = cmd;
        assert_eq!(brain.tick_count(), 1);
    }

    #[test]
    fn brain_telemetry_updates_each_tick() {
        let mut brain = Brain::new(BrainConfig::default());
        brain.tick(&make_frame(0.8, 0.9));
        assert_eq!(brain.telemetry().tick, 1);
        brain.tick(&make_frame(0.7, 0.85));
        assert_eq!(brain.telemetry().tick, 2);
        assert!(brain.telemetry().memory_active_count > 0);
    }

    #[test]
    fn brain_runs_100_ticks_without_panic() {
        let mut brain = Brain::new(BrainConfig::default());
        for i in 0..100 {
            let energy = 1.0 - (i as f32 * 0.005);
            let integrity = 1.0 - (i as f32 * 0.002);
            brain.tick(&make_frame(energy, integrity));
        }
        let t = brain.telemetry();
        assert_eq!(t.tick, 100);
        assert!(t.memory_utilization > 0.0);
    }

    #[test]
    fn brain_prediction_error_decreases_with_repeated_input() {
        let mut brain = Brain::new(BrainConfig::default());
        let frame = make_frame(0.8, 0.9);

        let mut errors = Vec::new();
        for _ in 0..500 {
            brain.tick(&frame);
            errors.push(brain.telemetry().prediction_error);
        }

        // Skip first few ticks (warmup, first prediction is None → error=0)
        let early_avg: f32 = errors[10..30].iter().sum::<f32>() / 20.0;
        let late_avg: f32 = errors[480..500].iter().sum::<f32>() / 20.0;

        assert!(
            late_avg < early_avg,
            "Prediction error should decrease with repeated input: early={early_avg}, late={late_avg}"
        );
    }

    #[test]
    fn brain_handles_extreme_inputs() {
        let mut brain = Brain::new(BrainConfig::default());

        // All zeros
        let frame_zero = SensoryFrame {
            vision: VisualField {
                width: 4,
                height: 3,
                color: vec![0.0; 48],
                depth: vec![0.0; 12],
            },
            velocity: Vec3::ZERO,
            facing: Vec3::Z,
            angular_velocity: 0.0,
            energy_signal: 0.0,
            integrity_signal: 0.0,
            energy_delta: 0.0,
            integrity_delta: 0.0,
            touch_contacts: vec![],
            tick: 0,
        };
        brain.tick(&frame_zero);
        let t = brain.telemetry();
        assert!(!t.prediction_error.is_nan(), "Should not produce NaN with zero inputs");
        assert!(!t.homeostatic_gradient.is_nan(), "Gradient should not be NaN");

        // All ones
        let frame_one = SensoryFrame {
            vision: VisualField {
                width: 4,
                height: 3,
                color: vec![1.0; 48],
                depth: vec![1.0; 12],
            },
            velocity: Vec3::ONE,
            facing: Vec3::Z,
            angular_velocity: 1.0,
            energy_signal: 1.0,
            integrity_signal: 1.0,
            energy_delta: 1.0,
            integrity_delta: 1.0,
            touch_contacts: vec![],
            tick: 1,
        };
        brain.tick(&frame_one);
        let t = brain.telemetry();
        assert!(!t.prediction_error.is_nan(), "Should not produce NaN with max inputs");
        assert!(!t.homeostatic_gradient.is_nan(), "Gradient should not be NaN");
    }
}
