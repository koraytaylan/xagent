use xagent_shared::{BrainConfig, MotorCommand, SensoryFrame};

use crate::action::ActionSelector;
use crate::capacity::CapacityManager;
use crate::encoder::{EncodedState, SensoryEncoder};
use crate::habituation::SensoryHabituation;
use crate::homeostasis::{HomeostaticMonitor, HomeostaticState};
use crate::memory::PatternMemory;
use crate::motor_fatigue::MotorFatigue;
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
    /// Sensory habituation: mean attenuation across encoded dimensions [0.1, 1.0].
    pub mean_attenuation: f32,
    /// Curiosity bonus from sensory monotony [0.0, 0.4].
    pub curiosity_bonus: f32,
    /// Motor fatigue factor [0.2, 1.0]. Low = fatigued.
    pub fatigue_factor: f32,
    /// Recent motor output variance (higher = more diverse).
    pub motor_variance: f32,
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

/// Per-tick snapshot of the brain's decision-making process.
/// Captured after each tick for the decision stream UI.
#[derive(Clone, Debug)]
pub struct DecisionSnapshot {
    /// Tick number when this decision was made.
    pub tick: u64,
    /// Motor output: forward thrust [-1, 1].
    pub motor_forward: f32,
    /// Motor output: turn rate [-1, 1].
    pub motor_turn: f32,
    /// Current exploration rate [0.1, 0.85].
    pub exploration_rate: f32,
    /// Homeostatic gradient (composite, smoothed).
    pub gradient: f32,
    /// Raw per-tick gradient (unsmoothed, used for credit).
    pub raw_gradient: f32,
    /// Urgency/distress level [0, 10].
    pub urgency: f32,
    /// Prediction error this tick.
    pub prediction_error: f32,
    /// Number of patterns recalled this tick.
    pub patterns_recalled: usize,
    /// Total credit magnitude applied this tick (sum of |credit| across all history entries).
    pub credit_magnitude: f32,
    /// Current energy level [0, 1].
    pub energy: f32,
    /// Current integrity level [0, 1].
    pub integrity: f32,
    /// Behavior phase label.
    pub phase: &'static str,
    /// Whether the agent is alive.
    pub alive: bool,
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
    pub habituation: SensoryHabituation,
    pub motor_fatigue: MotorFatigue,
    tick_count: u64,
    /// Latest telemetry snapshot.
    last_telemetry: BrainTelemetry,
    /// Latest decision snapshot for external observation.
    last_decision: Option<DecisionSnapshot>,
}

impl Brain {
    /// Create a new brain with the given capacity configuration.
    ///
    /// Initializes all subsystems (encoder, memory, predictor, action selector,
    /// homeostasis, capacity manager) with parameters derived from the config.
    pub fn new(config: BrainConfig) -> Self {
        let mut encoder = SensoryEncoder::new(config.representation_dim, config.visual_encoding_size);
        let memory = PatternMemory::new(config.memory_capacity, config.representation_dim);
        let predictor = Predictor::new(config.representation_dim);
        let action_selector = ActionSelector::new(config.representation_dim);
        let homeostasis = HomeostaticMonitor::new(config.distress_exponent);
        let capacity = CapacityManager::new(config.processing_slots);

        // Trigger encoder weight initialization with the standard 8×6 frame
        // so feature_count() is available for GPU pipeline setup.
        let blank = xagent_shared::SensoryFrame::new_blank(8, 6);
        let _ = encoder.encode(&blank);

        let repr_dim = config.representation_dim;

        Self {
            config,
            encoder,
            memory,
            predictor,
            action_selector,
            homeostasis,
            capacity,
            habituation: SensoryHabituation::new(repr_dim),
            motor_fatigue: MotorFatigue::new(),
            tick_count: 0,
            last_telemetry: BrainTelemetry::default(),
            last_decision: None,
        }
    }

    /// Process one tick: sensory input → motor output.
    pub fn tick(&mut self, frame: &SensoryFrame) -> MotorCommand {
        let encoded = self.encoder.encode(frame);
        self.tick_inner(frame, encoded, None)
    }

    /// Process one tick using GPU-computed encode and recall results.
    pub fn tick_gpu(
        &mut self,
        frame: &SensoryFrame,
        gpu_encoded: &[f32],
        gpu_similarities: &[f32],
    ) -> MotorCommand {
        let encoded = EncodedState::from_slice(gpu_encoded);
        self.tick_inner(frame, encoded, Some(gpu_similarities))
    }

    /// Shared implementation for both CPU and GPU tick paths.
    fn tick_inner(
        &mut self,
        frame: &SensoryFrame,
        encoded: EncodedState,
        gpu_similarities: Option<&[f32]>,
    ) -> MotorCommand {
        self.tick_count += 1;
        self.memory.advance_tick();

        // 1a. Sensory habituation: attenuate encoded state + compute curiosity
        self.habituation.update(encoded.data());
        let habituated = EncodedState::from_slice(self.habituation.habituated_state());

        // 1b. Compute homeostatic gradient from interoceptive signals
        let homeo_state: HomeostaticState =
            self.homeostasis.update(frame.energy_signal, frame.integrity_signal);

        // 2. Compute prediction error from previous tick's prediction
        let mut scalar_error = 0.0_f32;
        let modulated_lr = self.config.learning_rate * (1.0 + homeo_state.gradient.abs());

        if let Some(prev_prediction) = self.predictor.last_prediction() {
            let prev_prediction = prev_prediction.clone();
            scalar_error = self.predictor.prediction_error(&prev_prediction, &habituated);

            // Learn: update memory reinforcements
            self.memory.learn(&habituated, scalar_error, modulated_lr);

            // Compute error vector into scratch, then learn from it
            // We use the static version to avoid borrow conflict with learn()
            {
                let error_vec = Predictor::prediction_error_vec(&prev_prediction, &habituated);
                self.predictor.learn(&error_vec, prev_prediction.data(), modulated_lr);
            }

            // Learn: adapt encoder weights
            self.encoder.adapt(scalar_error, modulated_lr);

            self.predictor.record_error(scalar_error);
        }

        // 3. Adaptive recall budget based on prediction error
        let (recall_budget, _surprise_budget) =
            self.capacity.allocate_recall_budget_adaptive(scalar_error);

        // 4. Recall relevant patterns with similarity scores (GPU or CPU path)
        let recalled = match gpu_similarities {
            Some(sims) => self.memory.recall_with_gpu_similarities(sims, recall_budget),
            None => self.memory.recall(&habituated, recall_budget),
        };
        self.capacity.report_usage(recalled.len());

        // 5. Predict next state using pre-computed similarity scores
        let prediction = self.predictor.predict_weighted(&habituated, &recalled);

        // 5b. Multi-step rollout for prospective evaluation.
        let confidence = 1.0 - scalar_error.clamp(0.0, 1.0);
        let look_ahead = (confidence * MAX_LOOK_AHEAD as f32) as usize;
        let prospection_prediction = if look_ahead > 1 {
            self.predictor.rollout(&prediction, look_ahead - 1)
        } else {
            prediction.clone()
        };

        // 6. Store current state as a pattern
        self.memory.store(habituated.clone());

        // 7. Decay old patterns
        self.memory.decay(self.config.decay_rate);

        // 8. Select action based on predictions, homeostatic state, and prediction error.
        // Credit assignment receives the RAW per-tick gradient (not EMA composite) so
        // food/damage events produce a sharp, strong signal. The composite gradient is
        // still used for exploration rate and modulated learning rate (its smoothing is
        // appropriate there).
        let curiosity_bonus = self.habituation.curiosity_bonus();
        let command = self.action_selector.select(
            &habituated,
            &prospection_prediction,
            &recalled,
            homeo_state.raw_gradient,
            scalar_error,
            homeo_state.urgency,
            curiosity_bonus,
        );

        // 8b. Credit-driven encoder adaptation: amplify raw features that
        //     contributed to behaviourally relevant encoded dimensions.
        let credit_signal = self.action_selector.last_credit_signal();
        self.encoder.adapt_from_credit(credit_signal, modulated_lr);

        // 8c. Motor fatigue: dampen output when action variance is low.
        self.motor_fatigue.update(command.forward, command.turn);
        let fatigue = self.motor_fatigue.fatigue_factor();
        let command = MotorCommand {
            forward: command.forward * fatigue,
            turn: command.turn * fatigue,
            strafe: command.strafe,
            action: command.action,
        };

        // 9. Record prediction for next tick's error computation
        self.predictor.record_prediction(prediction);

        // 10. Compute behavior quality metrics
        let exploitation_ratio = self.action_selector.exploitation_ratio();
        let exploration_rate = self.action_selector.exploration_rate();
        let decision_quality = (1.0 - scalar_error.clamp(0.0, 1.0))
            * (1.0 - exploration_rate)
            * (1.0 + homeo_state.gradient).clamp(0.0, 2.0)
            / 2.0;

        // 11. Update telemetry
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
            mean_attenuation: self.habituation.mean_attenuation(),
            curiosity_bonus,
            fatigue_factor: self.motor_fatigue.fatigue_factor(),
            motor_variance: self.motor_fatigue.motor_variance(),
        };

        // 12. Capture decision snapshot for UI stream
        self.last_decision = Some(DecisionSnapshot {
            tick: self.tick_count,
            motor_forward: command.forward,
            motor_turn: command.turn,
            exploration_rate,
            gradient: homeo_state.gradient,
            raw_gradient: homeo_state.raw_gradient,
            urgency: homeo_state.urgency,
            prediction_error: scalar_error,
            patterns_recalled: recalled.len(),
            credit_magnitude: self.action_selector.last_credit_magnitude(),
            energy: frame.energy_signal,
            integrity: frame.integrity_signal,
            phase: self.last_telemetry.behavior_phase(),
            alive: true,
        });

        command
    }

    /// Get the latest telemetry snapshot.
    pub fn telemetry(&self) -> &BrainTelemetry {
        &self.last_telemetry
    }

    /// Get the latest decision snapshot (for UI decision stream).
    pub fn last_decision(&self) -> Option<&DecisionSnapshot> {
        self.last_decision.as_ref()
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
    /// led to death, then resets the homeostatic monitor so the respawn's
    /// health jump doesn't produce a false positive gradient.
    pub fn death_signal(&mut self) {
        self.action_selector.death_signal();
        self.homeostasis.reset();
    }

    /// Export the brain's learned state for cross-generation inheritance.
    ///
    /// Within-lifetime learning discovers spatial associations (e.g. "green-left
    /// -> turn left") that are encoded in the encoder and action selector weights.
    /// Without inheritance, this knowledge is discarded every generation and must
    /// be relearned from scratch -- making evolution unable to build on prior
    /// progress. By transferring the learned state to the next generation's
    /// champion, learning accumulates across generations.
    ///
    /// Includes encoder weights (credit-refined perceptual mapping), action
    /// policy weights (behavioural associations), and predictor weights
    /// (temporal model). All three are needed because the policy references
    /// specific encoder output patterns.
    pub fn export_learned_state(&self) -> LearnedState {
        LearnedState {
            encoder_weights: self.encoder.weights_snapshot(),
            encoder_biases: self.encoder.biases().to_vec(),
            action_weights: self.action_selector.export_weights(),
            predictor_weights: self.predictor.export_weights(),
            predictor_context_weight: self.predictor.export_context_weight(),
        }
    }

    /// Import a learned state from a previous generation's best performer.
    ///
    /// Only works when dimensions match (same BrainConfig). If sizes mismatch,
    /// the import is silently skipped and the brain keeps its fresh weights.
    pub fn import_learned_state(&mut self, state: &LearnedState) {
        self.encoder.import_weights(&state.encoder_weights, &state.encoder_biases);
        self.action_selector.import_weights(&state.action_weights);
        self.predictor.import_weights(&state.predictor_weights, state.predictor_context_weight);
    }
}

/// Snapshot of a brain's learned weights for cross-generation inheritance.
///
/// Captures the encoder (perceptual mapping refined by credit-driven Hebbian
/// learning), the behavioral policy (action selector: forward_weights +
/// turn_weights + biases), and temporal model (predictor). All three are
/// needed because the policy references specific encoder output patterns:
/// inheriting only the policy into a fresh encoder would be meaningless if
/// the encoder outputs differ.
///
/// Including the predictor is critical: without inherited predictions, the
/// agent has high prediction error -> high exploration rate (~80%) -> the
/// inherited policy is mostly ignored for the first ~500 ticks.
#[derive(Clone, Debug)]
pub struct LearnedState {
    /// Encoder projection weights, refined by credit-driven Hebbian learning.
    pub encoder_weights: Vec<f32>,
    /// Encoder bias terms.
    pub encoder_biases: Vec<f32>,
    /// Combined continuous motor weights: [forward_weights..., turn_weights..., forward_bias, turn_bias].
    pub action_weights: Vec<f32>,
    pub predictor_weights: Vec<f32>,
    pub predictor_context_weight: f32,
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

    #[test]
    fn encoder_feature_count_available_after_construction() {
        let brain = Brain::new(BrainConfig::default());
        // feature_count must be > 0 immediately after construction
        // (GPU pipeline reads it before the first tick).
        let fc = brain.encoder.feature_count();
        // 8×6 pixels × 4 channels (RGBD) + 9 interoceptive = 201
        assert_eq!(fc, 8 * 6 * 4 + 9, "feature_count should reflect 8×6 RGBD + interoception");
    }

    #[test]
    fn learned_state_roundtrip_preserves_weights() {
        let mut brain = Brain::new(BrainConfig::default());
        // Run a few ticks so the brain has non-trivial learned state
        // (encoder weights get refined by credit-driven adaptation).
        let frame = make_frame(0.8, 0.9);
        for _ in 0..20 {
            brain.tick(&frame);
        }

        let state = brain.export_learned_state();

        // Verify encoder weights are included in the exported state.
        assert!(
            !state.encoder_weights.is_empty(),
            "Exported state must include encoder weights"
        );
        assert!(
            !state.encoder_biases.is_empty(),
            "Exported state must include encoder biases"
        );

        // Create a fresh brain and import the learned state.
        let mut brain2 = Brain::new(BrainConfig::default());
        brain2.import_learned_state(&state);

        // After import, encoder weights and encodings should match.
        assert_eq!(
            brain.encoder.weights_snapshot(),
            brain2.encoder.weights_snapshot(),
            "Encoder weights should match after import"
        );
        let enc1 = brain.encoder.encode(&frame);
        let enc2 = brain2.encoder.encode(&frame);
        assert_eq!(enc1.data(), enc2.data(), "Encoder should produce identical output after import");
    }

    #[test]
    fn learned_state_import_skips_on_dimension_mismatch() {
        let mut brain_32 = Brain::new(BrainConfig { representation_dim: 32, ..BrainConfig::default() });
        brain_32.tick(&make_frame(0.5, 0.5));
        let state_32 = brain_32.export_learned_state();

        // Create a brain with different dimensions
        let mut brain_16 = Brain::new(BrainConfig { representation_dim: 16, ..BrainConfig::default() });
        let frame = make_frame(0.5, 0.5);
        brain_16.tick(&frame);
        let enc_before = brain_16.encoder.encode(&frame);

        // Import from mismatched dimensions — should be silently ignored
        brain_16.import_learned_state(&state_32);
        let enc_after = brain_16.encoder.encode(&frame);

        assert_eq!(
            enc_before.data(), enc_after.data(),
            "Mismatched import should leave brain unchanged"
        );
    }
}
