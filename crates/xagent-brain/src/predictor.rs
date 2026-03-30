//! State predictor: learns temporal transitions via online gradient descent.
//!
//! The predictor maintains a linear transform (with tanh activation) that maps
//! current encoded state → predicted next state. Prediction error is the
//! universal learning signal that drives the entire cognitive architecture.

use crate::encoder::EncodedState;
use crate::memory::PatternMemory;

/// Size of the prediction error history ring buffer.
const ERROR_HISTORY_LEN: usize = 128;

// --- Constants ---
const GRADIENT_CLAMP: f32 = 1.0;
const WEIGHT_CLAMP_RANGE: f32 = 3.0;
const CONTEXT_WEIGHT_LR: f32 = 0.01;
const CONTEXT_WEIGHT_MIN: f32 = 0.05;
const CONTEXT_WEIGHT_MAX: f32 = 0.5;

use crate::fast_tanh;

/// Predicts the next encoded state given the current state and recalled patterns.
///
/// Prediction error — the gap between what was predicted and what actually
/// happened — is the universal learning signal. It drives encoding adaptation,
/// memory reinforcement, and indirectly action selection.
///
/// Weights are updated via online gradient descent each tick so the predictor
/// continuously improves its internal model of state transitions.
pub struct Predictor {
    dim: usize,
    /// Prediction weights: [dim × dim], row-major linear transform.
    weights: Vec<f32>,
    /// Context blending weight (learned).
    context_weight: f32,
    /// Last prediction made (for computing error on next tick).
    last_prediction: Option<EncodedState>,
    /// Cached input to the last predict() call, needed for weight updates.
    last_input: Option<Vec<f32>>,
    /// Ring buffer of recent scalar prediction errors.
    error_history: Vec<f32>,
    /// Write cursor into error_history.
    error_cursor: usize,
    /// Total errors recorded (may exceed ERROR_HISTORY_LEN).
    error_count: u64,
    /// Reusable scratch buffer for prediction output.
    scratch: Vec<f32>,
}

impl Predictor {
    /// Create a new predictor with near-identity weight initialization.
    ///
    /// Weights start as 0.9 on the diagonal (predicting the next state ≈ current state)
    /// with small random off-diagonal noise to break symmetry.
    pub fn new(dim: usize) -> Self {
        // Near-identity with small off-diagonal noise
        let mut weights = vec![0.0; dim * dim];
        let mut rng = rand::rng();
        use rand::Rng;
        for i in 0..dim {
            for j in 0..dim {
                let idx = i * dim + j;
                if i == j {
                    weights[idx] = 0.9;
                } else {
                    weights[idx] = rng.random_range(-0.01..=0.01);
                }
            }
        }

        Self {
            dim,
            weights,
            context_weight: 0.2,
            last_prediction: None,
            last_input: None,
            error_history: vec![0.0; ERROR_HISTORY_LEN],
            error_cursor: 0,
            error_count: 0,
            scratch: vec![0.0; dim],
        }
    }

    /// Predict the next encoded state from current state + similarity-weighted
    /// recalled context.
    pub fn predict_weighted(
        &mut self,
        current: &EncodedState,
        recalled: &[(EncodedState, f32)],
    ) -> EncodedState {
        // Reuse scratch buffer instead of allocating
        for v in self.scratch.iter_mut() {
            *v = 0.0;
        }

        // Apply prediction weights to current state
        for i in 0..self.dim {
            let mut sum = 0.0;
            for j in 0..self.dim {
                sum += current.data()[j] * self.weights[j * self.dim + i];
            }
            self.scratch[i] = sum;
        }

        // Blend in recalled patterns weighted by their similarity (relevance)
        if !recalled.is_empty() {
            let total_sim: f32 = recalled.iter().map(|(_, s)| s.max(0.0)).sum();
            if total_sim > 1e-8 {
                for (state, sim) in recalled {
                    let w = self.context_weight * sim.max(0.0) / total_sim;
                    for i in 0..self.dim {
                        self.scratch[i] += state.data()[i] * w;
                    }
                }
            }
        }

        // Cache the effective input for weight updates
        self.last_input = Some(current.data().to_vec());

        // Normalize with fast tanh approximation
        for val in self.scratch.iter_mut() {
            *val = fast_tanh(*val);
        }

        EncodedState::from_slice(&self.scratch[..self.dim])
    }

    /// Convenience: predict with unweighted recalled states (uniform weight).
    pub fn predict(&mut self, current: &EncodedState, recalled: &[EncodedState]) -> EncodedState {
        let weighted: Vec<(EncodedState, f32)> = recalled
            .iter()
            .map(|s| {
                let sim = PatternMemory::cosine_similarity(current.data(), s.data()).max(0.0);
                (s.clone(), sim)
            })
            .collect();
        self.predict_weighted(current, &weighted)
    }

    /// Compute per-dimension error vector: predicted − actual.
    pub fn prediction_error_vec(predicted: &EncodedState, actual: &EncodedState) -> Vec<f32> {
        predicted
            .data()
            .iter()
            .zip(actual.data().iter())
            .map(|(p, a)| p - a)
            .collect()
    }

    /// Compute scalar prediction error (RMSE).
    pub fn prediction_error(&self, predicted: &EncodedState, actual: &EncodedState) -> f32 {
        let mse: f32 = predicted
            .data()
            .iter()
            .zip(actual.data().iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f32>()
            / self.dim as f32;
        mse.sqrt()
    }

    /// Update prediction weights using online gradient descent.
    ///
    /// Given the error vector (predicted − actual) and the predicted values,
    /// adjust weights to reduce future prediction error. Applies the tanh
    /// derivative and gradient clipping for stability.
    pub fn learn(&mut self, error_vec: &[f32], predicted: &[f32], learning_rate: f32) {
        let input = match &self.last_input {
            Some(inp) => inp.clone(),
            None => return,
        };
        let dim = self.dim;

        for i in 0..dim {
            let err_i = error_vec.get(i).copied().unwrap_or(0.0);
            let pred_i = predicted.get(i).copied().unwrap_or(0.0);
            // tanh'(x) = 1 - tanh(x)²
            let tanh_deriv = 1.0 - pred_i.powi(2);
            for j in 0..dim {
                let grad = (err_i * tanh_deriv * input[j]).clamp(-GRADIENT_CLAMP, GRADIENT_CLAMP);
                let idx = j * dim + i;
                self.weights[idx] -= learning_rate * grad;
                self.weights[idx] = self.weights[idx].clamp(-WEIGHT_CLAMP_RANGE, WEIGHT_CLAMP_RANGE);
            }
        }

        // Also adapt context blending weight
        let error_mag: f32 = error_vec.iter().map(|e| e * e).sum::<f32>().sqrt();
        // If error is high, increase context weight; if low, decrease it
        self.context_weight += learning_rate * CONTEXT_WEIGHT_LR * (error_mag - 0.5);
        self.context_weight = self.context_weight.clamp(CONTEXT_WEIGHT_MIN, CONTEXT_WEIGHT_MAX);
    }

    /// Record scalar prediction error into the history ring buffer.
    pub fn record_error(&mut self, error: f32) {
        self.error_history[self.error_cursor] = error;
        self.error_cursor = (self.error_cursor + 1) % ERROR_HISTORY_LEN;
        self.error_count += 1;
    }

    /// Average prediction error over the last N entries (or all if fewer).
    pub fn recent_avg_error(&self, window: usize) -> f32 {
        let len = self.error_history.len();
        let count = window.min(self.error_count as usize);
        if count == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for i in 0..count {
            let idx = (self.error_cursor + len - 1 - i) % len;
            sum += self.error_history[idx];
        }
        sum / count as f32
    }

    /// Full error history (for capacity manager / telemetry).
    pub fn error_count(&self) -> u64 {
        self.error_count
    }

    /// Store prediction for error computation on next tick.
    pub fn record_prediction(&mut self, prediction: EncodedState) {
        self.last_prediction = Some(prediction);
    }

    /// Retrieve last prediction.
    pub fn last_prediction(&self) -> Option<EncodedState> {
        self.last_prediction.clone()
    }

    /// Pure (stateless) prediction: applies the weight matrix + tanh without
    /// modifying any internal state. Used for multi-step rollout where we
    /// iteratively simulate the agent's trajectory without affecting learning.
    pub fn predict_pure(&self, current: &EncodedState) -> EncodedState {
        let mut output = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let mut sum = 0.0;
            for j in 0..self.dim {
                sum += current.data()[j] * self.weights[j * self.dim + i];
            }
            output[i] = fast_tanh(sum);
        }
        EncodedState::from_slice(&output[..self.dim])
    }

    /// Multi-step rollout: iteratively predict N steps into the future.
    ///
    /// This gives the agent "foresight" — instead of seeing one tick ahead
    /// (where the state barely changes), it can mentally simulate its current
    /// trajectory and detect dangers 30+ ticks away. The predictor applies its
    /// learned transition model repeatedly, so accuracy depends on how well
    /// it has learned the environment dynamics.
    ///
    /// No context patterns are used during rollout — the agent is simulating
    /// a hypothetical solo trajectory, not recalling memories.
    pub fn rollout(&self, initial: &EncodedState, steps: usize) -> EncodedState {
        let mut state = initial.clone();
        for _ in 0..steps {
            state = self.predict_pure(&state);
        }
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(vals: &[f32]) -> EncodedState {
        EncodedState::from_slice(vals)
    }

    #[test]
    fn prediction_error_is_zero_for_identical() {
        let pred = Predictor::new(4);
        let a = make_state(&[0.5, 0.3, -0.1, 0.2]);
        assert!((pred.prediction_error(&a, &a)) < 1e-6);
    }

    #[test]
    fn prediction_error_decreases_with_learning() {
        let dim = 4;
        let mut pred = Predictor::new(dim);

        // Repeated pattern: state A → state B
        let state_a = make_state(&[0.5, 0.3, -0.1, 0.2]);
        let state_b = make_state(&[0.2, -0.1, 0.4, 0.1]);

        let mut errors = Vec::new();
        for _ in 0..50 {
            let predicted = pred.predict(&state_a, &[]);
            let error = pred.prediction_error(&predicted, &state_b);
            let error_vec = Predictor::prediction_error_vec(&predicted, &state_b);
            pred.learn(&error_vec, predicted.data(), 0.05);
            pred.record_error(error);
            errors.push(error);
        }

        // Error should decrease
        let first_5_avg: f32 = errors[..5].iter().sum::<f32>() / 5.0;
        let last_5_avg: f32 = errors[45..].iter().sum::<f32>() / 5.0;
        assert!(
            last_5_avg < first_5_avg,
            "Error should decrease with learning: first={first_5_avg}, last={last_5_avg}"
        );
    }

    #[test]
    fn context_weighting_influences_prediction() {
        let dim = 4;
        let mut pred1 = Predictor::new(dim);
        let mut pred2 = Predictor::new(dim);

        let current = make_state(&[0.5, 0.3, -0.1, 0.2]);
        let context = make_state(&[0.0, 0.0, 1.0, 0.0]);

        // Use separate predictor instances so last_input doesn't get overwritten
        let pred_no_ctx = pred1.predict(&current, &[]);
        let pred_with_ctx = pred2.predict(&current, &[context]);

        // Predictions should differ when context is provided
        let diff: f32 = pred_no_ctx
            .data()
            .iter()
            .zip(pred_with_ctx.data().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.001, "Context should influence prediction, diff={diff}");
    }

    #[test]
    fn error_history_tracks_correctly() {
        let mut pred = Predictor::new(4);
        assert_eq!(pred.error_count(), 0);
        pred.record_error(0.5);
        pred.record_error(0.3);
        assert_eq!(pred.error_count(), 2);
        let avg = pred.recent_avg_error(2);
        assert!((avg - 0.4).abs() < 0.01);
    }

    #[test]
    fn ring_buffer_average_after_wraparound() {
        let mut pred = Predictor::new(4);
        // Record 200 errors to wrap around the 128-element ring buffer
        for i in 0..200 {
            pred.record_error(i as f32 * 0.01);
        }
        // Last 10 errors: i=190..200 → values 1.90, 1.91, ..., 1.99
        let avg = pred.recent_avg_error(10);
        let expected: f32 = (190..200).map(|i| i as f32 * 0.01).sum::<f32>() / 10.0;
        assert!(
            (avg - expected).abs() < 0.01,
            "Average after wraparound should be correct: got {avg}, expected {expected}"
        );
    }

    #[test]
    fn gradient_descent_converges_on_simple_pattern() {
        let dim = 4;
        let mut pred = Predictor::new(dim);
        let state_a = make_state(&[0.5, 0.3, -0.1, 0.2]);
        let state_b = make_state(&[0.2, -0.1, 0.4, 0.1]);

        let mut last_error = f32::MAX;
        for _ in 0..200 {
            let predicted = pred.predict(&state_a, &[]);
            let error = pred.prediction_error(&predicted, &state_b);
            let error_vec = Predictor::prediction_error_vec(&predicted, &state_b);
            pred.learn(&error_vec, predicted.data(), 0.05);
            last_error = error;
        }

        assert!(
            last_error < 0.15,
            "Gradient descent should converge: final error = {last_error}"
        );
    }

    #[test]
    fn predict_pure_matches_predict_output() {
        let dim = 4;
        let mut pred = Predictor::new(dim);
        let state = make_state(&[0.5, 0.3, -0.1, 0.2]);

        // predict() with no context should produce the same output as predict_pure()
        let mutable_result = pred.predict(&state, &[]);
        let pure_result = pred.predict_pure(&state);

        for (a, b) in mutable_result.data().iter().zip(pure_result.data().iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "predict_pure should match predict (no context): {a} vs {b}"
            );
        }
    }

    #[test]
    fn predict_pure_does_not_mutate_state() {
        let dim = 4;
        let pred = Predictor::new(dim);
        let state = make_state(&[0.5, 0.3, -0.1, 0.2]);

        // predict_pure takes &self, so this test is mostly a compile-time check
        // that it doesn't require &mut self
        let _r1 = pred.predict_pure(&state);
        let _r2 = pred.predict_pure(&state);
        assert_eq!(pred.error_count(), 0, "predict_pure should not modify predictor state");
    }

    #[test]
    fn rollout_converges_to_fixed_point() {
        let dim = 4;
        let pred = Predictor::new(dim);
        let state = make_state(&[0.5, 0.3, -0.1, 0.2]);

        // With near-identity weights (0.9 diagonal), rollout contracts toward zero
        let far = pred.rollout(&state, 100);
        let magnitude: f32 = far.data().iter().map(|x| x.abs()).sum();
        let initial_magnitude: f32 = state.data().iter().map(|x| x.abs()).sum();

        assert!(
            magnitude < initial_magnitude,
            "Long rollout should contract (near-identity shrinks): far={magnitude}, initial={initial_magnitude}"
        );
    }

    #[test]
    fn rollout_preserves_learned_dynamics() {
        let dim = 4;
        let mut pred = Predictor::new(dim);
        let state_a = make_state(&[0.5, 0.3, -0.1, 0.2]);
        let state_b = make_state(&[0.2, -0.1, 0.4, 0.1]);

        // Train the predictor: A → B transition
        for _ in 0..200 {
            let predicted = pred.predict(&state_a, &[]);
            let error_vec = Predictor::prediction_error_vec(&predicted, &state_b);
            pred.learn(&error_vec, predicted.data(), 0.05);
        }

        // 1-step rollout from A should be close to B
        let one_step = pred.rollout(&state_a, 1);
        let error_1 = pred.prediction_error(&one_step, &state_b);
        assert!(
            error_1 < 0.2,
            "1-step rollout should approximate learned transition: error={error_1}"
        );

        // Multi-step rollout should continue the trajectory (not crash/NaN)
        let far = pred.rollout(&state_a, 30);
        for val in far.data() {
            assert!(!val.is_nan(), "Rollout should not produce NaN");
            assert!(val.abs() <= 1.0, "Rollout values should stay in tanh range");
        }
    }
}
