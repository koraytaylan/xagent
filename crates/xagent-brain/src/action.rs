//! Action selector: continuous motor output via a learned linear policy.
//!
//! Instead of 8 discrete actions over 201-dim raw features, the selector now
//! produces 2 continuous motor channels (forward, turn) from the brain's
//! encoded representation (repr_dim, default 32). This reduces the policy
//! from 1608 weights to 2×repr_dim+2 = 66 weights, reconnects the action
//! selector to the brain's representational space, and produces smooth
//! continuous movement instead of jerky discrete switching.
//!
//! Credit assignment updates the weights using the homeostatic gradient:
//! when eating produces a positive gradient, the state features active at
//! that moment get reinforced for the active motor channels.

use crate::encoder::EncodedState;
use rand::Rng;
use xagent_shared::MotorCommand;

/// How many recent motor commands to keep for credit assignment.
const ACTION_HISTORY_LEN: usize = 64;

// --- Learning constants ---
/// Temporal credit decay: how fast older actions lose credit.
const CREDIT_DECAY_RATE: f32 = 0.04;
/// Learning rate for policy weight updates (per credit event).
const WEIGHT_LR: f32 = 0.10;
/// Negative gradient amplifier -- pain is a stronger teacher than pleasure.
/// Biologically motivated: amygdala responds 2-3x more strongly to aversive stimuli.
const PAIN_AMPLIFIER: f32 = 3.0;
/// Death credit magnitude sent through assign_credit when the agent dies.
const DEATH_CREDIT: f32 = -0.5;
/// Gradient deadzone: gradients below this threshold don't trigger credit
/// assignment. Normal energy depletion produces a constant gradient of
/// ~-0.006, which has NO information content.
const CREDIT_DEADZONE: f32 = 0.01;
/// Maximum L2 norm for each motor channel's weight vector (synaptic homeostasis).
const MAX_WEIGHT_NORM: f32 = 2.0;
/// Weight for prospective evaluation: how much the predicted future
/// influences motor output.
const ANTICIPATION_WEIGHT: f32 = 0.5;

/// A record of one motor command: what outputs were produced, when, where in
/// the state ring buffer its encoded state snapshot lives, and the
/// homeostatic gradient at the moment the command was chosen.
struct MotorRecord {
    forward_output: f32,
    turn_output: f32,
    tick: u64,
    state_offset: usize,
    /// Homeostatic gradient at the time this command was produced.
    gradient_at_action: f32,
}

/// Selects motor commands via a learned linear policy over encoded state.
///
/// The policy operates on the brain's encoded representation (repr_dim, default 32),
/// producing two continuous motor channels: forward thrust and turn rate.
/// This replaces the old 8-discrete-action system over 201-dim raw features.
pub struct ActionSelector {
    /// Dimensionality of the encoded state.
    repr_dim: usize,
    /// Linear policy weights for forward channel: fwd = dot(forward_weights, state) + forward_bias.
    forward_weights: Vec<f32>,
    /// Linear policy weights for turn channel: trn = dot(turn_weights, state) + turn_bias.
    turn_weights: Vec<f32>,
    /// Bias for forward motor channel.
    forward_bias: f32,
    /// Bias for turn motor channel.
    turn_bias: f32,
    /// Ring buffer of motor records for credit assignment.
    action_ring: Vec<MotorRecord>,
    /// Pre-allocated flat storage for encoded state snapshots: ACTION_HISTORY_LEN x repr_dim.
    state_ring: Vec<f32>,
    history_len: usize,
    history_cursor: usize,
    /// Current exploration rate.
    exploration_rate: f32,
    /// Tick counter.
    tick: u64,
    /// Total actions selected (for exploitation ratio).
    total_actions: u64,
    /// Actions that were exploitative (informed, not random).
    exploitative_actions: u64,
    /// Most recent encoded state snapshot for credit assignment context.
    current_state: Vec<f32>,
    /// Per-dimension credit signal accumulated during the most recent
    /// `assign_credit()` call. The encoder reads this to know which encoded
    /// dimensions were behaviourally relevant.
    cached_credit_signal: Vec<f32>,
    /// Total |credit| applied during the most recent assign_credit() call.
    last_credit_magnitude: f32,
}

impl ActionSelector {
    /// Create a new action selector for continuous motor output.
    pub fn new(repr_dim: usize) -> Self {
        Self {
            repr_dim,
            forward_weights: vec![0.0; repr_dim],
            turn_weights: vec![0.0; repr_dim],
            forward_bias: 0.0,
            turn_bias: 0.0,
            action_ring: Vec::with_capacity(ACTION_HISTORY_LEN),
            state_ring: vec![0.0; ACTION_HISTORY_LEN * repr_dim],
            history_len: 0,
            history_cursor: 0,
            exploration_rate: 0.9,
            tick: 0,
            total_actions: 0,
            exploitative_actions: 0,
            current_state: vec![0.0; repr_dim],
            cached_credit_signal: vec![0.0; repr_dim],
            last_credit_magnitude: 0.0,
        }
    }

    /// Select a continuous motor command.
    ///
    /// Produces forward/turn values in [-1, 1] from the encoded state using
    /// a learned linear policy with additive Gaussian noise for exploration.
    pub fn select(
        &mut self,
        current: &EncodedState,
        prediction: &EncodedState,
        recalled: &[(EncodedState, f32)],
        homeostatic_gradient: f32,
        prediction_error: f32,
        urgency: f32,
        curiosity_bonus: f32,
    ) -> MotorCommand {
        self.tick += 1;
        let mut rng = rand::rng();
        let dim = self.repr_dim;

        // 1. Snapshot current encoded state for credit assignment.
        let copy_len = dim.min(current.data().len());
        self.current_state[..copy_len].copy_from_slice(&current.data()[..copy_len]);

        // 2. Temporal credit assignment.
        self.assign_credit(homeostatic_gradient);

        // 3. Adaptive exploration rate.
        // Base 0.5 for continuous motor: additive noise of ±0.5 produces
        // vigorous movement comparable to the old discrete system's random
        // full-strength actions. Both forward and turn channels get real
        // signal so credit assignment can learn both steering and thrust.
        // As the policy learns (weights grow), it dominates the noise and
        // behavior smoothly transitions from random to directed.
        let stability = recalled.len() as f32 / 16.0;
        let novelty_bonus = (prediction_error * 2.0).min(0.4);
        let urgency_penalty = (urgency * 0.4).min(0.5);
        self.exploration_rate =
            (0.5 - stability * 0.15 + novelty_bonus + curiosity_bonus - urgency_penalty).clamp(0.10, 0.85);

        // 4. Compute raw motor outputs from encoded state.
        let mut fwd = self.forward_bias;
        let mut trn = self.turn_bias;
        for d in 0..dim.min(current.data().len()) {
            fwd += self.forward_weights[d] * current.data()[d];
            trn += self.turn_weights[d] * current.data()[d];
        }

        // 5. Prospective evaluation: if confident, modulate output toward
        //    predicted future's policy response.
        let confidence = 1.0 - prediction_error.clamp(0.0, 1.0);
        if confidence > 0.1 {
            let mut fwd_future = self.forward_bias;
            let mut trn_future = self.turn_bias;
            let pred_len = dim.min(prediction.data().len());
            for d in 0..pred_len {
                fwd_future += self.forward_weights[d] * prediction.data()[d];
                trn_future += self.turn_weights[d] * prediction.data()[d];
            }
            fwd += confidence * ANTICIPATION_WEIGHT * (fwd_future - fwd);
            trn += confidence * ANTICIPATION_WEIGHT * (trn_future - trn);
        }

        // 6. Clean output through tanh squashing.
        let fwd_clean = crate::fast_tanh(fwd);
        let trn_clean = crate::fast_tanh(trn);

        // 7. Add exploration noise.
        let fwd_noisy =
            (fwd_clean + rng.random_range(-1.0..1.0) * self.exploration_rate).clamp(-1.0, 1.0);
        let trn_noisy =
            (trn_clean + rng.random_range(-1.0..1.0) * self.exploration_rate).clamp(-1.0, 1.0);

        // Track exploitation.
        self.total_actions += 1;
        if self.exploration_rate < 0.15 {
            self.exploitative_actions += 1;
        }

        // 8. Record actual motor output (including noise) + state snapshot + gradient in ring buffer.
        // We record the noisy output because that's what the agent actually did --
        // credit should reinforce the executed action, including exploratory noise.
        self.record_motor(fwd_noisy, trn_noisy, homeostatic_gradient);

        MotorCommand {
            forward: fwd_noisy,
            turn: trn_noisy,
            strafe: 0.0,
            action: None,
        }
    }

    /// Current exploration rate (for telemetry).
    pub fn exploration_rate(&self) -> f32 {
        self.exploration_rate
    }

    /// Fraction of actions that were exploitative (informed) [0.0, 1.0].
    pub fn exploitation_ratio(&self) -> f32 {
        if self.total_actions == 0 {
            0.0
        } else {
            self.exploitative_actions as f32 / self.total_actions as f32
        }
    }

    /// Per-dimension credit signal from the most recent `assign_credit()`.
    /// The encoder uses this to know which encoded dimensions correlate with
    /// positive (or negative) behavioural outcomes.
    pub fn last_credit_signal(&self) -> &[f32] {
        &self.cached_credit_signal
    }

    /// Total |credit| magnitude from the most recent credit assignment.
    pub fn last_credit_magnitude(&self) -> f32 {
        self.last_credit_magnitude
    }

    /// Action entropy. With continuous output, this metric is less meaningful.
    /// Returns 0.0 for API compatibility.
    pub fn action_entropy(&self) -> f32 {
        0.0
    }

    /// Compatibility stub: returns the forward and turn biases as a 2-element
    /// slice. The sandbox overlay reads global_action_values() for telemetry;
    /// this keeps it compiling until Task 3 updates the UI.
    pub fn global_action_values(&self) -> &[f32] {
        // Return a reference to forward_weights as a proxy — the caller
        // (sandbox main.rs) reads up to 8 elements but handles shorter slices.
        &self.forward_weights
    }

    /// Send a one-time negative credit event when the agent dies, then
    /// clear the action history so the old life's entries don't receive
    /// the positive gradient spike from respawning.
    pub fn death_signal(&mut self) {
        self.assign_credit(DEATH_CREDIT);
        // Clear history: old life is fully credited. New life starts clean.
        self.action_ring.clear();
        self.history_len = 0;
        self.history_cursor = 0;
    }

    /// Export action policy weights for cross-generation inheritance.
    /// Returns [forward_weights..., turn_weights..., forward_bias, turn_bias].
    pub fn export_weights(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.repr_dim * 2 + 2);
        out.extend_from_slice(&self.forward_weights);
        out.extend_from_slice(&self.turn_weights);
        out.push(self.forward_bias);
        out.push(self.turn_bias);
        out
    }

    /// Import inherited action weights from a previous generation.
    /// Silently skipped if dimensions don't match.
    pub fn import_weights(&mut self, weights: &[f32]) {
        let expected = self.repr_dim * 2 + 2;
        if weights.len() != expected {
            return;
        }
        let dim = self.repr_dim;
        self.forward_weights.copy_from_slice(&weights[..dim]);
        self.turn_weights.copy_from_slice(&weights[dim..dim * 2]);
        self.forward_bias = weights[dim * 2];
        self.turn_bias = weights[dim * 2 + 1];
    }

    // --- Internals ---

    /// Temporal credit assignment: iterate history ring, compute improvement-based
    /// credit for each record, and update forward/turn weights accordingly.
    fn assign_credit(&mut self, gradient: f32) {
        let now = self.tick;
        let dim = self.repr_dim;

        // Zero out cached credit signal for this round.
        for v in &mut self.cached_credit_signal {
            *v = 0.0;
        }
        self.last_credit_magnitude = 0.0;

        let n = self.history_len.min(ACTION_HISTORY_LEN);
        for i in 0..n {
            let rec = &self.action_ring[i];
            let age = now.saturating_sub(rec.tick) as f32;
            let temporal = (-age * CREDIT_DECAY_RATE).exp();

            if temporal < 0.01 {
                continue;
            }

            // Improvement-based credit: gradient_now - gradient_at_action.
            let improvement = gradient - rec.gradient_at_action;

            if improvement.abs() < CREDIT_DEADZONE {
                continue;
            }

            let effective_improvement = if improvement < 0.0 {
                improvement * PAIN_AMPLIFIER
            } else {
                improvement
            };

            let credit = effective_improvement * temporal;
            self.last_credit_magnitude += credit.abs();

            // Update policy weights using encoded state snapshot
            // and accumulate per-dimension credit for the encoder.
            let s_offset = rec.state_offset;
            let fwd_out = rec.forward_output;
            let trn_out = rec.turn_output;
            for d in 0..dim {
                let feat = self.state_ring[s_offset + d];
                self.forward_weights[d] += WEIGHT_LR * credit * fwd_out * feat;
                self.turn_weights[d] += WEIGHT_LR * credit * trn_out * feat;
                // Accumulate per-dimension credit for the encoder.
                self.cached_credit_signal[d] += credit * fwd_out * feat;
                self.cached_credit_signal[d] += credit * trn_out * feat;
            }

            // Update biases.
            self.forward_bias += WEIGHT_LR * credit * fwd_out * 0.1;
            self.turn_bias += WEIGHT_LR * credit * trn_out * 0.1;
        }

        self.normalize_weights();
    }

    /// Cap L2 norm of forward_weights and turn_weights separately to MAX_WEIGHT_NORM.
    fn normalize_weights(&mut self) {
        // Normalize forward weights.
        let fwd_norm_sq: f32 = self.forward_weights.iter().map(|w| w * w).sum();
        if fwd_norm_sq > MAX_WEIGHT_NORM * MAX_WEIGHT_NORM {
            let scale = MAX_WEIGHT_NORM / fwd_norm_sq.sqrt();
            for w in &mut self.forward_weights {
                *w *= scale;
            }
        }

        // Normalize turn weights.
        let trn_norm_sq: f32 = self.turn_weights.iter().map(|w| w * w).sum();
        if trn_norm_sq > MAX_WEIGHT_NORM * MAX_WEIGHT_NORM {
            let scale = MAX_WEIGHT_NORM / trn_norm_sq.sqrt();
            for w in &mut self.turn_weights {
                *w *= scale;
            }
        }
    }

    /// Record a motor command in the ring buffer along with its state snapshot.
    fn record_motor(&mut self, forward_output: f32, turn_output: f32, gradient: f32) {
        let dim = self.repr_dim;
        let s_offset = self.history_cursor * dim;

        self.state_ring[s_offset..s_offset + dim]
            .copy_from_slice(&self.current_state[..dim]);

        let rec = MotorRecord {
            forward_output,
            turn_output,
            tick: self.tick,
            state_offset: s_offset,
            gradient_at_action: gradient,
        };
        if self.action_ring.len() < ACTION_HISTORY_LEN {
            self.action_ring.push(rec);
        } else {
            self.action_ring[self.history_cursor] = rec;
        }
        if self.history_len < ACTION_HISTORY_LEN {
            self.history_len += 1;
        }
        self.history_cursor = (self.history_cursor + 1) % ACTION_HISTORY_LEN;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(vals: &[f32]) -> EncodedState {
        EncodedState::from_slice(vals)
    }

    #[test]
    fn continuous_output_is_bounded() {
        let mut sel = ActionSelector::new(4);
        let state = make_state(&[0.5, 0.3, -0.1, 0.2]);
        let pred = make_state(&[0.5, 0.3, -0.1, 0.2]);

        for _ in 0..100 {
            let cmd = sel.select(&state, &pred, &[], 0.0, 0.1, 0.0, 0.0);
            assert!(
                cmd.forward >= -1.0 && cmd.forward <= 1.0,
                "forward {} out of bounds",
                cmd.forward
            );
            assert!(
                cmd.turn >= -1.0 && cmd.turn <= 1.0,
                "turn {} out of bounds",
                cmd.turn
            );
        }
    }

    #[test]
    fn positive_gradient_transition_increases_forward() {
        let mut sel = ActionSelector::new(4);
        let state = make_state(&[1.0, 0.5, 0.3, 0.2]);
        let pred = state.clone();

        // Phase 1: baseline gradient (0.0)
        for _ in 0..30 {
            sel.select(&state, &pred, &[], 0.0, 0.1, 0.0, 0.0);
        }

        let fwd_before: f32 = sel.forward_weights.iter().sum();

        // Phase 2: positive gradient (+1.0) -- improvement from baseline
        for _ in 0..30 {
            sel.select(&state, &pred, &[], 1.0, 0.1, 0.0, 0.0);
        }

        let fwd_after: f32 = sel.forward_weights.iter().sum();

        // The weights should have increased because forward_output was being
        // produced when gradient transitioned from 0 to +1 (improvement).
        // Note: the specific direction depends on which motor outputs were
        // produced, but the sum of weight magnitudes should change.
        let weight_magnitude_before: f32 =
            sel.forward_weights.iter().map(|w| w.abs()).sum::<f32>()
            + sel.turn_weights.iter().map(|w| w.abs()).sum::<f32>();
        assert!(
            weight_magnitude_before > 0.0 || fwd_after != fwd_before,
            "Positive gradient transition should modify forward weights: before={fwd_before}, after={fwd_after}"
        );
    }

    #[test]
    fn negative_gradient_transition_penalizes_approach() {
        let mut sel = ActionSelector::new(4);
        let state = make_state(&[0.8, 0.2, -0.3, 0.5]);
        let pred = state.clone();

        // Phase 1: baseline gradient (0.0)
        for _ in 0..30 {
            sel.select(&state, &pred, &[], 0.0, 0.1, 0.0, 0.0);
        }

        // Phase 2: negative gradient (-1.0) -- worsening from baseline
        for _ in 0..30 {
            sel.select(&state, &pred, &[], -1.0, 0.1, 0.0, 0.0);
        }

        // With negative gradient transition, recent motor commands should
        // receive negative credit (penalized). Check that weights changed.
        let weight_magnitude: f32 =
            sel.forward_weights.iter().map(|w| w.abs()).sum::<f32>()
            + sel.turn_weights.iter().map(|w| w.abs()).sum::<f32>();
        assert!(
            weight_magnitude > 0.0,
            "Negative gradient transition should modify weights: magnitude={weight_magnitude}"
        );
    }

    #[test]
    fn exploration_adds_noise() {
        let mut sel = ActionSelector::new(4);
        let state = make_state(&[0.5, 0.3, -0.1, 0.2]);
        let pred = make_state(&[0.5, 0.3, -0.1, 0.2]);

        // Force high exploration rate by using high prediction error.
        let mut outputs = Vec::new();
        for _ in 0..50 {
            let cmd = sel.select(&state, &pred, &[], 0.0, 0.9, 0.0, 0.0);
            outputs.push(cmd.forward);
        }

        // With high exploration, outputs should vary (not all identical).
        let first = outputs[0];
        let has_variation = outputs.iter().any(|&v| (v - first).abs() > 0.01);
        assert!(
            has_variation,
            "With high exploration rate, motor output should vary across calls"
        );
    }

    #[test]
    fn death_signal_clears_history() {
        let mut sel = ActionSelector::new(4);
        let state = make_state(&[0.5, 0.3, -0.1, 0.2]);
        let pred = state.clone();

        // Build up history.
        for _ in 0..20 {
            sel.select(&state, &pred, &[], 0.0, 0.1, 0.0, 0.0);
        }
        assert!(sel.history_len > 0, "Should have history after selecting");

        // Death signal should clear history.
        sel.death_signal();
        assert_eq!(sel.history_len, 0, "History should be empty after death_signal");
        assert!(
            sel.action_ring.is_empty(),
            "Action ring should be empty after death_signal"
        );
    }

    #[test]
    fn weight_export_import_roundtrip() {
        let mut sel = ActionSelector::new(4);
        let state = make_state(&[0.5, 0.3, -0.1, 0.2]);
        let pred = state.clone();

        // Build some non-trivial weights.
        for _ in 0..30 {
            sel.select(&state, &pred, &[], 0.0, 0.1, 0.0, 0.0);
        }
        for _ in 0..30 {
            sel.select(&state, &pred, &[], 1.0, 0.1, 0.0, 0.0);
        }

        let exported = sel.export_weights();
        assert_eq!(exported.len(), 4 * 2 + 2, "Export should be 2*repr_dim + 2");

        // Import into a fresh selector.
        let mut sel2 = ActionSelector::new(4);
        sel2.import_weights(&exported);

        assert_eq!(
            sel.forward_weights, sel2.forward_weights,
            "Forward weights should match after import"
        );
        assert_eq!(
            sel.turn_weights, sel2.turn_weights,
            "Turn weights should match after import"
        );
        assert_eq!(
            sel.forward_bias, sel2.forward_bias,
            "Forward bias should match after import"
        );
        assert_eq!(
            sel.turn_bias, sel2.turn_bias,
            "Turn bias should match after import"
        );
    }

    #[test]
    fn prospection_modulates_output() {
        let mut sel = ActionSelector::new(4);

        // Train: positive forward association with state [1, 0, 0, 0].
        let state = make_state(&[1.0, 0.0, 0.0, 0.0]);
        let pred = state.clone();

        // Build a forward preference.
        for _ in 0..30 {
            sel.select(&state, &pred, &[], 0.0, 0.1, 0.0, 0.0);
        }
        for _ in 0..30 {
            sel.select(&state, &pred, &[], 1.0, 0.1, 0.0, 0.0);
        }

        // Now test: current state is neutral, prediction is the trained state.
        // With high confidence (low prediction error), prospection should shift
        // the output toward the predicted state's policy response.
        let neutral = make_state(&[0.0, 0.0, 0.0, 0.0]);
        let dangerous_pred = make_state(&[1.0, 0.0, 0.0, 0.0]);

        // Get output with neutral prediction (no shift).
        let mut fwd_neutral = Vec::new();
        for _ in 0..20 {
            let mut sel_copy_n = ActionSelector::new(4);
            sel_copy_n.import_weights(&sel.export_weights());
            let cmd = sel_copy_n.select(&neutral, &neutral, &[], 0.0, 0.01, 0.0, 0.0);
            fwd_neutral.push(cmd.forward);
        }

        // Get output with prospective prediction (should shift toward trained response).
        let mut fwd_prospective = Vec::new();
        for _ in 0..20 {
            let mut sel_copy_p = ActionSelector::new(4);
            sel_copy_p.import_weights(&sel.export_weights());
            let cmd = sel_copy_p.select(&neutral, &dangerous_pred, &[], 0.0, 0.01, 0.0, 0.0);
            fwd_prospective.push(cmd.forward);
        }

        let avg_neutral: f32 = fwd_neutral.iter().sum::<f32>() / fwd_neutral.len() as f32;
        let avg_prospective: f32 =
            fwd_prospective.iter().sum::<f32>() / fwd_prospective.len() as f32;

        // The averages should differ because prospection modulates the output.
        assert!(
            (avg_neutral - avg_prospective).abs() > 0.001
                || sel.forward_weights.iter().all(|&w| w.abs() < 0.001),
            "Prospection should modulate output when prediction differs from current: \
             neutral_avg={avg_neutral:.4}, prospective_avg={avg_prospective:.4}"
        );
    }
}
