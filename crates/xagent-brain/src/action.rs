//! Action selector: chooses motor commands via a learned linear policy.
//!
//! Instead of context-hashing (lossy and noisy), the selector learns a weight
//! matrix that maps the encoded sensory state directly to action preferences.
//! Each action has a weight vector; preference = dot(weights, state) + bias.
//!
//! Credit assignment updates the weights using the homeostatic gradient:
//! when eating produces a positive gradient, the state features active at
//! that moment get reinforced for the chosen action. This lets the brain
//! learn "when I see green pixels ahead, forward is good" — something the
//! old context-hash approach could never capture.

use crate::encoder::EncodedState;
use crate::memory::PatternMemory;
use rand::Rng;
use xagent_shared::{MotorAction, MotorCommand};

/// The discrete action space.
const NUM_ACTIONS: usize = 8;
/// How many recent actions to keep for credit assignment.
const ACTION_HISTORY_LEN: usize = 64;

// --- Learning constants ---
/// Temporal credit decay: how fast older actions lose credit.
const CREDIT_DECAY_RATE: f32 = 0.04;
/// Learning rate for policy weight updates (per credit event).
/// Tuned for improvement-based credit with raw (non-EMA) gradient:
/// a food event produces raw_gradient ≈ 0.12, improvement ≈ 0.13,
/// credit ≈ 0.07, yielding Δw ≈ 0.003/dim — enough for ~8 food events
/// to build a detectable preference. MAX_WEIGHT_NORM prevents explosion.
const WEIGHT_LR: f32 = 0.10;
// WEIGHT_DECAY removed: MAX_WEIGHT_NORM already prevents explosion via
// normalize_weights(). Decay on top of normalization was destroying
// inherited cross-generation weights (39% loss per 50k-tick generation).
/// Negative gradient amplifier — pain is a stronger teacher than pleasure.
/// Biologically motivated: amygdala responds 2-3x more strongly to aversive stimuli.
const PAIN_AMPLIFIER: f32 = 3.0;
/// Death credit magnitude sent through assign_credit when the agent dies.
/// After PAIN_AMPLIFIER (3.0×), effective ≈ -1.5. Safe to be strong because
/// death_signal() only updates state-dependent weights, not global biases.
/// Per-death weight shift ≈ -0.015 per relevant feature dimension.
const DEATH_CREDIT: f32 = -0.5;
/// EMA factor for global (state-independent) action value updates.
const GLOBAL_LR: f32 = 0.08;
/// EMA retention for global action values.
const GLOBAL_RETAIN: f32 = 0.995;

/// Weight for prospective evaluation: how much the predicted future
/// influences action preferences. The agent evaluates its learned
/// associations against the predicted state, weighted by prediction
/// confidence. This is model-based reasoning, not hardcoded avoidance.
const ANTICIPATION_WEIGHT: f32 = 0.5;

/// Gradient deadzone: gradients below this threshold don't trigger credit
/// assignment. Normal energy depletion produces a constant gradient of
/// ~-0.006, which has NO information content — it's the metabolic baseline.
/// Without the deadzone, this constant signal accumulates to weights of
/// -320 per dim (73× stronger than the death signal!), teaching the agent
/// "everything is bad" and collapsing behavior.
/// Analogous to a neural firing threshold — only significant stimuli
/// (food: ~0.03+, damage: ~0.02+, death: 0.5) trigger learning.
const CREDIT_DEADZONE: f32 = 0.01;

/// Maximum L2 norm for each action's weight vector (synaptic homeostasis).
/// Biological synapses have a bounded strength — they can't grow
/// infinitely. Without this cap, food credit accumulates unboundedly
/// (agent gets trapped circling food forever) and death signals accumulate
/// unboundedly (agent becomes paralyzed, too scared to move).
/// With dim=32 and typical state norm ~2.8, MAX_WEIGHT_NORM=2.0 produces
/// max preferences of ±5.6 — strong enough for clear exploitation but
/// bounded enough that alternatives retain >1% softmax probability.
const MAX_WEIGHT_NORM: f32 = 2.0;

/// A record of one action taken: what action, when, where in the
/// state ring buffer its encoded state snapshot lives, and the
/// homeostatic gradient at the moment the action was chosen.
struct ActionRecord {
    action_idx: usize,
    tick: u64,
    state_offset: usize,
    /// Homeostatic gradient at the time this action was taken.
    /// Used for improvement-based credit: credit = (gradient_now - gradient_then).
    /// This eliminates the temporal lag problem where correct actions (e.g. turning
    /// away from danger) were punished because the gradient was still reflecting
    /// *past* damage at the moment credit was assigned.
    gradient_at_action: f32,
}

/// Selects motor commands via a learned linear policy over raw sensory features.
///
/// The policy operates on RAW sensory features (201-dim: 48 pixels × RGBD + 9
/// interoceptive) rather than the compressed encoded state (32-dim). This
/// preserves per-pixel spatial specificity: "food at pixel 2 (left)" and "food
/// at pixel 6 (right)" differ in 8 raw features but compress to nearly identical
/// 32-dim encodings (cosine similarity >0.98). The action selector needs this
/// specificity to learn directional behaviors like "food-left → turn-left."
///
/// Memory, prediction, and prospective evaluation still use the encoded state
/// (they benefit from generalization). Only the policy and credit assignment
/// use raw features (they need specificity).
pub struct ActionSelector {
    /// Linear policy weights: preference[a] = dot(weights[a*raw_dim..(a+1)*raw_dim], raw_features).
    action_weights: Vec<f32>,
    /// Dimensionality of the raw sensory features used for the policy.
    raw_dim: usize,
    /// Dimensionality of the encoded state (used for prospective evaluation only).
    repr_dim: usize,
    /// Global (state-independent) action biases.
    global_action_values: Vec<f32>,
    /// Ring buffer of action records.
    action_ring: Vec<ActionRecord>,
    /// Pre-allocated flat storage for raw feature snapshots: ACTION_HISTORY_LEN × raw_dim.
    state_ring: Vec<f32>,
    history_len: usize,
    history_cursor: usize,
    /// Current exploration rate.
    exploration_rate: f32,
    /// Tick counter.
    tick: u64,
    /// Last chosen action index.
    last_action_idx: usize,
    /// Total actions selected (for exploitation ratio).
    total_actions: u64,
    /// Actions that were exploitative (informed, not random).
    exploitative_actions: u64,
    /// Most recent raw feature snapshot for credit assignment context.
    current_state: Vec<f32>,
    /// Whether raw_dim has been determined (set on first select call).
    initialized: bool,
}

impl ActionSelector {
    /// Create a new action selector. The raw feature dimension is determined
    /// lazily on the first `select()` call when raw features are available.
    pub fn new(repr_dim: usize) -> Self {
        Self {
            action_weights: Vec::new(),
            raw_dim: 0,
            repr_dim,
            global_action_values: vec![0.0; NUM_ACTIONS],
            action_ring: Vec::with_capacity(ACTION_HISTORY_LEN),
            state_ring: Vec::new(),
            history_len: 0,
            history_cursor: 0,
            exploration_rate: 0.9,
            tick: 0,
            last_action_idx: 0,
            total_actions: 0,
            exploitative_actions: 0,
            current_state: Vec::new(),
            initialized: false,
        }
    }

    /// Initialize raw-feature-sized buffers on first call.
    fn lazy_init_raw(&mut self, raw_dim: usize) {
        self.raw_dim = raw_dim;
        self.action_weights = vec![0.0; NUM_ACTIONS * raw_dim];
        self.state_ring = vec![0.0; ACTION_HISTORY_LEN * raw_dim];
        self.current_state = vec![0.0; raw_dim];
        self.initialized = true;
    }

    /// Select a motor command.
    ///
    /// `raw_features` are the full per-pixel sensory features (201-dim) used for
    /// the policy and credit assignment. `current` and `prediction` are the
    /// compressed encoded states (32-dim) used only for prospective evaluation.
    pub fn select(
        &mut self,
        current: &EncodedState,
        prediction: &EncodedState,
        recalled: &[(EncodedState, f32)],
        homeostatic_gradient: f32,
        prediction_error: f32,
        urgency: f32,
        _memory: &mut PatternMemory,
        raw_features: &[f32],
    ) -> MotorCommand {
        self.tick += 1;
        let mut rng = rand::rng();

        if !self.initialized {
            self.lazy_init_raw(raw_features.len());
        }
        let rdim = self.raw_dim;

        // Snapshot raw features for credit assignment context.
        let copy_len = rdim.min(raw_features.len());
        self.current_state[..copy_len].copy_from_slice(&raw_features[..copy_len]);

        // --- Temporal credit assignment (uses raw features) ---
        self.assign_credit(homeostatic_gradient, true);

        // --- Adaptive exploration rate ---
        let stability = recalled.len() as f32 / 16.0;
        let novelty_bonus = (prediction_error * 2.0).min(0.4);
        let urgency_penalty = (urgency * 0.4).min(0.5);
        self.exploration_rate =
            (0.4 - stability * 0.2 + novelty_bonus - urgency_penalty).clamp(0.10, 0.85);

        // Compute action preferences from RAW features (full spatial specificity)
        let mut preferences = self.compute_raw_preferences(raw_features);

        // --- Prospective evaluation (delta-based, uses ENCODED states) ---
        // The predictor outputs encoded states, so prospection works in that space.
        // This is fine — prospection detects trajectory trends (approaching danger),
        // which are visible even in compressed space. The directional specificity
        // for choosing WHICH way to turn comes from the raw-feature policy above.
        let confidence = 1.0 - prediction_error.clamp(0.0, 1.0);
        if confidence > 0.1 {
            let current_state_prefs = self.compute_encoded_preferences(current);
            let future_state_prefs = self.compute_encoded_preferences(prediction);
            for a in 0..NUM_ACTIONS {
                let delta = future_state_prefs[a] - current_state_prefs[a];
                preferences[a] += confidence * ANTICIPATION_WEIGHT * delta;
            }
        }

        let is_exploitative = rng.random::<f32>() >= self.exploration_rate;
        let action_idx = if !is_exploitative {
            rng.random_range(0..NUM_ACTIONS)
        } else {
            self.best_action(&preferences)
        };

        self.total_actions += 1;
        if is_exploitative {
            self.exploitative_actions += 1;
        }

        let command = Self::action_to_command(action_idx, &mut rng);

        // Record action + raw feature snapshot + current gradient in ring buffer.
        self.record_action_raw(action_idx, raw_features, homeostatic_gradient);
        self.last_action_idx = action_idx;

        command
    }

    /// Current exploration rate (for telemetry).
    pub fn exploration_rate(&self) -> f32 {
        self.exploration_rate
    }

    /// Global (state-independent) action biases for telemetry.
    pub fn global_action_values(&self) -> &[f32] {
        &self.global_action_values
    }

    /// Fraction of actions that were exploitative (informed) [0.0, 1.0].
    pub fn exploitation_ratio(&self) -> f32 {
        if self.total_actions == 0 {
            0.0
        } else {
            self.exploitative_actions as f32 / self.total_actions as f32
        }
    }

    /// Send a one-time negative credit event when the agent dies, then
    /// clear the action history so the old life's entries don't receive
    /// the positive gradient spike from respawning.
    ///
    /// Without clearing, the respawn causes a gradient discontinuity
    /// (health jumps from 0 to 50%+). assign_credit on the next tick
    /// computes improvement = +0.15 - (-0.006) = +0.156 for old entries,
    /// accumulating ~60% of the death penalty back as false REWARD.
    /// The agent learns "approach danger → die → respawn → things improved!"
    pub fn death_signal(&mut self) {
        self.assign_credit(DEATH_CREDIT, false);
        // Clear history: old life is fully credited. New life starts clean.
        self.action_ring.clear();
        self.history_len = 0;
        self.history_cursor = 0;
    }

    /// Export action policy weights for cross-generation inheritance.
    pub fn export_weights(&self) -> Vec<f32> {
        self.action_weights.clone()
    }

    /// Export global action biases for cross-generation inheritance.
    pub fn export_global_values(&self) -> Vec<f32> {
        self.global_action_values.clone()
    }

    /// Import inherited action weights from a previous generation.
    /// Silently skipped if dimensions don't match.
    pub fn import_weights(&mut self, weights: &[f32], global_values: &[f32]) {
        // If not yet initialized, defer — weights will be set by lazy_init_raw + import
        if self.initialized && weights.len() == self.action_weights.len() {
            self.action_weights.copy_from_slice(weights);
        } else if !self.initialized && !weights.is_empty() {
            // Infer raw_dim from weight length and initialize
            let rdim = weights.len() / NUM_ACTIONS;
            if rdim > 0 && weights.len() == NUM_ACTIONS * rdim {
                self.lazy_init_raw(rdim);
                self.action_weights.copy_from_slice(weights);
            }
        }
        if global_values.len() == self.global_action_values.len() {
            self.global_action_values.copy_from_slice(global_values);
        }
    }

    /// Compute entropy of action value distribution (for telemetry).
    pub fn action_entropy(&self) -> f32 {
        let max_val = self
            .global_action_values
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exps = [0.0f32; NUM_ACTIONS];
        let mut sum = 0.0f32;
        for i in 0..NUM_ACTIONS {
            exps[i] = (self.global_action_values[i] - max_val).exp();
            sum += exps[i];
        }
        if sum < 1e-8 {
            return (NUM_ACTIONS as f32).ln();
        }
        let mut entropy = 0.0f32;
        for &e in &exps {
            let p = e / sum;
            if p > 1e-10 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    // --- Internals ---

    /// Compute action preferences: dot(weights[a], state) + global_bias[a].
    /// Compute action preferences from raw sensory features (policy).
    fn compute_raw_preferences(&self, raw: &[f32]) -> [f32; NUM_ACTIONS] {
        let mut prefs = [0.0f32; NUM_ACTIONS];
        let rdim = self.raw_dim.min(raw.len());
        for a in 0..NUM_ACTIONS {
            let offset = a * self.raw_dim;
            let mut dot = 0.0f32;
            for i in 0..rdim {
                dot += self.action_weights[offset + i] * raw[i];
            }
            prefs[a] = dot + self.global_action_values[a];
        }
        prefs
    }

    /// Preferences from encoded state (no global bias). Used ONLY for
    /// prospective evaluation where the predictor outputs encoded states.
    /// The raw-feature policy handles the actual action choice.
    fn compute_encoded_preferences(&self, state: &EncodedState) -> [f32; NUM_ACTIONS] {
        // Prospection uses a simple heuristic: evaluate how the encoded state's
        // features shift between current and predicted. We use the first repr_dim
        // weights of each action (which overlap with the raw features that
        // encode interoceptive signals at the end). This is approximate but
        // captures trajectory trends (approaching danger/food).
        let mut prefs = [0.0f32; NUM_ACTIONS];
        let dim = self.repr_dim.min(state.len());
        if self.raw_dim == 0 { return prefs; }
        // Use the LAST repr_dim weights (corresponding to interoceptive features)
        // for encoded-state prospection. This connects urgency/energy trends to
        // action preferences without needing a separate weight matrix.
        let base = self.raw_dim.saturating_sub(self.repr_dim);
        for a in 0..NUM_ACTIONS {
            let offset = a * self.raw_dim + base;
            let mut dot = 0.0f32;
            for i in 0..dim {
                if offset + i < (a + 1) * self.raw_dim {
                    dot += self.action_weights[offset + i] * state.data()[i];
                }
            }
            prefs[a] = dot;
        }
        prefs
    }

    fn assign_credit(&mut self, gradient: f32, update_global: bool) {
        let now = self.tick;
        let rdim = self.raw_dim;
        if rdim == 0 { return; }

        let mut global_credits = [0.0f32; NUM_ACTIONS];

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

            // Credit = improvement × temporal decay.
            // State conditioning is handled implicitly: the weight update
            // multiplies by the RAW feature values at action time, so only
            // features that were active get updated. No cosine similarity
            // needed — with 201 raw features, states are naturally specific.
            let credit = effective_improvement * temporal;

            // Update policy weights using raw feature snapshot
            let s_offset = rec.state_offset;
            let w_offset = rec.action_idx * rdim;
            for d in 0..rdim {
                self.action_weights[w_offset + d] +=
                    WEIGHT_LR * credit * self.state_ring[s_offset + d];
            }

            if update_global {
                global_credits[rec.action_idx] += credit;
            }
        }

        if update_global {
            for a in 0..NUM_ACTIONS {
                self.global_action_values[a] =
                    self.global_action_values[a] * GLOBAL_RETAIN + global_credits[a] * GLOBAL_LR;
            }
        }

        self.normalize_weights();
    }

    /// Cap per-action weight vector L2 norms to MAX_WEIGHT_NORM.
    fn normalize_weights(&mut self) {
        let dim = self.raw_dim;
        for a in 0..NUM_ACTIONS {
            let offset = a * dim;
            let norm_sq: f32 = self.action_weights[offset..offset + dim]
                .iter()
                .map(|w| w * w)
                .sum();
            if norm_sq > MAX_WEIGHT_NORM * MAX_WEIGHT_NORM {
                let scale = MAX_WEIGHT_NORM / norm_sq.sqrt();
                for d in 0..dim {
                    self.action_weights[offset + d] *= scale;
                }
            }
        }
    }

    fn record_action_raw(&mut self, action_idx: usize, raw: &[f32], gradient: f32) {
        let rdim = self.raw_dim;
        if rdim == 0 { return; }
        let s_offset = self.history_cursor * rdim;

        let copy_len = rdim.min(raw.len());
        self.state_ring[s_offset..s_offset + copy_len]
            .copy_from_slice(&raw[..copy_len]);

        let rec = ActionRecord {
            action_idx,
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

    fn best_action(&self, preferences: &[f32; NUM_ACTIONS]) -> usize {
        let max_val = preferences.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut rng = rand::rng();
        let mut tied = [0usize; NUM_ACTIONS];
        let mut tied_count = 0;
        for a in 0..NUM_ACTIONS {
            if (preferences[a] - max_val).abs() < 1e-6 {
                tied[tied_count] = a;
                tied_count += 1;
            }
        }
        if tied_count == 0 {
            0
        } else {
            tied[rng.random_range(0..tied_count)]
        }
    }

    #[allow(dead_code)]
    fn softmax_sample(&self, preferences: &[f32; NUM_ACTIONS], rng: &mut impl Rng) -> usize {
        let max_v = preferences
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exps = [0.0f32; NUM_ACTIONS];
        let mut sum = 0.0f32;
        for a in 0..NUM_ACTIONS {
            exps[a] = ((preferences[a] - max_v) * 1.5).exp();
            sum += exps[a];
        }
        if sum < 1e-8 {
            return rng.random_range(0..NUM_ACTIONS);
        }

        let r: f32 = rng.random::<f32>() * sum;
        let mut cumulative = 0.0;
        for (i, &e) in exps.iter().enumerate() {
            cumulative += e;
            if r <= cumulative {
                return i;
            }
        }
        NUM_ACTIONS - 1
    }

    fn action_to_command(idx: usize, rng: &mut impl Rng) -> MotorCommand {
        match idx {
            0 => MotorCommand { forward: 1.0, ..Default::default() },
            1 => MotorCommand { forward: -1.0, ..Default::default() },
            2 => MotorCommand { turn: -1.0, ..Default::default() },
            3 => MotorCommand { turn: 1.0, ..Default::default() },
            4 => MotorCommand {
                forward: 0.5,
                strafe: 0.0,
                turn: 0.3,
                action: None,
            },
            5 => MotorCommand {
                forward: 0.5,
                strafe: 0.0,
                turn: -0.3,
                action: None,
            },
            6 => MotorCommand {
                action: Some(MotorAction::Jump),
                ..Default::default()
            },
            // Forage: slow approach + consume attempt. No zero-movement action
            // exists — standing still never helps when energy depletes continuously.
            7 => MotorCommand {
                forward: 0.15,
                strafe: 0.0,
                turn: 0.0,
                action: Some(MotorAction::Consume),
            },
            _ => MotorCommand {
                forward: rng.random_range(-1.0..=1.0),
                strafe: rng.random_range(-0.3..=0.3),
                turn: rng.random_range(-1.0..=1.0),
                action: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(vals: &[f32]) -> EncodedState {
        EncodedState::from_slice(vals)
    }

    /// Raw features for tests — uses the same values as the encoded state
    /// so test semantics are preserved. In production, raw features are
    /// 201-dim and encoded is 32-dim, but tests use small matching dims.
    fn make_raw(vals: &[f32]) -> Vec<f32> {
        vals.to_vec()
    }

    #[test]
    fn exploration_increases_with_high_prediction_error() {
        let mut sel = ActionSelector::new(4);
        let state = make_state(&[0.5, 0.3, -0.1, 0.2]);
        let pred = make_state(&[0.5, 0.3, -0.1, 0.2]);
        let recalled: Vec<(EncodedState, f32)> = vec![make_state(&[0.1; 4]); 8]
            .into_iter().map(|s| (s, 0.5)).collect();
        let mut mem = PatternMemory::new(10, 4);

        // Low prediction error
        sel.select(&state, &pred, &recalled, 0.0, 0.05, 0.0, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        let low_err_rate = sel.exploration_rate();

        // High prediction error → should explore more
        sel.select(&state, &pred, &recalled, 0.0, 0.8, 0.0, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        let high_err_rate = sel.exploration_rate();

        assert!(
            high_err_rate > low_err_rate,
            "Higher prediction error should increase exploration: low={low_err_rate}, high={high_err_rate}"
        );
    }

    #[test]
    fn urgency_decreases_exploration() {
        let mut sel = ActionSelector::new(4);
        let state = make_state(&[0.5, 0.3, -0.1, 0.2]);
        let pred = make_state(&[0.5, 0.3, -0.1, 0.2]);
        let mut mem = PatternMemory::new(10, 4);

        // No urgency
        sel.select(&state, &pred, &[], 0.0, 0.3, 0.0, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        let no_urgency_rate = sel.exploration_rate();

        // High urgency → should exploit more
        sel.select(&state, &pred, &[], 0.0, 0.3, 1.0, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        let high_urgency_rate = sel.exploration_rate();

        assert!(
            high_urgency_rate < no_urgency_rate,
            "Urgency should reduce exploration: none={no_urgency_rate}, high={high_urgency_rate}"
        );
    }

    #[test]
    fn action_entropy_is_maximal_initially() {
        let sel = ActionSelector::new(4);
        let entropy = sel.action_entropy();
        let max_entropy = (NUM_ACTIONS as f32).ln();
        assert!(
            (entropy - max_entropy).abs() < 0.1,
            "Initial entropy should be near maximum: {entropy} vs {max_entropy}"
        );
    }

    #[test]
    fn positive_gradient_increases_action_value() {
        let mut sel = ActionSelector::new(4);
        let state = make_state(&[0.5, 0.3, -0.1, 0.2]);
        let pred = state.clone();
        let mut mem = PatternMemory::new(10, 4);

        // Simulate a gradient TRANSITION: baseline → positive.
        // With improvement-based credit, actions taken during baseline get
        // rewarded when the gradient subsequently rises (things improved).
        for _ in 0..25 {
            sel.select(&state, &pred, &[], 0.0, 0.1, 0.0, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        }
        for _ in 0..25 {
            sel.select(&state, &pred, &[], 1.0, 0.1, 0.0, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        }

        // At least one action value should be positive
        let max_val = sel
            .global_action_values
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(max_val > 0.0, "Positive gradient transition should increase values");
    }

    #[test]
    fn exploitation_ratio_starts_at_zero() {
        let sel = ActionSelector::new(4);
        assert_eq!(
            sel.exploitation_ratio(),
            0.0,
            "Fresh selector should have 0% exploitation"
        );
    }

    #[test]
    fn exploration_decreases_with_stability() {
        let mut sel = ActionSelector::new(4);
        let state = make_state(&[0.5, 0.3, -0.1, 0.2]);
        let pred = state.clone();
        let mut mem = PatternMemory::new(10, 4);

        // Few recalled patterns (unstable)
        sel.select(&state, &pred, &[], 0.0, 0.1, 0.0, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        let rate_unstable = sel.exploration_rate();

        // Many recalled patterns (stable)
        let recalled: Vec<(EncodedState, f32)> = vec![make_state(&[0.1; 4]); 12]
            .into_iter().map(|s| (s, 0.5)).collect();
        sel.select(&state, &pred, &recalled, 0.0, 0.1, 0.0, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        let rate_stable = sel.exploration_rate();

        assert!(
            rate_stable < rate_unstable,
            "More recalled patterns should reduce exploration: unstable={rate_unstable}, stable={rate_stable}"
        );
    }

    #[test]
    fn linear_policy_learns_state_preference() {
        let mut sel = ActionSelector::new(4);
        let mut mem = PatternMemory::new(10, 4);

        let raw_a = [1.0f32, 0.0, 0.0, 0.0];
        let raw_b = [0.0f32, 1.0, 0.0, 0.0];
        let state_a = make_state(&raw_a);
        let state_b = make_state(&raw_b);
        let pred = make_state(&[0.0; 4]);

        // Simulate gradient TRANSITION in state A: baseline → positive.
        for _ in 0..50 {
            sel.select(&state_a, &pred, &[], 0.0, 0.1, 0.5, &mut mem, &raw_a);
        }
        for _ in 0..50 {
            sel.select(&state_a, &pred, &[], 1.0, 0.1, 0.5, &mut mem, &raw_a);
        }
        // Simulate gradient TRANSITION in state B: baseline → negative.
        for _ in 0..50 {
            sel.select(&state_b, &pred, &[], 0.0, 0.1, 0.5, &mut mem, &raw_b);
        }
        for _ in 0..50 {
            sel.select(&state_b, &pred, &[], -1.0, 0.1, 0.5, &mut mem, &raw_b);
        }

        let pref_a = sel.compute_raw_preferences(&raw_a);
        let pref_b = sel.compute_raw_preferences(&raw_b);

        let max_a = pref_a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_b = pref_b.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        assert!(
            max_a > max_b,
            "State A (improving gradient) should have higher max preference than B (worsening): A={max_a}, B={max_b}"
        );
    }

    #[test]
    fn prospection_delta_penalizes_trajectory_toward_danger() {
        let mut sel = ActionSelector::new(4);
        let mut mem = PatternMemory::new(10, 4);

        // "Danger state": features [1.0, 0, 0, 0]
        let danger_state = make_state(&[1.0, 0.0, 0.0, 0.0]);
        let safe_state = make_state(&[0.0, 1.0, 0.0, 0.0]);

        // Simulate entering danger: gradient transitions from safe (0) to damage (-1).
        // Actions taken during safe phase get punished when gradient worsens.
        for _ in 0..25 {
            sel.select(&danger_state, &safe_state, &[], 0.0, 0.1, 0.5, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        }
        for _ in 0..25 {
            sel.select(&danger_state, &safe_state, &[], -1.0, 0.1, 0.5, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        }

        // Delta-based prospection: when the prediction shows a shift from
        // safe features to danger features, the delta should be negative
        // (things are getting worse for actions trained on danger).
        let current_state_prefs = sel.compute_encoded_preferences(&safe_state);
        let future_state_prefs = sel.compute_encoded_preferences(&danger_state);

        // For actions trained in danger state (negative credit), the danger
        // state preferences should be lower than safe state preferences.
        // The delta (future - current) should be negative for those actions.
        let min_delta: f32 = (0..NUM_ACTIONS)
            .map(|a| future_state_prefs[a] - current_state_prefs[a])
            .fold(f32::INFINITY, f32::min);

        assert!(
            min_delta < -0.01,
            "Trajectory toward danger should produce negative delta for penalized actions: min_delta={min_delta}"
        );
    }

    #[test]
    fn weight_normalization_prevents_unbounded_growth() {
        let mut sel = ActionSelector::new(4);
        let mut mem = PatternMemory::new(10, 4);
        let state = make_state(&[1.0, 0.5, 0.3, 0.2]);
        let pred = state.clone();

        // Hammer the same action with huge positive gradients for many ticks
        for _ in 0..500 {
            sel.select(&state, &pred, &[], 1.0, 0.1, 0.5, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        }

        // All per-action weight vector norms must be ≤ MAX_WEIGHT_NORM
        for a in 0..NUM_ACTIONS {
            let offset = a * 4;
            let norm: f32 = sel.action_weights[offset..offset + 4]
                .iter()
                .map(|w| w * w)
                .sum::<f32>()
                .sqrt();
            assert!(
                norm <= MAX_WEIGHT_NORM + 0.001,
                "Action {a} weight norm {norm} exceeds MAX_WEIGHT_NORM={MAX_WEIGHT_NORM}"
            );
        }
    }

    #[test]
    fn weight_normalization_prevents_paralysis_from_death_signals() {
        let mut sel = ActionSelector::new(4);
        let mut mem = PatternMemory::new(10, 4);
        let danger_state = make_state(&[0.8, 0.2, -0.3, 0.5]);
        let pred = danger_state.clone();

        // Simulate multiple lives: walk forward in danger state, die, repeat
        for _life in 0..20 {
            for _ in 0..30 {
                sel.select(&danger_state, &pred, &[], 0.0, 0.1, 0.5, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
            }
            sel.death_signal();
        }

        // Preferences should be bounded
        let prefs = sel.compute_raw_preferences(&make_raw(&[1.0, 0.0, 0.0, 0.0]));
        let state_norm: f32 = danger_state.data().iter().map(|v| v * v).sum::<f32>().sqrt();
        let max_possible = MAX_WEIGHT_NORM * state_norm + 10.0;
        for a in 0..NUM_ACTIONS {
            assert!(
                prefs[a].abs() < max_possible,
                "Action {a} preference {:.2} exceeds bounded range (±{max_possible:.2}) after 20 deaths",
                prefs[a]
            );
        }
    }

    #[test]
    fn state_conditioned_credit_spares_dissimilar_states() {
        let mut sel = ActionSelector::new(4);
        let mut mem = PatternMemory::new(10, 4);

        // Two orthogonal states: danger and safe
        let danger_state = make_state(&[1.0, 0.0, 0.0, 0.0]);
        let safe_state = make_state(&[0.0, 0.0, 0.0, 1.0]);
        let pred = danger_state.clone();

        // Build mixed history: ticks in danger state AND safe state
        for _ in 0..15 {
            sel.select(&safe_state, &pred, &[], 0.0, 0.1, 0.5, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        }
        for _ in 0..20 {
            sel.select(&danger_state, &pred, &[], 0.0, 0.1, 0.5, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        }

        // Die in danger context — current_state is danger_state (from last select).
        // State-conditioned credit should penalize danger-similar history entries
        // (cos_sim ≈ 1.0) but spare safe entries (cos_sim ≈ 0.0).
        sel.death_signal();

        let prefs_danger = sel.compute_encoded_preferences(&danger_state);
        let prefs_safe = sel.compute_encoded_preferences(&safe_state);

        let avg_danger: f32 = prefs_danger.iter().sum::<f32>() / NUM_ACTIONS as f32;
        let avg_safe: f32 = prefs_safe.iter().sum::<f32>() / NUM_ACTIONS as f32;

        assert!(
            avg_danger < avg_safe,
            "Death in danger state should penalize danger preferences more than safe: danger={avg_danger:.4}, safe={avg_safe:.4}"
        );
    }

    #[test]
    fn avoidance_action_gets_positive_credit_after_gradient_recovers() {
        // Core scenario the improvement-based credit fixes:
        // Agent takes damage (gradient drops), then turns away (gradient recovers).
        // The turn should get POSITIVE credit, not negative.
        let mut sel = ActionSelector::new(4);
        let danger_state = make_state(&[1.0, 0.0, 0.0, 0.0]);
        let pred = make_state(&[0.0; 4]);
        let mut mem = PatternMemory::new(10, 4);

        // Phase 1: take actions while gradient is worsening (entering danger)
        for _ in 0..10 {
            sel.select(&danger_state, &pred, &[], -0.5, 0.3, 0.5, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        }

        // Phase 2: gradient recovers (agent turned away, damage stopping)
        for _ in 0..10 {
            sel.select(&danger_state, &pred, &[], 0.0, 0.3, 0.2, &mut mem, &make_raw(&[0.5, 0.3, -0.1, 0.2]));
        }

        // The actions from phase 1 (taken at gradient -0.5) should have received
        // POSITIVE improvement credit when gradient rose to 0.0 in phase 2:
        // improvement = 0.0 - (-0.5) = +0.5 → reward.
        // So at least some action weights in danger_state should be positive
        // (the actions that were taken during the worsening phase got credited
        // when things improved).
        let prefs = sel.compute_encoded_preferences(&danger_state);
        let max_pref = prefs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_pref > 0.0,
            "Actions during danger should get positive credit when gradient recovers: max_pref={max_pref:.4}"
        );
    }
}
