//! Per-parameter mutation momentum for directed evolution.
//!
//! Each island maintains its own momentum vector that biases future mutations
//! toward directions that previously improved fitness. Momentum decays each
//! generation so stale signals fade naturally.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use xagent_shared::BrainConfig;

/// Accumulates per-parameter directional signal from successful mutations.
///
/// After each generation, winning offspring (those that beat their parent's
/// fitness) contribute their mutation deltas to the momentum. Future
/// perturbations are biased in the momentum direction — parameters with
/// strong momentum get pushed toward winning values, while parameters with
/// weak momentum stay near random noise.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MutationMomentum {
    /// Per-parameter momentum. Positive = trending upward, negative = trending down.
    momentum: HashMap<String, f32>,
    /// Per-generation decay factor (e.g., 0.9). Applied multiplicatively.
    decay: f32,
}

impl MutationMomentum {
    /// Create a new momentum tracker with the given decay rate.
    pub fn new(decay: f32) -> Self {
        Self {
            momentum: HashMap::new(),
            decay,
        }
    }

    /// Get momentum for a parameter (0.0 if not tracked).
    pub fn get(&self, param: &str) -> f32 {
        self.momentum.get(param).copied().unwrap_or(0.0)
    }

    /// Decay all momentum values by the decay factor.
    pub fn decay_step(&mut self) {
        for v in self.momentum.values_mut() {
            *v *= self.decay;
        }
        // Remove near-zero entries to keep the map clean
        self.momentum.retain(|_, v| v.abs() > 1e-8);
    }

    /// Biased perturbation for f32 parameters.
    ///
    /// Combines a random multiplicative factor (same as the old `perturb_f`)
    /// with an additive momentum nudge that shifts the center of perturbation.
    pub fn biased_perturb_f(
        &self,
        rng: &mut impl Rng,
        value: f32,
        param: &str,
        strength: f32,
    ) -> f32 {
        let lo = 1.0 - strength;
        let hi = 1.0 + strength;
        let random_factor: f32 = rng.random_range(lo..hi);
        let nudge = self.get(param);
        let biased_factor = random_factor + nudge;
        (value * biased_factor).max(0.0001)
    }

    /// Biased perturbation for usize parameters.
    pub fn biased_perturb_u(
        &self,
        rng: &mut impl Rng,
        value: usize,
        param: &str,
        strength: f32,
    ) -> usize {
        let lo = 1.0 - strength;
        let hi = 1.0 + strength;
        let random_factor: f32 = rng.random_range(lo..hi);
        let nudge = self.get(param);
        let biased_factor = random_factor + nudge;
        ((value as f32 * biased_factor).round() as usize).max(1)
    }

    /// Return the top N parameters by absolute momentum magnitude.
    pub fn top_params(&self, n: usize) -> Vec<(&str, f32)> {
        let mut entries: Vec<(&str, f32)> = self
            .momentum
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        entries.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(n);
        entries
    }

    /// Mutable access to the momentum map (for testing).
    #[cfg(test)]
    pub fn momentum_mut(&mut self) -> &mut HashMap<String, f32> {
        &mut self.momentum
    }

    /// Update momentum from a set of winning offspring.
    ///
    /// For each BrainConfig parameter, computes the average delta between parent
    /// and winners, then blends into momentum:
    ///   momentum[p] = decay * momentum[p] + (1 - decay) * avg_delta[p]
    ///
    /// Does nothing if `winners` is empty (no signal to learn from).
    pub fn update(&mut self, parent: &BrainConfig, winners: &[BrainConfig]) {
        if winners.is_empty() {
            return;
        }
        let n = winners.len() as f32;
        let blend = 1.0 - self.decay;

        let params: Vec<(&str, f32)> = vec![
            ("memory_capacity", parent.memory_capacity as f32),
            ("processing_slots", parent.processing_slots as f32),
            ("representation_dim", parent.representation_dim as f32),
            ("learning_rate", parent.learning_rate),
            ("decay_rate", parent.decay_rate),
            ("distress_exponent", parent.distress_exponent),
            ("habituation_sensitivity", parent.habituation_sensitivity),
            ("max_curiosity_bonus", parent.max_curiosity_bonus),
            ("fatigue_floor", parent.fatigue_floor),
        ];

        for (name, parent_val) in &params {
            let avg_delta: f32 = winners
                .iter()
                .map(|w| {
                    let w_val = match *name {
                        "memory_capacity" => w.memory_capacity as f32,
                        "processing_slots" => w.processing_slots as f32,
                        "representation_dim" => w.representation_dim as f32,
                        "learning_rate" => w.learning_rate,
                        "decay_rate" => w.decay_rate,
                        "distress_exponent" => w.distress_exponent,
                        "habituation_sensitivity" => w.habituation_sensitivity,
                        "max_curiosity_bonus" => w.max_curiosity_bonus,
                        "fatigue_floor" => w.fatigue_floor,
                        _ => *parent_val,
                    };
                    w_val - parent_val
                })
                .sum::<f32>()
                / n;

            if avg_delta.abs() > 1e-8 {
                let current = self.momentum.get(*name).copied().unwrap_or(0.0);
                let updated = self.decay * current + blend * avg_delta;
                if updated.abs() > 1e-8 {
                    self.momentum.insert((*name).to_string(), updated);
                } else {
                    self.momentum.remove(*name);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xagent_shared::BrainConfig;

    #[test]
    fn empty_momentum_returns_zero() {
        let m = MutationMomentum::new(0.9);
        assert_eq!(m.get("learning_rate"), 0.0);
        assert_eq!(m.get("nonexistent"), 0.0);
    }

    #[test]
    fn decay_reduces_values() {
        let mut m = MutationMomentum::new(0.5);
        m.momentum.insert("learning_rate".into(), 1.0);
        m.momentum.insert("decay_rate".into(), -0.8);

        m.decay_step();

        assert!((m.get("learning_rate") - 0.5).abs() < 1e-6);
        assert!((m.get("decay_rate") - (-0.4)).abs() < 1e-6);
    }

    #[test]
    fn decay_cleans_near_zero_entries() {
        let mut m = MutationMomentum::new(0.1);
        m.momentum.insert("tiny".into(), 1e-7);
        m.momentum.insert("big".into(), 1.0);

        m.decay_step();

        assert_eq!(m.get("tiny"), 0.0); // removed
        assert!((m.get("big") - 0.1).abs() < 1e-6); // kept
    }

    #[test]
    fn serialization_round_trip() {
        let mut m = MutationMomentum::new(0.9);
        m.momentum.insert("learning_rate".into(), 0.05);
        m.momentum.insert("decay_rate".into(), -0.03);

        let json = serde_json::to_string(&m).unwrap();
        let restored: MutationMomentum = serde_json::from_str(&json).unwrap();

        assert!((restored.decay - 0.9).abs() < 1e-6);
        assert!((restored.get("learning_rate") - 0.05).abs() < 1e-6);
        assert!((restored.get("decay_rate") - (-0.03)).abs() < 1e-6);
    }

    #[test]
    fn update_builds_positive_momentum() {
        let mut m = MutationMomentum::new(0.9);

        let parent = BrainConfig {
            learning_rate: 0.05,
            decay_rate: 0.001,
            ..BrainConfig::default()
        };

        // Winner has higher learning_rate, same decay_rate
        let winners = vec![BrainConfig {
            learning_rate: 0.06,
            decay_rate: 0.001,
            ..BrainConfig::default()
        }];

        m.update(&parent, &winners);

        // learning_rate delta = 0.06 - 0.05 = 0.01
        // momentum = 0.9 * 0.0 + 0.1 * 0.01 = 0.001
        assert!(m.get("learning_rate") > 0.0);
        // decay_rate unchanged — no momentum
        assert_eq!(m.get("decay_rate"), 0.0);
    }

    #[test]
    fn update_accumulates_across_calls() {
        let mut m = MutationMomentum::new(0.9);

        let parent = BrainConfig::default();
        let winners = vec![BrainConfig {
            learning_rate: parent.learning_rate + 0.01,
            ..BrainConfig::default()
        }];

        m.update(&parent, &winners);
        let after_one = m.get("learning_rate");

        m.update(&parent, &winners);
        let after_two = m.get("learning_rate");

        // Second update should strengthen momentum in same direction
        assert!(after_two > after_one);
    }

    #[test]
    fn update_opposing_signals_cancel() {
        let mut m = MutationMomentum::new(0.5); // fast decay for cleaner test

        let parent = BrainConfig::default();

        // First: winner increases learning_rate
        let up = vec![BrainConfig {
            learning_rate: parent.learning_rate + 0.02,
            ..BrainConfig::default()
        }];
        m.update(&parent, &up);
        let after_up = m.get("learning_rate");
        assert!(after_up > 0.0);

        // Second: winner decreases learning_rate by same amount
        let down = vec![BrainConfig {
            learning_rate: parent.learning_rate - 0.02,
            ..BrainConfig::default()
        }];
        m.update(&parent, &down);
        let after_down = m.get("learning_rate");

        // Should be smaller in magnitude than after_up (partially canceled)
        assert!(after_down.abs() < after_up.abs());
    }

    #[test]
    fn update_no_winners_is_noop() {
        let mut m = MutationMomentum::new(0.9);
        m.momentum.insert("learning_rate".into(), 0.05);

        let parent = BrainConfig::default();
        m.update(&parent, &[]); // no winners

        // Momentum unchanged
        assert!((m.get("learning_rate") - 0.05).abs() < 1e-6);
    }

    #[test]
    fn update_averages_across_multiple_winners() {
        let mut m = MutationMomentum::new(0.9);

        let parent = BrainConfig {
            learning_rate: 0.05,
            ..BrainConfig::default()
        };

        // Two winners: one went up +0.02, the other up +0.04
        // Average delta = +0.03
        let winners = vec![
            BrainConfig {
                learning_rate: 0.07,
                ..BrainConfig::default()
            },
            BrainConfig {
                learning_rate: 0.09,
                ..BrainConfig::default()
            },
        ];

        m.update(&parent, &winners);

        // momentum = 0.9 * 0.0 + 0.1 * 0.03 = 0.003
        let val = m.get("learning_rate");
        assert!((val - 0.003).abs() < 1e-5);
    }

    #[test]
    fn biased_perturb_f_no_momentum_stays_in_range() {
        let m = MutationMomentum::new(0.9); // empty momentum
        let mut rng = rand::rng();

        let value = 1.0;
        for _ in 0..100 {
            let result = m.biased_perturb_f(&mut rng, value, "learning_rate", 0.1);
            assert!(result >= 0.0001);
            // Without momentum, factor is in [0.9, 1.1], so result in [0.9, 1.1]
            assert!(
                result >= 0.89 && result <= 1.11,
                "result {} out of expected range",
                result
            );
        }
    }

    #[test]
    fn biased_perturb_f_with_momentum_shifts_distribution() {
        let mut m = MutationMomentum::new(0.9);
        m.momentum.insert("learning_rate".into(), 0.1);

        let mut rng = rand::rng();
        let value = 1.0;

        let mut sum = 0.0;
        let n = 1000;
        for _ in 0..n {
            sum += m.biased_perturb_f(&mut rng, value, "learning_rate", 0.1);
        }
        let avg = sum / n as f32;

        // Average should be above 1.0 (biased upward by momentum)
        assert!(
            avg > 1.0,
            "avg {} should be > 1.0 with positive momentum",
            avg
        );
    }

    #[test]
    fn biased_perturb_u_no_momentum_stays_reasonable() {
        let m = MutationMomentum::new(0.9);
        let mut rng = rand::rng();

        let value: usize = 100;
        for _ in 0..100 {
            let result = m.biased_perturb_u(&mut rng, value, "memory_capacity", 0.1);
            assert!(result >= 1);
            assert!(
                result >= 85 && result <= 115,
                "result {} out of expected range",
                result
            );
        }
    }

    #[test]
    fn biased_perturb_f_respects_min_clamp() {
        let mut m = MutationMomentum::new(0.9);
        m.momentum.insert("fatigue_floor".into(), -10.0);

        let mut rng = rand::rng();
        for _ in 0..100 {
            let result = m.biased_perturb_f(&mut rng, 0.001, "fatigue_floor", 0.5);
            assert!(result >= 0.0001, "result {} below min clamp", result);
        }
    }
}
