//! Per-parameter mutation momentum for directed evolution.
//!
//! Each island maintains its own momentum vector that biases future mutations
//! toward directions that previously improved fitness. Momentum decays each
//! generation so stale signals fade naturally.
//!
//! Momentum is dimensionless: it tracks the *relative* change winners made to
//! each parameter, because mutation is multiplicative (`value × factor`) and
//! the momentum nudge shifts that factor. Parameters span five orders of
//! magnitude (`decay_rate` ≈ 0.001 vs `memory_capacity` ≈ 2048), so absolute
//! deltas would turn one large-unit winner into a factor of ±hundreds and pin
//! every mutant to a clamp bound.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use xagent_shared::BrainConfig;

/// Largest shift, in multiplicative-factor units, that momentum may apply to a
/// mutation. With the adaptive mutation strength capped at 0.5 the random
/// factor never drops below 0.5, so a 0.25 cap keeps every biased factor in
/// `[0.25, 1.75]`: strictly positive (no sign flip, no collapse to a clamp
/// floor) and never more than a 75% move per generation. It also bounds any
/// oversized momentum restored from an older database.
pub const MAX_MOMENTUM_NUDGE: f32 = 0.25;

/// Floor on the parent value used as the relative-delta denominator, so a gene
/// sitting at (or next to) zero cannot produce a division by zero.
const RELATIVE_DELTA_DENOMINATOR_FLOOR: f32 = 1e-6;

/// Accumulates per-parameter directional signal from successful mutations.
///
/// After each generation, winning offspring (those that beat their parent's
/// fitness) contribute their relative mutation deltas to the momentum. Future
/// perturbations are biased in the momentum direction — parameters with
/// strong momentum get pushed toward winning values, while parameters with
/// weak momentum stay near random noise.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MutationMomentum {
    /// Per-parameter momentum as a relative (dimensionless) change, bounded to
    /// `±MAX_MOMENTUM_NUDGE`. Positive = trending upward, negative = trending down.
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

    /// Multiplicative-factor shift applied to a mutation of `param`: the stored
    /// momentum bounded to `±MAX_MOMENTUM_NUDGE`. The bound is re-applied here
    /// so momentum deserialized from a database written before the bound
    /// existed cannot push a factor negative.
    fn nudge(&self, param: &str) -> f32 {
        self.get(param)
            .clamp(-MAX_MOMENTUM_NUDGE, MAX_MOMENTUM_NUDGE)
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
    /// Combines a random multiplicative factor in `[1 - strength, 1 + strength)`
    /// with the bounded momentum nudge that shifts the center of perturbation.
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
        let biased_factor = random_factor + self.nudge(param);
        (value * biased_factor).max(0.0001)
    }

    /// Biased perturbation for usize parameters.
    ///
    /// The scaled value is stochastically rounded (rounded up with probability
    /// equal to its fractional part), so the expected result equals the scaled
    /// value. Nearest rounding would make small values absorbing: `1 × factor`
    /// rounds back to 1 for every factor below 1.5, so a gene that ever reached
    /// 1 could never grow again.
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
        let biased_factor = random_factor + self.nudge(param);
        let scaled = (value as f32 * biased_factor).max(0.0);
        let round_up = rng.random::<f32>() < scaled.fract();
        let rounded = if round_up {
            scaled.floor() + 1.0
        } else {
            scaled.floor()
        };
        (rounded as usize).max(1)
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
    /// For each BrainConfig parameter, computes the average relative delta
    /// between parent and winners, `(winner - parent) / |parent|`, then blends
    /// it into momentum and bounds the result to `±MAX_MOMENTUM_NUDGE`:
    ///   momentum[p] = decay * momentum[p] + (1 - decay) * avg_relative_delta[p]
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
            (
                "representation_dimension",
                parent.representation_dimension as f32,
            ),
            ("learning_rate", parent.learning_rate),
            ("decay_rate", parent.decay_rate),
            ("distress_exponent", parent.distress_exponent),
            ("habituation_sensitivity", parent.habituation_sensitivity),
            ("max_curiosity_bonus", parent.max_curiosity_bonus),
            ("fatigue_floor", parent.fatigue_floor),
            ("movement_speed", parent.movement_speed),
            ("gabor_wavelength", parent.gabor_wavelength),
            ("gabor_aspect_ratio", parent.gabor_aspect_ratio),
            ("dog_surround_ratio", parent.dog_surround_ratio),
            ("orientation_offset", parent.orientation_offset),
            ("horizontal_fov_degrees", parent.horizontal_fov_degrees),
            ("vertical_fov_degrees", parent.vertical_fov_degrees),
            ("smell_strength", parent.smell_strength),
        ];

        for (name, parent_val) in &params {
            let denominator = parent_val.abs().max(RELATIVE_DELTA_DENOMINATOR_FLOOR);
            let avg_delta: f32 = winners
                .iter()
                .map(|w| {
                    let w_val = match *name {
                        "memory_capacity" => w.memory_capacity as f32,
                        "processing_slots" => w.processing_slots as f32,
                        "representation_dimension" => w.representation_dimension as f32,
                        "learning_rate" => w.learning_rate,
                        "decay_rate" => w.decay_rate,
                        "distress_exponent" => w.distress_exponent,
                        "habituation_sensitivity" => w.habituation_sensitivity,
                        "max_curiosity_bonus" => w.max_curiosity_bonus,
                        "fatigue_floor" => w.fatigue_floor,
                        "movement_speed" => w.movement_speed,
                        "gabor_wavelength" => w.gabor_wavelength,
                        "gabor_aspect_ratio" => w.gabor_aspect_ratio,
                        "dog_surround_ratio" => w.dog_surround_ratio,
                        "orientation_offset" => w.orientation_offset,
                        "horizontal_fov_degrees" => w.horizontal_fov_degrees,
                        "vertical_fov_degrees" => w.vertical_fov_degrees,
                        "smell_strength" => w.smell_strength,
                        _ => *parent_val,
                    };
                    (w_val - parent_val) / denominator
                })
                .sum::<f32>()
                / n;

            if avg_delta.abs() > 1e-8 {
                let current = self.momentum.get(*name).copied().unwrap_or(0.0);
                let updated = (self.decay * current + blend * avg_delta)
                    .clamp(-MAX_MOMENTUM_NUDGE, MAX_MOMENTUM_NUDGE);
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

        // learning_rate relative delta = (0.06 - 0.05) / 0.05 = 0.2
        // momentum = 0.9 * 0.0 + 0.1 * 0.2 = 0.02
        assert!((m.get("learning_rate") - 0.02).abs() < 1e-5);
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

        // Two winners: one went up +0.02 (+40%), the other up +0.04 (+80%)
        // Average relative delta = +0.6
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

        // momentum = 0.9 * 0.0 + 0.1 * 0.6 = 0.06
        let val = m.get("learning_rate");
        assert!((val - 0.06).abs() < 1e-5);
    }

    #[test]
    fn update_is_dimensionless_across_parameter_scales() {
        let mut m = MutationMomentum::new(0.9);
        let parent = BrainConfig {
            memory_capacity: 128,
            learning_rate: 0.05,
            ..BrainConfig::default()
        };
        // Both winners double their parameter; the raw deltas differ by a
        // factor of ~2500 but the relative change is identical.
        let winners = vec![BrainConfig {
            memory_capacity: 256,
            learning_rate: 0.1,
            ..BrainConfig::default()
        }];

        m.update(&parent, &winners);

        assert!(
            (m.get("memory_capacity") - m.get("learning_rate")).abs() < 1e-6,
            "memory_capacity momentum {} must equal learning_rate momentum {}",
            m.get("memory_capacity"),
            m.get("learning_rate")
        );
    }

    #[test]
    fn update_bounds_momentum_to_max_nudge() {
        let mut m = MutationMomentum::new(0.9);
        let parent = BrainConfig {
            memory_capacity: 1,
            movement_speed: 1.0,
            ..BrainConfig::default()
        };
        let winners = vec![BrainConfig {
            memory_capacity: 2048,
            movement_speed: 100.0,
            ..BrainConfig::default()
        }];

        for _ in 0..50 {
            m.update(&parent, &winners);
        }

        assert!(m.get("memory_capacity") <= MAX_MOMENTUM_NUDGE);
        assert!(m.get("movement_speed") <= MAX_MOMENTUM_NUDGE);
        assert!(m.get("movement_speed") > 0.0);
    }

    #[test]
    fn oversized_restored_momentum_cannot_pin_mutants_to_bounds() {
        // Magnitudes taken from a database written when momentum stored raw
        // parameter deltas: +8.99 on movement_speed, -333.7 on memory_capacity.
        let restored: MutationMomentum = serde_json::from_str(
            r#"{"momentum":{"movement_speed":8.986292,"memory_capacity":-333.7462},"decay":0.9}"#,
        )
        .unwrap();
        let mut rng = rand::rng();
        let strength = 0.5;
        let speed = 50.0_f32;
        let memory: usize = 128;
        let lowest_factor = 1.0 - strength - MAX_MOMENTUM_NUDGE;
        let highest_factor = 1.0 + strength + MAX_MOMENTUM_NUDGE;

        for _ in 0..1000 {
            let mutated_speed =
                restored.biased_perturb_f(&mut rng, speed, "movement_speed", strength);
            assert!(
                mutated_speed <= speed * highest_factor,
                "speed {mutated_speed} escaped the bounded factor"
            );
            let mutated_memory =
                restored.biased_perturb_u(&mut rng, memory, "memory_capacity", strength);
            assert!(
                mutated_memory as f32 >= (memory as f32 * lowest_factor).floor(),
                "memory {mutated_memory} collapsed below the bounded factor"
            );
        }
    }

    #[test]
    fn integer_gene_at_one_is_not_absorbing() {
        let m = MutationMomentum::new(0.9);
        let mut rng = rand::rng();

        let grew = (0..1000).any(|_| m.biased_perturb_u(&mut rng, 1, "processing_slots", 0.5) > 1);

        assert!(grew, "a gene at 1 must be able to mutate upward");
    }

    #[test]
    fn integer_perturbation_is_unbiased_without_momentum() {
        let m = MutationMomentum::new(0.9);
        let mut rng = rand::rng();
        let draws = 20_000;
        let value = 3;

        let total: usize = (0..draws)
            .map(|_| m.biased_perturb_u(&mut rng, value, "processing_slots", 0.2))
            .sum();
        let mean = total as f32 / draws as f32;

        assert!(
            (mean - value as f32).abs() < 0.05,
            "mean {mean} should stay at {value}"
        );
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
