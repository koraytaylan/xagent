//! Per-parameter mutation momentum for directed evolution.
//!
//! Each island maintains its own momentum vector that biases future mutations
//! toward directions that previously improved fitness. Momentum decays each
//! generation so stale signals fade naturally.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
