//! Sensory habituation: post-encoder attenuation filter + curiosity bonus.
//!
//! Tracks per-dimension variance of the encoded state over time. Dimensions
//! that stop changing get attenuated (habituated). A curiosity bonus derived
//! from the mean attenuation drives exploration when input is monotonous.

/// EMA smoothing factor for variance tracking (~50-tick window).
const HABITUATION_EMA_ALPHA: f32 = 0.02;
/// Scales variance_ema into the [0, 1] attenuation range.
const SENSITIVITY: f32 = 15.0;
/// Minimum attenuation — never fully suppress a dimension.
const ATTENUATION_FLOOR: f32 = 0.1;
/// Maximum curiosity bonus (same scale as novelty_bonus in action selector).
const MAX_CURIOSITY_BONUS: f32 = 0.4;

/// Post-encoder filter that attenuates repetitive sensory input and produces
/// a curiosity bonus that drives exploration during monotony.
pub struct SensoryHabituation {
    prev_encoded: Vec<f32>,
    variance_ema: Vec<f32>,
    attenuation: Vec<f32>,
    habituated: Vec<f32>,
    curiosity_bonus: f32,
    tick: u64,
}

impl SensoryHabituation {
    /// Create a new habituation filter for the given representation dimension.
    pub fn new(repr_dim: usize) -> Self {
        Self {
            prev_encoded: vec![0.0; repr_dim],
            variance_ema: vec![0.0; repr_dim],
            attenuation: vec![1.0; repr_dim],
            habituated: vec![0.0; repr_dim],
            curiosity_bonus: 0.0,
            tick: 0,
        }
    }

    /// Update with a new encoded state. Computes attenuation and curiosity bonus.
    pub fn update(&mut self, encoded: &[f32]) {
        self.tick += 1;
        let dim = self.prev_encoded.len().min(encoded.len());

        let mut attenuation_sum = 0.0_f32;
        for i in 0..dim {
            let delta = (encoded[i] - self.prev_encoded[i]).abs();
            self.variance_ema[i] =
                (1.0 - HABITUATION_EMA_ALPHA) * self.variance_ema[i] + HABITUATION_EMA_ALPHA * delta;
            self.attenuation[i] = (self.variance_ema[i] * SENSITIVITY).clamp(ATTENUATION_FLOOR, 1.0);
            self.habituated[i] = encoded[i] * self.attenuation[i];
            attenuation_sum += self.attenuation[i];
            self.prev_encoded[i] = encoded[i];
        }

        let mean_attenuation = if dim > 0 {
            attenuation_sum / dim as f32
        } else {
            1.0
        };
        self.curiosity_bonus = (1.0 - mean_attenuation) * MAX_CURIOSITY_BONUS;
    }

    /// The habituated (attenuated) encoded state.
    pub fn habituated_state(&self) -> &[f32] {
        &self.habituated
    }

    /// Current curiosity bonus [0.0, MAX_CURIOSITY_BONUS].
    pub fn curiosity_bonus(&self) -> f32 {
        self.curiosity_bonus
    }

    /// Mean attenuation across all dimensions [ATTENUATION_FLOOR, 1.0].
    pub fn mean_attenuation(&self) -> f32 {
        if self.attenuation.is_empty() {
            return 1.0;
        }
        self.attenuation.iter().sum::<f32>() / self.attenuation.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_input_increases_habituation() {
        let mut hab = SensoryHabituation::new(4);
        let state = vec![0.5, 0.3, -0.1, 0.2];
        for _ in 0..200 {
            hab.update(&state);
        }
        assert!(
            hab.mean_attenuation() < 0.2,
            "Constant input should drive attenuation near floor: {}",
            hab.mean_attenuation()
        );
        assert!(
            hab.curiosity_bonus() > 0.3,
            "Constant input should produce high curiosity: {}",
            hab.curiosity_bonus()
        );
    }

    #[test]
    fn varying_input_keeps_attenuation_high() {
        let mut hab = SensoryHabituation::new(4);
        for i in 0..200 {
            let v = (i as f32 * 0.1).sin();
            let state = vec![v, -v, v * 0.5, v * 0.3];
            hab.update(&state);
        }
        assert!(
            hab.mean_attenuation() > 0.5,
            "Varying input should keep attenuation high: {}",
            hab.mean_attenuation()
        );
        assert!(
            hab.curiosity_bonus() < 0.2,
            "Varying input should produce low curiosity: {}",
            hab.curiosity_bonus()
        );
    }

    #[test]
    fn habituation_recovers_after_change() {
        let mut hab = SensoryHabituation::new(4);
        let constant = vec![0.5, 0.3, -0.1, 0.2];
        for _ in 0..200 {
            hab.update(&constant);
        }
        assert!(hab.mean_attenuation() < 0.2);
        let different = vec![-0.5, 0.8, 0.4, -0.3];
        for _ in 0..100 {
            hab.update(&different);
        }
        assert!(
            hab.mean_attenuation() < 0.3,
            "Should re-habituate to new constant: {}",
            hab.mean_attenuation()
        );
    }

    #[test]
    fn habituated_state_is_attenuated() {
        let mut hab = SensoryHabituation::new(4);
        let state = vec![1.0, 1.0, 1.0, 1.0];
        for _ in 0..200 {
            hab.update(&state);
        }
        for &v in hab.habituated_state() {
            assert!(v < 0.3, "Habituated value should be attenuated: {}", v);
        }
    }

    #[test]
    fn attenuation_is_bounded() {
        let mut hab = SensoryHabituation::new(4);
        for _ in 0..500 {
            hab.update(&[0.0, 0.0, 0.0, 0.0]);
        }
        for &a in &hab.attenuation {
            assert!(
                a >= ATTENUATION_FLOOR && a <= 1.0,
                "Attenuation {} out of bounds [{}, 1.0]",
                a, ATTENUATION_FLOOR
            );
        }
        assert!(
            hab.curiosity_bonus() >= 0.0 && hab.curiosity_bonus() <= MAX_CURIOSITY_BONUS,
            "Curiosity {} out of bounds [0, {}]",
            hab.curiosity_bonus(), MAX_CURIOSITY_BONUS
        );
    }
}
