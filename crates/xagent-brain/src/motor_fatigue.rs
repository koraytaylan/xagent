//! Motor fatigue: dampens motor output when action variance is low.
//!
//! Tracks recent forward and turn outputs in a ring buffer. When the variance
//! is low (repetitive motor commands), a fatigue factor reduces the effective
//! motor output. Recovery is immediate when output diversifies.

/// Ring buffer size for tracking recent motor outputs.
const FATIGUE_WINDOW: usize = 64;
/// Scales variance into the [0, 1] fatigue factor range.
const RECOVERY_SENSITIVITY: f32 = 5.0;
/// Minimum fatigue factor — never fully paralyze (80% max dampening).
const FATIGUE_FLOOR: f32 = 0.2;

/// Tracks motor output variance and produces a fatigue dampening factor.
pub struct MotorFatigue {
    forward_ring: Vec<f32>,
    turn_ring: Vec<f32>,
    cursor: usize,
    len: usize,
    fatigue_factor: f32,
    motor_variance: f32,
}

impl MotorFatigue {
    /// Create a new motor fatigue tracker.
    pub fn new() -> Self {
        Self {
            forward_ring: vec![0.0; FATIGUE_WINDOW],
            turn_ring: vec![0.0; FATIGUE_WINDOW],
            cursor: 0,
            len: 0,
            fatigue_factor: 1.0,
            motor_variance: 0.0,
        }
    }

    /// Record a motor command and update the fatigue factor.
    pub fn update(&mut self, forward: f32, turn: f32) {
        self.forward_ring[self.cursor] = forward;
        self.turn_ring[self.cursor] = turn;
        self.cursor = (self.cursor + 1) % FATIGUE_WINDOW;
        if self.len < FATIGUE_WINDOW {
            self.len += 1;
        }

        let fwd_var = Self::variance(&self.forward_ring[..self.len]);
        let turn_var = Self::variance(&self.turn_ring[..self.len]);
        self.motor_variance = fwd_var + turn_var;

        self.fatigue_factor = (self.motor_variance * RECOVERY_SENSITIVITY)
            .clamp(FATIGUE_FLOOR, 1.0);
    }

    /// Current fatigue factor [FATIGUE_FLOOR, 1.0]. Multiply motor output by this.
    pub fn fatigue_factor(&self) -> f32 {
        self.fatigue_factor
    }

    /// Current total motor variance (for telemetry).
    pub fn motor_variance(&self) -> f32 {
        self.motor_variance
    }

    /// Compute variance of a slice.
    fn variance(data: &[f32]) -> f32 {
        if data.len() < 2 {
            return 0.0;
        }
        let n = data.len() as f32;
        let mean = data.iter().sum::<f32>() / n;
        data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_output_causes_fatigue() {
        let mut mf = MotorFatigue::new();
        for _ in 0..FATIGUE_WINDOW {
            mf.update(0.5, 0.3);
        }
        assert!(
            mf.fatigue_factor() < 0.3,
            "Constant motor output should cause fatigue: {}",
            mf.fatigue_factor()
        );
        assert!(
            mf.motor_variance() < 0.01,
            "Constant output should have near-zero variance: {}",
            mf.motor_variance()
        );
    }

    #[test]
    fn varied_output_prevents_fatigue() {
        let mut mf = MotorFatigue::new();
        for i in 0..FATIGUE_WINDOW {
            let v = (i as f32 * 0.2).sin();
            mf.update(v, -v);
        }
        assert!(
            mf.fatigue_factor() > 0.8,
            "Varied output should prevent fatigue: {}",
            mf.fatigue_factor()
        );
    }

    #[test]
    fn fatigue_recovers_when_output_diversifies() {
        let mut mf = MotorFatigue::new();
        for _ in 0..FATIGUE_WINDOW {
            mf.update(0.5, 0.3);
        }
        assert!(mf.fatigue_factor() < 0.3);
        for i in 0..FATIGUE_WINDOW {
            let v = (i as f32 * 0.3).sin();
            mf.update(v, -v);
        }
        assert!(
            mf.fatigue_factor() > 0.8,
            "Fatigue should recover when output diversifies: {}",
            mf.fatigue_factor()
        );
    }

    #[test]
    fn fatigue_factor_is_bounded() {
        let mut mf = MotorFatigue::new();
        for _ in 0..200 {
            mf.update(0.0, 0.0);
        }
        assert!(
            mf.fatigue_factor() >= FATIGUE_FLOOR,
            "Fatigue factor should not go below floor: {}",
            mf.fatigue_factor()
        );
        for i in 0..200 {
            let v = if i % 2 == 0 { 1.0 } else { -1.0 };
            mf.update(v, -v);
        }
        assert!(
            mf.fatigue_factor() <= 1.0,
            "Fatigue factor should not exceed 1.0: {}",
            mf.fatigue_factor()
        );
    }

    #[test]
    fn initial_state_has_no_fatigue() {
        let mf = MotorFatigue::new();
        assert_eq!(mf.fatigue_factor(), 1.0);
        assert_eq!(mf.motor_variance(), 0.0);
    }
}
