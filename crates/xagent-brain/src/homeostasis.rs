//! Homeostatic monitor: tracks physiological signal trends across multiple timescales.
//!
//! This is the brain's ONLY evaluative signal — there is no explicit reward.
//! The gradient (improving vs. worsening) modulates learning rates and
//! action preferences throughout the cognitive architecture.

// --- Constants ---
const ENERGY_WEIGHT: f32 = 0.6;
const INTEGRITY_WEIGHT: f32 = 0.4;
const FAST_EMA_ALPHA: f32 = 0.6;
const MEDIUM_EMA_ALPHA: f32 = 0.04;
const SLOW_EMA_ALPHA: f32 = 0.004;
const GRADIENT_BLEND_FAST: f32 = 0.5;
const GRADIENT_BLEND_MEDIUM: f32 = 0.35;
const GRADIENT_BLEND_SLOW: f32 = 0.15;
const DISTRESS_SCALE: f32 = 10.0;
const MAX_DISTRESS: f32 = 10.0;

/// Tracks internal physiological signals and computes homeostatic gradient
/// across multiple timescales, plus a non-linear urgency signal.
///
/// The homeostatic monitor is NOT a "hunger module" or "fear module".
/// It simply tracks whether internal variables are trending toward or
/// away from stability. This gradient is the ONLY evaluative signal
/// in the entire brain — it modulates learning rates and action preferences.
pub struct HomeostaticMonitor {
    /// Previous energy signal.
    prev_energy: f32,
    /// Previous integrity signal.
    prev_integrity: f32,

    // --- Multi-timescale gradient EMAs ---
    /// Fast gradient (last ~5 ticks, α ≈ 0.4).
    gradient_fast: f32,
    /// Medium gradient (last ~50 ticks, α ≈ 0.04).
    gradient_medium: f32,
    /// Slow gradient (last ~500 ticks, α ≈ 0.004).
    gradient_slow: f32,

    /// Non-linear urgency signal [0, ∞). Increases rapidly as internal states
    /// approach critical levels.
    urgency: f32,

    /// Current energy level (cached for external queries).
    current_energy: f32,
    /// Current integrity level.
    current_integrity: f32,
}

/// Result of a homeostatic update, consumed by other brain modules.
#[derive(Clone, Debug)]
pub struct HomeostaticState {
    /// Composite gradient (blended from all timescales). Used for urgency,
    /// exploration rate, and modulated learning rate — contexts that benefit
    /// from temporal smoothing.
    pub gradient: f32,
    /// Raw per-tick gradient: energy_delta × 0.6 + integrity_delta × 0.4.
    /// No EMA smoothing. Used for credit assignment because actions need
    /// the *immediate* consequence signal, not a smoothed average. A food
    /// event that changes energy by +0.2 produces raw_gradient = +0.12,
    /// vs composite ≈ +0.04 (67% lost to EMA blending).
    pub raw_gradient: f32,
    /// Fast-timescale gradient.
    pub gradient_fast: f32,
    /// Medium-timescale gradient.
    pub gradient_medium: f32,
    /// Slow-timescale gradient.
    pub gradient_slow: f32,
    /// Urgency signal [0, ∞). Higher values mean the agent is in danger.
    pub urgency: f32,
}

impl HomeostaticMonitor {
    /// Create a new monitor assuming full health (energy=1.0, integrity=1.0).
    pub fn new() -> Self {
        Self {
            prev_energy: 1.0,
            prev_integrity: 1.0,
            gradient_fast: 0.0,
            gradient_medium: 0.0,
            gradient_slow: 0.0,
            urgency: 0.0,
            current_energy: 1.0,
            current_integrity: 1.0,
        }
    }

    /// Reset all EMAs to neutral. Called on death so the respawn's health
    /// jump doesn't produce a false positive gradient that rewards the
    /// actions from the previous life.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Update with current interoceptive signals. Returns full homeostatic state.
    ///
    /// Positive gradient = internal state improving (good).
    /// Negative gradient = internal state worsening (bad).
    pub fn update(&mut self, energy_signal: f32, integrity_signal: f32) -> HomeostaticState {
        // Guard against NaN inputs (e.g. from max_energy=0 division)
        let energy_signal = if energy_signal.is_finite() { energy_signal } else { 0.0 };
        let integrity_signal = if integrity_signal.is_finite() { integrity_signal } else { 0.0 };

        let energy_delta = energy_signal - self.prev_energy;
        let integrity_delta = integrity_signal - self.prev_integrity;

        // Combined raw gradient (integrity is weighted higher — damage is threatening)
        let raw_gradient = energy_delta * ENERGY_WEIGHT + integrity_delta * INTEGRITY_WEIGHT;

        // Update multi-timescale EMAs
        self.gradient_fast = self.gradient_fast * (1.0 - FAST_EMA_ALPHA) + raw_gradient * FAST_EMA_ALPHA;
        self.gradient_medium = self.gradient_medium * (1.0 - MEDIUM_EMA_ALPHA) + raw_gradient * MEDIUM_EMA_ALPHA;
        self.gradient_slow = self.gradient_slow * (1.0 - SLOW_EMA_ALPHA) + raw_gradient * SLOW_EMA_ALPHA;

        // --- Non-linear urgency ---
        let energy_distress = Self::distress_curve(energy_signal);
        let integrity_distress = Self::distress_curve(integrity_signal);
        self.urgency = (energy_distress + integrity_distress) * 0.5;

        // --- Composite gradient ---
        // Blend timescales, amplified by urgency
        let base_gradient = self.gradient_fast * GRADIENT_BLEND_FAST
            + self.gradient_medium * GRADIENT_BLEND_MEDIUM
            + self.gradient_slow * GRADIENT_BLEND_SLOW;
        let urgency_amplifier = 1.0 + self.urgency;
        let gradient = base_gradient * urgency_amplifier;

        self.prev_energy = energy_signal;
        self.prev_integrity = integrity_signal;
        self.current_energy = energy_signal;
        self.current_integrity = integrity_signal;

        HomeostaticState {
            gradient,
            raw_gradient,
            gradient_fast: self.gradient_fast,
            gradient_medium: self.gradient_medium,
            gradient_slow: self.gradient_slow,
            urgency: self.urgency,
        }
    }

    /// Current urgency signal (for external queries without updating).
    pub fn urgency(&self) -> f32 {
        self.urgency
    }

    /// Current composite gradient.
    pub fn gradient(&self) -> f32 {
        self.gradient_fast * GRADIENT_BLEND_FAST
            + self.gradient_medium * GRADIENT_BLEND_MEDIUM
            + self.gradient_slow * GRADIENT_BLEND_SLOW
    }

    /// Non-linear distress: maps a [0, 1] health level to [0, MAX_DISTRESS].
    /// distress(1.0) = 0, distress(0.5) = 2.5, distress(0.1) = 8.1, distress(0.01) ≈ 9.8.
    fn distress_curve(level: f32) -> f32 {
        let clamped = level.clamp(0.01, 1.0);
        let distress = (1.0 - clamped).powi(2) * DISTRESS_SCALE;
        distress.min(MAX_DISTRESS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_signals_produce_near_zero_gradient() {
        let mut hm = HomeostaticMonitor::new();
        // Feed the same (healthy) signals repeatedly
        for _ in 0..20 {
            let state = hm.update(0.8, 0.9);
            // Gradient should settle near zero
            assert!(
                state.gradient.abs() < 0.5,
                "Stable signals should give near-zero gradient, got {}",
                state.gradient
            );
        }
    }

    #[test]
    fn improving_signals_give_positive_gradient() {
        let mut hm = HomeostaticMonitor::new();
        // Start at a moderate level (not too far from initial 1.0 to avoid big negative spike)
        let mut energy = 0.5;
        let mut integrity = 0.5;
        hm.update(energy, integrity);
        // Steadily improve over many ticks to let EMAs settle
        for i in 1..=20 {
            energy += 0.02;
            integrity += 0.015;
            let state = hm.update(energy, integrity);
            if i > 8 {
                assert!(
                    state.gradient > 0.0,
                    "Improving signals should give positive gradient at step {i}: {}",
                    state.gradient
                );
            }
        }
    }

    #[test]
    fn worsening_signals_give_negative_gradient() {
        let mut hm = HomeostaticMonitor::new();
        hm.update(0.9, 0.9);
        // Steadily worsen
        for i in 1..=10 {
            let e = 0.9 - i as f32 * 0.06;
            let state = hm.update(e, 0.9 - i as f32 * 0.04);
            if i > 2 {
                assert!(
                    state.gradient < 0.0,
                    "Worsening signals should give negative gradient at step {i}: {}",
                    state.gradient
                );
            }
        }
    }

    #[test]
    fn urgency_increases_at_critical_levels() {
        let mut hm = HomeostaticMonitor::new();
        let state_healthy = hm.update(0.9, 0.9);
        let mut hm2 = HomeostaticMonitor::new();
        let state_critical = hm2.update(0.1, 0.1);

        assert!(
            state_critical.urgency > state_healthy.urgency,
            "Critical levels should produce higher urgency: healthy={}, critical={}",
            state_healthy.urgency,
            state_critical.urgency
        );
    }

    #[test]
    fn multi_timescale_gradients_converge_differently() {
        let mut hm = HomeostaticMonitor::new();
        // Single positive step
        hm.update(0.5, 0.5);
        let state = hm.update(0.6, 0.6);

        // Fast should react more than slow
        assert!(
            state.gradient_fast.abs() > state.gradient_slow.abs(),
            "Fast gradient should react more strongly: fast={}, slow={}",
            state.gradient_fast,
            state.gradient_slow
        );
    }

    #[test]
    fn distress_curve_is_bounded() {
        for i in 0..=100 {
            let level = i as f32 / 100.0;
            let d = HomeostaticMonitor::distress_curve(level);
            assert!(
                d >= 0.0 && d <= MAX_DISTRESS,
                "distress({level}) = {d} out of bounds [0, {MAX_DISTRESS}]"
            );
        }
    }

    #[test]
    fn gradient_responds_to_energy_drop() {
        let mut hm = HomeostaticMonitor::new();
        hm.update(1.0, 1.0);
        let state = hm.update(0.5, 1.0);
        assert!(
            state.gradient_fast < 0.0,
            "Energy drop should produce negative fast gradient: {}",
            state.gradient_fast
        );
    }

    #[test]
    fn gradient_responds_to_energy_gain() {
        let mut hm = HomeostaticMonitor::new();
        hm.update(0.5, 1.0);
        let state = hm.update(1.0, 1.0);
        assert!(
            state.gradient_fast > 0.0,
            "Energy gain should produce positive fast gradient: {}",
            state.gradient_fast
        );
    }

    #[test]
    fn urgency_increases_at_low_levels() {
        let mut hm1 = HomeostaticMonitor::new();
        let state_low = hm1.update(0.1, 0.5);
        let mut hm2 = HomeostaticMonitor::new();
        let state_high = hm2.update(0.9, 0.5);
        assert!(
            state_low.urgency > state_high.urgency * 2.0,
            "Urgency at low energy should be much higher: low={}, high={}",
            state_low.urgency,
            state_high.urgency
        );
    }

    #[test]
    fn multi_timescale_gradients_fast_reacts_quicker() {
        let mut hm = HomeostaticMonitor::new();
        // Feed stable signal then a sudden drop
        for _ in 0..50 {
            hm.update(0.8, 0.8);
        }
        let state = hm.update(0.5, 0.8);
        assert!(
            state.gradient_fast.abs() > state.gradient_slow.abs(),
            "Fast gradient should react more to sudden change: fast={}, slow={}",
            state.gradient_fast,
            state.gradient_slow
        );
    }
}
