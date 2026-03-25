//! Capacity manager: enforces per-tick processing budgets.
//!
//! Limited recall budget forces the brain to prioritize which patterns
//! to retrieve, giving rise to attention-like behavior. A surprise budget
//! reserves capacity for novel/unexpected patterns.

/// Manages the brain's processing capacity budget per tick.
///
/// The capacity manager enforces limits on how much work the brain
/// can do each tick. This forces the brain to prioritize — it cannot
/// recall all patterns or process all sensory details. Under pressure,
/// it must focus, which gives rise to attention-like behavior.
///
/// The recall budget is **adaptive**: high prediction error allocates more
/// slots to recall more context, while a "surprise budget" reserves
/// capacity for novel/unexpected patterns.
pub struct CapacityManager {
    /// Maximum recall operations per tick.
    max_recall_budget: usize,
    /// Current recall budget (may vary with state).
    current_recall_budget: usize,

    // --- Cognitive load tracking ---
    /// EMA of recall slots actually used.
    avg_recall_used: f32,
    /// EMA of prediction error (proxy for cognitive demand).
    avg_prediction_error: f32,
    /// Peak prediction error seen recently.
    peak_error: f32,
    /// How many ticks the surprise budget has been active.
    surprise_active_ticks: u32,

    // --- Surprise budget ---
    /// Fraction of budget reserved for novel patterns [0, 0.4].
    surprise_fraction: f32,
}

/// Snapshot of cognitive load for telemetry.
#[derive(Clone, Debug, Default)]
pub struct CognitiveLoad {
    pub recall_budget: usize,
    pub surprise_budget: usize,
    pub avg_utilization: f32,
    pub avg_prediction_error: f32,
}

impl CapacityManager {
    /// Create a new capacity manager with the given number of processing slots.
    ///
    /// All slots are initially available for recall; the surprise fraction starts at 10%.
    pub fn new(processing_slots: usize) -> Self {
        Self {
            max_recall_budget: processing_slots,
            current_recall_budget: processing_slots,
            avg_recall_used: 0.0,
            avg_prediction_error: 0.0,
            peak_error: 0.0,
            surprise_active_ticks: 0,
            surprise_fraction: 0.1,
        }
    }

    /// Update with latest prediction error, then allocate recall budget.
    /// Returns (recall_budget, surprise_budget).
    pub fn allocate_recall_budget_adaptive(&mut self, prediction_error: f32) -> (usize, usize) {
        // Update cognitive-load EMAs
        self.avg_prediction_error = self.avg_prediction_error * 0.9 + prediction_error * 0.1;
        self.peak_error = self.peak_error * 0.98_f32 + prediction_error * 0.02;

        // Adaptive recall budget: scale linearly with prediction error
        // Low error → use 50% of budget; high error → use up to 100%
        let error_scale = (0.5 + self.avg_prediction_error * 2.0).clamp(0.5, 1.0);
        let base_budget = (self.max_recall_budget as f32 * error_scale).round() as usize;

        // Surprise budget: reserve slots proportional to surprise level
        // Spike detection: if current error > 2× average → increase surprise fraction
        if prediction_error > self.avg_prediction_error * 2.0 && prediction_error > 0.05 {
            self.surprise_fraction = (self.surprise_fraction + 0.05).min(0.4);
            self.surprise_active_ticks = 0;
        } else {
            self.surprise_active_ticks += 1;
            // Decay surprise fraction if no surprises for a while
            if self.surprise_active_ticks > 20 {
                self.surprise_fraction = (self.surprise_fraction - 0.01).max(0.05);
            }
        }

        let surprise_budget =
            (self.max_recall_budget as f32 * self.surprise_fraction).ceil() as usize;
        let recall_budget = base_budget.min(self.max_recall_budget.saturating_sub(surprise_budget));

        self.current_recall_budget = recall_budget;
        (recall_budget, surprise_budget)
    }

    /// Simple allocation (backward-compatible).
    pub fn allocate_recall_budget(&mut self) -> usize {
        self.current_recall_budget = self.max_recall_budget;
        self.current_recall_budget
    }

    /// Report how many recall slots were actually used this tick.
    pub fn report_usage(&mut self, used: usize) {
        self.avg_recall_used = self.avg_recall_used * 0.9 + used as f32 * 0.1;
    }

    /// Current cognitive load snapshot.
    pub fn cognitive_load(&self) -> CognitiveLoad {
        let surprise_budget =
            (self.max_recall_budget as f32 * self.surprise_fraction).ceil() as usize;
        CognitiveLoad {
            recall_budget: self.current_recall_budget,
            surprise_budget,
            avg_utilization: self.avg_recall_used / self.max_recall_budget.max(1) as f32,
            avg_prediction_error: self.avg_prediction_error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_error_increases_recall_budget() {
        let mut cm = CapacityManager::new(16);
        let (low_budget, _) = cm.allocate_recall_budget_adaptive(0.01);
        let mut cm2 = CapacityManager::new(16);
        let (high_budget, _) = cm2.allocate_recall_budget_adaptive(0.9);

        assert!(
            high_budget >= low_budget,
            "High error should allocate at least as much budget: low={low_budget}, high={high_budget}"
        );
    }

    #[test]
    fn surprise_spike_increases_surprise_fraction() {
        let mut cm = CapacityManager::new(16);
        let initial_frac = cm.surprise_fraction;

        // Feed low errors to establish baseline
        for _ in 0..20 {
            cm.allocate_recall_budget_adaptive(0.01);
        }
        // Big spike
        cm.allocate_recall_budget_adaptive(0.9);
        assert!(
            cm.surprise_fraction > initial_frac,
            "Surprise spike should increase fraction"
        );
    }

    #[test]
    fn cognitive_load_reports_correctly() {
        let mut cm = CapacityManager::new(16);
        cm.allocate_recall_budget_adaptive(0.3);
        cm.report_usage(8);
        let load = cm.cognitive_load();
        assert!(load.recall_budget <= 16);
        assert!(load.avg_utilization > 0.0);
    }
}
