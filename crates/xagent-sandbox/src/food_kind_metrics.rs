//! Per-food-kind free-life measurements and run-summary types.
//!
//! Pure computation: no GPU handle is held and nothing is written.

use serde::{Deserialize, Serialize};
use xagent_brain::buffers::{
    FOOD_POSITION_X, FOOD_POSITION_Z, FOOD_RESPAWN_TIMER, FOOD_STATE_STRIDE,
};
use xagent_shared::FoodKind;

/// Numerator of the late-window fraction of a run (the last 30% of ticks).
pub const LATE_FRACTION_NUMERATOR: u64 = 3;
/// Denominator of the late-window fraction; integer arithmetic keeps the boundary exact.
pub const LATE_FRACTION_DENOMINATOR: u64 = 10;
/// Number of equal run quarters in which meals are counted.
pub const QUARTER_COUNT: usize = 4;

/// Which experimental arm a run belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Arm {
    /// Learning enabled.
    Learning,
    /// Learning frozen.
    Frozen,
}

/// Per-agent food-kind measurements for one run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentFoodKindSummary {
    /// Mean turn toward nourishing food over the whole run.
    pub turn_toward_nourishing_whole: Option<f32>,
    /// Mean turn toward nourishing food over the late window.
    pub turn_toward_nourishing_late: Option<f32>,
    /// Mean turn away from harmful food over the whole run.
    pub turn_away_harmful_whole: Option<f32>,
    /// Mean turn away from harmful food over the late window.
    pub turn_away_harmful_late: Option<f32>,
    /// Nourishing meals per quarter.
    pub nourishing_meals_by_quarter: [u32; 4],
    /// Harmful meals per quarter.
    pub harmful_meals_by_quarter: [u32; 4],
    /// Death count.
    pub deaths: u32,
    /// Readout balanced accuracy.
    pub readout_balanced_accuracy: Option<f32>,
    /// Readout accuracy with shuffled labels.
    pub readout_shuffled_accuracy: Option<f32>,
    /// Signed credit of the first pair.
    pub signed_credit_pair_first: f32,
    /// Signed credit of the second pair.
    pub signed_credit_pair_second: f32,
    /// Signed weight difference of the first pair.
    pub signed_weight_difference_pair_first: f32,
    /// Signed weight difference of the second pair.
    pub signed_weight_difference_pair_second: f32,
}

/// Summary of one run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunFoodKindSummary {
    /// Run seed.
    pub seed: u64,
    /// Arm of the run.
    pub arm: Arm,
    /// Ticks simulated.
    pub ticks: u64,
    /// Number of agents.
    pub agent_count: usize,
    /// Whether brains started fresh.
    pub fresh_brains: bool,
    /// Which kind is nourishing.
    pub nourishing_kind: FoodKind,
    /// Per-agent summaries.
    pub agents: Vec<AgentFoodKindSummary>,
}

/// Nearest available food of one kind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearestFood {
    /// Distance to the food.
    pub distance: f32,
    /// Signed bearing, as the kernel computes it.
    pub bearing: f32,
}

/// Nearest available item of `kind` strictly within `range`, if any.
#[must_use]
pub fn nearest_food_of_kind(
    position: [f32; 3],
    facing: [f32; 2],
    food_state: &[f32],
    kind: FoodKind,
    range: f32,
) -> Option<NearestFood> {
    let range_sq = range * range;
    let mut best: Option<(f32, f32, f32)> = None;
    for (i, item) in food_state.as_chunks::<FOOD_STATE_STRIDE>().0.iter().enumerate() {
        if FoodKind::of_item(i) != kind || item[FOOD_RESPAWN_TIMER] != 0.0 {
            continue;
        }
        let to_x = item[FOOD_POSITION_X] - position[0];
        let to_z = item[FOOD_POSITION_Z] - position[2];
        let d_sq = to_x * to_x + to_z * to_z;
        if d_sq < range_sq && best.is_none_or(|(b, _, _)| d_sq < b) {
            best = Some((d_sq, to_x, to_z));
        }
    }
    best.map(|(d_sq, to_x, to_z)| NearestFood {
        distance: d_sq.sqrt(),
        bearing: (facing[0] * to_z - facing[1] * to_x).atan2(facing[0] * to_x + facing[1] * to_z),
    })
}

/// Approach sign of a bearing: positive means toward the food.
#[must_use]
pub fn approach_sign(bearing: f32) -> f32 {
    if bearing == 0.0 {
        0.0
    } else {
        -bearing.signum()
    }
}

/// Mean turns over the whole run and the late window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KindTurnMeans {
    /// Mean turn toward nourishing food, whole run.
    pub turn_toward_nourishing_whole: Option<f32>,
    /// Mean turn toward nourishing food, late window.
    pub turn_toward_nourishing_late: Option<f32>,
    /// Mean turn away from harmful food, whole run.
    pub turn_away_harmful_whole: Option<f32>,
    /// Mean turn away from harmful food, late window.
    pub turn_away_harmful_late: Option<f32>,
}

/// Running sum and count.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Mean {
    sum: f64,
    count: u64,
}

impl Mean {
    fn add(&mut self, value: f32) {
        self.sum += f64::from(value);
        self.count += 1;
    }

    /// Mean narrowed to `f32`, or `None` without samples.
    #[allow(clippy::cast_possible_truncation)]
    fn mean(&self) -> Option<f32> {
        if self.count == 0 {
            return None;
        }
        // Counts stay far below 2^32 in practice, so the conversion is exact.
        let count = f64::from(u32::try_from(self.count).unwrap_or(u32::MAX));
        Some((self.sum / count) as f32)
    }
}

/// Accumulates executed turns relative to food bearings.
#[derive(Debug, Clone, PartialEq)]
pub struct KindTurnAccumulator {
    late_start: u64,
    toward_whole: Mean,
    toward_late: Mean,
    away_whole: Mean,
    away_late: Mean,
}

impl KindTurnAccumulator {
    /// New accumulator for a run of `total_ticks`.
    #[must_use]
    pub fn new(total_ticks: u64) -> Self {
        let late_start = total_ticks
            - total_ticks.saturating_mul(LATE_FRACTION_NUMERATOR) / LATE_FRACTION_DENOMINATOR;
        Self {
            late_start,
            toward_whole: Mean::default(),
            toward_late: Mean::default(),
            away_whole: Mean::default(),
            away_late: Mean::default(),
        }
    }

    /// Records one tick's executed turn against the available bearings.
    pub fn record(
        &mut self,
        tick: u64,
        executed_turn: f32,
        nourishing_bearing: Option<f32>,
        harmful_bearing: Option<f32>,
    ) {
        let late = tick >= self.late_start;
        if let Some(b) = nourishing_bearing {
            let v = executed_turn * approach_sign(b);
            self.toward_whole.add(v);
            if late {
                self.toward_late.add(v);
            }
        }
        if let Some(b) = harmful_bearing {
            let v = -executed_turn * approach_sign(b);
            self.away_whole.add(v);
            if late {
                self.away_late.add(v);
            }
        }
    }

    /// Means of all recorded samples.
    #[must_use]
    pub fn finish(&self) -> KindTurnMeans {
        KindTurnMeans {
            turn_toward_nourishing_whole: self.toward_whole.mean(),
            turn_toward_nourishing_late: self.toward_late.mean(),
            turn_away_harmful_whole: self.away_whole.mean(),
            turn_away_harmful_late: self.away_late.mean(),
        }
    }
}

/// Meals per quarter by kind.
#[derive(Debug, Clone, PartialEq)]
pub struct KindMealQuarters {
    /// Nourishing meals per quarter.
    pub nourishing_by_quarter: [u32; 4],
    /// Harmful meals per quarter.
    pub harmful_by_quarter: [u32; 4],
}

impl KindMealQuarters {
    /// Differences of cumulative `(nourishing, harmful)` counts; `None` if any decreases.
    #[must_use]
    pub fn from_cumulative(cumulative: [(u32, u32); 4]) -> Option<Self> {
        let mut nourishing = [0u32; 4];
        let mut harmful = [0u32; 4];
        let (mut prev_n, mut prev_h) = (0u32, 0u32);
        for (q, &(n, h)) in cumulative.iter().enumerate() {
            nourishing[q] = n.checked_sub(prev_n)?;
            harmful[q] = h.checked_sub(prev_h)?;
            prev_n = n;
            prev_h = h;
        }
        Some(Self {
            nourishing_by_quarter: nourishing,
            harmful_by_quarter: harmful,
        })
    }

    /// Nourishing share of meals in `quarter`, or `None` without meals.
    #[must_use]
    pub fn nourishing_share(&self, quarter: usize) -> Option<f32> {
        if quarter >= QUARTER_COUNT {
            return None;
        }
        let n = self.nourishing_by_quarter[quarter];
        let total = u64::from(n) + u64::from(self.harmful_by_quarter[quarter]);
        if total == 0 {
            return None;
        }
        Some(f64_to_f32(f64::from(n) / total as f64))
    }
}

/// Narrows an `f64` to `f32`.
#[allow(clippy::cast_possible_truncation)]
fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

/// Accumulates credit per pair kind.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct PairCreditAccumulator {
    first: f64,
    second: f64,
}

impl PairCreditAccumulator {
    /// Adds one tick's credit contribution to the pair of `pair_kind`.
    pub fn record(
        &mut self,
        pair_kind: FoodKind,
        td_error: f32,
        turn_noise_innovation: f32,
        whitened_left: f32,
        whitened_right: f32,
    ) {
        let v = f64::from(td_error)
            * f64::from(turn_noise_innovation)
            * f64::from(whitened_right - whitened_left);
        match pair_kind {
            FoodKind::First => self.first += v,
            FoodKind::Second => self.second += v,
        }
    }

    /// Accumulated credit of the pair of `pair_kind`.
    #[must_use]
    pub fn credit(&self, pair_kind: FoodKind) -> f32 {
        f64_to_f32(match pair_kind {
            FoodKind::First => self.first,
            FoodKind::Second => self.second,
        })
    }
}

/// `value` when the kinds match, else its negation.
#[must_use]
pub fn signed_for_nourishing(value: f32, pair_kind: FoodKind, nourishing_kind: FoodKind) -> f32 {
    if pair_kind == nourishing_kind {
        value
    } else {
        -value
    }
}

/// Right minus left weight, signed for the nourishing kind.
#[must_use]
pub fn signed_weight_difference(
    left_weight: f32,
    right_weight: f32,
    pair_kind: FoodKind,
    nourishing_kind: FoodKind,
) -> f32 {
    signed_for_nourishing(right_weight - left_weight, pair_kind, nourishing_kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn food(items: &[(f32, f32, f32)]) -> Vec<f32> {
        items
            .iter()
            .flat_map(|&(x, z, t)| {
                let mut s = [0.0; FOOD_STATE_STRIDE];
                s[FOOD_POSITION_X] = x;
                s[FOOD_POSITION_Z] = z;
                s[FOOD_RESPAWN_TIMER] = t;
                s
            })
            .collect()
    }

    #[test]
    fn nearest_food_kind_filter_and_bearing_sign() {
        // Item 0 First at distance 5, item 1 Second at distance 1.
        let fs = food(&[(5.0, 0.0, 0.0), (1.0, 0.0, 0.0)]);
        let n = nearest_food_of_kind([0.0; 3], [1.0, 0.0], &fs, FoodKind::First, 10.0).unwrap();
        assert!((n.distance - 5.0).abs() < 1e-6);
        assert!(n.bearing.abs() < 1e-6);
        let fs = food(&[(0.0, 2.0, 0.0)]);
        let n = nearest_food_of_kind([0.0; 3], [1.0, 0.0], &fs, FoodKind::First, 10.0).unwrap();
        assert!(n.bearing > 0.0);
    }

    #[test]
    fn nearest_food_skips_unavailable_items() {
        let fs = food(&[(1.0, 0.0, 5.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0)]);
        let n = nearest_food_of_kind([0.0; 3], [1.0, 0.0], &fs, FoodKind::First, 10.0).unwrap();
        assert!((n.distance - 3.0).abs() < 1e-6);
    }

    #[test]
    fn nearest_food_out_of_range_is_none() {
        let fs = food(&[(5.0, 0.0, 0.0)]);
        assert!(nearest_food_of_kind([0.0; 3], [1.0, 0.0], &fs, FoodKind::First, 5.0).is_none());
    }

    #[test]
    fn turn_accumulator_late_window_and_signs() {
        let mut a = KindTurnAccumulator::new(300_000);
        a.record(209_999, 1.0, Some(-0.5), None);
        let m = a.finish();
        assert_eq!(m.turn_toward_nourishing_whole, Some(1.0));
        assert_eq!(m.turn_toward_nourishing_late, None);
        a.record(210_000, 1.0, Some(-0.5), Some(-0.5));
        let m = a.finish();
        assert_eq!(m.turn_toward_nourishing_late, Some(1.0));
        assert_eq!(m.turn_away_harmful_late, Some(-1.0));
    }

    #[test]
    fn turn_accumulator_none_without_samples() {
        let m = KindTurnAccumulator::new(100).finish();
        assert_eq!(m.turn_toward_nourishing_whole, None);
        assert_eq!(m.turn_away_harmful_late, None);
    }

    #[test]
    fn meal_quarters_shares_with_zero_meal_quarter() {
        let q = KindMealQuarters::from_cumulative([(1, 1), (1, 1), (4, 1), (4, 2)]).unwrap();
        assert_eq!(q.nourishing_by_quarter, [1, 0, 3, 0]);
        assert_eq!(q.harmful_by_quarter, [1, 0, 0, 1]);
        assert_eq!(q.nourishing_share(0), Some(0.5));
        assert_eq!(q.nourishing_share(1), None);
        assert_eq!(q.nourishing_share(2), Some(1.0));
        assert_eq!(q.nourishing_share(4), None);
    }

    #[test]
    fn meal_quarters_rejects_decreasing_counts() {
        assert!(KindMealQuarters::from_cumulative([(2, 0), (1, 0), (3, 0), (4, 0)]).is_none());
        assert!(KindMealQuarters::from_cumulative([(0, 2), (0, 1), (0, 3), (0, 4)]).is_none());
    }

    #[test]
    fn pair_credit_sum_and_signing() {
        let mut c = PairCreditAccumulator::default();
        c.record(FoodKind::First, 2.0, 0.5, 1.0, 3.0);
        c.record(FoodKind::First, 1.0, -1.0, 0.0, 1.0);
        assert!((c.credit(FoodKind::First) - 1.0).abs() < 1e-6);
        assert_eq!(c.credit(FoodKind::Second), 0.0);
        let v = c.credit(FoodKind::First);
        assert_eq!(signed_for_nourishing(v, FoodKind::First, FoodKind::First), v);
        assert_eq!(signed_for_nourishing(v, FoodKind::First, FoodKind::Second), -v);
    }

    #[test]
    fn signed_weight_difference_sign() {
        assert_eq!(
            signed_weight_difference(1.0, 3.0, FoodKind::First, FoodKind::First),
            2.0
        );
        assert_eq!(
            signed_weight_difference(1.0, 3.0, FoodKind::Second, FoodKind::First),
            -2.0
        );
    }

    #[test]
    fn run_food_kind_summary_serde_round_trip() {
        let run = RunFoodKindSummary {
            seed: 7,
            arm: Arm::Frozen,
            ticks: 300_000,
            agent_count: 1,
            fresh_brains: true,
            nourishing_kind: FoodKind::Second,
            agents: vec![AgentFoodKindSummary {
                turn_toward_nourishing_whole: Some(0.25),
                turn_toward_nourishing_late: None,
                turn_away_harmful_whole: None,
                turn_away_harmful_late: Some(-0.5),
                nourishing_meals_by_quarter: [1, 2, 3, 4],
                harmful_meals_by_quarter: [4, 3, 2, 1],
                deaths: 2,
                readout_balanced_accuracy: Some(0.75),
                readout_shuffled_accuracy: None,
                signed_credit_pair_first: 1.5,
                signed_credit_pair_second: -1.5,
                signed_weight_difference_pair_first: 0.5,
                signed_weight_difference_pair_second: -0.5,
            }],
        };
        let json = serde_json::to_string(&run).unwrap();
        let back: RunFoodKindSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(run, back);
    }
}
