//! In-run summary with season-relative food-kind windows.
//!
//! Pure computation over copies of physics and food rows: no GPU handle is
//! held and nothing is written.

use std::fmt;

use serde::{Deserialize, Serialize};
use xagent_brain::buffers::{
    FOOD_STATE_STRIDE, PHYS_STRIDE, P_ALIVE, P_DEATH_COUNT, P_FACING_X, P_FACING_Z, P_FOOD_COUNT,
    P_HARMFUL_MEAL_COUNT, P_MOTOR_TURN_OUT, P_POS_X, P_POS_Z,
};
use xagent_shared::{EvolutionMode, FoodKind, Inheritance, SCENT_RANGE};

use crate::food_kind_metrics::nearest_food_of_kind;

/// Numerator of the early and late window fraction of a season.
pub const WINDOW_FRACTION_NUMERATOR: usize = 3;
/// Denominator of the early and late window fraction of a season.
pub const WINDOW_FRACTION_DENOMINATOR: usize = 10;
/// Counters at or above this value are no longer exact in `f32`.
pub const COUNTER_EXACT_LIMIT: f32 = 16_777_216.0;

/// Totals of one run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    /// Evolution mode of the run.
    pub mode: EvolutionMode,
    /// Inheritance policy; `None` in generation mode.
    pub inheritance: Option<Inheritance>,
    /// Run seed.
    pub seed: u64,
    /// Number of agent slots.
    pub agent_count: usize,
    /// Ticks simulated.
    pub ticks: u64,
    /// Agent-ticks simulated (`ticks * agent_count`).
    pub agent_ticks: u64,
    /// Whether food kinds were enabled.
    pub food_kinds_enabled: bool,
    /// Nourishing meals eaten.
    pub nourishing_meals: u64,
    /// Harmful meals eaten.
    pub harmful_meals: u64,
    /// Deaths.
    pub deaths: u64,
    /// Seasons begun.
    pub seasons: u64,
    /// Season-relative totals; `Some` iff food kinds were enabled.
    pub kinds: Option<KindWindowTotals>,
}

/// Season-relative window totals summed over all seasons.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KindWindowTotals {
    /// Sum of turn toward nourishing food over late windows.
    pub late_toward_nourishing_sum: f64,
    /// Turn samples behind `late_toward_nourishing_sum`.
    pub late_toward_nourishing_samples: u64,
    /// Sum of turn away from harmful food over late windows.
    pub late_away_harmful_sum: f64,
    /// Turn samples behind `late_away_harmful_sum`.
    pub late_away_harmful_samples: u64,
    /// Nourishing meals in late windows.
    pub late_nourishing_meals: u64,
    /// Harmful meals in late windows.
    pub late_harmful_meals: u64,
    /// Nourishing meals in early windows.
    pub early_nourishing_meals: u64,
    /// Harmful meals in early windows.
    pub early_harmful_meals: u64,
}

/// Why a sample or a call was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunSummaryError {
    /// Physics slice is not `agent_count * PHYS_STRIDE` long.
    PhysicsLengthMismatch,
    /// Food slice length is not a multiple of `FOOD_STATE_STRIDE`.
    FoodLengthMismatch,
    /// Sample tick is not after the previous sample tick.
    TickNotAfterPrevious,
    /// A counter fell below its baseline without a reset.
    CounterDecreased,
    /// A counter is non-finite, fractional, negative or too large.
    InvalidCounter,
    /// Agent index is not below the agent count.
    AgentOutOfRange,
    /// Food kinds are on but no nourishing kind is known.
    MissingNourishingKind,
    /// A size computation overflowed.
    SizeOverflow,
}

impl fmt::Display for RunSummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::PhysicsLengthMismatch => "physics slice length does not match the agent count",
            Self::FoodLengthMismatch => "food slice length is not a multiple of the item stride",
            Self::TickNotAfterPrevious => "sample tick is not after the previous sample",
            Self::CounterDecreased => "counter decreased without a baseline reset",
            Self::InvalidCounter => "counter is not an exact non-negative integer",
            Self::AgentOutOfRange => "agent index is out of range",
            Self::MissingNourishingKind => "food kinds are on but no nourishing kind is set",
            Self::SizeOverflow => "size computation overflowed",
        };
        f.write_str(text)
    }
}

impl std::error::Error for RunSummaryError {}

/// Sample-interval record buffered inside an open season.
#[derive(Clone, Copy, Default)]
struct Interval {
    nourishing_meals: u64,
    harmful_meals: u64,
    toward_sum: f64,
    toward_samples: u64,
    away_sum: f64,
    away_samples: u64,
}

/// Season whose intervals are still being collected.
struct OpenSeason {
    intervals: Vec<Interval>,
}

/// Collects a [`RunSummary`] from physics samples.
pub struct RunSummaryAccumulator {
    mode: EvolutionMode,
    inheritance: Option<Inheritance>,
    seed: u64,
    agent_count: usize,
    kinds_on: bool,
    nourishing: Option<FoodKind>,
    baselines: Vec<[u64; 3]>,
    last_tick: Option<u64>,
    nourishing_meals: u64,
    harmful_meals: u64,
    deaths: u64,
    seasons: u64,
    open: Option<OpenSeason>,
    totals: KindWindowTotals,
}

/// Approach sign of a bearing: positive means a turn toward the food.
fn approach_sign_of(bearing: f32) -> f32 {
    if bearing == 0.0 || bearing.is_nan() {
        0.0
    } else {
        -bearing.signum()
    }
}

/// The kind that is not `kind`.
fn other_kind(kind: FoodKind) -> FoodKind {
    match kind {
        FoodKind::First => FoodKind::Second,
        FoodKind::Second => FoodKind::First,
    }
}

/// Validates a raw counter value.
fn validated_counter(value: f32) -> Result<f32, RunSummaryError> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value >= COUNTER_EXACT_LIMIT {
        Err(RunSummaryError::InvalidCounter)
    } else {
        Ok(value)
    }
}

/// Converts a validated counter to `u64`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn counter_to_u64(value: f32) -> u64 {
    value as u64
}

fn read_counter(physics: &[f32], base: usize, slot: usize) -> Result<u64, RunSummaryError> {
    validated_counter(physics[base + slot]).map(counter_to_u64)
}

impl RunSummaryAccumulator {
    /// Creates an accumulator with all baselines at zero.
    #[must_use]
    pub fn new(
        mode: EvolutionMode,
        inheritance: Option<Inheritance>,
        seed: u64,
        agent_count: usize,
        food_kinds_enabled: bool,
    ) -> Self {
        Self {
            mode,
            inheritance,
            seed,
            agent_count,
            kinds_on: food_kinds_enabled,
            nourishing: None,
            baselines: vec![[0; 3]; agent_count],
            last_tick: None,
            nourishing_meals: 0,
            harmful_meals: 0,
            deaths: 0,
            seasons: 0,
            open: None,
            totals: KindWindowTotals {
                late_toward_nourishing_sum: 0.0,
                late_toward_nourishing_samples: 0,
                late_away_harmful_sum: 0.0,
                late_away_harmful_samples: 0,
                late_nourishing_meals: 0,
                late_harmful_meals: 0,
                early_nourishing_meals: 0,
                early_harmful_meals: 0,
            },
        }
    }

    /// Closes the open season and begins a new one. `_start_tick` is accepted
    /// for the interface and not used: intervals, not ticks, define windows.
    ///
    /// # Errors
    /// `MissingNourishingKind` when kinds are on and `nourishing` is `None`.
    pub fn begin_season(
        &mut self,
        _start_tick: u64,
        nourishing: Option<FoodKind>,
    ) -> Result<(), RunSummaryError> {
        if self.kinds_on && nourishing.is_none() {
            return Err(RunSummaryError::MissingNourishingKind);
        }
        self.close_season();
        self.seasons += 1;
        if self.kinds_on {
            self.nourishing = nourishing;
            self.open = Some(OpenSeason {
                intervals: Vec::new(),
            });
        }
        Ok(())
    }

    /// Replaces an agent slot: its next counters are deltas from zero.
    ///
    /// # Errors
    /// `AgentOutOfRange` when `agent` is not below the agent count.
    pub fn reset_agent_baseline(&mut self, agent: usize) -> Result<(), RunSummaryError> {
        let slot = self
            .baselines
            .get_mut(agent)
            .ok_or(RunSummaryError::AgentOutOfRange)?;
        *slot = [0; 3];
        Ok(())
    }

    /// Sets every baseline to zero (generation start).
    pub fn reset_all_baselines(&mut self) {
        self.baselines.fill([0; 3]);
    }

    /// Records one sample interval ending at `tick`.
    ///
    /// # Errors
    /// Any [`RunSummaryError`] describing the rejected input; state is
    /// unchanged on error.
    pub fn record_physics_sample(
        &mut self,
        tick: u64,
        physics: &[f32],
        food: Option<&[f32]>,
    ) -> Result<(), RunSummaryError> {
        let expected = self
            .agent_count
            .checked_mul(PHYS_STRIDE)
            .ok_or(RunSummaryError::SizeOverflow)?;
        if physics.len() != expected {
            return Err(RunSummaryError::PhysicsLengthMismatch);
        }
        if self.kinds_on {
            if food.is_some_and(|f| f.len() % FOOD_STATE_STRIDE != 0) {
                return Err(RunSummaryError::FoodLengthMismatch);
            }
            if self.open.is_none() {
                return Err(RunSummaryError::MissingNourishingKind);
            }
        }
        if self.last_tick.is_some_and(|last| tick <= last) {
            return Err(RunSummaryError::TickNotAfterPrevious);
        }

        let mut updated = Vec::with_capacity(self.agent_count);
        let mut deltas = [0_u64; 3];
        for agent in 0..self.agent_count {
            let base = agent * PHYS_STRIDE;
            let current = [
                read_counter(physics, base, P_FOOD_COUNT)?,
                read_counter(physics, base, P_DEATH_COUNT)?,
                if self.kinds_on {
                    read_counter(physics, base, P_HARMFUL_MEAL_COUNT)?
                } else {
                    0
                },
            ];
            let before = self.baselines[agent];
            for i in 0..3 {
                let delta = current[i]
                    .checked_sub(before[i])
                    .ok_or(RunSummaryError::CounterDecreased)?;
                deltas[i] = deltas[i]
                    .checked_add(delta)
                    .ok_or(RunSummaryError::SizeOverflow)?;
            }
            updated.push(current);
        }

        let mut interval = Interval {
            nourishing_meals: deltas[0],
            harmful_meals: deltas[2],
            ..Interval::default()
        };
        if let (Some(food), Some(kind)) = (food, self.nourishing) {
            if self.kinds_on {
                for agent in 0..self.agent_count {
                    let base = agent * PHYS_STRIDE;
                    if physics[base + P_ALIVE] == 0.0 {
                        continue;
                    }
                    let turn = physics[base + P_MOTOR_TURN_OUT];
                    if !turn.is_finite() {
                        continue;
                    }
                    let position = [physics[base + P_POS_X], 0.0, physics[base + P_POS_Z]];
                    let facing = [physics[base + P_FACING_X], physics[base + P_FACING_Z]];
                    if let Some(near) =
                        nearest_food_of_kind(position, facing, food, kind, SCENT_RANGE)
                    {
                        interval.toward_sum += f64::from(turn * approach_sign_of(near.bearing));
                        interval.toward_samples += 1;
                    }
                    if let Some(far) =
                        nearest_food_of_kind(position, facing, food, other_kind(kind), SCENT_RANGE)
                    {
                        interval.away_sum += f64::from(-turn * approach_sign_of(far.bearing));
                        interval.away_samples += 1;
                    }
                }
            }
        }

        self.baselines = updated;
        self.last_tick = Some(tick);
        self.nourishing_meals = self.nourishing_meals.saturating_add(deltas[0]);
        self.deaths = self.deaths.saturating_add(deltas[1]);
        self.harmful_meals = self.harmful_meals.saturating_add(deltas[2]);
        if let Some(open) = self.open.as_mut() {
            open.intervals.push(interval);
        }
        Ok(())
    }

    /// Ends the run after `ticks` ticks and returns the summary.
    ///
    /// # Errors
    /// `SizeOverflow` when `ticks * agent_count` overflows.
    pub fn finish(mut self, ticks: u64) -> Result<RunSummary, RunSummaryError> {
        let count = u64::try_from(self.agent_count).map_err(|_| RunSummaryError::SizeOverflow)?;
        let agent_ticks = ticks
            .checked_mul(count)
            .ok_or(RunSummaryError::SizeOverflow)?;
        self.close_season();
        Ok(RunSummary {
            mode: self.mode,
            inheritance: self.inheritance,
            seed: self.seed,
            agent_count: self.agent_count,
            ticks,
            agent_ticks,
            food_kinds_enabled: self.kinds_on,
            nourishing_meals: self.nourishing_meals,
            harmful_meals: self.harmful_meals,
            deaths: self.deaths,
            seasons: self.seasons,
            kinds: self.kinds_on.then_some(self.totals),
        })
    }

    /// Classifies the open season's intervals into early and late windows.
    fn close_season(&mut self) {
        let Some(open) = self.open.take() else {
            return;
        };
        let n = open.intervals.len();
        let w = n * WINDOW_FRACTION_NUMERATOR / WINDOW_FRACTION_DENOMINATOR;
        for interval in &open.intervals[..w] {
            self.totals.early_nourishing_meals += interval.nourishing_meals;
            self.totals.early_harmful_meals += interval.harmful_meals;
        }
        for interval in &open.intervals[n - w..] {
            self.totals.late_nourishing_meals += interval.nourishing_meals;
            self.totals.late_harmful_meals += interval.harmful_meals;
            self.totals.late_toward_nourishing_sum += interval.toward_sum;
            self.totals.late_toward_nourishing_samples += interval.toward_samples;
            self.totals.late_away_harmful_sum += interval.away_sum;
            self.totals.late_away_harmful_samples += interval.away_samples;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xagent_brain::buffers::{FOOD_POSITION_X, FOOD_POSITION_Z};

    fn rows(agents: usize) -> Vec<f32> {
        vec![0.0; agents * PHYS_STRIDE]
    }

    fn acc(agents: usize, kinds: bool) -> RunSummaryAccumulator {
        RunSummaryAccumulator::new(EvolutionMode::default(), None, 7, agents, kinds)
    }

    #[test]
    fn run_summary_meal_and_death_deltas() {
        let mut a = acc(2, false);
        let mut p = rows(2);
        p[P_FOOD_COUNT] = 1.0;
        p[PHYS_STRIDE + P_DEATH_COUNT] = 1.0;
        a.record_physics_sample(1, &p, None).unwrap();
        p[P_FOOD_COUNT] = 3.0;
        p[PHYS_STRIDE + P_FOOD_COUNT] = 2.0;
        a.record_physics_sample(2, &p, None).unwrap();
        a.reset_agent_baseline(1).unwrap();
        p[PHYS_STRIDE + P_FOOD_COUNT] = 1.0;
        p[PHYS_STRIDE + P_DEATH_COUNT] = 1.0;
        a.record_physics_sample(3, &p, None).unwrap();
        let s = a.finish(3).unwrap();
        assert_eq!(s.nourishing_meals, 3 + 2 + 1);
        assert_eq!(s.deaths, 1 + 1);
        assert_eq!(s.agent_ticks, 6);
    }

    #[test]
    fn run_summary_rejects_decreasing_counter_without_reset() {
        let mut a = acc(1, false);
        let mut p = rows(1);
        p[P_FOOD_COUNT] = 2.0;
        a.record_physics_sample(1, &p, None).unwrap();
        p[P_FOOD_COUNT] = 1.0;
        assert_eq!(
            a.record_physics_sample(2, &p, None),
            Err(RunSummaryError::CounterDecreased)
        );
    }

    #[test]
    fn run_summary_rejects_wrong_physics_length() {
        let mut a = acc(2, false);
        let p = rows(1);
        assert_eq!(
            a.record_physics_sample(1, &p, None),
            Err(RunSummaryError::PhysicsLengthMismatch)
        );
    }

    /// One agent whose nourishing (`harmful` false) or harmful (`harmful` true)
    /// meal counter rises by one each interval.
    fn run_meals(season_len: u64, a: &mut RunSummaryAccumulator, harmful: bool) {
        let mut p = rows(1);
        let start = a.last_tick.unwrap_or(0);
        for i in 1..=season_len {
            if harmful {
                p[P_HARMFUL_MEAL_COUNT] = (a.baselines[0][2] + 1) as f32;
            } else {
                p[P_FOOD_COUNT] = (a.baselines[0][0] + 1) as f32;
            }
            a.record_physics_sample(start + i, &p, None).unwrap();
        }
    }

    #[test]
    fn run_summary_season_windows_first_and_last_thirty_percent() {
        let mut a = acc(1, true);
        a.begin_season(0, Some(FoodKind::First)).unwrap();
        run_meals(10, &mut a, false);
        let s = a.finish(10).unwrap();
        let k = s.kinds.unwrap();
        assert_eq!(k.early_nourishing_meals, 3);
        assert_eq!(k.late_nourishing_meals, 3);

        let mut a = acc(1, true);
        a.begin_season(0, Some(FoodKind::First)).unwrap();
        run_meals(4, &mut a, true);
        let k = a.finish(4).unwrap().kinds.unwrap();
        assert_eq!(k.early_harmful_meals, 1);
        assert_eq!(k.late_harmful_meals, 1);
    }

    fn food_row(x: f32, z: f32) -> Vec<f32> {
        let mut f = vec![0.0; FOOD_STATE_STRIDE];
        f[FOOD_POSITION_X] = x;
        f[FOOD_POSITION_Z] = z;
        f
    }

    fn turning_agent() -> Vec<f32> {
        let mut p = rows(1);
        p[P_ALIVE] = 1.0;
        p[P_FACING_Z] = 1.0;
        p[P_MOTOR_TURN_OUT] = 1.0;
        p
    }

    #[test]
    fn run_summary_turn_judged_against_current_season_kind() {
        let mut a = acc(1, true);
        let p = turning_agent();
        let food = food_row(1.0, 0.0);
        let mut tick = 0;
        for kind in [FoodKind::First, FoodKind::Second] {
            a.begin_season(tick, Some(kind)).unwrap();
            for _ in 0..10 {
                tick += 1;
                a.record_physics_sample(tick, &p, Some(&food)).unwrap();
            }
        }
        let k = a.finish(tick).unwrap().kinds.unwrap();
        assert_eq!(k.late_toward_nourishing_sum, 3.0);
        assert_eq!(k.late_toward_nourishing_samples, 3);
        assert_eq!(k.late_away_harmful_sum, -3.0);
        assert_eq!(k.late_away_harmful_samples, 3);
    }

    #[test]
    fn run_summary_approach_sign_matches_food_kind_convention() {
        assert_eq!(approach_sign_of(-0.5), 1.0);
        assert_eq!(approach_sign_of(0.5), -1.0);
        assert_eq!(approach_sign_of(0.0), 0.0);
        let food = food_row(1.0, 0.0);
        let near = nearest_food_of_kind([0.0; 3], [0.0, 1.0], &food, FoodKind::First, SCENT_RANGE)
            .unwrap();
        assert!(approach_sign_of(near.bearing) * near.bearing < 0.0);
    }

    #[test]
    fn run_summary_kinds_off_has_no_kind_totals() {
        let mut a = acc(1, false);
        a.begin_season(0, None).unwrap();
        a.begin_season(5, None).unwrap();
        a.record_physics_sample(1, &rows(1), None).unwrap();
        let s = a.finish(1).unwrap();
        assert!(s.kinds.is_none());
        assert_eq!(s.seasons, 2);
    }

    #[test]
    fn run_summary_agent_ticks_overflow_is_error() {
        let a = acc(2, false);
        assert_eq!(a.finish(u64::MAX), Err(RunSummaryError::SizeOverflow));
    }

    #[test]
    fn run_summary_serde_round_trip() {
        for kinds in [true, false] {
            let mut a = acc(1, kinds);
            a.begin_season(0, kinds.then_some(FoodKind::First)).unwrap();
            a.record_physics_sample(1, &rows(1), None).unwrap();
            let s = a.finish(1).unwrap();
            let text = serde_json::to_string(&s).unwrap();
            let back: RunSummary = serde_json::from_str(&text).unwrap();
            assert_eq!(s, back);
        }
    }
}
