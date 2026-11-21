//! Telemetry records derived from supervisor data.
//!
//! Pure computation over standings, boundary details and run-summary totals:
//! no GPU handle, database, file or clock is used. A [`TelemetryBuilder`]
//! turns each supervisor sample into a [`TelemetryFrame`] holding slot
//! snapshots, an optional boundary record, series points, intake rows and a
//! progress record.

use std::collections::BTreeSet;

use serde_json::Value;
use xagent_shared::config::{BrainConfig, FoodKind, SUPERVISOR_SAMPLE_TICKS};

use crate::run_summary::KindWindowTotals;
use crate::supervisor::{BoundaryDetail, SlotStanding};

/// Samples between persisted frames when nothing else forces one.
pub const TELEMETRY_CADENCE_SAMPLES: u64 = 10;

/// One configuration field that differs between parent and newborn.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfigFieldDiff {
    /// Field name.
    pub field: String,
    /// Value in the parent's configuration.
    pub parent: Value,
    /// Value in the newborn's configuration.
    pub newborn: Value,
}

/// Lists the top-level fields whose values differ, in key order. A field
/// missing on one side reads as null. Empty when either side does not
/// serialize to an object.
#[must_use]
pub fn config_diff(parent: &BrainConfig, newborn: &BrainConfig) -> Vec<ConfigFieldDiff> {
    let (Ok(Value::Object(left)), Ok(Value::Object(right))) =
        (serde_json::to_value(parent), serde_json::to_value(newborn))
    else {
        return Vec::new();
    };
    let keys: BTreeSet<&String> = left.keys().chain(right.keys()).collect();
    let mut diffs = Vec::new();
    for key in keys {
        let parent = left.get(key).cloned().unwrap_or(Value::Null);
        let newborn = right.get(key).cloned().unwrap_or(Value::Null);
        if parent != newborn {
            diffs.push(ConfigFieldDiff {
                field: key.clone(),
                parent,
                newborn,
            });
        }
    }
    diffs
}

/// A boundary with the newborn's configuration difference.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryRecord {
    /// What the boundary looked at and decided.
    pub detail: BoundaryDetail,
    /// Occupant id of the newborn, for a replacement with a newborn input.
    pub newborn_occupant: Option<u64>,
    /// Fields that differ between parent and newborn configurations.
    pub config_diff: Vec<ConfigFieldDiff>,
    /// Ticks the retired occupant lived, for a replacement.
    pub retired_lifespan_ticks: Option<u64>,
}

/// A named series.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Meals per agent per tick since the previous persisted frame.
    FoodPerAgentTick,
    /// Deaths per agent per tick since the previous persisted frame.
    DeathsPerAgentTick,
    /// Spread of the finite window scores.
    WindowScoreSpread,
    /// Ticks the retired occupant lived.
    RetiredLifespanTicks,
    /// Best archive outcome.
    ArchiveBestOutcome,
    /// Median archive outcome.
    ArchiveMedianOutcome,
    /// Mean turn toward nourishing food over the closed late windows.
    LateTowardNourishing,
    /// Mean turn away from harmful food over the closed late windows.
    LateAwayHarmful,
    /// Share of late-window meals that were nourishing.
    LateNourishingShare,
}

impl Metric {
    /// The stored name of the metric.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FoodPerAgentTick => "food_per_agent_tick",
            Self::DeathsPerAgentTick => "deaths_per_agent_tick",
            Self::WindowScoreSpread => "window_score_spread",
            Self::RetiredLifespanTicks => "retired_lifespan_ticks",
            Self::ArchiveBestOutcome => "archive_best_outcome",
            Self::ArchiveMedianOutcome => "archive_median_outcome",
            Self::LateTowardNourishing => "late_toward_nourishing",
            Self::LateAwayHarmful => "late_away_harmful",
            Self::LateNourishingShare => "late_nourishing_share",
        }
    }
}

/// One point of a series.
#[derive(Clone, Debug, PartialEq)]
pub struct SeriesPoint {
    /// Tick of the point.
    pub tick: u64,
    /// Series the point belongs to.
    pub metric: Metric,
    /// Value of the point.
    pub value: f64,
}

/// Intake of one occupant at a persisted frame.
#[derive(Clone, Debug, PartialEq)]
pub struct IntakeRow {
    /// Occupant id.
    pub occupant_id: u64,
    /// Slot index.
    pub slot: usize,
    /// Ticks the occupant has lived.
    pub age_ticks: u64,
    /// Meals in the occupant's life.
    pub meals: u32,
}

/// Run progress at a sample.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgressRecord {
    /// Sample tick.
    pub tick: u64,
    /// Total ticks of the run, when known.
    pub total_ticks: Option<u64>,
    /// Agent-tick budget of the run.
    pub agent_tick_budget: u64,
    /// Ticks until the next boundary.
    pub ticks_to_boundary: u64,
    /// Currently nourishing kind, when food kinds are on.
    pub nourishing: Option<FoodKind>,
    /// Tick of the next season flip, when there is one.
    pub next_flip_tick: Option<u64>,
    /// Observed speed, when known.
    pub ticks_per_second: Option<f64>,
}

/// Everything telemetry derives from one sample.
#[derive(Clone, Debug, PartialEq)]
pub struct TelemetryFrame {
    /// Sample tick.
    pub tick: u64,
    /// Standing of every slot.
    pub slots: Vec<SlotStanding>,
    /// The boundary decided at this sample, if any.
    pub boundary: Option<BoundaryRecord>,
    /// Series points; empty unless the frame is persisted.
    pub series: Vec<SeriesPoint>,
    /// Intake rows; empty unless the frame is persisted.
    pub intake: Vec<IntakeRow>,
    /// Run progress.
    pub progress: ProgressRecord,
    /// Whether the frame is to be persisted.
    pub persist: bool,
}

/// The newborn of a replacement.
#[derive(Clone, Copy, Debug)]
pub struct NewbornInput<'a> {
    /// Occupant id of the newborn.
    pub newborn_occupant: u64,
    /// Configuration of the parent.
    pub parent_config: &'a BrainConfig,
    /// Configuration of the newborn.
    pub newborn_config: &'a BrainConfig,
}

/// Inputs of one sample.
#[derive(Clone, Copy, Debug)]
pub struct SampleInput<'a> {
    /// Sample tick.
    pub tick: u64,
    /// Standing of every slot.
    pub standings: &'a [SlotStanding],
    /// The boundary decided at this sample, if any.
    pub boundary: Option<&'a BoundaryDetail>,
    /// The newborn, when the boundary replaced a slot.
    pub replacement: Option<NewbornInput<'a>>,
    /// Archive outcomes in rank order.
    pub archive_outcomes: &'a [f32],
    /// Total ticks of the run, when known.
    pub total_ticks: Option<u64>,
    /// Agent-tick budget of the run.
    pub agent_tick_budget: u64,
    /// Tick of the next boundary.
    pub next_boundary_tick: u64,
    /// Currently nourishing kind.
    pub nourishing: Option<FoodKind>,
    /// Tick of the next season flip.
    pub next_flip_tick: Option<u64>,
    /// Observed speed.
    pub ticks_per_second: Option<f64>,
    /// Whether a season flipped at this tick.
    pub season_flipped: bool,
}

/// Builds telemetry frames from samples.
#[derive(Debug)]
pub struct TelemetryBuilder {
    agent_count: usize,
    prev_meals: Vec<u32>,
    prev_deaths: Vec<u32>,
    prev_occupant: Vec<Option<u64>>,
    prev_tick: u64,
    seen: KindWindowTotals,
    queued: Vec<SeriesPoint>,
}

/// Converts a count to `f64`; precision loss only matters beyond 2^53.
#[allow(clippy::cast_precision_loss)]
fn to_f64(value: u64) -> f64 {
    value as f64
}

fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    let value = numerator / denominator;
    (denominator != 0.0 && value.is_finite()).then_some(value)
}

impl TelemetryBuilder {
    /// Creates a builder for `agent_count` slots with all counters at zero.
    #[must_use]
    pub fn new(agent_count: usize) -> Self {
        Self {
            agent_count,
            prev_meals: vec![0; agent_count],
            prev_deaths: vec![0; agent_count],
            prev_occupant: vec![None; agent_count],
            prev_tick: 0,
            seen: KindWindowTotals {
                late_toward_nourishing_sum: 0.0,
                late_toward_nourishing_samples: 0,
                late_away_harmful_sum: 0.0,
                late_away_harmful_samples: 0,
                late_nourishing_meals: 0,
                late_harmful_meals: 0,
                early_nourishing_meals: 0,
                early_harmful_meals: 0,
            },
            queued: Vec::new(),
        }
    }

    /// Records the closed-season totals as of `tick` and queues the late-window
    /// points for the next frame.
    pub fn on_season_closed(&mut self, tick: u64, totals: &KindWindowTotals) {
        let prev = &self.seen;
        let toward_samples = totals
            .late_toward_nourishing_samples
            .saturating_sub(prev.late_toward_nourishing_samples);
        let toward_sum = totals.late_toward_nourishing_sum - prev.late_toward_nourishing_sum;
        let away_samples = totals
            .late_away_harmful_samples
            .saturating_sub(prev.late_away_harmful_samples);
        let away_sum = totals.late_away_harmful_sum - prev.late_away_harmful_sum;
        let nourishing = totals
            .late_nourishing_meals
            .saturating_sub(prev.late_nourishing_meals);
        let harmful = totals
            .late_harmful_meals
            .saturating_sub(prev.late_harmful_meals);
        let mut points = Vec::new();
        let mut push = |metric, value: Option<f64>| {
            if let Some(value) = value.filter(|v| v.is_finite()) {
                points.push(SeriesPoint {
                    tick,
                    metric,
                    value,
                });
            }
        };
        push(
            Metric::LateTowardNourishing,
            ratio(toward_sum, to_f64(toward_samples)),
        );
        push(
            Metric::LateAwayHarmful,
            ratio(away_sum, to_f64(away_samples)),
        );
        push(
            Metric::LateNourishingShare,
            ratio(
                to_f64(nourishing),
                to_f64(nourishing.saturating_add(harmful)),
            ),
        );
        self.queued.extend(points);
        self.seen = totals.clone();
    }

    /// Builds the frame for one sample.
    pub fn on_sample(&mut self, input: SampleInput<'_>) -> TelemetryFrame {
        let tick = input.tick;
        let cadence = SUPERVISOR_SAMPLE_TICKS.saturating_mul(TELEMETRY_CADENCE_SAMPLES);
        let persist = tick % cadence == 0 || input.boundary.is_some() || input.season_flipped;

        let boundary = input.boundary.map(|detail| {
            let mut record = BoundaryRecord {
                detail: detail.clone(),
                newborn_occupant: None,
                config_diff: Vec::new(),
                retired_lifespan_ticks: None,
            };
            if let BoundaryDetail::Replaced {
                tick: boundary_tick,
                retired_born_tick,
                ..
            } = detail
            {
                record.retired_lifespan_ticks = boundary_tick.checked_sub(*retired_born_tick);
                if let Some(newborn) = input.replacement {
                    record.newborn_occupant = Some(newborn.newborn_occupant);
                    record.config_diff = config_diff(newborn.parent_config, newborn.newborn_config);
                }
            }
            record
        });

        let progress = ProgressRecord {
            tick,
            total_ticks: input.total_ticks,
            agent_tick_budget: input.agent_tick_budget,
            ticks_to_boundary: input.next_boundary_tick.saturating_sub(tick),
            nourishing: input.nourishing,
            next_flip_tick: input.next_flip_tick,
            ticks_per_second: input.ticks_per_second,
        };

        let mut series = Vec::new();
        let mut intake = Vec::new();
        if persist {
            self.persisted_parts(&input, boundary.as_ref(), &mut series, &mut intake);
            series.append(&mut self.queued);
        }

        TelemetryFrame {
            tick,
            slots: input.standings.to_vec(),
            boundary,
            series,
            intake,
            progress,
            persist,
        }
    }

    fn persisted_parts(
        &mut self,
        input: &SampleInput<'_>,
        boundary: Option<&BoundaryRecord>,
        series: &mut Vec<SeriesPoint>,
        intake: &mut Vec<IntakeRow>,
    ) {
        let tick = input.tick;
        let mut push = |metric, value: Option<f64>| {
            if let Some(value) = value.filter(|v| v.is_finite()) {
                series.push(SeriesPoint {
                    tick,
                    metric,
                    value,
                });
            }
        };

        let mut meal_sum = 0_u64;
        let mut death_sum = 0_u64;
        for standing in input.standings {
            intake.push(IntakeRow {
                occupant_id: standing.occupant_id,
                slot: standing.slot,
                age_ticks: standing.age_ticks,
                meals: standing.meals,
            });
            if standing.slot >= self.agent_count {
                continue;
            }
            let changed = self.prev_occupant.get(standing.slot).copied().flatten()
                != Some(standing.occupant_id);
            let before_meals = self.prev_meals.get(standing.slot).copied().unwrap_or(0);
            let before_deaths = self.prev_deaths.get(standing.slot).copied().unwrap_or(0);
            let (before_meals, before_deaths) = if changed {
                (0, 0)
            } else {
                (before_meals, before_deaths)
            };
            meal_sum =
                meal_sum.saturating_add(u64::from(standing.meals.saturating_sub(before_meals)));
            death_sum =
                death_sum.saturating_add(u64::from(standing.deaths.saturating_sub(before_deaths)));
        }
        let span = tick.saturating_sub(self.prev_tick);
        let denominator =
            to_f64(u64::try_from(self.agent_count).unwrap_or(u64::MAX)) * to_f64(span);
        push(
            Metric::FoodPerAgentTick,
            ratio(to_f64(meal_sum), denominator),
        );
        push(
            Metric::DeathsPerAgentTick,
            ratio(to_f64(death_sum), denominator),
        );
        for standing in input.standings {
            if let (Some(meals), Some(deaths), Some(occupant)) = (
                self.prev_meals.get_mut(standing.slot),
                self.prev_deaths.get_mut(standing.slot),
                self.prev_occupant.get_mut(standing.slot),
            ) {
                *meals = standing.meals;
                *deaths = standing.deaths;
                *occupant = Some(standing.occupant_id);
            }
        }
        self.prev_tick = tick;

        let scores: Vec<f64> = input
            .standings
            .iter()
            .filter_map(|s| s.window_score)
            .filter(|s| s.is_finite())
            .map(f64::from)
            .collect();
        if scores.len() >= 2 {
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
            push(Metric::WindowScoreSpread, Some(max - min));
        }

        if let Some(BoundaryRecord {
            detail: BoundaryDetail::Replaced { .. },
            retired_lifespan_ticks: Some(lifespan),
            ..
        }) = boundary
        {
            push(Metric::RetiredLifespanTicks, Some(to_f64(*lifespan)));
        }

        let finite: Vec<f32> = input
            .archive_outcomes
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        push(
            Metric::ArchiveBestOutcome,
            finite.first().copied().map(f64::from),
        );
        push(
            Metric::ArchiveMedianOutcome,
            crate::supervisor::median(finite).map(f64::from),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::{EligibilityGround, ParentRef, ReplaceReason, TieBreak};

    fn standing(
        slot: usize,
        occupant: u64,
        meals: u32,
        deaths: u32,
        score: Option<f32>,
    ) -> SlotStanding {
        SlotStanding {
            slot,
            occupant_id: occupant,
            age_ticks: 500,
            alive: true,
            meals,
            deaths,
            window_score: score,
            peer_median: None,
            ground: EligibilityGround::NoFullWindow,
            parent: None,
        }
    }

    fn replaced(tick: u64, born: u64) -> BoundaryDetail {
        BoundaryDetail::Replaced {
            tick,
            candidates: Vec::new(),
            slot: 0,
            reason: ReplaceReason::Death,
            tiebreak: TieBreak {
                pool: vec![0],
                drawn_index: 0,
            },
            parent: ParentRef::PeerBirth(1),
            parent_draw: None,
            retired_occupant: 1,
            retired_born_tick: born,
        }
    }

    fn input<'a>(tick: u64, standings: &'a [SlotStanding]) -> SampleInput<'a> {
        SampleInput {
            tick,
            standings,
            boundary: None,
            replacement: None,
            archive_outcomes: &[],
            total_ticks: None,
            agent_tick_budget: 1000,
            next_boundary_tick: 5000,
            nourishing: None,
            next_flip_tick: None,
            ticks_per_second: None,
            season_flipped: false,
        }
    }

    fn value(frame: &TelemetryFrame, metric: Metric) -> Option<f64> {
        frame
            .series
            .iter()
            .find(|p| p.metric == metric)
            .map(|p| p.value)
    }

    fn close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("point present");
        assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
    }

    fn totals(
        toward_sum: f64,
        toward_samples: u64,
        away_sum: f64,
        away_samples: u64,
        nourishing_meals: u64,
        harmful_meals: u64,
    ) -> KindWindowTotals {
        KindWindowTotals {
            late_toward_nourishing_sum: toward_sum,
            late_toward_nourishing_samples: toward_samples,
            late_away_harmful_sum: away_sum,
            late_away_harmful_samples: away_samples,
            late_nourishing_meals: nourishing_meals,
            late_harmful_meals: harmful_meals,
            early_nourishing_meals: 0,
            early_harmful_meals: 0,
        }
    }

    #[test]
    fn config_diff_lists_changed_fields_in_key_order() {
        let base = BrainConfig::default();
        assert!(config_diff(&base, &base.clone()).is_empty());
        let mut other = base.clone();
        other.memory_capacity += 1;
        other.learning_rate += 0.25;
        let diff = config_diff(&base, &other);
        let names: Vec<&str> = diff.iter().map(|d| d.field.as_str()).collect();
        assert_eq!(names, ["learning_rate", "memory_capacity"]);
        assert_eq!(
            diff[0].parent,
            serde_json::to_value(base.learning_rate).unwrap()
        );
        assert_eq!(
            diff[0].newborn,
            serde_json::to_value(other.learning_rate).unwrap()
        );
        assert_eq!(
            diff[1].parent,
            serde_json::to_value(base.memory_capacity).unwrap()
        );
        assert_eq!(
            diff[1].newborn,
            serde_json::to_value(other.memory_capacity).unwrap()
        );
    }

    #[test]
    fn rates_plain_and_replacement() {
        let mut builder = TelemetryBuilder::new(2);
        let standings = [standing(0, 10, 4, 0, None), standing(1, 11, 2, 1, None)];
        let frame = builder.on_sample(input(1000, &standings));
        close(value(&frame, Metric::FoodPerAgentTick), 6.0 / 2000.0);
        close(value(&frame, Metric::DeathsPerAgentTick), 1.0 / 2000.0);
        assert_eq!(frame.intake.len(), 2);
        let standings = [standing(0, 10, 6, 0, None), standing(1, 11, 5, 1, None)];
        let frame = builder.on_sample(input(2000, &standings));
        close(value(&frame, Metric::FoodPerAgentTick), 5.0 / 2000.0);
        close(value(&frame, Metric::DeathsPerAgentTick), 0.0);
        let standings = [standing(0, 10, 6, 0, None), standing(1, 12, 1, 0, None)];
        let frame = builder.on_sample(input(3000, &standings));
        close(value(&frame, Metric::FoodPerAgentTick), 1.0 / 2000.0);
        close(value(&frame, Metric::DeathsPerAgentTick), 0.0);
    }

    #[test]
    fn spread_lifespan_archive() {
        let mut builder = TelemetryBuilder::new(3);
        let standings = [
            standing(0, 1, 0, 0, Some(1.0)),
            standing(1, 2, 0, 0, Some(4.0)),
            standing(2, 3, 0, 0, Some(f32::NAN)),
        ];
        let detail = replaced(1000, 400);
        let base = BrainConfig::default();
        let mut other = base.clone();
        other.memory_capacity += 1;
        let mut sample_input = input(1000, &standings);
        sample_input.boundary = Some(&detail);
        sample_input.replacement = Some(NewbornInput {
            newborn_occupant: 9,
            parent_config: &base,
            newborn_config: &other,
        });
        let outcomes = [3.0, f32::NAN, 1.0, 2.0];
        sample_input.archive_outcomes = &outcomes;
        let frame = builder.on_sample(sample_input);
        close(value(&frame, Metric::WindowScoreSpread), 3.0);
        close(value(&frame, Metric::RetiredLifespanTicks), 600.0);
        close(value(&frame, Metric::ArchiveBestOutcome), 3.0);
        close(value(&frame, Metric::ArchiveMedianOutcome), 2.0);
        let record = frame.boundary.unwrap();
        assert_eq!(record.newborn_occupant, Some(9));
        assert_eq!(record.config_diff.len(), 1);
        assert_eq!(record.retired_lifespan_ticks, Some(600));

        let outcomes = [4.0, 1.0, 2.0, 3.0];
        let mut sample_input = input(2000, &standings);
        sample_input.archive_outcomes = &outcomes;
        let frame = builder.on_sample(sample_input);
        close(value(&frame, Metric::ArchiveMedianOutcome), 2.5);
        assert_eq!(value(&frame, Metric::RetiredLifespanTicks), None);
        let frame = builder.on_sample(input(3000, &standings));
        assert_eq!(value(&frame, Metric::ArchiveBestOutcome), None);
        assert_eq!(value(&frame, Metric::ArchiveMedianOutcome), None);
    }

    #[test]
    fn persist_flags_cadence_boundary_and_season() {
        let mut builder = TelemetryBuilder::new(1);
        let standings = [standing(0, 1, 0, 0, None)];
        let frame = builder.on_sample(input(900, &standings));
        assert!(!frame.persist && frame.series.is_empty() && frame.intake.is_empty());
        assert_eq!(frame.slots.len(), 1);
        assert_eq!(frame.progress.ticks_to_boundary, 4100);
        assert!(builder.on_sample(input(1000, &standings)).persist);
        let detail = replaced(900, 0);
        let mut sample_input = input(900, &standings);
        sample_input.boundary = Some(&detail);
        assert!(builder.on_sample(sample_input).persist);
        let mut sample_input = input(900, &standings);
        sample_input.season_flipped = true;
        assert!(builder.on_sample(sample_input).persist);
    }

    #[test]
    fn late_window_points_from_closed_totals() {
        let mut builder = TelemetryBuilder::new(1);
        let standings = [standing(0, 1, 0, 0, None)];
        builder.on_season_closed(1000, &totals(6.0, 3, -3.0, 3, 3, 1));
        let mut sample_input = input(1000, &standings);
        sample_input.season_flipped = true;
        let frame = builder.on_sample(sample_input);
        close(value(&frame, Metric::LateTowardNourishing), 2.0);
        close(value(&frame, Metric::LateAwayHarmful), -1.0);
        close(value(&frame, Metric::LateNourishingShare), 0.75);
        builder.on_season_closed(2000, &totals(8.0, 4, -3.0, 3, 4, 1));
        let mut sample_input = input(2000, &standings);
        sample_input.season_flipped = true;
        let frame = builder.on_sample(sample_input);
        close(value(&frame, Metric::LateTowardNourishing), 2.0);
        assert_eq!(value(&frame, Metric::LateAwayHarmful), None);
        close(value(&frame, Metric::LateNourishingShare), 1.0);
        let frame = builder.on_sample(input(3000, &standings));
        assert_eq!(value(&frame, Metric::LateTowardNourishing), None);
    }

    #[test]
    fn omitted_points_on_zero_denominators_and_nan() {
        let mut builder = TelemetryBuilder::new(0);
        let frame = builder.on_sample(input(1000, &[]));
        assert!(frame.series.is_empty());
        let mut builder = TelemetryBuilder::new(1);
        let standings = [standing(0, 1, 1, 0, Some(f32::NAN))];
        let frame = builder.on_sample(input(1000, &standings));
        assert_eq!(value(&frame, Metric::WindowScoreSpread), None);
        let outcomes = [f32::NAN];
        let mut sample_input = input(1000, &standings);
        sample_input.archive_outcomes = &outcomes;
        let frame = builder.on_sample(sample_input);
        assert_eq!(value(&frame, Metric::ArchiveBestOutcome), None);
        assert_eq!(value(&frame, Metric::FoodPerAgentTick), None);
        builder.on_season_closed(2000, &totals(0.0, 0, 0.0, 0, 0, 0));
        let mut sample_input = input(2000, &standings);
        sample_input.season_flipped = true;
        let frame = builder.on_sample(sample_input);
        assert_eq!(value(&frame, Metric::LateTowardNourishing), None);
        assert_eq!(value(&frame, Metric::LateAwayHarmful), None);
        assert_eq!(value(&frame, Metric::LateNourishingShare), None);
    }
}
