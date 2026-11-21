//! Bounded, pure model of a supervisor run for the observatory views.
//!
//! The model folds one [`ObservatoryRecord`] at a time. Every collection obeys
//! a named cap however many records are folded, so memory and frame time stay
//! bounded on long runs. No GPU handle, database or clock is used here.

use std::collections::{BTreeMap, VecDeque};

use xagent_shared::{BrainConfig, FoodKind};

use crate::supervisor::{
    ArchiveEntry, BoundaryDetail, EligibilityGround as SourceGround, ParentRef as SourceParent,
    SlotStanding,
};
use crate::supervisor_telemetry::{
    BoundaryRecord as SourceBoundary, IntakeRow, Metric, ProgressRecord, TelemetryFrame,
};

pub mod history;
pub mod plots;
pub mod store_read;
pub mod tables;

/// Most slots the model tracks; a run never has more agents than this, and the
/// slot table is sized for it.
pub const MAX_SLOTS: usize = 256;
/// Most points kept per series; bounds chart memory and draw cost on long runs.
pub const MAX_SERIES_POINTS: usize = 4096;
/// Most boundary records kept; the oldest is dropped so the inspector stays small.
pub const MAX_BOUNDARY_RECORDS: usize = 4096;
/// Most timeline events kept; the oldest is dropped so the list stays scrollable.
pub const MAX_TIMELINE_EVENTS: usize = 20_000;
/// Most occupants tracked for lineage and learning views; bounds map size.
pub const MAX_TRACKED_OCCUPANTS: usize = 4096;
/// Most intake points kept per occupant; bounds each learning curve.
pub const MAX_INTAKE_POINTS: usize = 512;
/// Most archive rows kept; matches the size of the browsable archive.
pub const MAX_ARCHIVE_ROWS: usize = 1024;
/// Most bytes of display text kept per string, so one record cannot dominate a frame.
pub const MAX_DISPLAY_TEXT_BYTES: usize = 256;
/// Decimation keeps every element whose index is a multiple of this, which
/// halves an over-full series while preserving its shape.
pub const DECIMATION_STRIDE: usize = 2;

/// Where an inherited parent state comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParentSource {
    /// An archive entry.
    Archive {
        /// Archive entry id.
        entry_id: u64,
    },
    /// A living peer slot.
    Peer {
        /// Peer slot index.
        slot: usize,
    },
}

/// Which state of the parent the newborn inherited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InheritedState {
    /// The parent's birth brain.
    Birth,
    /// The parent's learned brain.
    Learned,
}

/// The parent of an occupant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParentRef {
    /// Occupant id of the parent.
    pub occupant_id: u64,
    /// Where the parent came from.
    pub source: ParentSource,
    /// Which parent state was inherited.
    pub state: InheritedState,
}

/// One row of the slot table.
#[derive(Clone, Debug, PartialEq)]
pub struct SlotRow {
    /// Slot index.
    pub slot: usize,
    /// Occupant of the slot.
    pub occupant_id: u64,
    /// Ticks the occupant has lived.
    pub age_ticks: u64,
    /// Whether the occupant is alive.
    pub alive: bool,
    /// Meals in the occupant's life.
    pub meals: u32,
    /// Deaths in the occupant's life.
    pub deaths: u32,
    /// Window score, when a full window exists.
    pub window_score: Option<f32>,
    /// Median of the peers' window scores.
    pub peer_median: Option<f32>,
    /// Parent of the occupant.
    pub parent: Option<ParentRef>,
}

/// Why a slot was eligible for replacement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EligibilityGround {
    /// The occupant died.
    Died,
    /// The window score is below the peer median.
    BelowMedian {
        /// The slot's window score.
        score: f32,
        /// Median of the peers' scores.
        median: f32,
    },
}

/// A slot that was eligible at a boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct EligibleSlot {
    /// Slot index.
    pub slot: usize,
    /// Why the slot was eligible.
    pub ground: EligibilityGround,
}

/// One configuration field that differs between parent and child.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfigDelta {
    /// Field name.
    pub field: String,
    /// Parent value as text.
    pub parent: String,
    /// Child value as text.
    pub child: String,
}

/// What one boundary looked at and decided.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryRecord {
    /// Boundary tick.
    pub tick: u64,
    /// Slots that were eligible.
    pub eligible: Vec<EligibleSlot>,
    /// The slot picked for replacement, if any.
    pub picked: Option<usize>,
    /// Description of the tie-break draw.
    pub tie_break: Option<String>,
    /// Parent of the newborn.
    pub parent: Option<ParentRef>,
    /// Configuration differences between parent and newborn.
    pub config_diff: Vec<ConfigDelta>,
    /// Why the boundary was idle.
    pub idle_reason: Option<String>,
}

/// One point of the population series.
#[derive(Clone, Debug, PartialEq)]
pub struct SeriesPoint {
    /// Tick of the point.
    pub tick: u64,
    /// Meals per agent per tick.
    pub food_per_agent_tick: f32,
    /// Deaths per agent per tick.
    pub deaths_per_agent_tick: f32,
    /// Spread of the window scores.
    pub score_spread: Option<f32>,
    /// Lifespan of the occupant retired at this tick.
    pub retired_lifespan: Option<u64>,
    /// Best archive outcome.
    pub archive_best: Option<f32>,
    /// Median archive outcome.
    pub archive_median: Option<f32>,
}

/// Late-window learning measures of an occupant.
#[derive(Clone, Debug, PartialEq)]
pub struct LateWindow {
    /// Meals per alive agent in the late window.
    pub meals_per_alive: f32,
    /// Turn toward food in the late window.
    pub turn_toward_food: f32,
    /// Share of sightings that ended in a meal.
    pub sighting_meal_share: f32,
}

/// One tracked occupant.
#[derive(Clone, Debug, PartialEq)]
pub struct OccupantRecord {
    /// Occupant id.
    pub occupant_id: u64,
    /// Parent of the occupant.
    pub parent: Option<ParentRef>,
    /// Tick the occupant was born.
    pub born_tick: u64,
    /// Tick the occupant was retired, while not alive in a slot.
    pub retired_tick: Option<u64>,
    /// Pairs of age ticks and cumulative meals.
    pub intake: Vec<(u64, u32)>,
    /// Late-window measures.
    pub late: Option<LateWindow>,
}

/// One archive row.
#[derive(Clone, Debug, PartialEq)]
pub struct ArchiveRow {
    /// Entry id.
    pub entry_id: u64,
    /// Occupant the entry records.
    pub occupant_id: u64,
    /// Life outcome used for ranking.
    pub outcome: f32,
    /// Ticks the occupant lived.
    pub age_ticks: u64,
    /// Heritable configuration.
    pub config: BrainConfig,
}

/// The current season.
#[derive(Clone, Debug, PartialEq)]
pub struct SeasonState {
    /// Currently nourishing food kind.
    pub nourishing: FoodKind,
    /// Tick of the next flip, when there is one.
    pub next_flip_tick: Option<u64>,
}

/// Run progress.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Progress {
    /// Latest tick.
    pub tick: u64,
    /// Tick budget, when the run has one.
    pub budget_ticks: Option<u64>,
    /// Observed speed, live runs only.
    pub ticks_per_sec: Option<f64>,
    /// Tick of the next boundary.
    pub next_boundary_tick: Option<u64>,
    /// Whether the run finished.
    pub finished: bool,
}

/// Kind of a timeline event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimelineKind {
    /// A slot was replaced.
    Replacement,
    /// A boundary was idle.
    Idle,
    /// The season flipped.
    Season,
    /// The archive changed.
    Archive,
    /// An occupant died.
    Death,
    /// A run-level event.
    Run,
}

/// One timeline event.
#[derive(Clone, Debug, PartialEq)]
pub struct TimelineEvent {
    /// Tick of the event.
    pub tick: u64,
    /// Kind of the event.
    pub kind: TimelineKind,
    /// Display text.
    pub text: String,
}

/// Where the model's records come from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelSource {
    /// A running supervisor.
    #[default]
    Live,
    /// A stored run opened read-only.
    Browse {
        /// Run id in the database.
        run_id: i64,
    },
}

/// One record the model folds.
#[derive(Clone, Debug, PartialEq)]
pub enum ObservatoryRecord {
    /// Snapshot of one slot.
    Slot(SlotRow),
    /// A boundary.
    Boundary(BoundaryRecord),
    /// A series point.
    Series(SeriesPoint),
    /// An occupant; upserts everything but the intake.
    Occupant(OccupantRecord),
    /// Intake of one occupant at an age.
    Intake {
        /// Occupant id.
        occupant_id: u64,
        /// Ticks the occupant has lived.
        age_ticks: u64,
        /// Cumulative meals.
        meals: u32,
    },
    /// An archive row.
    Archive(ArchiveRow),
    /// The season.
    Season(SeasonState),
    /// Run progress.
    Progress(Progress),
    /// A timeline event.
    Event(TimelineEvent),
}

/// The bounded observatory model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObservatoryModel {
    /// One row per slot, at most [`MAX_SLOTS`].
    pub slots: Vec<SlotRow>,
    /// Boundary records, oldest first.
    pub boundaries: VecDeque<BoundaryRecord>,
    /// Series points in tick order.
    pub series: Vec<SeriesPoint>,
    /// Tracked occupants by id.
    pub occupants: BTreeMap<u64, OccupantRecord>,
    /// Archive rows.
    pub archive: Vec<ArchiveRow>,
    /// Current season, when known.
    pub season: Option<SeasonState>,
    /// Run progress.
    pub progress: Progress,
    /// Timeline events, oldest first.
    pub timeline: VecDeque<TimelineEvent>,
    /// Source of the records.
    pub source: ModelSource,
}

/// Keeps every element whose index is a multiple of [`DECIMATION_STRIDE`] plus
/// the last element, once `items` exceeds `cap`.
fn decimate<T>(items: &mut Vec<T>, cap: usize) {
    if items.len() <= cap {
        return;
    }
    let last = items.len() - 1;
    let mut index = 0;
    items.retain(|_| {
        let keep = index % DECIMATION_STRIDE == 0 || index == last;
        index += 1;
        keep
    });
}

/// Truncates `text` to at most [`MAX_DISPLAY_TEXT_BYTES`] on a char boundary.
fn truncate_display(text: &mut String) {
    if text.len() <= MAX_DISPLAY_TEXT_BYTES {
        return;
    }
    let mut end = MAX_DISPLAY_TEXT_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
}

impl ObservatoryModel {
    /// Folds one record into the model, keeping every collection within its cap.
    pub fn fold(&mut self, record: ObservatoryRecord) {
        match record {
            ObservatoryRecord::Slot(row) => self.fold_slot(row),
            ObservatoryRecord::Boundary(mut record) => {
                for text in [&mut record.tie_break, &mut record.idle_reason]
                    .into_iter()
                    .flatten()
                {
                    truncate_display(text);
                }
                self.boundaries.push_back(record);
                if self.boundaries.len() > MAX_BOUNDARY_RECORDS {
                    self.boundaries.pop_front();
                }
            }
            ObservatoryRecord::Series(point) => {
                self.series.push(point);
                decimate(&mut self.series, MAX_SERIES_POINTS);
            }
            ObservatoryRecord::Occupant(record) => self.fold_occupant(record),
            ObservatoryRecord::Intake {
                occupant_id,
                age_ticks,
                meals,
            } => {
                if let Some(occupant) = self.occupants.get_mut(&occupant_id) {
                    occupant.intake.push((age_ticks, meals));
                    decimate(&mut occupant.intake, MAX_INTAKE_POINTS);
                }
            }
            ObservatoryRecord::Archive(row) => self.fold_archive(row),
            ObservatoryRecord::Season(season) => self.season = Some(season),
            ObservatoryRecord::Progress(progress) => self.progress = progress,
            ObservatoryRecord::Event(mut event) => {
                truncate_display(&mut event.text);
                self.timeline.push_back(event);
                if self.timeline.len() > MAX_TIMELINE_EVENTS {
                    self.timeline.pop_front();
                }
            }
        }
    }

    fn fold_slot(&mut self, row: SlotRow) {
        if row.slot >= MAX_SLOTS {
            return;
        }
        match self.slots.iter_mut().find(|held| held.slot == row.slot) {
            Some(held) => *held = row,
            None => self.slots.push(row),
        }
    }

    fn fold_occupant(&mut self, record: OccupantRecord) {
        if let Some(held) = self.occupants.get_mut(&record.occupant_id) {
            held.parent = record.parent;
            held.born_tick = record.born_tick;
            held.retired_tick = record.retired_tick;
            held.late = record.late;
            return;
        }
        if self.occupants.len() >= MAX_TRACKED_OCCUPANTS {
            let oldest_retired = self
                .occupants
                .values()
                .filter_map(|held| held.retired_tick.map(|tick| (tick, held.occupant_id)))
                .min();
            match oldest_retired {
                Some((_, id)) => {
                    self.occupants.remove(&id);
                }
                None => return,
            }
        }
        let mut record = record;
        decimate(&mut record.intake, MAX_INTAKE_POINTS);
        self.occupants.insert(record.occupant_id, record);
    }

    fn fold_archive(&mut self, row: ArchiveRow) {
        if let Some(held) = self
            .archive
            .iter_mut()
            .find(|held| held.entry_id == row.entry_id)
        {
            *held = row;
            return;
        }
        if self.archive.len() < MAX_ARCHIVE_ROWS {
            self.archive.push(row);
            return;
        }
        let lowest = self
            .archive
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.outcome
                    .total_cmp(&b.outcome)
                    .then(a.entry_id.cmp(&b.entry_id))
            })
            .map(|(index, held)| (index, held.outcome));
        if let Some((index, outcome)) = lowest {
            if row.outcome.total_cmp(&outcome).is_gt() {
                self.archive[index] = row;
            }
        }
    }
}

impl ObservatoryModel {
    /// Resolves a source parent against the archive rows and slot rows held so
    /// far. Returns `None` when the parent was never seen or was evicted, so
    /// lineage treats the occupant as a root.
    #[must_use]
    pub fn resolve_parent(&self, parent: &SourceParent) -> Option<ParentRef> {
        let (source, state, occupant_id) = match *parent {
            SourceParent::ArchiveEntry(entry_id) => {
                let row = self.archive.iter().find(|row| row.entry_id == entry_id)?;
                (
                    ParentSource::Archive { entry_id },
                    InheritedState::Birth,
                    row.occupant_id,
                )
            }
            SourceParent::PeerBirth(slot) => (
                ParentSource::Peer { slot },
                InheritedState::Birth,
                self.slot_occupant(slot)?,
            ),
            SourceParent::PeerLearned(slot) => (
                ParentSource::Peer { slot },
                InheritedState::Learned,
                self.slot_occupant(slot)?,
            ),
        };
        Some(ParentRef {
            occupant_id,
            source,
            state,
        })
    }

    fn slot_occupant(&self, slot: usize) -> Option<u64> {
        self.slots
            .iter()
            .find(|row| row.slot == slot)
            .map(|row| row.occupant_id)
    }

    /// Builds the slot record for a standing, resolving its parent.
    #[must_use]
    pub fn slot_record(&self, standing: &SlotStanding) -> ObservatoryRecord {
        ObservatoryRecord::Slot(SlotRow {
            slot: standing.slot,
            occupant_id: standing.occupant_id,
            age_ticks: standing.age_ticks,
            alive: standing.alive,
            meals: standing.meals,
            deaths: standing.deaths,
            window_score: standing.window_score,
            peer_median: standing.peer_median,
            parent: standing
                .parent
                .as_ref()
                .and_then(|parent| self.resolve_parent(parent)),
        })
    }

    /// Builds the boundary record for a telemetry boundary, resolving the
    /// parent. Call it before the replaced slot's row is updated so a peer
    /// parent names the occupant that was in the slot at the boundary.
    #[must_use]
    pub fn boundary_record(&self, source: &SourceBoundary) -> ObservatoryRecord {
        let mut record = BoundaryRecord::from(source);
        if let BoundaryDetail::Replaced { parent, .. } = &source.detail {
            record.parent = self.resolve_parent(parent);
        }
        ObservatoryRecord::Boundary(record)
    }
}

fn eligible_slots(standings: &[SlotStanding]) -> Vec<EligibleSlot> {
    standings
        .iter()
        .filter_map(|standing| {
            let ground = match standing.ground {
                SourceGround::Death => EligibilityGround::Died,
                SourceGround::BelowMedian { score, median } => {
                    EligibilityGround::BelowMedian { score, median }
                }
                _ => return None,
            };
            Some(EligibleSlot {
                slot: standing.slot,
                ground,
            })
        })
        .collect()
}

impl From<&SourceBoundary> for BoundaryRecord {
    fn from(source: &SourceBoundary) -> Self {
        let config_diff = source
            .config_diff
            .iter()
            .map(|diff| ConfigDelta {
                field: diff.field.clone(),
                parent: diff.parent.to_string(),
                child: diff.newborn.to_string(),
            })
            .collect();
        match &source.detail {
            BoundaryDetail::Replaced {
                tick,
                candidates,
                slot,
                tiebreak,
                ..
            } => Self {
                tick: *tick,
                eligible: eligible_slots(candidates),
                picked: Some(*slot),
                tie_break: Some(format!(
                    "pool {:?}, drew index {}",
                    tiebreak.pool, tiebreak.drawn_index
                )),
                parent: None,
                config_diff,
                idle_reason: None,
            },
            BoundaryDetail::Idle { tick, reason, .. } => Self {
                tick: *tick,
                eligible: Vec::new(),
                picked: None,
                tie_break: None,
                parent: None,
                config_diff,
                idle_reason: Some(format!(
                    "too young {}, no full window {}, no peer median {}, at or above median {}",
                    reason.too_young,
                    reason.no_full_window,
                    reason.no_peer_median,
                    reason.at_or_above_median
                )),
            },
        }
    }
}

impl From<&IntakeRow> for ObservatoryRecord {
    fn from(row: &IntakeRow) -> Self {
        Self::Intake {
            occupant_id: row.occupant_id,
            age_ticks: row.age_ticks,
            meals: row.meals,
        }
    }
}

impl From<&ProgressRecord> for Progress {
    fn from(record: &ProgressRecord) -> Self {
        Self {
            tick: record.tick,
            budget_ticks: record.total_ticks,
            ticks_per_sec: record.ticks_per_second,
            next_boundary_tick: Some(record.tick.saturating_add(record.ticks_to_boundary)),
            finished: false,
        }
    }
}

impl From<&ProgressRecord> for ObservatoryRecord {
    fn from(record: &ProgressRecord) -> Self {
        Self::Progress(Progress::from(record))
    }
}

impl SeasonState {
    /// The season a progress record carries, when food kinds are on.
    #[must_use]
    pub fn from_progress(record: &ProgressRecord) -> Option<Self> {
        record.nourishing.map(|nourishing| Self {
            nourishing,
            next_flip_tick: record.next_flip_tick,
        })
    }
}

impl From<&ArchiveEntry> for ArchiveRow {
    fn from(entry: &ArchiveEntry) -> Self {
        Self {
            entry_id: entry.entry_id,
            occupant_id: entry.occupant_id,
            outcome: entry.outcome,
            age_ticks: entry.retired_tick.saturating_sub(entry.born_tick),
            config: entry.config.clone(),
        }
    }
}

impl From<&ArchiveEntry> for ObservatoryRecord {
    fn from(entry: &ArchiveEntry) -> Self {
        Self::Archive(ArchiveRow::from(entry))
    }
}

/// Narrows a series value to the model's single precision.
#[allow(clippy::cast_possible_truncation)]
fn to_f32(value: f64) -> f32 {
    value as f32
}

impl ObservatoryRecord {
    /// The series point a telemetry frame carries, or `None` when the frame
    /// carries no series values.
    #[must_use]
    pub fn series_from_frame(frame: &TelemetryFrame) -> Option<Self> {
        if frame.series.is_empty() {
            return None;
        }
        let value = |metric: Metric| {
            frame
                .series
                .iter()
                .find(|point| point.metric == metric)
                .map(|point| to_f32(point.value))
        };
        Some(Self::Series(SeriesPoint {
            tick: frame.tick,
            food_per_agent_tick: value(Metric::FoodPerAgentTick).unwrap_or_default(),
            deaths_per_agent_tick: value(Metric::DeathsPerAgentTick).unwrap_or_default(),
            score_spread: value(Metric::WindowScoreSpread),
            retired_lifespan: frame
                .boundary
                .as_ref()
                .and_then(|boundary| boundary.retired_lifespan_ticks),
            archive_best: value(Metric::ArchiveBestOutcome),
            archive_median: value(Metric::ArchiveMedianOutcome),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Folds enough points to overflow a cap several times over.
    const DECIMATION_FOLD_FACTOR: usize = 3;
    /// Tick used for the sample records.
    const SAMPLE_TICK: u64 = 5;
    /// Occupant id used for the sample occupant.
    const SAMPLE_OCCUPANT: u64 = 7;
    /// Occupant id that is never tracked.
    const UNTRACKED_OCCUPANT: u64 = 99;
    /// Age of the sample intake point.
    const SAMPLE_AGE: u64 = 3;
    /// Meals of the sample intake point.
    const SAMPLE_MEALS: u32 = 2;
    /// Retirement tick of the sample occupant.
    const SAMPLE_RETIRED_TICK: u64 = 9;
    /// Next flip tick of the sample season.
    const SAMPLE_FLIP_TICK: u64 = 10;
    /// Meals per alive agent of the sample late window.
    const SAMPLE_LATE_MEALS: f32 = 0.5;
    /// Turn toward food of the sample late window.
    const SAMPLE_LATE_TURN: f32 = 0.25;
    /// Sighting meal share of the sample late window.
    const SAMPLE_LATE_SHARE: f32 = 0.75;
    /// Higher archive outcome used to replace a row.
    const SAMPLE_HIGH_OUTCOME: f32 = 2.0;
    /// Entry id of the sample archive row.
    const SAMPLE_ENTRY: u64 = 4;
    /// Slot of the sample peer parent.
    const SAMPLE_PEER_SLOT: usize = 2;

    fn count(cap: usize) -> u64 {
        u64::try_from(cap).unwrap_or(u64::MAX)
    }

    fn slot_row(slot: usize) -> SlotRow {
        SlotRow {
            slot,
            occupant_id: 0,
            age_ticks: 0,
            alive: true,
            meals: 0,
            deaths: 0,
            window_score: None,
            peer_median: None,
            parent: None,
        }
    }

    fn series_point(tick: u64) -> SeriesPoint {
        SeriesPoint {
            tick,
            food_per_agent_tick: 0.0,
            deaths_per_agent_tick: 0.0,
            score_spread: None,
            retired_lifespan: None,
            archive_best: None,
            archive_median: None,
        }
    }

    fn occupant(id: u64, retired_tick: Option<u64>) -> OccupantRecord {
        OccupantRecord {
            occupant_id: id,
            parent: None,
            born_tick: 0,
            retired_tick,
            intake: Vec::new(),
            late: None,
        }
    }

    fn archive_row(entry_id: u64, outcome: f32) -> ArchiveRow {
        ArchiveRow {
            entry_id,
            occupant_id: entry_id,
            outcome,
            age_ticks: 0,
            config: BrainConfig::default(),
        }
    }

    fn event(tick: u64, text: String) -> TimelineEvent {
        TimelineEvent {
            tick,
            kind: TimelineKind::Run,
            text,
        }
    }

    #[test]
    fn observatory_model_caps_hold() {
        let mut model = ObservatoryModel::default();
        model.fold(ObservatoryRecord::Slot(slot_row(MAX_SLOTS)));
        assert!(model.slots.is_empty());
        for slot in 0..=MAX_SLOTS {
            model.fold(ObservatoryRecord::Slot(slot_row(slot)));
        }
        assert_eq!(model.slots.len(), MAX_SLOTS);

        let over = count(MAX_TIMELINE_EVENTS) + 1;
        for tick in 0..over {
            model.fold(ObservatoryRecord::Boundary(BoundaryRecord {
                tick,
                eligible: Vec::new(),
                picked: None,
                tie_break: Some("x".repeat(MAX_DISPLAY_TEXT_BYTES + 1)),
                parent: None,
                config_diff: Vec::new(),
                idle_reason: None,
            }));
            model.fold(ObservatoryRecord::Series(series_point(tick)));
            model.fold(ObservatoryRecord::Event(event(
                tick,
                "é".repeat(MAX_DISPLAY_TEXT_BYTES),
            )));
        }
        assert!(model.boundaries.len() <= MAX_BOUNDARY_RECORDS);
        assert!(model.series.len() <= MAX_SERIES_POINTS);
        assert_eq!(model.timeline.len(), MAX_TIMELINE_EVENTS);
        assert!(model.timeline.iter().all(|e| {
            e.text.len() <= MAX_DISPLAY_TEXT_BYTES && e.text.chars().all(|c| c == 'é')
        }));
        assert!(model.boundaries.iter().all(|b| b
            .tie_break
            .as_ref()
            .is_some_and(|t| t.len() <= MAX_DISPLAY_TEXT_BYTES)));

        for id in 0..count(MAX_TRACKED_OCCUPANTS + 1) {
            model.fold(ObservatoryRecord::Occupant(occupant(id, Some(id))));
        }
        assert_eq!(model.occupants.len(), MAX_TRACKED_OCCUPANTS);

        for id in 0..count(MAX_ARCHIVE_ROWS + 1) {
            let outcome = f32::from(u16::try_from(id).unwrap_or(u16::MAX));
            model.fold(ObservatoryRecord::Archive(archive_row(id, outcome)));
        }
        assert_eq!(model.archive.len(), MAX_ARCHIVE_ROWS);
        assert!(model.archive.iter().all(|row| row.entry_id != 0));
    }

    #[test]
    fn observatory_model_decimation_keeps_ends() {
        let mut model = ObservatoryModel::default();
        let last = count(MAX_SERIES_POINTS * DECIMATION_FOLD_FACTOR);
        for tick in 0..=last {
            model.fold(ObservatoryRecord::Series(series_point(tick)));
        }
        assert!(model.series.len() <= MAX_SERIES_POINTS);
        assert_eq!(model.series.first().map(|p| p.tick), Some(0));
        assert_eq!(model.series.last().map(|p| p.tick), Some(last));

        model.fold(ObservatoryRecord::Occupant(occupant(1, None)));
        let last = count(MAX_INTAKE_POINTS * DECIMATION_FOLD_FACTOR);
        for tick in 0..=last {
            model.fold(ObservatoryRecord::Intake {
                occupant_id: 1,
                age_ticks: tick,
                meals: 0,
            });
        }
        let intake = &model.occupants[&1].intake;
        assert!(intake.len() <= MAX_INTAKE_POINTS);
        assert_eq!(intake.first().map(|p| p.0), Some(0));
        assert_eq!(intake.last().map(|p| p.0), Some(last));
    }

    #[test]
    fn observatory_model_living_never_evicted() {
        let mut model = ObservatoryModel::default();
        let living = MAX_TRACKED_OCCUPANTS - 1;
        for id in 0..count(living) {
            model.fold(ObservatoryRecord::Occupant(occupant(id, None)));
        }
        let retired = count(living);
        model.fold(ObservatoryRecord::Occupant(occupant(retired, Some(1))));
        assert_eq!(model.occupants.len(), MAX_TRACKED_OCCUPANTS);

        let newcomer = retired + 1;
        model.fold(ObservatoryRecord::Occupant(occupant(newcomer, None)));
        assert_eq!(model.occupants.len(), MAX_TRACKED_OCCUPANTS);
        assert!(!model.occupants.contains_key(&retired));
        assert!(model.occupants.contains_key(&newcomer));
        assert!((0..count(living)).all(|id| model.occupants.contains_key(&id)));

        let dropped = newcomer + 1;
        model.fold(ObservatoryRecord::Occupant(occupant(dropped, None)));
        assert!(!model.occupants.contains_key(&dropped));
        assert_eq!(model.occupants.len(), MAX_TRACKED_OCCUPANTS);
    }

    #[test]
    fn observatory_model_fold_basic() {
        let mut model = ObservatoryModel::default();
        assert_eq!(model.source, ModelSource::Live);
        assert!(model.season.is_none());

        model.fold(ObservatoryRecord::Slot(slot_row(0)));
        let mut replaced = slot_row(0);
        replaced.occupant_id = SAMPLE_OCCUPANT;
        model.fold(ObservatoryRecord::Slot(replaced));
        assert_eq!(model.slots.len(), 1);
        assert_eq!(model.slots[0].occupant_id, SAMPLE_OCCUPANT);

        model.fold(ObservatoryRecord::Boundary(BoundaryRecord {
            tick: SAMPLE_TICK,
            eligible: Vec::new(),
            picked: Some(0),
            tie_break: None,
            parent: None,
            config_diff: Vec::new(),
            idle_reason: None,
        }));
        assert_eq!(model.boundaries.len(), 1);

        model.fold(ObservatoryRecord::Series(series_point(SAMPLE_TICK)));
        assert_eq!(model.series.len(), 1);

        model.fold(ObservatoryRecord::Occupant(occupant(SAMPLE_OCCUPANT, None)));
        model.fold(ObservatoryRecord::Intake {
            occupant_id: SAMPLE_OCCUPANT,
            age_ticks: SAMPLE_AGE,
            meals: SAMPLE_MEALS,
        });
        let mut retired = occupant(SAMPLE_OCCUPANT, Some(SAMPLE_RETIRED_TICK));
        retired.late = Some(LateWindow {
            meals_per_alive: SAMPLE_LATE_MEALS,
            turn_toward_food: SAMPLE_LATE_TURN,
            sighting_meal_share: SAMPLE_LATE_SHARE,
        });
        model.fold(ObservatoryRecord::Occupant(retired));
        let held = &model.occupants[&SAMPLE_OCCUPANT];
        assert_eq!(held.retired_tick, Some(SAMPLE_RETIRED_TICK));
        assert_eq!(held.intake, vec![(SAMPLE_AGE, SAMPLE_MEALS)]);
        assert!(held.late.is_some());
        model.fold(ObservatoryRecord::Intake {
            occupant_id: UNTRACKED_OCCUPANT,
            age_ticks: 1,
            meals: 1,
        });
        assert!(!model.occupants.contains_key(&UNTRACKED_OCCUPANT));

        model.fold(ObservatoryRecord::Archive(archive_row(1, 1.0)));
        model.fold(ObservatoryRecord::Archive(archive_row(
            1,
            SAMPLE_HIGH_OUTCOME,
        )));
        assert_eq!(model.archive.len(), 1);
        assert!((model.archive[0].outcome - SAMPLE_HIGH_OUTCOME).abs() < f32::EPSILON);

        let season = SeasonState {
            nourishing: FoodKind::First,
            next_flip_tick: Some(SAMPLE_FLIP_TICK),
        };
        model.fold(ObservatoryRecord::Season(season.clone()));
        assert_eq!(model.season, Some(season));

        let progress = Progress {
            tick: SAMPLE_TICK,
            finished: true,
            ..Progress::default()
        };
        model.fold(ObservatoryRecord::Progress(progress.clone()));
        assert_eq!(model.progress, progress);

        model.fold(ObservatoryRecord::Event(event(
            SAMPLE_TICK,
            "run".to_owned(),
        )));
        assert_eq!(model.timeline.len(), 1);
    }

    #[test]
    fn observatory_model_parent_resolves_occupant() {
        let mut model = ObservatoryModel::default();
        assert!(model
            .resolve_parent(&SourceParent::ArchiveEntry(SAMPLE_ENTRY))
            .is_none());
        let mut row = archive_row(SAMPLE_ENTRY, 1.0);
        row.occupant_id = SAMPLE_OCCUPANT;
        model.fold(ObservatoryRecord::Archive(row));
        let mut peer = slot_row(SAMPLE_PEER_SLOT);
        peer.occupant_id = UNTRACKED_OCCUPANT;
        model.fold(ObservatoryRecord::Slot(peer));

        let from_archive = model
            .resolve_parent(&SourceParent::ArchiveEntry(SAMPLE_ENTRY))
            .expect("archive parent resolves");
        assert_eq!(from_archive.occupant_id, SAMPLE_OCCUPANT);
        assert_eq!(from_archive.state, InheritedState::Birth);
        let from_peer = model
            .resolve_parent(&SourceParent::PeerLearned(SAMPLE_PEER_SLOT))
            .expect("peer parent resolves");
        assert_eq!(from_peer.occupant_id, UNTRACKED_OCCUPANT);
        assert_eq!(from_peer.state, InheritedState::Learned);
    }
}
