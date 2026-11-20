//! Pure deterministic core of the continuous-life supervisor.
//!
//! The core owns one ledger per agent slot, a bounded archive of retired
//! lives and the random streams that drive replacement. It sees only
//! per-sample counters; it performs no GPU work, database access or file I/O,
//! so every decision is a pure function of the run seed, the configuration and
//! the sample sequence.
//!
//! Counters are exact `f32` integers; a count above [`COUNT_EXACT_LIMIT`] or a
//! non-finite value makes the score that depends on it NaN, and a NaN score is
//! never eligible, archived or ranked above a finite one. Counters are never
//! clamped.

use std::collections::{BTreeSet, VecDeque};
use std::fmt;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use xagent_brain::buffers::{
    AgentBrainState, PHYS_STRIDE, P_ALIVE, P_DEATH_COUNT, P_DISTANCE_TRAVELED, P_ENERGY_SPENT,
    P_FOOD_COUNT, P_POS_X, P_POS_Z,
};
use xagent_shared::{
    BrainConfig, GovernorConfig, Inheritance, SupervisorConfig, SupervisorConfigError,
    SUPERVISOR_SAMPLE_TICKS,
};

use crate::agent::{
    fresh_brain_state, mutate_sensory_genes, mutate_steering_weights, steering_population,
    HEATMAP_RES,
};
use crate::governor::{composite_fitness, exploration_grid_cells};
use crate::momentum::MutationMomentum;

/// Stream tag of the replacement-boundary schedule.
pub const BOUNDARY_STREAM: u64 = 0xB0A1_D5C4_0000_0001;
/// Stream tag of the food-kind season schedule.
pub const SEASON_STREAM: u64 = 0x5EA5_0000_0000_0002;
/// Stream tag of the candidate tie-break draw.
pub const TIEBREAK_STREAM: u64 = 0x71E8_0000_0000_0003;
/// Stream tag of the parent draw and newborn mutation.
pub const PARENT_STREAM: u64 = 0x9A4E_0000_0000_0004;
/// Stream tag of the initial population.
pub const INITIAL_STREAM: u64 = 0x1A17_0000_0000_0005;

/// Splitmix64 increment.
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// First splitmix64 multiplier.
const SPLITMIX_MUL_1: u64 = 0xBF58_476D_1CE4_E5B9;
/// Second splitmix64 multiplier.
const SPLITMIX_MUL_2: u64 = 0x94D0_49BB_1331_11EB;
/// First splitmix64 xor-shift.
const SPLITMIX_SHIFT_1: u32 = 30;
/// Second splitmix64 xor-shift.
const SPLITMIX_SHIFT_2: u32 = 27;
/// Final splitmix64 xor-shift.
const SPLITMIX_SHIFT_3: u32 = 31;

/// An `alive` physics value above this means the agent is alive.
const ALIVE_THRESHOLD: f32 = 0.5;

/// Largest counter an `f32` holds exactly (2^24).
const COUNT_EXACT_LIMIT: f32 = 16_777_216.0;

/// Derives the seed of one random stream: splitmix64 of `run_seed ^ tag`.
pub fn stream_seed(run_seed: u64, tag: u64) -> u64 {
    let mut z = (run_seed ^ tag).wrapping_add(SPLITMIX_GAMMA);
    z = (z ^ (z >> SPLITMIX_SHIFT_1)).wrapping_mul(SPLITMIX_MUL_1);
    z = (z ^ (z >> SPLITMIX_SHIFT_2)).wrapping_mul(SPLITMIX_MUL_2);
    z ^ (z >> SPLITMIX_SHIFT_3)
}

/// Jittered schedule of tick values on the sample grid.
pub struct TickSchedule {
    rng: SmallRng,
    low: u64,
    high: u64,
    next: u64,
}

impl TickSchedule {
    /// Schedule whose gaps are `SUPERVISOR_SAMPLE_TICKS * k`, with `k` uniform
    /// in `[ceil(m / 2), floor(3 * m / 2)]` and `m = mean_ticks /
    /// SUPERVISOR_SAMPLE_TICKS`. The first tick is one gap after tick 0.
    pub fn new(seed: u64, mean_ticks: u64) -> Self {
        let mean = mean_ticks / SUPERVISOR_SAMPLE_TICKS;
        let low = (mean - mean / HALF_DIVISOR_INT).max(1);
        let high = (mean + mean / HALF_DIVISOR_INT).max(low);
        let mut schedule = Self {
            rng: SmallRng::seed_from_u64(seed),
            low,
            high,
            next: 0,
        };
        schedule.advance();
        schedule
    }

    /// The tick the schedule currently points at.
    pub fn next_tick(&self) -> u64 {
        self.next
    }

    /// Moves to the following tick, one jittered gap later.
    pub fn advance(&mut self) {
        let k = self.rng.random_range(self.low..=self.high);
        self.next = self
            .next
            .saturating_add(SUPERVISOR_SAMPLE_TICKS.saturating_mul(k));
    }
}

/// Why a supervisor call was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum SupervisorError {
    /// The supervisor configuration is infeasible.
    Config(SupervisorConfigError),
    /// The world size is not finite and positive.
    InvalidWorldSize,
    /// A physics row did not have `PHYS_STRIDE` values.
    PhysicsRowLength { expected: usize, actual: usize },
    /// The counter slice did not have one entry per slot.
    CounterLength { expected: usize, actual: usize },
    /// A call named a tick other than the one expected.
    TickMismatch { expected: u64, actual: u64 },
    /// A plan named a slot that does not exist.
    UnknownSlot { slot: usize },
    /// A peer-learned parent was planned but no learned state was supplied.
    LearnedStateRequired,
    /// Learned state was supplied for a parent that does not use it.
    LearnedStateUnexpected,
    /// A plan named an archive entry that is no longer held.
    UnknownArchiveEntry { entry_id: u64 },
}

impl fmt::Display for SupervisorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(err) => write!(f, "invalid supervisor config: {err}"),
            Self::InvalidWorldSize => write!(f, "world size must be finite and positive"),
            Self::PhysicsRowLength { expected, actual } => {
                write!(f, "physics row has {actual} values, expected {expected}")
            }
            Self::CounterLength { expected, actual } => {
                write!(f, "got {actual} slot counters, expected {expected}")
            }
            Self::TickMismatch { expected, actual } => {
                write!(f, "tick {actual} does not match expected tick {expected}")
            }
            Self::UnknownSlot { slot } => write!(f, "slot {slot} does not exist"),
            Self::LearnedStateRequired => write!(f, "parent needs a learned state"),
            Self::LearnedStateUnexpected => write!(f, "parent takes no learned state"),
            Self::UnknownArchiveEntry { entry_id } => {
                write!(f, "archive entry {entry_id} is not held")
            }
        }
    }
}

impl std::error::Error for SupervisorError {}

/// Cumulative counters of one slot's current life at one sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotCounters {
    /// Whether the agent was alive at the sample.
    pub alive: bool,
    /// Deaths so far.
    pub deaths: f32,
    /// Food items consumed so far.
    pub food: f32,
    /// Distance travelled so far.
    pub distance: f32,
    /// Energy spent so far.
    pub energy_spent: f32,
    /// Heatmap cell of the position, `None` while dead or non-finite.
    pub cell: Option<u16>,
}

impl SlotCounters {
    /// The zeroed spawn row every life is compared against first.
    const ZERO: Self = Self {
        alive: false,
        deaths: 0.0,
        food: 0.0,
        distance: 0.0,
        energy_spent: 0.0,
        cell: None,
    };

    /// Reads the counters from one physics row of a world `world_size` wide.
    pub fn from_physics(row: &[f32], world_size: f32) -> Result<Self, SupervisorError> {
        if row.len() != PHYS_STRIDE {
            return Err(SupervisorError::PhysicsRowLength {
                expected: PHYS_STRIDE,
                actual: row.len(),
            });
        }
        let alive = row[P_ALIVE] > ALIVE_THRESHOLD;
        let (x, z) = (row[P_POS_X], row[P_POS_Z]);
        let cell = if alive && x.is_finite() && z.is_finite() {
            heatmap_cell(x, z, world_size)
        } else {
            None
        };
        Ok(Self {
            alive,
            deaths: row[P_DEATH_COUNT],
            food: row[P_FOOD_COUNT],
            distance: row[P_DISTANCE_TRAVELED],
            energy_spent: row[P_ENERGY_SPENT],
            cell,
        })
    }
}

/// Divisor that halves an integer (schedule bounds, median middle index and
/// parity).
const HALF_DIVISOR_INT: u64 = 2;

/// Divisor that halves an `f32` (the world is centred on the origin, so its
/// half extent is the offset to the grid origin; the median of an even count
/// is the mean of the middle pair).
const HALF_DIVISOR_F32: f32 = 2.0;

/// Heatmap cell index of a position, with the expressions and clamps of
/// `Agent::record_heatmap`.
fn heatmap_cell(x: f32, z: f32, world_size: f32) -> Option<u16> {
    let half = world_size / HALF_DIVISOR_F32;
    let cell = world_size / HEATMAP_RES as f32;
    let cx = (((x + half) / cell) as usize).min(HEATMAP_RES - 1);
    let cz = (((z + half) / cell) as usize).min(HEATMAP_RES - 1);
    u16::try_from(cz * HEATMAP_RES + cx).ok()
}

/// A counter as an integer, or `None` unless finite, non-negative, integral
/// and exactly representable.
fn count_u32(value: f32) -> Option<u32> {
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= COUNT_EXACT_LIMIT {
        Some(value as u32)
    } else {
        None
    }
}

/// `value` when finite and non-negative.
fn finite_non_negative(value: f32) -> Option<f32> {
    (value.is_finite() && value >= 0.0).then_some(value)
}

/// Growth of an integer counter between two samples.
fn count_delta(first: f32, last: f32) -> Option<u32> {
    count_u32(last)?.checked_sub(count_u32(first)?)
}

/// A distinct-cell count as the `u32` the fitness function takes.
fn cell_count(len: usize) -> u32 {
    u32::try_from(len).unwrap_or(u32::MAX)
}

/// Median of the values; the mean of the middle two when the count is even.
fn median(mut values: Vec<f32>) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f32::total_cmp);
    let half = usize::try_from(HALF_DIVISOR_INT).unwrap_or(usize::MAX);
    let mid = values.len() / half;
    if values.len() % half == 1 {
        Some(values[mid])
    } else {
        Some((values[mid - 1] + values[mid]) / HALF_DIVISOR_F32)
    }
}

/// Everything the supervisor remembers about one slot's current occupant.
pub struct SlotLedger {
    /// Identifier of the occupant, monotone from 0.
    pub occupant_id: u64,
    /// Tick the occupant was born.
    pub born_tick: u64,
    /// The brain the occupant was born with.
    pub birth_brain: AgentBrainState,
    /// The heritable configuration the occupant was born with.
    pub config: BrainConfig,
    pending_death: bool,
    window_len: usize,
    window: VecDeque<SlotCounters>,
    life_cells: BTreeSet<u16>,
}

/// Cap on the window samples a ledger preallocates, so a huge configured
/// window cannot reserve memory before any sample exists.
const LEDGER_PREALLOC_SAMPLES: usize = 1024;

impl SlotLedger {
    fn born(
        occupant_id: u64,
        born_tick: u64,
        birth_brain: AgentBrainState,
        config: BrainConfig,
        window_len: usize,
    ) -> Self {
        let mut window = VecDeque::with_capacity(window_len.min(LEDGER_PREALLOC_SAMPLES));
        window.push_back(SlotCounters::ZERO);
        Self {
            occupant_id,
            born_tick,
            birth_brain,
            config,
            pending_death: false,
            window_len,
            window,
            life_cells: BTreeSet::new(),
        }
    }

    /// Ticks lived at `tick`.
    pub fn age(&self, tick: u64) -> u64 {
        tick.saturating_sub(self.born_tick)
    }

    fn record(&mut self, counters: SlotCounters) {
        let previous = self.window.back().map_or(0.0, |sample| sample.deaths);
        if counters.deaths > previous {
            self.pending_death = true;
        }
        if let Some(cell) = counters.cell {
            self.life_cells.insert(cell);
        }
        self.window.push_back(counters);
        while self.window.len() > self.window_len {
            self.window.pop_front();
        }
    }

    /// Window score, or `None` until the ring holds a full window.
    fn full_window_score(&self) -> Option<f32> {
        if self.window.len() != self.window_len {
            return None;
        }
        Some(self.window_score())
    }

    fn window_score(&self) -> f32 {
        let (Some(first), Some(last)) = (self.window.front(), self.window.back()) else {
            return f32::NAN;
        };
        let deaths = count_delta(first.deaths, last.deaths);
        let food = count_delta(first.food, last.food);
        let distance = finite_non_negative(last.distance - first.distance);
        let energy = finite_non_negative(last.energy_spent - first.energy_spent);
        let cells: BTreeSet<u16> = self.window.iter().skip(1).filter_map(|s| s.cell).collect();
        match (deaths, food, distance, energy) {
            (Some(d), Some(f), Some(dist), Some(en)) => composite_fitness(
                d,
                f,
                cell_count(cells.len()),
                exploration_grid_cells(),
                dist,
                en,
            ),
            _ => f32::NAN,
        }
    }

    /// Outcome of the whole life so far.
    fn life_outcome(&self) -> f32 {
        let Some(last) = self.window.back() else {
            return f32::NAN;
        };
        let distance = finite_non_negative(last.distance);
        let energy = finite_non_negative(last.energy_spent);
        match (
            count_u32(last.deaths),
            count_u32(last.food),
            distance,
            energy,
        ) {
            (Some(d), Some(f), Some(dist), Some(en)) => composite_fitness(
                d,
                f,
                cell_count(self.life_cells.len()),
                exploration_grid_cells(),
                dist,
                en,
            ),
            _ => f32::NAN,
        }
    }
}

/// Why a slot is replaced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplaceReason {
    /// The occupant died since it was born.
    Death,
    /// The occupant's window score is below its peers' median.
    Underperformer,
}

/// Slots eligible at a boundary: death-eligible and underperformers, each in
/// slot order.
fn eligible_slots(
    slots: &[SlotLedger],
    tick: u64,
    config: &SupervisorConfig,
) -> (Vec<usize>, Vec<usize>) {
    let scores: Vec<Option<f32>> = slots.iter().map(SlotLedger::full_window_score).collect();
    let (mut deaths, mut under) = (Vec::new(), Vec::new());
    for (index, slot) in slots.iter().enumerate() {
        if slot.pending_death {
            deaths.push(index);
            continue;
        }
        let Some(own) = scores[index] else {
            continue;
        };
        if slot.age(tick) < config.min_age_ticks {
            continue;
        }
        let peers: Vec<f32> = scores
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .filter_map(|(_, score)| *score)
            .filter(|score| !score.is_nan())
            .collect();
        if median(peers).is_some_and(|middle| own < middle) {
            under.push(index);
        }
    }
    (deaths, under)
}

/// Picks the slot to replace: death-eligible first, else underperformers, with
/// exactly one draw when any candidate exists.
fn choose_candidate(
    deaths: &[usize],
    under: &[usize],
    rng: &mut SmallRng,
) -> Option<(usize, ReplaceReason)> {
    let (pool, reason) = if !deaths.is_empty() {
        (deaths, ReplaceReason::Death)
    } else if !under.is_empty() {
        (under, ReplaceReason::Underperformer)
    } else {
        return None;
    };
    Some((pool[rng.random_range(0..pool.len())], reason))
}

/// One retired life held by the archive.
#[derive(Clone, Debug)]
pub struct ArchiveEntry {
    /// Identifier, monotone from 1.
    pub entry_id: u64,
    /// Occupant the entry records.
    pub occupant_id: u64,
    /// Tick the occupant was born.
    pub born_tick: u64,
    /// Tick the occupant was retired.
    pub retired_tick: u64,
    /// Life outcome used for ranking.
    pub outcome: f32,
    /// Heritable configuration the occupant was born with.
    pub config: BrainConfig,
    /// Brain the occupant was born with.
    pub birth_brain: AgentBrainState,
}

/// Result of offering an entry to the archive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// The entry was not archived.
    Refused,
    /// The entry was archived, evicting the named entry when full.
    Admitted { evicted: Option<u64> },
}

/// Bounded archive of retired lives in rank order.
pub struct Archive {
    capacity: usize,
    entries: Vec<ArchiveEntry>,
    next_id: u64,
}

impl Archive {
    /// Empty archive holding at most `capacity` entries.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::new(),
            next_id: 1,
        }
    }

    /// The identifier the next admitted entry should carry.
    pub fn next_entry_id(&self) -> u64 {
        self.next_id
    }

    /// Offers `entry`: admitted below capacity, or when full and its outcome
    /// beats the worst held entry, which is evicted. Non-finite outcomes are
    /// refused.
    pub fn insert(&mut self, entry: ArchiveEntry) -> Admission {
        if !entry.outcome.is_finite() || self.capacity == 0 {
            return Admission::Refused;
        }
        let mut evicted = None;
        if self.entries.len() >= self.capacity {
            let beats_worst = self
                .entries
                .last()
                .is_some_and(|worst| entry.outcome.total_cmp(&worst.outcome).is_gt());
            if !beats_worst {
                return Admission::Refused;
            }
            evicted = self.entries.pop().map(|worst| worst.entry_id);
        }
        self.next_id = self.next_id.max(entry.entry_id.saturating_add(1));
        self.entries.push(entry);
        self.entries.sort_by(|a, b| {
            b.outcome
                .total_cmp(&a.outcome)
                .then(a.entry_id.cmp(&b.entry_id))
        });
        Admission::Admitted { evicted }
    }

    /// Entries in rank order: outcome descending, then entry id ascending.
    pub fn entries(&self) -> &[ArchiveEntry] {
        &self.entries
    }

    /// The entry with identifier `entry_id`.
    pub fn get(&self, entry_id: u64) -> Option<&ArchiveEntry> {
        self.entries.iter().find(|e| e.entry_id == entry_id)
    }

    /// Number of held entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the archive holds nothing.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Where a newborn's parent comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParentRef {
    /// An archive entry, by id; its birth brain is the parent.
    ArchiveEntry(u64),
    /// The birth brain of a living peer slot.
    PeerBirth(usize),
    /// The current learned state of a living peer slot, supplied by the driver.
    PeerLearned(usize),
}

/// A replacement chosen at a boundary, completed by the driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplacementPlan {
    /// Slot to replace.
    pub slot: usize,
    /// Why the slot is replaced.
    pub reason: ReplaceReason,
    /// Parent of the newborn.
    pub parent: ParentRef,
    /// Boundary tick the plan was made at.
    pub tick: u64,
}

/// What one recorded sample decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleDecision {
    /// The sample was recorded; it is not a boundary.
    Recorded,
    /// A boundary passed with no eligible slot.
    Idle,
    /// A boundary chose a slot to replace.
    Replace(ReplacementPlan),
}

/// A completed replacement, with everything the store writes.
#[derive(Clone, Debug)]
pub struct ReplacementOutcome {
    /// The newborn's brain.
    pub newborn_brain: AgentBrainState,
    /// The newborn's heritable configuration.
    pub newborn_config: BrainConfig,
    /// Occupant id given to the newborn.
    pub occupant_id: u64,
    /// The retired occupant's archive entry, when it was admitted.
    pub archived: Option<ArchiveEntry>,
    /// Entry evicted by the admission.
    pub evicted: Option<u64>,
}

/// One archive admission made at the run end.
#[derive(Clone, Debug)]
pub struct ArchiveChange {
    /// Slot whose occupant was archived.
    pub slot: usize,
    /// The admitted entry.
    pub entry: ArchiveEntry,
    /// Entry evicted by the admission.
    pub evicted: Option<u64>,
}

/// The supervisor: ledgers, archive, schedule and random streams.
pub struct Supervisor {
    config: SupervisorConfig,
    mutation_strength: f32,
    momentum_decay: f32,
    world_size: f32,
    slots: Vec<SlotLedger>,
    archive: Archive,
    boundary: TickSchedule,
    tiebreak_rng: SmallRng,
    parent_rng: SmallRng,
    next_tick: u64,
    next_occupant: u64,
    retired: Option<RetiredSlot>,
}

/// The archive admission of a retiring occupant, held until its replacement completes.
struct RetiredSlot {
    slot: usize,
    archived: Option<ArchiveEntry>,
    evicted: Option<u64>,
}

impl Supervisor {
    /// Builds the supervisor and its initial population from `run_seed`.
    pub fn new(
        governor: &GovernorConfig,
        brain: &BrainConfig,
        run_seed: u64,
        world_size: f32,
        agent_count: usize,
    ) -> Result<Self, SupervisorError> {
        let config = governor.supervisor.clone();
        config
            .validate(agent_count)
            .map_err(SupervisorError::Config)?;
        if !(world_size.is_finite() && world_size > 0.0) {
            return Err(SupervisorError::InvalidWorldSize);
        }
        let window_len = usize::try_from(config.score_window_ticks / SUPERVISOR_SAMPLE_TICKS + 1)
            .unwrap_or(usize::MAX);
        let mut initial = SmallRng::seed_from_u64(stream_seed(run_seed, INITIAL_STREAM));
        let template = fresh_brain_state(brain, &mut initial);
        let brains = steering_population(
            &template,
            agent_count,
            1,
            governor.mutation_strength,
            &mut initial,
        );
        let slots: Vec<SlotLedger> = (0u64..)
            .zip(brains)
            .map(|(id, state)| SlotLedger::born(id, 0, state, brain.clone(), window_len))
            .collect();
        Ok(Self {
            boundary: TickSchedule::new(
                stream_seed(run_seed, BOUNDARY_STREAM),
                config.boundary_interval_ticks,
            ),
            archive: Archive::new(config.archive_capacity),
            next_occupant: slots.len() as u64,
            slots,
            config,
            mutation_strength: governor.mutation_strength,
            momentum_decay: governor.momentum_decay,
            world_size,
            tiebreak_rng: SmallRng::seed_from_u64(stream_seed(run_seed, TIEBREAK_STREAM)),
            parent_rng: SmallRng::seed_from_u64(stream_seed(run_seed, PARENT_STREAM)),
            next_tick: SUPERVISOR_SAMPLE_TICKS,
            retired: None,
        })
    }

    /// World size counters' cells are computed for.
    pub fn world_size(&self) -> f32 {
        self.world_size
    }

    /// The ledger of `index`.
    pub fn slot(&self, index: usize) -> Result<&SlotLedger, SupervisorError> {
        self.slots
            .get(index)
            .ok_or(SupervisorError::UnknownSlot { slot: index })
    }

    /// The tick the next sample must carry.
    pub fn next_sample_tick(&self) -> u64 {
        self.next_tick
    }

    /// The archive of retired lives.
    pub fn archive(&self) -> &Archive {
        &self.archive
    }

    /// Records the sample at `tick` for every slot; at a boundary, chooses at
    /// most one slot to replace and its parent.
    pub fn record_sample(
        &mut self,
        tick: u64,
        counters: &[SlotCounters],
    ) -> Result<SampleDecision, SupervisorError> {
        if tick != self.next_tick {
            return Err(SupervisorError::TickMismatch {
                expected: self.next_tick,
                actual: tick,
            });
        }
        if counters.len() != self.slots.len() {
            return Err(SupervisorError::CounterLength {
                expected: self.slots.len(),
                actual: counters.len(),
            });
        }
        for (slot, sample) in self.slots.iter_mut().zip(counters) {
            slot.record(*sample);
        }
        self.next_tick += SUPERVISOR_SAMPLE_TICKS;
        if tick != self.boundary.next_tick() {
            return Ok(SampleDecision::Recorded);
        }
        self.boundary.advance();
        let (deaths, under) = eligible_slots(&self.slots, tick, &self.config);
        let Some((slot, reason)) = choose_candidate(&deaths, &under, &mut self.tiebreak_rng) else {
            return Ok(SampleDecision::Idle);
        };
        // Retire first: the retiree joins the archive before the parent draw.
        let entry = self.entry_for(slot, tick);
        let (archived, evicted) = self.archive_entry(entry);
        self.retired = Some(RetiredSlot {
            slot,
            archived,
            evicted,
        });
        let parent = self.plan_parent(slot);
        Ok(SampleDecision::Replace(ReplacementPlan {
            slot,
            reason,
            parent,
            tick,
        }))
    }

    fn plan_parent(&mut self, slot: usize) -> ParentRef {
        match self.config.inheritance {
            Inheritance::Darwinian if !self.archive.is_empty() => {
                let pick = self.parent_rng.random_range(0..self.archive.len());
                ParentRef::ArchiveEntry(self.archive.entries()[pick].entry_id)
            }
            Inheritance::Darwinian => ParentRef::PeerBirth(self.best_peer(slot)),
            Inheritance::WeightCopy => ParentRef::PeerLearned(self.best_peer(slot)),
        }
    }

    /// Best living peer: highest window score among full windows, else highest
    /// life outcome, ties to the lowest slot index.
    fn best_peer(&self, slot: usize) -> usize {
        self.best_by(slot, SlotLedger::full_window_score)
            .or_else(|| self.best_by(slot, |s| Some(s.life_outcome())))
            .unwrap_or(usize::from(slot == 0))
    }

    fn best_by(&self, slot: usize, score: impl Fn(&SlotLedger) -> Option<f32>) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (index, ledger) in self.slots.iter().enumerate() {
            let Some(value) = score(ledger).filter(|_| index != slot) else {
                continue;
            };
            let better =
                best.is_none_or(|(_, top)| value > top || (top.is_nan() && !value.is_nan()));
            if better {
                best = Some((index, value));
            }
        }
        best.map(|(index, _)| index)
    }

    fn resolve_parent(
        &self,
        parent: ParentRef,
        learned: Option<&AgentBrainState>,
    ) -> Result<(AgentBrainState, BrainConfig), SupervisorError> {
        match (parent, learned) {
            (ParentRef::PeerLearned(_), None) => Err(SupervisorError::LearnedStateRequired),
            (ParentRef::ArchiveEntry(_) | ParentRef::PeerBirth(_), Some(_)) => {
                Err(SupervisorError::LearnedStateUnexpected)
            }
            (ParentRef::ArchiveEntry(entry_id), None) => self
                .archive
                .get(entry_id)
                .map(|e| (e.birth_brain.clone(), e.config.clone()))
                .ok_or(SupervisorError::UnknownArchiveEntry { entry_id }),
            (ParentRef::PeerBirth(peer), None) => {
                let ledger = self.slot(peer)?;
                Ok((ledger.birth_brain.clone(), ledger.config.clone()))
            }
            (ParentRef::PeerLearned(peer), Some(state)) => {
                Ok((state.clone(), self.slot(peer)?.config.clone()))
            }
        }
    }

    /// Finishes `plan`: mutates the newborn from its parent, archives the
    /// retired occupant and installs the newborn in the slot. `learned_parent`
    /// is `Some` exactly for a peer-learned parent.
    pub fn complete_replacement(
        &mut self,
        plan: ReplacementPlan,
        tick: u64,
        learned_parent: Option<&AgentBrainState>,
    ) -> Result<ReplacementOutcome, SupervisorError> {
        if tick != plan.tick {
            return Err(SupervisorError::TickMismatch {
                expected: plan.tick,
                actual: tick,
            });
        }
        self.slot(plan.slot)?;
        let (parent_brain, parent_config) = self.resolve_parent(plan.parent, learned_parent)?;
        let (archived, evicted) = match self.retired.take() {
            Some(retired) if retired.slot == plan.slot => (retired.archived, retired.evicted),
            _ => return Err(SupervisorError::UnknownSlot { slot: plan.slot }),
        };
        let newborn_brain =
            mutate_steering_weights(&parent_brain, self.mutation_strength, &mut self.parent_rng);
        let newborn_config = mutate_sensory_genes(
            &parent_config,
            self.mutation_strength,
            &MutationMomentum::new(self.momentum_decay),
            &mut self.parent_rng,
        );
        let occupant_id = self.next_occupant;
        self.next_occupant += 1;
        let window_len = self.slots[plan.slot].window_len;
        self.slots[plan.slot] = SlotLedger::born(
            occupant_id,
            tick,
            newborn_brain.clone(),
            newborn_config.clone(),
            window_len,
        );
        Ok(ReplacementOutcome {
            newborn_brain,
            newborn_config,
            occupant_id,
            archived,
            evicted,
        })
    }

    /// The archive entry of the occupant of `slot` retiring at `tick`.
    fn entry_for(&self, slot: usize, tick: u64) -> ArchiveEntry {
        let ledger = &self.slots[slot];
        ArchiveEntry {
            entry_id: self.archive.next_entry_id(),
            occupant_id: ledger.occupant_id,
            born_tick: ledger.born_tick,
            retired_tick: tick,
            outcome: ledger.life_outcome(),
            config: ledger.config.clone(),
            birth_brain: ledger.birth_brain.clone(),
        }
    }

    fn archive_entry(&mut self, entry: ArchiveEntry) -> (Option<ArchiveEntry>, Option<u64>) {
        match self.archive.insert(entry.clone()) {
            Admission::Refused => (None, None),
            Admission::Admitted { evicted } => (Some(entry), evicted),
        }
    }

    /// Archives the living slots with `age >= min_age_ticks`, in slot order.
    pub fn finish(&mut self, tick: u64) -> Vec<ArchiveChange> {
        let mut changes = Vec::new();
        for slot in 0..self.slots.len() {
            if self.slots[slot].age(tick) < self.config.min_age_ticks {
                continue;
            }
            let entry = self.entry_for(slot, tick);
            if let (Some(entry), evicted) = self.archive_entry(entry) {
                changes.push(ArchiveChange {
                    slot,
                    entry,
                    evicted,
                });
            }
        }
        changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use glam::Vec3;

    /// Run seed shared by the fixtures; any fixed value keeps them reproducible.
    const SEED: u64 = 7;
    /// World side length of the fixtures, matching the default world size.
    const WORLD: f32 = 100.0;
    /// Boundary and season period, and the mean of the test schedules.
    const BOUNDARY_INTERVAL: u64 = 400;
    /// Minimum life age: three sample periods, older than one window.
    const MIN_AGE: u64 = 300;
    /// Score window: two sample periods, so a ring holds three samples.
    const SCORE_WINDOW: u64 = 200;
    /// Archive capacity used by the fixtures; also the slot count of most tests.
    const CAPACITY: usize = 3;
    /// Slot count of the tie-break determinism test.
    const FOUR_SLOTS: usize = 4;
    /// Index of the best peer in three-slot fixtures.
    const SLOT_BEST: usize = 2;
    /// Index of the last slot in the four-slot fixture.
    const SLOT_FOURTH: usize = 3;
    /// Phase modulus that makes the per-sample cell drift cyclic.
    const PHASE_MODULUS: u32 = 5;
    /// Distance added per sample period.
    const DISTANCE_RATE: u32 = 5;
    /// Energy spent per sample period.
    const ENERGY_RATE: u32 = 10;
    /// Cell id gap between slots, so slots never share a cell.
    const CELL_STRIDE: u16 = 100;
    /// Retirement tick stamped on fixture archive entries.
    const RETIRED_TICK: u64 = 1000;
    /// Sample-driving limit that covers several boundaries.
    const HORIZON: u64 = 5000;
    /// Later limit used to find a still-pending death.
    const LATE_HORIZON: u64 = 9000;
    /// Food rate of the fast slots.
    const FAST: u32 = 2;
    /// First sample tick of a ring.
    const TICK_1: u64 = 100;
    /// Second sample tick of a ring.
    const TICK_2: u64 = 200;
    /// Third sample tick of a ring.
    const TICK_3: u64 = 300;
    /// Smallest gap for a mean of `BOUNDARY_INTERVAL`: half the mean.
    const MIN_GAP: u64 = 200;
    /// Largest gap for that mean: one and a half means.
    const MAX_GAP: u64 = 600;
    /// Long schedule length.
    const LEN_LONG: usize = 50;
    /// Medium schedule length.
    const LEN_MED: usize = 30;
    /// Short schedule length.
    const LEN_SHORT: usize = 20;
    /// First run seed for schedule tests.
    const SEED_A: u64 = 9;
    /// Second, distinct run seed for schedule tests.
    const SEED_B: u64 = 10;
    /// Run seed for the stream-independence test.
    const STREAM_RUN_SEED: u64 = 5;
    /// Second archive entry id (the first is 1).
    const ENTRY_B: u64 = 2;
    /// Third archive entry id.
    const ENTRY_C: u64 = 3;
    /// Fourth archive entry id.
    const ENTRY_D: u64 = 4;
    /// Fifth archive entry id.
    const ENTRY_E: u64 = 5;
    /// Sixth archive entry id.
    const ENTRY_F: u64 = 6;
    /// Mid archive outcome.
    const SCORE_MID: f32 = 0.5;
    /// High archive outcome.
    const SCORE_HIGH: f32 = 0.9;
    /// Low archive outcome.
    const SCORE_LOW: f32 = 0.1;
    /// Outcome between low and mid.
    const SCORE_NEW: f32 = 0.3;
    /// Outcome below every archived one, so it is refused when full.
    const SCORE_TINY: f32 = 0.05;
    /// Outcome of the parent entry seeded in the archive.
    const SCORE_PARENT: f32 = 0.7;
    /// Brain seed of that parent entry.
    const PARENT_BRAIN_SEED: u64 = 42;
    /// Brain seed of the learned state.
    const LEARNED_SEED: u64 = 77;
    /// Seed for the death-first tie-break rng.
    const TIE_SEED: u64 = 3;
    /// Seed for the no-candidate rng pair.
    const TIE_SEED_B: u64 = 4;
    /// Seed for the same-choice rng pair.
    const TIE_SEED_C: u64 = 11;
    /// Occupant id the first newborn receives (three initial occupants).
    const FIRST_NEWBORN_ID: u64 = 3;
    /// Huge outcome magnitude used to show archived scores are ignored.
    const HUGE_SCORE: f32 = 1.0e9;
    /// Food count above `COUNT_EXACT_LIMIT`.
    const OVER_LIMIT_FOOD: f32 = 3.0e7;
    /// Expected food delta of the `WINDOW_SAMPLES` ring.
    const WIN_FOOD: u32 = 3;
    /// Expected distinct cells of the newer samples in that ring.
    const WIN_CELLS: u32 = 2;
    /// Expected distance delta of that ring.
    const WIN_DISTANCE: f32 = 40.0;
    /// Expected energy delta of that ring.
    const WIN_ENERGY: f32 = 15.0;
    /// One window-score sample: (deaths, food, distance, energy, cell).
    type WindowSample = (f32, f32, f32, f32, u16);
    /// Samples of the window-score test; the slice length is inferred, not written.
    const WINDOW_SAMPLES: &[WindowSample] = &[
        (0.0, 1.0, W_DIST_A, W_ENERGY_A, 1),
        (0.0, W_FOOD_B, W_DIST_B, W_ENERGY_B, W_CELL_B),
        (1.0, W_FOOD_C, W_DIST_C, W_ENERGY_C, W_CELL_C),
    ];
    /// Distance of the first window sample.
    const W_DIST_A: f32 = 10.0;
    /// Energy of the first window sample.
    const W_ENERGY_A: f32 = 5.0;
    /// Food of the second window sample.
    const W_FOOD_B: f32 = 3.0;
    /// Distance of the second window sample.
    const W_DIST_B: f32 = 30.0;
    /// Energy of the second window sample.
    const W_ENERGY_B: f32 = 12.0;
    /// Cell of the second window sample.
    const W_CELL_B: u16 = 2;
    /// Food of the third window sample.
    const W_FOOD_C: f32 = 4.0;
    /// Distance of the third window sample.
    const W_DIST_C: f32 = 50.0;
    /// Energy of the third window sample.
    const W_ENERGY_C: f32 = 20.0;
    /// Cell of the third window sample.
    const W_CELL_C: u16 = 3;
    /// Probe coordinate at the world's half-extent edge.
    const SPOT_EDGE: f32 = 50.0;
    /// Probe coordinate just inside the edge.
    const SPOT_INSIDE: f32 = 49.99;
    /// Probe coordinate beyond the world, to exercise clamping.
    const SPOT_FAR: f32 = 75.0;
    /// Probe coordinate beyond the world on the negative side.
    const SPOT_MID: f32 = 60.0;
    /// Small positive probe coordinate.
    const SPOT_SMALL: f32 = 10.0;
    /// Arbitrary interior probe x.
    const SPOT_ODD_X: f32 = 12.3;
    /// Arbitrary interior probe z magnitude.
    const SPOT_ODD_Z: f32 = 34.5;
    /// Energy of the first sample in the non-finite cases.
    const ENERGY_FIRST: f32 = 5.0;
    /// Energy of the second sample in the non-finite energy case.
    const ENERGY_SECOND: f32 = 6.0;
    /// Constant distance of the non-finite cases.
    const FLAT_DISTANCE: f32 = 10.0;
    /// Food of the second sample in the over-limit case.
    const FOOD_SECOND: f32 = 2.0;
    /// Positions probed against `Agent::record_heatmap`.
    const SPOTS: &[(f32, f32)] = &[
        (0.0, 0.0),
        (-SPOT_EDGE, -SPOT_EDGE),
        (SPOT_INSIDE, SPOT_INSIDE),
        (SPOT_EDGE, SPOT_EDGE),
        (SPOT_FAR, -SPOT_FAR),
        (-SPOT_MID, SPOT_SMALL),
        (SPOT_ODD_X, -SPOT_ODD_Z),
    ];

    fn governor(inheritance: Inheritance) -> GovernorConfig {
        GovernorConfig {
            supervisor: SupervisorConfig {
                boundary_interval_ticks: BOUNDARY_INTERVAL,
                min_age_ticks: MIN_AGE,
                score_window_ticks: SCORE_WINDOW,
                archive_capacity: CAPACITY,
                season_period_ticks: BOUNDARY_INTERVAL,
                inheritance,
                agent_tick_budget: 0,
            },
            ..GovernorConfig::default()
        }
    }

    fn supervisor(inheritance: Inheritance, agents: usize) -> Supervisor {
        Supervisor::new(
            &governor(inheritance),
            &BrainConfig::default(),
            SEED,
            WORLD,
            agents,
        )
        .expect("valid supervisor")
    }

    /// Counters of every slot at `tick`: slot `i` eats `rates[i]` per sample,
    /// and slots in `dead` carry one death.
    fn counters_at(tick: u64, rates: &[u32], dead: &[usize]) -> Vec<SlotCounters> {
        let n = u32::try_from(tick / SUPERVISOR_SAMPLE_TICKS).expect("test tick fits u32");
        let phase = u16::try_from(n % PHASE_MODULUS).expect("phase fits u16");
        rates
            .iter()
            .enumerate()
            .map(|(i, rate)| SlotCounters {
                alive: true,
                deaths: if dead.contains(&i) { 1.0 } else { 0.0 },
                food: f32::from(u16::try_from(rate * n).expect("test food fits u16")),
                distance: f32::from(
                    u16::try_from(DISTANCE_RATE * n).expect("test distance fits u16"),
                ),
                energy_spent: f32::from(
                    u16::try_from(ENERGY_RATE * n).expect("test energy fits u16"),
                ),
                cell: Some(u16::try_from(i).expect("slot fits u16") * CELL_STRIDE + phase),
            })
            .collect()
    }

    /// Samples until a boundary picks a slot; `None` when none does by `limit`.
    fn drive(
        sup: &mut Supervisor,
        rates: &[u32],
        dead: &[usize],
        limit: u64,
    ) -> Option<ReplacementPlan> {
        drive_with(sup, rates, dead, limit, false)
    }

    /// Like `drive`; with `poison`, the first dead slot reports NaN food, so
    /// its life outcome is non-finite and is refused by the archive.
    fn drive_with(
        sup: &mut Supervisor,
        rates: &[u32],
        dead: &[usize],
        limit: u64,
        poison: bool,
    ) -> Option<ReplacementPlan> {
        while sup.next_sample_tick() <= limit {
            let tick = sup.next_sample_tick();
            let mut counters = counters_at(tick, rates, dead);
            if let (true, Some(&first)) = (poison, dead.first()) {
                counters[first].food = f32::NAN;
            }
            let decision = sup.record_sample(tick, &counters).expect("sample accepted");
            if let SampleDecision::Replace(plan) = decision {
                return Some(plan);
            }
        }
        None
    }

    fn entry(entry_id: u64, outcome: f32, brain_seed: u64) -> ArchiveEntry {
        let mut rng = SmallRng::seed_from_u64(brain_seed);
        ArchiveEntry {
            entry_id,
            occupant_id: entry_id,
            born_tick: 0,
            retired_tick: RETIRED_TICK,
            outcome,
            config: BrainConfig::default(),
            birth_brain: fresh_brain_state(&BrainConfig::default(), &mut rng),
        }
    }

    fn schedule_ticks(seed: u64, count: usize) -> Vec<u64> {
        let mut schedule = TickSchedule::new(seed, BOUNDARY_INTERVAL);
        (0..count)
            .map(|_| {
                let tick = schedule.next_tick();
                schedule.advance();
                tick
            })
            .collect()
    }

    #[test]
    fn schedule_ticks_on_sample_grid_within_bounds() {
        let ticks = schedule_ticks(1, LEN_LONG);
        let mut previous = 0;
        for tick in ticks {
            let gap = tick - previous;
            assert_eq!(gap % SUPERVISOR_SAMPLE_TICKS, 0);
            assert!((MIN_GAP..=MAX_GAP).contains(&gap), "gap {gap}");
            previous = tick;
        }
    }

    #[test]
    fn schedule_same_seed_same_ticks() {
        assert_eq!(
            schedule_ticks(SEED_A, LEN_MED),
            schedule_ticks(SEED_A, LEN_MED)
        );
        assert_ne!(
            schedule_ticks(SEED_A, LEN_MED),
            schedule_ticks(SEED_B, LEN_MED)
        );
    }

    #[test]
    fn schedule_streams_are_independent() {
        let boundary_seed = stream_seed(STREAM_RUN_SEED, BOUNDARY_STREAM);
        let season_seed = stream_seed(STREAM_RUN_SEED, SEASON_STREAM);
        assert_ne!(boundary_seed, season_seed);
        let alone = schedule_ticks(boundary_seed, LEN_SHORT);
        let mut boundary = TickSchedule::new(boundary_seed, BOUNDARY_INTERVAL);
        let mut season = TickSchedule::new(season_seed, BOUNDARY_INTERVAL);
        let mut interleaved = Vec::new();
        for _ in 0..LEN_SHORT {
            interleaved.push(boundary.next_tick());
            boundary.advance();
            season.advance();
        }
        assert_eq!(alone, interleaved);
        assert_ne!(alone, schedule_ticks(season_seed, LEN_SHORT));
    }

    #[test]
    fn eligibility_pending_death_persists_until_replaced() {
        let mut sup = supervisor(Inheritance::Darwinian, CAPACITY);
        let rates = [1, 1, 1];
        let plan = drive(&mut sup, &rates, &[1], HORIZON).expect("death replaces");
        assert_eq!((plan.slot, plan.reason), (1, ReplaceReason::Death));
        let again = drive(&mut sup, &rates, &[1], LATE_HORIZON).expect("still pending");
        assert_eq!((again.slot, again.reason), (1, ReplaceReason::Death));
        sup.complete_replacement(again.clone(), again.tick, None)
            .expect("completes");
        assert!(!sup.slot(1).expect("slot").pending_death);
        assert_eq!(sup.slot(1).expect("slot").born_tick, again.tick);
    }

    #[test]
    fn eligibility_underperformer_strictly_below_peer_median() {
        let mut sup = supervisor(Inheritance::Darwinian, CAPACITY);
        let plan = drive(&mut sup, &[0, FAST, FAST], &[], HORIZON).expect("underperformer");
        assert_eq!((plan.slot, plan.reason), (0, ReplaceReason::Underperformer));
        let mut equal = supervisor(Inheritance::Darwinian, CAPACITY);
        assert!(drive(&mut equal, &[1, 1, 1], &[], HORIZON).is_none());
    }

    #[test]
    fn eligibility_requires_min_age_and_full_window() {
        let mut sup = supervisor(Inheritance::Darwinian, CAPACITY);
        let rates = [0, FAST, FAST];
        sup.record_sample(TICK_1, &counters_at(TICK_1, &rates, &[]))
            .expect("sample");
        let lenient = SupervisorConfig {
            min_age_ticks: TICK_1,
            ..sup.config.clone()
        };
        let (deaths, under) = eligible_slots(&sup.slots, TICK_1, &lenient);
        assert!(deaths.is_empty() && under.is_empty(), "window not full");
        sup.record_sample(TICK_2, &counters_at(TICK_2, &rates, &[]))
            .expect("sample");
        let (_, under) = eligible_slots(&sup.slots, TICK_2, &sup.config);
        assert!(under.is_empty(), "too young");
        let (_, under) = eligible_slots(&sup.slots, TICK_2, &lenient);
        assert_eq!(under, vec![0]);
    }

    #[test]
    fn eligibility_ignores_archived_scores() {
        let mut sup = supervisor(Inheritance::Darwinian, CAPACITY);
        for tick in [TICK_1, TICK_2, TICK_3] {
            sup.record_sample(tick, &counters_at(tick, &[0, FAST, FAST], &[]))
                .expect("sample");
        }
        let before = eligible_slots(&sup.slots, TICK_3, &sup.config);
        assert_eq!(
            sup.archive.insert(entry(1, HUGE_SCORE, 1)),
            Admission::Admitted { evicted: None }
        );
        assert_eq!(
            sup.archive.insert(entry(ENTRY_B, -HUGE_SCORE, ENTRY_B)),
            Admission::Admitted { evicted: None }
        );
        assert_eq!(before, eligible_slots(&sup.slots, TICK_3, &sup.config));
        assert_eq!(before.1, vec![0]);
    }

    #[test]
    fn tiebreak_death_first_and_one_slot() {
        let mut rng = SmallRng::seed_from_u64(TIE_SEED);
        let (slot, reason) = choose_candidate(&[0, SLOT_BEST], &[1], &mut rng).expect("candidate");
        assert_eq!(reason, ReplaceReason::Death);
        assert!(slot == 0 || slot == SLOT_BEST);
        let (slot, reason) = choose_candidate(&[], &[1], &mut rng).expect("candidate");
        assert_eq!((slot, reason), (1, ReplaceReason::Underperformer));
        let mut fresh = SmallRng::seed_from_u64(TIE_SEED_B);
        let mut untouched = SmallRng::seed_from_u64(TIE_SEED_B);
        assert!(choose_candidate(&[], &[], &mut fresh).is_none());
        assert_eq!(fresh.random::<u64>(), untouched.random::<u64>());
    }

    #[test]
    fn tiebreak_same_seed_same_choice() {
        let run = || {
            let mut sup = supervisor(Inheritance::Darwinian, FOUR_SLOTS);
            drive(
                &mut sup,
                &[1, 1, 1, 1],
                &[0, 1, SLOT_BEST, SLOT_FOURTH],
                HORIZON,
            )
            .expect("plan")
        };
        assert_eq!(run(), run());
        let all = [0, 1, SLOT_BEST, SLOT_FOURTH];
        let mut a = SmallRng::seed_from_u64(TIE_SEED_C);
        let mut b = SmallRng::seed_from_u64(TIE_SEED_C);
        assert_eq!(
            choose_candidate(&all, &[], &mut a),
            choose_candidate(&all, &[], &mut b)
        );
    }

    #[test]
    fn archive_bounded_rank_order_and_eviction() {
        let mut archive = Archive::new(CAPACITY);
        assert_eq!(
            archive.insert(entry(1, SCORE_MID, 1)),
            Admission::Admitted { evicted: None }
        );
        assert_eq!(
            archive.insert(entry(ENTRY_B, SCORE_HIGH, ENTRY_B)),
            Admission::Admitted { evicted: None }
        );
        assert_eq!(
            archive.insert(entry(ENTRY_C, SCORE_LOW, ENTRY_C)),
            Admission::Admitted { evicted: None }
        );
        let ids = |a: &Archive| a.entries().iter().map(|e| e.entry_id).collect::<Vec<_>>();
        assert_eq!(ids(&archive), vec![ENTRY_B, 1, ENTRY_C]);
        assert_eq!(
            archive.insert(entry(ENTRY_D, SCORE_NEW, ENTRY_D)),
            Admission::Admitted {
                evicted: Some(ENTRY_C)
            }
        );
        assert_eq!(ids(&archive), vec![ENTRY_B, 1, ENTRY_D]);
        assert_eq!(
            archive.insert(entry(ENTRY_E, SCORE_TINY, ENTRY_E)),
            Admission::Refused
        );
        assert_eq!(
            archive.insert(entry(ENTRY_F, SCORE_MID, ENTRY_F)),
            Admission::Admitted {
                evicted: Some(ENTRY_D)
            }
        );
        assert_eq!(ids(&archive), vec![ENTRY_B, 1, ENTRY_F]);
        assert_eq!(archive.len(), CAPACITY);
        assert!(archive.get(ENTRY_F).is_some() && archive.get(ENTRY_D).is_none());
    }

    #[test]
    fn archive_rejects_non_finite_outcome() {
        let mut archive = Archive::new(CAPACITY);
        for outcome in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(archive.insert(entry(1, outcome, 1)), Admission::Refused);
        }
        assert!(archive.is_empty());
    }

    fn reference_rng() -> SmallRng {
        SmallRng::seed_from_u64(stream_seed(SEED, PARENT_STREAM))
    }

    #[test]
    fn parent_draw_from_archive_uses_birth_brain() {
        let mut sup = supervisor(Inheritance::Darwinian, CAPACITY);
        let parent = entry(1, SCORE_PARENT, PARENT_BRAIN_SEED);
        sup.archive.insert(parent.clone());
        let plan = drive(&mut sup, &[1, 1, 1], &[0], HORIZON).expect("plan");
        // Retire then draw: the retiree is already archived when the parent is drawn.
        let pool = usize::from(u8::try_from(ENTRY_B).expect("entry count fits u8"));
        assert_eq!(sup.archive().len(), pool);
        assert_eq!(sup.archive().get(ENTRY_B).map(|e| e.occupant_id), Some(0));
        let mut rng = reference_rng();
        let pick = rng.random_range(0..pool);
        let chosen = sup.archive().entries()[pick].clone();
        assert_eq!(plan.parent, ParentRef::ArchiveEntry(chosen.entry_id));
        let outcome = sup
            .complete_replacement(plan.clone(), plan.tick, None)
            .expect("completes");
        assert_eq!(outcome.archived.map(|e| e.entry_id), Some(ENTRY_B));
        let expected =
            mutate_steering_weights(&chosen.birth_brain, sup.mutation_strength, &mut rng);
        assert_eq!(outcome.newborn_brain.brain_state, expected.brain_state);
    }

    #[test]
    fn parent_fallback_is_best_peer_birth_brain() {
        let mut sup = supervisor(Inheritance::Darwinian, CAPACITY);
        let plan = drive_with(&mut sup, &[0, 1, FAST], &[0], HORIZON, true).expect("plan");
        assert_eq!(plan.slot, 0);
        assert!(sup.archive().is_empty());
        assert_eq!(plan.parent, ParentRef::PeerBirth(SLOT_BEST));
    }

    #[test]
    fn newborn_is_mutation_of_parent_birth_brain() {
        let mut sup = supervisor(Inheritance::Darwinian, CAPACITY);
        let plan = drive_with(&mut sup, &[0, 1, FAST], &[0], HORIZON, true).expect("plan");
        let (peer_brain, peer_config) = {
            let peer = sup.slot(SLOT_BEST).expect("slot");
            (peer.birth_brain.clone(), peer.config.clone())
        };
        let outcome = sup
            .complete_replacement(plan.clone(), plan.tick, None)
            .expect("completes");
        let mut rng = reference_rng();
        let brain = mutate_steering_weights(&peer_brain, sup.mutation_strength, &mut rng);
        let config = mutate_sensory_genes(
            &peer_config,
            sup.mutation_strength,
            &MutationMomentum::new(sup.momentum_decay),
            &mut rng,
        );
        assert_eq!(outcome.newborn_brain.brain_state, brain.brain_state);
        assert_eq!(outcome.newborn_brain.patterns, brain.patterns);
        assert_eq!(outcome.newborn_config, config);
        assert_eq!(outcome.occupant_id, FIRST_NEWBORN_ID);
        let ledger = sup.slot(0).expect("slot");
        assert_eq!(
            (ledger.occupant_id, ledger.born_tick),
            (FIRST_NEWBORN_ID, plan.tick)
        );
    }

    #[test]
    fn weight_copy_requires_and_uses_learned_state() {
        let mut sup = supervisor(Inheritance::WeightCopy, CAPACITY);
        let plan = drive(&mut sup, &[0, 1, FAST], &[0], HORIZON).expect("plan");
        assert_eq!(plan.parent, ParentRef::PeerLearned(SLOT_BEST));
        assert_eq!(
            sup.complete_replacement(plan.clone(), plan.tick, None)
                .unwrap_err(),
            SupervisorError::LearnedStateRequired
        );
        let mut rng = SmallRng::seed_from_u64(LEARNED_SEED);
        let learned = fresh_brain_state(&BrainConfig::default(), &mut rng);
        let outcome = sup
            .complete_replacement(plan.clone(), plan.tick, Some(&learned))
            .expect("completes");
        let expected =
            mutate_steering_weights(&learned, sup.mutation_strength, &mut reference_rng());
        assert_eq!(outcome.newborn_brain.brain_state, expected.brain_state);

        let mut darwin = supervisor(Inheritance::Darwinian, CAPACITY);
        let plan = drive(&mut darwin, &[0, 1, FAST], &[0], HORIZON).expect("plan");
        assert_eq!(
            darwin
                .complete_replacement(plan.clone(), plan.tick, Some(&learned))
                .unwrap_err(),
            SupervisorError::LearnedStateUnexpected
        );
    }

    fn sample(deaths: f32, food: f32, distance: f32, energy: f32, cell: u16) -> SlotCounters {
        SlotCounters {
            alive: true,
            deaths,
            food,
            distance,
            energy_spent: energy,
            cell: Some(cell),
        }
    }

    fn ledger() -> SlotLedger {
        let mut rng = SmallRng::seed_from_u64(1);
        let brain = fresh_brain_state(&BrainConfig::default(), &mut rng);
        SlotLedger::born(0, 0, brain, BrainConfig::default(), CAPACITY)
    }

    #[test]
    fn window_score_matches_composite_fitness_over_window() {
        let mut slot = ledger();
        for (deaths, food, distance, energy, cell) in WINDOW_SAMPLES.iter().copied() {
            slot.record(sample(deaths, food, distance, energy, cell));
        }
        let expected = composite_fitness(
            1,
            WIN_FOOD,
            WIN_CELLS,
            exploration_grid_cells(),
            WIN_DISTANCE,
            WIN_ENERGY,
        );
        assert_eq!(slot.full_window_score(), Some(expected));

        let mut bad_energy = ledger();
        bad_energy.record(sample(0.0, 1.0, FLAT_DISTANCE, ENERGY_FIRST, 1));
        bad_energy.record(sample(0.0, 1.0, FLAT_DISTANCE, ENERGY_SECOND, 1));
        bad_energy.record(sample(0.0, 1.0, FLAT_DISTANCE, f32::NAN, 1));
        assert!(bad_energy.window_score().is_nan());

        let mut huge = ledger();
        huge.record(sample(0.0, 1.0, FLAT_DISTANCE, ENERGY_FIRST, 1));
        huge.record(sample(0.0, FOOD_SECOND, FLAT_DISTANCE, ENERGY_FIRST, 1));
        huge.record(sample(0.0, OVER_LIMIT_FOOD, FLAT_DISTANCE, ENERGY_FIRST, 1));
        assert!(huge.window_score().is_nan());
        assert!(huge.life_outcome().is_nan());
    }

    #[test]
    fn heatmap_cell_matches_agent_record_heatmap() {
        for (x, z) in SPOTS.iter().copied() {
            let mut agent = Agent::new(0, Vec3::ZERO, 0, BrainConfig::default(), 0);
            agent.body.body.position = Vec3::new(x, 0.0, z);
            agent.record_heatmap(WORLD);
            let expected = agent.heatmap.iter().position(|&n| n > 0).expect("one cell");
            assert_eq!(heatmap_cell(x, z, WORLD), u16::try_from(expected).ok());
        }
        let mut row = vec![0.0; PHYS_STRIDE];
        row[P_ALIVE] = 1.0;
        row[P_POS_X] = f32::NAN;
        let counters = SlotCounters::from_physics(&row, WORLD).expect("row");
        assert!(counters.alive && counters.cell.is_none());
        assert_eq!(
            SlotCounters::from_physics(&row[1..], WORLD),
            Err(SupervisorError::PhysicsRowLength {
                expected: PHYS_STRIDE,
                actual: PHYS_STRIDE - 1
            })
        );
    }

    #[test]
    fn core_source_has_no_gpu_types() {
        let source = include_str!("supervisor.rs");
        let core = source.split("#[cfg(test)]").next().expect("source");
        let forbidden = [
            concat!("wg", "pu"),
            concat!("Gpu", "Kernel"),
            concat!("rus", "qlite"),
            concat!("std::", "fs"),
            concat!("World", "State"),
        ];
        for name in forbidden {
            assert!(!core.contains(name), "core mentions {name}");
        }
    }
}
