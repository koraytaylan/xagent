//! Shared kernel-facing driver of the continuous-life supervisor.
//!
//! The driver borrows the caller's [`GpuKernel`] and owns the pure supervisor
//! core, its own [`WorldState`] for newborn spawn positions, an optional store
//! and an optional run-summary accumulator. The caller dispatches batches and
//! hands each 100-tick readback to [`SupervisorRun::on_sample`].

use std::fmt;

use rusqlite::{params, Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use xagent_brain::buffers::{AgentBrainState, PHYS_STRIDE};
use xagent_brain::gpu_kernel::{
    FoodKindSetError, KernelCheckpoint, KernelCpuState, KernelRestoreError, KernelSection,
    SlotReplaceError, SlotSpawn,
};
use xagent_brain::GpuKernel;
use xagent_shared::{FoodKind, FullConfig, SupervisorConfigError, SUPERVISOR_SAMPLE_TICKS};

use crate::agent::Agent;
use crate::run_summary::{KindTotalsBits, RunSummary, RunSummaryAccumulator, RunSummaryError};
use crate::supervisor::{
    stream_seed, ArchiveChange, EligibilityGround, ParentRef, ReplaceReason, ReplacementOutcome,
    ReplacementPlan, SampleDecision, SlotCounters, SlotStanding, Supervisor, SupervisorError,
    SupervisorState, TickSchedule, TickScheduleState, SEASON_STREAM,
};
use crate::supervisor_checkpoint::{decode, encode, Checkpoint, CheckpointError};
use crate::supervisor_store::{read_checkpoint, StoreError, SupervisorStore};
use crate::supervisor_telemetry::{NewbornInput, SampleInput, TelemetryBuilder, TelemetryFrame};
use crate::world::WorldState;

/// Tag xor-ed into the world seed to seed the kernel's initial agent reset.
pub const RESET_SEED_TAG: u64 = 0x5EED_0000_0000_0006;

/// What one sample did.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleReport {
    /// Tick of the sample.
    pub tick: u64,
    /// The boundary decided at this sample, if it was one.
    pub boundary: Option<BoundaryReport>,
    /// The nourishing kind a season flip set at this sample.
    pub season: Option<FoodKind>,
    /// Whether the run reached its tick budget.
    pub finished: bool,
}

/// A replacement boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryReport {
    /// Boundary tick.
    pub tick: u64,
    /// Replaced slot; `None` for an idle boundary.
    pub slot: Option<usize>,
    /// Why the slot was replaced.
    pub reason: Option<ReplaceReason>,
    /// Occupant id given to the newborn.
    pub occupant_id: Option<u64>,
    /// Parent of the newborn.
    pub parent: Option<ParentRef>,
    /// Archive entries held after the boundary.
    pub archive_len: usize,
    /// Outcome of the best archive entry.
    pub best_outcome: Option<f32>,
}

/// Result of a finished run.
#[derive(Clone, Debug)]
pub struct SupervisorFinish {
    /// The run summary when measuring.
    pub summary: Option<RunSummary>,
}

/// Why a supervisor run call failed.
#[derive(Debug)]
pub enum SupervisorRunError {
    /// The supervisor configuration is infeasible.
    Config(SupervisorConfigError),
    /// The supervisor core refused a call.
    Core(SupervisorError),
    /// The store failed.
    Store(StoreError),
    /// The run summary refused a sample.
    Summary(RunSummaryError),
    /// A slot write was refused.
    Slot(SlotReplaceError),
    /// The nourishing kind could not be set.
    FoodKind(FoodKindSetError),
    /// The kernel holds a different number of agents than the config.
    AgentCount { expected: usize, actual: u32 },
    /// The sample period is not a multiple of the kernel batch size.
    BatchSize { batch: u32 },
    /// A sample tick was not the expected one.
    TickMismatch { expected: u64, actual: u64 },
    /// A physics slice had the wrong length.
    PhysicsLength { expected: usize, actual: usize },
    /// Food kinds are on but the kernel has no nourishing kind.
    MissingNourishingKind,
    /// A slot index does not fit the kernel's index type.
    SlotRange { slot: usize },
    /// An occupant id does not fit the kernel's agent id type.
    OccupantRange { occupant_id: u64 },
    /// A checkpoint could not be written or read.
    Checkpoint(CheckpointError),
    /// The kernel refused a checkpoint.
    KernelRestore(KernelRestoreError),
}

impl fmt::Display for SupervisorRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(e) => write!(f, "invalid supervisor config: {e}"),
            Self::Core(e) => write!(f, "supervisor core: {e}"),
            Self::Store(e) => write!(f, "supervisor store: {e}"),
            Self::Summary(e) => write!(f, "run summary: {e}"),
            Self::Slot(e) => write!(f, "slot replacement: {e}"),
            Self::FoodKind(e) => write!(f, "food kind: {e}"),
            Self::AgentCount { expected, actual } => {
                write!(f, "kernel has {actual} agents, config needs {expected}")
            }
            Self::BatchSize { batch } => write!(
                f,
                "sample period {SUPERVISOR_SAMPLE_TICKS} is not a multiple of kernel batch size {batch}"
            ),
            Self::TickMismatch { expected, actual } => {
                write!(f, "sample tick {actual}, expected {expected}")
            }
            Self::PhysicsLength { expected, actual } => {
                write!(f, "physics has {actual} values, expected {expected}")
            }
            Self::MissingNourishingKind => write!(f, "kernel has no nourishing food kind"),
            Self::SlotRange { slot } => write!(f, "slot {slot} is out of range"),
            Self::OccupantRange { occupant_id } => {
                write!(f, "occupant id {occupant_id} is out of range")
            }
            Self::Checkpoint(e) => write!(f, "checkpoint: {e}"),
            Self::KernelRestore(e) => write!(f, "kernel restore: {e}"),
        }
    }
}

impl From<CheckpointError> for SupervisorRunError {
    fn from(e: CheckpointError) -> Self {
        Self::Checkpoint(e)
    }
}
impl From<KernelRestoreError> for SupervisorRunError {
    fn from(e: KernelRestoreError) -> Self {
        Self::KernelRestore(e)
    }
}

impl std::error::Error for SupervisorRunError {}

impl From<SupervisorConfigError> for SupervisorRunError {
    fn from(e: SupervisorConfigError) -> Self {
        Self::Config(e)
    }
}
impl From<SupervisorError> for SupervisorRunError {
    fn from(e: SupervisorError) -> Self {
        Self::Core(e)
    }
}
impl From<StoreError> for SupervisorRunError {
    fn from(e: StoreError) -> Self {
        Self::Store(e)
    }
}
impl From<RunSummaryError> for SupervisorRunError {
    fn from(e: RunSummaryError) -> Self {
        Self::Summary(e)
    }
}
impl From<SlotReplaceError> for SupervisorRunError {
    fn from(e: SlotReplaceError) -> Self {
        Self::Slot(e)
    }
}
impl From<FoodKindSetError> for SupervisorRunError {
    fn from(e: FoodKindSetError) -> Self {
        Self::FoodKind(e)
    }
}

/// The kind that is not `kind`.
fn other_kind(kind: FoodKind) -> FoodKind {
    match kind {
        FoodKind::First => FoodKind::Second,
        FoodKind::Second => FoodKind::First,
    }
}

/// Kernel index of a slot.
fn slot_index(slot: usize) -> Result<u32, SupervisorRunError> {
    u32::try_from(slot).map_err(|_| SupervisorRunError::SlotRange { slot })
}

/// Spawn parameters of a freshly built agent.
fn spawn_of(agent: &Agent) -> SlotSpawn {
    SlotSpawn {
        position: agent.body.body.position,
        max_energy: agent.body.body.internal.max_energy,
        max_integrity: agent.body.body.internal.max_integrity,
        memory_capacity: agent.brain_config.memory_capacity,
        processing_slots: agent.brain_config.processing_slots,
    }
}

/// Uploads the world's static data to the kernel.
fn upload_world(kernel: &GpuKernel, world: &WorldState) {
    let biomes = world.biome_map.grid_as_u32();
    let food_pos: Vec<(f32, f32, f32)> = world
        .food_items
        .iter()
        .map(|f| (f.position.x, f.position.y, f.position.z))
        .collect();
    let consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
    let timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
    kernel.upload_world(
        &world.terrain.heights,
        &biomes,
        &food_pos,
        &consumed,
        &timers,
    );
}

/// Rebuilds a telemetry builder through its public calls from the mirror.
///
/// The persisted frame is replayed as a season-flip sample, which always
/// persists, so it restores the previous standings and tick and empties the
/// queue; the recorded closures then restore the seen totals and the queue.
/// The frames this returns are dropped and nothing is written.
fn rebuild_telemetry(population: usize, mirror: &TelemetryMirror) -> TelemetryBuilder {
    let mut builder = TelemetryBuilder::new(population);
    if let Some(persisted) = &mirror.persisted {
        if let Some(base) = &mirror.seen_base {
            builder.on_season_closed(persisted.tick, &base.into());
        }
        let standings: Vec<SlotStanding> = persisted
            .standings
            .iter()
            .map(|s| SlotStanding {
                slot: s.slot,
                occupant_id: s.occupant_id,
                age_ticks: 0,
                alive: true,
                meals: s.meals,
                deaths: s.deaths,
                window_score: None,
                peer_median: None,
                ground: EligibilityGround::NoFullWindow,
                parent: None,
            })
            .collect();
        let _ = builder.on_sample(SampleInput {
            tick: persisted.tick,
            standings: &standings,
            boundary: None,
            replacement: None,
            archive_outcomes: &[],
            total_ticks: None,
            agent_tick_budget: 0,
            next_boundary_tick: 0,
            nourishing: None,
            next_flip_tick: None,
            ticks_per_second: None,
            season_flipped: true,
        });
    }
    for (tick, totals) in &mirror.closures {
        builder.on_season_closed(*tick, &totals.into());
    }
    builder
}

/// The supervisor driver over one borrowed kernel.
pub struct SupervisorRun {
    core: Supervisor,
    world: WorldState,
    store: Option<SupervisorStore>,
    accumulator: Option<RunSummaryAccumulator>,
    population: usize,
    kinds_on: bool,
    season: TickSchedule,
    nourishing: Option<FoodKind>,
    tick: u64,
    budget: u64,
    /// Occupant id the core assigns to the next newborn (mirrors the core's counter).
    next_occupant: u64,
    telemetry: TelemetryBuilder,
    recording: bool,
    telemetry_stats: Option<RunSummaryAccumulator>,
    last_ticks_per_second: Option<f64>,
    last_frame: Option<TelemetryFrame>,
    telemetry_warned: bool,
    config: FullConfig,
    spawn_draws: u64,
    telemetry_mirror: TelemetryMirror,
}

/// One slot of the last persisted telemetry frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TelemetryStanding {
    /// Slot index.
    pub slot: usize,
    /// Occupant of the slot.
    pub occupant_id: u64,
    /// Meals of the occupant.
    pub meals: u32,
    /// Deaths of the occupant.
    pub deaths: u32,
}

/// Standings and tick of the last persisted telemetry frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TelemetryPersisted {
    /// Tick of the frame.
    pub tick: u64,
    /// Standings of every slot at that tick.
    pub standings: Vec<TelemetryStanding>,
}

/// The inputs that rebuild the telemetry builder through its public calls.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TelemetryMirror {
    /// The last persisted frame.
    pub persisted: Option<TelemetryPersisted>,
    /// Season totals as last handed to the builder.
    pub seen: Option<KindTotalsBits>,
    /// Season totals as they stood at the last persisted frame.
    pub seen_base: Option<KindTotalsBits>,
    /// Season closures handed to the builder since the last persisted frame.
    pub closures: Vec<(u64, KindTotalsBits)>,
}

/// Everything a [`SupervisorRun`] holds that a checkpoint must carry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SupervisorRunState {
    /// The supervisor core.
    pub core: SupervisorState,
    /// The season schedule.
    pub(crate) season: TickScheduleState,
    /// Run summary accumulator when measuring.
    pub accumulator: Option<RunSummaryAccumulator>,
    /// Number of agent slots.
    pub population: usize,
    /// Whether food kinds are on.
    pub kinds_on: bool,
    /// Current nourishing kind.
    pub nourishing: Option<FoodKind>,
    /// Last sample tick handled.
    pub tick: u64,
    /// Agent-tick budget.
    pub budget: u64,
    /// Occupant id of the next newborn.
    pub next_occupant: u64,
    /// `safe_spawn_position` calls so far.
    pub spawn_draws: u64,
    /// Whether telemetry persistence is on.
    pub recording: bool,
    /// Telemetry statistics accumulator.
    pub telemetry_stats: Option<RunSummaryAccumulator>,
    /// Bits of the latest observed ticks per second.
    pub last_ticks_per_second_bits: Option<u64>,
    /// Whether the telemetry failure was reported.
    pub telemetry_warned: bool,
    /// Inputs rebuilding the telemetry builder.
    pub telemetry_mirror: TelemetryMirror,
    /// The frame of the last sample, not yet taken.
    #[serde(with = "frame_serde::opt_frame")]
    pub last_frame: Option<TelemetryFrame>,
}

/// Kernel section names a checkpoint may carry.
const KERNEL_SECTION_NAMES: [&str; 11] = [
    "agent_phys",
    "decision",
    "food_state",
    "food_flags",
    "sensory",
    "sensory_next",
    "brain_state",
    "brain_scratch",
    "pattern",
    "brain_config",
    "trail_ring",
];

/// Section holding the serialised [`FullConfig`].
const SECTION_CONFIG: &str = "config";
/// Section holding the serialised [`SupervisorRunState`].
const SECTION_RUN: &str = "run";
/// Prefix of every kernel section.
const SECTION_KERNEL_PREFIX: &str = "kernel.";
/// Name of the kernel CPU state section.
const SECTION_KERNEL_CPU: &str = "kernel.cpu";

/// A checkpoint decoded and checked, ready for [`SupervisorRun::resume`].
#[derive(Debug)]
pub struct ResumePoint {
    /// Run the checkpoint belongs to.
    pub run_id: i64,
    /// Tick of the checkpoint.
    pub tick: u64,
    /// Configuration the run was started with.
    pub config: FullConfig,
    /// Run state at the checkpoint.
    pub state: SupervisorRunState,
    /// Kernel state at the checkpoint.
    pub kernel: KernelCheckpoint,
    /// Physics rows at the checkpoint.
    pub physics: Vec<f32>,
    /// Food rows at the checkpoint, present exactly when the world has food items
    /// (the food kinds setting is not considered).
    pub food: Option<Vec<f32>>,
}

fn malformed(why: impl fmt::Display) -> CheckpointError {
    CheckpointError::Malformed(why.to_string())
}

/// Reads, decodes and config-checks the newest checkpoint.
fn load_envelope(db_path: &str) -> Result<(i64, Checkpoint, FullConfig), CheckpointError> {
    let stored = read_checkpoint(db_path)
        .map_err(malformed)?
        .ok_or(CheckpointError::Missing)?;
    let envelope = decode(&stored.payload)?;
    let config: FullConfig =
        serde_json::from_slice(envelope.section(SECTION_CONFIG)?).map_err(malformed)?;
    let db = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(malformed)?;
    db.busy_timeout(std::time::Duration::from_secs(5))
        .map_err(malformed)?;
    let row: (String, String, String) = db
        .query_row(
            "SELECT governor_config, brain_config, world_config FROM run WHERE id = ?1",
            params![stored.run_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(malformed)?;
    let current = (
        serde_json::to_string(&config.governor).map_err(malformed)?,
        serde_json::to_string(&config.brain).map_err(malformed)?,
        serde_json::to_string(&config.world).map_err(malformed)?,
    );
    if row != current {
        return Err(CheckpointError::ConfigMismatch);
    }
    Ok((stored.run_id, envelope, config))
}

/// Loads only the configuration of the newest checkpoint, checked against the run row.
///
/// # Errors
///
/// Returns a [`CheckpointError`] when the checkpoint is missing, damaged or
/// was written for a different configuration.
pub fn load_checkpoint_config(db_path: &str) -> Result<FullConfig, CheckpointError> {
    load_envelope(db_path).map(|(_, _, config)| config)
}

/// Loads and checks the newest checkpoint of the run in `db_path`.
///
/// # Errors
///
/// Returns a [`CheckpointError`] when the checkpoint is missing, damaged,
/// inconsistent with its run, or was written for a different configuration.
pub fn load_resume_point(db_path: &str) -> Result<ResumePoint, CheckpointError> {
    let (run_id, envelope, config) = load_envelope(db_path)?;
    let tick = envelope.tick;
    let state: SupervisorRunState =
        serde_json::from_slice(envelope.section(SECTION_RUN)?).map_err(malformed)?;
    let mut cpu = None;
    let mut sections = Vec::new();
    for (name, bytes) in &envelope.sections {
        let Some(short) = name.strip_prefix(SECTION_KERNEL_PREFIX) else {
            continue;
        };
        if name == SECTION_KERNEL_CPU {
            cpu = Some(KernelCpuState::from_bytes(bytes).map_err(malformed)?);
            continue;
        }
        let known = KERNEL_SECTION_NAMES
            .iter()
            .copied()
            .find(|n| *n == short)
            .ok_or_else(|| malformed(format!("unknown kernel section {name}")))?;
        sections.push(KernelSection {
            name: known,
            bytes: bytes.clone(),
        });
    }
    let cpu = cpu.ok_or(CheckpointError::Missing)?;
    let kernel = KernelCheckpoint::new(sections, cpu);
    if tick % SUPERVISOR_SAMPLE_TICKS != 0 {
        return Err(malformed("checkpoint tick is not on the sample grid"));
    }
    if state.tick >= tick {
        return Err(malformed("run tick is not before the checkpoint tick"));
    }
    let core = Supervisor::restore(state.core.clone()).map_err(malformed)?;
    if core.next_sample_tick() != tick {
        return Err(malformed(
            "next sample tick differs from the checkpoint tick",
        ));
    }
    if core.standings(state.tick).len() != config.governor.population_size {
        return Err(malformed("slot count differs from the population size"));
    }
    let physics = kernel.physics().to_vec();
    let food = kernel.food().map(<[f32]>::to_vec);
    let has_food = !WorldState::new(config.world.clone()).food_items.is_empty();
    if food.is_some() != has_food {
        return Err(malformed("food rows do not match the world's food"));
    }
    Ok(ResumePoint {
        run_id,
        tick,
        config,
        state,
        kernel,
        physics,
        food,
    })
}

/// Boundary report and replacement outcome of one sample.
type BoundaryResult = (Option<BoundaryReport>, Option<ReplacementOutcome>);

impl SupervisorRun {
    /// Validates, uploads the initial population and records the slots.
    ///
    /// # Errors
    ///
    /// Returns a [`SupervisorRunError`] for an infeasible config, a kernel of
    /// the wrong size or batch, or a failed write.
    pub fn start(
        kernel: &mut GpuKernel,
        config: &FullConfig,
        store: Option<SupervisorStore>,
        measure: bool,
    ) -> Result<Self, SupervisorRunError> {
        let population = config.governor.population_size;
        Self::validate(kernel, config, population)?;
        let seed = config.world.seed;
        let world = WorldState::new(config.world.clone());
        let core = Supervisor::new(
            &config.governor,
            &config.brain,
            seed,
            config.world.world_size,
            population,
        )?;
        let kinds_on = config.world.food_kinds_enabled;
        let accumulator = measure.then(|| {
            RunSummaryAccumulator::new(
                Some(config.governor.supervisor.inheritance),
                seed,
                population,
                kinds_on,
            )
        });
        let telemetry_stats = kinds_on.then(|| {
            RunSummaryAccumulator::new(
                Some(config.governor.supervisor.inheritance),
                seed,
                population,
                kinds_on,
            )
        });
        let recording = store.is_some();
        let sup = &config.governor.supervisor;
        let next_occupant = u64::try_from(population)
            .map_err(|_| SupervisorRunError::SlotRange { slot: population })?;
        let mut run = Self {
            core,
            world,
            store,
            accumulator,
            population,
            kinds_on,
            season: TickSchedule::new(stream_seed(seed, SEASON_STREAM), sup.season_period_ticks),
            nourishing: None,
            tick: 0,
            budget: sup.agent_tick_budget,
            next_occupant,
            telemetry: TelemetryBuilder::new(population),
            recording,
            telemetry_stats,
            last_ticks_per_second: None,
            last_frame: None,
            telemetry_warned: false,
            config: config.clone(),
            spawn_draws: 0,
            telemetry_mirror: TelemetryMirror::default(),
        };
        run.upload_population(kernel, config)?;
        run.begin_run_season(kernel)?;
        if let (Some(stats), Some(kind)) = (run.telemetry_stats.as_mut(), run.nourishing) {
            if let Err(e) = stats.begin_season(0, Some(kind)) {
                run.telemetry_failed(&e);
            }
        }
        Ok(run)
    }

    /// Rebuilds a run from `state` on a freshly built kernel, restoring the
    /// kernel from `checkpoint` and rewriting every slot row.
    ///
    /// # Errors
    ///
    /// Returns a [`SupervisorRunError`] for an infeasible config, a kernel of
    /// the wrong size or batch, a state that does not fit the config, a kernel
    /// that refuses the checkpoint, or a failed write.
    pub fn resume(
        kernel: &mut GpuKernel,
        config: &FullConfig,
        store: Option<SupervisorStore>,
        state: SupervisorRunState,
        checkpoint: &KernelCheckpoint,
    ) -> Result<Self, SupervisorRunError> {
        let population = config.governor.population_size;
        Self::validate(kernel, config, population)?;
        let SupervisorRunState {
            core,
            season,
            accumulator,
            population: state_population,
            kinds_on,
            nourishing,
            tick,
            budget,
            next_occupant,
            spawn_draws,
            recording,
            telemetry_stats,
            last_ticks_per_second_bits,
            telemetry_warned,
            telemetry_mirror,
            last_frame,
        } = state;
        if state_population != population || kinds_on != config.world.food_kinds_enabled {
            return Err(CheckpointError::ConfigMismatch.into());
        }
        let core = Supervisor::restore(core)?;
        let season = season.rebuild()?;
        let mut run = Self {
            core,
            world: WorldState::new(config.world.clone()),
            store,
            accumulator,
            population,
            kinds_on,
            season,
            nourishing,
            tick,
            budget,
            next_occupant,
            telemetry: rebuild_telemetry(population, &telemetry_mirror),
            recording,
            telemetry_stats,
            last_ticks_per_second: last_ticks_per_second_bits.map(f64::from_bits),
            last_frame,
            telemetry_warned,
            config: config.clone(),
            spawn_draws: 0,
            telemetry_mirror,
        };
        for _ in 0..spawn_draws {
            run.spawn_position();
        }
        upload_world(kernel, &run.world);
        kernel.restore_checkpoint(checkpoint)?;
        if let Some(store) = run.store.as_mut() {
            for slot in 0..population {
                store.record_slot(slot, run.core.slot(slot)?)?;
            }
        }
        Ok(run)
    }

    /// One newborn spawn position; every spawn draw goes through here.
    fn spawn_position(&mut self) -> glam::Vec3 {
        self.spawn_draws += 1;
        self.world.safe_spawn_position()
    }

    /// Whether a checkpoint is due at `tick`.
    pub fn checkpoint_due(&self, tick: u64) -> bool {
        self.store.is_some() && tick == self.core.next_boundary_tick()
    }

    /// Captures the run state; every field is named so a new one fails to compile.
    pub fn capture_state(&self) -> SupervisorRunState {
        let SupervisorRun {
            core,
            world,
            store,
            accumulator,
            population,
            kinds_on,
            season,
            nourishing,
            tick,
            budget,
            next_occupant,
            telemetry,
            recording,
            telemetry_stats,
            last_ticks_per_second,
            last_frame,
            telemetry_warned,
            config,
            spawn_draws,
            telemetry_mirror,
        } = self;
        // Not owned by the state: rebuilt on resume from the config, the
        // telemetry mirror, or the store.
        let _ = (world, store, config, telemetry);
        SupervisorRunState {
            core: core.capture(),
            season: TickScheduleState::capture(season),
            accumulator: accumulator.clone(),
            population: *population,
            kinds_on: *kinds_on,
            nourishing: *nourishing,
            tick: *tick,
            budget: *budget,
            next_occupant: *next_occupant,
            spawn_draws: *spawn_draws,
            recording: *recording,
            telemetry_stats: telemetry_stats.clone(),
            last_ticks_per_second_bits: last_ticks_per_second.map(f64::to_bits),
            telemetry_warned: *telemetry_warned,
            telemetry_mirror: telemetry_mirror.clone(),
            last_frame: last_frame.clone(),
        }
    }

    /// Writes the pre-sample state as one checkpoint at `tick`; without a
    /// store nothing is written.
    ///
    /// # Errors
    ///
    /// Returns [`SupervisorRunError::TickMismatch`] unless `tick` is the next
    /// sample tick, and other errors when serialisation or the store fails.
    pub fn write_checkpoint(
        &mut self,
        tick: u64,
        kernel: &KernelCheckpoint,
    ) -> Result<(), SupervisorRunError> {
        if tick != self.next_sample_tick() {
            return Err(SupervisorRunError::TickMismatch {
                expected: self.next_sample_tick(),
                actual: tick,
            });
        }
        if self.store.is_none() {
            return Ok(());
        }
        let mut sections = vec![
            (
                SECTION_CONFIG.to_string(),
                serde_json::to_vec(&self.config).map_err(malformed)?,
            ),
            (
                SECTION_RUN.to_string(),
                serde_json::to_vec(&self.capture_state()).map_err(malformed)?,
            ),
        ];
        for section in kernel.sections() {
            sections.push((
                format!("{SECTION_KERNEL_PREFIX}{}", section.name),
                section.bytes.clone(),
            ));
        }
        sections.push((SECTION_KERNEL_CPU.to_string(), kernel.cpu().to_bytes()));
        let payload = encode(&Checkpoint { tick, sections });
        if let Some(store) = self.store.as_mut() {
            store.write_checkpoint(tick, &payload)?;
        }
        Ok(())
    }

    fn validate(
        kernel: &GpuKernel,
        config: &FullConfig,
        population: usize,
    ) -> Result<(), SupervisorRunError> {
        config.governor.supervisor.validate(population)?;
        if usize::try_from(kernel.agent_count()).ok() != Some(population) {
            return Err(SupervisorRunError::AgentCount {
                expected: population,
                actual: kernel.agent_count(),
            });
        }
        let batch = kernel.kernel_batch_size();
        if batch == 0 || SUPERVISOR_SAMPLE_TICKS % u64::from(batch) != 0 {
            return Err(SupervisorRunError::BatchSize { batch });
        }
        Ok(())
    }

    /// Uploads world and agents, resets, writes every slot and records it.
    fn upload_population(
        &mut self,
        kernel: &mut GpuKernel,
        config: &FullConfig,
    ) -> Result<(), SupervisorRunError> {
        upload_world(kernel, &self.world);
        let mut agents: Vec<Agent> = Vec::with_capacity(self.population);
        for slot in 0..self.population {
            let id = slot_index(slot)?;
            let position = self.spawn_position();
            agents.push(Agent::new(id, position, id, config.brain.clone(), 0));
        }
        let rows: Vec<(glam::Vec3, f32, f32, usize, usize)> = agents
            .iter()
            .map(|a| {
                let s = spawn_of(a);
                (
                    s.position,
                    s.max_energy,
                    s.max_integrity,
                    s.memory_capacity,
                    s.processing_slots,
                )
            })
            .collect();
        kernel.upload_agents(&rows);
        kernel.reset_agents_seeded(&config.brain, config.world.seed ^ RESET_SEED_TAG);
        for (slot, agent) in agents.iter().enumerate() {
            let ledger = self.core.slot(slot)?;
            kernel.replace_agent_slot(
                slot_index(slot)?,
                &ledger.birth_brain,
                &ledger.config,
                &spawn_of(agent),
            )?;
            if let Some(store) = self.store.as_mut() {
                store.record_slot(slot, ledger)?;
            }
        }
        Ok(())
    }

    /// Opens the first season when food kinds are on.
    fn begin_run_season(&mut self, kernel: &GpuKernel) -> Result<(), SupervisorRunError> {
        if !self.kinds_on {
            return Ok(());
        }
        let kind = kernel
            .nourishing_food_kind()
            .ok_or(SupervisorRunError::MissingNourishingKind)?;
        self.nourishing = Some(kind);
        if let Some(acc) = self.accumulator.as_mut() {
            acc.begin_season(0, Some(kind))?;
        }
        if let Some(store) = self.store.as_mut() {
            store.record_season(0, kind)?;
        }
        Ok(())
    }

    /// Ticks simulated so far (the last sample handled).
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// The tick the next sample must carry.
    pub fn next_sample_tick(&self) -> u64 {
        self.core.next_sample_tick()
    }

    /// Total ticks of the run; `None` when the budget is unlimited.
    pub fn total_ticks(&self) -> Option<u64> {
        let population = u64::try_from(self.population).ok()?;
        (self.budget != 0 && population != 0).then(|| self.budget / population)
    }

    /// Whether the run reached its tick budget.
    pub fn is_finished(&self) -> bool {
        self.total_ticks().is_some_and(|total| self.tick >= total)
    }

    /// Birth brain of the current occupant of `slot`.
    ///
    /// # Errors
    ///
    /// Returns an error when `slot` does not exist.
    pub fn birth_state(&self, slot: usize) -> Result<&AgentBrainState, SupervisorRunError> {
        Ok(&self.core.slot(slot)?.birth_brain)
    }

    /// Handles the readback at `tick`: ledger, measurement, season, boundary.
    ///
    /// # Errors
    ///
    /// Returns a [`SupervisorRunError`] for a wrong tick or length, or when a
    /// kernel, store or summary step fails.
    pub fn on_sample(
        &mut self,
        kernel: &mut GpuKernel,
        tick: u64,
        physics: &[f32],
        food: Option<&[f32]>,
    ) -> Result<SampleReport, SupervisorRunError> {
        if tick != self.next_sample_tick() {
            return Err(SupervisorRunError::TickMismatch {
                expected: self.next_sample_tick(),
                actual: tick,
            });
        }
        let expected = self.population * PHYS_STRIDE;
        if physics.len() != expected {
            return Err(SupervisorRunError::PhysicsLength {
                expected,
                actual: physics.len(),
            });
        }
        let decision = self.core.record_sample(tick, &self.counters(physics)?)?;
        if let Some(acc) = self.accumulator.as_mut() {
            acc.record_physics_sample(tick, physics, food)?;
        }
        self.telemetry_physics(tick, physics, food);
        self.tick = tick;
        let standings = self.core.standings(tick);
        let had_boundary = !matches!(decision, SampleDecision::Recorded);
        let season = self.flip_season(kernel, tick)?;
        if let Some(kind) = season {
            self.telemetry_season(tick, kind);
        }
        let (boundary, outcome) = self.boundary(kernel, tick, decision)?;
        self.build_frame(
            tick,
            &standings,
            had_boundary,
            outcome.as_ref(),
            season.is_some(),
        );
        Ok(SampleReport {
            tick,
            boundary,
            season,
            finished: self.is_finished(),
        })
    }

    /// Turns recording off after a telemetry failure and reports it once.
    fn telemetry_failed(&mut self, error: &dyn fmt::Display) {
        self.recording = false;
        self.telemetry_stats = None;
        if !self.telemetry_warned {
            self.telemetry_warned = true;
            log::warn!("supervisor telemetry stopped: {error}");
        }
    }

    /// Feeds the telemetry accumulator one physics sample.
    fn telemetry_physics(&mut self, tick: u64, physics: &[f32], food: Option<&[f32]>) {
        if let Some(stats) = self.telemetry_stats.as_mut() {
            if let Err(e) = stats.record_physics_sample(tick, physics, food) {
                self.telemetry_failed(&e);
            }
        }
    }

    /// Opens a telemetry season and hands the closed totals to the builder.
    fn telemetry_season(&mut self, tick: u64, kind: FoodKind) {
        let Some(stats) = self.telemetry_stats.as_mut() else {
            return;
        };
        if let Err(e) = stats.begin_season(tick, Some(kind)) {
            self.telemetry_failed(&e);
            return;
        }
        if let Some(totals) = stats.closed_totals() {
            self.telemetry.on_season_closed(tick, totals);
            let bits = KindTotalsBits::from(totals);
            self.telemetry_mirror.seen = Some(bits.clone());
            self.telemetry_mirror.closures.push((tick, bits));
        }
    }

    /// Builds this sample's frame and persists it when recording.
    fn build_frame(
        &mut self,
        tick: u64,
        standings: &[SlotStanding],
        had_boundary: bool,
        outcome: Option<&ReplacementOutcome>,
        season_flipped: bool,
    ) {
        let archive_outcomes: Vec<f32> = self
            .core
            .archive()
            .entries()
            .iter()
            .map(|e| e.outcome)
            .collect();
        let frame = self.telemetry.on_sample(SampleInput {
            tick,
            standings,
            boundary: if had_boundary {
                self.core.last_boundary()
            } else {
                None
            },
            replacement: outcome.map(|o| NewbornInput {
                newborn_occupant: o.occupant_id,
                parent_config: &o.parent_config,
                newborn_config: &o.newborn_config,
            }),
            archive_outcomes: &archive_outcomes,
            total_ticks: self.total_ticks(),
            agent_tick_budget: self.budget,
            next_boundary_tick: self.core.next_boundary_tick(),
            nourishing: self.nourishing,
            next_flip_tick: self.kinds_on.then(|| self.season.next_tick()),
            ticks_per_second: self.last_ticks_per_second,
            season_flipped,
        });
        if self.recording && frame.persist {
            if let Some(store) = self.store.as_mut() {
                if let Err(e) = store.record_telemetry(&frame) {
                    self.telemetry_failed(&e);
                }
            }
        }
        if frame.persist {
            let mirror = &mut self.telemetry_mirror;
            mirror.persisted = Some(TelemetryPersisted {
                tick,
                standings: standings
                    .iter()
                    .map(|s| TelemetryStanding {
                        slot: s.slot,
                        occupant_id: s.occupant_id,
                        meals: s.meals,
                        deaths: s.deaths,
                    })
                    .collect(),
            });
            mirror.seen_base = mirror.seen.clone();
            mirror.closures.clear();
        }
        self.last_frame = Some(frame);
    }

    /// Turns telemetry persistence on or off.
    pub fn set_recording(&mut self, on: bool) {
        self.recording = on && self.store.is_some();
    }

    /// Stores the latest observed ticks per second; non-finite rates are ignored.
    pub fn note_ticks_per_second(&mut self, rate: f64) {
        if rate.is_finite() {
            self.last_ticks_per_second = Some(rate);
        }
    }

    /// Takes the frame of the last sample.
    pub fn take_frame(&mut self) -> Option<TelemetryFrame> {
        self.last_frame.take()
    }

    fn counters(&self, physics: &[f32]) -> Result<Vec<SlotCounters>, SupervisorRunError> {
        let size = self.core.world_size();
        physics
            .as_chunks::<PHYS_STRIDE>()
            .0
            .iter()
            .map(|row| SlotCounters::from_physics(row, size).map_err(Into::into))
            .collect()
    }

    /// Flips the nourishing kind when `tick` is a season tick.
    fn flip_season(
        &mut self,
        kernel: &mut GpuKernel,
        tick: u64,
    ) -> Result<Option<FoodKind>, SupervisorRunError> {
        let current = match self.nourishing {
            Some(kind) if self.kinds_on && tick == self.season.next_tick() => kind,
            _ => return Ok(None),
        };
        self.season.advance();
        let new = other_kind(current);
        kernel.set_nourishing_food_kind(new)?;
        self.nourishing = Some(new);
        if let Some(store) = self.store.as_mut() {
            store.record_season(tick, new)?;
        }
        if let Some(acc) = self.accumulator.as_mut() {
            acc.begin_season(tick, Some(new))?;
        }
        Ok(Some(new))
    }

    fn boundary(
        &mut self,
        kernel: &GpuKernel,
        tick: u64,
        decision: SampleDecision,
    ) -> Result<BoundaryResult, SupervisorRunError> {
        match decision {
            SampleDecision::Recorded => Ok((None, None)),
            SampleDecision::Idle => {
                if let Some(store) = self.store.as_mut() {
                    store.record_idle_boundary(tick)?;
                }
                Ok((Some(self.report(tick, None, None)), None))
            }
            SampleDecision::Replace(plan) => {
                let (report, outcome) = self.replace(kernel, tick, plan)?;
                Ok((Some(report), Some(outcome)))
            }
        }
    }

    fn replace(
        &mut self,
        kernel: &GpuKernel,
        tick: u64,
        plan: ReplacementPlan,
    ) -> Result<(BoundaryReport, ReplacementOutcome), SupervisorRunError> {
        // Every fallible conversion runs before the core commits.
        let slot = slot_index(plan.slot)?;
        let id =
            u32::try_from(self.next_occupant).map_err(|_| SupervisorRunError::OccupantRange {
                occupant_id: self.next_occupant,
            })?;
        let learned = match plan.parent {
            ParentRef::PeerLearned(peer) => Some(kernel.read_agent_state(slot_index(peer)?)),
            _ => None,
        };
        let outcome = self
            .core
            .complete_replacement(plan.clone(), tick, learned.as_ref())?;
        self.next_occupant = outcome.occupant_id.saturating_add(1);
        let position = self.spawn_position();
        let agent = Agent::new(id, position, slot, outcome.newborn_config.clone(), tick);
        kernel.replace_agent_slot(
            slot,
            &outcome.newborn_brain,
            &outcome.newborn_config,
            &spawn_of(&agent),
        )?;
        if let Some(acc) = self.accumulator.as_mut() {
            acc.reset_agent_baseline(plan.slot)?;
        }
        if let Some(stats) = self.telemetry_stats.as_mut() {
            if let Err(e) = stats.reset_agent_baseline(plan.slot) {
                self.telemetry_failed(&e);
            }
        }
        if let Some(store) = self.store.as_mut() {
            store.record_boundary(tick, &plan, &outcome, outcome.archived.as_ref())?;
        }
        let report = self.report(tick, Some(&plan), Some(outcome.occupant_id));
        Ok((report, outcome))
    }

    fn report(
        &self,
        tick: u64,
        plan: Option<&ReplacementPlan>,
        occupant_id: Option<u64>,
    ) -> BoundaryReport {
        let archive = self.core.archive();
        BoundaryReport {
            tick,
            slot: plan.map(|p| p.slot),
            reason: plan.map(|p| p.reason),
            occupant_id,
            parent: plan.map(|p| p.parent),
            archive_len: archive.len(),
            best_outcome: archive.entries().first().map(|e| e.outcome),
        }
    }

    /// Archives eligible living slots, records the ticks and returns the summary.
    ///
    /// # Errors
    ///
    /// Returns a [`SupervisorRunError`] when a store write or the summary fails.
    pub fn finish(mut self) -> Result<SupervisorFinish, SupervisorRunError> {
        let changes: Vec<ArchiveChange> = self.core.finish(self.tick);
        if let Some(store) = self.store.as_mut() {
            store.record_archive_changes(self.tick, &changes, self.core.archive())?;
            store.record_ticks(self.tick)?;
        }
        let summary = self
            .accumulator
            .take()
            .map(|acc| acc.finish(self.tick))
            .transpose()?;
        Ok(SupervisorFinish { summary })
    }
}

/// Bit-exact serde for [`TelemetryFrame`] and its parts, through remote
/// definitions of the public telemetry and supervisor types.
mod frame_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use serde_json::Value;
    use xagent_shared::FoodKind;

    use crate::supervisor::{
        BoundaryDetail, EligibilityGround, IdleReason, ParentDraw, ParentRef, ReplaceReason,
        SlotStanding, TieBreak,
    };
    use crate::supervisor_checkpoint::f32_bits;
    use crate::supervisor_telemetry::{
        BoundaryRecord, ConfigFieldDiff, IntakeRow, Metric, ProgressRecord, SeriesPoint,
        TelemetryFrame,
    };

    mod opt_f32 {
        use serde::{Deserialize, Deserializer, Serialize, Serializer};

        #[allow(clippy::ref_option, clippy::trivially_copy_pass_by_ref)]
        pub fn serialize<S: Serializer>(v: &Option<f32>, s: S) -> Result<S::Ok, S::Error> {
            v.map(f32::to_bits).serialize(s)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f32>, D::Error> {
            Ok(Option::<u32>::deserialize(d)?.map(f32::from_bits))
        }
    }

    mod f64_bits {
        use serde::{Deserialize, Deserializer, Serialize, Serializer};

        #[allow(clippy::trivially_copy_pass_by_ref)]
        pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
            v.to_bits().serialize(s)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
            Ok(f64::from_bits(u64::deserialize(d)?))
        }
    }

    mod opt_f64 {
        use serde::{Deserialize, Deserializer, Serialize, Serializer};

        #[allow(clippy::ref_option)]
        pub fn serialize<S: Serializer>(v: &Option<f64>, s: S) -> Result<S::Ok, S::Error> {
            v.map(f64::to_bits).serialize(s)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
            Ok(Option::<u64>::deserialize(d)?.map(f64::from_bits))
        }
    }

    /// A `with` module for `Vec<$ty>` or `Option<$ty>` through a remote definition.
    macro_rules! wrapped {
        ($module:ident, vec, $ty:ty, $def:literal) => {
            pub mod $module {
                use super::*;

                #[derive(Serialize, Deserialize)]
                struct W(#[serde(with = $def)] $ty);

                pub fn serialize<S: Serializer>(v: &[$ty], s: S) -> Result<S::Ok, S::Error> {
                    s.collect_seq(v.iter().map(|x| W(x.clone())))
                }

                pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<$ty>, D::Error> {
                    Ok(Vec::<W>::deserialize(d)?.into_iter().map(|w| w.0).collect())
                }
            }
        };
        ($module:ident, opt, $ty:ty, $def:literal) => {
            pub mod $module {
                use super::*;

                #[derive(Serialize, Deserialize)]
                struct W(#[serde(with = $def)] $ty);

                #[allow(clippy::ref_option)]
                pub fn serialize<S: Serializer>(v: &Option<$ty>, s: S) -> Result<S::Ok, S::Error> {
                    v.clone().map(W).serialize(s)
                }

                pub fn deserialize<'de, D: Deserializer<'de>>(
                    d: D,
                ) -> Result<Option<$ty>, D::Error> {
                    Ok(Option::<W>::deserialize(d)?.map(|w| w.0))
                }
            }
        };
    }

    #[derive(Serialize, Deserialize)]
    #[serde(remote = "EligibilityGround")]
    enum EligibilityGroundDef {
        Death,
        BelowMedian {
            #[serde(with = "f32_bits")]
            score: f32,
            #[serde(with = "f32_bits")]
            median: f32,
        },
        TooYoung,
        NoFullWindow,
        NoPeerMedian,
        AtOrAboveMedian {
            #[serde(with = "f32_bits")]
            score: f32,
            #[serde(with = "f32_bits")]
            median: f32,
        },
    }

    #[derive(Serialize, Deserialize)]
    #[serde(remote = "SlotStanding")]
    struct SlotStandingDef {
        slot: usize,
        occupant_id: u64,
        age_ticks: u64,
        alive: bool,
        meals: u32,
        deaths: u32,
        #[serde(with = "opt_f32")]
        window_score: Option<f32>,
        #[serde(with = "opt_f32")]
        peer_median: Option<f32>,
        #[serde(with = "EligibilityGroundDef")]
        ground: EligibilityGround,
        parent: Option<ParentRef>,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(remote = "ReplaceReason")]
    enum ReplaceReasonDef {
        Death,
        Underperformer,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(remote = "TieBreak")]
    struct TieBreakDef {
        pool: Vec<usize>,
        drawn_index: usize,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(remote = "ParentDraw")]
    struct ParentDrawDef {
        pool_len: usize,
        drawn_index: usize,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(remote = "IdleReason")]
    struct IdleReasonDef {
        too_young: usize,
        no_full_window: usize,
        no_peer_median: usize,
        at_or_above_median: usize,
    }

    wrapped!(standings, vec, SlotStanding, "SlotStandingDef");
    wrapped!(opt_draw, opt, ParentDraw, "ParentDrawDef");

    #[derive(Serialize, Deserialize)]
    #[serde(remote = "BoundaryDetail")]
    enum BoundaryDetailDef {
        Replaced {
            tick: u64,
            #[serde(with = "standings")]
            candidates: Vec<SlotStanding>,
            slot: usize,
            #[serde(with = "ReplaceReasonDef")]
            reason: ReplaceReason,
            #[serde(with = "TieBreakDef")]
            tiebreak: TieBreak,
            parent: ParentRef,
            #[serde(with = "opt_draw")]
            parent_draw: Option<ParentDraw>,
            retired_occupant: u64,
            retired_born_tick: u64,
        },
        Idle {
            tick: u64,
            #[serde(with = "standings")]
            standings: Vec<SlotStanding>,
            #[serde(with = "IdleReasonDef")]
            reason: IdleReason,
        },
    }

    #[derive(Serialize, Deserialize)]
    #[serde(remote = "ConfigFieldDiff")]
    struct ConfigFieldDiffDef {
        field: String,
        parent: Value,
        newborn: Value,
    }

    wrapped!(diffs, vec, ConfigFieldDiff, "ConfigFieldDiffDef");

    #[derive(Serialize, Deserialize)]
    #[serde(remote = "BoundaryRecord")]
    struct BoundaryRecordDef {
        #[serde(with = "BoundaryDetailDef")]
        detail: BoundaryDetail,
        newborn_occupant: Option<u64>,
        #[serde(with = "diffs")]
        config_diff: Vec<ConfigFieldDiff>,
        retired_lifespan_ticks: Option<u64>,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(remote = "Metric")]
    enum MetricDef {
        FoodPerAgentTick,
        DeathsPerAgentTick,
        WindowScoreSpread,
        RetiredLifespanTicks,
        ArchiveBestOutcome,
        ArchiveMedianOutcome,
        LateTowardNourishing,
        LateAwayHarmful,
        LateNourishingShare,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(remote = "SeriesPoint")]
    struct SeriesPointDef {
        tick: u64,
        #[serde(with = "MetricDef")]
        metric: Metric,
        #[serde(with = "f64_bits")]
        value: f64,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(remote = "IntakeRow")]
    struct IntakeRowDef {
        occupant_id: u64,
        slot: usize,
        age_ticks: u64,
        meals: u32,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(remote = "ProgressRecord")]
    struct ProgressRecordDef {
        tick: u64,
        total_ticks: Option<u64>,
        agent_tick_budget: u64,
        ticks_to_boundary: u64,
        nourishing: Option<FoodKind>,
        next_flip_tick: Option<u64>,
        #[serde(with = "opt_f64")]
        ticks_per_second: Option<f64>,
    }

    wrapped!(opt_boundary, opt, BoundaryRecord, "BoundaryRecordDef");
    wrapped!(series, vec, SeriesPoint, "SeriesPointDef");
    wrapped!(intake, vec, IntakeRow, "IntakeRowDef");

    #[derive(Serialize, Deserialize)]
    #[serde(remote = "TelemetryFrame")]
    struct TelemetryFrameDef {
        tick: u64,
        #[serde(with = "standings")]
        slots: Vec<SlotStanding>,
        #[serde(with = "opt_boundary")]
        boundary: Option<BoundaryRecord>,
        #[serde(with = "series")]
        series: Vec<SeriesPoint>,
        #[serde(with = "intake")]
        intake: Vec<IntakeRow>,
        #[serde(with = "ProgressRecordDef")]
        progress: ProgressRecord,
        persist: bool,
    }

    wrapped!(opt_frame, opt, TelemetryFrame, "TelemetryFrameDef");
}
