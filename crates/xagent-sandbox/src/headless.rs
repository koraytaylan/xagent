//! Headless (no-window) supervisor run and free-life utilities.
//!
//! Uses GpuKernel for all simulation — physics, brain, food, and
//! death/respawn run entirely on GPU via fused kernel dispatch.

use std::fmt;
use std::path::PathBuf;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use xagent_shared::{BrainConfig, FullConfig, SUPERVISOR_SAMPLE_TICKS};

use crate::supervisor_checkpoint::CheckpointError;
use crate::supervisor_run::{
    load_resume_point, ResumePoint, SampleReport, SupervisorFinish, SupervisorRun,
    SupervisorRunError,
};
use crate::supervisor_store::{
    inspect_database, load_archive_entry_read_only, ArchiveLoadError, DatabaseKind, StoreError,
    SupervisorStore,
};

use xagent_brain::buffers::{
    fixed_tail_base, SenseBlockOffsets, P_FACING_X, P_FACING_Z, P_HARMFUL_MEAL_COUNT,
};
use xagent_brain::buffers::{
    BrainLayout, BrainTickProbe, TrunkLesions, O_PREDICTOR_CONTEXT_WEIGHT, PHYS_STRIDE, P_ALIVE,
    P_DEATH_COUNT, P_FOOD_COUNT, P_IN_DANGER_BIOME, P_MOTOR_TURN_OUT, P_NEAREST_DANGER_BEARING,
    P_NEAREST_DANGER_DISTANCE, P_NEAREST_FOOD_BEARING, P_NEAREST_FOOD_DISTANCE, P_POS_X, P_POS_Y,
    P_POS_Z, TD_TRACE_DECAY,
};
use xagent_brain::kind_readout::{fit_kind_readout, ReadoutSample};
use xagent_brain::{AgentBrainState, GpuKernel};
use xagent_shared::{FoodKind, SCENT_RANGE};

use crate::agent::{fresh_brain_state, Agent, MAX_AGENTS};
use crate::food_kind_metrics::{
    nearest_food_of_kind, signed_for_nourishing, signed_weight_difference, AgentFoodKindSummary,
    Arm, KindMealQuarters, KindTurnAccumulator, PairCreditAccumulator, RunFoodKindSummary,
    QUARTER_COUNT,
};
use crate::free_life::metrics::{AgentAccumulator, AgentMetrics, FreeLifeSample, MetricsConfig};
use crate::free_life::tick_log::TickLogWriter;
use crate::tick_window::{frame_from_snapshot, Pin, PinKind, TickWindow, TickWindowConfig};
use crate::tick_window_store::TickWindowStore;
use crate::world::WorldState;

/// Options of a headless run beyond the config.
#[derive(Clone, Debug, Default)]
pub struct HeadlessOptions {
    /// Where to write the run summary as JSON; `None` skips measurement.
    pub run_summary: Option<PathBuf>,
    /// Stop right after the first checkpoint at or after this tick.
    pub stop_after_checkpoint_at_or_after: Option<u64>,
}

/// Why a headless run failed.
#[derive(Debug)]
pub enum HeadlessError {
    /// The stored checkpoint is missing or refused.
    Checkpoint(CheckpointError),
    /// The supervisor store failed.
    Store(StoreError),
    /// The supervisor run failed.
    SupervisorRun(SupervisorRunError),
    /// The run summary could not be encoded, produced or written; encode
    /// failures arrive as `std::io::Error` through serde_json's conversion.
    Summary(std::io::Error),
    /// The tick window could not be captured or stored.
    TickWindow(String),
}

impl fmt::Display for HeadlessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpoint(e) => write!(f, "{e}"),
            Self::Store(e) => write!(f, "{e}"),
            Self::SupervisorRun(e) => write!(f, "{e}"),
            Self::Summary(e) => write!(f, "run summary could not be written: {e}"),
            Self::TickWindow(message) => write!(f, "tick window failed: {message}"),
        }
    }
}

impl std::error::Error for HeadlessError {}

impl From<CheckpointError> for HeadlessError {
    fn from(err: CheckpointError) -> Self {
        Self::Checkpoint(err)
    }
}

impl From<StoreError> for HeadlessError {
    fn from(err: StoreError) -> Self {
        Self::Store(err)
    }
}

impl From<SupervisorRunError> for HeadlessError {
    fn from(err: SupervisorRunError) -> Self {
        Self::SupervisorRun(err)
    }
}

impl From<std::io::Error> for HeadlessError {
    fn from(err: std::io::Error) -> Self {
        Self::Summary(err)
    }
}

impl From<serde_json::Error> for HeadlessError {
    fn from(err: serde_json::Error) -> Self {
        Self::Summary(err.into())
    }
}

/// Exit status of a headless run that failed.
const FAILURE_EXIT_STATUS: i32 = 1;

/// Run the headless supervisor: no window, no rendering, max speed.
/// An error is printed and the process exits with status 1.
pub fn run_headless(config: FullConfig, db_path: &str, resume: bool, _has_gpu: bool) {
    if let Err(err) =
        run_headless_with_options(config, db_path, resume, &HeadlessOptions::default())
    {
        eprintln!("{err}");
        std::process::exit(FAILURE_EXIT_STATUS);
    }
}

/// Run the headless supervisor driver.
///
/// # Errors
///
/// Returns a [`HeadlessError`] when the database holds a generation-mode run
/// (the file is left untouched) or for a failed supervisor run, store or
/// summary write.
pub fn run_headless_with_options(
    config: FullConfig,
    db_path: &str,
    resume: bool,
    options: &HeadlessOptions,
) -> Result<(), HeadlessError> {
    match inspect_database(std::path::Path::new(db_path))? {
        DatabaseKind::Missing | DatabaseKind::Empty | DatabaseKind::Supervisor => {}
        DatabaseKind::LegacyGenerations { generation } => {
            return Err(HeadlessError::Store(StoreError::LegacyGenerations {
                path: db_path.to_owned(),
                generation,
            }));
        }
    }
    if resume {
        return resume_supervisor(db_path, &config, options);
    }
    run_supervisor(config, db_path, options, None)
}

/// One sample period as the kernel's batch tick count, checked at compile time.
const SAMPLE_BATCH_TICKS: u32 = {
    assert!(SUPERVISOR_SAMPLE_TICKS <= u32::MAX as u64);
    SUPERVISOR_SAMPLE_TICKS as u32
};

/// Run a fresh supervisor session with an optional tick window.
///
/// # Errors
///
/// Returns any error of the supervisor run.
pub fn run_headless_supervisor_with_tick_window(
    config: FullConfig,
    db_path: &str,
    options: &HeadlessOptions,
    tick_window: Option<TickWindowConfig>,
) -> Result<(), HeadlessError> {
    run_supervisor(config, db_path, options, tick_window)
}

/// Run the supervisor loop on one kernel and write the summary when asked.
fn run_supervisor(
    config: FullConfig,
    db_path: &str,
    options: &HeadlessOptions,
    tick_window: Option<TickWindowConfig>,
) -> Result<(), HeadlessError> {
    run_supervisor_inner(config, db_path, options, true, tick_window)
}

/// Run the supervisor loop, persisting decision telemetry only when `record` is set.
///
/// # Errors
///
/// Returns a [`HeadlessError`] for a failed store, run or summary write.
pub fn run_supervisor_with_recording(
    config: FullConfig,
    db_path: &str,
    options: &HeadlessOptions,
    record: bool,
) -> Result<(), HeadlessError> {
    run_supervisor_inner(config, db_path, options, record, None)
}

fn run_supervisor_inner(
    config: FullConfig,
    db_path: &str,
    options: &HeadlessOptions,
    record: bool,
    tick_window: Option<TickWindowConfig>,
) -> Result<(), HeadlessError> {
    let store = SupervisorStore::create(db_path, &config)?;
    // A population past u32 has a last slot index the kernel cannot address.
    let population = u32::try_from(config.governor.population_size).map_err(|_| {
        SupervisorRunError::SlotRange {
            slot: config.governor.population_size - 1,
        }
    })?;
    let batch = SAMPLE_BATCH_TICKS;
    let food_count = WorldState::new(config.world.clone()).food_items.len();
    let mut kernel = GpuKernel::new(population, food_count, &config.brain, &config.world);
    let mut run = SupervisorRun::start(
        &mut kernel,
        &config,
        Some(store),
        options.run_summary.is_some(),
    )?;
    run.set_recording(record);
    let mut capture = match tick_window {
        Some(window_config) => {
            let window = TickWindow::new(window_config.clone()).map_err(window_error)?;
            let store = TickWindowStore::open_for_latest_supervisor_run(db_path, window_config)
                .map_err(window_error)?;
            Some((window, store))
        }
        None => None,
    };
    println!("Supervisor run | Population: {population}");
    let mut next_progress = 1_u64;
    drive_supervisor(
        run,
        &mut kernel,
        options,
        batch,
        &mut next_progress,
        capture.as_mut(),
    )
}

/// Convert a tick-window failure into a [`HeadlessError`].
fn window_error(err: impl fmt::Display) -> HeadlessError {
    HeadlessError::TickWindow(err.to_string())
}

/// Record one sample's frame and pins after the run has consumed it.
fn capture_sample(
    window: &mut TickWindow,
    store: &mut TickWindowStore,
    report: &SampleReport,
    physics: &[f32],
    food: Option<&[f32]>,
) -> Result<(), HeadlessError> {
    let frame =
        frame_from_snapshot(report.tick, physics, food.unwrap_or(&[])).map_err(window_error)?;
    window.capture(frame.clone());
    store.record_frame(&frame).map_err(window_error)?;
    let mut pins = Vec::new();
    if let Some(slot) = report.boundary.as_ref().and_then(|b| b.slot) {
        pins.push(Pin {
            tick: report.tick,
            kind: PinKind::Replacement { slot },
        });
    }
    if report.season.is_some() {
        pins.push(Pin {
            tick: report.tick,
            kind: PinKind::SeasonFlip,
        });
    }
    for pin in pins {
        window.pin(pin);
        store.record_pin(pin).map_err(window_error)?;
    }
    Ok(())
}

/// Run the loop to completion (or the stop option) and write the summary.
fn drive_supervisor(
    mut run: SupervisorRun,
    kernel: &mut GpuKernel,
    options: &HeadlessOptions,
    batch: u32,
    next_progress: &mut u64,
    mut capture: Option<&mut (TickWindow, TickWindowStore)>,
) -> Result<(), HeadlessError> {
    while !run.is_finished() {
        let Some((report, _physics, _food)) =
            supervisor_step(&mut run, kernel, batch, options, capture.as_deref_mut())?
        else {
            return Ok(());
        };
        print_supervisor_progress(&report, run.total_ticks(), next_progress);
    }
    let finish = run.finish()?;
    write_run_summary(finish, options)
}

/// Resume the checkpointed supervisor run and continue it.
fn resume_supervisor(
    db_path: &str,
    supplied: &FullConfig,
    options: &HeadlessOptions,
) -> Result<(), HeadlessError> {
    if !std::path::Path::new(db_path).exists() {
        return Err(CheckpointError::Missing.into());
    }
    let point = load_resume_point(db_path)?;
    let ResumePoint {
        run_id,
        tick,
        config: stored,
        state,
        kernel: checkpoint,
        physics,
        food,
    } = point;
    if serde_json::to_string(supplied)? != serde_json::to_string(&stored)? {
        eprintln!(
            "The supplied config differs from the checkpoint; the checkpoint config is used."
        );
    }
    let population = u32::try_from(stored.governor.population_size).map_err(|_| {
        SupervisorRunError::SlotRange {
            slot: stored.governor.population_size - 1,
        }
    })?;
    let food_count = WorldState::new(stored.world.clone()).food_items.len();
    let mut kernel = GpuKernel::new(population, food_count, &stored.brain, &stored.world);
    let store = SupervisorStore::reopen_for_resume(db_path, run_id, tick)?;
    let mut run = SupervisorRun::resume(&mut kernel, &stored, Some(store), state, &checkpoint)?;
    println!("Supervisor resume | Population: {population} | tick {tick}");
    let mut next_progress = 1_u64;
    let report = run.on_sample(&mut kernel, tick, &physics, food.as_deref())?;
    print_supervisor_progress(&report, run.total_ticks(), &mut next_progress);
    drive_supervisor(
        run,
        &mut kernel,
        options,
        SAMPLE_BATCH_TICKS,
        &mut next_progress,
        None,
    )
}

/// A sample report with the physics and food copies it was made from.
type StepSample = (SampleReport, Vec<f32>, Option<Vec<f32>>);

/// Dispatch one sample period, read it back and hand it to the run. Returns
/// `None` when the stop option ended the run right after a checkpoint. When a
/// tick window is given, the sample is captured after `on_sample` returns.
fn supervisor_step(
    run: &mut SupervisorRun,
    kernel: &mut GpuKernel,
    batch: u32,
    options: &HeadlessOptions,
    mut capture: Option<&mut (TickWindow, TickWindowStore)>,
) -> Result<Option<StepSample>, HeadlessError> {
    let started = Instant::now();
    let sample_tick = run.next_sample_tick();
    if run.checkpoint_due(sample_tick) {
        kernel.dispatch_ticks(run.tick(), batch);
        let checkpoint = kernel.read_checkpoint_blocking();
        run.write_checkpoint(sample_tick, &checkpoint)?;
        if options
            .stop_after_checkpoint_at_or_after
            .is_some_and(|v| sample_tick >= v)
        {
            return Ok(None);
        }
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            run.note_ticks_per_second(f64::from(batch) / elapsed);
        }
        let physics = checkpoint.physics().to_vec();
        let food = checkpoint.food().map(<[f32]>::to_vec);
        let report = run.on_sample(kernel, sample_tick, &physics, food.as_deref())?;
        if let Some((window, store)) = capture.as_deref_mut() {
            capture_sample(window, store, &report, &physics, food.as_deref())?;
        }
        return Ok(Some((report, physics, food)));
    }
    kernel.dispatch_batch(run.tick(), batch);
    while !kernel.try_collect_state_snapshot() {
        std::thread::yield_now();
    }
    let elapsed = started.elapsed().as_secs_f64();
    if elapsed > 0.0 {
        run.note_ticks_per_second(f64::from(batch) / elapsed);
    }
    let physics = kernel.cached_state().to_vec();
    let food = kernel.cached_food_state().map(<[f32]>::to_vec);
    let report = run.on_sample(kernel, sample_tick, &physics, food.as_deref())?;
    if let Some((window, store)) = capture {
        capture_sample(window, store, &report, &physics, food.as_deref())?;
    }
    Ok(Some((report, physics, food)))
}

/// Number of progress steps across a bounded budget.
const PROGRESS_STEPS: u64 = 10;

/// Percent scale of the progress label.
const PERCENT: u64 = 100;

/// One line per replacement and season flip, plus one every 10% of the budget.
fn print_supervisor_progress(report: &SampleReport, total: Option<u64>, next_progress: &mut u64) {
    if let Some(boundary) = report.boundary.as_ref().filter(|b| b.slot.is_some()) {
        println!(
            "tick {} | replaced slot {:?} ({:?}) | archive {}",
            boundary.tick, boundary.slot, boundary.reason, boundary.archive_len
        );
    }
    if let Some(kind) = report.season {
        println!("tick {} | season flip: nourishing {:?}", report.tick, kind);
    }
    if let Some(total) = total.filter(|t| *t > 0) {
        while *next_progress <= PROGRESS_STEPS
            && report.tick >= total.saturating_mul(*next_progress) / PROGRESS_STEPS
        {
            println!(
                "tick {} / {} ({}%)",
                report.tick,
                total,
                *next_progress * PERCENT / PROGRESS_STEPS
            );
            *next_progress += 1;
        }
    }
}

/// Write the run summary when one was requested.
fn write_run_summary(
    finish: SupervisorFinish,
    options: &HeadlessOptions,
) -> Result<(), HeadlessError> {
    let Some(path) = options.run_summary.as_ref() else {
        return Ok(());
    };
    let summary = finish.summary.ok_or_else(|| {
        HeadlessError::Summary(std::io::Error::other("the run produced no summary"))
    })?;
    let json = serde_json::to_string_pretty(&summary)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Where a free life's birth brain comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BrainSource {
    /// One freshly seeded birth brain shared by every agent.
    Fresh,
    /// A supervisor archive entry, read only.
    Archive {
        /// Path of the supervisor database.
        db_path: String,
        /// Rank of the entry in the archive, best first.
        rank: usize,
    },
}

/// Default number of agents in a free life.
pub const DEFAULT_FREE_LIFE_AGENTS: usize = 10;

/// Default number of simulation ticks in a free life.
pub const DEFAULT_FREE_LIFE_TICKS: u64 = 300_000;

/// Name of the `turn-weights` lesion in a lesion list.
const LESION_TURN_WEIGHTS: &str = "turn-weights";
/// Name of the `forward-weights` lesion in a lesion list.
const LESION_FORWARD_WEIGHTS: &str = "forward-weights";
/// Name of the `memory-blend` lesion in a lesion list.
const LESION_MEMORY_BLEND: &str = "memory-blend";
/// Name of the `critic` lesion in a lesion list.
const LESION_CRITIC: &str = "critic";
/// List entry selecting every lesion.
const LESION_ALL: &str = "all";
/// List entry selecting no lesion.
const LESION_NONE: &str = "none";

/// Report label of a freshly seeded birth brain.
const SOURCE_FRESH: &str = "fresh";
/// Report label of a supervisor archive birth brain.
const SOURCE_ARCHIVE: &str = "archive";

/// Options of one seeded free life.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FreeLifeOptions {
    /// Number of agents, in `1..=MAX_AGENTS`.
    pub agent_count: usize,
    /// Simulation ticks; a non-zero multiple of the kernel batch size.
    pub ticks: u64,
    /// Seed of the world and of every random choice.
    pub seed: u64,
    /// Birth brain source.
    pub brain_source: BrainSource,
    /// Trunk parts left out of the brain.
    pub lesions: TrunkLesions,
    /// Optional per-tick log file.
    pub log_path: Option<PathBuf>,
}

/// Why a free life could not run.
#[derive(Debug)]
pub enum FreeLifeError {
    /// No GPU adapter is available.
    NoGpu,
    /// The brain config's vision stride is not 1.
    VisionStride(u32),
    /// The agent count is outside `1..=MAX_AGENTS`.
    AgentCount(usize),
    /// The tick count is zero or not a multiple of the kernel batch size.
    Ticks {
        /// Requested ticks.
        ticks: u64,
        /// Kernel batch size.
        batch: u32,
    },
    /// A lesion list names an unknown part.
    UnknownLesion(String),
    /// A readback returned fewer values than the run needs.
    ReadbackLength {
        /// Which readback.
        what: &'static str,
        /// Values needed.
        expected: usize,
        /// Values returned.
        found: usize,
    },
    /// The tick log could not be written.
    LogIo(String),
    /// The food-kind run was asked to start from a stored brain.
    FoodKindStoredBrain,
    /// The kernel drew no nourishing food kind.
    NoNourishingKind,
    /// A cumulative meal count decreased.
    MealCounts,
    /// A required sense is missing from the brain layout.
    MissingSense(&'static str),
    /// The food-kind summary could not be serialised or written.
    SummaryIo(String),
    /// The archive entry could not be loaded.
    ArchiveLoad(ArchiveLoadError),
    /// The archive entry's brain does not fit its config's layout.
    ArchiveLayout {
        /// Brain length the config's layout needs.
        expected: usize,
        /// Brain length stored.
        found: usize,
    },
}

impl fmt::Display for FreeLifeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArchiveLoad(err) => write!(f, "cannot load archive entry: {err:?}"),
            Self::ArchiveLayout { expected, found } => write!(
                f,
                "archive brain has {found} values, its config needs {expected}"
            ),
            Self::FoodKindStoredBrain => write!(
                f,
                "--food-kinds cannot start from a stored brain; use a fresh brain"
            ),
            Self::NoNourishingKind => write!(f, "the kernel drew no nourishing food kind"),
            Self::MealCounts => write!(f, "a cumulative meal count decreased"),
            Self::MissingSense(name) => write!(f, "brain layout has no '{name}' sense"),
            Self::SummaryIo(msg) => write!(f, "food-kind summary error: {msg}"),
            Self::NoGpu => write!(f, "no GPU adapter available"),
            Self::VisionStride(s) => write!(f, "vision stride must be 1, got {s}"),
            Self::AgentCount(n) => write!(f, "agent count must be in 1..={MAX_AGENTS}, got {n}"),
            Self::Ticks { ticks, batch } => write!(
                f,
                "ticks must be a non-zero multiple of the kernel batch size {batch}, got {ticks}"
            ),
            Self::UnknownLesion(name) => write!(f, "unknown lesion '{name}'"),
            Self::ReadbackLength {
                what,
                expected,
                found,
            } => write!(
                f,
                "{what} readback returned {found} values, expected {expected}"
            ),
            Self::LogIo(msg) => write!(f, "tick log error: {msg}"),
        }
    }
}

impl std::error::Error for FreeLifeError {}

/// Result of one seeded free life.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FreeLifeReport {
    pub agent_count: usize,
    pub ticks: u64,
    pub seed: u64,
    /// `"fresh"` or `"archive"`.
    pub brain_source: String,
    /// Entry id of the archive entry; `Some` only for the archive source.
    #[serde(default)]
    pub archive_entry_id: Option<i64>,
    /// Names of the lesioned parts.
    pub lesions: Vec<String>,
    pub kernel_batch_size: u32,
    /// Brain ticks run (one sample per agent each).
    pub brain_ticks: u64,
    pub agents: Vec<AgentMetrics>,
}

/// Parse a comma list of lesion names into a lesion set.
pub fn parse_lesions(list: &str) -> Result<TrunkLesions, FreeLifeError> {
    let mut lesions = TrunkLesions::NONE;
    if list.trim().is_empty() {
        return Ok(lesions);
    }
    for item in list.split(',') {
        match item.trim() {
            LESION_TURN_WEIGHTS => lesions.turn_weights = true,
            LESION_FORWARD_WEIGHTS => lesions.forward_weights = true,
            LESION_MEMORY_BLEND => lesions.memory_blend = true,
            LESION_CRITIC => lesions.critic = true,
            LESION_ALL => lesions = TrunkLesions::ALL,
            LESION_NONE => {}
            other => return Err(FreeLifeError::UnknownLesion(other.to_string())),
        }
    }
    Ok(lesions)
}

/// Names of the lesioned parts, in a fixed order.
pub fn lesion_names(lesions: TrunkLesions) -> Vec<&'static str> {
    [
        (lesions.turn_weights, LESION_TURN_WEIGHTS),
        (lesions.forward_weights, LESION_FORWARD_WEIGHTS),
        (lesions.memory_blend, LESION_MEMORY_BLEND),
        (lesions.critic, LESION_CRITIC),
    ]
    .into_iter()
    .filter_map(|(on, name)| on.then_some(name))
    .collect()
}

/// Birth brain and config chosen by the options.
struct Birth {
    brain_config: BrainConfig,
    stored_brain: Option<AgentBrainState>,
    archive_entry_id: Option<i64>,
}

/// Resolve the birth source, check its layout and the vision stride.
fn resolve_birth(config: &FullConfig, source: &BrainSource) -> Result<Birth, FreeLifeError> {
    let birth = match source {
        BrainSource::Fresh => Birth {
            brain_config: config.brain.clone(),
            stored_brain: None,
            archive_entry_id: None,
        },
        BrainSource::Archive { db_path, rank } => {
            let stored =
                load_archive_entry_read_only(db_path, *rank).map_err(FreeLifeError::ArchiveLoad)?;
            let expected = BrainLayout::from_config(&stored.config).brain_stride;
            let found = stored.brain.brain_state.len();
            if expected != found {
                return Err(FreeLifeError::ArchiveLayout { expected, found });
            }
            let entry_id = i64::try_from(stored.entry_id).map_err(|_| {
                FreeLifeError::ArchiveLoad(ArchiveLoadError::MalformedBlob {
                    entry_id: stored.entry_id,
                })
            })?;
            Birth {
                brain_config: stored.config,
                stored_brain: Some(stored.brain),
                archive_entry_id: Some(entry_id),
            }
        }
    };
    if birth.brain_config.vision_stride != 1 {
        return Err(FreeLifeError::VisionStride(
            birth.brain_config.vision_stride,
        ));
    }
    Ok(birth)
}

/// Build the world and create the kernel without uploading anything.
fn create_kernel(
    config: &FullConfig,
    brain_config: &BrainConfig,
    options: &FreeLifeOptions,
) -> (GpuKernel, WorldState) {
    let mut world_config = config.world.clone();
    world_config.seed = options.seed;
    let world = WorldState::new(world_config.clone());
    let kernel = GpuKernel::new(
        u32::try_from(options.agent_count).unwrap_or(u32::MAX),
        world.food_items.len(),
        brain_config,
        &world_config,
    );
    (kernel, world)
}

/// Upload the world and the agents to the kernel.
fn upload_kernel(
    kernel: &GpuKernel,
    world: &WorldState,
    brain_config: &BrainConfig,
    options: &FreeLifeOptions,
) {
    let biomes = world.biome_map.grid_as_u32();
    let food_pos: Vec<(f32, f32, f32)> = world
        .food_items
        .iter()
        .map(|f| (f.position.x, f.position.y, f.position.z))
        .collect();
    let food_consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
    let food_timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
    kernel.upload_world(
        &world.terrain.heights,
        &biomes,
        &food_pos,
        &food_consumed,
        &food_timers,
    );
    let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = (0..options.agent_count)
        .map(|i| {
            let id = u32::try_from(i).unwrap_or(u32::MAX);
            let agent = Agent::new(id, world.safe_spawn_position(), id, brain_config.clone(), 0);
            (
                agent.body.body.position,
                agent.body.body.internal.max_energy,
                agent.body.body.internal.max_integrity,
                brain_config.memory_capacity,
                brain_config.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agent_data);
}

/// Convert a non-negative kernel counter held in an `f32` to a count.
fn counter_value(v: f32) -> u64 {
    if v.is_finite() && v > 0.0 {
        v.round() as u64
    } else {
        0
    }
}

/// Build one agent's sample from its physics row and the brain-tick probe.
fn build_sample(brain_tick: u64, phys: &[f32], probe: &BrainTickProbe) -> FreeLifeSample {
    FreeLifeSample {
        brain_tick,
        alive: phys[P_ALIVE] > 0.5,
        meals_total: counter_value(phys[P_FOOD_COUNT]),
        deaths_total: counter_value(phys[P_DEATH_COUNT]),
        on_hazard: phys[P_IN_DANGER_BIOME] > 0.5,
        motor_turn: phys[P_MOTOR_TURN_OUT],
        food_distance: phys[P_NEAREST_FOOD_DISTANCE],
        food_bearing: phys[P_NEAREST_FOOD_BEARING],
        hazard_distance: phys[P_NEAREST_DANGER_DISTANCE],
        hazard_bearing: phys[P_NEAREST_DANGER_BEARING],
        encoding: probe.encoding.to_vec(),
        pathway_inputs: probe.pathway_inputs,
        reward: probe.reward,
        value: probe.value,
        td_error: probe.td_error,
        forward_noise: probe.forward_noise,
        turn_noise: probe.turn_noise,
        turn_innovation: probe.turn_innovation,
        encoding_turn_output: probe.encoding_turn_output,
        scent_turn_output: probe.scent_turn_output,
        vision_turn_output: probe.vision_turn_output,
        encoding_forward_output: probe.encoding_forward_output,
    }
}

/// Feed one brain tick's samples to the accumulators and the optional log.
fn feed_tick(
    brain_tick: u64,
    probes: &[BrainTickProbe],
    state: &[f32],
    accumulators: &mut [AgentAccumulator],
    writer: &mut Option<TickLogWriter>,
) -> Result<(), FreeLifeError> {
    let agents = accumulators.len();
    if probes.len() != agents {
        return Err(FreeLifeError::ReadbackLength {
            what: "brain tick probe",
            expected: agents,
            found: probes.len(),
        });
    }
    let needed = agents.saturating_mul(PHYS_STRIDE);
    if state.len() < needed {
        return Err(FreeLifeError::ReadbackLength {
            what: "physics state",
            expected: needed,
            found: state.len(),
        });
    }
    for (i, (acc, probe)) in accumulators.iter_mut().zip(probes).enumerate() {
        let phys = &state[i * PHYS_STRIDE..(i + 1) * PHYS_STRIDE];
        let sample = build_sample(brain_tick, phys, probe);
        acc.push(&sample);
        if let Some(w) = writer.as_mut() {
            let agent = u32::try_from(i).unwrap_or(u32::MAX);
            w.write_sample(agent, &sample)
                .map_err(|e| FreeLifeError::LogIo(e.to_string()))?;
        }
    }
    Ok(())
}

/// Write the birth brain and heritable config to every agent.
fn seed_birth_brains(
    kernel: &GpuKernel,
    birth: &Birth,
    agent_count: usize,
    rng: &mut rand::rngs::SmallRng,
) {
    let state = match &birth.stored_brain {
        Some(brain) => brain.clone(),
        None => fresh_brain_state(&birth.brain_config, rng),
    };
    for i in 0..agent_count {
        let idx = u32::try_from(i).unwrap_or(u32::MAX);
        kernel.write_agent_state(idx, &state);
        kernel.write_agent_heritable_config(idx, &birth.brain_config);
    }
}

/// Dispatch every batch and feed each brain tick's samples.
fn run_batches(
    kernel: &mut GpuKernel,
    batches: u64,
    batch_size: u32,
    accumulators: &mut [AgentAccumulator],
    writer: &mut Option<TickLogWriter>,
    mut tracker: Option<&mut FoodKindTracker>,
) -> Result<(), FreeLifeError> {
    let mut probes: Vec<BrainTickProbe> = Vec::new();
    for batch in 0..batches {
        // `dispatch_batch` always reports success.
        let _dispatched = kernel.dispatch_batch(batch * u64::from(batch_size), batch_size);
        // The probe read borrows the kernel shared and the physics read borrows
        // it mutably, so the probe read goes first.
        probes.clear();
        kernel.read_brain_tick_probes_blocking(&mut probes);
        let state = kernel.read_full_state_blocking();
        feed_tick(batch, &probes, state, accumulators, writer)?;
        if let Some(t) = tracker.as_deref_mut() {
            t.observe(kernel, batch, &probes)?;
        }
    }
    Ok(())
}

/// Food-kind run configuration; every field is opt-in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoodKindRun {
    /// Hold the turn weights for the whole run.
    pub freeze_policy: bool,
    /// File that receives the run summary as JSON.
    pub summary_path: Option<PathBuf>,
}

/// Name of the first odour sense in the brain layout.
const SENSE_SMELL: &str = "smell";
/// Name of the second odour sense in the brain layout.
const SENSE_SECOND_ODOUR: &str = "second_odour";
/// Index of the left channel of an odour pair.
const CHANNEL_LEFT: usize = 0;
/// Index of the right channel of an odour pair.
const CHANNEL_RIGHT: usize = 1;
/// Multiplier that spreads agent indices over the shuffle seed space.
const SHUFFLE_SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Shuffle seed of the readout of one agent.
fn readout_shuffle_seed(run_seed: u64, agent_index: usize) -> u64 {
    let index = u64::try_from(agent_index).unwrap_or(u64::MAX);
    run_seed ^ index.wrapping_add(1).wrapping_mul(SHUFFLE_SEED_STRIDE)
}

/// Narrow a kernel counter to `u32`, saturating.
fn counter_u32(v: f32) -> u32 {
    u32::try_from(counter_value(v)).unwrap_or(u32::MAX)
}

/// Sense blocks with offsets rebased onto absolute `brain_state` indices.
/// Layout offsets are relative to the fixed-tail base, which moves with the
/// feature count.
fn absolute_sense_blocks(layout: &BrainLayout) -> Vec<SenseBlockOffsets> {
    let shift = fixed_tail_base(layout.brain_stride) - O_PREDICTOR_CONTEXT_WEIGHT;
    let add = |o: Option<usize>| o.map(|v| v + shift);
    layout
        .sense_blocks()
        .iter()
        .map(|b| SenseBlockOffsets {
            mean: b.mean + shift,
            covariance: b.covariance + shift,
            whitening: add(b.whitening),
            whitened_input: b.whitened_input + shift,
            turn_weights: b.turn_weights + shift,
            turn_trace: b.turn_trace + shift,
            forward_weights: add(b.forward_weights),
            forward_trace: add(b.forward_trace),
            critic_weights: add(b.critic_weights),
            critic_trace: add(b.critic_trace),
            ..*b
        })
        .collect()
}

/// Turn weights of every sense, per sense then per channel.
fn turn_weights_of(brain: &[f32], blocks: &[SenseBlockOffsets]) -> Vec<Vec<f32>> {
    blocks
        .iter()
        .map(|b| brain[b.turn_weights..b.turn_weights + b.channels].to_vec())
        .collect()
}

/// Per-agent running measurements of a food-kind run.
struct AgentKindTrack {
    turns: KindTurnAccumulator,
    credit: PairCreditAccumulator,
    samples: Vec<ReadoutSample>,
    cumulative: [(u32, u32); QUARTER_COUNT],
}

/// Reads snapshots (including read-only per-agent brain-state copies) and
/// logs only; never writes kernel state.
struct FoodKindTracker {
    nourishing: FoodKind,
    harmful: FoodKind,
    blocks: Vec<SenseBlockOffsets>,
    smell: usize,
    second: usize,
    boundaries: [u64; QUARTER_COUNT],
    agents: Vec<AgentKindTrack>,
}

impl FoodKindTracker {
    fn new(kernel: &GpuKernel, agent_count: usize, batches: u64) -> Result<Self, FreeLifeError> {
        let nourishing = kernel
            .nourishing_food_kind()
            .ok_or(FreeLifeError::NoNourishingKind)?;
        let harmful = match nourishing {
            FoodKind::First => FoodKind::Second,
            FoodKind::Second => FoodKind::First,
        };
        let layout = kernel.brain_layout();
        let find = |name: &'static str| {
            layout
                .senses()
                .iter()
                .position(|s| s.name == name)
                .ok_or(FreeLifeError::MissingSense(name))
        };
        let smell = find(SENSE_SMELL)?;
        let second = find(SENSE_SECOND_ODOUR)?;
        let quarters = u64::try_from(QUARTER_COUNT).unwrap_or(1);
        let mut boundaries = [0u64; QUARTER_COUNT];
        for (q, b) in boundaries.iter_mut().enumerate() {
            let step = u64::try_from(q + 1).unwrap_or(1);
            *b = batches.saturating_mul(step) / quarters;
        }
        let agents = (0..agent_count)
            .map(|_| AgentKindTrack {
                turns: KindTurnAccumulator::new(batches),
                credit: PairCreditAccumulator::default(),
                samples: Vec::new(),
                cumulative: [(0, 0); QUARTER_COUNT],
            })
            .collect();
        Ok(Self {
            nourishing,
            harmful,
            blocks: absolute_sense_blocks(layout),
            smell,
            second,
            boundaries,
            agents,
        })
    }

    /// Record one brain tick from the physics snapshot, food snapshot, brain log
    /// and a read-only copy of each brain state (the only published source of
    /// the whitened sense inputs).
    fn observe(
        &mut self,
        kernel: &mut GpuKernel,
        brain_tick: u64,
        probes: &[BrainTickProbe],
    ) -> Result<(), FreeLifeError> {
        // The snapshot staged by the dispatch is already complete.
        let _collected = kernel.try_collect_state_snapshot();
        let state = kernel.cached_state();
        let needed = self.agents.len().saturating_mul(PHYS_STRIDE);
        if state.len() < needed {
            return Err(FreeLifeError::ReadbackLength {
                what: "physics snapshot",
                expected: needed,
                found: state.len(),
            });
        }
        let food = kernel
            .cached_food_state()
            .ok_or(FreeLifeError::ReadbackLength {
                what: "food snapshot",
                expected: 1,
                found: 0,
            })?;
        for (i, (track, probe)) in self.agents.iter_mut().zip(probes).enumerate() {
            let phys = &state[i * PHYS_STRIDE..(i + 1) * PHYS_STRIDE];
            let position = [phys[P_POS_X], phys[P_POS_Y], phys[P_POS_Z]];
            let facing = [phys[P_FACING_X], phys[P_FACING_Z]];
            let near = nearest_food_of_kind(position, facing, food, self.nourishing, SCENT_RANGE);
            let far = nearest_food_of_kind(position, facing, food, self.harmful, SCENT_RANGE);
            track.turns.record(
                brain_tick,
                phys[P_MOTOR_TURN_OUT],
                near.map(|n| n.bearing),
                far.map(|n| n.bearing),
            );
            let index = u32::try_from(i).unwrap_or(u32::MAX);
            let brain = kernel.read_agent_state(index).brain_state;
            for (kind, sense) in [
                (FoodKind::First, self.smell),
                (FoodKind::Second, self.second),
            ] {
                let at = self.blocks[sense].whitened_input;
                track.credit.record(
                    kind,
                    probe.td_error,
                    probe.turn_innovation,
                    brain[at + CHANNEL_LEFT],
                    brain[at + CHANNEL_RIGHT],
                );
            }
            let label = match (near, far) {
                (Some(n), Some(f)) => Some(n.distance <= f.distance),
                (Some(_), None) => Some(true),
                (None, Some(_)) => Some(false),
                (None, None) => None,
            };
            if let Some(label) = label {
                let features: Vec<f32> = self
                    .blocks
                    .iter()
                    .flat_map(|b| brain[b.whitened_input..b.whitened_input + b.channels].to_vec())
                    .collect();
                track.samples.push(ReadoutSample { features, label });
            }
            for (q, boundary) in self.boundaries.iter().enumerate() {
                if brain_tick + 1 == *boundary {
                    track.cumulative[q] = (
                        counter_u32(phys[P_FOOD_COUNT]),
                        counter_u32(phys[P_HARMFUL_MEAL_COUNT]),
                    );
                }
            }
        }
        Ok(())
    }
}

/// Result of a food-kind free life.
#[derive(Clone, Debug)]
pub struct FoodKindOutcome {
    /// The ordinary free-life report.
    pub report: FreeLifeReport,
    /// The run summary, as written to the summary file.
    pub summary: RunFoodKindSummary,
    /// Sense names in layout order.
    pub sense_names: Vec<&'static str>,
    /// Birth turn weights, per agent, sense and channel.
    pub birth_turn_weights: Vec<Vec<Vec<f32>>>,
    /// Final turn weights, per agent, sense and channel.
    pub final_turn_weights: Vec<Vec<Vec<f32>>>,
}

/// Run one seeded free life with the food-kind scenario.
pub fn run_free_life_food_kind(
    config: &FullConfig,
    options: &FreeLifeOptions,
    food_kind: &FoodKindRun,
) -> Result<FoodKindOutcome, FreeLifeError> {
    if options.brain_source != BrainSource::Fresh {
        return Err(FreeLifeError::FoodKindStoredBrain);
    }
    let mut kind_config = config.clone();
    kind_config.world.food_kinds_enabled = true;
    kind_config.brain.second_odour_sense_enabled = true;
    let (_report, outcome) = run_free_life_inner(&kind_config, options, Some(food_kind))?;
    let outcome = outcome.ok_or(FreeLifeError::NoNourishingKind)?;
    if let Some(path) = &food_kind.summary_path {
        let json = serde_json::to_string_pretty(&outcome.summary)
            .map_err(|e| FreeLifeError::SummaryIo(e.to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| FreeLifeError::SummaryIo(format!("{}: {e}", path.display())))?;
    }
    Ok(outcome)
}

/// Run one seeded free life and report per-agent metrics.
pub fn run_free_life(
    config: &FullConfig,
    options: &FreeLifeOptions,
) -> Result<FreeLifeReport, FreeLifeError> {
    run_free_life_inner(config, options, None).map(|(report, _)| report)
}

fn run_free_life_inner(
    config: &FullConfig,
    options: &FreeLifeOptions,
    food_kind: Option<&FoodKindRun>,
) -> Result<(FreeLifeReport, Option<FoodKindOutcome>), FreeLifeError> {
    if !GpuKernel::is_available() {
        return Err(FreeLifeError::NoGpu);
    }
    if options.agent_count == 0 || options.agent_count > MAX_AGENTS {
        return Err(FreeLifeError::AgentCount(options.agent_count));
    }
    let birth = resolve_birth(config, &options.brain_source)?;
    let (mut kernel, world) = create_kernel(config, &birth.brain_config, options);
    let batch_size = kernel.kernel_batch_size();
    if options.ticks == 0 || options.ticks % u64::from(batch_size) != 0 {
        return Err(FreeLifeError::Ticks {
            ticks: options.ticks,
            batch: batch_size,
        });
    }
    upload_kernel(&kernel, &world, &birth.brain_config, options);
    let batches = options.ticks / u64::from(batch_size);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(options.seed);
    kernel.reset_agents_seeded(&birth.brain_config, rng.random());
    seed_birth_brains(&kernel, &birth, options.agent_count, &mut rng);
    kernel.set_trunk_lesions(options.lesions);
    kernel.set_brain_tick_probe(true);

    let expected_rows = u64::try_from(options.agent_count)
        .ok()
        .and_then(|n| n.checked_mul(batches))
        .unwrap_or(u64::MAX);
    let mut writer = match &options.log_path {
        Some(path) => Some(
            TickLogWriter::create(path, expected_rows)
                .map_err(|e| FreeLifeError::LogIo(e.to_string()))?,
        ),
        None => None,
    };
    let metrics_config = MetricsConfig {
        trace_decay: f64::from(TD_TRACE_DECAY),
        total_brain_ticks: batches,
    };
    let mut accumulators: Vec<AgentAccumulator> = (0..options.agent_count)
        .map(|_| AgentAccumulator::new(metrics_config))
        .collect();
    let (mut tracker, birth_turn_weights) = match food_kind {
        Some(run) => prepare_food_kind(&mut kernel, run, options.agent_count, batches)?,
        None => (None, Vec::new()),
    };
    run_batches(
        &mut kernel,
        batches,
        batch_size,
        &mut accumulators,
        &mut writer,
        tracker.as_mut(),
    )?;
    if let Some(w) = writer {
        w.finish()
            .map_err(|e| FreeLifeError::LogIo(e.to_string()))?;
    }
    let source_label = if birth.archive_entry_id.is_some() {
        SOURCE_ARCHIVE
    } else {
        SOURCE_FRESH
    };
    let report = FreeLifeReport {
        agent_count: options.agent_count,
        ticks: options.ticks,
        seed: options.seed,
        brain_source: source_label.to_string(),
        archive_entry_id: birth.archive_entry_id,
        lesions: lesion_names(options.lesions)
            .into_iter()
            .map(str::to_string)
            .collect(),
        kernel_batch_size: batch_size,
        brain_ticks: batches,
        agents: accumulators
            .into_iter()
            .map(AgentAccumulator::finish)
            .collect(),
    };
    let (Some(run), Some(tracker)) = (food_kind, tracker) else {
        return Ok((report, None));
    };
    let outcome = build_food_kind_outcome(
        &mut kernel,
        &tracker,
        options,
        run,
        report.clone(),
        birth_turn_weights,
    )?;
    Ok((report, Some(outcome)))
}

/// Food-kind tracker and birth turn weights; sets the freeze flag for the
/// frozen arm before the first dispatch.
fn prepare_food_kind(
    kernel: &mut GpuKernel,
    run: &FoodKindRun,
    agent_count: usize,
    batches: u64,
) -> Result<(Option<FoodKindTracker>, Vec<Vec<Vec<f32>>>), FreeLifeError> {
    let tracker = FoodKindTracker::new(kernel, agent_count, batches)?;
    let birth = read_turn_weights(kernel, agent_count);
    if run.freeze_policy {
        kernel.set_freeze_steering_weights(true);
    }
    Ok((Some(tracker), birth))
}

/// Every agent's turn weights, per sense then per channel.
fn read_turn_weights(kernel: &GpuKernel, agent_count: usize) -> Vec<Vec<Vec<f32>>> {
    let blocks = absolute_sense_blocks(kernel.brain_layout());
    (0..agent_count)
        .map(|i| {
            let index = u32::try_from(i).unwrap_or(u32::MAX);
            turn_weights_of(&kernel.read_agent_state(index).brain_state, &blocks)
        })
        .collect()
}

/// End-of-run weight read, readouts and summary of a food-kind run.
fn build_food_kind_outcome(
    kernel: &mut GpuKernel,
    tracker: &FoodKindTracker,
    options: &FreeLifeOptions,
    run: &FoodKindRun,
    report: FreeLifeReport,
    birth_turn_weights: Vec<Vec<Vec<f32>>>,
) -> Result<FoodKindOutcome, FreeLifeError> {
    let final_turn_weights = read_turn_weights(kernel, options.agent_count);
    let physics = kernel.read_full_state_blocking();
    let mut agents = Vec::new();
    for (i, track) in tracker.agents.iter().enumerate() {
        let weights = &final_turn_weights[i];
        let diff = |sense: usize, kind: FoodKind| {
            signed_weight_difference(
                weights[sense][CHANNEL_LEFT],
                weights[sense][CHANNEL_RIGHT],
                kind,
                tracker.nourishing,
            )
        };
        let quarters =
            KindMealQuarters::from_cumulative(track.cumulative).ok_or(FreeLifeError::MealCounts)?;
        let means = track.turns.finish();
        let score = fit_kind_readout(&track.samples, readout_shuffle_seed(options.seed, i));
        agents.push(AgentFoodKindSummary {
            turn_toward_nourishing_whole: means.turn_toward_nourishing_whole,
            turn_toward_nourishing_late: means.turn_toward_nourishing_late,
            turn_away_harmful_whole: means.turn_away_harmful_whole,
            turn_away_harmful_late: means.turn_away_harmful_late,
            nourishing_meals_by_quarter: quarters.nourishing_by_quarter,
            harmful_meals_by_quarter: quarters.harmful_by_quarter,
            deaths: counter_u32(physics[i * PHYS_STRIDE + P_DEATH_COUNT]),
            readout_balanced_accuracy: score.as_ref().map(|s| s.balanced_accuracy),
            readout_shuffled_accuracy: score.as_ref().map(|s| s.shuffled_balanced_accuracy),
            signed_credit_pair_first: signed_for_nourishing(
                track.credit.credit(FoodKind::First),
                FoodKind::First,
                tracker.nourishing,
            ),
            signed_credit_pair_second: signed_for_nourishing(
                track.credit.credit(FoodKind::Second),
                FoodKind::Second,
                tracker.nourishing,
            ),
            signed_weight_difference_pair_first: diff(tracker.smell, FoodKind::First),
            signed_weight_difference_pair_second: diff(tracker.second, FoodKind::Second),
        });
    }
    let summary = RunFoodKindSummary {
        seed: options.seed,
        arm: if run.freeze_policy {
            Arm::Frozen
        } else {
            Arm::Learning
        },
        ticks: options.ticks,
        agent_count: options.agent_count,
        fresh_brains: true,
        nourishing_kind: tracker.nourishing,
        agents,
    };
    let sense_names = kernel
        .brain_layout()
        .senses()
        .iter()
        .map(|s| s.name)
        .collect();
    let outcome = FoodKindOutcome {
        report,
        summary,
        sense_names,
        birth_turn_weights,
        final_turn_weights,
    };
    Ok(outcome)
}
