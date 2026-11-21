//! Headless (no-window) evolution loop and tree dump utilities.
//!
//! Uses GpuKernel for all simulation — physics, brain, food, and
//! death/respawn run entirely on GPU via fused kernel dispatch.

use std::fmt;
use std::path::PathBuf;
use std::time::Instant;

use log::info;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use xagent_shared::{BrainConfig, EvolutionMode, FullConfig, SUPERVISOR_SAMPLE_TICKS};

use crate::run_summary::{RunSummaryAccumulator, RunSummaryError};
use crate::supervisor_checkpoint::CheckpointError;
use crate::supervisor_run::{
    load_resume_point, ResumePoint, SampleReport, SupervisorFinish, SupervisorRun,
    SupervisorRunError,
};
use crate::supervisor_store::{
    load_archive_entry_read_only, ArchiveLoadError, StoreError, SupervisorStore,
};

use xagent_brain::buffers::{
    fixed_tail_base, SenseBlockOffsets, P_FACING_X, P_FACING_Z, P_HARMFUL_MEAL_COUNT,
};
use xagent_brain::buffers::{
    BrainLayout, BrainTickProbe, TrunkLesions, ENCODED_DIMENSION, O_ACTION_FORWARD_WEIGHTS,
    O_ACTION_TURN_WEIGHTS, O_PREDICTOR_CONTEXT_WEIGHT, PHYS_STRIDE, PREDICTOR_DIMENSION, P_ALIVE,
    P_APPROACH_SENSE_RANGE_TICKS, P_APPROACH_TURNS_TOWARD, P_AVOIDANCE_SENSE_RANGE_TICKS,
    P_AVOIDANCE_TURNS_OPPOSING, P_AVOIDANCE_TURN_AWAY, P_DANGER_PATH_LENGTH, P_DEATH_COUNT,
    P_DISTANCE_TRAVELED, P_ENERGY_SPENT, P_FOOD_COUNT, P_HAZARD_ENTRIES, P_IN_DANGER_BIOME,
    P_MOTOR_TURN_OUT, P_NEAREST_DANGER_BEARING, P_NEAREST_DANGER_DISTANCE, P_NEAREST_FOOD_BEARING,
    P_NEAREST_FOOD_DISTANCE, P_POS_X, P_POS_Y, P_POS_Z, P_TICKS_ALIVE, TD_TRACE_DECAY,
};
use xagent_brain::kind_readout::{fit_kind_readout, ReadoutSample};
use xagent_brain::{AgentBrainState, GpuKernel};
use xagent_shared::{FoodKind, SCENT_RANGE};

use crate::agent::{fresh_brain_state, steering_population, Agent, MAX_AGENTS};
use crate::food_kind_metrics::{
    nearest_food_of_kind, signed_for_nourishing, signed_weight_difference, AgentFoodKindSummary,
    Arm, KindMealQuarters, KindTurnAccumulator, PairCreditAccumulator, RunFoodKindSummary,
    QUARTER_COUNT,
};
use crate::free_life::metrics::{AgentAccumulator, AgentMetrics, FreeLifeSample, MetricsConfig};
use crate::free_life::tick_log::TickLogWriter;
use crate::governor::{
    compute_approach_intent_fraction, compute_avoidance_intent_fraction,
    compute_danger_dwell_fraction, load_champion_read_only, AdvanceResult, ChampionCapture,
    ChampionLoadError, Governor,
};
use crate::world::WorldState;

/// Chunk size for dispatch_batch calls in the standard headless run.
/// Between chunks we read back positions for heatmap recording.
const HEATMAP_INTERVAL: u32 = 100;

/// Number of position samples taken per generation in the validation dispatch loop.
/// 4 samples (one per quarter of the tick budget) satisfies the within-life tracker's
/// quarter-boundary requirements while keeping the GPU readback count low.  The
/// actual dispatch batch size is `tick_budget / VALIDATION_HEATMAP_SAMPLES`, clamped
/// to at least `HEATMAP_INTERVAL` (100 ticks) so it is always a valid batch.
const VALIDATION_HEATMAP_SAMPLES: u32 = 4;

fn repeated_config(seed: &BrainConfig, count: usize) -> Vec<BrainConfig> {
    vec![seed.clone(); count]
}

/// Upload one shared steering genome per `group_size` agents. `template` is
/// the champion when a generation inherits one, and a fresh birth brain when
/// it does not.
fn write_steering_population(
    kernel: &GpuKernel,
    agents: &[Agent],
    template: &AgentBrainState,
    group_size: usize,
    strength: f32,
    rng: &mut impl Rng,
) {
    let genomes = steering_population(template, agents.len(), group_size, strength, rng);
    for (agent, genome) in agents.iter().zip(genomes.iter()) {
        kernel.write_agent_state(agent.brain_idx, genome);
    }
}

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
    /// The agent-tick budget is not a whole number of generations.
    InvalidAgentTickBudget,
}

impl fmt::Display for HeadlessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpoint(e) => write!(f, "{e}"),
            Self::Store(e) => write!(f, "{e}"),
            Self::SupervisorRun(e) => write!(f, "{e}"),
            Self::Summary(e) => write!(f, "run summary could not be written: {e}"),
            Self::InvalidAgentTickBudget => write!(
                f,
                "the agent-tick budget must be a positive whole multiple of population times tick budget"
            ),
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

/// Run the headless evolution loop: no window, no rendering, max speed.
/// Picks the driver from `governor.evolution_mode`; errors are printed.
pub fn run_headless(config: FullConfig, db_path: &str, resume: bool, _has_gpu: bool) {
    if let Err(err) =
        run_headless_with_options(config, db_path, resume, &HeadlessOptions::default())
    {
        eprintln!("{err}");
    }
}

/// Run the headless driver the config's evolution mode selects.
///
/// # Errors
///
/// Returns a [`HeadlessError`] for an unsupported option combination or a
/// failed supervisor run, store or summary write.
pub fn run_headless_with_options(
    config: FullConfig,
    db_path: &str,
    resume: bool,
    options: &HeadlessOptions,
) -> Result<(), HeadlessError> {
    match config.governor.evolution_mode {
        EvolutionMode::Generations => {
            let Some(path) = options.run_summary.as_ref() else {
                run_generations(config, db_path, resume, None)?;
                return Ok(());
            };
            let mut acc = RunSummaryAccumulator::new(
                EvolutionMode::Generations,
                None,
                config.world.seed,
                config.governor.population_size,
                config.world.food_kinds_enabled,
            );
            let ticks = run_generations(config, db_path, resume, Some(&mut acc))?;
            let summary = acc.finish(ticks).map_err(summary_error)?;
            let json = serde_json::to_string_pretty(&summary)?;
            std::fs::write(path, json)?;
            Ok(())
        }
        EvolutionMode::Supervisor => {
            if resume {
                return resume_supervisor(db_path, &config, options);
            }
            run_supervisor(config, db_path, options)
        }
    }
}

/// One sample period as the kernel's batch tick count, checked at compile time.
const SAMPLE_BATCH_TICKS: u32 = {
    assert!(SUPERVISOR_SAMPLE_TICKS <= u32::MAX as u64);
    SUPERVISOR_SAMPLE_TICKS as u32
};

/// Run the supervisor loop on one kernel and write the summary when asked.
fn run_supervisor(
    config: FullConfig,
    db_path: &str,
    options: &HeadlessOptions,
) -> Result<(), HeadlessError> {
    run_supervisor_with_recording(config, db_path, options, true)
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
    println!("Supervisor run | Population: {population}");
    let mut next_progress = 1_u64;
    drive_supervisor(run, &mut kernel, options, batch, &mut next_progress)
}

/// Run the loop to completion (or the stop option) and write the summary.
fn drive_supervisor(
    mut run: SupervisorRun,
    kernel: &mut GpuKernel,
    options: &HeadlessOptions,
    batch: u32,
    next_progress: &mut u64,
) -> Result<(), HeadlessError> {
    while !run.is_finished() {
        let Some(report) = supervisor_step(&mut run, kernel, batch, options)? else {
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
    )
}

/// Dispatch one sample period, read it back and hand it to the run. Returns
/// `None` when the stop option ended the run right after a checkpoint.
fn supervisor_step(
    run: &mut SupervisorRun,
    kernel: &mut GpuKernel,
    batch: u32,
    options: &HeadlessOptions,
) -> Result<Option<SampleReport>, HeadlessError> {
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
        return Ok(Some(run.on_sample(
            kernel,
            sample_tick,
            &physics,
            food.as_deref(),
        )?));
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
    Ok(Some(run.on_sample(
        kernel,
        sample_tick,
        &physics,
        food.as_deref(),
    )?))
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

/// Convert a run-summary rejection into a [`HeadlessError`].
fn summary_error(err: RunSummaryError) -> HeadlessError {
    HeadlessError::Summary(std::io::Error::other(err.to_string()))
}

/// Convert an agent-tick budget into the `max_generations` that runs it.
///
/// The loop checks `evolution_complete()` at the top of each generation,
/// which is `max > 0 && generation + 1 >= max`. A fresh governor starts at
/// `generation = 0`, and `advance` breeds (increments `generation`) only when
/// it does not finish. So generation indices `0..max-1` run, which is
/// `max - 1` generations. With `q = budget / (population * tick_budget)`
/// generations wanted, `max = q + 1`. Resumed runs are out of scope.
///
/// # Errors
///
/// [`HeadlessError::InvalidAgentTickBudget`] when an input is zero, the
/// budget is not a multiple of `population * tick_budget`, or on overflow.
pub fn max_generations_for_agent_ticks(
    budget: u64,
    population: usize,
    tick_budget: u64,
) -> Result<u64, HeadlessError> {
    let population =
        u64::try_from(population).map_err(|_| HeadlessError::InvalidAgentTickBudget)?;
    let product = population
        .checked_mul(tick_budget)
        .ok_or(HeadlessError::InvalidAgentTickBudget)?;
    if budget == 0 || product == 0 || budget % product != 0 {
        return Err(HeadlessError::InvalidAgentTickBudget);
    }
    (budget / product)
        .checked_add(1)
        .ok_or(HeadlessError::InvalidAgentTickBudget)
}

/// The generation-mode loop: a governor, generations until complete or
/// interrupted. Returns the ticks of completed generations.
fn run_generations(
    config: FullConfig,
    db_path: &str,
    resume: bool,
    mut summary: Option<&mut RunSummaryAccumulator>,
) -> Result<u64, HeadlessError> {
    let mut completed_ticks: u64 = 0;
    info!("Running headless evolution");
    let world_json = serde_json::to_string(&config.world).unwrap_or_default();
    let start_time = Instant::now();

    let mut governor = if resume {
        println!("Resuming from {}", db_path);
        Governor::resume(db_path).expect("Failed to resume from database")
    } else {
        Governor::new_seeded(
            db_path,
            config.governor.clone(),
            &config.brain,
            &world_json,
            config.world.seed,
        )
        .expect("Failed to initialize governor database")
    };

    let seed_config = governor.current_config().unwrap_or(config.brain.clone());
    let mut simulation_ticks = governor.simulation_ticks();
    println!(
        "Population: {} | Tick budget: {} | Elitism: {} | Patience: {}",
        governor.config.population_size,
        governor.config.tick_budget,
        governor.config.elitism_count,
        governor.config.patience,
    );

    let mut current_configs = repeated_config(&seed_config, governor.config.population_size);

    // Brain state inherited from the stored champion of the node the current
    // generation was bred from. The first generation has none, so it starts
    // from one shared birth brain.
    let mut inherited_state: Option<AgentBrainState> = None;
    let mut inherited_mutation_strength: f32 = governor.config.mutation_strength;
    let repeats = governor.config.eval_repeats.max(1);

    // Create GpuKernel once — reused across generations via reset_agents().
    let pop_size = governor.config.population_size;
    let world = WorldState::new(config.world.clone());
    let food_count = world.food_items.len();
    let mut kernel = GpuKernel::new(pop_size as u32, food_count, &seed_config, &config.world);

    loop {
        if governor.evolution_complete() {
            println!(
                "\n✓ Evolution complete after {} generations",
                governor.generation
            );
            break;
        }

        // Initialize world and agents for this generation
        let world = WorldState::new(config.world.clone());
        let mut agents: Vec<Agent> = current_configs
            .iter()
            .enumerate()
            .map(|(i, cfg)| {
                let pos = world.safe_spawn_position();
                Agent::new(i as u32, pos, i as u32, cfg.clone(), 0)
            })
            .collect();

        // Upload world data
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

        // Upload agent physics state
        let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = agents
            .iter()
            .map(|a| {
                (
                    a.body.body.position,
                    a.body.body.internal.max_energy,
                    a.body.body.internal.max_integrity,
                    a.brain_config.memory_capacity,
                    a.brain_config.processing_slots,
                )
            })
            .collect();
        kernel.upload_agents(&agent_data);

        // Every random choice below comes from the run's stream, so a seeded
        // run reproduces.
        let mut rng = rand::rngs::SmallRng::seed_from_u64(governor.draw_generation_seed());

        // `reset_agents_seeded()` does two things:
        // 1. Writes the shader config uniform — population-wide brain
        //    tuning values (learning_rate, decay_rate, distress_exponent,
        //    metabolic_rate, integrity_scale) plus layout constants
        //    (DIM, feature_count, memory_cap, recall_k).
        // 2. Seeds per-agent brain_state with initial heritable values
        //    from this BrainConfig (habituation_sensitivity,
        //    max_curiosity_bonus, fatigue_recovery_sensitivity,
        //    fatigue_floor).
        // Per-agent physical fields (memory_capacity, processing_slots)
        // come from `upload_agents()` above. Any inherited or mutated
        // AgentBrainState written via `write_agent_state()` below
        // overrides the reset-seeded brain_state values.
        kernel.reset_agents_seeded(&current_configs[0], rng.random());

        let template = match &inherited_state {
            Some(state) => state.clone(),
            None => fresh_brain_state(&current_configs[0], &mut rng),
        };
        write_steering_population(
            &kernel,
            &agents,
            &template,
            repeats,
            inherited_mutation_strength,
            &mut rng,
        );

        // Patch per-agent heritable config values so each agent's
        // brain_state reflects its own BrainConfig genome (not just
        // config[0] from reset_agents or the champion's values).
        for (i, agent) in agents.iter().enumerate() {
            kernel.write_agent_heritable_config(agent.brain_idx, &current_configs[i]);
        }
        let birth_states: Vec<AgentBrainState> = agents
            .iter()
            .map(|agent| kernel.read_agent_state(agent.brain_idx))
            .collect();

        governor.gen_tick = 0;

        let generation_start_tick = completed_ticks;
        if let Some(acc) = summary.as_deref_mut() {
            acc.reset_all_baselines();
            acc.begin_season(generation_start_tick, kernel.nourishing_food_kind())
                .map_err(summary_error)?;
        }

        // Run generation in chunks
        let gen_start = Instant::now();
        let tick_budget = governor.config.tick_budget;
        let mut ticks_done: u64 = 0;

        while ticks_done < tick_budget {
            let remaining = (tick_budget - ticks_done).min(HEATMAP_INTERVAL as u64) as u32;
            kernel.dispatch_batch(ticks_done, remaining);
            ticks_done += remaining as u64;

            // Advance governor tick counter
            governor.advance_ticks(u64::from(remaining));

            // Drain async readback, then sample cached state for heatmap
            while !kernel.try_collect_state_snapshot() {
                std::thread::yield_now();
            }
            if let Some(acc) = summary.as_deref_mut() {
                acc.record_physics_sample(
                    generation_start_tick + ticks_done,
                    kernel.cached_state(),
                    kernel.cached_food_state(),
                )
                .map_err(summary_error)?;
            }
            let state = kernel.cached_state();
            for i in 0..agents.len() {
                let base = i * PHYS_STRIDE;
                let alive = state[base + P_ALIVE] > 0.5;
                if alive {
                    agents[i].body.body.position = glam::Vec3::new(
                        state[base + P_POS_X],
                        state[base + P_POS_Y],
                        state[base + P_POS_Z],
                    );
                    agents[i].record_heatmap(config.world.world_size);
                }
            }

            // Feed the governor's within-life tracker (it snapshots quarter
            // boundaries from gen_tick).
            let mut cumulative_food = 0_u64;
            let mut cumulative_alive = 0_u64;
            for i in 0..agents.len() {
                let base = i * PHYS_STRIDE;
                cumulative_food += state[base + P_FOOD_COUNT] as u64;
                cumulative_alive += state[base + P_TICKS_ALIVE] as u64;
            }
            governor.record_within_life_sample(cumulative_food, cumulative_alive);

            if governor.gen_tick % (governor.config.tick_budget / 10).max(1) == 0 {
                let pct =
                    (governor.gen_tick as f32 / governor.config.tick_budget as f32 * 100.0) as u32;
                print!("\rGen {} [{:>3}%]", governor.generation, pct);
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
        }

        // Extract fitness stats from final state
        let state = kernel.cached_state();
        let mut final_food = 0_u64;
        let mut final_alive = 0_u64;
        for i in 0..agents.len() {
            let base = i * PHYS_STRIDE;
            agents[i].food_consumed = state[base + P_FOOD_COUNT] as u32;
            agents[i].total_ticks_alive = state[base + P_TICKS_ALIVE] as u64;
            agents[i].distance_traveled = state[base + P_DISTANCE_TRAVELED];
            agents[i].energy_spent = state[base + P_ENERGY_SPENT];
            agents[i].danger_path_length = state[base + P_DANGER_PATH_LENGTH];
            agents[i].avoidance_sense_range_ticks = state[base + P_AVOIDANCE_SENSE_RANGE_TICKS];
            agents[i].avoidance_turns_opposing = state[base + P_AVOIDANCE_TURNS_OPPOSING];
            agents[i].avoidance_turn_away = state[base + P_AVOIDANCE_TURN_AWAY];
            agents[i].hazard_entries = state[base + P_HAZARD_ENTRIES];
            agents[i].approach_sense_range_ticks = state[base + P_APPROACH_SENSE_RANGE_TICKS];
            agents[i].approach_turns_toward = state[base + P_APPROACH_TURNS_TOWARD];
            agents[i].death_count = state[base + P_DEATH_COUNT] as u32;
            final_food += u64::from(agents[i].food_consumed);
            final_alive += agents[i].total_ticks_alive;
        }
        // Final sample fills any quarter the chunk loop did not land on.
        governor.record_within_life_sample(final_food, final_alive);
        let (first_quarter_rate, last_quarter_rate) = governor.within_life_rates();

        println!();
        let gen_elapsed = gen_start.elapsed();

        let fitness = governor.evaluate(&agents);

        // Learning metrics describe the top-ranked agent's brain.
        let best_idx = fitness.first().map(|f| f.agent_index).unwrap_or(0);
        let best_state = agents
            .get(best_idx)
            .map(|a| kernel.read_agent_state(a.brain_idx));
        // The weight-norm layout must come from the top agent's own config —
        // configs are per-agent, and a mismatched layout would misplace the
        // tail offsets into its brain_state.
        let best_config = current_configs.get(best_idx).unwrap_or(&current_configs[0]);
        log_learning_metrics(
            &agents,
            best_state.as_ref(),
            best_config,
            first_quarter_rate,
            last_quarter_rate,
        );
        governor.log_generation(&fitness);
        println!(
            "  Time: {:.1}s | {:.0} ticks/sec",
            gen_elapsed.as_secs_f64(),
            governor.config.tick_budget as f64 / gen_elapsed.as_secs_f64(),
        );

        governor.update_wall_time(start_time.elapsed().as_secs_f64());
        completed_ticks = completed_ticks.saturating_add(tick_budget);
        simulation_ticks = simulation_ticks.saturating_add(governor.config.tick_budget);
        governor.update_simulation_ticks(simulation_ticks);

        match governor.advance(&fitness) {
            AdvanceResult::Continue {
                configs,
                messages,
                mutation_strength,
                champion_capture,
                inherit_from_node,
            } => {
                for msg in &messages {
                    println!("{}", msg);
                }
                inherited_state = resolve_inherited_brain(
                    &governor,
                    &birth_states,
                    champion_capture,
                    inherit_from_node,
                );
                current_configs = configs;
                inherited_mutation_strength = mutation_strength;
            }
            AdvanceResult::Finished { messages } => {
                for msg in &messages {
                    println!("{}", msg);
                }
                break;
            }
        }
    }

    let total_time = start_time.elapsed();
    println!(
        "\nTotal wall time: {:.1}s | {} generations",
        total_time.as_secs_f64(),
        governor.generation,
    );
    Ok(completed_ticks)
}

/// Store the accepted node's champion brain and return the brain the next
/// generation inherits: the stored champion of the node its configs were bred
/// from. The champion's brain is the one it was born with (`birth_states`,
/// indexed like the agents), the genome its fitness was earned with; what it
/// learned in its life is not passed on. Mirrors the live sandbox handoff, so
/// a node is always evaluated with the brain lineage its config came from.
fn resolve_inherited_brain(
    governor: &Governor,
    birth_states: &[AgentBrainState],
    champion_capture: Option<ChampionCapture>,
    inherit_from_node: Option<i64>,
) -> Option<AgentBrainState> {
    let captured = champion_capture.and_then(|capture| {
        let champion = birth_states.get(capture.agent_index)?.clone();
        governor.store_champion_brain(capture.node_id, &champion);
        Some((capture.node_id, champion))
    });
    let node_id = inherit_from_node?;
    match captured {
        Some((captured_node, champion)) if captured_node == node_id => Some(champion),
        _ => governor.champion_brain(node_id),
    }
}

/// Per-generation learning metrics: behavioral signal (food per 1k
/// alive-ticks) plus the policy weight norms of the generation's best
/// agent. These stay flat for a population that isn't learning and should
/// trend upward once credit assignment reaches food-approach actions.
/// Printed alongside the fitness line so headless runs double as
/// before/after measurement records.
fn log_learning_metrics(
    agents: &[Agent],
    best_state: Option<&AgentBrainState>,
    config: &BrainConfig,
    first_quarter_rate: f64,
    last_quarter_rate: f64,
) {
    let total_food: u64 = agents.iter().map(|a| u64::from(a.food_consumed)).sum();
    let total_deaths: u64 = agents.iter().map(|a| u64::from(a.death_count)).sum();
    // Foraging rate per 1k alive-ticks: total food normalized by the
    // life-time the population actually accrued. Unlike food-per-life this is
    // robust to death count — an active forager that dies often still scores
    // its foraging honestly — so it is the cleaner cross-generation learning
    // signal. (`total_ticks_alive` is preserved across respawn.)
    let total_alive_ticks: u64 = agents.iter().map(|a| a.total_ticks_alive).sum();
    let food_per_1k = if total_alive_ticks > 0 {
        total_food as f64 / total_alive_ticks as f64 * 1000.0
    } else {
        0.0
    };

    let mut weight_norms = String::new();
    if let Some(state) = best_state {
        let layout = BrainLayout::new(config.vision_width, config.vision_height);
        // Tail offsets are vision-independent deltas from the context-weight
        // slot; rebase them onto this layout's dynamic position.
        let tail_base = layout.feature_count * ENCODED_DIMENSION
            + ENCODED_DIMENSION
            + PREDICTOR_DIMENSION * ENCODED_DIMENSION;
        let forward_base = tail_base + (O_ACTION_FORWARD_WEIGHTS - O_PREDICTOR_CONTEXT_WEIGHT);
        let turn_base = tail_base + (O_ACTION_TURN_WEIGHTS - O_PREDICTOR_CONTEXT_WEIGHT);
        if state.brain_state.len() >= turn_base + ENCODED_DIMENSION {
            let l2_norm = |base: usize| -> f32 {
                state.brain_state[base..base + ENCODED_DIMENSION]
                    .iter()
                    .map(|w| w * w)
                    .sum::<f32>()
                    .sqrt()
            };
            weight_norms = format!(
                " | w_fwd {:.3} | w_turn {:.3}",
                l2_norm(forward_base),
                l2_norm(turn_base),
            );
        }
    }
    println!(
        "  Food: {total_food} | Deaths: {total_deaths} | Food/1k-ticks: {food_per_1k:.3} \
| Learn q1→q4: {first_quarter_rate:.3} → {last_quarter_rate:.3}{weight_norms}"
    );
}

/// Print evolution tree from database and exit.
pub fn dump_tree(db_path: &str) {
    let mut gov = match Governor::resume(db_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to open database '{}': {}", db_path, e);
            std::process::exit(1);
        }
    };

    let nodes = gov.tree_nodes();
    if nodes.is_empty() {
        println!("No evolution nodes found in {}", db_path);
        return;
    }

    println!("Evolution Tree ({} nodes):", nodes.len());
    println!("─────────────────────────────────────────────");
    for node in &nodes {
        let indent = "  ".repeat(node.generation as usize);
        let fitness_str = node
            .best_fitness
            .map(|f| format!("{:.4}", f))
            .unwrap_or_else(|| "—".into());
        let mutation_str = if node.mutations.is_empty() {
            String::new()
        } else {
            let parts: Vec<String> = node
                .mutations
                .iter()
                .map(|(p, d)| format!("{}{}", p, if *d > 0.0 { "↑" } else { "↓" }))
                .collect();
            format!(" ({})", parts.join(" "))
        };
        let status_marker = match node.status.as_str() {
            "failed" => " ✗",
            "exhausted" => " ⊘",
            "successful" => " ✓",
            "active" if Some(node.id) == gov.current_node_id => " ★",
            _ => "",
        };
        println!(
            "{}Gen {:>3} [{}] fitness={}{}{} ",
            indent, node.generation, node.id, fitness_str, mutation_str, status_marker,
        );
    }
}

/// Run danger-percept production A/B validation with bootstrap 95% confidence intervals.
/// Baseline: danger_percept_enabled=false (all other flags off).
/// ON: danger_percept_enabled=true (all other flags off).
/// Measures avoidance-intent, approach-intent, survival (ticks_alive), and steering_alignment
/// across N bootstrap replicates at production scale.
///
/// `tick_budget_override`: if non-zero, replaces the governor's default 1 M-tick budget per
/// generation. Use a smaller value (e.g. 10_000) to make N=100 production-scale bootstrap
/// feasible on hardware where 1 M ticks × 50 gen × 100 pop × 200 arm-calls is prohibitive;
/// 0 means keep the governor's configured value.
pub fn validate_danger_percept(
    config: FullConfig,
    num_generations: u64,
    population: u32,
    num_replicates: usize,
    tick_budget_override: u64,
) {
    let effective_tick_budget = if tick_budget_override > 0 {
        tick_budget_override
    } else {
        config.governor.tick_budget
    };

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DANGER-PERCEPT PRODUCTION A/B VALIDATION");
    println!(
        "Running N={} bootstrap replicates at population {} × {} generations × {} ticks/gen",
        num_replicates, population, num_generations, effective_tick_budget
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Collect metrics across N replicates: baseline (danger_percept OFF) and ON.
    let mut baseline_avoidance_intent: Vec<f32> = Vec::new();
    let mut baseline_approach_intent: Vec<f32> = Vec::new();
    let mut baseline_ticks_alive: Vec<f32> = Vec::new();
    let mut baseline_steering_alignment: Vec<f32> = Vec::new();

    let mut on_avoidance_intent: Vec<f32> = Vec::new();
    let mut on_approach_intent: Vec<f32> = Vec::new();
    let mut on_ticks_alive: Vec<f32> = Vec::new();
    let mut on_steering_alignment: Vec<f32> = Vec::new();

    for replicate in 0..num_replicates {
        println!(
            "\n[Replicate {}/{}] Running baseline (danger_percept OFF) and ON arms...",
            replicate + 1,
            num_replicates
        );

        // Generate a seeded but independent world for this replicate.
        let mut replicate_config = config.clone();
        replicate_config.governor.population_size = population as usize;
        replicate_config.world.seed = config.world.seed.wrapping_add(replicate as u64);
        // Apply tick_budget_override when set: both arms use the same budget for a fair A/B.
        if tick_budget_override > 0 {
            replicate_config.governor.tick_budget = tick_budget_override;
        }

        // Baseline run: all flags off, danger_percept_enabled=false
        let baseline_stats =
            run_headless_with_flags(replicate_config.clone(), num_generations, false, false);

        // ON run: danger_percept_enabled=true, all other flags off
        let on_stats = run_headless_with_flags(replicate_config, num_generations, true, false);

        // Record metrics for this replicate.
        baseline_avoidance_intent.push(baseline_stats.mean_avoidance_intent_fraction);
        baseline_approach_intent.push(baseline_stats.mean_approach_intent_fraction);
        baseline_ticks_alive.push(baseline_stats.mean_ticks_alive as f32);
        baseline_steering_alignment.push(baseline_stats.mean_steering_alignment);

        on_avoidance_intent.push(on_stats.mean_avoidance_intent_fraction);
        on_approach_intent.push(on_stats.mean_approach_intent_fraction);
        on_ticks_alive.push(on_stats.mean_ticks_alive as f32);
        on_steering_alignment.push(on_stats.mean_steering_alignment);
    }

    // Compute bootstrap metrics: point estimate (mean), lower/upper 95% CI, effect.
    let avoidance_baseline = compute_bootstrap_metric(&baseline_avoidance_intent, None);
    let avoidance_on =
        compute_bootstrap_metric(&on_avoidance_intent, Some(avoidance_baseline.point));

    let approach_baseline = compute_bootstrap_metric(&baseline_approach_intent, None);
    let approach_on = compute_bootstrap_metric(&on_approach_intent, Some(approach_baseline.point));

    let ticks_alive_baseline = compute_bootstrap_metric(&baseline_ticks_alive, None);
    let ticks_alive_on =
        compute_bootstrap_metric(&on_ticks_alive, Some(ticks_alive_baseline.point));

    let steering_baseline = compute_bootstrap_metric(&baseline_steering_alignment, None);
    let steering_on =
        compute_bootstrap_metric(&on_steering_alignment, Some(steering_baseline.point));

    // Output JSON — include run parameters alongside metrics so the artifact is self-describing.
    let json_output = serde_json::json!({
        "run_parameters": {
            "num_replicates": num_replicates,
            "population": population,
            "num_generations": num_generations,
            "tick_budget_per_generation": effective_tick_budget,
        },
        "avoidance_intent_baseline": avoidance_baseline,
        "avoidance_intent_on": avoidance_on,
        "approach_intent_baseline": approach_baseline,
        "approach_intent_on": approach_on,
        "mean_ticks_alive_baseline": ticks_alive_baseline,
        "mean_ticks_alive_on": ticks_alive_on,
        "steering_alignment_baseline": steering_baseline,
        "steering_alignment_on": steering_on,
    });

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BOOTSTRAP RESULTS (JSON)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{}",
        serde_json::to_string_pretty(&json_output).unwrap_or_default()
    );

    // Save JSON output.
    let json_path = "danger_percept_bootstrap.json";
    match std::fs::write(
        json_path,
        serde_json::to_string_pretty(&json_output).unwrap_or_default(),
    ) {
        Ok(()) => println!("\nJSON output saved to ./{}", json_path),
        Err(e) => eprintln!("Failed to write {}: {}", json_path, e),
    }

    // Print decision rule.
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DECISION RULE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    print_danger_percept_decision_rule(
        &avoidance_baseline,
        &avoidance_on,
        &approach_baseline,
        &approach_on,
        &ticks_alive_baseline,
        &ticks_alive_on,
        &steering_baseline,
        &steering_on,
    );
}

/// Run innate-instinct prove-or-kill A/B benchmark.
/// Baseline: innate_instincts_enabled=false (blank slate, learning from scratch).
/// ON: innate_instincts_enabled=true (seeded instinct patterns + learning).
/// Both arms use identical seeded populations and worlds (deterministic mutations from config.world.seed).
/// Evaluates three gates: survival (+10%), alignment (>=0.4), food-per-death (>=2.0, real food/deaths).
pub fn validate_innate_instincts(config: FullConfig, num_generations: u64) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("INNATE-INSTINCT VALIDATION");
    println!(
        "Running {} generations with innate_instincts_enabled OFF (baseline) then ON",
        num_generations
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let (_baseline_stats, _on_stats, passed) = run_innate_instinct_ab(config, num_generations);

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!(
        "\nFinal Decision: {}",
        if passed {
            "✓✓✓ ALL GATES PASS"
        } else {
            "✗✗✗ GATE FAILURE"
        }
    );
}

/// Run innate-instinct A/B comparison and evaluate gates.
/// Returns (baseline_stats, on_stats, passed: bool).
pub fn run_innate_instinct_ab(
    config: FullConfig,
    num_generations: u64,
) -> (ValidationStats, ValidationStats, bool) {
    // Baseline: innate_instincts_enabled=false (blank slate, learning from scratch).
    // All other flags off — isolates the innate-instinct flag as the only variable.
    println!("\n=== Baseline (innate_instincts_enabled=false) ===");
    let baseline_stats = run_headless_with_flags(config.clone(), num_generations, false, false);

    // ON: innate_instincts_enabled=true (seeded instincts).
    // All other flags remain off — same single-variable isolation.
    println!("\n=== ON (innate_instincts_enabled=true) ===");
    let on_stats = run_headless_with_flags(config.clone(), num_generations, false, true);

    // Apply gates.
    println!("\n=== Gate Evaluation ===");

    let survival_gate = on_stats.mean_ticks_alive
        >= ((baseline_stats.mean_ticks_alive as f32) * (1.0 + INSTINCT_SURVIVAL_MARGIN)) as u64;
    println!(
        "Survival gate (ON >= baseline × {:.1}%): baseline={}, ON={} → {}",
        INSTINCT_SURVIVAL_MARGIN * 100.0,
        baseline_stats.mean_ticks_alive,
        on_stats.mean_ticks_alive,
        if survival_gate {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );

    let alignment_gate = on_stats.mean_avoidance_intent_fraction >= INSTINCT_ALIGNMENT_FLOOR;
    println!(
        "Alignment gate (avoidance-intent >= {:.1}): ON={:.3} → {}",
        INSTINCT_ALIGNMENT_FLOOR,
        on_stats.mean_avoidance_intent_fraction,
        if alignment_gate {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );

    // REAL food consumed (population mean), never mean_fitness (a composite).
    let food_per_death_on = if on_stats.mean_death_count > FOOD_PER_DEATH_ZERO_GUARD {
        on_stats.mean_food_consumed / on_stats.mean_death_count
    } else {
        f32::INFINITY
    };
    let food_per_death_gate = food_per_death_on >= INSTINCT_FOOD_PER_DEATH_MIN;
    println!(
        "Food-per-death gate (ratio >= {:.1}): ON={:.2} → {}",
        INSTINCT_FOOD_PER_DEATH_MIN,
        food_per_death_on,
        if food_per_death_gate {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );

    let passed = survival_gate && alignment_gate && food_per_death_gate;

    (baseline_stats, on_stats, passed)
}

/// One metric's bootstrap summary over N replicates: the point estimate,
/// its 95% CI bounds (2.5th/97.5th percentile), and the ON−baseline effect.
/// Recorded for ticks_alive, speed_correlation, fitness, and danger_dwell so the
/// flip/retire/defer rule is backed by precision, not point-estimate noise.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BootstrapMetric {
    pub point: f32,
    pub lower_ci: f32,
    pub upper_ci: f32,
    pub effect: f32,
}

/// Statistics collected during a headless run.
#[derive(Clone, Debug)]
pub struct ValidationStats {
    pub mean_fitness: f32,
    pub mean_movement_speed: f32,
    pub mean_ticks_alive: u64,
    /// Mean deaths per agent across all agents and all generations in this run.
    /// Uncapped: unlike `mean_ticks_alive`, this is never pinned to `tick_budget`.
    /// A value of 0.0 means no agent ever died — a population-viability red flag.
    pub mean_death_count: f32,
    /// Mean food consumed per agent across all agents and all generations.
    /// Used for food-per-death ratio calculation in instinct validation.
    pub mean_food_consumed: f32,
    pub speed_fitness_correlation: f32,
    pub death_speed_regression: f32,
    pub food_per_energy_vs_speed_slope: f32,
    pub mean_danger_dwell_fraction: f32,
    pub mean_avoidance_intent_fraction: f32,
    /// Mean approach-intent fraction: sum of approach turns toward food divided by sum of
    /// approach sense-range ticks, population aggregate. Paired with avoidance-intent
    /// for the danger-percept A/B evaluation.
    pub mean_approach_intent_fraction: f32,
    /// Mean per-agent steering alignment: how well the chosen turn tracks the danger/food
    /// gradient. Recorded as a deferral signal in danger-percept A/B to explain why intent
    /// improves but steering stays in the chance band (credit-path bottleneck).
    pub mean_steering_alignment: f32,
    /// Population-mean movement_speed per generation (chronological order).
    /// Used to assess whether speed stops ratcheting toward 100 under the ON flags.
    pub speed_trajectory_per_gen: Vec<f32>,
}

/// Guard value for the food-per-death denominator.
/// Treats mean_death_count values below this threshold as effectively zero, returning
/// f32::INFINITY instead of dividing. Chosen at 1e-4 to absorb floating-point imprecision
/// near zero (e.g., a sub-1-per-10000 death rate is meaninglessly small as a divisor).
const FOOD_PER_DEATH_ZERO_GUARD: f32 = 1e-4;

/// Gate: ON (instincts seeded) must improve survival over baseline by at least this margin.
/// Tuned at 0.10 (10%) to require meaningful benefit while tolerating natural variance.
const INSTINCT_SURVIVAL_MARGIN: f32 = 0.10;

/// Gate: mean avoidance-intent fraction (turns opposing danger bearing) in ON run
/// must exceed this floor. Tuned at 0.4 to require substantial steering-alignment.
const INSTINCT_ALIGNMENT_FLOOR: f32 = 0.4;

/// Gate: ON run's food-per-death ratio (mean food count / mean death count) must
/// exceed this threshold. Tuned at 2.0 to require at least 2 food consumed per death.
const INSTINCT_FOOD_PER_DEATH_MIN: f32 = 2.0;

/// Run headless evolution and collect statistics with specified flags.
///
/// `danger_percept_enabled` adds the dedicated danger bearing/distance senses;
/// `innate_instincts_enabled` seeds the instinct patterns for the innate-instinct
/// A/B harness. Everything else runs at the config's own values, so a baseline
/// arm and an ON arm differ only in the flag under test.
fn run_headless_with_flags(
    mut config: FullConfig,
    num_generations: u64,
    danger_percept_enabled: bool,
    innate_instincts_enabled: bool,
) -> ValidationStats {
    config.brain.danger_percept_enabled = danger_percept_enabled;
    config.brain.innate_instincts_enabled = innate_instincts_enabled;

    println!(
        "  Flags: danger_percept={}, speed_cost_exponent={}, innate_instincts={}",
        danger_percept_enabled, config.brain.speed_cost_exponent, innate_instincts_enabled
    );

    // Create a temporary database for this run
    let temp_db = format!(
        "xagent-validation-{}-{}.db",
        if danger_percept_enabled || innate_instincts_enabled {
            "ON"
        } else {
            "BASELINE"
        },
        std::process::id()
    );
    let mut governor = Governor::new(&temp_db, config.governor.clone(), &config.brain, "")
        .expect("Failed to initialize validation governor");

    let seed_config = config.brain.clone();
    // Both arms share this config. Steering genomes are drawn from a seed
    // derived from the world seed, so the same world seed makes the arms'
    // initial turn policies identical.
    let mut current_configs = repeated_config(&seed_config, governor.config.population_size);

    let mut inherited_state: Option<AgentBrainState> = None;
    let mut inherited_mutation_strength: f32 = governor.config.mutation_strength;
    let repeats = governor.config.eval_repeats.max(1);

    let pop_size = governor.config.population_size;
    let world = WorldState::new(config.world.clone());
    let food_count = world.food_items.len();
    let mut kernel = GpuKernel::new(pop_size as u32, food_count, &seed_config, &config.world);

    let mut all_fitness: Vec<Vec<crate::governor::AgentFitness>> = Vec::new();
    // Per-generation mean movement_speed (for the trajectory check).
    let mut speed_trajectory_per_gen: Vec<f32> = Vec::new();

    for generation_index in 0..num_generations {
        if governor.evolution_complete() {
            break;
        }

        let world = WorldState::new(config.world.clone());
        let mut agents: Vec<Agent> = current_configs
            .iter()
            .enumerate()
            .map(|(i, cfg)| {
                let pos = world.safe_spawn_position();
                Agent::new(i as u32, pos, i as u32, cfg.clone(), 0)
            })
            .collect();

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

        let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = agents
            .iter()
            .map(|a| {
                (
                    a.body.body.position,
                    a.body.body.internal.max_energy,
                    a.body.body.internal.max_integrity,
                    a.brain_config.memory_capacity,
                    a.brain_config.processing_slots,
                )
            })
            .collect();
        kernel.upload_agents(&agent_data);
        kernel.reset_agents_seeded(&current_configs[0], config.world.seed);

        let mut rng =
            rand::rngs::SmallRng::seed_from_u64(config.world.seed.wrapping_add(generation_index));
        let template = match &inherited_state {
            Some(state) => state.clone(),
            None => fresh_brain_state(&current_configs[0], &mut rng),
        };
        write_steering_population(
            &kernel,
            &agents,
            &template,
            repeats,
            inherited_mutation_strength,
            &mut rng,
        );

        for (i, agent) in agents.iter().enumerate() {
            kernel.write_agent_heritable_config(agent.brain_idx, &current_configs[i]);
        }
        let birth_states: Vec<AgentBrainState> = agents
            .iter()
            .map(|agent| kernel.read_agent_state(agent.brain_idx))
            .collect();

        governor.gen_tick = 0;

        let tick_budget = governor.config.tick_budget;
        let mut ticks_done: u64 = 0;

        // Adaptive dispatch interval: tick_budget / VALIDATION_HEATMAP_SAMPLES gives exactly
        // VALIDATION_HEATMAP_SAMPLES readbacks per generation, clamped to at least
        // HEATMAP_INTERVAL (100 ticks) to avoid sub-stride batches.  This balances heatmap
        // coverage (cells_explored) against GPU round-trip overhead regardless of tick_budget.
        let validation_interval = ((tick_budget / u64::from(VALIDATION_HEATMAP_SAMPLES))
            .max(u64::from(HEATMAP_INTERVAL))) as u32;

        while ticks_done < tick_budget {
            // Use the adaptive dispatch interval to keep Metal/Vulkan round-trip overhead low
            // while still sampling enough positions for meaningful heatmap coverage.  Both arms
            // see the same interval, so the relative A/B comparison is unbiased.
            let remaining = (tick_budget - ticks_done).min(validation_interval as u64) as u32;
            kernel.dispatch_batch(ticks_done, remaining);
            ticks_done += remaining as u64;
            governor.advance_ticks(u64::from(remaining));

            while !kernel.try_collect_state_snapshot() {
                std::thread::yield_now();
            }
            let state = kernel.cached_state();
            for i in 0..agents.len() {
                let base = i * PHYS_STRIDE;
                let alive = state[base + P_ALIVE] > 0.5;
                if alive {
                    agents[i].body.body.position = glam::Vec3::new(
                        state[base + P_POS_X],
                        state[base + P_POS_Y],
                        state[base + P_POS_Z],
                    );
                    agents[i].record_heatmap(config.world.world_size);
                }
            }

            let mut cumulative_food = 0_u64;
            let mut cumulative_alive = 0_u64;
            for i in 0..agents.len() {
                let base = i * PHYS_STRIDE;
                cumulative_food += state[base + P_FOOD_COUNT] as u64;
                cumulative_alive += state[base + P_TICKS_ALIVE] as u64;
            }
            governor.record_within_life_sample(cumulative_food, cumulative_alive);
        }

        let state = kernel.cached_state();
        for i in 0..agents.len() {
            let base = i * PHYS_STRIDE;
            agents[i].food_consumed = state[base + P_FOOD_COUNT] as u32;
            agents[i].total_ticks_alive = state[base + P_TICKS_ALIVE] as u64;
            agents[i].distance_traveled = state[base + P_DISTANCE_TRAVELED];
            agents[i].energy_spent = state[base + P_ENERGY_SPENT];
            agents[i].danger_path_length = state[base + P_DANGER_PATH_LENGTH];
            agents[i].avoidance_sense_range_ticks = state[base + P_AVOIDANCE_SENSE_RANGE_TICKS];
            agents[i].avoidance_turns_opposing = state[base + P_AVOIDANCE_TURNS_OPPOSING];
            agents[i].avoidance_turn_away = state[base + P_AVOIDANCE_TURN_AWAY];
            agents[i].hazard_entries = state[base + P_HAZARD_ENTRIES];
            agents[i].approach_sense_range_ticks = state[base + P_APPROACH_SENSE_RANGE_TICKS];
            agents[i].approach_turns_toward = state[base + P_APPROACH_TURNS_TOWARD];
            agents[i].death_count = state[base + P_DEATH_COUNT] as u32;
        }

        let (_first_quarter_rate, _last_quarter_rate) = governor.within_life_rates();
        let fitness = governor.evaluate(&agents);

        // Record per-generation mean speed for the trajectory check.
        // AgentFitness.config.movement_speed is the canonical pairing of speed
        // to this agent's evaluation result; using the fitness slice avoids any
        // index-order mismatch with `evaluate()`'s internal sort.
        let gen_mean_speed = if fitness.is_empty() {
            0.0
        } else {
            fitness.iter().map(|f| f.config.movement_speed).sum::<f32>() / fitness.len() as f32
        };
        speed_trajectory_per_gen.push(gen_mean_speed);

        // Record for aggregate analysis
        all_fitness.push(fitness.clone());

        governor.log_generation(&fitness);

        match governor.advance(&fitness) {
            AdvanceResult::Continue {
                configs,
                messages: _,
                mutation_strength,
                champion_capture,
                inherit_from_node,
            } => {
                inherited_state = resolve_inherited_brain(
                    &governor,
                    &birth_states,
                    champion_capture,
                    inherit_from_node,
                );
                current_configs = configs;
                inherited_mutation_strength = mutation_strength;
            }
            AdvanceResult::Finished { .. } => {
                break;
            }
        }
    }

    // Flatten all generations into a single slice for aggregate metrics.
    // Each AgentFitness entry carries its own `config.movement_speed`, so speed
    // is always paired with the correct fitness record — no separate speed vec
    // needed, and no risk of index mismatch from evaluate()'s internal sort.
    let all_agents_fitness: Vec<crate::governor::AgentFitness> =
        all_fitness.into_iter().flatten().collect();

    let mean_fitness = if all_agents_fitness.is_empty() {
        0.0
    } else {
        all_agents_fitness
            .iter()
            .map(|f| f.composite_fitness)
            .sum::<f32>()
            / all_agents_fitness.len() as f32
    };

    // Aggregate mean speed — derived from AgentFitness.config.movement_speed so
    // it is always aligned with the fitness record for the same agent.
    let mean_movement_speed = if all_agents_fitness.is_empty() {
        0.0
    } else {
        all_agents_fitness
            .iter()
            .map(|f| f.config.movement_speed)
            .sum::<f32>()
            / all_agents_fitness.len() as f32
    };

    let mean_ticks_alive = if all_agents_fitness.is_empty() {
        0
    } else {
        let total: u64 = all_agents_fitness.iter().map(|f| f.total_ticks_alive).sum();
        total / all_agents_fitness.len() as u64
    };

    // Build aligned speed / fitness slices for correlation and regression.
    // Using AgentFitness.config.movement_speed guarantees each speed value is
    // the genome of the same agent whose fitness/death/foraging appear in the
    // parallel position — evaluate()'s sort does not break the pairing.
    let speeds: Vec<f32> = all_agents_fitness
        .iter()
        .map(|f| f.config.movement_speed)
        .collect();
    let fitnesses: Vec<f32> = all_agents_fitness
        .iter()
        .map(|f| f.composite_fitness)
        .collect();
    let death_counts: Vec<f32> = all_agents_fitness
        .iter()
        .map(|f| f.death_count as f32)
        .collect();
    // Mean deaths per agent: uncapped metric not pinned to tick_budget.
    // If agents never die, this is 0.0 — a population-viability red flag.
    let mean_death_count = if death_counts.is_empty() {
        0.0
    } else {
        death_counts.iter().sum::<f32>() / death_counts.len() as f32
    };
    let mean_food_consumed = if all_agents_fitness.is_empty() {
        0.0
    } else {
        all_agents_fitness
            .iter()
            .map(|f| f.food_consumed as f32)
            .sum::<f32>()
            / all_agents_fitness.len() as f32
    };
    let food_per_energy: Vec<f32> = all_agents_fitness
        .iter()
        .map(|f| {
            if f.energy_spent > 0.001 {
                f.food_consumed as f32 / f.energy_spent
            } else {
                0.0
            }
        })
        .collect();

    let speed_fitness_correlation = compute_correlation(&speeds, &fitnesses);
    let death_speed_regression = compute_regression(&speeds, &death_counts);
    let food_per_energy_vs_speed_slope = compute_regression(&speeds, &food_per_energy);

    // danger_dwell_fraction and avoidance_intent_fraction: population-aggregate
    // statistics computed through the shared reducer, identical to production.
    let mean_danger_dwell_fraction = if all_agents_fitness.is_empty() {
        0.0
    } else {
        compute_danger_dwell_fraction(&all_agents_fitness)
    };

    let mean_avoidance_intent_fraction = if all_agents_fitness.is_empty() {
        0.0
    } else {
        compute_avoidance_intent_fraction(&all_agents_fitness)
    };

    let mean_approach_intent_fraction = if all_agents_fitness.is_empty() {
        0.0
    } else {
        compute_approach_intent_fraction(&all_agents_fitness)
    };

    // Steering alignment: average of approach and avoidance intent fractions.
    // Both are bounded [0, 1]; their mean reflects overall directional steering effectiveness.
    // At chance, this is ~0.46-0.5 (the empirical baseline for vision-conditional steering).
    let mean_steering_alignment =
        (mean_approach_intent_fraction + mean_avoidance_intent_fraction) / 2.0;

    // Clean up temp database and its sidecars
    let _ = std::fs::remove_file(&temp_db);
    let _ = std::fs::remove_file(format!("{}-wal", temp_db));
    let _ = std::fs::remove_file(format!("{}-shm", temp_db));

    ValidationStats {
        mean_fitness,
        mean_movement_speed,
        mean_ticks_alive,
        mean_death_count,
        mean_food_consumed,
        speed_fitness_correlation,
        death_speed_regression,
        food_per_energy_vs_speed_slope,
        mean_danger_dwell_fraction,
        mean_avoidance_intent_fraction,
        mean_approach_intent_fraction,
        mean_steering_alignment,
        speed_trajectory_per_gen,
    }
}

/// Lower percentile bound for the 95% bootstrap confidence interval (2.5th percentile).
const CI_LOWER_PERCENTILE: f32 = 0.025;
/// Upper percentile bound for the 95% bootstrap confidence interval (97.5th percentile).
const CI_UPPER_PERCENTILE: f32 = 0.975;

/// Compute bootstrap metric: point estimate, 95% CI bounds, and effect size.
/// If baseline_point is None, effect is computed as zero (no comparison).
fn compute_bootstrap_metric(values: &[f32], baseline_point: Option<f32>) -> BootstrapMetric {
    if values.is_empty() {
        return BootstrapMetric {
            point: 0.0,
            lower_ci: 0.0,
            upper_ci: 0.0,
            effect: 0.0,
        };
    }

    // Point estimate: mean of replicates.
    let point = values.iter().sum::<f32>() / values.len() as f32;

    // 95% CI: 2.5th and 97.5th percentile.
    // Use floor for the lower index and ceil for the upper index, then clamp so that
    // lower_idx <= upper_idx regardless of N (necessary for N < 40 where the two
    // percentile indices would otherwise cross).
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let last = values.len() - 1;
    let lower_idx = ((CI_LOWER_PERCENTILE * last as f32).floor() as usize).min(last);
    let upper_idx = ((CI_UPPER_PERCENTILE * last as f32).ceil() as usize)
        .min(last)
        .max(lower_idx);

    let lower_ci = sorted[lower_idx];
    let upper_ci = sorted[upper_idx];

    // Effect: ON − baseline.
    let effect = baseline_point.map_or(0.0, |b| point - b);

    BootstrapMetric {
        point,
        lower_ci,
        upper_ci,
        effect,
    }
}

/// Lower bound of the "steering at chance" band. A value below this is considered regressed.
/// 0.38 mirrors the upper bound (0.62) symmetrically around 0.5 (random chance).
const STEERING_CHANCE_FLOOR: f32 = 0.38;
/// Upper bound of the chance band; also the minimum threshold for "good" steering.
/// Using a single constant avoids having two names for the same boundary (0.62).
const STEERING_GOOD_THRESHOLD: f32 = 0.62;
/// Acceptable ±band for ticks_alive stability between baseline and ON arms.
/// ±5% keeps the survival gate sensitive enough to catch real regressions while
/// tolerating the stochastic noise typical of 50-generation evolutionary runs.
const SURVIVAL_STABILITY_BAND: f32 = 0.05;

/// Print danger-percept decision rule and verdict.
/// Thresholds define flip (intent up, survival stable, steering good), retire (intent down,
/// survival down), or defer (intent/survival pass but steering at chance, unlock when credit
/// path improves).
fn print_danger_percept_decision_rule(
    avoidance_baseline: &BootstrapMetric,
    avoidance_on: &BootstrapMetric,
    approach_baseline: &BootstrapMetric,
    approach_on: &BootstrapMetric,
    ticks_alive_baseline: &BootstrapMetric,
    ticks_alive_on: &BootstrapMetric,
    steering_baseline: &BootstrapMetric,
    steering_on: &BootstrapMetric,
) {
    println!("\nThresholds:");
    println!("  (a) avoidance-intent ON > baseline (lower CI of ON > point of baseline)");
    println!(
        "  (b) survival ON within ±{:.0}% of baseline (CI overlap)",
        SURVIVAL_STABILITY_BAND * 100.0
    );
    println!(
        "  (c) steering_alignment ON: good if > {:.2}, chance if in [{:.2}, {:.2}]",
        STEERING_GOOD_THRESHOLD, STEERING_CHANCE_FLOOR, STEERING_GOOD_THRESHOLD
    );

    // Check gates.
    let lower_survival_band = ticks_alive_baseline.point * (1.0 - SURVIVAL_STABILITY_BAND);
    let upper_survival_band = ticks_alive_baseline.point * (1.0 + SURVIVAL_STABILITY_BAND);
    let gate_a = avoidance_on.lower_ci > avoidance_baseline.point;
    let gate_b =
        ticks_alive_on.point >= lower_survival_band && ticks_alive_on.point <= upper_survival_band;
    let steering_is_good = steering_on.point > STEERING_GOOD_THRESHOLD;
    let steering_is_chance =
        steering_on.point >= STEERING_CHANCE_FLOOR && steering_on.point <= STEERING_GOOD_THRESHOLD;

    println!("\nGate Evaluation:");
    println!(
        "  (a) ON avoidance_intent {:.4} (CI: [{:.4}, {:.4}]) > baseline {:.4}? → {}",
        avoidance_on.point,
        avoidance_on.lower_ci,
        avoidance_on.upper_ci,
        avoidance_baseline.point,
        if gate_a { "✓" } else { "✗" }
    );
    println!(
        "  (b) ON ticks_alive {:.0} in [{:.0}, {:.0}]? → {}",
        ticks_alive_on.point,
        lower_survival_band,
        upper_survival_band,
        if gate_b { "✓" } else { "✗" }
    );
    let steering_label = if steering_is_good {
        format!("GOOD (> {:.2})", STEERING_GOOD_THRESHOLD)
    } else if steering_is_chance {
        format!(
            "CHANCE-BAND ({:.2}-{:.2}, credit path bottleneck)",
            STEERING_CHANCE_FLOOR, STEERING_GOOD_THRESHOLD
        )
    } else {
        format!("REGRESSED (< {:.2})", STEERING_CHANCE_FLOOR)
    };
    println!(
        "  (c) ON steering_alignment {:.4} (baseline {:.4}): {}",
        steering_on.point, steering_baseline.point, steering_label
    );

    // Print auxiliary metrics for context.
    println!("\nAuxiliary Metrics:");
    println!(
        "  approach_intent ON: {:.4} (baseline {:.4}, Δ {:.4})",
        approach_on.point, approach_baseline.point, approach_on.effect
    );
    println!(
        "  avoidance_intent ON: {:.4} (baseline {:.4}, Δ {:.4})",
        avoidance_on.point, avoidance_baseline.point, avoidance_on.effect
    );

    // Decision logic:
    // FLIP: avoidance intent up AND survival stable.
    // RETIRE: avoidance intent down OR survival regressed.
    // DEFER: avoidance intent and survival pass BUT steering stays at chance,
    //        document unlock condition (steering must exceed 0.62).
    let verdict = if gate_a && gate_b && steering_is_good {
        "FLIP: Avoidance-intent up, survival stable, steering good. Danger-percept is ready to ship."
    } else if gate_a && gate_b && steering_is_chance {
        "DEFER: Avoidance-intent up, survival stable, but steering remains in the chance band [0.38, 0.62]. \
         Unlock condition: steering_alignment must exceed 0.62 on the default learning path (requires credit-path improvements) before danger-percept-enabled contributes independently. \
         The percept is wired and improves intent, but the credit path (not the percept) limits steering effectiveness."
    } else if !gate_a || !gate_b {
        "RETIRE: Avoidance-intent did not improve beyond baseline OR survival regressed. Mechanism does not show benefit at production scale."
    } else {
        "DEFER: Inconclusive. Steering regressed beyond the chance band, requiring investigation before integration."
    };

    println!("\n>>> VERDICT: {}", verdict);
}

/// Compute Pearson correlation between two vectors.
fn compute_correlation(x: &[f32], y: &[f32]) -> f32 {
    if x.len() < 2 || x.len() != y.len() {
        return 0.0;
    }

    let n = x.len() as f32;
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;

    let mut numerator = 0.0;
    let mut sum_x_sq = 0.0;
    let mut sum_y_sq = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator += dx * dy;
        sum_x_sq += dx * dx;
        sum_y_sq += dy * dy;
    }

    let denominator = (sum_x_sq * sum_y_sq).sqrt();
    if denominator < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compute linear regression slope (y ~ x).
fn compute_regression(x: &[f32], y: &[f32]) -> f32 {
    if x.len() < 2 || x.len() != y.len() {
        return 0.0;
    }

    let n = x.len() as f32;
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator += dx * dy;
        denominator += dx * dx;
    }

    if denominator < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Print validation metrics to console.
#[allow(dead_code)]
fn print_validation_metrics(baseline: &ValidationStats, on_stats: &ValidationStats) {
    println!("\nMetric                            Baseline        On              Delta");
    println!("──────────────────────────────────────────────────────────────────────");

    let print_metric = |label: &str, baseline_val: f32, on_val: f32| {
        let delta = on_val - baseline_val;
        let delta_str = if delta >= 0.0 {
            format!("+{:.4}", delta)
        } else {
            format!("{:.4}", delta)
        };
        println!(
            "{:<35} {:<15.4} {:<15.4} {}",
            label, baseline_val, on_val, delta_str
        );
    };

    print_metric(
        "speed-fitness correlation",
        baseline.speed_fitness_correlation,
        on_stats.speed_fitness_correlation,
    );
    print_metric(
        "death-speed regression slope",
        baseline.death_speed_regression,
        on_stats.death_speed_regression,
    );
    print_metric(
        "food-per-energy vs speed slope",
        baseline.food_per_energy_vs_speed_slope,
        on_stats.food_per_energy_vs_speed_slope,
    );
    print_metric("mean fitness", baseline.mean_fitness, on_stats.mean_fitness);
    print_metric(
        "mean movement_speed",
        baseline.mean_movement_speed,
        on_stats.mean_movement_speed,
    );
    print_metric(
        "mean ticks_alive",
        baseline.mean_ticks_alive as f32,
        on_stats.mean_ticks_alive as f32,
    );
    print_metric(
        "mean danger_dwell_fraction",
        baseline.mean_danger_dwell_fraction,
        on_stats.mean_danger_dwell_fraction,
    );
    print_metric(
        "mean avoidance_intent_fraction",
        baseline.mean_avoidance_intent_fraction,
        on_stats.mean_avoidance_intent_fraction,
    );

    // Print per-generation speed trajectory for both runs.
    println!("\nPer-generation mean speed trajectory:");
    println!(
        "  Baseline: {}",
        baseline
            .speed_trajectory_per_gen
            .iter()
            .map(|s| format!("{:.1}", s))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "  On:       {}",
        on_stats
            .speed_trajectory_per_gen
            .iter()
            .map(|s| format!("{:.1}", s))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// Where a free life's birth brain comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BrainSource {
    /// One freshly seeded birth brain shared by every agent.
    Fresh,
    /// The stored champion of a governor database, read only.
    Champion {
        /// Path of the governor database.
        db_path: String,
        /// Node to load; `None` selects the newest stored champion.
        node_id: Option<i64>,
    },
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
/// Report label of a stored champion birth brain.
const SOURCE_CHAMPION: &str = "champion";
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
    /// The champion could not be loaded.
    ChampionLoad(ChampionLoadError),
    /// The champion's brain does not fit its config's layout.
    ChampionLayout {
        /// Brain length the config's layout needs.
        expected: usize,
        /// Brain length stored.
        found: usize,
    },
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
    /// The food-kind run was asked to start from a stored champion brain.
    FoodKindChampion,
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
            Self::FoodKindChampion => write!(
                f,
                "--food-kinds cannot start from a stored champion brain; use a fresh brain"
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
            Self::ChampionLoad(err) => write!(f, "cannot load champion: {err}"),
            Self::ChampionLayout { expected, found } => write!(
                f,
                "champion brain has {found} values, its config needs {expected}"
            ),
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
    /// `"fresh"`, `"champion"` or `"archive"`.
    pub brain_source: String,
    pub champion_node_id: Option<i64>,
    pub champion_generation: Option<i64>,
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
    champion_brain: Option<AgentBrainState>,
    node_id: Option<i64>,
    generation: Option<i64>,
    archive_entry_id: Option<i64>,
}

/// Resolve the birth source, check its layout and the vision stride.
fn resolve_birth(config: &FullConfig, source: &BrainSource) -> Result<Birth, FreeLifeError> {
    let birth = match source {
        BrainSource::Fresh => Birth {
            brain_config: config.brain.clone(),
            champion_brain: None,
            node_id: None,
            generation: None,
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
                champion_brain: Some(stored.brain),
                node_id: None,
                generation: None,
                archive_entry_id: Some(entry_id),
            }
        }
        BrainSource::Champion { db_path, node_id } => {
            let stored =
                load_champion_read_only(db_path, *node_id).map_err(FreeLifeError::ChampionLoad)?;
            let expected = BrainLayout::from_config(&stored.config).brain_stride;
            let found = stored.brain.brain_state.len();
            if expected != found {
                return Err(FreeLifeError::ChampionLayout { expected, found });
            }
            Birth {
                brain_config: stored.config,
                champion_brain: Some(stored.brain),
                node_id: Some(stored.node_id),
                generation: Some(stored.generation),
                archive_entry_id: None,
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
    let state = match &birth.champion_brain {
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
        return Err(FreeLifeError::FoodKindChampion);
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
    } else if birth.node_id.is_some() {
        SOURCE_CHAMPION
    } else {
        SOURCE_FRESH
    };
    let report = FreeLifeReport {
        agent_count: options.agent_count,
        ticks: options.ticks,
        seed: options.seed,
        brain_source: source_label.to_string(),
        champion_node_id: birth.node_id,
        champion_generation: birth.generation,
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
        fresh_brains: report.champion_node_id.is_none(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_correlation_extremes() {
        // Perfect positive correlation: x = y
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = compute_correlation(&x, &y);
        assert!(
            (corr - 1.0).abs() < 1e-6,
            "perfect correlation should be ~1.0"
        );

        // Perfect negative correlation: x = -y
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
        let corr = compute_correlation(&x, &y);
        assert!(
            (corr - (-1.0)).abs() < 1e-6,
            "perfect negative correlation should be ~-1.0"
        );

        // Zero correlation: constant x
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = compute_correlation(&x, &y);
        assert!(
            corr.abs() < 1e-6,
            "constant x should give ~0 correlation (not NaN)"
        );

        // Insufficient data
        let x = vec![1.0];
        let y = vec![1.0];
        let corr = compute_correlation(&x, &y);
        assert!(corr == 0.0, "insufficient data should return 0.0");
    }

    #[test]
    fn compute_regression_slope() {
        // Known slope: y = 2*x
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let slope = compute_regression(&x, &y);
        assert!((slope - 2.0).abs() < 1e-6, "slope of y=2x should be 2.0");

        // Known slope: y = -0.5*x
        let x = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let y = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
        let slope = compute_regression(&x, &y);
        assert!(
            (slope - (-0.5)).abs() < 1e-6,
            "slope of y=-0.5x should be -0.5"
        );

        // Insufficient data
        let x = vec![1.0];
        let y = vec![1.0];
        let slope = compute_regression(&x, &y);
        assert!(slope == 0.0, "insufficient data should return 0.0");
    }

    // ── innate-instinct validation gate predicates ──────────────────────

    #[test]
    fn innate_gate_survival_requires_10_percent_improvement() {
        // Test the survival gate predicate: ON ticks must exceed baseline by 10%.
        let baseline_ticks = 100000u64;
        let on_ticks_below = 109999u64; // Just below 110% threshold
        let on_ticks_at = 110000u64; // Exactly at 110%
        let on_ticks_above = 110001u64; // Above 110%

        let threshold = (baseline_ticks as f32 * (1.0 + INSTINCT_SURVIVAL_MARGIN)) as u64;

        // Below threshold should fail
        assert!(
            on_ticks_below < threshold,
            "test setup: on_ticks_below should be below threshold"
        );

        // At or above threshold should pass
        assert!(
            on_ticks_at >= threshold,
            "test setup: on_ticks_at should be at threshold"
        );
        assert!(
            on_ticks_above >= threshold,
            "test setup: on_ticks_above should be above threshold"
        );
    }

    #[test]
    fn innate_gate_alignment_requires_0_4_avoidance_intent() {
        // Test alignment gate predicate: avoidance-intent fraction must be >= 0.4.
        let below_floor = 0.39999_f32;
        let at_floor = 0.4_f32;
        let above_floor = 0.40001_f32;

        // Below floor should fail gate
        assert!(
            below_floor < INSTINCT_ALIGNMENT_FLOOR,
            "below_floor should fail alignment gate"
        );

        // At or above floor should pass gate
        assert!(
            at_floor >= INSTINCT_ALIGNMENT_FLOOR,
            "at_floor should pass alignment gate"
        );
        assert!(
            above_floor >= INSTINCT_ALIGNMENT_FLOOR,
            "above_floor should pass alignment gate"
        );
    }

    #[test]
    fn innate_gate_food_per_death_requires_2_0_ratio() {
        // Test food-per-death gate predicate: ratio must be >= 2.0.
        let mean_food = 10.0_f32;
        let death_count_below = 5.00001_f32; // ratio = 10 / 5.00001 ≈ 1.9999 < 2.0
        let death_count_at = 5.0_f32; // ratio = 10 / 5.0 = 2.0
        let death_count_above = 4.99999_f32; // ratio = 10 / 4.99999 ≈ 2.00001 > 2.0

        let ratio_below = mean_food / death_count_below;
        let ratio_at = mean_food / death_count_at;
        let ratio_above = mean_food / death_count_above;

        // Below threshold should fail gate
        assert!(
            ratio_below < INSTINCT_FOOD_PER_DEATH_MIN,
            "ratio_below should fail food-per-death gate"
        );

        // At or above threshold should pass gate
        assert!(
            ratio_at >= INSTINCT_FOOD_PER_DEATH_MIN,
            "ratio_at should pass food-per-death gate"
        );
        assert!(
            ratio_above >= INSTINCT_FOOD_PER_DEATH_MIN,
            "ratio_above should pass food-per-death gate"
        );
    }

    #[test]
    fn innate_gate_food_per_death_infinity_with_zero_deaths() {
        // Edge case: zero deaths yields infinity, which always passes food-per-death gate.
        let mean_food = 10.0_f32;
        let zero_deaths = 0.0_f32;

        let food_per_death = if zero_deaths > FOOD_PER_DEATH_ZERO_GUARD {
            mean_food / zero_deaths
        } else {
            f32::INFINITY
        };

        // Infinity should pass the gate
        assert!(
            food_per_death >= INSTINCT_FOOD_PER_DEATH_MIN,
            "food_per_death=infinity should pass the threshold"
        );
    }

    #[test]
    fn innate_gate_all_three_predicates_required() {
        // Verify that all three gates are conjunctive (all must be true).
        // Test a scenario where two pass but one fails.

        // Survival gate passes (ON = baseline + 15%)
        let baseline_ticks = 100000u64;
        let on_ticks = 115000u64;

        // Alignment gate passes (ON avoidance = 0.5 > 0.4)
        let on_avoidance = 0.5_f32;

        // Food-per-death gate FAILS (ratio = 1.5 < 2.0)
        let on_food = 7.5_f32;
        let on_death = 5.0_f32;
        let on_food_per_death = on_food / on_death;

        // Individual gate evaluations
        let survival_gate =
            on_ticks >= ((baseline_ticks as f32) * (1.0 + INSTINCT_SURVIVAL_MARGIN)) as u64;
        let alignment_gate = on_avoidance >= INSTINCT_ALIGNMENT_FLOOR;
        let food_per_death_gate = on_food_per_death >= INSTINCT_FOOD_PER_DEATH_MIN;

        // Verify individual gate states
        assert!(survival_gate, "survival gate should pass");
        assert!(alignment_gate, "alignment gate should pass");
        assert!(!food_per_death_gate, "food-per-death gate should fail");

        // Combined gate must fail
        let combined_gate = survival_gate && alignment_gate && food_per_death_gate;
        assert!(
            !combined_gate,
            "combined gate must fail when any individual gate fails"
        );
    }
}
