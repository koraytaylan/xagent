//! Simulation worker: owns the GPU kernel and the simulation-cadence state.
//!
//! The render/UI loop no longer drives simulation progress. Instead a
//! background worker thread owns the fused [`GpuKernel`], advances ticks on a
//! wall-clock-driven cadence, enforces the generation tick budget, and
//! publishes CPU-visible state to the main thread through bounded channels.
//! The main thread sends [`SimCommand`]s (speed, pause, selection, champion
//! readback, population reset, shutdown) and drains [`SimEvent`]s (snapshots,
//! telemetry, generation-boundary, champion state, logs, errors), applying the
//! newest snapshot to its CPU-side caches and rendering from those.
//!
//! Snapshots and telemetry are *sampled observations*: the channel drops stale
//! snapshots under back-pressure and the consumer keeps only the newest
//! (latest-wins). Control events (kernel-ready, generation-boundary, champion
//! state) are always delivered. The worker exits on [`SimCommand::Shutdown`] or
//! command-channel disconnect and is joined in [`SimRuntime`]'s `Drop`.

use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use xagent_brain::buffers::{BrainLayout, TrailSample, PATTERN_STRIDE};
use xagent_brain::{AgentBrainState, AgentTelemetry, GpuKernel};
use xagent_sandbox::agent::{fresh_brain_state, mutate_brain_state, steering_population};
use xagent_sandbox::supervisor_checkpoint::CheckpointError;
use xagent_sandbox::supervisor_run::{load_resume_point, SampleReport, SupervisorRun};
use xagent_sandbox::supervisor_store::SupervisorStore;
use xagent_sandbox::supervisor_telemetry::{BoundaryRecord, TelemetryFrame};
use xagent_sandbox::tick_window::{frame_from_snapshot, Pin, PinKind, TickWindowConfig};
use xagent_sandbox::tick_window_store::TickWindowStore;
use xagent_shared::{BrainConfig, FullConfig, WorldConfig, SUPERVISOR_SAMPLE_TICKS};

use crate::app::{PendingUpload, SIM_DT};

/// Maximum CPU-visible physics-snapshot publication rate. Display smoothness
/// does not require more than one authoritative state sample per 60 Hz frame,
/// so snapshot requests are throttled to this rate independently of how fast
/// the worker advances compute.
const STATE_SNAPSHOT_MAX_HZ: f64 = 60.0;

/// Minimum wall-clock interval between state-snapshot requests, derived from
/// [`STATE_SNAPSHOT_MAX_HZ`].
const STATE_SNAPSHOT_MIN_INTERVAL: Duration =
    Duration::from_micros((1_000_000.0 / STATE_SNAPSHOT_MAX_HZ) as u64);

/// Maximum selected-agent telemetry request rate. Telemetry is heavier than
/// physics state (it copies sensory, decision, and brain slices) and drives
/// charts rather than body placement, so it publishes below display rate.
const TELEMETRY_MAX_HZ: f64 = 30.0;

/// Minimum wall-clock interval between selected-agent telemetry requests,
/// derived from [`TELEMETRY_MAX_HZ`].
const TELEMETRY_MIN_INTERVAL: Duration =
    Duration::from_micros((1_000_000.0 / TELEMETRY_MAX_HZ) as u64);

/// Worker loop sleep while running — short enough for fine tick granularity,
/// long enough to avoid a busy-spin between dispatches.
const RUNNING_SLEEP: Duration = Duration::from_millis(1);

/// Worker loop sleep while paused — the worker only needs to stay responsive to
/// commands and drain any trailing readbacks.
const PAUSED_SLEEP: Duration = Duration::from_millis(8);

/// Bounded capacity for the command channel (main → worker). Commands are rare.
const COMMAND_CAPACITY: usize = 64;

/// Bounded capacity for the event channel (worker → main). Snapshots/telemetry
/// are dropped under back-pressure (latest-wins); control events block briefly.
const EVENT_CAPACITY: usize = 256;

/// Clamp a requested tick batch to the remaining generation budget.
///
/// Returns the number of ticks that may be dispatched without overshooting
/// `tick_budget`, together with whether the clamp actually reduced the batch
/// (so the caller can count generation-boundary clamps). A generation ends
/// exactly at its configured `tick_budget`: once `gen_tick` reaches it the
/// remaining budget is zero and no further ticks dispatch. A `tick_budget` of
/// zero means "unlimited" and never clamps.
fn clamp_ticks_to_generation_budget(
    ticks_to_run: u32,
    tick_budget: u64,
    gen_tick: u64,
) -> (u32, bool) {
    if tick_budget == 0 {
        return (ticks_to_run, false);
    }
    let remaining = tick_budget.saturating_sub(gen_tick);
    // A 64-bit remaining budget can exceed `u32`; saturate so the clamp never
    // truncates a large budget down to a small `u32` and stalls the batch.
    let remaining_ticks = u32::try_from(remaining).unwrap_or(u32::MAX);
    if remaining_ticks < ticks_to_run {
        (remaining_ticks, true)
    } else {
        (ticks_to_run, false)
    }
}

/// Clamp a requested tick batch so it ends on the next supervisor sample tick.
///
/// A `sample_ticks` of zero means "no grid" and never clamps. Otherwise the
/// result is at most the distance from `tick` to the next multiple of
/// `sample_ticks`, so a dispatch never crosses a sample it does not end on.
fn clamp_ticks_to_supervisor_sample(ticks_to_run: u32, tick: u64, sample_ticks: u64) -> u32 {
    if sample_ticks == 0 {
        return ticks_to_run;
    }
    let to_next = sample_ticks - tick % sample_ticks;
    let to_next = u32::try_from(to_next).unwrap_or(u32::MAX);
    ticks_to_run.min(to_next)
}

/// Shape a dispatch so every trail-sample boundary falls on a dispatch end or
/// inside a run of whole kernel batches that started on a boundary.
///
/// The GPU samples agent positions at each multiple of `interval` ticks (one
/// kernel batch), and only at the end of a kernel batch, so the sampled path
/// is the same at every speed multiplier only if each dispatch (a) covers
/// whole brain cycles and (b) never straddles a boundary it does not end on.
/// A dispatch that starts on a boundary may run several whole batches; one
/// that starts inside a batch stops at the next boundary. Returns the number
/// of ticks to dispatch, at least `brain_tick_stride` for any `ticks` at
/// least that large.
fn align_dispatch_to_trail_boundary(
    ticks: u32,
    tick: u64,
    interval: u32,
    brain_tick_stride: u32,
) -> u32 {
    let whole_cycles = ticks - ticks % brain_tick_stride;
    let into_interval = u32::try_from(tick % u64::from(interval)).unwrap_or(0);
    if into_interval != 0 {
        return whole_cycles.min(interval - into_interval);
    }
    if whole_cycles >= interval {
        whole_cycles - whole_cycles % interval
    } else {
        whole_cycles
    }
}

/// An owned, CPU-visible snapshot of GPU simulation state.
///
/// Owns its vectors so it can cross the worker → main channel without borrowing
/// the kernel's caches. `physics` is `PHYS_STRIDE` floats per agent; `food` is
/// `FOOD_STATE_STRIDE` floats per food item (empty when the world has no food).
pub struct StateSnapshot {
    pub physics: Vec<f32>,
    pub food: Vec<f32>,
    /// Every trail sample still held in the GPU ring, oldest first. A snapshot
    /// repeats samples the previous one carried, so applying only the newest
    /// snapshot (latest-wins) still sees every sample the ring retains.
    pub trail_samples: Vec<TrailSample>,
    pub tick: u64,
    pub generation_tick: u64,
    /// Generation epoch of the population this state belongs to (the epoch the
    /// worker adopted at its last start or reset). The main thread drops plain
    /// snapshots whose epoch is not its current one, so state published before
    /// a population reset is never applied to the next generation's agents.
    pub generation_epoch: u64,
}

/// Champion brain state to seed into the next generation on reset.
///
/// The first `champion_slots` agents inherit the champion's exact brain. The
/// rest inherit mutated copies. When the reset freezes steering weights, each
/// following group of `champion_slots` agents shares one mutation of the turn
/// weights and the turn bias, and every other weight stays equal to the
/// champion's.
pub struct InheritedBrain {
    pub champion: AgentBrainState,
    pub mutation_strength: f32,
    pub champion_slots: usize,
}

/// Everything the worker needs to reset to the next generation's population.
pub struct ResetRequest {
    /// World + agent rows for the next generation.
    pub upload: PendingUpload,
    /// Brain config used to reseed brain state before inheritance is applied.
    pub brain_config: BrainConfig,
    /// Generation tick budget for the next generation (0 = unlimited).
    pub tick_budget: u64,
    /// Inherited champion state, or `None` to keep the freshly reseeded state.
    pub inherited: Option<InheritedBrain>,
    /// Whether the worker should resume (unpause) once the reset completes.
    pub resume: bool,
    /// Vary the turn-policy weights across repeat-groups (the steering
    /// genome search) instead of varying `BrainConfig`. Lifetime learning
    /// keeps updating them, but a lineage passes on the weights its champion
    /// was born with, not what it learned.
    pub search_steering_genome: bool,
    /// Generation epoch of the new population; every snapshot the worker
    /// publishes after the reset carries it.
    pub generation_epoch: u64,
}

/// Commands from the main thread to the simulation worker.
pub enum SimCommand {
    /// Pause or resume tick advancement.
    SetPaused(bool),
    /// Set the simulation speed multiplier.
    SetSpeed(u32),
    /// Set the selected agent (by kernel brain index) for telemetry.
    SelectAgent(u32),
    /// Request the brain an agent was born with this generation: the genome
    /// its fitness was earned with, before any lifetime learning.
    RequestBirthState { agent_index: u32, request_id: u64 },
    /// Reset to the next generation's population (boxed: much larger than the
    /// other variants, which would otherwise bloat every `SimCommand`).
    ResetPopulation(Box<ResetRequest>),
    /// Drain pending readbacks and exit the worker thread.
    Shutdown,
}

/// Events from the simulation worker to the main thread.
pub enum SimEvent {
    /// The kernel finished creation and uploads; simulation can proceed.
    KernelReady { agent_count: u32 },
    /// A fresh CPU-visible physics/food snapshot (latest-wins; may be dropped
    /// under back-pressure).
    Snapshot(StateSnapshot),
    /// Selected-agent telemetry for the agent at `agent_index` (brain index).
    Telemetry {
        agent_index: u32,
        telemetry: AgentTelemetry,
    },
    /// The generation tick budget was reached; carries the final state of the
    /// generation. The worker has paused itself and awaits the handoff.
    GenerationBudgetReached(StateSnapshot),
    /// Reply to [`SimCommand::RequestBirthState`].
    BirthState {
        request_id: u64,
        state: Option<AgentBrainState>,
    },
    /// A supervisor sample that carried a boundary, a season flip or the end of
    /// the run.
    // The payload is consumed by the UI consumer, which a later task adds.
    #[allow(dead_code)]
    SupervisorSample(SampleReport),
    /// One telemetry frame per supervisor sample (latest-wins; sent with
    /// `try_send` and dropped under back-pressure).
    #[allow(dead_code)]
    SupervisorTelemetry(Box<TelemetryFrame>),
    /// A boundary record, delivered as a control event ahead of the
    /// `SupervisorSample` of the same sample.
    #[allow(dead_code)]
    SupervisorBoundary(Box<BoundaryRecord>),
    /// Human-readable worker log line.
    Log(String),
    /// Worker error (e.g. no GPU adapter); the worker may be degraded.
    Error(String),
}

/// Supervisor-mode setup: the run configuration and optional store path.
#[allow(dead_code)] // Constructed by the supervisor entry point and tests only.
pub struct SupervisorWorkerInit {
    pub config: FullConfig,
    pub db_path: Option<String>,
    /// Restore the run from the checkpoint in `db_path` instead of starting a
    /// new one.
    pub resume: bool,
}

/// Initial configuration handed to [`SimRuntime::start`].
pub struct SimInit {
    pub agent_count: u32,
    pub food_count: usize,
    pub brain_config: BrainConfig,
    pub world_config: WorldConfig,
    pub upload: PendingUpload,
    pub tick_budget: u64,
    pub speed_multiplier: u32,
    pub paused: bool,
    /// Selected agent brain index for telemetry.
    pub selected_agent: u32,
    /// Generation epoch of the initial population (see
    /// [`StateSnapshot::generation_epoch`]).
    pub generation_epoch: u64,
    /// Search the turn policy across repeat-groups (see
    /// [`ResetRequest::search_steering_genome`]). Free play and tests leave
    /// this false.
    pub search_steering_genome: bool,
    /// Agents that share one steering genome. Matches `eval_repeats`.
    pub steering_group_size: usize,
    /// Turn-policy mutation strength for the first generation, before a
    /// champion brain exists. Later generations use [`InheritedBrain`].
    pub steering_mutation_strength: f32,
    /// Supervisor mode: the worker owns one `SupervisorRun` instead of
    /// generations. `None` keeps generation mode.
    pub supervisor: Option<SupervisorWorkerInit>,
}

/// Partition a batch of drained events into the newest current-epoch state
/// snapshot (latest-wins) and the ordered control events.
///
/// Intermediate [`SimEvent::Snapshot`]s are superseded by the most recent one.
/// Snapshots whose `generation_epoch` differs from `current_epoch` are dropped:
/// they describe a population the main thread has already evaluated (published
/// while the worker sat paused at the budget, or in flight when a reset was
/// sent), and applying them would feed the previous generation's positions,
/// counters and cumulative food into the next generation. Every other event —
/// including [`SimEvent::GenerationBudgetReached`], whose snapshot is the
/// authoritative end-of-generation state and which the worker emits exactly
/// once per population — is returned in arrival order so the consumer
/// processes them deterministically.
pub fn partition_events(
    events: Vec<SimEvent>,
    current_epoch: u64,
) -> (Option<StateSnapshot>, Vec<SimEvent>) {
    let mut latest_snapshot = None;
    let mut control = Vec::new();
    for event in events {
        match event {
            SimEvent::Snapshot(snapshot) if snapshot.generation_epoch == current_epoch => {
                latest_snapshot = Some(snapshot);
            }
            SimEvent::Snapshot(_) => {}
            other => control.push(other),
        }
    }
    (latest_snapshot, control)
}

/// Handle to the simulation worker thread.
///
/// Owns the command sender and event receiver; joins the worker in `Drop` after
/// requesting shutdown so the background thread always terminates deterministically.
pub struct SimRuntime {
    // Both channel ends are `Option` so `Drop` can disconnect them *before*
    // joining the worker (see the `Drop` impl for why that ordering matters).
    command_tx: Option<SyncSender<SimCommand>>,
    event_rx: Option<Receiver<SimEvent>>,
    join_handle: Option<JoinHandle<()>>,
}

impl SimRuntime {
    /// Spawn the worker thread for the given initial configuration.
    pub fn start(init: SimInit) -> Self {
        let (command_tx, command_rx) = sync_channel::<SimCommand>(COMMAND_CAPACITY);
        let (event_tx, event_rx) = sync_channel::<SimEvent>(EVENT_CAPACITY);
        let join_handle = std::thread::Builder::new()
            .name("xagent-sim-worker".into())
            .spawn(move || run_worker(init, &command_rx, &event_tx))
            .expect("failed to spawn sim worker thread");
        Self {
            command_tx: Some(command_tx),
            event_rx: Some(event_rx),
            join_handle: Some(join_handle),
        }
    }

    /// Send a command to the worker. Dropped silently if the worker has exited.
    pub fn send(&self, command: SimCommand) {
        if let Some(command_tx) = &self.command_tx {
            let _ = command_tx.send(command);
        }
    }

    /// Drain all currently pending events in arrival order (non-blocking).
    pub fn drain_events(&self) -> Vec<SimEvent> {
        let mut events = Vec::new();
        if let Some(event_rx) = &self.event_rx {
            while let Ok(event) = event_rx.try_recv() {
                events.push(event);
            }
        }
        events
    }
}

impl Drop for SimRuntime {
    fn drop(&mut self) {
        // Deterministic teardown that cannot hang `join()`. A worker blocked on
        // a full event channel mid-`send` never reaches the command-drain loop,
        // so a `Shutdown` command alone is not enough. Disconnect *both* channel
        // ends first: dropping `event_rx` makes the blocked `send` return an
        // error so the worker proceeds, and dropping `command_tx` makes the
        // worker's `recv`/`try_recv` observe a closed channel and exit. The
        // best-effort `Shutdown` lets a non-blocked worker exit one iteration
        // sooner.
        if let Some(command_tx) = &self.command_tx {
            let _ = command_tx.try_send(SimCommand::Shutdown);
        }
        self.event_rx = None;
        self.command_tx = None;
        if let Some(handle) = self.join_handle.take() {
            if let Err(panic) = handle.join() {
                log::error!("[SIM] worker thread panicked: {panic:?}");
            }
        }
    }
}

/// Worker-side instrumentation, the simulation half of the runtime-decoupling
/// baseline (the render half — frames rendered — lives on the main thread).
#[derive(Default)]
struct WorkerCounters {
    sim_ticks: u64,
    dispatch_calls: u64,
    state_snapshot_requests: u64,
    state_snapshots_collected: u64,
    telemetry_requests: u64,
    telemetry_snapshots_collected: u64,
    generation_clamps: u64,
}

/// Owns the GPU kernel and all simulation-cadence scheduling state.
struct Worker {
    kernel: GpuKernel,
    brain_config: BrainConfig,
    world_config: WorldConfig,

    paused: bool,
    speed_multiplier: u32,
    selected_agent: u32,

    tick: u64,
    gen_tick: u64,
    /// Epoch stamped on every published snapshot; adopted from `SimInit` and
    /// from each `ResetRequest`.
    generation_epoch: u64,
    tick_budget: u64,
    /// Set once `GenerationBudgetReached` has been emitted for the current
    /// generation, cleared on reset, so the boundary event fires exactly once.
    budget_reached: bool,

    sim_accumulator: f64,
    last_frame: Instant,

    last_state_snapshot_request: Option<Instant>,
    last_telemetry_request: Option<Instant>,
    last_telemetry_agent: Option<u32>,

    counters: WorkerCounters,
    last_counters_log: Instant,
    steering_group_size: usize,
    steering_mutation_strength: f32,
    /// Each agent's brain as written at the start of the current generation,
    /// indexed by brain index. What a champion passes on.
    birth_states: Vec<AgentBrainState>,
    /// The supervisor driver; `None` in generation mode or after it ended.
    supervisor: Option<SupervisorRun>,
    /// Whether the worker was built for supervisor mode.
    supervisor_mode: bool,
    /// Control events queued for blocking delivery on the next step.
    supervisor_events: Vec<SimEvent>,
    /// Newest telemetry frame awaiting a non-blocking send; a newer frame
    /// replaces an older one.
    pending_telemetry: Option<Box<TelemetryFrame>>,
    /// Wall-clock instant and tick of the previous supervisor sample.
    supervisor_sample_clock: Option<(Instant, u64)>,
    /// Persistent tick-window capture; `None` without a supervisor database or
    /// after a store failure.
    tick_window_store: Option<TickWindowStore>,
}

/// Start the supervisor run on `kernel`, opening the store when configured.
fn start_supervisor(
    kernel: &mut GpuKernel,
    init: &SupervisorWorkerInit,
    counts: (u32, usize),
) -> Result<(SupervisorRun, u64, Option<SampleReport>), String> {
    if init.resume {
        let (run, tick, report) = resume_supervisor(kernel, init, counts)?;
        return Ok((run, tick, Some(report)));
    }
    let store = match init.db_path.as_deref() {
        Some(path) => Some(
            SupervisorStore::create(path, &init.config)
                .map_err(|e| format!("supervisor store: {e:?}"))?,
        ),
        None => None,
    };
    SupervisorRun::start(kernel, &init.config, store, false)
        .map(|run| (run, 0, None))
        .map_err(|e| format!("supervisor start: {e}"))
}

/// Restore the run from the checkpoint in `init.db_path` and replay its
/// checkpoint sample. Every refusal happens before any store write.
fn resume_supervisor(
    kernel: &mut GpuKernel,
    init: &SupervisorWorkerInit,
    (agent_count, init_food_count): (u32, usize),
) -> Result<(SupervisorRun, u64, SampleReport), String> {
    let path = init
        .db_path
        .as_deref()
        .ok_or_else(|| "checkpoint resume needs a database path".to_string())?;
    let point = load_resume_point(path).map_err(|e| e.to_string())?;
    let stored = serde_json::to_string(&point.config).map_err(|e| e.to_string())?;
    let requested = serde_json::to_string(&init.config).map_err(|e| e.to_string())?;
    if stored != requested {
        return Err(CheckpointError::ConfigMismatch.to_string());
    }
    let food_count = xagent_sandbox::world::WorldState::new(point.config.world.clone())
        .food_items
        .len();
    if usize::try_from(agent_count).ok() != Some(point.config.governor.population_size)
        || init_food_count != food_count
    {
        return Err("checkpoint kernel counts differ from worker init".to_string());
    }
    let store = SupervisorStore::reopen_for_resume(path, point.run_id, point.tick)
        .map_err(|e| format!("supervisor store: {e:?}"))?;
    let mut run = SupervisorRun::resume(
        kernel,
        &init.config,
        Some(store),
        point.state,
        &point.kernel,
    )
    .map_err(|e| format!("supervisor resume: {e}"))?;
    let report = run
        .on_sample(kernel, point.tick, &point.physics, None)
        .map_err(|e| format!("supervisor resume: {e}"))?;
    Ok((run, point.tick, report))
}

/// Birth brains of every slot of `run`, indexed by slot.
fn supervisor_birth_states(
    run: &SupervisorRun,
    population: u32,
) -> Result<Vec<AgentBrainState>, String> {
    (0..usize::try_from(population).unwrap_or(0))
        .map(|slot| {
            run.birth_state(slot)
                .cloned()
                .map_err(|e| format!("supervisor birth state {slot}: {e}"))
        })
        .collect()
}

/// Read back every agent's brain before the generation's first tick
/// (blocking; the worker is between generations). Inheritance passes these
/// on, so offspring get a champion's genome rather than what it learned in
/// its life.
fn read_birth_states(kernel: &GpuKernel) -> Vec<AgentBrainState> {
    (0..kernel.agent_count())
        .map(|index| kernel.read_agent_state(index))
        .collect()
}

/// Whether an inherited champion brain matches the layout `brain_config`
/// builds, so it can be written without misplacing offsets. A stored brain can
/// disagree when the run's vision or retina settings changed since it was
/// captured; it is then skipped (and logged) and the agents keep fresh brains.
fn inherited_brain_fits(champion: &AgentBrainState, brain_config: &BrainConfig) -> bool {
    let brain_stride = BrainLayout::from_config(brain_config).brain_stride;
    let fits =
        champion.brain_state.len() == brain_stride && champion.patterns.len() == PATTERN_STRIDE;
    if !fits {
        log::warn!(
            "[SIM] inherited brain skipped: {} brain / {} pattern values, layout expects {} / {}",
            champion.brain_state.len(),
            champion.patterns.len(),
            brain_stride,
            PATTERN_STRIDE
        );
    }
    fits
}

/// Patch per-agent heritable config tail slots into the GPU kernel.
/// Called after world/agent uploads and after inherited brain states are written,
/// because write_agent_state overwrites the entire brain-state buffer.
fn patch_agent_configs(kernel: &GpuKernel, configs: &[BrainConfig]) {
    for (index, config) in configs.iter().enumerate() {
        let Ok(agent_index) = u32::try_from(index) else {
            break;
        };
        if agent_index >= kernel.agent_count() {
            break;
        }
        kernel.write_agent_heritable_config(agent_index, config);
    }
}

impl Worker {
    /// Create the kernel and upload the initial world + agents.
    fn new(init: SimInit) -> Self {
        let mut kernel = GpuKernel::new(
            init.agent_count,
            init.food_count,
            &init.brain_config,
            &init.world_config,
        );
        let supervisor_mode = init.supervisor.is_some();
        let mut paused = init.paused;
        let mut tick_budget = init.tick_budget;
        let mut supervisor = None;
        let mut supervisor_events = Vec::new();
        let mut tick_window_store: Option<TickWindowStore> = None;
        let birth_states;
        let mut initial_tick = 0;
        if let Some(sup_init) = init.supervisor.as_ref() {
            tick_budget = 0;
            match start_supervisor(&mut kernel, sup_init, (init.agent_count, init.food_count)) {
                Ok((run, start_tick, replay)) => {
                    match supervisor_birth_states(&run, kernel.agent_count()) {
                        Ok(states) => {
                            birth_states = states;
                            supervisor = Some(run);
                            initial_tick = start_tick;
                            if let Some(path) = sup_init.db_path.as_deref() {
                                match TickWindowStore::open_for_latest_supervisor_run(
                                    path,
                                    TickWindowConfig::default(),
                                ) {
                                    Ok(store) => tick_window_store = Some(store),
                                    Err(error) => {
                                        let line = format!("tick window disabled: open: {error}");
                                        supervisor_events.push(SimEvent::Log(line));
                                    }
                                }
                            }
                            if let Some(report) = replay {
                                if report.boundary.is_some()
                                    || report.season.is_some()
                                    || report.finished
                                {
                                    supervisor_events.push(SimEvent::SupervisorSample(report));
                                }
                            }
                        }
                        Err(message) => {
                            birth_states = Vec::new();
                            paused = true;
                            supervisor_events.push(SimEvent::Error(message));
                        }
                    }
                }
                Err(message) => {
                    birth_states = Vec::new();
                    paused = true;
                    supervisor_events.push(SimEvent::Error(message));
                }
            }
        } else {
            kernel.upload_world(
                &init.upload.heights,
                &init.upload.biomes,
                &init.upload.food_pos,
                &init.upload.food_consumed,
                &init.upload.food_timers,
            );
            kernel.upload_agents(&init.upload.agent_data);
            if init.search_steering_genome {
                let mut rng = rand::rng();
                let birth = fresh_brain_state(&init.brain_config, &mut rng);
                let count = usize::try_from(kernel.agent_count()).unwrap_or(0);
                let genomes = steering_population(
                    &birth,
                    count,
                    init.steering_group_size,
                    init.steering_mutation_strength,
                    &mut rng,
                );
                kernel.batch_write_agent_states(count, |index| genomes[index].clone());
            }
            patch_agent_configs(&kernel, &init.upload.agent_configs);
            birth_states = read_birth_states(&kernel);
        }
        Self {
            kernel,
            brain_config: init.brain_config,
            world_config: init.world_config,
            paused,
            speed_multiplier: init.speed_multiplier,
            selected_agent: init.selected_agent,
            tick: initial_tick,
            gen_tick: 0,
            generation_epoch: init.generation_epoch,
            tick_budget,
            budget_reached: false,
            sim_accumulator: 0.0,
            last_frame: Instant::now(),
            last_state_snapshot_request: None,
            last_telemetry_request: None,
            last_telemetry_agent: None,
            counters: WorkerCounters::default(),
            last_counters_log: Instant::now(),
            steering_group_size: init.steering_group_size.max(1),
            steering_mutation_strength: init.steering_mutation_strength,
            birth_states,
            supervisor,
            supervisor_mode,
            supervisor_events,
            pending_telemetry: None,
            supervisor_sample_clock: None,
            tick_window_store,
        }
    }

    /// Apply a command. Returns `true` if the worker should exit.
    fn handle_command(&mut self, command: SimCommand, event_tx: &SyncSender<SimEvent>) -> bool {
        match command {
            SimCommand::SetPaused(paused) => {
                self.paused = paused;
                // Resuming clears the stale wall-clock delta so a long pause
                // does not dump a huge catch-up batch on the next step.
                self.last_frame = Instant::now();
            }
            SimCommand::SetSpeed(speed) => self.speed_multiplier = speed.max(1),
            SimCommand::SelectAgent(index) => self.selected_agent = index,
            SimCommand::RequestBirthState {
                agent_index,
                request_id,
            } => {
                let state = usize::try_from(agent_index)
                    .ok()
                    .and_then(|index| self.birth_states.get(index))
                    .cloned();
                let _ = event_tx.send(SimEvent::BirthState { request_id, state });
            }
            SimCommand::ResetPopulation(_) if self.supervisor_mode => {
                log::warn!("[SIM] ResetPopulation ignored in supervisor mode");
            }
            SimCommand::ResetPopulation(request) => {
                let resume = request.resume;
                self.reset_population(*request);
                let _ = event_tx.send(SimEvent::Log(format!(
                    "[SIM] reset to {} agents ({})",
                    self.kernel.agent_count(),
                    if resume { "running" } else { "paused" },
                )));
            }
            SimCommand::Shutdown => return true,
        }
        false
    }

    /// Reset the kernel to the next generation's population, applying inherited
    /// champion/mutant brain states, then optionally resume.
    fn reset_population(&mut self, request: ResetRequest) {
        let next_agent_count = u32::try_from(request.upload.agent_data.len()).unwrap_or(u32::MAX);
        self.brain_config = request.brain_config;
        self.generation_epoch = request.generation_epoch;

        if self.kernel.agent_count() == next_agent_count {
            // Population size unchanged — reseed in place. Spin on the
            // non-blocking reset until the staging buffers drain, draining
            // readbacks each iteration. Bounded so a stuck slot can't hang the
            // worker forever.
            let mut reset_done = false;
            for _ in 0..10_000 {
                if self.kernel.try_reset_agents(&self.brain_config) {
                    reset_done = true;
                    break;
                }
                self.kernel.try_collect_state_snapshot();
                std::thread::sleep(RUNNING_SLEEP);
            }
            if !reset_done {
                log::error!("[SIM] agent reset did not complete; staging buffers stuck");
            }
        } else {
            // Population size changed — rebuild the kernel from scratch.
            self.kernel = GpuKernel::new(
                next_agent_count,
                request.upload.food_pos.len(),
                &self.brain_config,
                &self.world_config,
            );
            self.kernel.upload_world(
                &request.upload.heights,
                &request.upload.biomes,
                &request.upload.food_pos,
                &request.upload.food_consumed,
                &request.upload.food_timers,
            );
        }
        self.kernel.upload_agents(&request.upload.agent_data);

        let inherited = request
            .inherited
            .filter(|inherited| inherited_brain_fits(&inherited.champion, &self.brain_config));
        if request.search_steering_genome {
            let mut rng = rand::rng();
            let (template, strength, group_size) = if let Some(inherited) = inherited {
                (
                    inherited.champion,
                    inherited.mutation_strength,
                    inherited.champion_slots,
                )
            } else {
                (
                    fresh_brain_state(&self.brain_config, &mut rng),
                    self.steering_mutation_strength,
                    self.steering_group_size,
                )
            };
            let count = usize::try_from(self.kernel.agent_count()).unwrap_or(0);
            let genomes = steering_population(&template, count, group_size, strength, &mut rng);
            self.kernel
                .batch_write_agent_states(count, |index| genomes[index].clone());
        } else if let Some(inherited) = inherited {
            let count = usize::try_from(self.kernel.agent_count()).unwrap_or(0);
            let champion = inherited.champion;
            let strength = inherited.mutation_strength;
            let champion_slots = inherited.champion_slots;
            self.kernel.batch_write_agent_states(count, |index| {
                if index < champion_slots {
                    champion.clone()
                } else {
                    mutate_brain_state(&champion, strength)
                }
            });
        }
        patch_agent_configs(&self.kernel, &request.upload.agent_configs);
        self.birth_states = read_birth_states(&self.kernel);

        self.tick_budget = request.tick_budget;
        self.tick = 0;
        self.gen_tick = 0;
        self.budget_reached = false;
        self.sim_accumulator = 0.0;
        self.last_state_snapshot_request = None;
        self.last_frame = Instant::now();
        self.paused = !request.resume;
    }

    /// One scheduler iteration: collect/publish readbacks, request the next
    /// state/telemetry samples on their own cadence, then advance compute
    /// (unless paused) and enforce the generation budget.
    ///
    /// Returns `true` if a compute dispatch happened, so the caller can skip the
    /// idle sleep and keep draining a backlog at GPU speed.
    fn step(&mut self, event_tx: &SyncSender<SimEvent>) -> bool {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f64().min(0.05);
        self.last_frame = now;

        // Collect + publish readbacks every iteration so map_async callbacks
        // progress and the newest state reaches the UI even between dispatches.
        if self.kernel.try_collect_state_snapshot() {
            self.counters.state_snapshots_collected += 1;
            self.publish_snapshot(event_tx);
        }
        if let Some((agent_index, telemetry)) = self.kernel.try_collect_telemetry() {
            // Label the event with the agent the readback was *requested for*,
            // not the currently-selected agent: a readback that completes after
            // the selection changed would otherwise be applied to the wrong
            // agent. The main thread drops telemetry whose index does not match
            // its selection.
            self.counters.telemetry_snapshots_collected += 1;
            let _ = event_tx.try_send(SimEvent::Telemetry {
                agent_index,
                telemetry,
            });
        }

        // Publication is scheduled independently of dispatch: request state at
        // 60 Hz and telemetry at 30 Hz so the UI keeps updating even while the
        // GPU is busy with a long dispatch. Telemetry also fires immediately on
        // a selection change, even while paused.
        self.maybe_request_state_snapshot();
        self.maybe_request_telemetry();

        self.maybe_log_counters();

        // Control events are never dropped: block until delivered.
        for event in self.supervisor_events.drain(..) {
            let _ = event_tx.send(event);
        }
        // Telemetry is latest-wins: never block, drop when the channel is full.
        if let Some(frame) = self.pending_telemetry.take() {
            let _ = event_tx.try_send(SimEvent::SupervisorTelemetry(frame));
        }

        if self.paused {
            return false;
        }
        let dispatched = self.advance_compute(dt);
        if dispatched {
            self.check_generation_budget(event_tx);
        }
        dispatched
    }

    /// Accumulate wall time and dispatch one clamped kernel-batch when due.
    /// Returns `true` if a dispatch was submitted.
    ///
    /// Each dispatch is capped to a single kernel-batch
    /// (`vision_stride * brain_tick_stride` ticks) so the GPU publishes fresh
    /// state frequently (smooth display) instead of advancing in one long block,
    /// and a backlog drains across loop iterations rather than in one giant
    /// submit. A kernel-batch is the unit that keeps the vision-stride cadence
    /// intact, so once enough time has accumulated for a full batch, perception
    /// is unchanged. At low speed (less than one batch accumulated) the dispatch
    /// is a shorter remainder batch — the same partial-batch decomposition the
    /// kernel has always produced at sub-batch tick counts.
    fn advance_compute(&mut self, dt: f64) -> bool {
        if self.supervisor_mode && self.supervisor.is_none() {
            return false;
        }
        let sim_delta_time = SIM_DT as f64;
        self.sim_accumulator += dt * self.speed_multiplier as f64;

        let min_dispatch = self.kernel.brain_tick_stride();

        // Hand `dispatch_ticks` up to `MAX_FUSED_BATCHES` kernel-batches so a
        // high-speed backlog fuses into one submit instead of paying the
        // per-100-tick submit tax. The generation-budget clamp below stays
        // UPSTREAM of `dispatch_ticks`, so a fused batch still stops exactly at
        // `tick_budget`. At low speed `raw_ticks` stays small (the accumulator
        // only fills at `dt * speed`), so a single batch still dispatches and
        // interactive cadence is unchanged.
        let dispatch_cap = self
            .kernel
            .kernel_batch_size()
            .saturating_mul(xagent_brain::MAX_FUSED_BATCHES)
            .max(min_dispatch);

        // Raise the accumulator cap in lockstep so it can actually fill the
        // wider dispatch cap at high multipliers; `dispatch_cap` already exceeds
        // `min_dispatch`, preserving the original low-speed floor.
        let max_accumulator = sim_delta_time
            * (self.speed_multiplier as f64 * 3.0)
                .max(dispatch_cap as f64)
                .max(min_dispatch as f64 + 2.0);
        self.sim_accumulator = self.sim_accumulator.min(max_accumulator);

        let raw_ticks = ((self.sim_accumulator / sim_delta_time) as u32).min(dispatch_cap);
        let mut ticks_to_run = if raw_ticks >= min_dispatch {
            align_dispatch_to_trail_boundary(
                raw_ticks,
                self.tick,
                self.kernel.kernel_batch_size(),
                min_dispatch,
            )
        } else {
            0
        };

        let (clamped, did_clamp) =
            clamp_ticks_to_generation_budget(ticks_to_run, self.tick_budget, self.gen_tick);
        ticks_to_run = clamped;
        if did_clamp {
            self.counters.generation_clamps += 1;
        }
        if self.supervisor.is_some() {
            ticks_to_run =
                clamp_ticks_to_supervisor_sample(ticks_to_run, self.tick, SUPERVISOR_SAMPLE_TICKS);
        }

        if ticks_to_run == 0 || self.kernel.dispatch_backlog_full() {
            return false;
        }

        self.kernel.dispatch_ticks(self.tick, ticks_to_run);
        self.counters.dispatch_calls += 1;
        self.counters.sim_ticks += u64::from(ticks_to_run);

        self.sim_accumulator -= ticks_to_run as f64 * sim_delta_time;
        self.tick += u64::from(ticks_to_run);
        self.gen_tick = self.gen_tick.saturating_add(u64::from(ticks_to_run));
        self.on_supervisor_sample();
        true
    }

    /// Log `reason` once and stop capturing tick windows; the run continues.
    fn disable_tick_window(&mut self, reason: String) {
        self.supervisor_events
            .push(SimEvent::Log(format!("tick window disabled: {reason}")));
        self.tick_window_store = None;
    }

    /// Record a pin at the current tick, disabling capture on failure.
    fn record_tick_window_pin(&mut self, kind: PinKind) {
        let tick = self.tick;
        if let Some(store) = self.tick_window_store.as_mut() {
            if let Err(error) = store.record_pin(Pin { tick, kind }) {
                self.disable_tick_window(format!("pin failed: {error}"));
            }
        }
    }

    /// Hand the supervisor its sample when the tick reaches the next sample
    /// tick. Fails closed: an error or the end of the run drops the run and
    /// pauses, so nothing dispatches afterwards.
    fn on_supervisor_sample(&mut self) {
        let Some(mut run) = self.supervisor.take() else {
            return;
        };
        if self.tick != run.next_sample_tick() {
            self.supervisor = Some(run);
            return;
        }
        let physics = if run.checkpoint_due(self.tick) {
            let checkpoint = self.kernel.read_checkpoint_blocking();
            if let Err(error) = run.write_checkpoint(self.tick, &checkpoint) {
                self.supervisor_events.push(SimEvent::Error(format!(
                    "supervisor sample failed: {error}"
                )));
                self.paused = true;
                return;
            }
            checkpoint.physics().to_vec()
        } else {
            self.kernel.read_full_state_blocking().to_vec()
        };
        let food = self.kernel.cached_food_state().map(<[f32]>::to_vec);
        if let Some((at, tick0)) = self.supervisor_sample_clock {
            let elapsed = at.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                run.note_ticks_per_second((self.tick - tick0) as f64 / elapsed);
            }
        }
        self.supervisor_sample_clock = Some((Instant::now(), self.tick));
        match run.on_sample(&mut self.kernel, self.tick, &physics, food.as_deref()) {
            Ok(report) => {
                if let Some(slot) = report.boundary.as_ref().and_then(|b| b.slot) {
                    match run.birth_state(slot) {
                        Ok(state) => {
                            if let Some(entry) = self.birth_states.get_mut(slot) {
                                *entry = state.clone();
                            }
                        }
                        Err(error) => {
                            log::error!("[SIM] supervisor birth state {slot} failed: {error}");
                        }
                    }
                }
                if let Some(slot) = report.boundary.as_ref().and_then(|b| b.slot) {
                    self.record_tick_window_pin(PinKind::Replacement { slot });
                }
                if report.season.is_some() {
                    self.record_tick_window_pin(PinKind::SeasonFlip);
                }
                let finished = report.finished;
                if let Some(frame) = run.take_frame() {
                    if let Some(record) = frame.boundary.as_ref() {
                        self.supervisor_events
                            .push(SimEvent::SupervisorBoundary(Box::new(record.clone())));
                    }
                    self.pending_telemetry = Some(Box::new(frame));
                }
                if report.boundary.is_some() || report.season.is_some() || finished {
                    self.supervisor_events
                        .push(SimEvent::SupervisorSample(report));
                }
                if finished {
                    if let Err(error) = run.finish() {
                        log::error!("[SIM] supervisor finish failed: {error}");
                    }
                    self.paused = true;
                } else {
                    self.supervisor = Some(run);
                }
            }
            Err(error) => {
                self.supervisor_events.push(SimEvent::Error(format!(
                    "supervisor sample failed: {error}"
                )));
                self.paused = true;
            }
        }
    }

    /// Request a CPU-visible state snapshot at most `STATE_SNAPSHOT_MAX_HZ`.
    ///
    /// The throttle is consumed only when a staging slot was free (a copy was
    /// submitted), so a dropped request retries next iteration. Decoupled from
    /// dispatch so display keeps refreshing while the GPU grinds a long batch.
    fn maybe_request_state_snapshot(&mut self) {
        let due = self
            .last_state_snapshot_request
            .map_or(true, |t| t.elapsed() >= STATE_SNAPSHOT_MIN_INTERVAL);
        if !due {
            return;
        }
        self.counters.state_snapshot_requests += 1;
        if self.kernel.request_state_snapshot() {
            self.last_state_snapshot_request = Some(Instant::now());
        }
    }

    /// Emit `GenerationBudgetReached` exactly once when the budget is reached,
    /// then pause so the handoff can proceed.
    fn check_generation_budget(&mut self, event_tx: &SyncSender<SimEvent>) {
        if self.tick_budget == 0 || self.budget_reached || self.gen_tick < self.tick_budget {
            return;
        }
        let snapshot = self.force_snapshot();
        // Control event: deliver even under snapshot back-pressure.
        let _ = event_tx.send(SimEvent::GenerationBudgetReached(snapshot));
        self.paused = true;
        self.budget_reached = true;
    }

    /// Read the exact end-of-generation state synchronously.
    ///
    /// The generation boundary drives fitness evaluation and champion selection,
    /// so it must publish the precise final physics — not whatever the async
    /// staging ring happens to have collected, which can be several ticks stale
    /// or a not-yet-final slot. A blocking readback is appropriate here: it runs
    /// once per generation while the worker is paused. Food state is display-only
    /// at the boundary (it is not an input to fitness), so the latest cached food
    /// is sufficient.
    fn force_snapshot(&mut self) -> StateSnapshot {
        let physics = self.kernel.read_full_state_blocking().to_vec();
        let food = self
            .kernel
            .cached_food_state()
            .map(<[f32]>::to_vec)
            .unwrap_or_default();
        StateSnapshot {
            physics,
            food,
            trail_samples: self.kernel.collected_trail_samples(),
            tick: self.tick,
            generation_tick: self.gen_tick,
            generation_epoch: self.generation_epoch,
        }
    }

    /// Build an owned snapshot from the kernel's latest collected caches.
    fn build_snapshot(&self) -> StateSnapshot {
        let food = self
            .kernel
            .cached_food_state()
            .map(<[f32]>::to_vec)
            .unwrap_or_default();
        StateSnapshot {
            physics: self.kernel.cached_state().to_vec(),
            food,
            trail_samples: self.kernel.collected_trail_samples(),
            tick: self.tick,
            generation_tick: self.gen_tick,
            generation_epoch: self.generation_epoch,
        }
    }

    /// Publish the latest collected snapshot, dropping it under back-pressure
    /// (latest-wins): the next snapshot carries newer state, so a dropped one is
    /// never missed. The generation boundary delivers its final state through
    /// the always-sent [`SimEvent::GenerationBudgetReached`] instead.
    fn publish_snapshot(&mut self, event_tx: &SyncSender<SimEvent>) {
        let snapshot = self.build_snapshot();
        if self.tick_window_store.is_some() {
            match frame_from_snapshot(self.tick, &snapshot.physics, &snapshot.food) {
                Ok(frame) => {
                    let recorded = self
                        .tick_window_store
                        .as_mut()
                        .map(|store| store.record_frame(&frame));
                    if let Some(Err(error)) = recorded {
                        self.disable_tick_window(format!("frame failed: {error}"));
                    }
                }
                Err(error) => {
                    self.disable_tick_window(format!("frame build failed: {error}"));
                }
            }
        }
        if let Err(TrySendError::Full(_)) = event_tx.try_send(SimEvent::Snapshot(snapshot)) {
            // Channel full: dropping this snapshot is the intended latest-wins
            // back-pressure.
        }
    }

    /// Request selected-agent telemetry on a selection change or at cadence.
    fn maybe_request_telemetry(&mut self) {
        if self.selected_agent >= self.kernel.agent_count() {
            return;
        }
        let selection_changed = self.last_telemetry_agent != Some(self.selected_agent);
        let cadence_due = self
            .last_telemetry_request
            .map_or(true, |t| t.elapsed() >= TELEMETRY_MIN_INTERVAL);
        if selection_changed || cadence_due {
            self.kernel.request_agent_telemetry(self.selected_agent);
            self.counters.telemetry_requests += 1;
            self.last_telemetry_request = Some(Instant::now());
            self.last_telemetry_agent = Some(self.selected_agent);
        }
    }

    /// Emit the worker-side decoupling counters at most once per second.
    fn maybe_log_counters(&mut self) {
        /// Minimum wall-clock interval between worker counter log lines.
        const COUNTER_LOG_INTERVAL: Duration = Duration::from_secs(1);
        if self.last_counters_log.elapsed() < COUNTER_LOG_INTERVAL {
            return;
        }
        self.last_counters_log = Instant::now();
        let counters = &self.counters;
        // `kernel_batch` = vision_stride * brain_tick_stride: the GPU work unit.
        // A value of 1 means every tick re-runs the full global+vision+brain
        // passes (≈10-30x the per-tick cost of the default stride 10), which
        // caps achievable ticks/sec well below the requested multiplier.
        log::debug!(
            "[SIM-WORKER] sim_ticks={} dispatch_calls={} state_requests={} \
             state_collected={} telemetry_requests={} telemetry_collected={} \
             generation_clamps={} tick={} gen_tick={}/{} \
             brain_tick_stride={} kernel_batch={}",
            counters.sim_ticks,
            counters.dispatch_calls,
            counters.state_snapshot_requests,
            counters.state_snapshots_collected,
            counters.telemetry_requests,
            counters.telemetry_snapshots_collected,
            counters.generation_clamps,
            self.tick,
            self.gen_tick,
            self.tick_budget,
            self.kernel.brain_tick_stride(),
            self.kernel.kernel_batch_size(),
        );

        // Per-batch throughput probe: submit-return vs GPU-complete wall time
        // and the submit/batch fusion ratio. The GPU-complete column is
        // non-zero only under `XAGENT_PROBE_GPU_WAIT=1`.
        let probe_batches = self.kernel.probe_kernel_batches();
        let probe_submits = self.kernel.probe_submit_count();
        let submit_nanos = self.kernel.probe_submit_return_nanos();
        let complete_nanos = self.kernel.probe_gpu_complete_nanos();
        let per_batch = |total: u64| total.checked_div(probe_batches).unwrap_or(0);
        log::debug!(
            "[SIM-PROBE] kernel_batches={} submits={} submit_return_ns={} \
             submit_return_ns_per_batch={} gpu_complete_ns={} \
             gpu_complete_ns_per_batch={}",
            probe_batches,
            probe_submits,
            submit_nanos,
            per_batch(submit_nanos),
            complete_nanos,
            per_batch(complete_nanos),
        );
    }

    /// Sleep duration for this iteration — shorter while actively running.
    fn idle_sleep(&self) -> Duration {
        if self.paused {
            PAUSED_SLEEP
        } else {
            RUNNING_SLEEP
        }
    }
}

/// Worker thread entry point. Owns the kernel for its entire lifetime and exits
/// on [`SimCommand::Shutdown`] or command-channel disconnect.
fn run_worker(init: SimInit, command_rx: &Receiver<SimCommand>, event_tx: &SyncSender<SimEvent>) {
    if !GpuKernel::is_available() {
        let _ = event_tx.send(SimEvent::Error(
            "no GPU adapter available; sim worker idle".into(),
        ));
        // Degraded loop: stay responsive to shutdown / channel close so the
        // runtime still tears down cleanly on headless machines.
        loop {
            match command_rx.recv() {
                Ok(SimCommand::Shutdown) | Err(_) => return,
                Ok(_) => {}
            }
        }
    }

    let mut worker = Worker::new(init);
    let _ = event_tx.send(SimEvent::KernelReady {
        agent_count: worker.kernel.agent_count(),
    });

    loop {
        loop {
            match command_rx.try_recv() {
                Ok(command) => {
                    if worker.handle_command(command, event_tx) {
                        return;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }

        // Only sleep when caught up (or paused). While a dispatch backlog
        // remains, loop immediately so it drains at GPU speed — the kernel's
        // in-flight dispatch bound paces the worker, and snapshots publish as
        // fast as the GPU completes batches.
        let busy = worker.step(event_tx);
        if !busy {
            std::thread::sleep(worker.idle_sleep());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── generation-budget clamp ──────────────────────────────────────

    #[test]
    fn align_keeps_whole_batches_from_a_boundary() {
        // Batch of 100 ticks, brain stride 10, starting on a boundary.
        assert_eq!(align_dispatch_to_trail_boundary(350, 0, 100, 10), 300);
        assert_eq!(align_dispatch_to_trail_boundary(100, 500, 100, 10), 100);
    }

    #[test]
    fn align_runs_sub_batch_dispatches_from_a_boundary_whole() {
        // Fewer ticks than a batch cannot straddle a boundary; round to cycles.
        assert_eq!(align_dispatch_to_trail_boundary(37, 200, 100, 10), 30);
        assert_eq!(align_dispatch_to_trail_boundary(10, 200, 100, 10), 10);
    }

    #[test]
    fn align_stops_at_the_next_boundary_from_inside_a_batch() {
        assert_eq!(align_dispatch_to_trail_boundary(350, 30, 100, 10), 70);
        assert_eq!(align_dispatch_to_trail_boundary(40, 30, 100, 10), 40);
        assert_eq!(align_dispatch_to_trail_boundary(70, 30, 100, 10), 70);
    }

    #[test]
    fn align_reaches_every_boundary_from_any_dispatch_size() {
        // Whatever dispatch sizes the wall clock produces, stepping with the
        // aligned size must land on every multiple of the interval.
        for sizes in [[10_u32, 10], [37, 250], [999, 13]] {
            let mut tick = 0_u64;
            let mut boundaries_seen = 0_u64;
            for step in 0..400 {
                let raw = sizes[step % 2];
                let run = align_dispatch_to_trail_boundary(raw, tick, 100, 10);
                assert!(run >= 10, "dispatch must stay at least one brain cycle");
                let next = tick + u64::from(run);
                // No boundary is jumped over: crossing one means ending on it
                // or starting on one and covering whole batches.
                let crossed = next / 100 - tick / 100;
                if crossed > 0 {
                    assert!(
                        next % 100 == 0,
                        "dispatch {tick}->{next} straddles a boundary"
                    );
                    boundaries_seen += crossed;
                }
                tick = next;
            }
            assert_eq!(boundaries_seen, tick / 100);
        }
    }

    #[test]
    fn clamp_leaves_batch_untouched_with_budget_to_spare() {
        let (clamped, did_clamp) = clamp_ticks_to_generation_budget(64, 1000, 800);
        assert_eq!(clamped, 64);
        assert!(!did_clamp);
    }

    #[test]
    fn clamp_caps_batch_to_remaining_budget() {
        // tick_budget 100, gen_tick 97 → only 3 ticks remain; a 50-tick batch
        // must be clamped to 3 so the generation stops exactly at the budget.
        let (clamped, did_clamp) = clamp_ticks_to_generation_budget(50, 100, 97);
        assert_eq!(clamped, 3);
        assert!(did_clamp);
    }

    #[test]
    fn clamp_to_exact_boundary_yields_zero_and_flags_clamp() {
        let (clamped, did_clamp) = clamp_ticks_to_generation_budget(10, 100, 100);
        assert_eq!(clamped, 0);
        assert!(did_clamp);
    }

    #[test]
    fn clamp_treats_zero_budget_as_unlimited() {
        let (clamped, did_clamp) = clamp_ticks_to_generation_budget(500, 0, 1_000_000);
        assert_eq!(clamped, 500);
        assert!(!did_clamp);
    }

    // ── inherited-brain layout guard ─────────────────────────────────

    #[test]
    fn inherited_brain_fits_only_the_configured_layout() {
        let config = BrainConfig::default();
        let brain_stride = BrainLayout::from_config(&config).brain_stride;

        assert!(inherited_brain_fits(
            &AgentBrainState::new_for(brain_stride),
            &config
        ));
        assert!(!inherited_brain_fits(
            &AgentBrainState::new_for(brain_stride + 1),
            &config
        ));
        let mut short_patterns = AgentBrainState::new_for(brain_stride);
        short_patterns.patterns.pop();
        assert!(!inherited_brain_fits(&short_patterns, &config));
    }

    // ── latest-wins event partitioning ───────────────────────────────

    fn snapshot_with_tick(tick: u64) -> StateSnapshot {
        snapshot_with_tick_and_epoch(tick, 0)
    }

    fn snapshot_with_tick_and_epoch(tick: u64, generation_epoch: u64) -> StateSnapshot {
        StateSnapshot {
            physics: Vec::new(),
            food: Vec::new(),
            trail_samples: Vec::new(),
            tick,
            generation_tick: tick,
            generation_epoch,
        }
    }

    #[test]
    fn partition_events_keeps_only_newest_snapshot() {
        let events = vec![
            SimEvent::Snapshot(snapshot_with_tick(1)),
            SimEvent::Log("a".into()),
            SimEvent::Snapshot(snapshot_with_tick(2)),
            SimEvent::Snapshot(snapshot_with_tick(3)),
        ];
        let (latest, control) = partition_events(events, 0);
        assert_eq!(latest.expect("a snapshot is present").tick, 3);
        // The single non-snapshot event is preserved.
        assert_eq!(control.len(), 1);
        assert!(matches!(control[0], SimEvent::Log(_)));
    }

    #[test]
    fn partition_events_preserves_control_order_and_budget_snapshot() {
        let events = vec![
            SimEvent::KernelReady { agent_count: 4 },
            SimEvent::Snapshot(snapshot_with_tick(10)),
            SimEvent::GenerationBudgetReached(snapshot_with_tick(99)),
            SimEvent::BirthState {
                request_id: 7,
                state: None,
            },
        ];
        let (latest, control) = partition_events(events, 0);
        // The regular snapshot at tick 10 is the newest plain snapshot.
        assert_eq!(latest.expect("a snapshot is present").tick, 10);
        // Control events keep arrival order; the budget snapshot is NOT folded
        // into latest-wins (it is the authoritative end-of-generation state).
        assert_eq!(control.len(), 3);
        assert!(matches!(
            control[0],
            SimEvent::KernelReady { agent_count: 4 }
        ));
        assert!(matches!(
            control[1],
            SimEvent::GenerationBudgetReached(ref s) if s.tick == 99
        ));
        assert!(matches!(
            control[2],
            SimEvent::BirthState { request_id: 7, .. }
        ));
    }

    #[test]
    fn partition_events_drops_snapshots_from_other_epochs() {
        // Epoch 0 is the population already evaluated: its snapshots (even a
        // newer tick) must not win over the current epoch's.
        let events = vec![
            SimEvent::Snapshot(snapshot_with_tick_and_epoch(100_000, 0)),
            SimEvent::Snapshot(snapshot_with_tick_and_epoch(12, 1)),
            SimEvent::Snapshot(snapshot_with_tick_and_epoch(100_000, 0)),
        ];
        let (latest, control) = partition_events(events, 1);
        let latest = latest.expect("the current-epoch snapshot is kept");
        assert_eq!(latest.generation_epoch, 1);
        assert_eq!(latest.tick, 12);
        assert!(control.is_empty());

        let only_stale = vec![SimEvent::Snapshot(snapshot_with_tick_and_epoch(100_000, 0))];
        let (latest, _) = partition_events(only_stale, 1);
        assert!(latest.is_none(), "a stale-epoch snapshot must be dropped");
    }

    #[test]
    fn partition_events_handles_no_snapshot() {
        let events = vec![SimEvent::Log("only logs".into())];
        let (latest, control) = partition_events(events, 0);
        assert!(latest.is_none());
        assert_eq!(control.len(), 1);
    }

    // ── worker orchestration ──────────────────────────────────────────

    /// Build a single-agent world upload from a fresh default world.
    fn test_upload() -> (PendingUpload, usize) {
        use xagent_sandbox::world::WorldState;
        use xagent_shared::WorldConfig;

        let world = WorldState::new(WorldConfig::default());
        let upload = PendingUpload {
            heights: world.terrain.heights.clone(),
            biomes: world.biome_map.grid_as_u32(),
            food_pos: world
                .food_items
                .iter()
                .map(|f| (f.position.x, f.position.y, f.position.z))
                .collect(),
            food_consumed: world.food_items.iter().map(|f| f.consumed).collect(),
            food_timers: world.food_items.iter().map(|f| f.respawn_timer).collect(),
            agent_data: vec![(
                world.safe_spawn_position(),
                100.0,
                100.0,
                BrainConfig::default().memory_capacity,
                BrainConfig::default().processing_slots,
            )],
            agent_configs: vec![BrainConfig::default()],
        };
        (upload, world.food_items.len())
    }

    /// Serialize GPU-heavy tests: the guard is parked in a thread-local and
    /// released when the test thread exits.
    fn hold_gpu_test_lock() {
        static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        thread_local! {
            static HELD: std::cell::RefCell<Option<std::sync::MutexGuard<'static, ()>>> =
                const { std::cell::RefCell::new(None) };
        }
        HELD.with(|held| {
            let mut held = held.borrow_mut();
            if held.is_none() {
                *held = Some(
                    GPU_TEST_LOCK
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                );
            }
        });
    }

    fn test_init(tick_budget: u64, paused: bool, speed_multiplier: u32) -> SimInit {
        hold_gpu_test_lock();
        let (upload, food_count) = test_upload();
        SimInit {
            agent_count: 1,
            food_count,
            brain_config: BrainConfig::default(),
            world_config: WorldConfig::default(),
            upload,
            tick_budget,
            speed_multiplier,
            paused,
            selected_agent: 0,
            generation_epoch: 0,
            search_steering_genome: false,
            steering_group_size: 1,
            steering_mutation_strength: 0.0,
            supervisor: None,
        }
    }

    /// Build a two-agent world upload whose per-agent configs differ only in
    /// `gabor_wavelength` (the first Gabor visual-genome tail slot). Both
    /// agents spawn at the same safe position; the population size is fixed so
    /// `reset_population` takes the in-place reseed path. Returns the upload and
    /// the world's food count (mirrors `test_upload`).
    fn two_agent_upload_with_wavelengths(
        wavelength_a: f32,
        wavelength_b: f32,
    ) -> (PendingUpload, usize) {
        use xagent_sandbox::world::WorldState;
        use xagent_shared::WorldConfig;

        let world = WorldState::new(WorldConfig::default());
        let spawn = world.safe_spawn_position();
        let make_config = |wavelength: f32| BrainConfig {
            gabor_wavelength: wavelength,
            ..BrainConfig::default()
        };
        let config_a = make_config(wavelength_a);
        let config_b = make_config(wavelength_b);
        let agent_row = |config: &BrainConfig| {
            (
                spawn,
                100.0,
                100.0,
                config.memory_capacity,
                config.processing_slots,
            )
        };
        let upload = PendingUpload {
            heights: world.terrain.heights.clone(),
            biomes: world.biome_map.grid_as_u32(),
            food_pos: world
                .food_items
                .iter()
                .map(|f| (f.position.x, f.position.y, f.position.z))
                .collect(),
            food_consumed: world.food_items.iter().map(|f| f.consumed).collect(),
            food_timers: world.food_items.iter().map(|f| f.respawn_timer).collect(),
            agent_data: vec![agent_row(&config_a), agent_row(&config_b)],
            agent_configs: vec![config_a, config_b],
        };
        (upload, world.food_items.len())
    }

    /// Read agent `index`'s `O_GABOR_WAVELENGTH` brain-state tail slot. The tail
    /// base is derived from the read-back length (the dynamic `brain_stride`),
    /// never a hardcoded stride, so this works for any `BrainLayout`.
    fn read_gabor_wavelength(kernel: &GpuKernel, index: u32) -> f32 {
        use xagent_brain::buffers::{
            fixed_tail_base, O_GABOR_WAVELENGTH, O_PREDICTOR_CONTEXT_WEIGHT,
        };

        let state = kernel.read_agent_state(index);
        let tail_base = fixed_tail_base(state.brain_state.len());
        let slot = tail_base + (O_GABOR_WAVELENGTH - O_PREDICTOR_CONTEXT_WEIGHT);
        state.brain_state[slot]
    }

    /// The interactive worker must re-apply each agent's heritable visual genome
    /// after inheritance overwrites the brain-state tail. Inheritance writes the
    /// champion's exact state (carrying the champion's `gabor_wavelength`) into
    /// every champion slot; `patch_agent_configs` then restores each agent's own
    /// config. Without that patch step every agent's vision silently reverts to
    /// the champion's, so this test fails on the un-patched worker and passes
    /// once the patch covers the visual tail slots with per-agent genome values.
    /// GPU-gated.
    #[test]
    fn worker_reset_applies_visual_genome_after_inheritance() {
        if !GpuKernel::is_available() {
            eprintln!("Skipping: no GPU/fallback adapter available");
            return;
        }

        const WAVELENGTH_A: f32 = 3.0;
        const WAVELENGTH_B: f32 = 9.0;

        // Two agents with distinct gabor_wavelength. `Worker::new` runs
        // `patch_agent_configs`, so agent 0's tail already carries WAVELENGTH_A.
        let (upload, food_count) = two_agent_upload_with_wavelengths(WAVELENGTH_A, WAVELENGTH_B);
        let mut worker = Worker::new(SimInit {
            agent_count: 2,
            food_count,
            brain_config: BrainConfig::default(),
            world_config: WorldConfig::default(),
            upload,
            tick_budget: 0,
            speed_multiplier: 1,
            paused: true,
            selected_agent: 0,
            generation_epoch: 0,
            search_steering_genome: false,
            steering_group_size: 1,
            steering_mutation_strength: 0.0,
            supervisor: None,
        });

        // The champion is agent 0's inherited state: it carries WAVELENGTH_A in
        // its tail. Seeding it into *both* champion slots overwrites both tails
        // with WAVELENGTH_A — so if the worker fails to re-patch, agent 1 keeps
        // the champion's WAVELENGTH_A instead of its own WAVELENGTH_B.
        let champion = worker.kernel.read_agent_state(0);
        assert!(
            (read_gabor_wavelength(&worker.kernel, 0) - WAVELENGTH_A).abs() < 1e-4,
            "precondition: Worker::new should have patched agent 0's wavelength"
        );

        let (reset_upload, _) = two_agent_upload_with_wavelengths(WAVELENGTH_A, WAVELENGTH_B);
        worker.reset_population(ResetRequest {
            upload: reset_upload,
            brain_config: BrainConfig::default(),
            tick_budget: 0,
            inherited: Some(InheritedBrain {
                champion,
                mutation_strength: 0.0,
                champion_slots: 2,
            }),
            resume: false,
            generation_epoch: 1,
            search_steering_genome: false,
        });

        // Snapshots published after the reset carry the request's epoch.
        assert_eq!(worker.build_snapshot().generation_epoch, 1);

        let agent_0 = read_gabor_wavelength(&worker.kernel, 0);
        let agent_1 = read_gabor_wavelength(&worker.kernel, 1);
        assert!(
            (agent_0 - WAVELENGTH_A).abs() < 1e-4,
            "agent 0 wavelength {agent_0} != own config {WAVELENGTH_A}"
        );
        assert!(
            (agent_1 - WAVELENGTH_B).abs() < 1e-4,
            "agent 1 wavelength {agent_1} reverted to champion's instead of own \
             config {WAVELENGTH_B} — worker did not re-apply the visual genome \
             after inheritance"
        );
    }

    /// A champion passes on the brain it was born with, not what it learned:
    /// after a stretch of lifetime learning, the worker's reply to
    /// `RequestBirthState` must still equal the state written before the
    /// first tick. GPU-gated.
    #[test]
    fn worker_passes_on_the_birth_brain_not_the_learned_one() {
        if !GpuKernel::is_available() {
            eprintln!("Skipping: no GPU/fallback adapter available");
            return;
        }

        const LIFETIME_TICKS: u32 = 300;
        const REQUEST_ID: u64 = 3;

        let (upload, food_count) = two_agent_upload_with_wavelengths(5.0, 5.0);
        let mut worker = Worker::new(SimInit {
            agent_count: 2,
            food_count,
            brain_config: BrainConfig::default(),
            world_config: WorldConfig::default(),
            upload,
            tick_budget: 0,
            speed_multiplier: 1,
            paused: true,
            selected_agent: 0,
            generation_epoch: 0,
            search_steering_genome: true,
            steering_group_size: 1,
            steering_mutation_strength: 0.1,
            supervisor: None,
        });
        let birth = worker.kernel.read_agent_state(0);
        worker.kernel.dispatch_batch(0, LIFETIME_TICKS);
        let learned = worker.kernel.read_agent_state(0);
        assert_ne!(
            learned.brain_state, birth.brain_state,
            "precondition: {LIFETIME_TICKS} ticks of lifetime learning should change the brain"
        );

        let (event_tx, event_rx) = sync_channel(1);
        worker.handle_command(
            SimCommand::RequestBirthState {
                agent_index: 0,
                request_id: REQUEST_ID,
            },
            &event_tx,
        );
        let Ok(SimEvent::BirthState {
            request_id: REQUEST_ID,
            state: Some(passed_on),
        }) = event_rx.try_recv()
        else {
            panic!("the worker should reply with agent 0's birth state");
        };
        assert!(
            passed_on.brain_state == birth.brain_state && passed_on.patterns == birth.patterns,
            "the champion passed on its learned brain instead of its birth brain"
        );
    }

    /// Drain events through `pick`, returning the first non-`None` mapping or
    /// `None` if nothing matched within ~10 s (generous for a slow GPU).
    fn drain_until<T>(
        runtime: &SimRuntime,
        mut pick: impl FnMut(SimEvent) -> Option<T>,
    ) -> Option<T> {
        for _ in 0..2000 {
            for event in runtime.drain_events() {
                if let Some(value) = pick(event) {
                    return Some(value);
                }
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        None
    }

    /// Starting a runtime and shutting it down must terminate the worker
    /// thread deterministically. On a headless machine the worker takes the
    /// degraded no-GPU path and still honors `Shutdown`; with a GPU it builds a
    /// real kernel and tears it down. Either way `Drop` joins without hanging.
    /// (No GPU required.)
    #[test]
    fn sim_runtime_starts_and_shuts_down() {
        let runtime = SimRuntime::start(test_init(100, true, 1));
        runtime.send(SimCommand::Shutdown);
        // Dropping the runtime joins the worker; reaching the end of the test
        // without hanging proves clean shutdown.
        drop(runtime);
    }

    /// End-to-end generation handoff at the worker boundary: the worker reaches
    /// the tick budget and emits `GenerationBudgetReached`, replies to a champion
    /// `RequestBirthState` while paused, and resumes the next generation on
    /// `ResetPopulation`. Exercises the orchestration the kernel-level tests do
    /// not. GPU-gated.
    #[test]
    fn worker_runs_generation_budget_handoff() {
        if !GpuKernel::is_available() {
            eprintln!("Skipping: no GPU/fallback adapter available");
            return;
        }
        const BUDGET: u64 = 400;
        let runtime = SimRuntime::start(test_init(BUDGET, false, 100));

        // 1. The worker stops exactly at the budget and emits the boundary.
        let budget_tick = drain_until(&runtime, |event| match event {
            SimEvent::GenerationBudgetReached(snapshot) => Some(snapshot.generation_tick),
            _ => None,
        })
        .expect("GenerationBudgetReached should arrive");
        assert!(
            budget_tick >= BUDGET,
            "boundary fired at gen_tick {budget_tick}, expected >= {BUDGET}"
        );

        // 2. Champion brain-state readback while paused.
        runtime.send(SimCommand::RequestBirthState {
            agent_index: 0,
            request_id: 42,
        });
        let champion = drain_until(&runtime, |event| match event {
            SimEvent::BirthState {
                request_id: 42,
                state,
            } => Some(state),
            _ => None,
        })
        .expect("BirthState reply should arrive");
        assert!(champion.is_some(), "champion brain state should read back");

        // 3. Reset to the next generation and resume.
        let (upload, _) = test_upload();
        runtime.send(SimCommand::ResetPopulation(Box::new(ResetRequest {
            upload,
            brain_config: BrainConfig::default(),
            tick_budget: BUDGET,
            inherited: champion.map(|champion| InheritedBrain {
                champion,
                mutation_strength: 0.1,
                champion_slots: 1,
            }),
            resume: true,
            generation_epoch: 1,
            search_steering_genome: false,
        })));

        // The worker logs the reset once it processes the command...
        let reset_logged = drain_until(&runtime, |event| match event {
            SimEvent::Log(message) if message.contains("reset to") => Some(()),
            _ => None,
        });
        assert!(
            reset_logged.is_some(),
            "worker should log the population reset"
        );

        // ...and then resumes dispatch, publishing fresh snapshots again.
        let resumed = drain_until(&runtime, |event| match event {
            SimEvent::Snapshot(_) => Some(()),
            _ => None,
        });
        assert!(
            resumed.is_some(),
            "worker should resume publishing after reset"
        );

        drop(runtime);
    }

    /// Simulation progress is independent of whether the main thread drains
    /// events. With events left undrained (snapshots dropped under
    /// back-pressure, never blocking the worker), the worker keeps advancing
    /// ticks across two windows. This is the redraw-stall guarantee. GPU-gated.
    #[test]
    fn worker_advances_independently_of_event_draining() {
        if !GpuKernel::is_available() {
            eprintln!("Skipping: no GPU/fallback adapter available");
            return;
        }
        // Unlimited budget so the only events are snapshots/telemetry, which use
        // non-blocking sends — the worker cannot stall on a full channel here.
        let runtime = SimRuntime::start(test_init(0, false, 100));

        let max_tick = |runtime: &SimRuntime| {
            runtime
                .drain_events()
                .into_iter()
                .filter_map(|event| match event {
                    SimEvent::Snapshot(snapshot) => Some(snapshot.tick),
                    _ => None,
                })
                .max()
        };

        // Wait until the kernel is built (shader compilation can take a while)
        // so the measurement windows are not skewed by startup latency, then
        // clear the startup backlog.
        let ready = drain_until(&runtime, |event| {
            matches!(event, SimEvent::KernelReady { .. }).then_some(())
        });
        assert!(ready.is_some(), "kernel should become ready");
        let _ = runtime.drain_events();

        // Sample the worker's progress over an undrained window: never drain
        // mid-window, so this proves the worker advances without the main thread
        // consuming events. Poll generously across windows rather than trusting
        // a single fixed wall-clock budget — a software adapter (CI lavapipe) is
        // far slower than a hardware GPU at shader compilation and the first
        // async readback, so a fixed sleep races the first published snapshot.
        // Mirrors the `drain_until` budget used by the sibling GPU tests. Each
        // iteration's pre-sample sleep is itself a real undrained window.
        let advance_past = |minimum: u64| -> Option<u64> {
            for _ in 0..50 {
                std::thread::sleep(Duration::from_millis(200));
                if let Some(tick) = max_tick(&runtime) {
                    if tick > minimum {
                        return Some(tick);
                    }
                }
            }
            None
        };

        // A nonzero tick proves the worker advances without the main thread.
        let first = advance_past(0).expect("worker should advance ticks while undrained");

        // Further progress must follow: the worker is not gated on event
        // consumption (the redraw-stall guarantee), and it never blocked on the
        // (now-drained) snapshot channel.
        let second = advance_past(first).expect("worker should keep advancing while undrained");
        assert!(
            second > first,
            "worker should keep advancing while undrained: {second} !> {first}"
        );

        drop(runtime);
    }

    // ── supervisor mode ───────────────────────────────────────────────

    /// Pure fail-closed rule: `Ok(true)` runs, `Ok(false)` skips, `Err` fails.
    fn supervisor_gpu_guard_decision(
        available: bool,
        required: bool,
    ) -> Result<bool, &'static str> {
        if available {
            Ok(true)
        } else if required {
            Err("XAGENT_REQUIRE_GPU is set but no GPU adapter is available")
        } else {
            Ok(false)
        }
    }

    /// Whether a supervisor GPU test should run; panics when the GPU is
    /// required and missing.
    fn supervisor_gpu_or_skip() -> bool {
        let decision = supervisor_gpu_guard_decision(
            GpuKernel::is_available(),
            std::env::var_os("XAGENT_REQUIRE_GPU").is_some(),
        );
        match decision {
            Ok(true) => {
                hold_gpu_test_lock();
                true
            }
            Ok(false) => {
                eprintln!("Skipping: no GPU/fallback adapter available");
                false
            }
            Err(message) => panic!("{message}"),
        }
    }

    /// Agents in a supervisor test run.
    const SUPERVISOR_TEST_AGENTS: usize = 10;

    fn supervisor_test_config_with_budget(agent_tick_budget: u64) -> FullConfig {
        let mut config = FullConfig::default();
        config.governor.population_size = SUPERVISOR_TEST_AGENTS;
        config.governor.supervisor.boundary_interval_ticks = 1_000;
        config.governor.supervisor.min_age_ticks = 2_000;
        config.governor.supervisor.score_window_ticks = 1_000;
        config.governor.supervisor.season_period_ticks = 3_000;
        config.governor.supervisor.agent_tick_budget = agent_tick_budget;
        config
    }

    fn supervisor_test_config() -> FullConfig {
        supervisor_test_config_with_budget(100_000)
    }

    fn supervisor_init(config: &FullConfig, db_path: Option<String>) -> SimInit {
        let (upload, _) = test_upload();
        let food_count = xagent_sandbox::world::WorldState::new(config.world.clone())
            .food_items
            .len();
        let mut init = test_init(0, false, 1);
        init.agent_count = u32::try_from(config.governor.population_size).expect("fits");
        init.food_count = food_count;
        init.brain_config = config.brain.clone();
        init.world_config = config.world.clone();
        init.upload = upload;
        init.supervisor = Some(SupervisorWorkerInit {
            config: config.clone(),
            db_path,
            resume: false,
        });
        init
    }

    /// Advance with a synthetic wall-clock delta until `target` ticks or the
    /// worker pauses; returns the reports queued for delivery.
    fn supervisor_run_to(worker: &mut Worker, dt: f64, target: u64) -> Vec<SampleReport> {
        let mut reports = Vec::new();
        for _ in 0..1_000_000 {
            if worker.tick >= target || worker.paused {
                break;
            }
            worker.advance_compute(dt);
            for event in worker.supervisor_events.drain(..) {
                if let SimEvent::SupervisorSample(report) = event {
                    reports.push(report);
                }
            }
        }
        reports
    }

    /// Delta that yields roughly `ticks` ticks of accumulation at speed 1.
    fn supervisor_dt(ticks: f64) -> f64 {
        (ticks + 0.01) * f64::from(SIM_DT)
    }

    /// Tick a boundary record was decided at.
    fn boundary_record_tick(record: &BoundaryRecord) -> u64 {
        match &record.detail {
            xagent_sandbox::supervisor::BoundaryDetail::Replaced { tick, .. }
            | xagent_sandbox::supervisor::BoundaryDetail::Idle { tick, .. } => *tick,
        }
    }

    /// Advance until `target` ticks or pause, returning every queued event in
    /// order. When `discard` is set, telemetry frames and boundary events are
    /// dropped as they appear.
    fn supervisor_collect_events(
        worker: &mut Worker,
        dt: f64,
        target: u64,
        discard: bool,
    ) -> Vec<SimEvent> {
        let mut events = Vec::new();
        for _ in 0..1_000_000 {
            if worker.tick >= target || worker.paused {
                break;
            }
            worker.advance_compute(dt);
            if discard {
                worker.pending_telemetry = None;
                worker
                    .supervisor_events
                    .retain(|event| !matches!(event, SimEvent::SupervisorBoundary(_)));
            }
            events.append(&mut worker.supervisor_events);
        }
        events
    }

    #[test]
    fn worker_resume_matches_uninterrupted() {
        if !supervisor_gpu_or_skip() {
            return;
        }
        const TARGET: u64 = 4_000;
        let mut config = supervisor_test_config();
        // Food kinds on so the resumed run also exercises the season state.
        config.world.food_kinds_enabled = true;
        let dt = supervisor_dt(10.0);
        let dir = tempfile::tempdir().expect("tempdir");
        let path_a = dir.path().join("a.db").to_string_lossy().into_owned();
        let path_b = dir.path().join("b.db").to_string_lossy().into_owned();

        let reports_a = {
            let mut worker = Worker::new(supervisor_init(&config, Some(path_a.clone())));
            let reports = supervisor_run_to(&mut worker, dt, TARGET);
            assert_eq!(worker.tick, TARGET);
            reports
        };
        std::fs::copy(&path_a, &path_b).expect("copy database");
        let point = xagent_sandbox::supervisor_run::load_resume_point(&path_b).expect("checkpoint");
        let resume_tick = point.tick;
        assert!(resume_tick > 0);

        let mut init = supervisor_init(&config, Some(path_b));
        if let Some(sup) = init.supervisor.as_mut() {
            sup.resume = true;
        }
        let mut worker = Worker::new(init);
        assert_eq!(worker.tick, resume_tick);
        let mut reports_b: Vec<SampleReport> = worker
            .supervisor_events
            .drain(..)
            .filter_map(|event| match event {
                SimEvent::SupervisorSample(report) => Some(report),
                _ => None,
            })
            .collect();
        reports_b.extend(supervisor_run_to(&mut worker, dt, TARGET));
        assert!(!reports_b.is_empty(), "resumed worker reported nothing");
        let after = |reports: &[SampleReport]| -> Vec<SampleReport> {
            reports
                .iter()
                .filter(|r| r.tick > resume_tick)
                .cloned()
                .collect()
        };
        assert_eq!(after(&reports_a), after(&reports_b));
    }

    #[test]
    fn worker_resume_invalid_checkpoint_errors_and_pauses() {
        if !supervisor_gpu_or_skip() {
            return;
        }
        let mut config = supervisor_test_config();
        config.world.food_kinds_enabled = true;
        let dt = supervisor_dt(10.0);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("c.db").to_string_lossy().into_owned();
        {
            let mut worker = Worker::new(supervisor_init(&config, Some(path.clone())));
            supervisor_run_to(&mut worker, dt, 1_500);
        }
        let corrupted = {
            let db = rusqlite::Connection::open(&path).expect("open");
            let mut payload: Vec<u8> = db
                .query_row("SELECT payload FROM supervisor_checkpoint", [], |row| {
                    row.get(0)
                })
                .expect("payload");
            let middle = payload.len() / 2;
            payload[middle] ^= 0xFF;
            db.execute("UPDATE supervisor_checkpoint SET payload = ?1", [&payload])
                .expect("corrupt");
            payload
        };

        let mut init = supervisor_init(&config, Some(path.clone()));
        if let Some(sup) = init.supervisor.as_mut() {
            sup.resume = true;
        }
        let worker = Worker::new(init);
        let errors: Vec<&String> = worker
            .supervisor_events
            .iter()
            .filter_map(|event| match event {
                SimEvent::Error(text) => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("checkpoint"), "{}", errors[0]);
        assert_eq!(worker.supervisor_events.len(), 1);
        assert!(worker.paused);
        assert!(worker.supervisor.is_none());
        assert_eq!(worker.counters.dispatch_calls, 0);
        assert_eq!(worker.tick, 0);
        let db = rusqlite::Connection::open(&path).expect("open");
        let stored: Vec<u8> = db
            .query_row("SELECT payload FROM supervisor_checkpoint", [], |row| {
                row.get(0)
            })
            .expect("payload");
        assert_eq!(stored, corrupted);

        let mut init = supervisor_init(&config, None);
        if let Some(sup) = init.supervisor.as_mut() {
            sup.resume = true;
        }
        let worker = Worker::new(init);
        assert_eq!(worker.supervisor_events.len(), 1);
        assert!(matches!(worker.supervisor_events[0], SimEvent::Error(_)));
        assert!(worker.paused);
    }

    /// Collect a fresh state snapshot and publish it through the worker.
    fn collect_and_publish(worker: &mut Worker, event_tx: &SyncSender<SimEvent>) {
        while !worker.kernel.request_state_snapshot() {
            std::hint::spin_loop();
        }
        while !worker.kernel.try_collect_state_snapshot() {
            std::hint::spin_loop();
        }
        worker.publish_snapshot(event_tx);
    }

    #[test]
    fn tick_window_store_persists_frames_in_bounds() {
        if !supervisor_gpu_or_skip() {
            return;
        }
        let config = supervisor_test_config();
        let dt = supervisor_dt(10.0);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("w.db").to_string_lossy().into_owned();
        let (event_tx, _event_rx) = sync_channel(4);
        let mut worker = Worker::new(supervisor_init(&config, Some(path.clone())));
        assert!(worker.tick_window_store.is_some());
        for step in 1..=5 {
            supervisor_run_to(&mut worker, dt, step * 200);
            collect_and_publish(&mut worker, &event_tx);
        }
        assert!(!worker.paused);
        let final_tick = worker.tick;
        drop(worker);
        let db = rusqlite::Connection::open(&path).expect("open");
        let ticks: Vec<i64> = db
            .prepare("SELECT tick FROM supervisor_replay_frame")
            .expect("prepare")
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<Result<_, _>>()
            .expect("rows");
        assert!(!ticks.is_empty());
        for tick in ticks {
            assert!(tick >= 0);
            assert!(u64::try_from(tick).expect("non-negative") <= final_tick);
        }
    }

    #[test]
    fn tick_window_absent_without_database() {
        let config = supervisor_test_config();
        if supervisor_gpu_or_skip() {
            let worker = Worker::new(supervisor_init(&config, None));
            assert!(worker.tick_window_store.is_none());
            let worker = Worker::new(test_init(0, false, 1));
            assert!(worker.tick_window_store.is_none());
        }
    }

    #[test]
    fn tick_window_store_failure_logs_and_continues() {
        if !supervisor_gpu_or_skip() {
            return;
        }
        let config = supervisor_test_config();
        let dt = supervisor_dt(10.0);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("f.db").to_string_lossy().into_owned();
        let (event_tx, _event_rx) = sync_channel(4);
        let mut worker = Worker::new(supervisor_init(&config, Some(path.clone())));
        assert!(worker.tick_window_store.is_some());
        {
            let db = rusqlite::Connection::open(&path).expect("open");
            db.execute_batch("DROP TABLE supervisor_replay_frame;")
                .expect("drop");
        }
        supervisor_run_to(&mut worker, dt, 200);
        collect_and_publish(&mut worker, &event_tx);
        let logs = worker
            .supervisor_events
            .iter()
            .filter(|event| matches!(event, SimEvent::Log(_)))
            .count();
        assert_eq!(logs, 1);
        assert!(worker.tick_window_store.is_none());
        assert!(!worker.paused);
        let before = worker.tick;
        supervisor_run_to(&mut worker, dt, before + 200);
        assert!(worker.tick > before);
        assert!(!worker.paused);
    }

    #[test]
    fn worker_checkpoint_adds_no_submit() {
        if !supervisor_gpu_or_skip() {
            return;
        }
        const TARGET: u64 = 3_000;
        let config = supervisor_test_config();
        let dt = supervisor_dt(10.0);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("d.db").to_string_lossy().into_owned();
        let measure = |db_path: Option<String>| {
            let mut worker = Worker::new(supervisor_init(&config, db_path));
            let events = supervisor_collect_events(&mut worker, dt, TARGET, false);
            let samples = events
                .iter()
                .filter(|event| matches!(event, SimEvent::SupervisorSample(_)))
                .count();
            (
                worker.counters.dispatch_calls,
                worker.counters.sim_ticks,
                worker.kernel.queue_submit_count(),
                samples,
            )
        };
        let with_db = measure(Some(path));
        let without_db = measure(None);
        assert_eq!(with_db, without_db);
    }

    #[test]
    fn supervisor_telemetry_partition_keeps_new_variants_in_order() {
        use xagent_sandbox::supervisor::{BoundaryDetail, IdleReason};
        use xagent_sandbox::supervisor_telemetry::ProgressRecord;

        let record = BoundaryRecord {
            detail: BoundaryDetail::Idle {
                tick: 1_000,
                standings: Vec::new(),
                reason: IdleReason {
                    too_young: 0,
                    no_full_window: 0,
                    no_peer_median: 0,
                    at_or_above_median: 0,
                },
            },
            newborn_occupant: None,
            config_diff: Vec::new(),
            retired_lifespan_ticks: None,
        };
        let frame = TelemetryFrame {
            tick: 1_000,
            slots: Vec::new(),
            boundary: None,
            series: Vec::new(),
            intake: Vec::new(),
            progress: ProgressRecord {
                tick: 1_000,
                total_ticks: None,
                agent_tick_budget: 0,
                ticks_to_boundary: 0,
                nourishing: None,
                next_flip_tick: None,
                ticks_per_second: None,
            },
            persist: false,
        };
        let events = vec![
            SimEvent::SupervisorBoundary(Box::new(record)),
            SimEvent::Log("a".into()),
            SimEvent::SupervisorTelemetry(Box::new(frame)),
        ];
        let (latest, control) = partition_events(events, 0);
        assert!(latest.is_none());
        assert_eq!(control.len(), 3);
        assert!(matches!(control[0], SimEvent::SupervisorBoundary(_)));
        assert!(matches!(control[1], SimEvent::Log(_)));
        assert!(matches!(control[2], SimEvent::SupervisorTelemetry(_)));
    }

    #[test]
    fn supervisor_telemetry_boundary_control_events_follow_samples() {
        if !supervisor_gpu_or_skip() {
            return;
        }
        let config = supervisor_test_config();
        let mut init = supervisor_init(&config, None);
        init.speed_multiplier = 1_000;
        let mut worker = Worker::new(init);
        let events = supervisor_collect_events(&mut worker, 0.05, 3_000, false);

        let mut boundary_reports = 0_usize;
        let mut boundary_events = 0_usize;
        for (index, event) in events.iter().enumerate() {
            match event {
                SimEvent::SupervisorBoundary(_) => boundary_events += 1,
                SimEvent::SupervisorSample(report) if report.boundary.is_some() => {
                    boundary_reports += 1;
                    assert!(index > 0, "a boundary sample must follow its boundary");
                    match &events[index - 1] {
                        SimEvent::SupervisorBoundary(record) => {
                            assert_eq!(boundary_record_tick(record), report.tick);
                        }
                        _ => panic!("sample at {} lacks a preceding boundary", report.tick),
                    }
                }
                _ => {}
            }
        }
        assert!(boundary_reports >= 2, "got {boundary_reports} boundaries");
        assert_eq!(boundary_events, boundary_reports);
    }

    #[test]
    fn supervisor_telemetry_frame_dropped_when_channel_full() {
        if !supervisor_gpu_or_skip() {
            return;
        }
        let config = supervisor_test_config();
        let mut init = supervisor_init(&config, None);
        init.speed_multiplier = 1_000;
        let mut worker = Worker::new(init);
        let (event_tx, event_rx) = sync_channel::<SimEvent>(1);
        event_tx
            .send(SimEvent::Log("full".into()))
            .expect("channel has room");
        let before = worker.counters.dispatch_calls;
        for _ in 0..10_000_000 {
            if worker.tick >= 900 || worker.paused {
                break;
            }
            worker.step(&event_tx);
        }
        assert!(worker.tick >= 900, "worker stalled at {}", worker.tick);
        assert!(worker.counters.dispatch_calls > before);
        assert!(matches!(event_rx.try_recv(), Ok(SimEvent::Log(_))));
        assert!(event_rx.try_recv().is_err());
    }

    #[test]
    fn supervisor_telemetry_dispatch_schedule_unchanged_when_frames_discarded() {
        if !supervisor_gpu_or_skip() {
            return;
        }
        let config = supervisor_test_config();
        let build = || {
            let mut init = supervisor_init(&config, None);
            init.speed_multiplier = 1_000;
            Worker::new(init)
        };
        let reports = |events: Vec<SimEvent>| -> Vec<SampleReport> {
            events
                .into_iter()
                .filter_map(|event| match event {
                    SimEvent::SupervisorSample(report) => Some(report),
                    _ => None,
                })
                .collect()
        };
        let mut a = build();
        let mut b = build();
        let reports_a = reports(supervisor_collect_events(&mut a, 0.05, 3_000, false));
        let reports_b = reports(supervisor_collect_events(&mut b, 0.05, 3_000, true));
        assert_eq!(a.tick, b.tick);
        assert_eq!(a.counters.dispatch_calls, b.counters.dispatch_calls);
        assert_eq!(
            a.kernel.probe_kernel_batches(),
            b.kernel.probe_kernel_batches()
        );
        assert_eq!(reports_a, reports_b);
        let batch = u64::from(a.kernel.kernel_batch_size());
        assert_eq!(a.kernel.probe_kernel_batches(), a.tick / batch);
    }

    #[test]
    fn supervisor_gpu_guard_fails_closed_when_required() {
        assert!(supervisor_gpu_guard_decision(false, true).is_err());
        assert_eq!(supervisor_gpu_guard_decision(false, false), Ok(false));
        assert_eq!(supervisor_gpu_guard_decision(true, true), Ok(true));
        assert_eq!(supervisor_gpu_guard_decision(true, false), Ok(true));
    }

    #[test]
    fn supervisor_clamp_never_crosses_sample_grid() {
        assert_eq!(clamp_ticks_to_supervisor_sample(500, 0, 0), 500);
        assert_eq!(clamp_ticks_to_supervisor_sample(1000, 30, 100), 70);
        assert_eq!(clamp_ticks_to_supervisor_sample(50, 0, 100), 50);
        assert_eq!(clamp_ticks_to_supervisor_sample(100, 100, 100), 100);
        assert_eq!(clamp_ticks_to_supervisor_sample(1000, 100, 100), 100);
        for run in [10_u32, 37, 250, 999] {
            let mut tick = 0_u64;
            let mut samples = 0_u64;
            for _ in 0..500 {
                let ticks = clamp_ticks_to_supervisor_sample(run, tick, 100);
                assert!(ticks > 0);
                let next = tick + u64::from(ticks);
                let crossed = next / 100 - tick / 100;
                assert!(crossed == 0 || next % 100 == 0, "{tick}->{next}");
                samples += crossed;
                tick = next;
            }
            assert_eq!(samples, tick / 100);
        }
    }

    #[test]
    fn supervisor_worker_matches_headless_driver() {
        if !supervisor_gpu_or_skip() {
            return;
        }
        const TARGET: u64 = 2_000;
        let config = supervisor_test_config();
        let keep = |r: &SampleReport| r.boundary.is_some() || r.season.is_some() || r.finished;

        let reference = {
            let mut kernel = GpuKernel::new(
                u32::try_from(SUPERVISOR_TEST_AGENTS).expect("fits"),
                xagent_sandbox::world::WorldState::new(config.world.clone())
                    .food_items
                    .len(),
                &config.brain,
                &config.world,
            );
            let mut run =
                SupervisorRun::start(&mut kernel, &config, None, false).expect("run starts");
            let mut reports = Vec::new();
            let mut tick = 0;
            while tick < TARGET {
                kernel.dispatch_batch(tick, 100);
                while !kernel.try_collect_state_snapshot() {
                    std::hint::spin_loop();
                }
                tick += 100;
                let physics = kernel.cached_state().to_vec();
                let report = run
                    .on_sample(&mut kernel, tick, &physics, None)
                    .expect("sample is handled");
                if keep(&report) {
                    reports.push(report);
                }
            }
            reports
        };
        assert!(!reference.is_empty(), "reference run produced no reports");

        let mut results = Vec::new();
        for (speed, dt) in [(1_u32, supervisor_dt(10.0)), (1_000, 0.05)] {
            let mut init = supervisor_init(&config, None);
            init.speed_multiplier = speed;
            let mut worker = Worker::new(init);
            let reports = supervisor_run_to(&mut worker, dt, TARGET);
            assert_eq!(worker.tick, TARGET);
            assert_eq!(reports, reference, "speed {speed}");
            results.push(worker.counters.dispatch_calls);
        }
        assert_ne!(results[0], results[1], "dispatch counts should differ");
    }

    #[test]
    fn supervisor_worker_uses_one_kernel_and_exact_batches() {
        if !supervisor_gpu_or_skip() {
            return;
        }
        let config = supervisor_test_config();
        let mut init = supervisor_init(&config, None);
        init.speed_multiplier = 1_000;
        let mut worker = Worker::new(init);
        let batch = u64::from(worker.kernel.kernel_batch_size());
        let check = |worker: &Worker| {
            assert_eq!(worker.kernel.probe_kernel_batches(), worker.tick / batch);
        };

        supervisor_run_to(&mut worker, 0.05, 1_500);
        check(&worker);
        let (event_tx, _event_rx) = sync_channel(8);
        let (upload, _) = test_upload();
        let before = worker.kernel.probe_kernel_batches();
        worker.handle_command(
            SimCommand::ResetPopulation(Box::new(ResetRequest {
                upload,
                brain_config: BrainConfig::default(),
                tick_budget: 0,
                inherited: None,
                resume: true,
                generation_epoch: 1,
                search_steering_genome: false,
            })),
            &event_tx,
        );
        assert_eq!(worker.kernel.probe_kernel_batches(), before);
        check(&worker);

        supervisor_run_to(&mut worker, 0.05, 3_000);
        assert!(worker.tick >= 3_000 && worker.tick % SUPERVISOR_SAMPLE_TICKS == 0);
        check(&worker);
        assert_eq!(
            worker.kernel.agent_count(),
            u32::try_from(config.governor.population_size).expect("fits")
        );
    }

    #[test]
    fn supervisor_worker_ignores_reset_population() {
        if !supervisor_gpu_or_skip() {
            return;
        }
        let config = supervisor_test_config();
        let mut worker = Worker::new(supervisor_init(&config, None));
        supervisor_run_to(&mut worker, 0.05, 200);
        let (tick, agents) = (worker.tick, worker.kernel.agent_count());
        let (event_tx, event_rx) = sync_channel(8);
        let (upload, _) = test_upload();
        worker.handle_command(
            SimCommand::ResetPopulation(Box::new(ResetRequest {
                upload,
                brain_config: BrainConfig::default(),
                tick_budget: 50,
                inherited: None,
                resume: false,
                generation_epoch: 9,
                search_steering_genome: false,
            })),
            &event_tx,
        );
        assert_eq!(worker.tick, tick);
        assert_eq!(worker.kernel.agent_count(), agents);
        assert_eq!(worker.tick_budget, 0);
        assert_eq!(worker.generation_epoch, 0);
        assert!(!worker.paused);
        assert!(worker.supervisor.is_some());
        assert!(event_rx.try_recv().is_err(), "no reset log expected");
    }

    #[test]
    fn supervisor_worker_pauses_when_finished() {
        if !supervisor_gpu_or_skip() {
            return;
        }
        let config = supervisor_test_config_with_budget(50_000);
        let mut init = supervisor_init(&config, None);
        init.speed_multiplier = 1_000;
        let mut worker = Worker::new(init);
        let reports = supervisor_run_to(&mut worker, 0.05, u64::MAX);
        assert!(worker.paused);
        assert!(worker.supervisor.is_none());
        assert_eq!(worker.tick, 5_000);
        assert!(reports.last().is_some_and(|r| r.finished));
        let dispatches = worker.counters.dispatch_calls;
        worker.paused = false;
        assert!(!worker.advance_compute(0.05));
        assert_eq!(worker.counters.dispatch_calls, dispatches);
    }

    #[test]
    fn supervisor_worker_answers_birth_state_from_ledger() {
        if !supervisor_gpu_or_skip() {
            return;
        }
        let config = supervisor_test_config();
        let mut init = supervisor_init(&config, None);
        init.speed_multiplier = 1_000;
        let mut worker = Worker::new(init);
        assert_eq!(worker.birth_states.len(), SUPERVISOR_TEST_AGENTS);
        supervisor_run_to(&mut worker, 0.05, 3_000);
        let bits = |s: &AgentBrainState| -> Vec<u32> {
            s.brain_state
                .iter()
                .chain(&s.patterns)
                .map(|v| v.to_bits())
                .collect()
        };
        let (event_tx, event_rx) = sync_channel(SUPERVISOR_TEST_AGENTS);
        for slot in 0..SUPERVISOR_TEST_AGENTS {
            worker.handle_command(
                SimCommand::RequestBirthState {
                    agent_index: u32::try_from(slot).expect("fits"),
                    request_id: u64::try_from(slot).expect("fits"),
                },
                &event_tx,
            );
            let Ok(SimEvent::BirthState {
                state: Some(state), ..
            }) = event_rx.try_recv()
            else {
                panic!("slot {slot} should answer a birth state");
            };
            let run = worker.supervisor.as_ref().expect("run is live");
            let ledger = run.birth_state(slot).expect("slot exists");
            assert_eq!(bits(&state), bits(ledger), "slot {slot}");
        }
    }
}
