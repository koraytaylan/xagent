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

use xagent_brain::{AgentBrainState, AgentTelemetry, GpuKernel};
use xagent_sandbox::agent::mutate_brain_state;
use xagent_shared::{BrainConfig, WorldConfig};

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

/// An owned, CPU-visible snapshot of GPU simulation state.
///
/// Owns its vectors so it can cross the worker → main channel without borrowing
/// the kernel's caches. `physics` is `PHYS_STRIDE` floats per agent; `food` is
/// `FOOD_STATE_STRIDE` floats per food item (empty when the world has no food).
pub struct StateSnapshot {
    pub physics: Vec<f32>,
    pub food: Vec<f32>,
    pub tick: u64,
    pub generation_tick: u64,
}

/// Champion brain state to seed into the next generation on reset.
///
/// The first `champion_slots` agents inherit the champion's exact weights; the
/// rest inherit mutated copies, reproducing the neuroevolution seeding the
/// main-thread transition used to perform.
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
}

/// Commands from the main thread to the simulation worker.
pub enum SimCommand {
    /// Pause or resume tick advancement.
    SetPaused(bool),
    /// Set the simulation speed multiplier.
    SetSpeed(u32),
    /// Set the selected agent (by kernel brain index) for telemetry.
    SelectAgent(u32),
    /// Request a blocking champion brain-state readback while paused.
    RequestAgentState { agent_index: u32, request_id: u64 },
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
    /// Reply to [`SimCommand::RequestAgentState`].
    AgentState {
        request_id: u64,
        state: Option<AgentBrainState>,
    },
    /// Human-readable worker log line.
    Log(String),
    /// Worker error (e.g. no GPU adapter); the worker may be degraded.
    Error(String),
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
}

/// Partition a batch of drained events into the newest state snapshot
/// (latest-wins) and the ordered control events.
///
/// Intermediate [`SimEvent::Snapshot`]s are superseded by the most recent one;
/// every other event — including [`SimEvent::GenerationBudgetReached`], whose
/// snapshot is the authoritative end-of-generation state — is returned in
/// arrival order so the consumer processes them deterministically.
pub fn partition_events(events: Vec<SimEvent>) -> (Option<StateSnapshot>, Vec<SimEvent>) {
    let mut latest_snapshot = None;
    let mut control = Vec::new();
    for event in events {
        match event {
            SimEvent::Snapshot(snapshot) => latest_snapshot = Some(snapshot),
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
        let kernel = GpuKernel::new(
            init.agent_count,
            init.food_count,
            &init.brain_config,
            &init.world_config,
        );
        kernel.upload_world(
            &init.upload.heights,
            &init.upload.biomes,
            &init.upload.food_pos,
            &init.upload.food_consumed,
            &init.upload.food_timers,
        );
        kernel.upload_agents(&init.upload.agent_data);
        patch_agent_configs(&kernel, &init.upload.agent_configs);
        Self {
            kernel,
            brain_config: init.brain_config,
            world_config: init.world_config,
            paused: init.paused,
            speed_multiplier: init.speed_multiplier,
            selected_agent: init.selected_agent,
            tick: 0,
            gen_tick: 0,
            tick_budget: init.tick_budget,
            budget_reached: false,
            sim_accumulator: 0.0,
            last_frame: Instant::now(),
            last_state_snapshot_request: None,
            last_telemetry_request: None,
            last_telemetry_agent: None,
            counters: WorkerCounters::default(),
            last_counters_log: Instant::now(),
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
            SimCommand::RequestAgentState {
                agent_index,
                request_id,
            } => {
                // Blocking readback is safe here: the handoff only requests
                // champion state while the worker is paused at a generation
                // boundary.
                let state = if agent_index < self.kernel.agent_count() {
                    Some(self.kernel.read_agent_state(agent_index))
                } else {
                    None
                };
                let _ = event_tx.send(SimEvent::AgentState { request_id, state });
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

        if let Some(inherited) = request.inherited {
            let count = self.kernel.agent_count() as usize;
            let champion = inherited.champion;
            let strength = inherited.mutation_strength;
            let champion_slots = inherited.champion_slots;
            self.kernel.batch_write_agent_states(count, |i| {
                if i < champion_slots {
                    champion.clone()
                } else {
                    mutate_brain_state(&champion, strength)
                }
            });
        }
        patch_agent_configs(&self.kernel, &request.upload.agent_configs);

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
            raw_ticks
        } else {
            0
        };

        let (clamped, did_clamp) =
            clamp_ticks_to_generation_budget(ticks_to_run, self.tick_budget, self.gen_tick);
        ticks_to_run = clamped;
        if did_clamp {
            self.counters.generation_clamps += 1;
        }

        if ticks_to_run == 0 {
            return false;
        }

        self.kernel.dispatch_ticks(self.tick, ticks_to_run);
        self.counters.dispatch_calls += 1;
        self.counters.sim_ticks += u64::from(ticks_to_run);

        self.sim_accumulator -= ticks_to_run as f64 * sim_delta_time;
        self.tick += u64::from(ticks_to_run);
        self.gen_tick = self.gen_tick.saturating_add(u64::from(ticks_to_run));
        true
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
            tick: self.tick,
            generation_tick: self.gen_tick,
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
            tick: self.tick,
            generation_tick: self.gen_tick,
        }
    }

    /// Publish the latest collected snapshot, dropping it under back-pressure
    /// (latest-wins): the next snapshot carries newer state, so a dropped one is
    /// never missed. The generation boundary delivers its final state through
    /// the always-sent [`SimEvent::GenerationBudgetReached`] instead.
    fn publish_snapshot(&self, event_tx: &SyncSender<SimEvent>) {
        let snapshot = self.build_snapshot();
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
        // remains, loop immediately so it drains at GPU speed — `queue.submit`
        // back-pressure paces the worker without a fixed throttle, and snapshots
        // publish as fast as the GPU completes batches.
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

    // ── latest-wins event partitioning ───────────────────────────────

    fn snapshot_with_tick(tick: u64) -> StateSnapshot {
        StateSnapshot {
            physics: Vec::new(),
            food: Vec::new(),
            tick,
            generation_tick: tick,
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
        let (latest, control) = partition_events(events);
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
            SimEvent::AgentState {
                request_id: 7,
                state: None,
            },
        ];
        let (latest, control) = partition_events(events);
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
            SimEvent::AgentState { request_id: 7, .. }
        ));
    }

    #[test]
    fn partition_events_handles_no_snapshot() {
        let events = vec![SimEvent::Log("only logs".into())];
        let (latest, control) = partition_events(events);
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

    fn test_init(tick_budget: u64, paused: bool, speed_multiplier: u32) -> SimInit {
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
            FIXED_TAIL_SIZE, O_GABOR_WAVELENGTH, O_PREDICTOR_CONTEXT_WEIGHT,
        };

        let state = kernel.read_agent_state(index);
        let tail_base = state.brain_state.len() - FIXED_TAIL_SIZE;
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
        });

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
    /// `RequestAgentState` while paused, and resumes the next generation on
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
        runtime.send(SimCommand::RequestAgentState {
            agent_index: 0,
            request_id: 42,
        });
        let champion = drain_until(&runtime, |event| match event {
            SimEvent::AgentState {
                request_id: 42,
                state,
            } => Some(state),
            _ => None,
        })
        .expect("AgentState reply should arrive");
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
}
