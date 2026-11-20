//! Shared kernel-facing driver of the continuous-life supervisor.
//!
//! The driver borrows the caller's [`GpuKernel`] and owns the pure supervisor
//! core, its own [`WorldState`] for newborn spawn positions, an optional store
//! and an optional run-summary accumulator. The caller dispatches batches and
//! hands each 100-tick readback to [`SupervisorRun::on_sample`].

use std::fmt;

use xagent_brain::buffers::{AgentBrainState, PHYS_STRIDE};
use xagent_brain::gpu_kernel::{FoodKindSetError, SlotReplaceError, SlotSpawn};
use xagent_brain::GpuKernel;
use xagent_shared::{
    EvolutionMode, FoodKind, FullConfig, SupervisorConfigError, SUPERVISOR_SAMPLE_TICKS,
};

use crate::agent::Agent;
use crate::run_summary::{RunSummary, RunSummaryAccumulator, RunSummaryError};
use crate::supervisor::{
    stream_seed, ArchiveChange, ParentRef, ReplaceReason, ReplacementPlan, SampleDecision,
    SlotCounters, Supervisor, SupervisorError, TickSchedule, SEASON_STREAM,
};
use crate::supervisor_store::{StoreError, SupervisorStore};
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
        }
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
}

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
                EvolutionMode::Supervisor,
                Some(config.governor.supervisor.inheritance),
                seed,
                population,
                kinds_on,
            )
        });
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
        };
        run.upload_population(kernel, config)?;
        run.begin_run_season(kernel)?;
        Ok(run)
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
        let agents: Vec<Agent> = (0..self.population)
            .map(|slot| {
                let id = slot_index(slot)?;
                Ok(Agent::new(
                    id,
                    self.world.safe_spawn_position(),
                    id,
                    config.brain.clone(),
                    0,
                ))
            })
            .collect::<Result<_, SupervisorRunError>>()?;
        let agents: Vec<Agent> = agents;
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
        self.tick = tick;
        let season = self.flip_season(kernel, tick)?;
        let boundary = self.boundary(kernel, tick, decision)?;
        Ok(SampleReport {
            tick,
            boundary,
            season,
            finished: self.is_finished(),
        })
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
    ) -> Result<Option<BoundaryReport>, SupervisorRunError> {
        match decision {
            SampleDecision::Recorded => Ok(None),
            SampleDecision::Idle => {
                if let Some(store) = self.store.as_mut() {
                    store.record_idle_boundary(tick)?;
                }
                Ok(Some(self.report(tick, None, None)))
            }
            SampleDecision::Replace(plan) => self.replace(kernel, tick, plan).map(Some),
        }
    }

    fn replace(
        &mut self,
        kernel: &GpuKernel,
        tick: u64,
        plan: ReplacementPlan,
    ) -> Result<BoundaryReport, SupervisorRunError> {
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
        let agent = Agent::new(
            id,
            self.world.safe_spawn_position(),
            slot,
            outcome.newborn_config.clone(),
            tick,
        );
        kernel.replace_agent_slot(
            slot,
            &outcome.newborn_brain,
            &outcome.newborn_config,
            &spawn_of(&agent),
        )?;
        if let Some(acc) = self.accumulator.as_mut() {
            acc.reset_agent_baseline(plan.slot)?;
        }
        if let Some(store) = self.store.as_mut() {
            store.record_boundary(tick, &plan, &outcome, outcome.archived.as_ref())?;
        }
        Ok(self.report(tick, Some(&plan), Some(outcome.occupant_id)))
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
