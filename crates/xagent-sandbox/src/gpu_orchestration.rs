//! Main-thread orchestration of the simulation worker.
//!
//! The GPU kernel and all simulation-cadence scheduling live in the
//! [`crate::sim_runtime`] worker thread. This module is the main-thread side of
//! that boundary: it starts the worker for the current population, forwards
//! control commands (speed, pause, selection) on change, drains worker events
//! each frame, applies the newest state snapshot to the CPU-side agent caches,
//! and records the per-snapshot overlays/histories/replay. It performs no GPU
//! work and never calls `GpuKernel` directly.

use std::time::{Duration, Instant};

use glam::Vec3;

use xagent_brain::buffers::{
    PHYS_STRIDE, P_ALIVE, P_AVOIDANCE_SENSE_RANGE_TICKS, P_AVOIDANCE_TURNS_OPPOSING,
    P_DANGER_PATH_LENGTH, P_DEATH_COUNT, P_DISTANCE_TRAVELED, P_ENERGY, P_ENERGY_SPENT,
    P_EXPLORATION_RATE_OUT, P_FACING_X, P_FACING_Y, P_FACING_Z, P_FATIGUE_FACTOR_OUT, P_FOOD_COUNT,
    P_GRADIENT_OUT, P_INTEGRITY, P_LAST_DEATH_TICK, P_MAX_ENERGY, P_MAX_INTEGRITY, P_MOTOR_FWD_OUT,
    P_MOTOR_TURN_OUT, P_POS_X, P_POS_Y, P_POS_Z, P_PREDICTION_ERROR, P_TICKS_ALIVE, P_URGENCY_OUT,
    P_VEL_X, P_VEL_Y, P_VEL_Z, P_YAW,
};
use xagent_brain::AgentTelemetry;

use crate::app::{App, PendingUpload};
use crate::sim_runtime::{
    partition_events, SimCommand, SimEvent, SimInit, SimRuntime, StateSnapshot,
};

impl App {
    /// Build the world + agent upload for the current population, or `None`
    /// when there is no world. Shared by worker startup and generation reset.
    pub(crate) fn build_pending_upload(&self) -> Option<PendingUpload> {
        let world = self.world.as_ref()?;
        Some(PendingUpload {
            heights: world.terrain.heights.clone(),
            biomes: world.biome_map.grid_as_u32(),
            food_pos: world
                .food_items
                .iter()
                .map(|f| (f.position.x, f.position.y, f.position.z))
                .collect(),
            food_consumed: world.food_items.iter().map(|f| f.consumed).collect(),
            food_timers: world.food_items.iter().map(|f| f.respawn_timer).collect(),
            agent_data: self
                .agents
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
                .collect(),
            agent_configs: self
                .agents
                .iter()
                .map(|agent| agent.brain_config.clone())
                .collect(),
        })
    }

    /// Start the simulation worker for the current world and population.
    ///
    /// Builds the initial world/agent upload and kernel-creation parameters,
    /// spawns the worker, and seeds the forwarded-control cache so the first
    /// `sync_worker_controls` only sends commands that actually changed. No-op
    /// when there is no world or population yet.
    pub(crate) fn start_sim_worker(&mut self) {
        if self.agents.is_empty() {
            return;
        }
        let Some(upload) = self.build_pending_upload() else {
            return;
        };
        let (food_count, world_config) = match &self.world {
            Some(world) => (world.food_items.len(), world.config.clone()),
            None => return,
        };

        let agent_count = u32::try_from(self.agents.len()).unwrap_or(u32::MAX);
        let selected_agent = self
            .agents
            .get(self.selected_agent_idx)
            .map_or(0, |a| a.brain_idx);

        let init = SimInit {
            agent_count,
            food_count,
            brain_config: self.brain_config.clone(),
            world_config,
            upload,
            tick_budget: self.governor_config.tick_budget,
            speed_multiplier: self.speed_multiplier,
            paused: self.paused,
            selected_agent,
        };

        self.sim_runtime = Some(SimRuntime::start(init));
        self.sent_speed = Some(self.speed_multiplier);
        self.sent_paused = Some(self.paused);
        self.sent_selected_agent = Some(selected_agent);
        self.cached_food_state = None;
        self.log_msg(format!("[GPU] sim worker starting ({agent_count} agents)"));
    }

    /// Drop the simulation worker, shutting down and joining its thread.
    pub(crate) fn stop_sim_worker(&mut self) {
        // Dropping the handle requests Shutdown and joins the worker thread.
        self.sim_runtime = None;
        self.sent_speed = None;
        self.sent_paused = None;
        self.sent_selected_agent = None;
        self.pending_generation = None;
        self.cached_food_state = None;
    }

    /// Forward speed, pause, and selection to the worker, but only on change.
    ///
    /// The worker pauses itself at a generation boundary and resumes only on
    /// `ResetPopulation`, so while a handoff is pending the main thread keeps
    /// the worker paused regardless of the user's running state.
    pub(crate) fn sync_worker_controls(&mut self) {
        if self.sim_runtime.is_none() {
            return;
        }

        let speed = self.speed_multiplier;
        if self.sent_speed != Some(speed) {
            if let Some(runtime) = &self.sim_runtime {
                runtime.send(SimCommand::SetSpeed(speed));
            }
            self.sent_speed = Some(speed);
        }

        let desired_paused = self.paused || self.pending_generation.is_some();
        if self.sent_paused != Some(desired_paused) {
            if let Some(runtime) = &self.sim_runtime {
                runtime.send(SimCommand::SetPaused(desired_paused));
            }
            self.sent_paused = Some(desired_paused);
        }

        let selected_brain = self
            .agents
            .get(self.selected_agent_idx)
            .map(|a| a.brain_idx);
        if let Some(brain_idx) = selected_brain {
            if self.sent_selected_agent != Some(brain_idx) {
                if let Some(runtime) = &self.sim_runtime {
                    runtime.send(SimCommand::SelectAgent(brain_idx));
                }
                self.sent_selected_agent = Some(brain_idx);
            }
        }
    }

    /// Drain all pending worker events, applying the newest snapshot and
    /// processing control events (kernel-ready, telemetry, generation boundary,
    /// champion state, logs) in order.
    pub(crate) fn drain_sim_events(&mut self) {
        let Some(runtime) = &self.sim_runtime else {
            return;
        };
        let events = runtime.drain_events();
        if events.is_empty() {
            return;
        }
        let (latest_snapshot, control) = partition_events(events);

        // Latest-wins: apply only the newest plain snapshot.
        if let Some(snapshot) = latest_snapshot {
            self.apply_state_snapshot(&snapshot);
            self.on_fresh_snapshot();
        }

        for event in control {
            match event {
                SimEvent::KernelReady { agent_count } => {
                    self.log_msg(format!(
                        "[GPU] sim worker kernel ready ({agent_count} agents)"
                    ));
                }
                // Plain snapshots are folded into latest-wins above.
                SimEvent::Snapshot(_) => {}
                SimEvent::Telemetry {
                    agent_index,
                    telemetry,
                } => self.apply_telemetry(agent_index, telemetry),
                SimEvent::GenerationBudgetReached(snapshot) => {
                    self.apply_state_snapshot(&snapshot);
                    self.on_fresh_snapshot();
                    self.runtime_counters.generation_boundaries += 1;
                    self.on_generation_budget_reached();
                }
                SimEvent::AgentState { request_id, state } => {
                    self.on_champion_state(request_id, state);
                }
                SimEvent::Log(message) => self.log_msg(message),
                SimEvent::Error(message) => self.log_msg(format!("[GPU] {message}")),
            }
        }
    }

    /// Apply a worker physics/food snapshot to the CPU-side agent caches.
    ///
    /// Authoritative for position/yaw/alive/energy/integrity/velocity and the
    /// `cached_*` motor/gradient/urgency/prediction/exploration/fatigue fields,
    /// plus the food state that backs the food mesh and mini-map.
    fn apply_state_snapshot(&mut self, snapshot: &StateSnapshot) {
        // The worker owns tick advancement, so derive the ticks/sec display from
        // the delta between published snapshots (it resets to 0 across a
        // generation boundary, which `saturating_sub` reports as no progress
        // rather than a spurious spike).
        self.tps_tick_count += snapshot.tick.saturating_sub(self.tick);
        self.tick = snapshot.tick;
        // Sync the governor's per-generation counter from the snapshot to keep
        // the evolution UI's progress accurate.
        if let Some(governor) = self.governor.as_mut() {
            governor.gen_tick = snapshot.generation_tick;
        }
        let state = &snapshot.physics;
        let death_reference_tick = snapshot.tick.saturating_sub(1);
        for i in 0..self.agents.len() {
            let base = i * PHYS_STRIDE;
            // Guard every physics field we read below. Use the stride's highest
            // offset so adding new fields doesn't silently leave the guard stale.
            if base + PHYS_STRIDE > state.len() {
                break;
            }
            let a = &mut self.agents[i];
            a.body.body.position = Vec3::new(
                state[base + P_POS_X],
                state[base + P_POS_Y],
                state[base + P_POS_Z],
            );
            a.body.body.alive = state[base + P_ALIVE] > 0.5;
            a.body.yaw = state[base + P_YAW];
            a.body.body.internal.energy = state[base + P_ENERGY];
            a.body.body.internal.integrity = state[base + P_INTEGRITY];
            a.body.body.internal.max_energy = state[base + P_MAX_ENERGY];
            a.body.body.internal.max_integrity = state[base + P_MAX_INTEGRITY];
            a.body.body.velocity = Vec3::new(
                state[base + P_VEL_X],
                state[base + P_VEL_Y],
                state[base + P_VEL_Z],
            );
            a.food_consumed = state[base + P_FOOD_COUNT] as u32;
            a.total_ticks_alive = state[base + P_TICKS_ALIVE] as u64;
            a.distance_traveled = state[base + P_DISTANCE_TRAVELED];
            a.energy_spent = state[base + P_ENERGY_SPENT];
            a.danger_path_length = state[base + P_DANGER_PATH_LENGTH];
            a.avoidance_sense_range_ticks = state[base + P_AVOIDANCE_SENSE_RANGE_TICKS];
            a.avoidance_turns_opposing = state[base + P_AVOIDANCE_TURNS_OPPOSING];
            let new_deaths = state[base + P_DEATH_COUNT] as u32;
            let gpu_death_tick = state[base + P_LAST_DEATH_TICK] as u64;
            a.apply_death_count_readback(new_deaths, gpu_death_tick, death_reference_tick);
            a.body.body.facing = Vec3::new(
                state[base + P_FACING_X],
                state[base + P_FACING_Y],
                state[base + P_FACING_Z],
            );
            a.cached_prediction_error = state[base + P_PREDICTION_ERROR];
            a.cached_exploration_rate = state[base + P_EXPLORATION_RATE_OUT];
            a.cached_fatigue_factor = state[base + P_FATIGUE_FACTOR_OUT];
            a.cached_motor.forward = state[base + P_MOTOR_FWD_OUT];
            a.cached_motor.turn = state[base + P_MOTOR_TURN_OUT];
            a.cached_gradient = state[base + P_GRADIENT_OUT];
            a.cached_urgency = state[base + P_URGENCY_OUT];
        }

        // Feed the population's cumulative foraging totals to the governor's
        // within-life tracker (it snapshots them at each quarter of the tick
        // budget). Summed before the mutable governor borrow below.
        let cumulative_food: u64 = self.agents.iter().map(|a| u64::from(a.food_consumed)).sum();
        let cumulative_alive: u64 = self.agents.iter().map(|a| a.total_ticks_alive).sum();
        if let Some(governor) = self.governor.as_mut() {
            governor.record_within_life_sample(cumulative_food, cumulative_alive);
        }

        // Keep the previous food cache rather than overwriting with the empty
        // vector the worker sends before its first food readback.
        if !snapshot.food.is_empty() {
            match &mut self.cached_food_state {
                Some(food) => food.clone_from(&snapshot.food),
                none => *none = Some(snapshot.food.clone()),
            }
        }

        self.runtime_counters.snapshots_applied += 1;
        self.snap_dirty = true;
        self.hud_dirty = true;

        // Diagnostic: log first snapshot Y vs terrain height.
        if !self.readback_logged {
            if let Some(world) = &self.world {
                let n = self.agents.len().min(5);
                if n > 0 {
                    for i in 0..n {
                        let a = &self.agents[i];
                        let p = a.body.body.position;
                        let terrain_y = world.terrain.height_at(p.x, p.z);
                        log::info!(
                            "[TERRAIN-DIAG] Agent {} pos=({:.2}, {:.2}, {:.2}) terrain_y={:.2} diff={:.2}",
                            i, p.x, p.y, p.z, terrain_y, p.y - terrain_y,
                        );
                    }
                    self.readback_logged = true;
                }
            }
        }
    }

    /// Sample per-snapshot CPU-side state that follows the published cadence:
    /// overlays/histories, replay, and the food/heatmap dirty flags.
    fn on_fresh_snapshot(&mut self) {
        self.record_overlays_and_histories();
        self.record_replay_tick();
        self.heatmap_dirty = true;
        self.food_dirty = true;
    }

    /// Apply selected-agent telemetry to the matching agent's derived caches.
    ///
    /// Telemetry owns only the selected agent's vision color and the derived
    /// brain fields; the physics snapshot stays authoritative for everything
    /// else. Ignored when the telemetry is for an agent that is no longer the
    /// selected one (e.g. an in-flight readback after a selection change).
    fn apply_telemetry(&mut self, agent_index: u32, telemetry: AgentTelemetry) {
        let Some(a) = self.agents.get_mut(self.selected_agent_idx) else {
            return;
        };
        if a.brain_idx != agent_index {
            return;
        }
        a.cached_frame.vision.color = telemetry.vision_color;
        a.cached_mean_attenuation = telemetry.mean_attenuation;
        a.cached_curiosity_bonus = telemetry.curiosity_bonus;
        a.cached_staleness = telemetry.staleness;
    }

    /// Emit the render-side runtime-decoupling counters at most once per second.
    ///
    /// The simulation-side counters are logged by the worker thread; together
    /// the two log lines show simulation progress decoupled from render cadence.
    /// Call once per rendered frame; the internal timer rate-limits the output.
    pub(crate) fn log_runtime_counters(&mut self) {
        /// Minimum wall-clock interval between runtime-counter log lines.
        const COUNTER_LOG_INTERVAL: Duration = Duration::from_secs(1);
        if self.last_counters_log.elapsed() < COUNTER_LOG_INTERVAL {
            return;
        }
        self.last_counters_log = Instant::now();
        let counters = &self.runtime_counters;
        log::debug!(
            "[RENDER] frames={} snapshots_applied={} generation_boundaries={}",
            counters.frames_rendered,
            counters.snapshots_applied,
            counters.generation_boundaries,
        );
    }
}
