//! GPU kernel lifecycle and world/agent upload staging.
//!
//! Owns `App::ensure_gpu_kernel`, which drives creation of the fused
//! `GpuKernel` on a background thread, collects its data on the main thread,
//! and applies any brain state deferred from a generation transition that
//! straddled the kernel recreation.

use glam::Vec3;

use xagent_brain::buffers::{
    PHYS_STRIDE, P_ALIVE, P_DEATH_COUNT, P_ENERGY, P_EXPLORATION_RATE_OUT, P_FACING_X, P_FACING_Y,
    P_FACING_Z, P_FATIGUE_FACTOR_OUT, P_FOOD_COUNT, P_GRADIENT_OUT, P_INTEGRITY, P_LAST_DEATH_TICK,
    P_MAX_ENERGY, P_MAX_INTEGRITY, P_MOTOR_FWD_OUT, P_MOTOR_TURN_OUT, P_POS_X, P_POS_Y, P_POS_Z,
    P_PREDICTION_ERROR, P_TICKS_ALIVE, P_URGENCY_OUT, P_VEL_X, P_VEL_Y, P_VEL_Z, P_YAW,
};
use xagent_brain::GpuKernel;
use xagent_sandbox::agent::mutate_brain_state;

use crate::app::{App, PendingUpload, SIM_DT};

impl App {
    /// Ensure GpuKernel is initialized for the current population.
    pub(crate) fn ensure_gpu_kernel(&mut self) {
        if self.gpu_kernel.is_some() {
            return;
        }
        if self.agents.is_empty() {
            return;
        }
        let world = match &self.world {
            Some(w) => w,
            None => return,
        };

        // Check if background creation finished.
        if let Some(ref handle) = self.pending_kernel {
            if handle.is_finished() {
                let handle = self.pending_kernel.take().unwrap();
                let mk = handle
                    .join()
                    .expect("fused kernel background thread panicked");
                // Upload world + agent data (fast, main thread).
                if let Some(upload) = self.pending_upload.take() {
                    mk.upload_world(
                        &upload.heights,
                        &upload.biomes,
                        &upload.food_pos,
                        &upload.food_consumed,
                        &upload.food_timers,
                    );
                    mk.upload_agents(&upload.agent_data);
                }
                let ac = mk.agent_count();

                // Apply deferred inherited state from a generation transition
                // that occurred while the kernel was being recreated.
                if let Some((ref state, mutation_strength)) = self.deferred_inherited.take() {
                    let repeats = self.governor_config.eval_repeats.max(1);
                    let n = self.agents.len();
                    let champion = state.clone();
                    mk.batch_write_agent_states(n, |i| {
                        if i < repeats {
                            champion.clone()
                        } else {
                            mutate_brain_state(state, mutation_strength)
                        }
                    });
                }

                self.gpu_kernel = Some(mk);
                self.log_msg(format!("[GPU] GpuKernel ready ({} agents)", ac));
            }
            return; // still creating
        }

        // Collect data for upload (kept on main thread).
        let agent_count = self.agents.len() as u32;
        let food_count = world.food_items.len();
        let brain_config = self.brain_config.clone();
        let world_config = world.config.clone();

        self.pending_upload = Some(PendingUpload {
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
        });

        // Spawn background thread for device + shader compilation.
        self.pending_kernel = Some(std::thread::spawn(move || {
            GpuKernel::new(agent_count, food_count, &brain_config, &world_config)
        }));
        self.log_msg("[GPU] Creating GpuKernel (background)...".into());
    }

    /// Advance the simulation for one frame: dispatch fixed-timestep ticks,
    /// collect the every-frame state readback, record per-tick overlay/history
    /// data, and check for generation completion.
    ///
    /// Ticks are skipped while paused, while a generation transition is in
    /// flight (to avoid GPU contention with the async readback/reset), and while
    /// the kernel is being recreated in the background (to prevent a catch-up
    /// hitch when it lands).
    pub(crate) fn step_simulation(&mut self, dt: f32) {
        let sim_active =
            !self.paused && self.gen_transition.is_none() && self.pending_kernel.is_none();
        if !sim_active {
            return;
        }

        self.dispatch_sim_ticks(dt);

        // Collect GPU staging data and update agent positions regardless of
        // whether ticks were dispatched this frame, so visuals stay smooth at
        // low speed multipliers where dispatches happen infrequently.
        self.collect_state_readback();
        self.record_overlays_and_histories();
        self.heatmap_dirty = true;

        // ── Generation completion check (after tick batch) ──
        if self.gen_transition.is_none() {
            if let Some(gov) = &self.governor {
                if gov.generation_complete() {
                    self.advance_generation();
                }
            }
        }
    }

    /// Dispatch the fixed-timestep simulation batch for this frame.
    ///
    /// Accumulates wall time into a fixed-step budget and dispatches only when
    /// at least one full brain-tick stride has accumulated, so every dispatch
    /// includes a brain cycle and produces motor commands. The accumulator keeps
    /// its fractional remainder across frames so no sim-time is lost.
    fn dispatch_sim_ticks(&mut self, dt: f32) {
        let sim_delta_time = SIM_DT as f64;
        self.sim_accumulator += dt as f64 * self.speed_multiplier as f64;
        // Only dispatch when the accumulator has enough for at least
        // brain_tick_stride ticks; sub-stride dispatches would be physics-only
        // (no brain cycles), leaving agents with stale motor outputs.
        let min_dispatch = self
            .gpu_kernel
            .as_ref()
            .map_or(10, |kernel| kernel.brain_tick_stride());
        // Cap must allow at least min_dispatch ticks to accumulate, otherwise
        // low speeds (1x, 2x) can never reach the dispatch threshold.
        let max_accumulator =
            sim_delta_time * (self.speed_multiplier as f64 * 3.0).max(min_dispatch as f64 + 2.0);
        self.sim_accumulator = self.sim_accumulator.min(max_accumulator);
        let raw_ticks = ((self.sim_accumulator / sim_delta_time) as u32)
            .min(self.gpu_tick_budget)
            .min(500);
        let ticks_to_run = if raw_ticks >= min_dispatch {
            raw_ticks
        } else {
            0
        };

        if ticks_to_run == 0 {
            return;
        }

        let mut dispatched = false;
        if let Some(ref mut kernel) = self.gpu_kernel {
            kernel.dispatch_batch(self.tick, ticks_to_run);

            self.sim_accumulator -= ticks_to_run as f64 * sim_delta_time;

            self.gpu_tick_budget =
                (self.gpu_tick_budget + self.gpu_tick_budget / 4 + 1).min(64_000);

            self.tick += ticks_to_run as u64;
            self.tps_tick_count += ticks_to_run as u64;
            self.snap_dirty = true;

            if let Some(gov) = &mut self.governor {
                for _ in 0..ticks_to_run {
                    gov.tick();
                }
            }

            // Async telemetry readback for the selected agent.
            // `request_agent_telemetry` is gated internally: it no-ops if a
            // readback for the same agent is already pending, and clears the old
            // pending if the agent changed.
            if self.selected_agent_idx < self.agents.len() {
                let brain_idx = self.agents[self.selected_agent_idx].brain_idx;

                kernel.request_agent_telemetry(brain_idx);

                // Collect any completed readback (non-blocking)
                if let Some(tel) = kernel.try_collect_telemetry() {
                    let a = &mut self.agents[self.selected_agent_idx];
                    // Only update fields NOT already populated by the every-frame
                    // physics readback (collect_state_readback). Physics readback
                    // sets: cached_motor, cached_gradient, cached_urgency,
                    // cached_fatigue_factor, cached_prediction_error,
                    // cached_exploration_rate.
                    a.cached_frame.vision.color = tel.vision_color;
                    a.cached_mean_attenuation = tel.mean_attenuation;
                    a.cached_curiosity_bonus = tel.curiosity_bonus;
                    a.cached_staleness = tel.staleness;
                }
            }

            self.food_dirty = true;
            dispatched = true;
        }

        if dispatched {
            self.record_replay_tick();
        }
    }

    /// Collect the every-frame physics readback and apply it to agent bodies.
    ///
    /// Authoritative for position/yaw/alive/energy/integrity/velocity and the
    /// `cached_*` motor/gradient/urgency/prediction/exploration/fatigue fields.
    fn collect_state_readback(&mut self) {
        let Some(kernel) = self.gpu_kernel.as_mut() else {
            return;
        };
        if !kernel.try_collect_state() {
            return;
        }
        let state = kernel.cached_state();
        for i in 0..self.agents.len() {
            let base = i * PHYS_STRIDE;
            // Guard every physics field we read below. Use the stride's highest
            // offset (PHYS_STRIDE - 1) so adding new fields doesn't silently
            // leave the guard stale.
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
            let new_deaths = state[base + P_DEATH_COUNT] as u32;
            let gpu_death_tick = state[base + P_LAST_DEATH_TICK] as u64;
            a.apply_death_count_readback(new_deaths, gpu_death_tick, self.tick.saturating_sub(1));
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

        self.snap_dirty = true;
        self.hud_dirty = true;

        // Diagnostic: log first readback Y vs terrain height
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
}
