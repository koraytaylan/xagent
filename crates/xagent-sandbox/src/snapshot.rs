//! Per-tick snapshot assembly helpers for the egui/sparkline UI.
//!
//! Hosts the free `record_agent_histories` helper plus the `App` methods that
//! assemble the throttled per-frame UI snapshots: the agent list
//! (`rebuild_agent_snapshots`), the evolution overview (`update_evo_snapshot`),
//! the mini-map world state (`update_world_snapshot`), and the lazily-built
//! biome image (`build_biome_image`). `record_overlays_and_histories` captures
//! the per-tick heatmap/trail/sparkline data that feeds these snapshots.

use std::time::Instant;

use xagent_brain::buffers::{FOOD_STATE_STRIDE, F_POS_X, F_POS_Z, F_RESPAWN_TIMER};
use xagent_sandbox::agent::Agent;
use xagent_sandbox::ui::AgentSnapshot;

use crate::app::{App, REBUILD_THROTTLE};

/// Record per-tick telemetry into agent sparkline histories.
pub(crate) fn record_agent_histories(agent: &mut Agent) {
    let cap = 10_000;
    macro_rules! push_hist {
        ($h:expr, $v:expr) => {
            if $h.len() >= cap {
                $h.pop_front();
            }
            $h.push_back($v);
        };
    }
    push_hist!(
        agent.prediction_error_history,
        agent.cached_prediction_error
    );
    push_hist!(
        agent.exploration_rate_history,
        agent.cached_exploration_rate
    );
    let ef = agent.body.body.internal.energy / agent.body.body.internal.max_energy.max(0.001);
    push_hist!(agent.energy_history, ef.clamp(0.0, 1.0));
    let inf =
        agent.body.body.internal.integrity / agent.body.body.internal.max_integrity.max(0.001);
    push_hist!(agent.integrity_history, inf.clamp(0.0, 1.0));
    push_hist!(agent.fatigue_history, agent.cached_fatigue_factor);
}

impl App {
    /// Record per-tick heatmap/trail occupancy for living agents and append the
    /// latest telemetry to every agent's sparkline histories.
    pub(crate) fn record_overlays_and_histories(&mut self) {
        // Heatmap + trail recording
        if let Some(world) = &self.world {
            for agent in &mut self.agents {
                if agent.body.body.alive {
                    agent.record_heatmap(world.config.world_size);
                    agent.record_trail();
                }
            }
        }

        // Sparkline histories
        for agent in &mut self.agents {
            record_agent_histories(agent);
        }
    }

    /// Rebuild the cached per-agent UI snapshots (throttled to ~10 Hz).
    ///
    /// Forces a rebuild when the agent count or chart window changed; otherwise
    /// honors the snapshot throttle. Each snapshot carries the tail of the
    /// agent's history buffers sized to twice the chart window.
    pub(crate) fn rebuild_agent_snapshots(&mut self) {
        // Force rebuild when agents changed size or chart window changed.
        if self.cached_agent_snaps.len() != self.agents.len()
            || self.chart_window != self.last_snap_chart_window
        {
            self.snap_dirty = true;
        }
        let snap_rebuild_due = self.snap_dirty
            && (self.paused || self.last_snapshot_rebuild.elapsed() >= REBUILD_THROTTLE);
        if !snap_rebuild_due {
            return;
        }

        let snap_window = self.chart_window * 2;
        self.cached_agent_snaps = self
            .agents
            .iter()
            .map(|a| {
                let tail = |d: &std::collections::VecDeque<f32>| -> Vec<f32> {
                    let skip = d.len().saturating_sub(snap_window);
                    d.iter().skip(skip).copied().collect()
                };
                AgentSnapshot {
                    id: a.id,
                    generation: a.generation,
                    energy: a.body.body.internal.energy,
                    max_energy: a.body.body.internal.max_energy,
                    integrity: a.body.body.internal.integrity,
                    max_integrity: a.body.body.internal.max_integrity,
                    alive: a.body.body.alive,
                    deaths: a.death_count,
                    color: a.color,
                    longest_life: a.longest_life,
                    exploration_rate: a.cached_exploration_rate,
                    prediction_error: a.cached_prediction_error,
                    forward_weight_norm: 0.0, // GPU telemetry TBD
                    turn_weight_norm: 0.0,    // GPU telemetry TBD
                    prediction_error_history: tail(&a.prediction_error_history),
                    exploration_rate_history: tail(&a.exploration_rate_history),
                    energy_history: tail(&a.energy_history),
                    integrity_history: tail(&a.integrity_history),
                    gradient: a.cached_gradient,
                    urgency: a.cached_urgency,
                    food_consumed: a.food_consumed,
                    total_ticks_alive: a.total_ticks_alive,
                    motor_forward: a.cached_motor.forward,
                    motor_turn: a.cached_motor.turn,
                    phase: "GPU", // GPU telemetry TBD
                    vision_color: a.cached_frame.vision.color.clone(),
                    vision_width: a.cached_frame.vision.width,
                    vision_height: a.cached_frame.vision.height,
                    position: [
                        a.body.body.position.x,
                        a.body.body.position.y,
                        a.body.body.position.z,
                    ],
                    yaw: a.body.yaw,
                    mean_attenuation: a.cached_mean_attenuation,
                    curiosity_bonus: a.cached_curiosity_bonus,
                    fatigue_factor: a.cached_fatigue_factor,
                    staleness: a.cached_staleness,
                    fatigue_history: tail(&a.fatigue_history),
                }
            })
            .collect();
        self.snap_dirty = false;
        self.last_snap_chart_window = self.chart_window;
        if !self.paused {
            self.last_snapshot_rebuild = Instant::now();
        }
    }

    /// Refresh the evolution snapshot fields the UI reads (governor progress,
    /// accumulated wall time, and the once-per-second ticks/sec average).
    pub(crate) fn update_evo_snapshot(&mut self) {
        if let Some(gov) = &mut self.governor {
            self.evo_snapshot.gen_tick = gov.gen_tick;
            self.evo_snapshot.generation = gov.generation;
            self.evo_snapshot.tree_nodes = gov.tree_nodes();
            self.evo_snapshot.current_node_id = gov.current_node_id;
            self.evo_snapshot
                .fitness_history
                .clone_from(gov.fitness_history_by_island());
            self.evo_snapshot.best_fitness = gov.best_score();
        }
        let wall = self.evo_wall_accumulated
            + self
                .evo_wall_segment_start
                .map(|s| s.elapsed().as_secs_f64())
                .unwrap_or(0.0);
        self.evo_snapshot.wall_time_secs = wall;
        let tps_elapsed = self.tps_last_reset.elapsed().as_secs_f64();
        if tps_elapsed >= 1.0 {
            self.tps_display = self.tps_tick_count as f64 / tps_elapsed;
            self.tps_tick_count = 0;
            self.tps_last_reset = Instant::now();
        }
        self.evo_snapshot.ticks_per_sec = self.tps_display;
    }

    /// Refresh the mini-map world snapshot's food positions and world size.
    ///
    /// Prefers the GPU food state (authoritative once the kernel runs), falling
    /// back to the CPU world's food items before the first readback.
    pub(crate) fn update_world_snapshot(&mut self) {
        let Some(world) = &self.world else {
            return;
        };
        let gpu_food = self
            .gpu_kernel
            .as_ref()
            .and_then(|mk| mk.cached_food_state());
        if let Some(food) = gpu_food {
            self.world_snapshot.food_positions = food
                .chunks_exact(FOOD_STATE_STRIDE)
                .filter(|c| c[F_RESPAWN_TIMER] <= 0.0)
                .map(|c| [c[F_POS_X], c[F_POS_Z]])
                .collect();
        } else {
            self.world_snapshot.food_positions = world
                .food_items
                .iter()
                .filter(|f| !f.consumed)
                .map(|f| [f.position.x, f.position.z])
                .collect();
        }
        self.world_snapshot.world_size = world.config.world_size;
    }

    /// Build the mini-map biome image when it has not been uploaded yet.
    ///
    /// Returns `None` once the biome texture exists, so callers upload it only
    /// on the first frame (the texture upload itself requires an egui context).
    pub(crate) fn build_biome_image(&self) -> Option<egui::ColorImage> {
        if self.world_snapshot.biome_texture.is_some() {
            return None;
        }
        let world = self.world.as_ref()?;
        use xagent_sandbox::world::biome::BiomeType;
        let res = 256usize;
        let ws = world.config.world_size;
        let half = ws / 2.0;
        let cell = ws / res as f32;
        let mut pixels = Vec::with_capacity(res * res);
        for row in 0..res {
            let z = -half + (row as f32 + 0.5) * cell;
            for col in 0..res {
                let x = -half + (col as f32 + 0.5) * cell;
                let biome = world.biome_map.biome_at(x, z);
                let c = match biome {
                    BiomeType::FoodRich => egui::Color32::from_rgb(25, 70, 20),
                    BiomeType::Barren => egui::Color32::from_rgb(60, 50, 30),
                    BiomeType::Danger => egui::Color32::from_rgb(80, 25, 15),
                };
                pixels.push(c);
            }
        }
        Some(egui::ColorImage {
            size: [res, res],
            pixels,
        })
    }
}
