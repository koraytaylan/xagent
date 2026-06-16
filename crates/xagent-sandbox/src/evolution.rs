//! Evolution action handling, population spawning, and the async
//! generation-transition state machine.
//!
//! This module hosts `App::handle_evolution_action` plus the cluster of
//! methods that together drive a generation boundary — evaluating fitness,
//! reading back the champion's brain state, resetting or recreating the
//! GPU kernel, and spawning the next generation's agents.

use std::time::Instant;

use glam::Vec3;
use rand::Rng;

use xagent_brain::AgentBrainState;
use xagent_sandbox::agent::{mutate_config, Agent, MAX_AGENTS};
use xagent_sandbox::governor::{reset_database, AdvanceResult, Governor};
use xagent_sandbox::ui::{EvolutionAction, EvolutionSnapshot, EvolutionState};
use xagent_shared::BrainConfig;

use crate::app::{App, PendingGeneration};
use crate::sim_runtime::{InheritedBrain, ResetRequest, SimCommand};

impl App {
    /// Spawn a new agent with the given BrainConfig at a safe random position
    /// (not in a danger biome). The brain_idx is the agent's array index.
    pub(crate) fn spawn_agent(&mut self, config: BrainConfig, generation: u32) {
        if self.agents.len() >= MAX_AGENTS {
            return;
        }
        let Some(world) = &self.world else { return };

        let pos = world.safe_spawn_position();

        let id = self.next_agent_id;
        self.next_agent_id += 1;

        let brain_idx = self.agents.len() as u32;
        let mut agent = Agent::new(id, pos, brain_idx, config, self.tick);
        agent.generation = generation;

        self.agents.push(agent);
    }

    pub(crate) fn handle_evolution_action(&mut self, action: EvolutionAction) {
        match action {
            EvolutionAction::None => {}
            EvolutionAction::Start => {
                let brain_config = self.evo_snapshot.edit_brain.clone();
                let gov_config = self.evo_snapshot.edit_governor.clone();
                let world_json = serde_json::to_string(&self.world_config).unwrap_or_default();
                match Governor::new(
                    &self.db_path,
                    gov_config.clone(),
                    &brain_config,
                    &world_json,
                ) {
                    Ok(gov) => {
                        self.governor = Some(gov);
                        self.governor_config = gov_config.clone();
                        self.brain_config = brain_config;
                        self.evo_snapshot.state = EvolutionState::Running;
                        self.evo_snapshot.population_size = gov_config.population_size;
                        self.evo_snapshot.tick_budget = gov_config.tick_budget;
                        self.evo_snapshot.elitism_count = gov_config.elitism_count;
                        self.evo_snapshot.patience = gov_config.patience;
                        self.evo_snapshot.max_generations = gov_config.max_generations;
                        self.evo_snapshot.eval_repeats = gov_config.eval_repeats;
                        self.evo_snapshot.num_islands = gov_config.num_islands;
                        self.evo_snapshot.migration_interval = gov_config.migration_interval;
                        self.evo_wall_accumulated = 0.0;
                        self.evo_wall_segment_start = Some(Instant::now());
                        self.tps_tick_count = 0;
                        self.tps_last_reset = Instant::now();
                        self.tps_display = 0.0;
                        self.paused = false;
                        self.stop_sim_worker();
                        self.spawn_evolution_population();
                        self.start_sim_worker();
                        self.log_msg("[EVOLUTION] Started new run".into());
                    }
                    Err(e) => {
                        self.log_msg(format!("[EVOLUTION] Failed to start: {}", e));
                    }
                }
            }
            EvolutionAction::Resume => match Governor::resume(&self.db_path) {
                Ok(gov) => {
                    let cfg = gov.current_config();
                    self.evo_snapshot.state = EvolutionState::Running;
                    self.evo_snapshot.generation = gov.generation;
                    self.governor_config = gov.config.clone();
                    if let Some(c) = &cfg {
                        self.brain_config = c.clone();
                    }
                    self.governor = Some(gov);
                    self.evo_wall_accumulated = 0.0;
                    self.evo_wall_segment_start = Some(Instant::now());
                    self.tps_tick_count = 0;
                    self.tps_last_reset = Instant::now();
                    self.tps_display = 0.0;
                    self.paused = false;
                    self.stop_sim_worker();
                    self.spawn_evolution_population();
                    self.start_sim_worker();
                    self.log_msg("[EVOLUTION] Resumed from database".into());
                }
                Err(e) => {
                    self.log_msg(format!("[EVOLUTION] Failed to resume: {}", e));
                }
            },
            EvolutionAction::Pause => {
                self.evo_snapshot.state = EvolutionState::Paused;
                self.paused = true;
                self.snap_dirty = true;
                if let Some(start) = self.evo_wall_segment_start.take() {
                    self.evo_wall_accumulated += start.elapsed().as_secs_f64();
                }
                self.tps_display = 0.0;
                self.log_msg("[EVOLUTION] Paused".into());
            }
            EvolutionAction::Unpause => {
                self.evo_snapshot.state = EvolutionState::Running;
                self.paused = false;
                self.evo_wall_segment_start = Some(Instant::now());
                self.tps_tick_count = 0;
                self.tps_last_reset = Instant::now();
                self.log_msg("[EVOLUTION] Resumed".into());
            }
            EvolutionAction::Reset => {
                self.governor = None;
                self.stop_sim_worker();
                self.agents.clear();
                self.next_agent_id = 0;
                self.tick = 0;
                self.paused = true;
                if let Err(e) = reset_database(&self.db_path) {
                    self.log_msg(format!("[EVOLUTION] Failed to reset DB: {}", e));
                }
                self.evo_snapshot = EvolutionSnapshot::default();
                self.evo_snapshot.edit_brain = self.brain_config.clone();
                self.evo_snapshot.edit_governor = self.governor_config.clone();
                self.log_msg("[EVOLUTION] Reset — ready to start fresh".into());
            }
        }
    }

    pub(crate) fn spawn_evolution_population(&mut self) {
        self.agents.clear();
        self.next_agent_id = 0;
        self.tick = 0;
        let seed = if let Some(gov) = &self.governor {
            gov.current_config().unwrap_or(self.brain_config.clone())
        } else {
            self.brain_config.clone()
        };
        let pop_size = self.governor_config.population_size;
        let repeats = self.governor_config.eval_repeats.max(1);
        let unique_count = (pop_size / repeats).max(1);

        // Build unique configs matching breed_next_generation structure
        // so that reduce_fitness grouping (agent_index / eval_repeats)
        // correctly averages same-config runs.
        let mut unique_configs = vec![seed.clone()]; // slot 0: champion
        for _ in 1..unique_count {
            unique_configs.push(mutate_config(&seed));
        }

        // Repeat each config eval_repeats times
        for uc in &unique_configs {
            for _ in 0..repeats {
                if self.agents.len() >= pop_size {
                    break;
                }
                self.spawn_agent(uc.clone(), 0);
            }
        }

        // Start replay recording
        if let Some(world) = &self.world {
            let agent_info: Vec<(u32, [f32; 3])> =
                self.agents.iter().map(|a| (a.id, a.color)).collect();
            let initial_food: Vec<[f32; 3]> = world
                .food_items
                .iter()
                .map(|f| [f.position.x, f.position.y, f.position.z])
                .collect();
            let gen = self.governor.as_ref().map_or(0, |g| g.generation as u32);
            self.recording = Some(xagent_sandbox::replay::GenerationRecording::new(
                gen,
                &agent_info,
                &initial_food,
                self.governor_config.tick_budget as usize,
                self.brain_config.vision_width,
                self.brain_config.vision_height,
            ));
        }
    }

    pub(crate) fn spawn_population_from_configs(&mut self, configs: &[BrainConfig]) {
        self.agents.clear();
        self.next_agent_id = 0;
        self.tick = 0;
        for cfg in configs {
            self.spawn_agent(cfg.clone(), 0);
        }

        // Start replay recording
        if let Some(world) = &self.world {
            let agent_info: Vec<(u32, [f32; 3])> =
                self.agents.iter().map(|a| (a.id, a.color)).collect();
            let initial_food: Vec<[f32; 3]> = world
                .food_items
                .iter()
                .map(|f| [f.position.x, f.position.y, f.position.z])
                .collect();
            let gen = self.governor.as_ref().map_or(0, |g| g.generation as u32);
            self.recording = Some(xagent_sandbox::replay::GenerationRecording::new(
                gen,
                &agent_info,
                &initial_food,
                self.governor_config.tick_budget as usize,
                self.brain_config.vision_width,
                self.brain_config.vision_height,
            ));
        }
    }

    /// Handle the worker reaching the generation tick budget.
    ///
    /// The end-of-generation snapshot has already been applied to the CPU
    /// agents by the event drain. Persist the recording, evaluate fitness, and
    /// — when evolution continues — request the champion's brain state from the
    /// worker so the next generation can inherit it. `Finished` results pause
    /// the run (the worker has already paused itself at the budget).
    pub(crate) fn on_generation_budget_reached(&mut self) {
        // Persist the recording to SQLite before moving to the next generation.
        if let (Some(ref recording), Some(ref mut gov)) = (&self.recording, &mut self.governor) {
            gov.store_recording(recording);
        }
        self.last_recording = self.recording.take();
        let wall_secs = self.evo_wall_accumulated
            + self
                .evo_wall_segment_start
                .map(|s| s.elapsed().as_secs_f64())
                .unwrap_or(0.0);

        let (result, champion_brain_idx) = {
            let gov = match self.governor.as_mut() {
                Some(g) => g,
                None => return,
            };

            let fitness = gov.evaluate(&self.agents);
            gov.log_generation(&fitness);
            gov.update_wall_time(wall_secs);

            let result = gov.advance(&fitness);

            // The champion is the top-ranked agent; map its array index to its
            // kernel brain index for the readback request.
            let champion_brain_idx = if matches!(result, AdvanceResult::Continue { .. }) {
                fitness
                    .first()
                    .map(|f| f.agent_index)
                    .and_then(|idx| self.agents.get(idx))
                    .map(|a| a.brain_idx)
            } else {
                None
            };

            (result, champion_brain_idx)
        };

        match result {
            AdvanceResult::Continue { .. } => {
                if let Some(brain_idx) = champion_brain_idx {
                    let request_id = self.champion_request_counter;
                    self.champion_request_counter += 1;
                    if let Some(runtime) = &self.sim_runtime {
                        runtime.send(SimCommand::RequestAgentState {
                            agent_index: brain_idx,
                            request_id,
                        });
                    }
                    self.pending_generation = Some(PendingGeneration {
                        result,
                        champion_request_id: request_id,
                    });
                } else {
                    // No champion to inherit (no agents) — reset straight away.
                    self.finish_generation_continue(result, None);
                }
            }
            AdvanceResult::Finished { messages } => {
                for msg in &messages {
                    self.log_msg(msg.clone());
                }
                self.evo_snapshot.state = EvolutionState::Paused;
                self.paused = true;
            }
        }
    }

    /// Apply the worker's champion brain-state reply and start the next
    /// generation, ignoring a reply whose id does not match the pending request.
    pub(crate) fn on_champion_state(&mut self, request_id: u64, state: Option<AgentBrainState>) {
        let Some(pending) = self.pending_generation.take() else {
            return;
        };
        if pending.champion_request_id != request_id {
            // Stale reply (a newer request superseded it) — keep waiting.
            self.pending_generation = Some(pending);
            return;
        }
        self.finish_generation_continue(pending.result, state);
    }

    /// Spawn the next generation and command the worker to reset to it.
    ///
    /// The first `eval_repeats` slots inherit the champion's exact brain state;
    /// the rest inherit mutated copies. The worker applies the inheritance and
    /// resumes (unless the user paused during the handoff).
    fn finish_generation_continue(
        &mut self,
        result: AdvanceResult,
        champion_state: Option<AgentBrainState>,
    ) {
        let AdvanceResult::Continue {
            configs,
            messages,
            mutation_strength,
        } = result
        else {
            self.pending_generation = None;
            return;
        };
        for msg in &messages {
            self.log_msg(msg.clone());
        }
        self.spawn_population_from_configs(&configs);

        let Some(upload) = self.build_pending_upload() else {
            self.pending_generation = None;
            return;
        };
        let champion_slots = self.governor_config.eval_repeats.max(1);
        let inherited = champion_state.map(|champion| InheritedBrain {
            champion,
            mutation_strength,
            champion_slots,
        });
        let resume = !self.paused;
        let request = ResetRequest {
            upload,
            brain_config: self.brain_config.clone(),
            tick_budget: self.governor_config.tick_budget,
            inherited,
            resume,
        };
        if let Some(runtime) = &self.sim_runtime {
            runtime.send(SimCommand::ResetPopulation(Box::new(request)));
        }
        self.pending_generation = None;
    }

    /// Spawn a child agent near a parent, with mutated config.
    #[allow(dead_code)]
    pub(crate) fn spawn_child(&mut self, parent_idx: usize) {
        if self.agents.len() >= MAX_AGENTS {
            return;
        }
        let parent = &self.agents[parent_idx];
        let parent_id = parent.id;
        let parent_gen = parent.generation;
        let parent_pos = parent.body.body.position;
        let parent_config = parent.brain_config.clone();

        let child_config = mutate_config(&parent_config);
        let mut rng = rand::rng();
        let offset = Vec3::new(
            rng.random_range(-5.0..5.0_f32),
            0.0,
            rng.random_range(-5.0..5.0_f32),
        );
        let child_pos = parent_pos + offset;

        let id = self.next_agent_id;
        self.next_agent_id += 1;

        let Some(world) = &self.world else { return };
        let half = world.config.world_size / 2.0 - 1.0;
        let cx = child_pos.x.clamp(-half, half);
        let cz = child_pos.z.clamp(-half, half);
        let cy = world.terrain.height_at(cx, cz) + 1.0;

        let brain_idx = self.agents.len() as u32;
        let mut child = Agent::new(
            id,
            Vec3::new(cx, cy, cz),
            brain_idx,
            child_config,
            self.tick,
        );
        child.generation = parent_gen + 1;

        println!(
            "[REPRODUCE] Agent {} (gen {}) → child Agent {} (gen {}) at ({:.1}, {:.1})",
            parent_id, parent_gen, id, child.generation, cx, cz
        );
        // `mem_cost` / `proc_cost`: metabolic-cost proxies — kernel widths
        // are fixed at MEMORY_CAP=128, RECALL_K=16 (see issue #106).
        println!(
            "  Child config: mem_cost={} proc_cost={} dim={} lr={:.4} decay={:.4}",
            child.brain_config.memory_capacity,
            child.brain_config.processing_slots,
            child.brain_config.representation_dimension,
            child.brain_config.learning_rate,
            child.brain_config.decay_rate,
        );

        self.agents.push(child);
    }
}
