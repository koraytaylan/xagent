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
use xagent_sandbox::agent::{mutate_brain_state, mutate_config, Agent, MAX_AGENTS};
use xagent_sandbox::governor::{reset_database, AdvanceResult, Governor};
use xagent_sandbox::ui::{EvolutionAction, EvolutionSnapshot, EvolutionState};
use xagent_shared::BrainConfig;

use crate::app::{App, GenTransition};

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
                        self.gpu_kernel = None;
                        self.pending_kernel = None;
                        self.pending_upload = None;
                        self.spawn_evolution_population();
                        self.ensure_gpu_kernel();
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
                    self.gpu_kernel = None;
                    self.pending_kernel = None;
                    self.pending_upload = None;
                    self.spawn_evolution_population();
                    self.ensure_gpu_kernel();
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
                self.gpu_kernel = None;
                self.pending_kernel = None;
                self.pending_upload = None;
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

    /// Evaluate the current generation and begin the async transition.
    /// The actual GPU work (readback, reset, upload) spans multiple frames
    /// via `poll_gen_transition`.
    pub(crate) fn advance_generation(&mut self) {
        // Persist the recording to SQLite before moving to the next generation
        if let (Some(ref recording), Some(ref mut gov)) = (&self.recording, &mut self.governor) {
            gov.store_recording(recording);
        }
        self.last_recording = self.recording.take();
        let wall_secs = self.evo_wall_accumulated
            + self
                .evo_wall_segment_start
                .map(|s| s.elapsed().as_secs_f64())
                .unwrap_or(0.0);

        // Evaluate fitness and decide whether evolution continues.
        // Only kick off the async brain-state readback when the result is
        // Continue — when Finished, no readback is needed.
        let (result, readback_requested) = {
            let gov = match self.governor.as_mut() {
                Some(g) => g,
                None => return,
            };

            let fitness = gov.evaluate(&self.agents);
            gov.log_generation(&fitness);
            gov.update_wall_time(wall_secs);

            let result = gov.advance(&fitness);

            // Only read back the best agent's brain state when evolution
            // continues — Finished needs no inherited state.
            let mut readback_requested = false;
            if matches!(result, AdvanceResult::Continue { .. }) {
                let best_idx = fitness.first().map(|f| f.agent_index).unwrap_or(0);
                if let Some(a) = self.agents.get(best_idx) {
                    if let Some(mk) = self.gpu_kernel.as_mut() {
                        if mk.request_agent_state(a.brain_idx) {
                            readback_requested = true;
                        }
                    }
                }
            }

            (result, readback_requested)
        };

        match result {
            AdvanceResult::Continue { .. } if readback_requested => {
                self.gen_transition = Some(GenTransition::AwaitingReadback { result });
            }
            AdvanceResult::Continue { .. } => {
                // No readback was issued (no best agent or no GPU kernel) —
                // skip directly to reset to avoid deadlocking in AwaitingReadback.
                self.gen_transition = Some(GenTransition::AwaitingReset {
                    result,
                    inherited_state: None,
                    spawned: false,
                });
            }
            AdvanceResult::Finished { .. } => {
                // Skip readback phase entirely — go straight to finish.
                self.gen_transition = Some(GenTransition::AwaitingReset {
                    result,
                    inherited_state: None,
                    spawned: false,
                });
            }
        }
    }

    /// Drive the multi-frame generation transition state machine.
    /// Called every frame; returns true while a transition is still in progress.
    pub(crate) fn poll_gen_transition(&mut self) -> bool {
        let transition = match self.gen_transition.take() {
            Some(t) => t,
            None => return false,
        };

        match transition {
            GenTransition::AwaitingReadback { result } => {
                if self.gpu_kernel.is_none() {
                    // No kernel — proceed without inherited state.
                    self.gen_transition = Some(GenTransition::AwaitingReset {
                        result,
                        inherited_state: None,
                        spawned: false,
                    });
                    return true;
                }

                // Poll for the async agent state readback.
                let collected = self.gpu_kernel.as_mut().unwrap().try_collect_agent_state();

                match collected {
                    Some(Some(state)) => {
                        // Readback complete — move to reset phase.
                        self.gen_transition = Some(GenTransition::AwaitingReset {
                            result,
                            inherited_state: Some(state),
                            spawned: false,
                        });
                    }
                    Some(None) => {
                        // map_async error — proceed without inherited state
                        // rather than hanging in AwaitingReadback forever.
                        self.gen_transition = Some(GenTransition::AwaitingReset {
                            result,
                            inherited_state: None,
                            spawned: false,
                        });
                    }
                    None => {
                        // Still waiting — re-enqueue for next frame.
                        self.gen_transition = Some(GenTransition::AwaitingReadback { result });
                    }
                }
                true
            }
            GenTransition::AwaitingReset {
                result,
                inherited_state,
                spawned,
            } => {
                let (done, spawned) =
                    self.try_finish_generation(&result, inherited_state.as_ref(), spawned);
                if !done {
                    // Reset not ready yet — re-enqueue for next frame.
                    self.gen_transition = Some(GenTransition::AwaitingReset {
                        result,
                        inherited_state,
                        spawned,
                    });
                    return true;
                }
                false
            }
        }
    }

    /// Try to complete a generation transition. Returns `(done, spawned)`
    /// where `done` is false if the GPU staging buffers aren't ready yet
    /// (caller should retry next frame), and `spawned` tracks whether
    /// population has been spawned (to avoid re-spawning on retry).
    pub(crate) fn try_finish_generation(
        &mut self,
        result: &AdvanceResult,
        inherited_state: Option<&AgentBrainState>,
        mut spawned: bool,
    ) -> (bool, bool) {
        match result {
            AdvanceResult::Continue {
                configs,
                messages,
                mutation_strength,
            } => {
                if !spawned {
                    for msg in messages {
                        self.log_msg(msg.clone());
                    }
                    self.spawn_population_from_configs(configs);
                    spawned = true;
                }
                let repeats = self.governor_config.eval_repeats.max(1);

                // Reuse existing kernel if population size matches,
                // otherwise drop and recreate in background.
                let can_reuse = self
                    .gpu_kernel
                    .as_ref()
                    .is_some_and(|mk| mk.agent_count() == self.agents.len() as u32);

                if can_reuse {
                    let mk = self.gpu_kernel.as_mut().unwrap();
                    // Non-blocking reset — return false to retry next frame
                    // instead of falling back to a blocking busy-wait.
                    if !mk.try_reset_agents(&self.brain_config) {
                        return (false, spawned);
                    }
                    let agent_data: Vec<_> = self
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
                        .collect();
                    mk.upload_agents(&agent_data);
                } else {
                    self.gpu_kernel = None;
                    self.pending_kernel = None;
                    self.pending_upload = None;
                    self.ensure_gpu_kernel();
                    // Kernel is being recreated in the background —
                    // keep the transition active until it's ready.
                    if self.gpu_kernel.is_none() {
                        return (false, spawned);
                    }
                }

                // Inherit learned weights: champions get exact brain state,
                // mutants get perturbed weights for neuroevolution.
                if let Some(state) = inherited_state {
                    if let Some(ref mk) = self.gpu_kernel {
                        let ms = *mutation_strength;
                        let n = self.agents.len();
                        // Pre-clone once for all champion slots to avoid
                        // cloning inside the hot closure on every call.
                        let champion = state.clone();
                        mk.batch_write_agent_states(n, |i| {
                            if i < repeats {
                                champion.clone()
                            } else {
                                mutate_brain_state(state, ms)
                            }
                        });
                    } else {
                        // Kernel is being recreated in the background
                        // (population size changed). Stash inherited state
                        // so it can be applied once the kernel is ready.
                        self.deferred_inherited = Some((state.clone(), *mutation_strength));
                    }
                }
            }
            AdvanceResult::Finished { messages } => {
                for msg in messages {
                    self.log_msg(msg.clone());
                }
                self.evo_snapshot.state = EvolutionState::Paused;
                self.paused = true;
            }
        }
        (true, spawned)
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
