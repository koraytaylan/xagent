//! GPU kernel lifecycle and world/agent upload staging.
//!
//! Owns `App::ensure_gpu_kernel`, which drives creation of the fused
//! `GpuKernel` on a background thread, collects its data on the main thread,
//! and applies any brain state deferred from a generation transition that
//! straddled the kernel recreation.

use xagent_brain::GpuKernel;
use xagent_sandbox::agent::mutate_brain_state;

use crate::app::{App, PendingUpload};

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
}
