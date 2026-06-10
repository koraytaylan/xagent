//! Render-side helpers that build UI/overlay geometry from `App` state.
//!
//! Hosts the per-frame render orchestration (`render_frame`), the throttled
//! dynamic-mesh rebuilds (`rebuild_dynamic_meshes`), the agent instance buffer
//! upload (`update_agent_instances`), the HUD/text overlay rebuild
//! (`rebuild_hud_text`), plus `build_hud_bars` and the screen-space agent
//! picker (`pick_agent_at_cursor`). The name avoids colliding with the
//! library's `renderer` module.

use std::time::Instant;

use winit::event_loop::ActiveEventLoop;

use xagent_sandbox::agent::srgb_to_linear;
use xagent_sandbox::overlay;
use xagent_sandbox::renderer::font::TextItem;
use xagent_sandbox::renderer::hud::HudBar;
use xagent_sandbox::renderer::{GpuMesh, InstanceData};
use xagent_sandbox::ui::{EvolutionAction, Tab, TabContext};

use crate::app::{App, REBUILD_THROTTLE};
use crate::ui_chrome::{draw_agent_sidebar, draw_console, draw_top_bar, TopBarState};

impl App {
    /// Pick the agent closest to the cursor via screen-space projection.
    pub(crate) fn pick_agent_at_cursor(&mut self) {
        let Some(renderer) = &self.renderer else {
            return;
        };
        let w = renderer.config.width as f32;
        let h = renderer.config.height as f32;
        if w < 1.0 || h < 1.0 {
            return;
        }

        // Normalized device coordinates [-1, 1]
        let ndc_x = (self.cursor_pos.0 as f32 / w) * 2.0 - 1.0;
        let ndc_y = 1.0 - (self.cursor_pos.1 as f32 / h) * 2.0;

        let vp = self.camera.view_projection_matrix();
        let mut best_idx: Option<usize> = None;
        let mut best_dist_sq = f32::MAX;

        for (i, agent) in self.agents.iter().enumerate() {
            if !agent.body.body.alive {
                continue;
            }
            let pos = agent.body.body.position;
            let clip = vp * glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);
            if clip.w <= 0.0 {
                continue;
            } // behind camera
            let sx = clip.x / clip.w;
            let sy = clip.y / clip.w;
            let d = (sx - ndc_x).powi(2) + (sy - ndc_y).powi(2);
            if d < best_dist_sq {
                best_dist_sq = d;
                best_idx = Some(i);
            }
        }

        // Only select if click is reasonably close (within ~10% of screen)
        if let Some(idx) = best_idx {
            if best_dist_sq < 0.05 {
                self.selected_agent_idx = idx;
                self.agents[idx].trail_dirty = true;
                let a = &self.agents[idx];
                println!("[SELECT] Agent {} (gen {})", a.id, a.generation);
                self.hud_dirty = true;
            }
        }
    }

    /// Build HUD overlay bars for the selected agent.
    pub(crate) fn build_hud_bars(&self) -> Vec<HudBar> {
        let Some(agent) = self.agents.get(self.selected_agent_idx) else {
            return Vec::new();
        };

        let energy = agent.body.body.internal.energy_signal();
        let integrity = agent.body.body.internal.integrity_signal();
        let pred_err = agent.cached_prediction_error;
        let explore = agent.cached_exploration_rate;

        let bar_w = 0.35;
        let bar_h = 0.025;
        let left = -0.98;
        let top = 0.97;
        let gap = 0.035;
        let bg = [0.15, 0.15, 0.15];

        vec![
            // Energy — green
            HudBar {
                x: left,
                y: top,
                width: bar_w,
                height: bar_h,
                fill: energy,
                color: [0.2, 0.85, 0.2],
                bg_color: bg,
            },
            // Integrity — blue
            HudBar {
                x: left,
                y: top - gap,
                width: bar_w,
                height: bar_h,
                fill: integrity,
                color: [0.3, 0.5, 1.0],
                bg_color: bg,
            },
            // Prediction error — red
            HudBar {
                x: left,
                y: top - gap * 2.0,
                width: bar_w,
                height: bar_h,
                fill: pred_err,
                color: [0.95, 0.2, 0.15],
                bg_color: bg,
            },
            // Exploration rate — yellow
            HudBar {
                x: left,
                y: top - gap * 3.0,
                width: bar_w,
                height: bar_h,
                fill: explore,
                color: [0.95, 0.85, 0.1],
                bg_color: bg,
            },
        ]
    }

    /// Rebuild the throttled dynamic overlay meshes (food, heatmap, all-agent
    /// trails) and the per-frame selection marker.
    ///
    /// Food and trail rebuilds run at most every `REBUILD_THROTTLE` (or every
    /// frame while paused); the heatmap rebuilds when enabled and dirty.
    pub(crate) fn rebuild_dynamic_meshes(&mut self) {
        let mesh_rebuild_due = self.paused || self.last_mesh_rebuild.elapsed() >= REBUILD_THROTTLE;

        let mut did_rebuild = false;

        if self.food_dirty && mesh_rebuild_due {
            if let (Some(renderer), Some(world), Some(food_gpu)) =
                (&self.renderer, &self.world, &mut self.food_gpu)
            {
                let fm = world.food_mesh();
                food_gpu.update_from_mesh(&renderer.queue, &fm);
                self.food_dirty = false;
                did_rebuild = true;
            }
        }

        // ── rebuild heatmap overlay ─────────────────────────
        if self.heatmap_enabled && self.heatmap_dirty {
            if let (Some(renderer), Some(world), Some(heatmap_gpu)) =
                (&self.renderer, &self.world, &mut self.heatmap_gpu)
            {
                if let Some(agent) = self.agents.get(self.selected_agent_idx) {
                    let mesh = overlay::build_heatmap_mesh(
                        &agent.heatmap,
                        world.config.world_size,
                        &world.terrain,
                    );
                    heatmap_gpu.update_from_mesh(&renderer.queue, &mesh);
                }
            }
            self.heatmap_dirty = false;
        } else if !self.heatmap_enabled {
            if let Some(heatmap_gpu) = &mut self.heatmap_gpu {
                heatmap_gpu.num_indices = 0;
            }
        }

        // ── rebuild trail overlay for ALL agents (throttled) ──
        if mesh_rebuild_due {
            if let (Some(renderer), Some(trail_gpu)) = (&self.renderer, &mut self.trail_gpu) {
                let any_dirty = self.agents.iter().any(|a| a.trail_dirty);
                if any_dirty {
                    let agent_data: Vec<(&[[f32; 3]], &[f32; 3], bool)> = self
                        .agents
                        .iter()
                        .map(|a| (a.trail.as_slice(), &a.color as &[f32; 3], a.body.body.alive))
                        .collect();
                    let mesh = overlay::build_all_trails_mesh(&agent_data);
                    if mesh.indices.is_empty() {
                        trail_gpu.num_indices = 0;
                    } else {
                        trail_gpu.update_from_mesh(&renderer.queue, &mesh);
                    }
                    for a in &mut self.agents {
                        a.trail_dirty = false;
                    }
                    did_rebuild = true;
                }
            }
        }

        if did_rebuild && !self.paused {
            self.last_mesh_rebuild = Instant::now();
        }

        // ── rebuild selection marker above focused agent ──────
        if let (Some(renderer), Some(marker_gpu)) = (&self.renderer, &mut self.marker_gpu) {
            if let Some(agent) = self.agents.get(self.selected_agent_idx) {
                if agent.body.body.alive {
                    let mesh = overlay::build_marker_mesh(agent.body.body.position);
                    marker_gpu.update_from_mesh(&renderer.queue, &mesh);
                } else {
                    marker_gpu.num_indices = 0;
                }
            } else {
                marker_gpu.num_indices = 0;
            }
        }
    }

    /// Upload the agent instance buffer (positions + palette colors) when the
    /// HUD is dirty. Dead agents render in a muted gray.
    pub(crate) fn update_agent_instances(&mut self) {
        if !self.hud_dirty {
            return;
        }
        let instances: Vec<InstanceData> = self
            .agents
            .iter()
            .map(|a| {
                // Convert the agent's palette color from sRGB to linear so the
                // sRGB framebuffer produces the final color matching the sidebar.
                let color = if !a.body.body.alive {
                    [
                        srgb_to_linear(0.25),
                        srgb_to_linear(0.25),
                        srgb_to_linear(0.25),
                    ]
                } else {
                    [
                        srgb_to_linear(a.color[0]),
                        srgb_to_linear(a.color[1]),
                        srgb_to_linear(a.color[2]),
                    ]
                };
                InstanceData {
                    position: a.body.body.position.into(),
                    color,
                    scale: 2.0,
                    _pad: 0.0,
                }
            })
            .collect();
        self.agent_instance_count = instances.len() as u32;

        if let (Some(renderer), Some(buf)) = (&self.renderer, &self.agent_instance_buffer) {
            if !instances.is_empty() {
                renderer
                    .queue
                    .write_buffer(buf, 0, bytemuck::cast_slice(&instances));
            }
        }
    }

    /// Rebuild the HUD bars and overlay text (FPS, agent count, selected-agent
    /// vitals) when the HUD is dirty, then clear the dirty flag.
    pub(crate) fn rebuild_hud_text(&mut self) {
        if !self.hud_dirty {
            return;
        }
        self.cached_hud_bars = self.build_hud_bars();

        let mut text_items: Vec<TextItem> = Vec::new();

        // FPS counter at top-right
        text_items.push(TextItem {
            text: format!("FPS: {:.0}", self.fps),
            x: 0.62,
            y: 0.97,
            scale: 0.04,
            color: [0.0, 0.0, 0.0, 1.0],
        });

        // Agent count below FPS
        let alive = self.agents.iter().filter(|a| a.body.body.alive).count();
        text_items.push(TextItem {
            text: format!("Agents: {}/{}", alive, self.agents.len()),
            x: 0.62,
            y: 0.93,
            scale: 0.03,
            color: [0.0, 0.0, 0.0, 0.8],
        });

        // HUD bar labels
        let label_x = -0.98 + 0.36;
        let label_scale = 0.03;
        let label_color = [0.0, 0.0, 0.0, 0.9];
        let label_top = 0.97;
        let label_gap = 0.035;

        if let Some(agent) = self.agents.get(self.selected_agent_idx) {
            let energy = agent.body.body.internal.energy_signal();
            let integrity = agent.body.body.internal.integrity_signal();
            let pred_err = agent.cached_prediction_error;
            let explore = agent.cached_exploration_rate;

            text_items.push(TextItem {
                text: format!("Energy: {:.0}%", energy * 100.0),
                x: label_x,
                y: label_top,
                scale: label_scale,
                color: label_color,
            });
            text_items.push(TextItem {
                text: format!("Integrity: {:.0}%", integrity * 100.0),
                x: label_x,
                y: label_top - label_gap,
                scale: label_scale,
                color: label_color,
            });
            text_items.push(TextItem {
                text: format!("PredErr: {:.2}", pred_err),
                x: label_x,
                y: label_top - label_gap * 2.0,
                scale: label_scale,
                color: label_color,
            });
            text_items.push(TextItem {
                text: format!("Explore: {:.0}%", explore * 100.0),
                x: label_x,
                y: label_top - label_gap * 3.0,
                scale: label_scale,
                color: label_color,
            });

            // Info block below bars
            let info_y = label_top - label_gap * 4.5;
            let info_scale = 0.028;
            let info_color = [0.1, 0.1, 0.2, 0.9];
            text_items.push(TextItem {
                text: format!(
                    "Agent {} | Gen {} | Deaths: {}",
                    agent.id, agent.generation, agent.death_count
                ),
                x: -0.98,
                y: info_y,
                scale: info_scale,
                color: info_color,
            });
            text_items.push(TextItem {
                text: "Phase: GPU (TBD)".to_string(),
                x: -0.98,
                y: info_y - 0.035,
                scale: info_scale,
                color: info_color,
            });
        }

        if let Some(renderer) = &mut self.renderer {
            renderer.update_hud(&self.cached_hud_bars, &[]);
            renderer.update_text(&text_items);
        }

        self.hud_dirty = false;
    }

    /// Render one frame: assemble the UI snapshots, draw the 3D scene to the
    /// offscreen viewport, and render the egui chrome (top bar, console,
    /// sidebar) plus the tabbed dock. Returns any evolution action requested by
    /// the UI this frame so the caller can apply it outside the renderer borrow.
    pub(crate) fn render_frame(&mut self, event_loop: &ActiveEventLoop) -> EvolutionAction {
        // Assemble per-frame UI snapshots before borrowing the renderer.
        self.rebuild_agent_snapshots();
        self.update_evo_snapshot();
        self.update_world_snapshot();
        // Pre-build biome image pixels if the texture is not yet created.
        let mut biome_image = self.build_biome_image();

        let inst_buf = self.agent_instance_buffer.as_ref();
        let inst_count = self.agent_instance_count;

        let mut pending_evo_action = EvolutionAction::None;

        if let Some(renderer) = &mut self.renderer {
            let t = self.terrain_gpu.as_ref();
            let f = self.food_gpu.as_ref();
            let h = self.heatmap_gpu.as_ref().filter(|g| g.num_indices > 0);
            let tr = self.trail_gpu.as_ref().filter(|g| g.num_indices > 0);
            let mk = self.marker_gpu.as_ref().filter(|g| g.num_indices > 0);
            let mut mesh_vec: Vec<&GpuMesh> = Vec::with_capacity(5);
            if let Some(t) = t {
                mesh_vec.push(t);
            }
            if let Some(f) = f {
                mesh_vec.push(f);
            }
            if let Some(h) = h {
                mesh_vec.push(h);
            }
            if let Some(tr) = tr {
                mesh_vec.push(tr);
            }
            if let Some(mk) = mk {
                mesh_vec.push(mk);
            }

            let vp = self.camera.view_projection_matrix();

            // ── Offscreen 3D → egui surface pipeline ──────────
            match renderer.begin_frame() {
                Ok(mut frame_ctx) => {
                    // 1) Render 3D scene to offscreen viewport texture
                    if self.render_3d {
                        if let Some(egui) = &self.egui {
                            renderer.render_3d_offscreen(
                                &mesh_vec,
                                &vp,
                                inst_buf,
                                inst_count,
                                &mut frame_ctx.encoder,
                                &egui.viewport_color_view,
                                &egui.viewport_depth_view,
                            );
                        }
                    }

                    // 2) Render egui UI to the surface (viewport texture embedded)
                    if let (Some(egui), Some(window)) = (&mut self.egui, &self.window) {
                        let screen = egui_wgpu::ScreenDescriptor {
                            size_in_pixels: [renderer.config.width, renderer.config.height],
                            pixels_per_point: window.scale_factor() as f32,
                        };

                        let fps = self.fps;
                        let wall_time_secs = self.evo_snapshot.wall_time_secs;
                        let ticks_per_sec = self.evo_snapshot.ticks_per_sec;
                        let render_3d = self.render_3d;
                        let speed_multiplier = self.speed_multiplier;
                        let evo_state = self.evo_snapshot.state.clone();
                        let viewport_tex_id = egui.viewport_texture_id;
                        let ppp = window.scale_factor() as f32;
                        let mut desired_vp = (0u32, 0u32);
                        let selected_idx = self.selected_agent_idx;

                        let agent_snaps = self.cached_agent_snaps.as_slice();

                        // Move snapshot out so we can pass &mut to the closure
                        let mut evo_snap = std::mem::take(&mut self.evo_snapshot);
                        let gen_tick = evo_snap.gen_tick;
                        let tick_budget = evo_snap.tick_budget;
                        let evo_generation = evo_snap.generation;
                        let best_fitness = evo_snap.best_fitness;
                        let mut evo_action = EvolutionAction::None;

                        let mut world_snap = std::mem::take(&mut self.world_snapshot);

                        let console_lines: Vec<&str> =
                            self.console_log.iter().map(|s| s.as_str()).collect();

                        let mut clicked_agent_idx: Option<usize> = None;
                        let mut open_agent_tab: Option<u32> = None;
                        let mut vp_hovered = false;
                        let mut chart_win = self.chart_window;
                        let mut sort_mode = self.sort_mode;
                        let dock_state = &mut self.dock_state;
                        let replay_state = &mut self.replay_state;
                        let last_recording = self.last_recording.as_ref();
                        let orbit_mode = &mut self.orbit_mode;

                        let top_bar = TopBarState {
                            fps,
                            agent_count: agent_snaps.len(),
                            evo_state: &evo_state,
                            wall_time_secs,
                            speed_multiplier,
                            ticks_per_sec,
                            best_fitness,
                            gen_tick,
                            tick_budget,
                            generation: evo_generation,
                            render_3d,
                        };

                        egui.render(
                            window,
                            &renderer.device,
                            &renderer.queue,
                            &mut frame_ctx.encoder,
                            &frame_ctx.view,
                            screen,
                            |ctx| {
                                // Build biome texture once (requires ctx)
                                if let Some(image) = biome_image.take() {
                                    world_snap.biome_texture = Some(ctx.load_texture(
                                        "biome_map",
                                        image,
                                        egui::TextureOptions::NEAREST,
                                    ));
                                }

                                draw_top_bar(ctx, &top_bar, &mut evo_action);
                                draw_console(ctx, &console_lines);
                                let sidebar = draw_agent_sidebar(
                                    ctx,
                                    agent_snaps,
                                    &mut sort_mode,
                                    selected_idx,
                                );
                                clicked_agent_idx = sidebar.clicked_agent_idx;
                                open_agent_tab = sidebar.open_agent_tab;

                                // ── Central dock area (tabs) ─────────
                                egui::CentralPanel::default().frame(egui::Frame::NONE).show(
                                    ctx,
                                    |ui| {
                                        let mut tab_ctx = TabContext {
                                            viewport_tex_id,
                                            ppp,
                                            desired_vp: &mut desired_vp,
                                            viewport_hovered: &mut vp_hovered,
                                            chart_window: &mut chart_win,
                                            agents: agent_snaps,
                                            evolution: &mut evo_snap,
                                            evolution_action: &mut evo_action,
                                            world: &world_snap,
                                            replay: replay_state,
                                            recording: last_recording,
                                            orbit_mode,
                                        };
                                        egui_dock::DockArea::new(dock_state)
                                            .style(egui_dock::Style::from_egui(ui.style().as_ref()))
                                            .show_inside(ui, &mut tab_ctx);
                                    },
                                );
                            },
                        );

                        self.viewport_hovered = vp_hovered;
                        self.chart_window = chart_win;
                        self.sort_mode = sort_mode;

                        // Restore evolution snapshot (may have been mutated by UI)
                        self.evo_snapshot = evo_snap;
                        // Restore world snapshot (biome texture may have been created)
                        self.world_snapshot = world_snap;

                        // Defer evolution action to after the renderer borrow
                        pending_evo_action = evo_action;

                        // Handle agent selection from sidebar click
                        if let Some(idx) = clicked_agent_idx {
                            self.selected_agent_idx = idx;
                        }

                        // Handle double-click → open agent detail tab
                        if let Some(agent_id) = open_agent_tab {
                            let tab = Tab::AgentDetail(agent_id);
                            // Check if tab already exists
                            let already_open =
                                self.dock_state.iter_all_tabs().any(|(_, t)| *t == tab);
                            if !already_open {
                                self.dock_state.push_to_focused_leaf(tab);
                            }
                        }

                        // Resize offscreen textures if the panel changed size (takes effect next frame)
                        if desired_vp.0 > 0 && desired_vp.1 > 0 {
                            egui.resize_viewport(&renderer.device, desired_vp.0, desired_vp.1);
                            // Update camera aspect to match viewport
                            self.camera.aspect = desired_vp.0 as f32 / desired_vp.1.max(1) as f32;
                        }
                    }

                    renderer.finish_frame(frame_ctx);
                }
                Err(wgpu::SurfaceError::Lost) => {
                    let w = renderer.config.width;
                    let h = renderer.config.height;
                    renderer.resize(w, h);
                }
                Err(wgpu::SurfaceError::OutOfMemory) => {
                    log::error!("Out of memory!");
                    self.print_session_summary();
                    event_loop.exit();
                }
                Err(e) => {
                    log::warn!("Surface error: {:?}", e);
                }
            }
        }

        pending_evo_action
    }
}
