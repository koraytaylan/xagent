//! Fixed egui chrome panels that frame the central dock area.
//!
//! These free functions render the static top toolbar, bottom console, and
//! left agent-list sidebar. They take plain snapshot data and never borrow
//! `App`, keeping the egui render closure in `render_pipeline` free of large
//! inline panel bodies. The central tabbed dock area is rendered separately
//! via `egui_dock` and `TabContext`.

use xagent_sandbox::ui::{AgentSnapshot, EvolutionAction, EvolutionState, SortMode};

/// Read-only state rendered by the top toolbar.
pub(crate) struct TopBarState<'a> {
    pub(crate) fps: f32,
    pub(crate) agent_count: usize,
    pub(crate) evo_state: &'a EvolutionState,
    pub(crate) wall_time_secs: f64,
    pub(crate) speed_multiplier: u32,
    pub(crate) ticks_per_sec: f64,
    pub(crate) best_fitness: f32,
    pub(crate) gen_tick: u64,
    pub(crate) tick_budget: u64,
    pub(crate) generation: u32,
    pub(crate) render_3d: bool,
}

/// Agent interactions captured from the sidebar during one frame.
pub(crate) struct SidebarInteraction {
    /// Single-clicked agent (selects it).
    pub(crate) clicked_agent_idx: Option<usize>,
    /// Double-clicked agent id (opens its detail tab).
    pub(crate) open_agent_tab: Option<u32>,
}

/// Render the top toolbar. Records any evolution control button press into
/// `evo_action`.
pub(crate) fn draw_top_bar(
    ctx: &egui::Context,
    state: &TopBarState,
    evo_action: &mut EvolutionAction,
) {
    egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("xagent")
                    .strong()
                    .color(egui::Color32::from_rgb(120, 200, 255)),
            );
            ui.separator();
            ui.label(format!("FPS: {:.0}", state.fps));
            ui.separator();
            ui.label(format!("Agents: {}", state.agent_count));
            ui.separator();
            match state.evo_state {
                EvolutionState::Idle => {
                    ui.label(egui::RichText::new("⏹ IDLE").color(egui::Color32::GRAY));
                }
                EvolutionState::HasSession { generation } => {
                    ui.label(
                        egui::RichText::new(format!("💾 SESSION @ Gen {}", generation))
                            .color(egui::Color32::from_rgb(100, 180, 255)),
                    );
                }
                EvolutionState::Running => {
                    ui.label(
                        egui::RichText::new("▶ RUNNING")
                            .color(egui::Color32::from_rgb(50, 200, 80)),
                    );
                }
                EvolutionState::Paused => {
                    ui.label(egui::RichText::new("⏸ PAUSED").color(egui::Color32::YELLOW));
                }
            }
            if matches!(
                state.evo_state,
                EvolutionState::Running | EvolutionState::Paused
            ) {
                ui.separator();
                let hours = (state.wall_time_secs / 3600.0) as u64;
                let mins = ((state.wall_time_secs % 3600.0) / 60.0) as u64;
                let secs = (state.wall_time_secs % 60.0) as u64;
                ui.label(format!("{}h {:02}m {:02}s", hours, mins, secs));
                ui.separator();
                let speed_label = crate::speed_label(state.speed_multiplier);
                ui.label(egui::RichText::new(format!("⏩ {}", speed_label)).color(
                    if state.speed_multiplier > 1 {
                        egui::Color32::from_rgb(255, 200, 50)
                    } else {
                        egui::Color32::GRAY
                    },
                ));
                ui.separator();
                ui.label(format!("{:.0} ticks/s", state.ticks_per_sec));
            }
            if matches!(
                state.evo_state,
                EvolutionState::Running | EvolutionState::Paused
            ) && state.best_fitness >= 0.0
            {
                ui.separator();
                ui.label(
                    egui::RichText::new(format!("Best: {:.4}", state.best_fitness))
                        .color(egui::Color32::from_rgb(50, 200, 80)),
                );
            }
            // Generation progress bar (compact, in toolbar)
            if matches!(
                state.evo_state,
                EvolutionState::Running | EvolutionState::Paused
            ) && state.tick_budget > 0
            {
                ui.separator();
                let progress = state.gen_tick as f32 / state.tick_budget as f32;
                ui.add(
                    egui::ProgressBar::new(progress)
                        .text(format!(
                            "Gen {} — {:.0}%",
                            state.generation,
                            progress * 100.0
                        ))
                        .desired_width(160.0)
                        .animate(matches!(state.evo_state, EvolutionState::Running)),
                );
            }
            if !state.render_3d {
                ui.separator();
                ui.label(
                    egui::RichText::new("⚡ FAST (G)").color(egui::Color32::from_rgb(255, 160, 50)),
                );
            }
            // ── Right-aligned controls ──
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if matches!(
                    state.evo_state,
                    EvolutionState::Running | EvolutionState::Paused
                ) {
                    if ui
                        .add(egui::Button::new(
                            egui::RichText::new("🗑 Reset")
                                .color(egui::Color32::from_rgb(220, 80, 80)),
                        ))
                        .clicked()
                    {
                        *evo_action = EvolutionAction::Reset;
                    }
                    if matches!(state.evo_state, EvolutionState::Running) {
                        if ui.button("⏸ Pause").clicked() {
                            *evo_action = EvolutionAction::Pause;
                        }
                    } else if ui.button("▶ Resume").clicked() {
                        *evo_action = EvolutionAction::Unpause;
                    }
                }
            });
        });
    });
}

/// Render the bottom console panel with severity-colored log lines.
pub(crate) fn draw_console(ctx: &egui::Context, console_lines: &[&str]) {
    egui::TopBottomPanel::bottom("console")
        .resizable(true)
        .default_height(120.0)
        .show(ctx, |ui| {
            ui.label(
                egui::RichText::new("Console")
                    .small()
                    .color(egui::Color32::GRAY),
            );
            ui.separator();
            let scroll_width = ui.available_width();
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .min_scrolled_width(scroll_width)
                .show(ui, |ui| {
                    ui.set_min_width(scroll_width);
                    for line in console_lines {
                        let color = if line.contains("ERROR") || line.contains("Failed to") {
                            egui::Color32::from_rgb(255, 100, 100)
                        } else if line.contains("best") || line.contains("Beat parent") {
                            egui::Color32::from_rgb(80, 220, 80)
                        } else if line.contains("Failed")
                            || line.contains("exhausted")
                            || line.contains("backtracking")
                        {
                            egui::Color32::from_rgb(255, 140, 80)
                        } else if line.contains("Migration") {
                            egui::Color32::from_rgb(100, 180, 255)
                        } else if line.contains("Momentum") {
                            egui::Color32::from_rgb(180, 160, 255)
                        } else {
                            egui::Color32::LIGHT_GRAY
                        };
                        ui.label(
                            egui::RichText::new(*line)
                                .monospace()
                                .size(11.0)
                                .color(color),
                        );
                    }
                });
        });
}

/// Render the left agent-list sidebar with the sort selector and per-agent
/// rows. Returns any selection/open interaction captured this frame.
pub(crate) fn draw_agent_sidebar(
    ctx: &egui::Context,
    agents: &[AgentSnapshot],
    sort_mode: &mut SortMode,
    selected_idx: usize,
) -> SidebarInteraction {
    let mut clicked_agent_idx: Option<usize> = None;
    let mut open_agent_tab: Option<u32> = None;
    egui::SidePanel::left("agent_list")
        .resizable(true)
        .default_width(200.0)
        .show(ctx, |ui| {
            ui.label(egui::RichText::new("Agents").strong().size(14.0));
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Sort:")
                        .small()
                        .color(egui::Color32::GRAY),
                );
                egui::ComboBox::from_id_salt("agent_sort")
                    .selected_text(sort_mode.label())
                    .width(90.0)
                    .show_ui(ui, |ui| {
                        for mode in SortMode::ALL {
                            ui.selectable_value(sort_mode, mode, mode.label());
                        }
                    });
            });
            ui.add_space(2.0);
            egui::ScrollArea::vertical().show(ui, |ui| {
                let mut sorted_indices: Vec<usize> = (0..agents.len()).collect();
                match *sort_mode {
                    SortMode::Id => {} // already sorted by id
                    SortMode::Energy => {
                        sorted_indices.sort_by(|&a, &b| {
                            let ea = agents[a].energy / agents[a].max_energy.max(0.001);
                            let eb = agents[b].energy / agents[b].max_energy.max(0.001);
                            eb.partial_cmp(&ea).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    SortMode::Integrity => {
                        sorted_indices.sort_by(|&a, &b| {
                            let ia = agents[a].integrity / agents[a].max_integrity.max(0.001);
                            let ib = agents[b].integrity / agents[b].max_integrity.max(0.001);
                            ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    SortMode::Deaths => {
                        sorted_indices.sort_by(|&a, &b| agents[a].deaths.cmp(&agents[b].deaths));
                    }
                    SortMode::LongestLife => {
                        sorted_indices
                            .sort_by(|&a, &b| agents[b].longest_life.cmp(&agents[a].longest_life));
                    }
                    SortMode::PredictionError => {
                        sorted_indices.sort_by(|&a, &b| {
                            agents[a]
                                .prediction_error
                                .partial_cmp(&agents[b].prediction_error)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    SortMode::Fitness => {
                        sorted_indices.sort_by(|&a, &b| {
                            let fa = agents[a].food_consumed as f64
                                + agents[a].total_ticks_alive as f64 * 0.001;
                            let fb = agents[b].food_consumed as f64
                                + agents[b].total_ticks_alive as f64 * 0.001;
                            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                }
                for &idx in &sorted_indices {
                    let snap = &agents[idx];
                    let is_selected = idx == selected_idx;
                    let color = egui::Color32::from_rgb(
                        (snap.color[0] * 255.0) as u8,
                        (snap.color[1] * 255.0) as u8,
                        (snap.color[2] * 255.0) as u8,
                    );
                    let frame = if is_selected {
                        egui::Frame::NONE
                            .fill(egui::Color32::from_rgba_premultiplied(60, 60, 80, 255))
                            .inner_margin(4.0)
                            .corner_radius(3.0)
                    } else {
                        egui::Frame::NONE.inner_margin(4.0)
                    };
                    let response = frame.show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let (rect, _) = ui
                                .allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                            ui.painter().circle_filled(rect.center(), 5.0, color);
                            let status = if !snap.alive { "💀" } else { "" };
                            ui.label(format!(
                                "Agent {} (g{}) {}",
                                snap.id, snap.generation, status
                            ));
                        });
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(snap.phase).small().color(
                                match snap.phase {
                                    "ADAPTED" => egui::Color32::from_rgb(80, 200, 80),
                                    "LEARNING" => egui::Color32::from_rgb(200, 200, 80),
                                    "EXPLORING" => egui::Color32::from_rgb(200, 140, 60),
                                    _ => egui::Color32::from_rgb(150, 150, 150),
                                },
                            ));
                            ui.label(
                                egui::RichText::new(format!(
                                    "| D:{} F:{}",
                                    snap.deaths, snap.food_consumed
                                ))
                                .small()
                                .color(egui::Color32::GRAY),
                            );
                        });
                    });
                    let resp = response.response.interact(egui::Sense::click());
                    if resp.clicked() {
                        clicked_agent_idx = Some(idx);
                    }
                    if resp.double_clicked() {
                        open_agent_tab = Some(snap.id);
                    }
                    ui.add_space(2.0);
                }
            });
        });
    SidebarInteraction {
        clicked_agent_idx,
        open_agent_tab,
    }
}
