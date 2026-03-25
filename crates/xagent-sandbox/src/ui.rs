//! egui integration layer — immediate-mode UI rendered on top of the 3D scene.
//!
//! Manages the offscreen render target for the 3D viewport and the egui overlay.

use egui_wgpu::ScreenDescriptor;

/// Tab types for the dock area.
#[derive(Clone, Debug, PartialEq)]
pub enum Tab {
    /// The 3D sandbox viewport (always open, cannot be closed).
    Sandbox,
    /// Agent detail view, keyed by agent ID.
    AgentDetail(u32),
}

/// Snapshot of a single agent's state, copied once per frame for the UI.
pub struct AgentSnapshot {
    pub id: u32,
    pub gen: u32,
    pub energy: f32,
    pub max_energy: f32,
    pub integrity: f32,
    pub max_integrity: f32,
    pub alive: bool,
    pub deaths: u32,
    pub color: [f32; 3],
    pub longest_life: u64,
    pub exploration_rate: f32,
    pub prediction_error: f32,
    pub action_weights: Vec<f32>,
    pub prediction_error_history: Vec<f32>,
    pub exploration_rate_history: Vec<f32>,
    pub energy_history: Vec<f32>,
    pub integrity_history: Vec<f32>,
    pub action_weight_history: Vec<[f32; 8]>,
}

/// Context passed to the TabViewer so it can render both viewport and agent detail tabs.
pub struct TabContext<'a> {
    pub viewport_tex_id: egui::TextureId,
    pub ppp: f32,
    pub desired_vp: &'a mut (u32, u32),
    pub viewport_hovered: &'a mut bool,
    /// Number of history samples visible in the detail chart (zoom level).
    pub chart_window: &'a mut usize,
    pub agents: &'a [AgentSnapshot],
}

/// Holds egui state needed across frames: context, winit integration, wgpu renderer,
/// and the offscreen texture used to embed the 3D viewport inside an egui panel.
pub struct EguiIntegration {
    pub ctx: egui::Context,
    winit_state: egui_winit::State,
    wgpu_renderer: egui_wgpu::Renderer,

    // Offscreen 3D viewport
    viewport_color_format: wgpu::TextureFormat,
    pub viewport_color: wgpu::Texture,
    pub viewport_color_view: wgpu::TextureView,
    pub viewport_depth_view: wgpu::TextureView,
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub viewport_texture_id: egui::TextureId,
}

impl EguiIntegration {
    /// Create a new egui integration sharing the existing wgpu device.
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        window: &winit::window::Window,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Self {
        let ctx = egui::Context::default();

        let winit_state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            None, // native_pixels_per_point — let egui auto-detect
            None, // theme
            None, // max_texture_side
        );

        let mut wgpu_renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            None, // no depth
            1,    // msaa samples
            false, // dithering
        );

        let (color_tex, color_view, depth_view) =
            Self::create_offscreen_textures(device, viewport_width, viewport_height, surface_format);

        let texture_id = wgpu_renderer.register_native_texture(
            device,
            &color_view,
            wgpu::FilterMode::Linear,
        );

        Self {
            ctx,
            winit_state,
            wgpu_renderer,
            viewport_color_format: surface_format,
            viewport_color: color_tex,
            viewport_color_view: color_view,
            viewport_depth_view: depth_view,
            viewport_width,
            viewport_height,
            viewport_texture_id: texture_id,
        }
    }

    fn create_offscreen_textures(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        color_format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
        let w = width.max(1);
        let h = height.max(1);

        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_color"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        (color_tex, color_view, depth_view)
    }

    /// Resize the offscreen viewport textures. Call when the egui panel changes size.
    pub fn resize_viewport(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if w == self.viewport_width && h == self.viewport_height {
            return;
        }
        self.viewport_width = w;
        self.viewport_height = h;

        let (color_tex, color_view, depth_view) =
            Self::create_offscreen_textures(device, w, h, self.viewport_color_format);
        self.viewport_color = color_tex;
        self.viewport_color_view = color_view;
        self.viewport_depth_view = depth_view;

        // Re-register the texture with egui using the same TextureId
        self.wgpu_renderer.update_egui_texture_from_wgpu_texture(
            device,
            &self.viewport_color_view,
            wgpu::FilterMode::Linear,
            self.viewport_texture_id,
        );
    }

    /// Forward a winit event to egui. Returns `true` if egui consumed it
    /// (meaning the app should NOT process it as a camera/sim control).
    pub fn on_window_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        let response = self.winit_state.on_window_event(window, event);
        response.consumed
    }

    /// Run the egui frame, tessellate, and record a render pass into `encoder`.
    ///
    /// `ui_fn` receives the egui context and should call `egui::SidePanel`, `Window`, etc.
    pub fn render(
        &mut self,
        window: &winit::window::Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        screen: ScreenDescriptor,
        ui_fn: impl FnMut(&egui::Context),
    ) {
        let raw_input = self.winit_state.take_egui_input(window);
        let full_output = self.ctx.run(raw_input, ui_fn);

        self.winit_state
            .handle_platform_output(window, full_output.platform_output);

        let tris = self
            .ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        // Upload changed textures (font atlas on first frame, etc.)
        for (id, image_delta) in &full_output.textures_delta.set {
            self.wgpu_renderer
                .update_texture(device, queue, *id, image_delta);
        }

        self.wgpu_renderer
            .update_buffers(device, queue, encoder, &tris, &screen);

        // egui render pass — Clear to dark background, then draw UI (which embeds viewport texture)
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.12, g: 0.12, b: 0.14, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // wgpu 24 returns RenderPass<'encoder>; egui-wgpu 0.31 expects
            // RenderPass<'static>. forget_lifetime() drops the borrow.
            let mut pass = pass.forget_lifetime();
            self.wgpu_renderer.render(&mut pass, &tris, &screen);
        }

        // Free textures no longer needed
        for id in &full_output.textures_delta.free {
            self.wgpu_renderer.free_texture(id);
        }
    }
}

// ── TabViewer for egui_dock ──────────────────────────────────────────────

impl<'a> egui_dock::TabViewer for TabContext<'a> {
    type Tab = Tab;

    fn id(&mut self, tab: &mut Self::Tab) -> egui::Id {
        match tab {
            Tab::Sandbox => egui::Id::new("tab_sandbox"),
            Tab::AgentDetail(id) => egui::Id::new("tab_agent_detail").with(*id),
        }
    }

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            Tab::Sandbox => "🌍 Sandbox".into(),
            Tab::AgentDetail(id) => {
                let name = self.agents.iter()
                    .find(|a| a.id == *id)
                    .map(|a| format!("Agent {} (g{})", a.id, a.gen))
                    .unwrap_or_else(|| format!("Agent {} (?)", id));
                format!("🧠 {}", name).into()
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            Tab::Sandbox => {
                let avail = ui.available_size();
                *self.desired_vp = (
                    (avail.x * self.ppp) as u32,
                    (avail.y * self.ppp) as u32,
                );
                let resp = ui.add(
                    egui::Image::new(egui::load::SizedTexture::new(
                        self.viewport_tex_id,
                        avail,
                    ))
                    .sense(egui::Sense::click_and_drag()),
                );
                *self.viewport_hovered = resp.hovered() || resp.dragged();
            }
            Tab::AgentDetail(id) => {
                if let Some(snap) = self.agents.iter().find(|a| a.id == *id) {
                    egui::ScrollArea::vertical()
                        .id_salt(format!("agent_detail_scroll_{}", id))
                        .show(ui, |ui| {
                            Self::render_agent_detail(ui, snap, self.chart_window);
                        });
                } else {
                    ui.label(format!("Agent {} no longer exists.", id));
                }
            }
        }
    }

    fn closeable(&mut self, tab: &mut Self::Tab) -> bool {
        // Sandbox tab cannot be closed
        !matches!(tab, Tab::Sandbox)
    }
}

impl<'a> TabContext<'a> {
    fn render_agent_detail(ui: &mut egui::Ui, snap: &AgentSnapshot, chart_window: &mut usize) {
        let color = egui::Color32::from_rgb(
            (snap.color[0] * 255.0) as u8,
            (snap.color[1] * 255.0) as u8,
            (snap.color[2] * 255.0) as u8,
        );

        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
            ui.painter().circle_filled(rect.center(), 7.0, color);
            ui.heading(format!(
                "Agent {} (Gen {})",
                snap.id, snap.gen
            ));
            if !snap.alive {
                ui.label(egui::RichText::new("💀 DEAD").color(egui::Color32::RED));
            }
        });
        ui.separator();

        ui.columns(2, |cols| {
            // Left column: vitals + statistics
            cols[0].label(egui::RichText::new("Vitals").strong());
            cols[0].add_space(4.0);

            let energy_frac = snap.energy / snap.max_energy.max(0.001);
            let integrity_frac = snap.integrity / snap.max_integrity.max(0.001);

            egui::Grid::new("vitals_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(&mut cols[0], |ui| {
                    ui.label("Energy:");
                    ui.add(egui::ProgressBar::new(energy_frac)
                        .text(format!("{:.1}/{:.0}", snap.energy, snap.max_energy))
                        .fill(if energy_frac > 0.5 {
                            egui::Color32::from_rgb(80, 200, 80)
                        } else if energy_frac > 0.25 {
                            egui::Color32::YELLOW
                        } else {
                            egui::Color32::from_rgb(220, 60, 60)
                        }));
                    ui.end_row();

                    ui.label("Integrity:");
                    ui.add(egui::ProgressBar::new(integrity_frac)
                        .text(format!("{:.1}/{:.0}", snap.integrity, snap.max_integrity))
                        .fill(egui::Color32::from_rgb(100, 150, 255)));
                    ui.end_row();
                });

            cols[0].add_space(8.0);
            cols[0].label(egui::RichText::new("Statistics").strong());
            cols[0].add_space(4.0);
            cols[0].label(format!("Deaths: {}", snap.deaths));
            cols[0].label(format!("Longest life: {} ticks", snap.longest_life));

            // Right column: brain info + action weights
            cols[1].label(egui::RichText::new("Brain").strong());
            cols[1].add_space(4.0);
            cols[1].label(format!("Exploration: {:.1}%", snap.exploration_rate * 100.0));
            cols[1].label(format!("Prediction error: {:.4}", snap.prediction_error));

            cols[1].add_space(8.0);
            cols[1].label(egui::RichText::new("Action Weights").strong());
            cols[1].add_space(4.0);

            let action_names = [
                "Forward", "Back", "Left", "Right",
                "Turn L", "Turn R", "Consume", "Forage",
            ];
            let max_w = snap.action_weights.iter().copied()
                .fold(0.0f32, f32::max).max(0.001);

            egui::Grid::new("action_weights_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(&mut cols[1], |ui| {
                    for (i, w) in snap.action_weights.iter().enumerate() {
                        let name = action_names.get(i).unwrap_or(&"?");
                        ui.label(egui::RichText::new(*name).monospace().small());
                        ui.add(egui::ProgressBar::new(w / max_w)
                            .text(format!("{:.3}", w))
                            .desired_width(120.0));
                        ui.end_row();
                    }
                });
        });

        // ── Combined history chart ──────────────────────────────────
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("History").strong());
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(format!("Window: {} ticks  (scroll to zoom)", *chart_window))
                    .small()
                    .color(egui::Color32::GRAY),
            );
        });
        ui.add_space(4.0);

        // Legend
        ui.horizontal(|ui| {
            for (label, color) in [
                ("Energy", egui::Color32::from_rgb(80, 200, 80)),
                ("Integrity", egui::Color32::from_rgb(100, 150, 255)),
                ("Pred. Error", egui::Color32::from_rgb(200, 140, 60)),
                ("Exploration", egui::Color32::from_rgb(180, 100, 220)),
            ] {
                let (dot, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                ui.painter().circle_filled(dot.center(), 4.0, color);
                ui.label(egui::RichText::new(label).small().color(egui::Color32::GRAY));
                ui.add_space(6.0);
            }
        });
        ui.add_space(2.0);

        let chart_height = 120.0;
        let avail_w = ui.available_width().max(60.0);
        let (rect, chart_resp) = ui.allocate_exact_size(
            egui::vec2(avail_w, chart_height),
            egui::Sense::hover(),
        );

        // Scroll-to-zoom on chart area
        if chart_resp.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                let factor = if scroll > 0.0 { 0.8 } else { 1.25 };
                let new_w = ((*chart_window as f32) * factor).round() as usize;
                *chart_window = new_w.clamp(30, 10_000);
            }
        }

        let painter = ui.painter();

        // Background + faint grid lines
        painter.rect_filled(rect, 2.0, egui::Color32::from_gray(25));
        for frac in [0.25, 0.5, 0.75] {
            let y = rect.bottom() - frac * rect.height();
            painter.line_segment(
                [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                egui::Stroke::new(0.5, egui::Color32::from_gray(50)),
            );
        }

        // Slice each series to the visible window (most recent N samples)
        let window = *chart_window;

        let series_data: [(&[f32], egui::Color32); 4] = [
            (&snap.energy_history, egui::Color32::from_rgb(80, 200, 80)),
            (&snap.integrity_history, egui::Color32::from_rgb(100, 150, 255)),
            (&snap.prediction_error_history, egui::Color32::from_rgb(200, 140, 60)),
            (&snap.exploration_rate_history, egui::Color32::from_rgb(180, 100, 220)),
        ];
        for &(full_data, color) in &series_data {
            let start = full_data.len().saturating_sub(window);
            let data = &full_data[start..];
            if data.len() < 2 {
                continue;
            }
            let n = data.len();
            let points: Vec<egui::Pos2> = data.iter().enumerate().map(|(i, &v)| {
                let x = rect.left() + (i as f32 / (n - 1) as f32) * rect.width();
                let y = rect.bottom() - v.clamp(0.0, 1.0) * rect.height();
                egui::pos2(x, y)
            }).collect();
            let stroke = egui::Stroke::new(1.5, color);
            for pair in points.windows(2) {
                painter.line_segment([pair[0], pair[1]], stroke);
            }
        }

        // ── Action Weights history chart ────────────────────────────
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Action Weight History").strong());
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("(same zoom as above)")
                    .small()
                    .color(egui::Color32::GRAY),
            );
        });
        ui.add_space(4.0);

        let action_names = [
            "Forward", "Back", "Left", "Right",
            "Turn L", "Turn R", "Consume", "Forage",
        ];
        let action_colors = [
            egui::Color32::from_rgb(255, 100, 100),  // Forward - red
            egui::Color32::from_rgb(100, 100, 255),  // Back - blue
            egui::Color32::from_rgb(255, 200, 50),   // Left - yellow
            egui::Color32::from_rgb(50, 220, 200),   // Right - cyan
            egui::Color32::from_rgb(200, 100, 255),  // Turn L - purple
            egui::Color32::from_rgb(255, 150, 50),   // Turn R - orange
            egui::Color32::from_rgb(100, 255, 100),  // Consume - green
            egui::Color32::from_rgb(255, 255, 255),  // Forage - white
        ];

        // Legend row
        ui.horizontal_wrapped(|ui| {
            for i in 0..8 {
                let (dot, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                ui.painter().circle_filled(dot.center(), 4.0, action_colors[i]);
                ui.label(egui::RichText::new(action_names[i]).small().color(egui::Color32::GRAY));
                ui.add_space(4.0);
            }
        });
        ui.add_space(2.0);

        let aw_chart_height = 120.0;
        let aw_avail_w = ui.available_width().max(60.0);
        let (aw_rect, aw_resp) = ui.allocate_exact_size(
            egui::vec2(aw_avail_w, aw_chart_height),
            egui::Sense::hover(),
        );

        // Scroll-to-zoom (shared with vitals chart)
        if aw_resp.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                let factor = if scroll > 0.0 { 0.8 } else { 1.25 };
                let new_w = ((*chart_window as f32) * factor).round() as usize;
                *chart_window = new_w.clamp(30, 10_000);
            }
        }

        let aw_painter = ui.painter();

        // Background + grid
        aw_painter.rect_filled(aw_rect, 2.0, egui::Color32::from_gray(25));
        for frac in [0.25, 0.5, 0.75] {
            let y = aw_rect.bottom() - frac * aw_rect.height();
            aw_painter.line_segment(
                [egui::pos2(aw_rect.left(), y), egui::pos2(aw_rect.right(), y)],
                egui::Stroke::new(0.5, egui::Color32::from_gray(50)),
            );
        }

        // Compute visible window and auto-scale Y range
        let aw_hist = &snap.action_weight_history;
        let aw_start = aw_hist.len().saturating_sub(window);
        let aw_slice = &aw_hist[aw_start..];

        if aw_slice.len() >= 2 {
            let mut vmin = f32::MAX;
            let mut vmax = f32::MIN;
            for snap_aw in aw_slice {
                for &v in snap_aw {
                    if v < vmin { vmin = v; }
                    if v > vmax { vmax = v; }
                }
            }
            let range = (vmax - vmin).max(0.001);
            let n = aw_slice.len();

            for action_idx in 0..8 {
                let points: Vec<egui::Pos2> = aw_slice.iter().enumerate().map(|(i, aw)| {
                    let x = aw_rect.left() + (i as f32 / (n - 1) as f32) * aw_rect.width();
                    let t = (aw[action_idx] - vmin) / range;
                    let y = aw_rect.bottom() - t * aw_rect.height();
                    egui::pos2(x, y)
                }).collect();
                let stroke = egui::Stroke::new(1.5, action_colors[action_idx]);
                for pair in points.windows(2) {
                    aw_painter.line_segment([pair[0], pair[1]], stroke);
                }
            }
        }
    }
}
