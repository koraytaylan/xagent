# Sandbox Tab & UI Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overhaul the sandbox viewport: add a toolbar with orbit-agent button, implement FPS-style camera, darken the out-of-map void, replace death/spawn console noise with evolution insights, and remove energy/integrity bars from the sidebar.

**Architecture:** Changes span camera.rs (FPS movement + orbit mode), main.rs (input routing, toolbar, console, sidebar), renderer/mod.rs (clear color), and agent/senses.rs (sky color for agent vision). No new files needed.

**Tech Stack:** Rust, egui, wgpu, glam

---

### Task 1: Darken out-of-map clear color

The 3D viewport's clear color is sky blue `(0.53, 0.81, 0.92)` — visible wherever terrain mesh doesn't cover. Change it to a dark tone matching the app's egui background `(0.12, 0.12, 0.14)`.

**Files:**
- Modify: `crates/xagent-sandbox/src/renderer/mod.rs:650-651` (offscreen pass)
- Modify: `crates/xagent-sandbox/src/renderer/mod.rs:779-783` (standalone pass)
- Modify: `crates/xagent-sandbox/src/agent/senses.rs:130` (agent vision sky color — keep blue so agents can still distinguish sky from terrain)

- [ ] **Step 1: Change offscreen 3D pass clear color**

In `renderer/mod.rs`, find the offscreen pass clear color at line 650-651:

```rust
load: wgpu::LoadOp::Clear(wgpu::Color {
    r: 0.53, g: 0.81, b: 0.92, a: 1.0,
}),
```

Change to:

```rust
load: wgpu::LoadOp::Clear(wgpu::Color {
    r: 0.12, g: 0.12, b: 0.14, a: 1.0,
}),
```

- [ ] **Step 2: Change standalone pass clear color**

In the same file, find the second clear color at lines 779-783:

```rust
load: wgpu::LoadOp::Clear(wgpu::Color {
    r: 0.53,
    g: 0.81,
    b: 0.92,
    a: 1.0,
}),
```

Change to:

```rust
load: wgpu::LoadOp::Clear(wgpu::Color {
    r: 0.12,
    g: 0.12,
    b: 0.14,
    a: 1.0,
}),
```

- [ ] **Step 3: Keep agent vision sky color as-is**

Do NOT change `agent/senses.rs:130`. Agents need to perceive sky differently from terrain — the blue sky is part of their sensory input, not a display choice.

- [ ] **Step 4: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`

- [ ] **Step 5: Commit**

```bash
git add crates/xagent-sandbox/src/renderer/mod.rs
git commit -m "fix: darken out-of-map void to match app theme"
```

---

### Task 2: Implement FPS-style camera movement

Currently the camera requires left-mouse-drag to look around. Change to FPS-style: right-click-and-hold to enable mouse look (no drag required — just moving the mouse while right button is held rotates the camera). WASD movement should work while right-clicking.

**Files:**
- Modify: `crates/xagent-sandbox/src/renderer/camera.rs` (mouse look behavior)
- Modify: `crates/xagent-sandbox/src/main.rs:1067-1081` (key input)
- Modify: `crates/xagent-sandbox/src/main.rs:1165-1183` (mouse input routing)

- [ ] **Step 1: Change mouse look to right-click activation**

In `main.rs`, find the mouse button handling (around line 1165-1176). Currently left-click toggles `is_mouse_dragging`. Change to right-click:

Find:

```rust
winit::event::MouseButton::Left => {
```

In the block that sets `self.camera.is_mouse_dragging`, change `Left` to `Right`.

- [ ] **Step 2: Verify it compiles and the behavior is correct**

Run: `cargo build 2>&1 | tail -5`

FPS-style means: hold right-click to look around, WASD to move, release right-click to interact with UI. The existing camera code already handles this — `is_mouse_dragging` gates `process_mouse_move`, and WASD keys are independent. The only change needed is switching from left to right mouse button.

- [ ] **Step 3: Commit**

```bash
git add crates/xagent-sandbox/src/main.rs
git commit -m "feat: FPS-style camera — right-click to look around"
```

---

### Task 3: Add orbit-agent camera mode with toolbar button

Add an orbit mode that makes the camera orbit the selected agent. A toggle button in a toolbar at the top of the sandbox tab controls it. When active, the camera keeps a fixed distance and orbits around the agent's position. Mouse drag rotates the orbit, scroll zooms in/out, but the camera always looks at the agent.

**Files:**
- Modify: `crates/xagent-sandbox/src/renderer/camera.rs` (add orbit mode)
- Modify: `crates/xagent-sandbox/src/ui.rs:484-498` (sandbox tab — add toolbar)
- Modify: `crates/xagent-sandbox/src/ui.rs` (TabContext — add orbit state)
- Modify: `crates/xagent-sandbox/src/main.rs` (pass orbit state, update camera per frame)

- [ ] **Step 1: Add orbit state to Camera**

In `camera.rs`, add fields to the Camera struct:

```rust
    pub orbit_mode: bool,
    pub orbit_target: Vec3,
    pub orbit_distance: f32,
    pub orbit_yaw: f32,
    pub orbit_pitch: f32,
```

Initialize in `Camera::new()`:

```rust
    orbit_mode: false,
    orbit_target: Vec3::ZERO,
    orbit_distance: 30.0,
    orbit_yaw: 0.0,
    orbit_pitch: -0.5,
```

- [ ] **Step 2: Add orbit update method**

In `camera.rs`, add a method that positions the camera based on orbit parameters:

```rust
    /// Update camera to orbit around the target position.
    /// Call this instead of `update()` when orbit mode is active.
    pub fn update_orbit(&mut self, target: Vec3) {
        self.orbit_target = target;

        // Compute camera position on a sphere around the target
        let x = self.orbit_distance * self.orbit_yaw.cos() * self.orbit_pitch.cos();
        let y = self.orbit_distance * self.orbit_pitch.sin();
        let z = self.orbit_distance * self.orbit_yaw.sin() * self.orbit_pitch.cos();
        self.position = self.orbit_target + Vec3::new(x, y, z);

        // Look at target — derive yaw/pitch from the look direction
        let dir = (self.orbit_target - self.position).normalize();
        self.yaw = dir.z.atan2(dir.x);
        self.pitch = dir.y.asin();
    }

    /// Process mouse movement for orbit rotation.
    pub fn process_orbit_mouse_move(&mut self, x: f64, y: f64) {
        if !self.is_mouse_dragging {
            self.last_mouse_pos = None;
            return;
        }
        if let Some((last_x, last_y)) = self.last_mouse_pos {
            let sensitivity = 0.005;
            let dx = (x - last_x) as f32;
            let dy = (y - last_y) as f32;
            self.orbit_yaw += dx * sensitivity;
            self.orbit_pitch = (self.orbit_pitch - dy * sensitivity)
                .clamp(-1.4, -0.05); // Keep above ground, not too steep
        }
        self.last_mouse_pos = Some((x, y));
    }

    /// Scroll-wheel zoom for orbit mode: changes orbit distance.
    pub fn process_orbit_scroll(&mut self, delta: f32) {
        self.orbit_distance = (self.orbit_distance - delta * 2.0).clamp(5.0, 200.0);
    }
```

- [ ] **Step 3: Add orbit toggle to sandbox tab toolbar**

In `ui.rs`, the Sandbox tab currently just shows the viewport image. Add a toolbar above it. Change the `Tab::Sandbox` arm (around line 484-498):

First, add `orbit_mode: &'a mut bool` to `TabContext` struct and pass it from `main.rs`.

Then update the Sandbox tab rendering:

```rust
Tab::Sandbox => {
    // Toolbar
    ui.horizontal(|ui| {
        let orbit_label = if *self.orbit_mode { "🎯 Orbiting" } else { "🎯 Orbit Agent" };
        if ui.selectable_label(*self.orbit_mode, orbit_label).clicked() {
            *self.orbit_mode = !*self.orbit_mode;
        }
    });
    ui.separator();

    // Viewport
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
```

- [ ] **Step 4: Add orbit_mode field to App and wire into TabContext**

In `main.rs`, add `orbit_mode: bool` to the App struct. Initialize to `false`. Pass `&mut self.orbit_mode` to `TabContext` as `orbit_mode`.

- [ ] **Step 5: Route camera input through orbit mode**

In `main.rs`, in the per-frame camera update (around line 1211), change:

```rust
// Before:
self.camera.update(dt);

// After:
if self.camera.orbit_mode {
    if let Some(agent) = self.agents.get(self.selected_agent_idx) {
        let target = agent.body.body.position;
        self.camera.update_orbit(target);
    }
} else {
    self.camera.update(dt);
}
```

In the mouse move handler, route to orbit or free-look:

```rust
if self.camera.orbit_mode {
    self.camera.process_orbit_mouse_move(x, y);
} else {
    self.camera.process_mouse_move(x, y);
}
```

In the scroll handler:

```rust
if self.camera.orbit_mode {
    self.camera.process_orbit_scroll(delta);
} else {
    self.camera.process_scroll(delta);
}
```

Sync orbit_mode between App and Camera:

```rust
self.camera.orbit_mode = self.orbit_mode;
```

Do this before the camera update each frame.

- [ ] **Step 6: When orbit mode is toggled on, initialize orbit from current camera**

When `orbit_mode` changes from false to true, set initial orbit parameters:

```rust
// Capture current distance and angles relative to the selected agent
if let Some(agent) = self.agents.get(self.selected_agent_idx) {
    let diff = self.camera.position - agent.body.body.position;
    self.camera.orbit_distance = diff.length().clamp(5.0, 200.0);
    self.camera.orbit_yaw = diff.z.atan2(diff.x);
    self.camera.orbit_pitch = (diff.y / diff.length()).asin().clamp(-1.4, -0.05);
}
```

- [ ] **Step 7: Verify it compiles**

Run: `cargo build 2>&1 | tail -10`

- [ ] **Step 8: Commit**

```bash
git add crates/xagent-sandbox/src/renderer/camera.rs crates/xagent-sandbox/src/ui.rs crates/xagent-sandbox/src/main.rs
git commit -m "feat: orbit-agent camera mode with toolbar toggle in sandbox tab"
```

---

### Task 4: Replace death/spawn console noise with evolution insights

Currently the console shows `[DEATH]`, `[SPAWN]`, `[RESPAWN]` messages for every agent. These fire constantly and are noise. Remove them and keep only evolution-relevant messages. Also improve the existing evolution messages to surface parameter trend insights.

**Files:**
- Modify: `crates/xagent-sandbox/src/main.rs:1485-1522` (death/respawn logging)
- Modify: `crates/xagent-sandbox/src/main.rs:2189-2204` (console color coding)
- Modify: `crates/xagent-sandbox/src/governor.rs` (enhance evolution log messages)

- [ ] **Step 1: Remove death/spawn/respawn console messages**

In `main.rs`, find the death logging (around line 1491). There should be a `log_msg` or direct `console_log.push_back` with `[DEATH]`. Comment out or remove the line that pushes the death message.

Similarly find and remove `[RESPAWN]` messages (around line 1511) and `[SPAWN]` messages (around line 463, 476).

Keep the `println!` for terminal output if you want, but remove the `console_log.push_back` calls for these events.

- [ ] **Step 2: Update console color coding**

In `main.rs` (around line 2189-2204), the console colors `[DEATH]` red and `[SPAWN]`/`[RESPAWN]` green. Since these messages are removed, update the color logic to highlight evolution events:

```rust
for line in &console_lines {
    let color = if line.contains("best") || line.contains("Beat parent") {
        egui::Color32::from_rgb(80, 220, 80) // green for progress
    } else if line.contains("Failed") || line.contains("exhausted") || line.contains("backtracking") {
        egui::Color32::from_rgb(255, 140, 80) // orange for setbacks
    } else if line.contains("Migration") {
        egui::Color32::from_rgb(100, 180, 255) // blue for migration
    } else if line.contains("ERROR") || line.contains("Failed to") {
        egui::Color32::from_rgb(255, 100, 100) // red for errors
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
```

- [ ] **Step 3: Add momentum insight to evolution success messages**

In `governor.rs`, the `advance()` method logs messages like `[EVOLUTION] New global best` and `[EVOLUTION] Beat parent`. Enhance these to include which parameters had the strongest momentum — this tells the user what the evolution is "focusing on."

Add a helper method to `MutationMomentum`:

In `momentum.rs`, add:

```rust
    /// Return the top N parameters by absolute momentum magnitude.
    pub fn top_params(&self, n: usize) -> Vec<(&str, f32)> {
        let mut entries: Vec<(&str, f32)> = self.momentum.iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        entries.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(n);
        entries
    }
```

Then in `governor.rs`, in the `advance()` method, after the success log messages, add momentum insight. Find the `log_msg` calls for success and append parameter info:

Where the governor logs success, format the top momentum params. Since `advance()` returns log messages via `Vec<String>`, find where those strings are built and append momentum info. For example:

```rust
// After identifying winners and updating momentum:
let top = self.momentums[self.active_island].top_params(3);
if !top.is_empty() {
    let trends: Vec<String> = top.iter()
        .map(|(name, val)| {
            let dir = if *val > 0.0 { "↑" } else { "↓" };
            let short = match *name {
                "memory_capacity" => "mem",
                "processing_slots" => "slots",
                "representation_dim" => "repr",
                "learning_rate" => "lr",
                "decay_rate" => "decay",
                "distress_exponent" => "distress",
                "habituation_sensitivity" => "hab",
                "max_curiosity_bonus" => "curiosity",
                "fatigue_recovery_sensitivity" => "fat_rec",
                "fatigue_floor" => "fat_fl",
                other => other,
            };
            format!("{}{}", short, dir)
        })
        .collect();
    msgs.push(format!("[EVOLUTION] Momentum trending: {}", trends.join(", ")));
}
```

Check how `advance()` returns messages — it likely returns a `Vec<String>` that main.rs feeds to `log_msg`. Add the momentum trend message to that vec.

- [ ] **Step 4: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`

- [ ] **Step 5: Commit**

```bash
git add crates/xagent-sandbox/src/main.rs crates/xagent-sandbox/src/governor.rs crates/xagent-sandbox/src/momentum.rs
git commit -m "feat: replace death/spawn console noise with evolution insights and momentum trends"
```

---

### Task 5: Remove energy/integrity bars from sidebar

The sidebar shows compact E/I progress bars for each agent. Remove them — the information is available in the agent detail tab and the HUD bars.

**Files:**
- Modify: `crates/xagent-sandbox/src/main.rs:2308-2335` (sidebar vitals grid)

- [ ] **Step 1: Remove the vitals grid from each sidebar agent entry**

In `main.rs`, in the sidebar agent list rendering (inside the `frame.show()` closure), find the grid block that renders E/I bars. It starts around line 2308 with the comment `// Compact vitals: E/I bars + combined history chart` and the grid at line 2313-2335.

Remove lines from `// Compact vitals` through the grid's closing `});` — that's approximately lines 2308-2335. Keep the phase/death/food line that follows.

Also remove the `energy_frac` and `integrity_frac` variable declarations since they'll be unused.

- [ ] **Step 2: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```bash
git add crates/xagent-sandbox/src/main.rs
git commit -m "feat: remove energy/integrity bars from sidebar"
```
