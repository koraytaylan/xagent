# Project: xagent

## Contributing Rules
All rules in [CONTRIBUTING.md](CONTRIBUTING.md) must be strictly followed. That file is the single source of truth for code style, naming, numeric safety, GPU/buffer safety, WGSL safety, async readback, state invariants, concurrency, performance, serialization compatibility, testing, CI discipline, logging, and related rules.

## Build & Test
- `cargo check -p xagent-sandbox` — quick compile check for the sandbox crate
- `cargo test -p xagent-sandbox` — runs 83 lib unit + 11 bin unit + 60 integration tests (154 total). GPU tests self-skip without an adapter; CI/dev installs Mesa lavapipe.

## Architecture
- `crates/xagent-sandbox/src/governor.rs` — evolution state machine, SQLite persistence
- `crates/xagent-sandbox/src/ui.rs` — egui 0.31 immediate-mode UI, `EvolutionSnapshot` bridges governor↔UI
- `crates/xagent-sandbox/src/sim_runtime.rs` — simulation worker thread: owns `GpuKernel` and all sim-cadence scheduling, advances ticks on a wall-clock cadence, enforces the generation tick budget, and publishes CPU-visible state to the main thread via bounded `SimCommand`/`SimEvent` channels (latest-wins snapshots)
- `crates/xagent-sandbox/src/main.rs` — app loop: drains worker events, applies the newest snapshot to CPU agent caches, drives the generation handoff, and renders (no GPU dispatch/readback in the redraw path)
- `crates/xagent-sandbox/src/gpu_orchestration.rs` — main-thread side of the worker boundary: start/stop worker, drain events, apply snapshots, forward speed/pause/selection on change
- DB migrations are idempotent: `let _ = db.execute_batch("ALTER TABLE ... ADD COLUMN ...");`
- `crates/xagent-brain/src/gpu_kernel.rs` — fused kernel: single dispatch(agent_count,1,1) per vision-stride cycle, the sole GPU abstraction for all simulation. Compute (`dispatch_ticks`) is split from CPU-visible publication (`request_state_snapshot` / `try_collect_state_snapshot`); `dispatch_batch` is the combined compatibility wrapper
- `crates/xagent-brain/src/buffers.rs` — GPU buffer layout constants, sensory packing, AgentBrainState, AgentTelemetry
- `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl` — fused per-agent kernel (physics + food detect + death/respawn + brain, looped over vision_stride cycles)
- `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl` — the 7 cooperative brain passes; credit assignment is a TD(λ) actor-critic (value head + eligibility traces in `brain_state`), no history ring
- `crates/xagent-brain/src/shaders/kernel/global_tick.wgsl` — grid rebuild + collision pass (dispatched as (1,1,1))

## egui Gotchas
- `ui.available_size().y` is INFINITY inside `ScrollArea::vertical()` — use `available_width()` and let content drive height
- Nested ScrollAreas cause horizontal scrollbar — avoid wrapping a tab in outer ScrollArea if inner panes have their own
- `return` inside `ui.collapsing` closure exits the closure only; in a bare block it exits the whole method
- `ui.columns` closure borrows `&mut Ui` — clone data or read between column group calls to avoid borrow conflicts
- `ui.group` sizes to content, not its parent rect — use `ui.set_min_size(rect_size - padding)` inside the group closure to fill allocated space. Same applies to child UIs created via `ui.new_child(UiBuilder)`.
- `allocate_ui` inside `ui.horizontal` inherits horizontal layout — children stack sideways. Use `ui.new_child(UiBuilder::new().max_rect(rect).layout(top_down))` for manual rect-based pane layouts instead.
- `CollapsingHeader` toggles on any header click. For arrow-only toggle, use `CollapsingState::show_header` — it renders the native arrow (toggle on arrow only) and takes a closure for custom header content.

## CI/CD
- `.github/workflows/ci.yml` — check + test on Linux, triggers on push/PR to `develop`
- `.github/workflows/release.yml` — tag-triggered (`v*`) release: test → build 4 targets → changelog → merge to `main` → GitHub Release
- `cliff.toml` — git-cliff config for conventional commit changelog generation
- `rust-toolchain.toml` — pins Rust stable channel for CI and local dev
- Release flow: tag on `develop` (`git tag v0.x.0 && git push origin v0.x.0`) triggers the full pipeline

## Specs & Plans
- Specs: `docs/superpowers/specs/`, Plans: `docs/superpowers/plans/`
