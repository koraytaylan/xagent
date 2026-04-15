# Project: xagent

## Contributing Rules
All rules in [CONTRIBUTING.md](CONTRIBUTING.md) must be strictly followed. That file is the single source of truth for code style, naming, GPU/buffer safety, async readback, concurrency, performance, and testing rules.

## Build & Test
- `cargo check -p xagent-sandbox` — quick compile check for the sandbox crate
- `cargo test -p xagent-sandbox` — runs 58 lib unit + 3 bin unit + 34 integration tests (95 total)

## Architecture
- `crates/xagent-sandbox/src/governor.rs` — evolution state machine, SQLite persistence
- `crates/xagent-sandbox/src/ui.rs` — egui 0.31 immediate-mode UI, `EvolutionSnapshot` bridges governor↔UI
- `crates/xagent-sandbox/src/main.rs` — app loop, pipes governor data to snapshot before paint closure
- DB migrations are idempotent: `let _ = db.execute_batch("ALTER TABLE ... ADD COLUMN ...");`
- `crates/xagent-brain/src/gpu_kernel.rs` — fused kernel: single dispatch(agent_count,1,1) per vision-stride cycle, the sole GPU abstraction for all simulation
- `crates/xagent-brain/src/buffers.rs` — GPU buffer layout constants, sensory packing, AgentBrainState, AgentTelemetry
- `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl` — fused per-agent kernel (physics + food detect + death/respawn + brain, looped over vision_stride cycles)
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
