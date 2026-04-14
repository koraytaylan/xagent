# Project: xagent

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

## Performance Invariants
- Per-tick simulation logic belongs in WGSL shaders, never in Rust
- The CPU main loop submits GPU dispatches (batched) and collects async readback results (non-blocking)
- Recording/telemetry/history functions run once per frame, sampling the latest state
- No CPU-side work should scale with `ticks_to_run` — up to 64,000 ticks may be scheduled per frame/batch call, with GPU work dispatched in chunks
- Never clone large collections (`Vec`, `HashMap`) in per-frame hot paths — use `clone_from()` or borrow
- Use squared-distance comparisons to avoid `sqrt()` in loops

## GPU & Buffer Rules
- All buffer offsets must derive from `BrainLayout` / kernel config, never hardcoded constants
- Validate index/count inputs against kernel state before computing buffer offsets
- Constants shared between Rust and WGSL must have a single canonical source for each constant in each shader/pipeline; use whichever source that shader already treats as authoritative (e.g., `wgsl_physics_constants()` or `wconfig`)
- Never define the same constant in multiple headers that get concatenated — import or reference the canonical one for that shader/pipeline

## Async Readback Rules
- Track in-flight state explicitly: never overwrite pending async operations without cleanup
- Always handle both success and failure paths for `map_async` — stuck-forever states are bugs
- When multiple sources provide the same field (e.g., phys readback vs telemetry), document which is authoritative and never overwrite fresher data with stale
- Unmap staging buffers on all paths: success, error, agent switch, and generation reset

## Concurrency Rules
- All SQLite connections must set `busy_timeout` — default is 0 (instant fail)
- Background threads must have a shutdown path: drop sender → recv returns Disconnected → thread exits
- `Drop` impls must join threads and log panics, never silently discard
- Never use `.expect()` for fallible I/O/GPU/thread operations — return `Result` or `Option`

## Testing Rules
- New concurrency-sensitive code paths require end-to-end tests exercising failure modes
- Cache invalidation logic must be tested: populate → invalidate → verify fresh
- Non-trivial algorithms (color generation, distance checks) need unit tests with boundary cases

## Code Style
- Specs: `docs/superpowers/specs/`, Plans: `docs/superpowers/plans/`
- Commit prefixes: `feat:`, `fix:`, `perf:`
- Docstrings must match implementation — "non-blocking" means non-blocking, "pre-allocates" means pre-allocates
- Distinguish "start operation" from "poll/collect operation" in API naming
- Avoid abbreviations in identifiers — use full words (e.g. `vision_width` not `vision_w`, `position` not `pos`). Exceptions: buffer layout constants (O_*, P_*, CFG_*), loop variables, external API types.
- No commented-out code or tombstone comments — delete removed code completely, git history preserves everything
