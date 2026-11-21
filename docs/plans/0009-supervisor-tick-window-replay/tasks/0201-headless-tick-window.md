---
id: headless-tick-window
title: "Headless capture and capture-on/off digest gate"
workstream: "0200"
kind: task
role: developer
depends_on: [tick-window-core]
gated: false
touches: [crates/xagent-sandbox/src/headless.rs, crates/xagent-sandbox/tests/supervisor_tick_window_digest.rs]
status: todo
merged_as: ""
checks:
  - {name: focused-proof, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_tick_window_digest", timeout_ms: 3600000}
  - {name: supervisor-headless, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_headless", timeout_ms: 3600000}
---

# Headless capture and capture-on/off digest gate

## Outcome

`headless.rs` can run a supervisor session with an optional tick window. When the window is on, every supervisor sample is captured into an in-memory `TickWindow` from the physics and food copies the loop already makes, pinned on replacement and season-flip boundaries, and persisted through `TickWindowStore`. With the window off, behaviour is unchanged. A new GPU test file proves capture on and off produce the same supervisor digest, the same non-replay table rows and the same run summary bytes, and that captured frames surround each recorded replacement pin.

## Scope

- `headless.rs` only, plus the new test file.
- Add `pub fn run_headless_supervisor_with_tick_window(config: FullConfig, db_path: &str, options: &HeadlessOptions, tick_window: Option<TickWindowConfig>) -> Result<(), HeadlessError>`. It returns `HeadlessError::TickWindow(String)` when `config.evolution_mode` is not `Supervisor`, before any database write.
- Add the `HeadlessError::TickWindow(String)` variant with a `Display` arm.
- `run_supervisor` gains the same `Option<TickWindowConfig>` parameter. `run_headless_with_options` calls it with `None`, so its behaviour and bytes are unchanged.
- `supervisor_step` returns the existing `on_sample` result together with the `physics` and `food` copies it already makes. It does not make new copies or kernel calls.
- Use `tick_window::frame_from_snapshot`, `TickWindow::capture`, `TickWindow::pin`, `TickWindowStore::open_for_latest_supervisor_run`, `record_frame` and `record_pin` from `tick-window-core`.
- Capture runs only after `run.on_sample` returns, inside a helper with no kernel parameter.
- Open the store after `SupervisorRun::start`, and only when a window config is given.
- Do not change `HeadlessOptions` fields; its literal sites in `main.rs` and tests must stay as they are.

## Non-goals

- No new kernel, GPU dispatch, readback or WGSL change.
- No change to `supervisor_run.rs`, `SampleReport`, `BoundaryReport`, `SupervisorConfig` or any supervisor decision, schedule or random stream.
- No CLI flag and no change to `main.rs`.
- No change to `replay.rs`, `replay_coord.rs` or generation recording.
- No change to existing tests in `tests/supervisor_headless.rs`.

## Decisions

- **Pins**: after a sample, if `report.boundary` names a replaced slot (not an idle boundary), record `Pin { tick: report.tick, kind: PinKind::Replacement { slot } }`. If `report.season` is `Some`, record `Pin { tick: report.tick, kind: PinKind::SeasonFlip }`. Read `BoundaryReport` and `SampleReport` in `supervisor_run.rs` for the field names; they already exist.
- **Frame**: `frame_from_snapshot(report.tick, &physics, food.as_deref().unwrap_or(&[]))`. A frame error becomes `HeadlessError::TickWindow(message)` and stops the run.
- **Helper**: `fn capture_sample(window: &mut TickWindow, store: &mut TickWindowStore, report: &SampleReport, physics: &[f32], food: Option<&[f32]>) -> Result<(), HeadlessError>`. Its parameter list has no `kernel` argument, and its body names no kernel type.
- **Digest in the test**: the 0006 digest helper is test-local in the 0006 test files. Recompute the supervisor digest in the new test with the same inputs and the same fold that `tests/supervisor_headless.rs` uses in `supervisor_headless_store_is_reproducible`. Do not edit that file.
- **Seeded config**: copy `supervisor_config()` from `tests/supervisor_headless.rs` into the new file. Use the shortest tick budget that yields at least one replacement pin and at least one season-flip pin. The test asserts both counts, so a budget that produces neither fails the test instead of passing silently.
- **Timestamp columns**: exclude any column whose name ends in `_at` or contains `time`, listed explicitly in the test after reading the schema.
- **INV-2 proof**: the test reads `headless.rs` with `include_str!`, extracts the text of `fn capture_sample` up to its closing brace, and asserts it contains no `kernel` substring. It also asserts the `capture_sample` call site is after the `on_sample` call inside `supervisor_step`.
- **GPU guard**: the file starts with the `gpu_adapter_present_when_required` pattern from `tests/supervisor_headless.rs`. It returns early only when no adapter is present and `XAGENT_REQUIRE_GPU` is unset. It panics when the adapter is absent and `XAGENT_REQUIRE_GPU` is set.

## Steps

**Steps:**
1. In `headless.rs`, add the `HeadlessError::TickWindow(String)` variant and its `Display` arm. Add `run_headless_supervisor_with_tick_window`, which rejects a non-supervisor `evolution_mode` before opening any database. Make `run_supervisor` take `Option<TickWindowConfig>` and pass `None` from `run_headless_with_options`.
2. In `headless.rs`, change `supervisor_step` to return `(on_sample result, physics, food)` using the copies it already makes. Update its call sites only. Keep the `on_sample` call and its arguments unchanged.
3. In `headless.rs`, add `capture_sample` with the signature in Decisions. Inside `run_supervisor`, when the window config is `Some`, open `TickWindowStore::open_for_latest_supervisor_run` after `SupervisorRun::start` and call `capture_sample` after each sample. Leave the no-window path unchanged.
4. Create `tests/supervisor_tick_window_digest.rs` with the GPU guard, a copy of `supervisor_config()`, a temp-directory helper, and a helper that runs the headless supervisor with capture off (`None`) and on (`Some(TickWindowConfig::default())`) into two tempfile databases.
5. In the same file, add test (1): assert equal recomputed supervisor digests, equal rows in every table except `supervisor_replay_frame` and `supervisor_replay_pin` (with timestamp columns excluded), equal run-summary JSON bytes, and that the capture-on database has at least one replacement pin and one season-flip pin.
6. In the same file, add test (2) for the bounds of captured frames and pins (INV-5 proof: every replacement pin has at least one frame in `[t - H, t + H]`, with `H = PIN_HALF_WINDOW_TICKS`), and test (3) for the INV-2 source scan described in Decisions.

- **Done when:** the check `focused-proof` passes with the tests in step 5 and step 6 reported ok: capture-on and capture-off digests, non-replay table rows and run-summary bytes are identical (INV-1); the capture-on database holds frames within `MAX_WINDOW_FRAMES` and `MAX_FRAMES_PER_PIN` bounds and the INV-2 source scan finds no kernel call in `capture_sample` (INV-2); and every recorded replacement pin has frames inside `[t - H, t + H]` (INV-5); and the check `supervisor-headless` also passes with no test failing.
