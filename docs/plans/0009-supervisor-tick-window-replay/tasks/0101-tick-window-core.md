---
id: tick-window-core
title: "Tick window core, codec and bounded store"
workstream: "0100"
kind: task
role: developer
depends_on: []
gated: false
touches: [crates/xagent-sandbox/src/lib.rs, crates/xagent-sandbox/src/tick_window.rs, crates/xagent-sandbox/src/tick_window_store.rs]
status: done
merged_as: "6c5bb8479811b97ae2314fffa9578fd133f2729e"
checks:
  - {name: focused-proof, command: "cargo test -p xagent-sandbox --lib tick_window"}
---

# Tick window core, codec and bounded store

## Outcome

`lib.rs` declares the two new modules. `tick_window.rs` holds bounds, config, pins, the frame, the rolling `TickWindow` and a versioned codec. `tick_window_store.rs` holds the additive schema, a writer that keeps the database equal to the in-memory window, and a read-only loader.

## Scope

- `lib.rs`: two `pub mod` lines. Both new files carry inline `#[cfg(test)] mod tests`.
- Reuse `crate::replay::TickRecord` unchanged. No plan or task ids in comments or names; the GPU kernel type name never appears in the two files (the scan test uses `concat!("Gpu", "Kernel")`).

## Non-goals

- No other files, no `Cargo.toml` change, no pin-selection policy, no migration.

## Decisions

- **Constants** (`pub const`, `///` docs): `DEFAULT_TICK_WINDOW_TICKS: u64 = 20_000`, `MAX_WINDOW_FRAMES: usize = 4_096`, `PIN_HALF_WINDOW_TICKS: u64 = 2_000`, `MAX_PINNED_WINDOWS: usize = 32`, `MAX_FRAMES_PER_PIN: usize = 512`, `MAX_FRAME_AGENTS: usize = 1_024`, `MAX_FRAME_FOOD: usize = 65_536`, `PHASE_NOT_RECORDED: u8 = 0`, `FRAME_CODEC_VERSION: u32 = 1`.
- **Config**: `TickWindowConfig { window_ticks, max_frames, pin_half_window_ticks, max_pins, max_frames_per_pin }`, public fields, `Default` from the constants, `validate()` gives `InvalidConfig(field)` for any zero field.
- **Pins**: `PinKind { Replacement { slot: usize }, SeasonFlip }`, `Pin { tick, kind }`. Duplicate means same tick and same variant; slot ignored.
- **Frame**: `WindowFrame { tick: u64, agents: Vec<replay::TickRecord>, food: Vec<[f32; 4]> }`.
- **record_from_physics_row(row: &[f32])**: position from `P_POS_X/Y/Z`, `yaw = P_YAW`, `alive = row[P_ALIVE] > 0.5`, `energy`, `integrity`, `motor_forward = P_MOTOR_FWD_OUT`, `motor_turn = P_MOTOR_TURN_OUT`, `exploration_rate = P_EXPLORATION_RATE_OUT`, `prediction_error = P_PREDICTION_ERROR`, `gradient = P_GRADIENT_OUT`, `raw_gradient = P_RAW_GRADIENT_OUT`, `urgency = P_URGENCY_OUT`, `fatigue_factor = P_FATIGUE_FACTOR_OUT`. Others: `phase = PHASE_NOT_RECORDED`, `vision_color = None`, rest zero. `PHASE_NOT_RECORDED` equals RANDOM's 0 in `replay.rs`; the binding ARCHITECTURE fixes the value.
- **frame_from_snapshot(tick, physics, food)**: `PhysicsLength` unless a multiple of `PHYS_STRIDE`; `TooManyAgents` above `MAX_FRAME_AGENTS`; `FoodLength` unless a multiple of `FOOD_STATE_STRIDE`; `TooManyFood` above `MAX_FRAME_FOOD`.
- **TickWindow::new(config) -> Result**: validates. `capture(frame) -> bool`: `false`, nothing stored, when `frame.tick <= newest`; else append and prune. `pin(pin)`: ignored if duplicate; else append, drop index 0 when over `max_pins`, prune.
- **prune**: (1) for each pin in insertion order, take frames with `|tick - pin.tick| <= pin_half_window_ticks`, sort by `(distance, tick)`, mark the first `max_frames_per_pin` as pinned. (2) Drop unmarked frames with `newest - tick > window_ticks` (saturating). (3) If unmarked frames still exceed `max_frames`, drop the oldest unmarked. Pinned frames are never dropped; order preserved.
- **Accessors**: `frames()` tick order, `frames_in(start, end)` inclusive, `pins()`, `newest_tick() -> Option<u64>`, `len()` (pinned plus rolling), `config()`.
- **Codec** (little-endian): `u32` version, `u64` tick, `u32` agent count, agent records, `u32` food count, food as four `f32`. Agent: `f32` x, y, z, yaw; `u8` alive; `f32` energy, integrity, motor_forward, motor_turn, exploration_rate, prediction_error, gradient, raw_gradient, urgency, credit_magnitude; `u16` patterns_recalled; `u8` phase; `f32` mean_attenuation, curiosity_bonus, fatigue_factor, staleness; `u8` vision flag, then if 1 a `u32` count and that many `f32`.
  Decode: wrong version `UnsupportedVersion`; count above bound `CountOutOfBounds`; each count must fit remaining bytes before allocating, else `Truncated`; leftover bytes `TrailingBytes`; non-finite agent or food position `NonFinite`.
- **Errors**: `TickWindowError { InvalidConfig(&'static str), PhysicsLength(usize), FoodLength(usize), TooManyAgents(usize), TooManyFood(usize), UnsupportedVersion(u32), Truncated, TrailingBytes, CountOutOfBounds(u64), NonFinite }`, with `Debug, PartialEq, Eq, Display, Error`.
- **Schema**: `ensure_schema(&Connection) -> rusqlite::Result<()>` runs the two `CREATE TABLE IF NOT EXISTS` statements from ARCHITECTURE. Pin kinds: `"replacement"` (slot set), `"season"` (slot NULL).
- **Writer**: `TickWindowStore { conn, run_id, window }`. `open_for_latest_supervisor_run(db_path, config)`: read-write open, pragma `PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;`, `SELECT id FROM run WHERE mode = 'supervisor' ORDER BY id DESC LIMIT 1` (none gives `NoSupervisorRun`), `ensure_schema`, empty window. `record_frame(&WindowFrame)` and `record_pin(Pin)`: one transaction each, call the window method, then delete this run's frame rows whose tick is not in `window.frames()`, and replace this run's pin rows with `window.pins()`. A rejected capture writes nothing.
- **Loader**: `load_read_only(db_path, run_id: Option<i64>, config)` opens `SQLITE_OPEN_READ_ONLY | SQLITE_OPEN_NO_MUTEX`, busy timeout only, no pragma or DML. `None` = latest supervisor run; `Some(id)` must be a supervisor run, else `NoSupervisorRun`. Missing replay tables give an empty window. Pins first (`TooManyRows` over `max_pins`; `UnknownPinKind`), then frames by tick with a `LIMIT` of `max_frames + max_pins * max_frames_per_pin + 1` (`TooManyRows`); each blob via `decode_frame`, tick mismatch gives `TickMismatch`.
- **Store error**: `TickWindowStoreError { Sqlite(rusqlite::Error), Window(TickWindowError), NoSupervisorRun, UnknownPinKind(String), TooManyRows { table, limit }, TickMismatch { row, frame } }`, with `Display`, `Error`, `From`.

## Steps

**Steps:**
1. Add the two `pub mod` lines to `lib.rs`. Create `tick_window.rs` with constants, `TickWindowError`, `TickWindowConfig`, `Pin`, `PinKind`, `WindowFrame`.
2. In `tick_window.rs`, add `record_from_physics_row`, `frame_from_snapshot`, `TickWindow` and its prune rule. Add the in-module tests for config validation, rolling eviction, the rolling and per-pin caps, pin caps, out-of-order frames, pinned survival, duplicate pins, bad snapshot lengths and not-recorded fields.
3. In `tick_window.rs`, add `encode_frame` and `decode_frame`. Add codec tests for round trip, wrong version, truncation, trailing bytes, oversized count and non-finite position, and `tick_window_source_names_no_gpu_kernel`, which scans `include_str!` of both new files for `concat!("Gpu", "Kernel")`.
4. Create `tick_window_store.rs` with the pragma, `ensure_schema`, `TickWindowStoreError`, `TickWindowStore`, `open_for_latest_supervisor_run`, `record_frame` and `record_pin`.
5. In `tick_window_store.rs`, add `load_read_only`. Add tests for idempotent schema, loading a database seeded with `supervisor_store::ensure_schema` and a minimal `run (id, mode)` table, loading a database without replay tables, and refusing a database with no supervisor run.
6. Add writer and loader tests: disk rows equal the window after every write, disk pins survive eviction, `load_read_only` leaves bytes unchanged, and loader rejection of a bad blob, oversized row count and tick mismatch. Use `tempfile::tempdir()`.

- **Done when:** the check `focused-proof` passes with every test from steps 2 to 6 reported ok, including `tick_window_source_names_no_gpu_kernel` (INV-2); the rolling eviction, rolling cap, per-pin cap and pin cap tests (INV-3); `store_disk_bounds_match_window` (INV-4); `tick_window_pinned_frames_survive_rolling_eviction` and `store_disk_pins_survive_rolling_eviction` (INV-5); `store_schema_is_idempotent` and `store_loads_pre_existing_supervisor_database` (INV-6); `store_load_read_only_leaves_bytes_unchanged` (INV-7); and the codec rejection tests plus `store_load_rejects_wrong_version_blob` and `store_load_rejects_oversized_row_count` (INV-8). A failing test, or zero matched tests, fails the task.
