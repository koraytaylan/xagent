# Contributing to XAgent

These rules are distilled from hundreds of code review comments through PR #100. They represent the project's hard-won invariants.

## Tooling

All code must pass the following before merge:

```bash
cargo fmt --all -- --check   # formatting
cargo clippy --workspace --all-targets -- -D warnings   # lints (pedantic enabled)
cargo test -p xagent-sandbox # tests
```

These are enforced by CI. Run them locally before pushing.

## Code Style

We follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) and
enforce `clippy::pedantic` at the workspace level. The rules below supplement those
defaults.

### Naming

| Rule | Example |
|------|---------|
| No single-letter variables outside closures, iterators, or trivial math (`x`, `y`). | `gw` → `grid_width`, `c` → `contact` |
| No bare abbreviations. Spell out names so they read clearly. | `ppp` → `pixels_per_point`, `fc` → `feature_count` |
| Domain abbreviations that are universal in the project may be used if documented here. | `buf` (buffer), `wt`/`wts` (weight/weights) |
| Buffer layout constants (`O_*`, `P_*`, `CFG_*`), loop variables, and external API types are exempt. | |

### Magic Numbers

Any numeric literal that isn't `0`, `1`, `-1`, `0.0`, or `1.0` should be a named
`const` with a doc comment explaining *why* that value was chosen:

```rust
/// Death penalty decay factor for composite fitness scoring.
const DEATH_PENALTY_DECAY: f32 = 0.5;
```

### Function Length

If a function exceeds roughly 50 lines, consider splitting it into smaller,
single-responsibility helpers. Very long functions (100+ lines) should be
reviewed and refactored as part of normal code review.

### Documentation

- All public types and functions should have `///` doc comments.
- Module-level `//!` docs should describe purpose and key concepts.
- Docstrings must match implementation — "non-blocking" means non-blocking, "pre-allocates" means pre-allocates. Rename functions when behavior changes (e.g., `read_agent_telemetry` → `read_agent_telemetry_blocking`).
- Comments that make performance claims ("non-blocking", "pre-allocates", "reuses allocation") are contractual — verify them on every change.
- When changing a numeric value, default, range, or behavior, search for all comments and docs referencing the old value. `grep` for both the old literal and the concept name.
- Comments referencing removed or renamed parameters, fields, or functions must be updated or deleted in the same commit.
- Distinguish "start operation" from "poll/collect operation" in API naming. A function that both starts and polls should document that clearly.
- Keep PR descriptions synchronized with code. Constant values, radius sizes, and architectural claims must reflect what the code actually does.
- No commented-out code or tombstone comments. Delete removed code completely — git history preserves everything.
- No stale TODOs. If the referenced work is done or abandoned, delete the TODO.
- Do not write values to buffers or fields that nothing reads. Unused writes waste bandwidth and mislead readers about data flow.
- We plan to enable `#![warn(missing_docs)]` incrementally per crate.

### Logging

- In library code and long-lived runtime paths, use `log::warn!` / `log::error!` (or `tracing`) for warnings and errors. Do not use `eprintln!` or `println!` there, because they bypass log filtering and are invisible in structured logging setups. Direct `println!` / `eprintln!` output is acceptable for intentional user-facing CLI messages and in tests/benches where stdout/stderr is part of the interface or harness behavior.

### Commit Messages

Use conventional commit prefixes: `feat:`, `fix:`, `perf:`, `refactor:`, `doc:`, `chore:`.

## Numeric Safety

- Never use `as` for potentially lossy integer conversions (`u64` → `i64`, `u64` → `u32`). Use `try_into()` or `TryFrom` and handle the error.
- Size and count multiplications that derive buffer sizes must use `checked_mul()` or validate against a reasonable upper bound before allocating.
- Any value used as a divisor must be guarded against zero — use `max(value, epsilon)` or check before dividing. This applies to both Rust and WGSL.
- After subtracting from a bounded quantity (energy, integrity), immediately clamp to the valid range in the same scope. Do not defer invariant enforcement to the next tick.

## GPU & Buffer Safety

- All buffer offsets must derive from `BrainLayout` / kernel config, never hardcoded constants. Hardcoded strides like `SENSORY_STRIDE` break when `BrainLayout` uses non-default vision dimensions.
- Validate index and count inputs against kernel state before computing buffer offsets. Out-of-bounds offsets trigger wgpu validation errors or silent corruption.
- Constants shared between Rust and WGSL must have a single canonical source within a given shader pipeline / concatenated header set. A pipeline may use either the `wgsl_physics_constants()` template or the `wconfig` uniform buffer for a given constant, but do not define the same constant from both sources when headers are combined. Use named constants, not magic indices (e.g., `WC_FOOD_RADIUS`, not `wc(7u)`).
- Rust structs used as WGSL `var<uniform>` buffers must match WGSL struct alignment rules (16-byte aligned struct size, `vec4`-aligned members). Use `#[repr(C)]` and explicit padding, or validate with `assert_eq!(std::mem::size_of::<T>() % 16, 0)`.
- When agents have per-instance heritable config, GPU upload functions must iterate all agents, not just apply the first config. Document whether a reset/upload function applies to a single agent or the entire population.

## WGSL Safety

- `select()` evaluates all arguments before choosing — guard divisions with `max(denominator, epsilon)` instead of relying on the condition to skip them.
- When synchronizing reads/writes to `var<storage>` buffers within a workgroup, use `storageBarrier()` together with `workgroupBarrier()` when you need both memory visibility and execution synchronization.
- Early-return in a workgroup shader is only safe when the return condition is uniform across the entire workgroup and occurs before the first barrier; otherwise, restructure control flow to ensure all invocations reach each barrier.
- When removing or renaming a constant or function in a shared WGSL header, grep all shaders that concatenate that header to verify no dangling references.

## Async Readback

- Track in-flight state explicitly. Never overwrite a pending `map_async` operation without unmapping/cleaning up the previous one first.
- Always handle both success and failure paths. If any `map_async` callback fails, the system must clean up and allow retry — "stuck forever" states are bugs.
- Establish data authority. When physics readback and telemetry readback both provide the same field, document which is authoritative and never overwrite fresher data with stale async results.
- Unmap staging buffers on all paths: success, error, agent switch, and generation reset.

## State Invariants

- After modifying a value that has downstream invariants (e.g., energy), immediately re-establish the invariant (clamp, death check) in the same scope. Do not defer invariant enforcement to the next tick.
- When associating cause (action) with effect (reward/gradient), verify temporal alignment: the gradient recorded with a motor command must reflect the outcome of that command, not the previous tick's outcome.
- State machines must handle all states for all input conditions. If a transition requires a precondition (e.g., non-empty agent list), guard the transition — do not enter a stuck state.

## Concurrency

- All SQLite connections must set `busy_timeout`. The default is 0 (instant `SQLITE_BUSY` failure). Both main and background connections need matching timeout policies.
- Background threads must have a deterministic shutdown path. Drop sender → `recv` returns `Disconnected` → thread exits. `Drop` impls must join threads and log any panics.
- Return `Result` or `Option` from fallible operations. Never use `.expect()` for I/O, GPU, or thread operations. Silent failures are worse than explicit errors.

## Performance

- Per-tick simulation logic belongs in WGSL shaders, never in Rust.
- The CPU main loop submits GPU dispatches (batched) and collects async readback results (non-blocking).
- No CPU-side work should scale with `ticks_to_run` — the sandbox loop dispatches GPU work in chunks of at most 500 ticks per frame/batch call, while the internal accumulated tick budget may grow up to 64,000.
- Never clone large collections in per-frame hot paths. Use `clone_from()` for in-place updates or borrow patterns. A throttled rebuild is useless if you deep-clone every frame.
- Use squared-distance comparisons to avoid `sqrt()` in loops. Both CPU and GPU code should use `_SQ` variants for radius checks.
- Only mark throttle windows as consumed when work actually happens. Updating `last_rebuild` without rebuilding silently wastes the throttle budget.

## Serialization & Config Compatibility

- When renaming or restructuring serialized fields (`serde` JSON, SQLite columns), add `#[serde(alias = "old_name")]` or a migration path so existing configs and saved data still load.
- Binary persistence formats (BLOBs, recordings) must document their endianness and validate decoded sizes against expected dimensions. Prefer little-endian for portability.

## Testing

- New concurrency-sensitive code paths require end-to-end tests exercising the async lifecycle: request → complete → consume, and request → fail → cleanup.
- Cache invalidation must be tested: populate → invalidate → verify fresh data returned.
- Non-trivial algorithms need unit tests with representative inputs and boundary conditions.
- Tests must be falsifiable: a test that passes for all inputs (tautology, arithmetic identity) is not a test. If deleting the code under test would not break the test, the test is wrong.
- Tests must assert invariants, not skip violations. Filtering out bad data and asserting on the remainder hides bugs.
- Test names and doc comments must accurately describe what is verified. A claim like "verified here" requires an actual assertion.
- Tests must not depend on hardware availability (GPU). Gate hardware-dependent tests behind a feature flag or `#[cfg]` attribute.

## CI Discipline

- Benchmark and release-profile steps should generally be gated on `workflow_dispatch` or schedule triggers; running them on pushes to `develop` is acceptable when they are intentionally non-blocking CI signal.
- Clippy must lint `--all-targets` to catch issues in tests, benches, and examples.

## Incremental Cleanup

We do not require a big-bang rewrite. Existing code is cleaned up file-by-file
as it is touched. Priority targets:

1. `governor.rs` — extract magic numbers into named constants; split `advance()`.
2. `buffers.rs` — replace short abbreviations (`gw`, `go`, `fc`).
3. `ui.rs` — rename `ppp`; extract inline RGB colors into a palette module.

Files that already meet the standard (e.g., `config.rs`, `body.rs`) should be
left as-is.
