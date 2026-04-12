# Dynamic Vision Dimensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make vision dimensions fully dynamic so changing `vision_width`/`vision_height` in `BrainConfig` works end-to-end without breaking the vision display or simulation.

**Architecture:** The WGSL shader side already derives all vision constants from `VISION_W`/`VISION_H` (string-replaced at compile time). The GPU readback and buffer sizing already use the dynamic `BrainLayout`. The fix targets two areas: (1) the vision dispatch shader's workgroup limit of 256 threads that caps ray count, and (2) dead/stale Rust code that hardcodes 8x6 constants.

**Tech Stack:** Rust, WGSL (WebGPU shading language), wgpu

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/xagent-brain/src/shaders/kernel/vision_tick.wgsl` | Modify | Fix ray loop to handle > 256 rays |
| `crates/xagent-brain/src/buffers.rs` | Modify | Remove dead code, make `pack_sensory_frame` layout-aware, update tests |

---

### Task 1: Fix vision dispatch to support > 256 rays per agent

The `vision_tick.wgsl` shader has `@workgroup_size(256)` and guards ray casting with `if (tid < VISION_RAYS)`. With 32×24 = 768 rays, only threads 0–255 cast rays, leaving 256–767 empty. Fix by having each thread loop over multiple rays.

**Files:**
- Modify: `crates/xagent-brain/src/shaders/kernel/vision_tick.wgsl`

- [ ] **Step 1: Update `vision_tick` to loop over rays**

Replace the single-ray-per-thread guard with a strided loop so each of the 256 threads handles `ceil(VISION_RAYS / 256)` rays:

```wgsl
// ── Vision dispatch: multi-workgroup, one workgroup per agent ───────────────
// dispatch(agent_count, 1, 1) — each workgroup's 256 threads cooperatively
// cast all VISION_RAYS rays (looping when VISION_RAYS > 256),
// then thread 0 packs proprioception/interoception/touch into sensory_buf.

@compute @workgroup_size(256)
fn vision_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
) {
    let agent_id = wgid.x;
    let tid = lid.x;

    if (agent_phys[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }

    // Each thread casts rays in a strided loop (handles VISION_RAYS > 256)
    for (var ray = tid; ray < VISION_RAYS; ray += 256u) {
        vision_single_ray(agent_id, ray);
    }
    storageBarrier(); workgroupBarrier();

    // Thread 0 packs sensory data (proprioception, interoception, touch)
    if (tid == 0u) {
        phase_vision_senses(agent_id);
    }
}
```

- [ ] **Step 2: Verify shader compiles**

Run: `cargo check -p xagent-brain`
Expected: compiles clean (WGSL is validated at pipeline creation, but Rust compilation confirms no syntax issues in the include)

- [ ] **Step 3: Commit**

```bash
git add crates/xagent-brain/src/shaders/kernel/vision_tick.wgsl
git commit -m "fix: loop vision rays so workgroup handles > 256 rays per agent"
```

---

### Task 2: Make `pack_sensory_frame` layout-aware

`pack_sensory_frame` uses hardcoded `VISION_COLOR_COUNT` (192) and `VISION_DEPTH_COUNT` (48). It's currently only called from its own test, but it's a public API that should work correctly for any vision dimensions.

**Files:**
- Modify: `crates/xagent-brain/src/buffers.rs:290-370` (function)
- Modify: `crates/xagent-brain/src/buffers.rs:737-742` (test)

- [ ] **Step 1: Update the test to use a non-default layout**

Add a test that packs a frame with non-default vision dimensions:

```rust
#[test]
fn pack_sensory_frame_dynamic_layout() {
    let layout = BrainLayout::new(12, 8);
    let frame = SensoryFrame::new_blank(12, 8);
    let mut buf = vec![0.0_f32; layout.sensory_stride];
    pack_sensory_frame(&frame, &layout, &mut buf);
    assert!(buf.iter().all(|v| v.is_finite()));
}
```

Run: `cargo test -p xagent-brain pack_sensory_frame_dynamic_layout`
Expected: FAIL — `pack_sensory_frame` doesn't accept a `&BrainLayout` parameter

- [ ] **Step 2: Add `&BrainLayout` parameter to `pack_sensory_frame`**

Change the signature and body to use layout-derived counts:

```rust
/// Pack a SensoryFrame into a flat f32 slice for GPU upload.
/// Selects up to 4 highest-intensity touch contacts, zero-pads the rest.
pub fn pack_sensory_frame(frame: &SensoryFrame, layout: &BrainLayout, out: &mut [f32]) {
    debug_assert!(out.len() >= layout.sensory_stride);

    let mut offset = 0;

    // Vision color
    let color_len = frame.vision.color.len().min(layout.vision_color_count);
    out[offset..offset + color_len].copy_from_slice(&frame.vision.color[..color_len]);
    for i in color_len..layout.vision_color_count {
        out[offset + i] = 0.0;
    }
    offset += layout.vision_color_count;

    // Vision depth
    let depth_len = frame.vision.depth.len().min(layout.vision_depth_count);
    out[offset..offset + depth_len].copy_from_slice(&frame.vision.depth[..depth_len]);
    for i in depth_len..layout.vision_depth_count {
        out[offset + i] = 0.0;
    }
    offset += layout.vision_depth_count;

    // ... rest of function unchanged (velocity, facing, angular_vel, energy, etc.)
```

- [ ] **Step 3: Update existing test to pass layout**

```rust
#[test]
fn pack_sensory_frame_fills_buffer() {
    let layout = BrainLayout::default();
    let frame = SensoryFrame::new_blank(layout.vision_width, layout.vision_height);
    let mut buf = vec![0.0_f32; layout.sensory_stride];
    pack_sensory_frame(&frame, &layout, &mut buf);
    assert!(buf.iter().all(|v| v.is_finite()));
}
```

- [ ] **Step 4: Run tests to verify both pass**

Run: `cargo test -p xagent-brain pack_sensory_frame`
Expected: 2 tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/xagent-brain/src/buffers.rs
git commit -m "fix: make pack_sensory_frame layout-aware for dynamic vision dimensions"
```

---

### Task 3: Remove dead WGSL generator functions and clean up stale constants

`wgsl_constants()` and `wgsl_physics_constants()` are never called — the fused kernel uses `common.wgsl` with string replacements instead. Remove them to eliminate confusion about which constants are authoritative. Also remove the hardcoded `VISION_W`, `VISION_H`, `VISION_COLOR_COUNT`, `VISION_DEPTH_COUNT` Rust constants (they exist only for the now-layout-aware `pack_sensory_frame` and dead code). Keep `NON_VISUAL_COUNT` (used by `BrainLayout::new`).

**Files:**
- Modify: `crates/xagent-brain/src/buffers.rs`

- [ ] **Step 1: Remove `wgsl_constants()` function**

Delete the function at lines 372–466. It's never called (verified via grep).

- [ ] **Step 2: Remove `wgsl_physics_constants()` function**

Delete the function at lines 468–575. It's never called (verified via grep).

- [ ] **Step 3: Remove hardcoded vision constants**

Remove these lines:
```rust
pub const VISION_W: usize = 8;
pub const VISION_H: usize = 6;
pub const VISION_COLOR_COUNT: usize = VISION_W * VISION_H * 4;
pub const VISION_DEPTH_COUNT: usize = VISION_W * VISION_H;
pub const SENSORY_STRIDE: usize = VISION_COLOR_COUNT + VISION_DEPTH_COUNT + NON_VISUAL_COUNT;
```

Keep `NON_VISUAL_COUNT` — it's used by `BrainLayout::new()` and is vision-independent.

- [ ] **Step 4: Update convenience wrappers to derive layout from config**

`init_brain_state()` and `build_config()` use `BrainLayout::default()`. Update them to derive layout from the config's `vision_width`/`vision_height`:

```rust
pub fn init_brain_state(config: &BrainConfig, rng: &mut impl rand::Rng) -> Vec<f32> {
    init_brain_state_for(config, &BrainLayout::new(config.vision_width, config.vision_height), rng)
}

pub fn build_config(config: &BrainConfig) -> Vec<f32> {
    build_config_for(config, &BrainLayout::new(config.vision_width, config.vision_height))
}
```

- [ ] **Step 5: Update `AgentBrainState::new()` to accept brain_stride**

`AgentBrainState::new()` uses hardcoded `BRAIN_STRIDE`. Since it's only used in tests, change it to require explicit size (it already has `new_for(brain_stride)` — just remove the default `new()`):

```rust
// Remove:
// pub fn new() -> Self { Self::new_for(BRAIN_STRIDE) }
```

Or keep it using `BrainLayout::default()`:

```rust
pub fn new() -> Self {
    Self::new_for(BrainLayout::default().brain_stride)
}
```

- [ ] **Step 6: Fix all compilation errors from removed constants**

Any remaining references to the removed constants need updating. These will be in tests (Task 4 handles those).

- [ ] **Step 7: Verify compilation**

Run: `cargo check -p xagent-brain`
Expected: compiles clean

- [ ] **Step 8: Commit**

```bash
git add crates/xagent-brain/src/buffers.rs
git commit -m "fix: remove dead WGSL generators and hardcoded vision constants"
```

---

### Task 4: Update tests for dynamic vision

Tests currently assert against removed hardcoded constants. Update them to use `BrainLayout::default()` values or test dynamic layouts directly.

**Files:**
- Modify: `crates/xagent-brain/src/buffers.rs` (test module, lines 709+)

- [ ] **Step 1: Update `sensory_stride_matches_feature_count`**

```rust
#[test]
fn sensory_stride_matches_feature_count() {
    let layout = BrainLayout::default();
    // Default 8x6: 192 color + 48 depth + 27 non-visual = 267 sensory
    assert_eq!(layout.sensory_stride, 267);
    // Feature count excludes 2 non-visual fields (energy_delta, integrity_delta)
    assert_eq!(layout.feature_count, 265);
    assert!(layout.sensory_stride >= layout.feature_count);
}
```

- [ ] **Step 2: Update `brain_stride_is_consistent`**

```rust
#[test]
fn brain_stride_is_consistent() {
    let layout = BrainLayout::default();
    let expected_tail_end = layout.feature_count * DIM + DIM + DIM * DIM + FIXED_TAIL_SIZE;
    assert_eq!(layout.brain_stride, expected_tail_end);
}
```

- [ ] **Step 3: Update `brain_layout_default_matches_constants`**

Remove assertions against deleted constants. The test already checks internal consistency:

```rust
#[test]
fn brain_layout_default_matches_constants() {
    let layout = BrainLayout::default();
    assert_eq!(layout.vision_width, 8);
    assert_eq!(layout.vision_height, 6);
    // Verify default layout values for 8x6 vision
    assert_eq!(layout.vision_color_count, 192);
    assert_eq!(layout.vision_depth_count, 48);
    assert_eq!(layout.sensory_stride, 267);
    assert_eq!(layout.feature_count, 265);
}
```

- [ ] **Step 4: Update `init_brain_state_has_correct_length`**

```rust
#[test]
fn init_brain_state_has_correct_length() {
    let config = BrainConfig::default();
    let layout = BrainLayout::new(config.vision_width, config.vision_height);
    let mut rng = rand::rng();
    let state = init_brain_state(&config, &mut rng);
    assert_eq!(state.len(), layout.brain_stride);
}
```

- [ ] **Step 5: Update `pack_sensory_frame_fills_buffer` (if not done in Task 2)**

Already handled in Task 2.

- [ ] **Step 6: Update any remaining test references to removed constants**

Check for `BRAIN_STRIDE`, `SENSORY_STRIDE`, `FEATURE_COUNT`, `VISION_W`, `VISION_H` in tests and replace with `BrainLayout` equivalents.

- [ ] **Step 7: Add test for large vision dimensions**

```rust
#[test]
fn brain_layout_large_vision() {
    let layout = BrainLayout::new(32, 24);
    assert_eq!(layout.vision_color_count, 32 * 24 * 4);
    assert_eq!(layout.vision_depth_count, 32 * 24);
    assert_eq!(layout.sensory_stride, 32 * 24 * 5 + NON_VISUAL_COUNT);
    // Verify brain_stride is consistent
    let expected = layout.feature_count * DIM + DIM + DIM * DIM + FIXED_TAIL_SIZE;
    assert_eq!(layout.brain_stride, expected);
}
```

- [ ] **Step 8: Run all tests**

Run: `cargo test -p xagent-brain`
Expected: all tests pass

- [ ] **Step 9: Commit**

```bash
git add crates/xagent-brain/src/buffers.rs
git commit -m "fix: update tests for dynamic vision layout, remove hardcoded constant assertions"
```

---

### Task 5: Final validation — clippy, fmt, full test suite

**Files:** All modified files

- [ ] **Step 1: Run rustfmt**

Run: `cargo fmt --all -- --check`
Expected: no formatting issues

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: no warnings

- [ ] **Step 3: Run full test suite**

Run: `cargo test -p xagent-sandbox`
Expected: all tests pass

- [ ] **Step 4: Fix any issues and commit**

```bash
git commit -m "fix: address clippy/fmt issues"
```
