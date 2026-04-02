# Fix Stagnant Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three root causes preventing evolutionary progress: food invisibility in ray-marching, `representation_dim` mutation destroying weight inheritance, and touch contacts not reaching the encoder.

**Architecture:** Unify the two divergent ray-march functions into one with an `AgentSlice` enum, lock `representation_dim` out of mutation/crossover, and wire touch contacts into the encoder's feature vector as 16 additional input features (top 4 contacts × 4 features each).

**Tech Stack:** Rust, glam (Vec3), xagent-shared (SensoryFrame, TouchContact, BrainConfig), xagent-brain (SensoryEncoder)

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `crates/xagent-sandbox/src/agent/senses.rs` | Unify `march_ray` + `march_ray_positions` into `march_ray_unified` with `AgentSlice` enum. Update both `sample_vision` and `sample_vision_positions` to call unified function. Remove old `march_ray` and `march_ray_positions`. |
| Modify | `crates/xagent-sandbox/src/agent/mod.rs:264-283` | Lock `representation_dim` in `mutate_config_with_strength` |
| Modify | `crates/xagent-sandbox/src/agent/mod.rs:312-330` | Lock `representation_dim` in `crossover_config` |
| Modify | `crates/xagent-brain/src/encoder.rs:9` | Update `NON_VISUAL_FEATURES` from 9 to 25 |
| Modify | `crates/xagent-brain/src/encoder.rs:227-278` | Add touch contact encoding in `extract_features_into` |
| Modify | `crates/xagent-sandbox/tests/integration.rs` | Add tests for food visibility in ray-march, touch contacts in sensory frame |
| Modify | `crates/xagent-brain/src/encoder.rs:282-505` | Add tests for touch contact encoding, feature count with touch |
| Modify | `README.md` | Update sensory docs, vision hit table, representation_dim description, known limitations |
| Modify | `EVOLUTION_JOURNEY.md` | Add entries for food blindness bug (refined), repr_dim inheritance bug, touch encoding gap |
| Modify | `crates/xagent-sandbox/README.md` | Update vision hit table, touch section, mutation docs, known limitations |

---

### Task 1: Unify Ray-Marching Functions

**Files:**
- Modify: `crates/xagent-sandbox/src/agent/senses.rs`

- [ ] **Step 1: Add the `AgentSlice` enum and `march_ray_unified` function**

Replace both `march_ray` (lines 122–186) and `march_ray_positions` (lines 317–363) with a single unified function. Add the `AgentSlice` enum just before the function.

In `senses.rs`, replace the block from `/// Fixed-step ray marching` (line 112) through the end of `march_ray` (line 186) with:

```rust
/// Discriminated union for the two agent-list representations.
/// Avoids duplicating the ray-march loop.
enum AgentSlice<'a> {
    Others(&'a [OtherAgent]),
    Positions { all: &'a [(Vec3, bool)], self_index: usize },
}

/// Fixed-step ray marching for terrain, food, and agent intersection.
///
/// Advances a ray from `origin` in direction `dir` in increments of `step` units,
/// checking for food items, other agents, and terrain at each step. Returns the
/// hit color and distance traveled. If no hit within `max_dist`, returns sky color.
///
/// Food items are rendered as bright lime-green, distinct from biome colors.
/// This makes food *physically visible* — the brain must still learn that this
/// color correlates with positive gradient. Without food visibility, agents have
/// zero directional signal for food.
fn march_ray_unified(
    origin: Vec3,
    dir: Vec3,
    world: &WorldState,
    max_dist: f32,
    step: f32,
    agents: AgentSlice,
) -> ([f32; 4], f32) {
    let sky: [f32; 4] = [0.53, 0.81, 0.92, 1.0];
    let agent_color: [f32; 4] = [0.9, 0.2, 0.6, 1.0];
    let food_color: [f32; 4] = [0.70, 0.95, 0.20, 1.0];
    let agent_radius_sq: f32 = 1.5 * 1.5;
    let food_radius_sq: f32 = 1.0 * 1.0;

    // Early-out: upward-pointing rays above terrain are almost certainly sky.
    if dir.y > 0.3 {
        let origin_h = world.terrain.height_at(origin.x, origin.z);
        if origin.y > origin_h {
            return (sky, max_dist);
        }
    }

    let mut t = 0.0_f32;
    while t < max_dist {
        let p = origin + dir * t;

        // Check food items via spatial grid (O(1) per step)
        for idx in world.food_grid.query_nearby(p.x, p.z) {
            let food = &world.food_items[idx];
            if food.consumed {
                continue;
            }
            let diff = p - food.position;
            if diff.length_squared() < food_radius_sq {
                return (food_color, t);
            }
        }

        // Check agent hits — dispatch on slice variant
        match &agents {
            AgentSlice::Others(others) => {
                for other in *others {
                    if !other.alive {
                        continue;
                    }
                    let diff = p - other.position;
                    if diff.length_squared() < agent_radius_sq {
                        return (agent_color, t);
                    }
                }
            }
            AgentSlice::Positions { all, self_index } => {
                for (j, (other_pos, alive)) in all.iter().enumerate() {
                    if j == *self_index || !alive {
                        continue;
                    }
                    let diff = p - *other_pos;
                    if diff.length_squared() < agent_radius_sq {
                        return (agent_color, t);
                    }
                }
            }
        }

        // Check terrain hit
        let gh = world.terrain.height_at(p.x, p.z);
        if p.y <= gh {
            let c = match world.biome_map.biome_at(p.x, p.z) {
                BiomeType::FoodRich => [0.15, 0.50, 0.10, 1.0],
                BiomeType::Barren => [0.50, 0.40, 0.20, 1.0],
                BiomeType::Danger => [0.60, 0.20, 0.10, 1.0],
            };
            return (c, t);
        }
        t += step;
    }
    (sky, max_dist)
}
```

- [ ] **Step 2: Update `sample_vision` to call unified function**

In `sample_vision` (line 99), change:

```rust
            let (color, depth) = march_ray(pos, ray, world, max_dist, step, others);
```

to:

```rust
            let (color, depth) = march_ray_unified(pos, ray, world, max_dist, step, AgentSlice::Others(others));
```

- [ ] **Step 3: Update `sample_vision_positions` to call unified function**

In `sample_vision_positions` (line 303), change:

```rust
            let (color, depth) = march_ray_positions(pos, ray, world, max_dist, step, all_positions, self_index);
```

to:

```rust
            let (color, depth) = march_ray_unified(pos, ray, world, max_dist, step, AgentSlice::Positions { all: all_positions, self_index });
```

- [ ] **Step 4: Remove old `march_ray_positions`**

Delete the old `march_ray_positions` function (the block from `/// Ray marching using shared positions slice.` through the closing brace at line 363). This function is now replaced by `march_ray_unified`.

- [ ] **Step 5: Build and verify compilation**

Run: `cargo build -p xagent-sandbox 2>&1 | head -20`
Expected: Compiles without errors (warnings about unused code are OK).

- [ ] **Step 6: Run existing tests to verify no regressions**

Run: `cargo test --workspace 2>&1 | tail -20`
Expected: All existing tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/xagent-sandbox/src/agent/senses.rs
git commit -m "feat: unify ray-marching into single function with food visibility

Merge march_ray and march_ray_positions into march_ray_unified with
AgentSlice enum dispatch. Both code paths now check food items (lime
green [0.70, 0.95, 0.20]). Previously, march_ray_positions — used by
both main.rs and headless.rs — skipped food entirely, making agents
completely blind to food during evolution runs."
```

---

### Task 2: Lock `representation_dim`

**Files:**
- Modify: `crates/xagent-sandbox/src/agent/mod.rs:264-283` (mutate_config_with_strength)
- Modify: `crates/xagent-sandbox/src/agent/mod.rs:312-330` (crossover_config)

- [ ] **Step 1: Lock `representation_dim` in `mutate_config_with_strength`**

In `crates/xagent-sandbox/src/agent/mod.rs`, line 275, change:

```rust
        representation_dim: momentum.biased_perturb_u(&mut rng, parent.representation_dim, "representation_dim", strength).min(MAX_REPR_DIM),
```

to:

```rust
        representation_dim: parent.representation_dim,
```

- [ ] **Step 2: Lock `representation_dim` in `crossover_config`**

In `crates/xagent-sandbox/src/agent/mod.rs`, lines 326–330, change:

```rust
        representation_dim: (if rng.random::<f32>() < 0.5 {
            a.representation_dim
        } else {
            b.representation_dim
        }).min(MAX_REPR_DIM),
```

to:

```rust
        representation_dim: a.representation_dim,
```

- [ ] **Step 3: Build and run tests**

Run: `cargo test --workspace 2>&1 | tail -20`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/xagent-sandbox/src/agent/mod.rs
git commit -m "fix: lock representation_dim to prevent weight inheritance breakage

Remove representation_dim from mutation and crossover. When this value
changed, all three weight imports (encoder, action selector, predictor)
were silently skipped due to dimension mismatch, forcing the child to
start from random initialization. The field remains in BrainConfig for
population-level configuration but no longer evolves."
```

---

### Task 3: Wire Touch Contacts Into Encoder

**Files:**
- Modify: `crates/xagent-brain/src/encoder.rs`

- [ ] **Step 1: Update `NON_VISUAL_FEATURES` constant**

In `crates/xagent-brain/src/encoder.rs`, line 9, change:

```rust
const NON_VISUAL_FEATURES: usize = 9;
```

to:

```rust
/// Number of non-visual features: velocity_mag, facing(x,y,z), angular_vel,
/// energy, integrity, energy_delta, integrity_delta (9 proprioceptive/interoceptive)
/// + 4 touch contacts × 4 features each (direction.x, direction.z, intensity,
/// normalized surface_tag) = 25 total.
const NON_VISUAL_FEATURES: usize = 25;

/// Maximum number of touch contacts encoded into the feature vector.
/// The strongest contacts by intensity are selected; remaining slots are zero-padded.
const MAX_TOUCH_CONTACTS: usize = 4;

/// Features per touch contact: direction.x, direction.z, intensity, surface_tag/4.0.
const TOUCH_FEATURES_PER_CONTACT: usize = 4;
```

Also remove the old doc comment on line 7–8 (the one that says `/// Number of non-visual features: velocity_mag, facing(x,y,z), angular_vel,` and `/// energy, integrity, energy_delta, integrity_delta.`) since the new constant definition includes its own doc comment.

- [ ] **Step 2: Add touch encoding to `extract_features_into`**

In `crates/xagent-brain/src/encoder.rs`, after line 277 (`self.feature_scratch[cursor + 8] = frame.integrity_delta;`), add the touch contact encoding block:

```rust

        // Touch contacts: top MAX_TOUCH_CONTACTS by intensity, 4 features each.
        // The brain receives direction, proximity, and an opaque surface tag —
        // it must learn what each tag value correlates with through experience.
        let touch_start = cursor + 9;
        let mut contact_count = frame.touch_contacts.len().min(MAX_TOUCH_CONTACTS);

        if contact_count > 0 {
            // Find the top contacts by intensity without allocating.
            // We use a simple selection: for each slot, find the strongest
            // unused contact. With MAX_TOUCH_CONTACTS=4, this is 4×N comparisons
            // which is fine for typical contact counts (<10).
            let mut used = [false; 64]; // more than enough for any realistic contact count
            for slot in 0..contact_count {
                let mut best_idx = usize::MAX;
                let mut best_intensity = -1.0_f32;
                for (ci, contact) in frame.touch_contacts.iter().enumerate() {
                    if ci < used.len() && !used[ci] && contact.intensity > best_intensity {
                        best_intensity = contact.intensity;
                        best_idx = ci;
                    }
                }
                if best_idx == usize::MAX {
                    // Fewer usable contacts than expected
                    contact_count = slot;
                    break;
                }
                used[best_idx] = true;
                let c = &frame.touch_contacts[best_idx];
                let base = touch_start + slot * TOUCH_FEATURES_PER_CONTACT;
                self.feature_scratch[base] = c.direction.x;
                self.feature_scratch[base + 1] = c.direction.z;
                self.feature_scratch[base + 2] = c.intensity;
                self.feature_scratch[base + 3] = c.surface_tag as f32 / 4.0;
            }
        }

        // Zero-pad remaining touch slots
        for slot in contact_count..MAX_TOUCH_CONTACTS {
            let base = touch_start + slot * TOUCH_FEATURES_PER_CONTACT;
            self.feature_scratch[base] = 0.0;
            self.feature_scratch[base + 1] = 0.0;
            self.feature_scratch[base + 2] = 0.0;
            self.feature_scratch[base + 3] = 0.0;
        }
```

- [ ] **Step 3: Build and run tests**

Run: `cargo test --workspace 2>&1 | tail -30`
Expected: Some existing encoder tests may fail because `feature_count` changed from 201 to 217. We'll fix those in the next step.

- [ ] **Step 4: Update existing encoder test `feature_count_includes_depth_and_interoception`**

In `crates/xagent-brain/src/encoder.rs`, in the test `feature_count_includes_depth_and_interoception` (around line 434), change:

```rust
        // 4×3 pixels × 4 channels (RGBD) + 9 interoceptive = 57
        assert_eq!(feats.len(), 4 * 3 * 4 + 9);
```

to:

```rust
        // 4×3 pixels × 4 channels (RGBD) + 25 non-visual (9 proprioceptive/interoceptive + 16 touch) = 73
        assert_eq!(feats.len(), 4 * 3 * 4 + NON_VISUAL_FEATURES);
```

- [ ] **Step 5: Run all tests to verify**

Run: `cargo test --workspace 2>&1 | tail -20`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/xagent-brain/src/encoder.rs
git commit -m "feat: wire touch contacts into encoder feature vector

Encode top 4 touch contacts by intensity into 16 additional input
features (direction.x, direction.z, intensity, normalized surface_tag).
NON_VISUAL_FEATURES increases from 9 to 25. This gives agents tactile
perception — food proximity, wall proximity, hazard zones, and other
agents — which was previously computed but never fed to the brain."
```

---

### Task 4: Add Tests

**Files:**
- Modify: `crates/xagent-sandbox/tests/integration.rs`
- Modify: `crates/xagent-brain/src/encoder.rs` (test module)

- [ ] **Step 1: Add food visibility integration test**

In `crates/xagent-sandbox/tests/integration.rs`, add after the last test (line 369):

```rust

#[test]
fn vision_detects_food_items() {
    let world = test_world();

    // Find an unconsumed food item
    let food = match world.food_items.iter().find(|f| !f.consumed) {
        Some(f) => f,
        None => return, // no food in default world (unlikely)
    };

    // Place agent looking directly at the food, close enough to see it
    let to_food = (food.position - Vec3::new(food.position.x - 10.0, food.position.y, food.position.z)).normalize();
    let agent_pos = food.position - to_food * 10.0;
    let spawn_y = world.terrain.height_at(agent_pos.x, agent_pos.z) + 2.0;
    let mut agent = agent_at(Vec3::new(agent_pos.x, spawn_y, agent_pos.z));
    // Face toward the food
    agent.body.facing = to_food;

    let mut frame = xagent_shared::SensoryFrame::new_blank(8, 6);
    xagent_sandbox::agent::senses::extract_senses(&agent, &world, 0, &mut frame);

    // Check if any pixel in the vision field has the food color (lime green: R≈0.70, G≈0.95)
    let food_green_threshold_g = 0.90;
    let food_green_threshold_r = 0.60;
    let mut found_food_pixel = false;
    let pixels = (frame.vision.width * frame.vision.height) as usize;
    for px in 0..pixels {
        let base = px * 4;
        let r = frame.vision.color[base];
        let g = frame.vision.color[base + 1];
        if r > food_green_threshold_r && g > food_green_threshold_g {
            found_food_pixel = true;
            break;
        }
    }

    // We can't guarantee the food is in the FOV (depends on world layout),
    // so this test checks the mechanism works rather than guaranteeing a hit.
    // Place agent very close and facing directly at food for a reliable check.
    let close_pos = food.position - Vec3::new(3.0, 0.0, 0.0);
    let close_y = world.terrain.height_at(close_pos.x, close_pos.z) + 2.0;
    let mut close_agent = agent_at(Vec3::new(close_pos.x, close_y, close_pos.z));
    close_agent.body.facing = Vec3::X; // face toward food (food is +X from agent)

    let mut close_frame = xagent_shared::SensoryFrame::new_blank(8, 6);
    xagent_sandbox::agent::senses::extract_senses(&close_agent, &world, 0, &mut close_frame);

    let mut close_found = false;
    for px in 0..pixels {
        let base = px * 4;
        let r = close_frame.vision.color[base];
        let g = close_frame.vision.color[base + 1];
        if r > food_green_threshold_r && g > food_green_threshold_g {
            close_found = true;
            break;
        }
    }

    assert!(
        found_food_pixel || close_found,
        "At least one vision approach should detect food as lime-green pixels"
    );
}

#[test]
fn vision_with_positions_detects_food_items() {
    let world = test_world();

    // Find an unconsumed food item
    let food = match world.food_items.iter().find(|f| !f.consumed) {
        Some(f) => f,
        None => return,
    };

    // Place agent close to food, facing it
    let close_pos = food.position - Vec3::new(3.0, 0.0, 0.0);
    let close_y = world.terrain.height_at(close_pos.x, close_pos.z) + 2.0;
    let mut agent = agent_at(Vec3::new(close_pos.x, close_y, close_pos.z));
    agent.body.facing = Vec3::X;

    // Use the positions-based extraction (the path used during evolution)
    let all_positions: Vec<(Vec3, bool)> = vec![(agent.body.position, true)];
    let mut frame = xagent_shared::SensoryFrame::new_blank(8, 6);
    xagent_sandbox::agent::senses::extract_senses_with_positions(
        &agent, &world, 0, &all_positions, 0, &mut frame,
    );

    let food_green_threshold_g = 0.90;
    let food_green_threshold_r = 0.60;
    let pixels = (frame.vision.width * frame.vision.height) as usize;
    let mut found_food = false;
    for px in 0..pixels {
        let base = px * 4;
        let r = frame.vision.color[base];
        let g = frame.vision.color[base + 1];
        if r > food_green_threshold_r && g > food_green_threshold_g {
            found_food = true;
            break;
        }
    }

    assert!(
        found_food,
        "extract_senses_with_positions should detect food as lime-green pixels (the critical evolution bug)"
    );
}

#[test]
fn touch_contacts_populated_near_food() {
    let world = test_world();

    // Find an unconsumed food item
    let food = match world.food_items.iter().find(|f| !f.consumed) {
        Some(f) => f,
        None => return,
    };

    // Place agent within touch range of food (< 3.0 units)
    let agent_pos = Vec3::new(food.position.x + 1.0, food.position.y, food.position.z);
    let spawn_y = world.terrain.height_at(agent_pos.x, agent_pos.z) + 2.0;
    let agent = agent_at(Vec3::new(agent_pos.x, spawn_y, agent_pos.z));

    let mut frame = xagent_shared::SensoryFrame::new_blank(8, 6);
    xagent_sandbox::agent::senses::extract_senses(&agent, &world, 0, &mut frame);

    let has_food_touch = frame.touch_contacts.iter().any(|c| c.surface_tag == 1 && c.intensity > 0.0);
    assert!(
        has_food_touch,
        "Agent within 3 units of food should have a TOUCH_FOOD contact (tag=1)"
    );
}
```

- [ ] **Step 2: Add encoder touch feature test**

In `crates/xagent-brain/src/encoder.rs`, add to the test module after the last test (before the closing `}`):

```rust

    #[test]
    fn touch_contacts_appear_in_features() {
        use xagent_shared::TouchContact;

        let mut enc = SensoryEncoder::new(8, 4);
        let mut frame = make_frame(0.5, 0.5);

        // Add touch contacts
        frame.touch_contacts = vec![
            TouchContact {
                direction: Vec3::new(1.0, 0.0, 0.0),
                intensity: 0.8,
                surface_tag: 1,
            },
            TouchContact {
                direction: Vec3::new(0.0, 0.0, -1.0),
                intensity: 0.3,
                surface_tag: 2,
            },
        ];

        let feats = enc.extract_features(&frame);

        // Non-visual features start after visual pixels
        // 4×3 pixels × 4 channels = 48, then 9 proprioceptive/interoceptive, then touch
        let touch_start = 4 * 3 * 4 + 9;

        // First touch slot should be the strongest contact (intensity 0.8)
        assert!((feats[touch_start] - 1.0).abs() < 1e-6, "direction.x of strongest contact");
        assert!((feats[touch_start + 1] - 0.0).abs() < 1e-6, "direction.z of strongest contact");
        assert!((feats[touch_start + 2] - 0.8).abs() < 1e-6, "intensity of strongest contact");
        assert!((feats[touch_start + 3] - 0.25).abs() < 1e-6, "surface_tag/4.0 of strongest contact (tag 1)");

        // Second touch slot should be the weaker contact (intensity 0.3)
        let slot2 = touch_start + 4;
        assert!((feats[slot2] - 0.0).abs() < 1e-6, "direction.x of second contact");
        assert!((feats[slot2 + 1] - (-1.0)).abs() < 1e-6, "direction.z of second contact");
        assert!((feats[slot2 + 2] - 0.3).abs() < 1e-6, "intensity of second contact");
        assert!((feats[slot2 + 3] - 0.5).abs() < 1e-6, "surface_tag/4.0 of second contact (tag 2)");

        // Third and fourth slots should be zero-padded
        let slot3 = touch_start + 8;
        for i in 0..4 {
            assert!((feats[slot3 + i]).abs() < 1e-6, "slot 3 feature {} should be zero-padded", i);
        }
        let slot4 = touch_start + 12;
        for i in 0..4 {
            assert!((feats[slot4 + i]).abs() < 1e-6, "slot 4 feature {} should be zero-padded", i);
        }
    }

    #[test]
    fn touch_contacts_sorted_by_intensity() {
        use xagent_shared::TouchContact;

        let mut enc = SensoryEncoder::new(8, 4);
        let mut frame = make_frame(0.5, 0.5);

        // Add contacts in non-sorted order (weakest first)
        frame.touch_contacts = vec![
            TouchContact {
                direction: Vec3::new(0.0, 0.0, 1.0),
                intensity: 0.1,
                surface_tag: 3,
            },
            TouchContact {
                direction: Vec3::new(1.0, 0.0, 0.0),
                intensity: 0.9,
                surface_tag: 1,
            },
            TouchContact {
                direction: Vec3::new(-1.0, 0.0, 0.0),
                intensity: 0.5,
                surface_tag: 4,
            },
        ];

        let feats = enc.extract_features(&frame);
        let touch_start = 4 * 3 * 4 + 9;

        // Slot 0: strongest (0.9, tag 1)
        assert!((feats[touch_start + 2] - 0.9).abs() < 1e-6, "slot 0 should have intensity 0.9");
        assert!((feats[touch_start + 3] - 0.25).abs() < 1e-6, "slot 0 should have tag 1 (0.25)");

        // Slot 1: second strongest (0.5, tag 4)
        let slot1 = touch_start + 4;
        assert!((feats[slot1 + 2] - 0.5).abs() < 1e-6, "slot 1 should have intensity 0.5");
        assert!((feats[slot1 + 3] - 1.0).abs() < 1e-6, "slot 1 should have tag 4 (1.0)");

        // Slot 2: weakest (0.1, tag 3)
        let slot2 = touch_start + 8;
        assert!((feats[slot2 + 2] - 0.1).abs() < 1e-6, "slot 2 should have intensity 0.1");
        assert!((feats[slot2 + 3] - 0.75).abs() < 1e-6, "slot 2 should have tag 3 (0.75)");
    }

    #[test]
    fn no_touch_contacts_produces_zero_features() {
        let mut enc = SensoryEncoder::new(8, 4);
        let frame = make_frame(0.5, 0.5); // no touch contacts

        let feats = enc.extract_features(&frame);
        let touch_start = 4 * 3 * 4 + 9;

        // All 16 touch features should be zero
        for i in 0..16 {
            assert!(
                feats[touch_start + i].abs() < 1e-6,
                "touch feature {} should be zero with no contacts, got {}",
                i,
                feats[touch_start + i]
            );
        }
    }
```

- [ ] **Step 3: Run all tests**

Run: `cargo test --workspace 2>&1 | tail -30`
Expected: All tests pass (existing + new).

- [ ] **Step 4: Commit**

```bash
git add crates/xagent-sandbox/tests/integration.rs crates/xagent-brain/src/encoder.rs
git commit -m "test: add tests for food visibility, touch encoding, and contact sorting

Integration tests verify both extract_senses and extract_senses_with_positions
detect food as lime-green pixels, and that touch contacts are populated near food.
Encoder unit tests verify touch contact encoding: correct feature positions,
intensity-based sorting, zero-padding, and surface_tag normalization."
```

---

### Task 5: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `EVOLUTION_JOURNEY.md`
- Modify: `crates/xagent-sandbox/README.md`

- [ ] **Step 1: Update `README.md` — sensory apparatus description**

In `README.md`, line 145, change:

```
- Sensory apparatus: 8×6 raycast vision (ray step 1.0), touch contacts (food, terrain edges, hazards, other agents), proprioception, interoception
```

to:

```
- Sensory apparatus: 8×6 raycast vision (ray step 1.0, detects terrain, food, and other agents), touch contacts (food, terrain edges, hazards, other agents — top 4 encoded into brain), proprioception, interoception
```

- [ ] **Step 2: Update `README.md` — agent vision description**

In `README.md`, line 146, change:

```
- **Agent vision**: ray marching detects terrain, food, and other agents (rendered as magenta `[0.9, 0.2, 0.6, 1.0]` in the visual field)
```

to:

```
- **Agent vision**: ray marching detects terrain (biome-colored), food items (lime green `[0.70, 0.95, 0.20, 1.0]`), and other agents (magenta `[0.9, 0.2, 0.6, 1.0]`) in the visual field
```

- [ ] **Step 3: Update `README.md` — representation_dim parameter description**

In `README.md`, line 270, change:

```
| `representation_dim` | Internal representation vector length. Smaller → more compression, more abstraction. |
```

to:

```
| `representation_dim` | Internal representation vector length. Fixed across generations (not evolved) to preserve weight inheritance. Smaller → more compression, more abstraction. |
```

- [ ] **Step 4: Update `README.md` — known limitations table**

In `README.md`, the "Known Limitations" is in section 12 "Future Directions". Actually, check the sandbox README for the known limitations table instead — that's where it lives. The main README doesn't have a "Simplified vision" limitation entry, so skip this sub-step for the main README.

- [ ] **Step 5: Update `EVOLUTION_JOURNEY.md` — refine Issue #5 entry**

In `EVOLUTION_JOURNEY.md`, lines 53–60 (the existing "Agent Couldn't See Food" entry), change:

```markdown
### 5. Agent Couldn't See Food

**Symptom:** 600+ generations, no food-seeking behavior.

**Root cause:** Ray marching checked terrain biome colors and other agents, but never food items. Inside a green biome, every direction looked identical. Zero directional signal.

**Fix:** Added food item detection to ray marching via spatial grid lookup.

**Lesson:** Before optimizing a learning algorithm, verify the agent has access to the information it needs to learn from.
```

to:

```markdown
### 5. Agent Couldn't See Food (The Duplication Bug)

**Symptom:** 600+ generations, no food-seeking behavior. Agents acted completely randomly after 10+ hours of evolution. Flat fitness graph.

**Root cause:** Two separate ray-march functions existed: `march_ray` (checked food, agents, terrain) and `march_ray_positions` (checked agents and terrain only — no food). Both `main.rs` and `headless.rs` used `extract_senses_with_positions`, which called the broken function. Agents were completely blind to food during all actual simulation runs. Credit assignment reinforced noise because there was no distinguishable visual state before eating food.

**Fix:** Unified both functions into `march_ray_unified` with an `AgentSlice` enum that dispatches agent-hit checks. Food detection (via spatial grid lookup, lime green `[0.70, 0.95, 0.20]`) is now in the single code path. Added integration tests that verify both extraction paths see food.

**Lesson:** Code duplication is a bug factory. When two functions do the same thing, they will eventually diverge. The fix isn't adding the missing code to both — it's eliminating the duplication. Also: before optimizing a learning algorithm, verify the agent has access to the information it needs to learn from.
```

- [ ] **Step 6: Add Issue #15 to `EVOLUTION_JOURNEY.md`**

After the entry for issue #14 ("Mutations Were Random Walks", ending around line 147), add:

```markdown

### 15. `representation_dim` Mutation Destroyed Weight Inheritance

**Symptom:** Offspring frequently started from random initialization despite weight inheritance code being present.

**Root cause:** `representation_dim` was evolvable via both mutation and crossover. When a child's `representation_dim` differed from the parent's, all three weight imports — encoder (`repr_dim × feature_count`), action selector (`repr_dim × 2 + 2`), and predictor (`repr_dim × repr_dim`) — were silently skipped because the dimension check in `import_weights` failed. The child started from a fresh random brain every time this happened.

**Fix:** Locked `representation_dim` — removed it from both `mutate_config_with_strength` and `crossover_config`. The field remains in `BrainConfig` for population-level configuration but never changes across generations.

**Lesson:** Silent failures are the worst kind. The `import_weights` guard that skips mismatched dimensions is correct safety behavior, but the upstream code that freely mutated the dimension was silently defeating the entire inheritance mechanism. When a safety guard fires, it should log — or better, the condition should be prevented upstream.

### 16. Touch Contacts Computed But Never Encoded

**Symptom:** Agents had no tactile sense despite touch infrastructure being fully wired on the sensing side.

**Root cause:** `detect_touch()` and `detect_touch_positions()` populated `frame.touch_contacts` every tick (food proximity, terrain edges, hazard zones, other agents). But `extract_features_into()` in the encoder never read `touch_contacts` — the feature vector contained only visual and proprioceptive/interoceptive features (9 non-visual features). Touch data was computed and discarded.

**Fix:** Encoder now encodes the top 4 touch contacts (by intensity) into 16 additional features: direction.x, direction.z, intensity, and normalized surface_tag for each. `NON_VISUAL_FEATURES` increased from 9 to 25.

**Lesson:** An untested data path is a broken data path. The sensing side was correct; the encoding side never consumed it. Integration tests that verify end-to-end data flow (sensor → encoder → feature vector) catch this class of bug.
```

- [ ] **Step 7: Update `crates/xagent-sandbox/README.md` — vision hit color table**

In `crates/xagent-sandbox/README.md`, around lines 420–429 (the vision hit color table), change:

```markdown
**Hit color** is determined by what the ray strikes:

| Target | RGBA |
|---|---|
| FoodRich terrain | `[0.15, 0.50, 0.10, 1.0]` |
| Barren terrain | `[0.50, 0.40, 0.20, 1.0]` |
| Danger terrain | `[0.60, 0.20, 0.10, 1.0]` |
| Other agent | `[0.90, 0.20, 0.60, 1.0]` (magenta) |
| Sky (miss) | `[0.53, 0.81, 0.92, 1.0]` (early-out) |
```

to:

```markdown
**Hit color** is determined by what the ray strikes first (checked in this order):

| Target | RGBA | Priority |
|---|---|---|
| Food item | `[0.70, 0.95, 0.20, 1.0]` (lime green) | 1st — spatial grid lookup |
| Other agent | `[0.90, 0.20, 0.60, 1.0]` (magenta) | 2nd |
| FoodRich terrain | `[0.15, 0.50, 0.10, 1.0]` | 3rd |
| Barren terrain | `[0.50, 0.40, 0.20, 1.0]` | 3rd |
| Danger terrain | `[0.60, 0.20, 0.10, 1.0]` | 3rd |
| Sky (miss) | `[0.53, 0.81, 0.92, 1.0]` (early-out) | fallback |

A single unified `march_ray_unified` function handles all targets via an `AgentSlice` enum dispatch, ensuring both the `extract_senses` and `extract_senses_with_positions` paths see food identically.
```

- [ ] **Step 8: Update `crates/xagent-sandbox/README.md` — mutation docs**

In `crates/xagent-sandbox/README.md`, around line 920, find the mutation paragraph and update the relevant sentence. Change:

```
`visual_encoding_size` is preserved (must match the sensory pipeline).
```

to:

```
`visual_encoding_size` and `representation_dim` are preserved (visual_encoding_size must match the sensory pipeline; representation_dim is locked to prevent weight inheritance breakage across generations).
```

- [ ] **Step 9: Update `crates/xagent-sandbox/README.md` — known limitations**

In `crates/xagent-sandbox/README.md`, around line 1085, change:

```
| Simplified vision | Agents see biome-colored raycasts, not the actual rendered scene. No agent/food visibility in visual field. |
```

to:

```
| Simplified vision | Agents see biome-colored raycasts, not the actual rendered scene. Food items and other agents are visible as distinct colors (lime green and magenta respectively). |
```

- [ ] **Step 10: Update `crates/xagent-sandbox/README.md` — "What the Brain Sees" section**

In `crates/xagent-sandbox/README.md`, around lines 862–871, change:

```markdown
### What the Brain "Sees" vs What the Renderer Shows

The brain receives an **8 × 6 biome-colored depth image** from simplified raycasting.
The renderer shows the actual **textured terrain mesh** with height-blended vertex
colors. These are related but not identical:

- The brain's vision uses flat biome colors (3 terrain options + agent magenta + sky).
- The renderer shows smooth height-based color gradients per biome.
- The brain **can see other agents** in its visual field (rendered as magenta
  `[0.9, 0.2, 0.6, 1.0]`), as well as detecting them via touch contacts.
```

to:

```markdown
### What the Brain "Sees" vs What the Renderer Shows

The brain receives an **8 × 6 colored depth image** from simplified raycasting.
The renderer shows the actual **textured terrain mesh** with height-blended vertex
colors. These are related but not identical:

- The brain's vision uses flat biome colors (3 terrain options + food lime-green + agent magenta + sky).
- The renderer shows smooth height-based color gradients per biome with 3D food/agent cubes.
- The brain **can see food items** (lime green `[0.70, 0.95, 0.20, 1.0]`) and **other agents** (magenta `[0.9, 0.2, 0.6, 1.0]`) in its visual field.
- The brain also receives **touch contacts** — the top 4 contacts by intensity are encoded into 16 features (direction, intensity, surface tag).
```

- [ ] **Step 11: Update `crates/xagent-sandbox/README.md` — sensory pipeline diagram**

In `crates/xagent-sandbox/README.md`, around lines 830–860, the "How Raw World State Becomes a SensoryFrame" diagram has the `sample_vision()` box. Update the comment text inside the box. Change:

```
│  ├─ terrain (heights)  │────►│ sample_vision()       │
│  ├─ biome_map          │     │  8×6 grid, 90° FOV    │──► vision.color [192 floats]
│  └─ food_items         │     │  50.0 max depth       │──► vision.depth [48 floats]
│                        │     │  1.0 ray step         │
│                        │     │  detects agents too   │
│                        │     └───────────────────────┘
```

to:

```
│  ├─ terrain (heights)  │────►│ march_ray_unified()   │
│  ├─ biome_map          │     │  8×6 grid, 90° FOV    │──► vision.color [192 floats]
│  └─ food_items         │     │  50.0 max depth       │──► vision.depth [48 floats]
│                        │     │  1.0 ray step         │
│                        │     │  food + agents + terrain│
│                        │     └───────────────────────┘
```

- [ ] **Step 12: Run final test pass**

Run: `cargo test --workspace 2>&1 | tail -20`
Expected: All tests pass.

- [ ] **Step 13: Commit**

```bash
git add README.md EVOLUTION_JOURNEY.md crates/xagent-sandbox/README.md
git commit -m "docs: update READMEs and EVOLUTION_JOURNEY for evolution fixes

Update vision hit tables to include food (lime green). Add EVOLUTION_JOURNEY
entries for the duplication bug (#5 refined), representation_dim inheritance
breakage (#15), and touch encoding gap (#16). Update mutation docs to note
representation_dim is locked. Fix known limitations to reflect food/agent
visibility in vision."
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Unify `march_ray` + `march_ray_positions` | Task 1 |
| `AgentSlice` enum | Task 1, Step 1 |
| Food detection with lime green color | Task 1, Step 1 |
| `sample_vision` calls unified function | Task 1, Step 2 |
| `sample_vision_positions` calls unified function | Task 1, Step 3 |
| Remove old `march_ray_positions` | Task 1, Step 4 |
| Lock `representation_dim` in `mutate_config_with_strength` | Task 2, Step 1 |
| Lock `representation_dim` in `crossover_config` | Task 2, Step 2 |
| `NON_VISUAL_FEATURES` 9 → 25 | Task 3, Step 1 |
| Touch encoding in `extract_features_into` | Task 3, Step 2 |
| Top 4 by intensity, zero-padded | Task 3, Step 2 |
| Tests for food visibility | Task 4, Step 1 |
| Tests for touch encoding | Task 4, Step 2 |
| Update README.md | Task 5, Steps 1–4 |
| Update EVOLUTION_JOURNEY.md | Task 5, Steps 5–6 |
| Update sandbox README.md | Task 5, Steps 7–11 |

**Placeholder scan:** No TBDs, TODOs, or "add appropriate" patterns found.

**Type consistency:** `AgentSlice` enum is used consistently in Task 1. `NON_VISUAL_FEATURES`, `MAX_TOUCH_CONTACTS`, `TOUCH_FEATURES_PER_CONTACT` are defined in Task 3 and referenced in Task 4 tests. `march_ray_unified` name is consistent across Task 1 code and Task 5 docs.
