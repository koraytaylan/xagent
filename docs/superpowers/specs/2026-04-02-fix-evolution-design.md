# Fix Stagnant Evolution — Design Spec

## Problem

After 10 hours of evolution, agents behave completely randomly with flat fitness graphs. Investigation revealed three root causes:

1. **Agents are blind to food.** `march_ray_positions` — the ray-marching function used by both `main.rs` and `headless.rs` — does not check food items. Rays pass straight through food. The separate `march_ray` function checks food correctly but is only used by the unused `extract_senses_with_others` path. Credit assignment reinforces noise because there is no distinguishable visual state before eating.

2. **`representation_dim` mutation silently destroys weight inheritance.** When a child mutates `representation_dim`, all three weight imports (encoder, action selector, predictor) are silently skipped due to dimension mismatch. The child starts from random initialization every generation this happens.

3. **Touch contacts computed but never encoded.** `detect_touch` and `detect_touch_positions` populate `frame.touch_contacts` every tick, but `extract_features_into` in the encoder never reads them. Agents have no tactile sense despite the infrastructure being fully wired on the sensing side.

## Design

### 1. Unified Ray-Marching

Merge `march_ray` (senses.rs:122) and `march_ray_positions` (senses.rs:317) into a single function. The two functions duplicate the entire ray-march loop; the only differences are food checking (present in `march_ray`, missing in `march_ray_positions`) and the agent-hit dispatch (one uses `&[OtherAgent]`, the other uses `&[(Vec3, bool)]` with a self-index).

**Unified function:**

```rust
enum AgentSlice<'a> {
    Others(&'a [OtherAgent]),
    Positions { all: &'a [(Vec3, bool)], self_index: usize },
}

fn march_ray_unified(
    origin: Vec3,
    dir: Vec3,
    world: &WorldState,
    max_dist: f32,
    step: f32,
    agents: AgentSlice,
) -> ([f32; 4], f32)
```

Single loop body, in order:
1. Food check via `world.food_grid.query_nearby` — returns `[0.70, 0.95, 0.20, 1.0]` (lime green)
2. Agent check — dispatches on `AgentSlice` variant, returns `[0.9, 0.2, 0.6, 1.0]` (hot pink)
3. Terrain check via heightmap + biome color

`sample_vision` and `sample_vision_positions` both call `march_ray_unified`. One source of truth, can never drift apart again.

Food renders as a visually distinct lime green stimulus. The brain receives raw RGBD pixels — no semantic "food" label. The agent must learn through evolution that this color correlates with energy gain.

### 2. Lock `representation_dim`

Remove `representation_dim` from evolutionary variation:

- **`mutate_config_with_strength`**: Replace `biased_perturb_u(... "representation_dim" ...)` with `representation_dim: parent.representation_dim`
- **`crossover_config`**: Replace the coin-flip with `representation_dim: a.representation_dim`

The field remains in `BrainConfig` for population-level configuration (default 32, tiny 16, large 64). It is set once at population init and never changes across generations.

This guarantees encoder weights (`repr_dim × feature_count`), action selector weights (`repr_dim × 2 + 2`), and predictor weights (`repr_dim × repr_dim`) are always dimension-compatible during `import_weights`, so inherited weights are never silently dropped.

### 3. Touch Contact Encoding

Wire `frame.touch_contacts` into the encoder's feature vector in `extract_features_into`.

**Encoding:** Top 4 contacts by intensity, each contributing 4 features:
- `direction.x` — horizontal bearing component
- `direction.z` — horizontal bearing component
- `intensity` — proximity strength, 0.0–1.0
- `surface_tag / 4.0` — normalized opaque tag (values 0.25, 0.50, 0.75, 1.0 for tags 1–4)

Zero-padded when fewer than 4 contacts. Fixed 16 features regardless of actual contact count.

**Changes:**
- `NON_VISUAL_FEATURES`: 9 → 25
- `extract_features_into`: after interoceptive features (cursor+8), sort contacts by intensity descending, write top 4 × 4 features, zero-pad remainder
- `feature_count` adjusts automatically via `lazy_init` (201 → 217 for 8×6 vision)

**Breaking change:** Input dimension change invalidates all existing encoder weight matrices. First generation after this change initializes fresh. Acceptable because the food visibility fix fundamentally changes perception — prior weights encoded a world without visible food.

## What This Does NOT Change

- **Motor fatigue** — intentional anti-loop mechanism, stays as designed
- **Credit assignment** — the architecture is sound; it just had no signal to work with because food was invisible
- **Mutation rates / evolutionary parameters** — wait for results with working perception before tuning
- **`march_ray` callers** — `sample_vision` and `sample_vision_positions` are the only callers; both switch to the unified function

## Expected Outcome

With food visible as a distinct color in the vision strip, credit assignment can finally correlate visual state (lime green pixels) with energy gain. Over generations, natural selection should favor agents whose encoder weights amplify the food-color features, and whose action policies turn toward those features. Touch encoding provides additional gradient signal (tactile proximity to food, walls, agents). Locked `representation_dim` ensures weight inheritance works reliably across all generations.
