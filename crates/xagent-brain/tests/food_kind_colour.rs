//! A food item's vision colour depends on its kind only. With kinds on a
//! `Second` item renders the second-kind colour and every other item the base
//! colour; with kinds off every item renders the base colour. The colour never
//! reveals which kind is nourishing.

use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, FoodKind, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
const FULL_METER: f32 = 100.0;
/// Ticks run so the vision pass has filled the sensory buffer.
const TICKS: u64 = 2;
/// Four items put each kind at two indices.
const FOOD: usize = 4;
/// Number of food kinds; each appears at FOOD / KIND_COUNT indices.
const KIND_COUNT: usize = 2;
const WORLD_SEED: u64 = 5;
const RESET_SEED: u64 = 11;
/// Resets tried while looking for a changed nourishing kind.
const MAX_RESETS: u64 = 32;
const AGENT_POS: (f32, f32, f32) = (2.0, 1.0, 0.0);
/// Straight ahead of the agent, inside the view.
/// Outside the eat radius, on a vision sample (three steps of 1.2).
const IN_VIEW: (f32, f32, f32) = (2.0, 1.0, 3.6);
/// Behind the agent, outside the view.
const OUT_OF_VIEW: (f32, f32, f32) = (2.0, 1.0, -3.6);
const BASE_COLOR: [f32; 4] = [0.7, 0.95, 0.2, 1.0];
const SECOND_KIND_COLOR: [f32; 4] = [0.85, 0.3, 0.75, 1.0];

fn gpu_or_skip() -> bool {
    if GpuKernel::is_available() {
        return true;
    }
    assert!(
        std::env::var_os("XAGENT_REQUIRE_GPU").is_none(),
        "no GPU adapter available but XAGENT_REQUIRE_GPU is set"
    );
    eprintln!("Skipping: no GPU/fallback adapter available");
    false
}

fn brain_config() -> BrainConfig {
    BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        movement_speed: 0.0,
        ..BrainConfig::default()
    }
}

fn world_config(kinds: bool) -> WorldConfig {
    WorldConfig {
        seed: WORLD_SEED,
        food_kinds_enabled: kinds,
        ..WorldConfig::default()
    }
}

/// Upload the scene with the item at `visible` in view and the other behind
/// the agent, then return the rendered vision pixels.
fn render(kernel: &mut GpuKernel, brain: &BrainConfig, visible: usize) -> Vec<f32> {
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    let food: Vec<(f32, f32, f32)> = (0..FOOD)
        .map(|i| if i == visible { IN_VIEW } else { OUT_OF_VIEW })
        .collect();
    kernel.upload_world(&heights, &biomes, &food, &[false; FOOD], &[0.0; FOOD]);
    kernel.upload_agents(&[(
        glam::Vec3::new(AGENT_POS.0, AGENT_POS.1, AGENT_POS.2),
        FULL_METER,
        FULL_METER,
        brain.memory_capacity,
        brain.processing_slots,
    )]);
    for tick in 0..TICKS {
        kernel.dispatch_batch(tick, 1);
    }
    kernel.read_agent_telemetry_blocking(0).vision_color
}

fn count_pixels(view: &[f32], color: [f32; 4]) -> usize {
    view.as_chunks::<4>()
        .0
        .iter()
        .filter(|p| **p == color)
        .count()
}

fn kernel_for(kinds: bool, brain: &BrainConfig) -> GpuKernel {
    let mut kernel = GpuKernel::new(1, FOOD, brain, &world_config(kinds));
    kernel.reset_agents_seeded(brain, RESET_SEED);
    kernel
}

#[test]
fn food_kind_colour_default_off_is_baseline() {
    if !gpu_or_skip() {
        return;
    }
    let brain = brain_config();
    let mut kernel = kernel_for(false, &brain);
    for item in 0..FOOD {
        let view = render(&mut kernel, &brain, item);
        assert!(
            count_pixels(&view, BASE_COLOR) > 0,
            "item {item}: no baseline hit pixels"
        );
        assert_eq!(count_pixels(&view, SECOND_KIND_COLOR), 0, "item {item}");
    }
}

#[test]
fn food_kind_colour_by_kind() {
    if !gpu_or_skip() {
        return;
    }
    let brain = brain_config();
    let mut kernel = kernel_for(true, &brain);
    let mut seen = [0_usize; KIND_COUNT];
    for item in 0..FOOD {
        let kind = FoodKind::of_item(item);
        seen[kind.index() as usize] += 1;
        let (expected, other) = match kind {
            FoodKind::Second => (SECOND_KIND_COLOR, BASE_COLOR),
            FoodKind::First => (BASE_COLOR, SECOND_KIND_COLOR),
        };
        let view = render(&mut kernel, &brain, item);
        assert!(
            count_pixels(&view, expected) > 0,
            "item {item} ({kind:?}): no hit pixels"
        );
        assert_eq!(count_pixels(&view, other), 0, "item {item} ({kind:?})");
    }
    assert_eq!(
        seen,
        [FOOD / KIND_COUNT; KIND_COUNT],
        "each kind must appear at two indices"
    );
}

#[test]
fn food_kind_colour_ignores_assignment() {
    if !gpu_or_skip() {
        return;
    }
    let brain = brain_config();
    let mut kernel = kernel_for(true, &brain);
    let initial = kernel.nourishing_food_kind();
    let before: Vec<Vec<u32>> = (0..FOOD)
        .map(|i| {
            let view = render(&mut kernel, &brain, i);
            view.iter().map(|v| v.to_bits()).collect()
        })
        .collect();
    let mut changed = false;
    for i in 0..MAX_RESETS {
        kernel.reset_agents_seeded(&brain, RESET_SEED + i);
        if kernel.nourishing_food_kind() != initial {
            changed = true;
            break;
        }
    }
    assert!(changed, "the nourishing kind never changed");
    for (i, expected) in before.iter().enumerate() {
        let view = render(&mut kernel, &brain, i);
        let bits: Vec<u32> = view.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            &bits, expected,
            "item {i} pixels changed with the assignment"
        );
    }
}
