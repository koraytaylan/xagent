//! The kernel redraws its nourishing food kind at construction and on every
//! reset that runs, following the config draw sequence.

use xagent_brain::buffers::food_kinds_slot_value;
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, FoodKind, WorldConfig};

/// Number of resets after construction.
const N: u64 = 8;
const AGENTS: u32 = 4;
const FOOD: usize = 4;
const RESET_SEED: u64 = 11;

/// Pure decision: `Ok(true)` run, `Ok(false)` skip, `Err` fail closed.
fn gpu_guard_decision(available: bool, required: bool) -> Result<bool, &'static str> {
    if available {
        Ok(true)
    } else if required {
        Err("no GPU adapter available but XAGENT_REQUIRE_GPU is set")
    } else {
        Ok(false)
    }
}

fn gpu_or_skip() -> bool {
    let required = std::env::var_os("XAGENT_REQUIRE_GPU").is_some();
    match gpu_guard_decision(GpuKernel::is_available(), required) {
        Ok(true) => true,
        Ok(false) => {
            eprintln!("Skipping: no GPU/fallback adapter available");
            false
        }
        Err(message) => panic!("{message}"),
    }
}

fn world(enabled: bool, seed: u64) -> WorldConfig {
    WorldConfig {
        seed,
        food_kinds_enabled: enabled,
        ..WorldConfig::default()
    }
}

/// Kinds read after construction and after `N` mixed resets.
fn kernel_sequence(world: &WorldConfig) -> Vec<Option<FoodKind>> {
    let brain = BrainConfig::default();
    let mut kernel = GpuKernel::new(AGENTS, FOOD, &brain, world);
    let mut seq = vec![kernel.nourishing_food_kind()];
    for i in 0..N {
        if i % 2 == 0 {
            kernel.reset_agents_seeded(&brain, RESET_SEED + i);
        } else {
            kernel.reset_agents(&brain);
        }
        seq.push(kernel.nourishing_food_kind());
    }
    seq
}

#[test]
fn food_kind_draw_off_by_default() {
    if !gpu_or_skip() {
        return;
    }
    let world = WorldConfig::default();
    let brain = BrainConfig::default();
    let mut kernel = GpuKernel::new(AGENTS, FOOD, &brain, &world);
    assert_eq!(kernel.nourishing_food_kind(), None);
    kernel.reset_agents_seeded(&brain, RESET_SEED);
    assert_eq!(kernel.nourishing_food_kind(), None);
    assert_eq!(food_kinds_slot_value(kernel.nourishing_food_kind()), 0.0);
}

#[test]
fn food_kind_draw_redraws_on_every_reset() {
    if !gpu_or_skip() {
        return;
    }
    let world = world(true, 5);
    let expected: Vec<Option<FoodKind>> = (0..=N)
        .map(|i| Some(world.nourishing_food_kind(i)))
        .collect();
    assert_eq!(kernel_sequence(&world), expected);
}

#[test]
fn food_kind_draw_is_seed_deterministic() {
    if !gpu_or_skip() {
        return;
    }
    let world = world(true, 9);
    assert_eq!(kernel_sequence(&world), kernel_sequence(&world));
}

#[test]
fn food_kind_draw_packs_world_config_slot() {
    if !gpu_or_skip() {
        return;
    }
    let world = world(true, 5);
    let brain = BrainConfig::default();
    let kernel = GpuKernel::new(AGENTS, FOOD, &brain, &world);
    let kind = kernel.nourishing_food_kind();
    assert_eq!(kind, Some(world.nourishing_food_kind(0)));
    assert_eq!(
        food_kinds_slot_value(kind),
        food_kinds_slot_value(Some(world.nourishing_food_kind(0)))
    );
}

#[test]
fn gpu_guard_fails_closed_when_required() {
    assert!(gpu_guard_decision(false, true).is_err());
    assert_eq!(gpu_guard_decision(false, false), Ok(false));
    assert_eq!(gpu_guard_decision(true, true), Ok(true));
}
