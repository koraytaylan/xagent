//! Harmful meals act only through integrity and are counted separately;
//! nourishing meals, and every meal with food kinds off, behave as before.

use xagent_brain::buffers::{
    harmful_meal_integrity_loss, PHYS_STRIDE, P_ALIVE, P_DEATH_COUNT, P_ENERGY, P_FOOD_COUNT,
    P_HARMFUL_MEAL_COUNT, P_INTEGRITY, P_LAST_DEATH_TICK,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, FoodKind, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
const FULL_METER: f32 = 100.0;
const FOOD_HEIGHT: f32 = 0.35;
/// Items sit this far apart on the x axis, well outside each other's reach.
const ITEM_SPACING: f32 = 20.0;
/// The control agent stands far from both items and eats nothing.
const CONTROL_X: f32 = -40.0;
/// One brain cycle at the default stride.
const ONE_CYCLE: u32 = 10;
/// Energy at 40% of max, below the 50% gate for integrity regeneration.
const LOW_ENERGY: f32 = 0.4 * FULL_METER;
/// Tick the lethal dispatch starts at, so a recorded death tick is nonzero.
const DISPATCH_TICK: u64 = 7;
/// `DISPATCH_TICK` as the f32 the kernel stores in `P_LAST_DEATH_TICK`.
const DISPATCH_TICK_F32: f32 = 7.0;
/// A lethal start integrity: this fraction of the harmful meal's loss.
const LETHAL_FRACTION_OF_LOSS: f32 = 0.5;
/// Number of food items in the test world.
const ITEM_COUNT: usize = 2;
/// f32 rounding.
const LOSS_TOLERANCE: f32 = 1e-3;
/// Agents in the test world: two at items, one control.
const AGENTS: u32 = 3;
const CONTROL: usize = 2;

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

struct Outcome {
    nourishing: Option<FoodKind>,
    physics: Vec<f32>,
    food_energy: f32,
}

impl Outcome {
    fn at(&self, agent: usize, slot: usize) -> f32 {
        self.physics[agent * PHYS_STRIDE + slot]
    }
}

/// Item index of the harmful kind given the nourishing kind.
fn harmful_item(nourishing: FoodKind) -> usize {
    match nourishing {
        FoodKind::First => 1,
        FoodKind::Second => 0,
    }
}

/// Agents 0 and 1 each stand at their own item; agent 2 is a control. With
/// `low_energy` every agent starts at 40% energy; `harmful_integrity` sets
/// the integrity of the agent at the harmful item.
fn run(kinds: bool, low_energy: bool, harmful_integrity: Option<f32>) -> Outcome {
    let brain = BrainConfig {
        movement_speed: 0.0,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: 4,
        food_kinds_enabled: kinds,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(AGENTS, 2, &brain, &world);
    kernel.reset_agents_seeded(&brain, 8);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    kernel.upload_world(
        &heights,
        &biomes,
        &[(0.0, FOOD_HEIGHT, 0.0), (ITEM_SPACING, FOOD_HEIGHT, 0.0)],
        &[false, false],
        &[0.0, 0.0],
    );
    let xs = [0.0, ITEM_SPACING, CONTROL_X];
    let bodies: Vec<_> = xs
        .iter()
        .map(|&x| {
            (
                glam::Vec3::new(x, 1.0, 0.0),
                FULL_METER,
                FULL_METER,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&bodies);
    let nourishing = kernel.nourishing_food_kind();
    for agent in 0..AGENTS {
        if low_energy {
            kernel.write_agent_physics_fields(agent, &[(P_ENERGY, LOW_ENERGY)]);
        }
    }
    if let (Some(kind), Some(integrity)) = (nourishing, harmful_integrity) {
        kernel.write_agent_physics_fields(
            u32::try_from(harmful_item(kind)).expect("item index fits u32"),
            &[(P_INTEGRITY, integrity)],
        );
    }
    kernel.dispatch_batch(DISPATCH_TICK, ONE_CYCLE);
    Outcome {
        nourishing,
        physics: kernel.read_full_state_blocking().to_vec(),
        food_energy: world.food_energy_value,
    }
}

fn expected_loss(food_energy: f32) -> f32 {
    harmful_meal_integrity_loss(food_energy, FULL_METER, FULL_METER)
}

#[test]
fn food_kind_meals_default_off_every_item_nourishes() {
    if !gpu_or_skip() {
        return;
    }
    let out = run(false, false, None);
    assert_eq!(out.nourishing, None);
    for item in 0..ITEM_COUNT {
        let gain = out.at(item, P_ENERGY) - out.at(CONTROL, P_ENERGY);
        assert!(
            (gain - out.food_energy).abs() <= LOSS_TOLERANCE,
            "item {item} gave {gain}, expected {}",
            out.food_energy
        );
        assert_eq!(out.at(item, P_FOOD_COUNT), 1.0);
        assert_eq!(out.at(item, P_INTEGRITY), FULL_METER);
        assert_eq!(out.at(item, P_HARMFUL_MEAL_COUNT), 0.0);
    }
}

#[test]
fn food_kind_meals_nourishing_and_harmful() {
    if !gpu_or_skip() {
        return;
    }
    let out = run(true, true, None);
    let kind = out.nourishing.expect("kinds on draw a nourishing kind");
    let harmful = harmful_item(kind);
    let nourishing = 1 - harmful;
    let control_energy = out.at(CONTROL, P_ENERGY);

    let gain = out.at(nourishing, P_ENERGY) - control_energy;
    assert!(
        (gain - out.food_energy).abs() <= LOSS_TOLERANCE,
        "gain {gain}"
    );
    assert_eq!(out.at(nourishing, P_FOOD_COUNT), 1.0);
    assert_eq!(out.at(nourishing, P_HARMFUL_MEAL_COUNT), 0.0);

    let harmful_gain = out.at(harmful, P_ENERGY) - control_energy;
    assert!(
        harmful_gain.abs() <= LOSS_TOLERANCE,
        "harmful gain {harmful_gain}"
    );
    let lost = FULL_METER - out.at(harmful, P_INTEGRITY);
    let expected = expected_loss(out.food_energy);
    assert!(
        (lost - expected).abs() <= LOSS_TOLERANCE,
        "lost {lost}, expected {expected}"
    );
    assert_eq!(out.at(harmful, P_HARMFUL_MEAL_COUNT), 1.0);
    assert_eq!(out.at(harmful, P_FOOD_COUNT), 0.0);
}

#[test]
fn food_kind_meals_lethal_harmful_meal_marks_death() {
    if !gpu_or_skip() {
        return;
    }
    let world = WorldConfig::default();
    let integrity = expected_loss(world.food_energy_value) * LETHAL_FRACTION_OF_LOSS;
    let out = run(true, true, Some(integrity));
    let harmful = harmful_item(out.nourishing.expect("kinds on"));
    assert_eq!(out.at(harmful, P_DEATH_COUNT), 1.0);
    assert_eq!(out.at(harmful, P_LAST_DEATH_TICK), DISPATCH_TICK_F32);
}

#[test]
fn food_kind_meals_harmful_counter_survives_respawn() {
    if !gpu_or_skip() {
        return;
    }
    let world = WorldConfig::default();
    let integrity = expected_loss(world.food_energy_value) * LETHAL_FRACTION_OF_LOSS;
    let out = run(true, true, Some(integrity));
    let harmful = harmful_item(out.nourishing.expect("kinds on"));
    assert_eq!(out.at(harmful, P_DEATH_COUNT), 1.0);
    assert_eq!(out.at(harmful, P_ALIVE), 1.0);
    assert_eq!(out.at(harmful, P_INTEGRITY), FULL_METER);
    assert_eq!(out.at(harmful, P_HARMFUL_MEAL_COUNT), 1.0);
    assert_eq!(out.at(harmful, P_FOOD_COUNT), 0.0);
}
