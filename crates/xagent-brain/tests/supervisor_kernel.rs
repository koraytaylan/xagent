//! Single-slot replacement and the nourishing-kind setter on the kernel.

use xagent_brain::buffers::{
    AgentBrainState, PATTERN_STRIDE, PHYS_STRIDE, P_FOOD_COUNT, P_HARMFUL_MEAL_COUNT,
};
use xagent_brain::gpu_kernel::{
    slot_spawn_physics_row, FoodKindSetError, SlotReplaceError, SlotSpawn,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, FoodKind, WorldConfig};

const AGENT_COUNT: usize = 3;
const TARGET_SLOT: u32 = 1;
const WARMUP_BATCHES: u64 = 3;
const MEAL_ITEM: usize = 0;
const BIRTH_SEED: u64 = 11;
const RESET_SEED: u64 = 8;
const FOOD_ITEMS: usize = 2;
const TERRAIN_SIDE: usize = 129;
const BIOME_SIDE: usize = 256;
const FOOD_HEIGHT: f32 = 0.35;
const FULL_METER: f32 = 100.0;
const AGENT_HEIGHT: f32 = 1.0;
const POSITION_STEP: f32 = 7.0;
const WORLD_SEED: u64 = 4;
const SHORT_LENGTH: usize = 3;

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

fn spawns(brain: &BrainConfig) -> Vec<SlotSpawn> {
    (0..AGENT_COUNT)
        .map(|i| SlotSpawn {
            position: glam::Vec3::new(POSITION_STEP * i as f32, AGENT_HEIGHT, -POSITION_STEP),
            max_energy: FULL_METER,
            max_integrity: FULL_METER,
            memory_capacity: brain.memory_capacity,
            processing_slots: brain.processing_slots,
        })
        .collect()
}

fn upload(kernel: &GpuKernel, spawns: &[SlotSpawn]) {
    let agents: Vec<_> = spawns
        .iter()
        .map(|s| {
            (
                s.position,
                s.max_energy,
                s.max_integrity,
                s.memory_capacity,
                s.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agents);
}

fn count(n: usize) -> u32 {
    u32::try_from(n).expect("count fits u32")
}

fn brains(kernel: &GpuKernel) -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..count(AGENT_COUNT))
        .map(|i| {
            let s = kernel.read_agent_state(i);
            (s.brain_state, s.patterns)
        })
        .collect()
}

fn row_of(physics: &[f32], agent: usize) -> Vec<f32> {
    physics[agent * PHYS_STRIDE..(agent + 1) * PHYS_STRIDE].to_vec()
}

fn spawn_row_values(spawn: &SlotSpawn) -> Vec<f32> {
    slot_spawn_physics_row(spawn)
        .into_iter()
        .map(|(_, v)| v)
        .collect()
}

fn plain_kernel(brain: &BrainConfig) -> GpuKernel {
    let world = WorldConfig::default();
    GpuKernel::new(count(AGENT_COUNT), 1, brain, &world)
}

#[test]
fn gpu_guard_fails_closed_when_required() {
    assert!(gpu_guard_decision(false, true).is_err());
    assert_eq!(gpu_guard_decision(false, false), Ok(false));
    assert_eq!(gpu_guard_decision(true, true), Ok(true));
    assert_eq!(gpu_guard_decision(true, false), Ok(true));
}

#[test]
fn spawn_row_matches_upload_agents() {
    if !gpu_or_skip() {
        return;
    }
    let brain = BrainConfig::default();
    let mut kernel = plain_kernel(&brain);
    let spawns = spawns(&brain);
    upload(&kernel, &spawns);
    let physics = kernel.read_full_state_blocking().to_vec();
    for (i, spawn) in spawns.iter().enumerate() {
        assert_eq!(row_of(&physics, i), spawn_row_values(spawn), "agent {i}");
    }
}

#[test]
fn replace_slot_resets_row_and_writes_brain() {
    if !gpu_or_skip() {
        return;
    }
    let brain = BrainConfig::default();
    let mut kernel = plain_kernel(&brain);
    kernel.reset_agents_seeded(&brain, RESET_SEED);
    let spawns = spawns(&brain);
    upload(&kernel, &spawns);
    let batch = kernel.kernel_batch_size();
    for b in 0..WARMUP_BATCHES {
        kernel.dispatch_batch(b * u64::from(batch), batch);
    }
    let before_physics = kernel.read_full_state_blocking().to_vec();
    let before_brains = brains(&kernel);

    let mut birth_kernel = plain_kernel(&brain);
    birth_kernel.reset_agents_seeded(&brain, BIRTH_SEED);
    let birth = birth_kernel.read_agent_state(TARGET_SLOT);

    let target = TARGET_SLOT as usize;
    kernel
        .replace_agent_slot(TARGET_SLOT, &birth, &brain, &spawns[target])
        .expect("replace");
    let after_physics = kernel.read_full_state_blocking().to_vec();
    let after_brains = brains(&kernel);

    assert_eq!(
        row_of(&after_physics, target),
        spawn_row_values(&spawns[target])
    );
    assert_eq!(after_brains[target].0, birth.brain_state);
    assert_eq!(after_brains[target].1, birth.patterns);
    for i in (0..AGENT_COUNT).filter(|&i| i != target) {
        assert_eq!(
            row_of(&after_physics, i),
            row_of(&before_physics, i),
            "row {i}"
        );
        assert_eq!(after_brains[i], before_brains[i], "brain {i}");
    }
}

#[test]
fn replace_slot_rejects_bad_index_and_lengths() {
    if !gpu_or_skip() {
        return;
    }
    let brain = BrainConfig::default();
    let mut kernel = plain_kernel(&brain);
    kernel.reset_agents_seeded(&brain, RESET_SEED);
    let spawns = spawns(&brain);
    upload(&kernel, &spawns);
    let before_physics = kernel.read_full_state_blocking().to_vec();
    let before_brains = brains(&kernel);
    let good = kernel.read_agent_state(TARGET_SLOT);
    let target = TARGET_SLOT as usize;

    let err = kernel
        .replace_agent_slot(count(AGENT_COUNT), &good, &brain, &spawns[target])
        .expect_err("bad index");
    assert!(matches!(err, SlotReplaceError::IndexOutOfRange { .. }));

    let short_brain = AgentBrainState {
        brain_state: vec![0.0; SHORT_LENGTH],
        patterns: good.patterns.clone(),
    };
    let err = kernel
        .replace_agent_slot(TARGET_SLOT, &short_brain, &brain, &spawns[target])
        .expect_err("short brain");
    assert!(matches!(err, SlotReplaceError::BrainStateLength { .. }));

    let short_patterns = AgentBrainState {
        brain_state: good.brain_state.clone(),
        patterns: vec![0.0; PATTERN_STRIDE - 1],
    };
    let err = kernel
        .replace_agent_slot(TARGET_SLOT, &short_patterns, &brain, &spawns[target])
        .expect_err("short patterns");
    assert!(matches!(err, SlotReplaceError::PatternLength { .. }));

    assert_eq!(kernel.read_full_state_blocking().to_vec(), before_physics);
    assert_eq!(brains(&kernel), before_brains);
}

#[test]
fn set_kind_refused_when_kinds_off() {
    if !gpu_or_skip() {
        return;
    }
    let brain = BrainConfig::default();
    let mut kernel = plain_kernel(&brain);
    let result = kernel.set_nourishing_food_kind(FoodKind::Second);
    assert!(matches!(result, Err(FoodKindSetError::FoodKindsDisabled)));
    assert_eq!(kernel.nourishing_food_kind(), None);
}

#[test]
fn set_kind_keeps_agents_and_draw_counter() {
    if !gpu_or_skip() {
        return;
    }
    let brain = BrainConfig::default();
    let world = WorldConfig {
        seed: WORLD_SEED,
        food_kinds_enabled: true,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(count(AGENT_COUNT), 1, &brain, &world);
    kernel.reset_agents_seeded(&brain, RESET_SEED);
    upload(&kernel, &spawns(&brain));
    let before_physics = kernel.read_full_state_blocking().to_vec();
    let before_brains = brains(&kernel);

    kernel
        .set_nourishing_food_kind(FoodKind::Second)
        .expect("kinds on");
    assert_eq!(kernel.nourishing_food_kind(), Some(FoodKind::Second));
    assert_eq!(kernel.read_full_state_blocking().to_vec(), before_physics);
    assert_eq!(brains(&kernel), before_brains);

    kernel.reset_agents_seeded(&brain, RESET_SEED);
    assert_eq!(
        kernel.nourishing_food_kind(),
        Some(world.nourishing_food_kind(1))
    );
}

/// One agent on one item at the origin; returns the physics after one cycle.
fn meal_outcome(kind: FoodKind) -> Vec<f32> {
    let brain = BrainConfig {
        movement_speed: 0.0,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: WORLD_SEED,
        food_kinds_enabled: true,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(1, FOOD_ITEMS, &brain, &world);
    kernel.reset_agents_seeded(&brain, RESET_SEED);
    kernel.set_nourishing_food_kind(kind).expect("kinds on");
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    let mut items = vec![(POSITION_STEP * 10.0, FOOD_HEIGHT, 0.0); FOOD_ITEMS];
    items[MEAL_ITEM] = (0.0, FOOD_HEIGHT, 0.0);
    kernel.upload_world(
        &heights,
        &biomes,
        &items,
        &[false; FOOD_ITEMS],
        &[0.0; FOOD_ITEMS],
    );
    kernel.upload_agents(&[(
        glam::Vec3::new(0.0, AGENT_HEIGHT, 0.0),
        FULL_METER,
        FULL_METER,
        brain.memory_capacity,
        brain.processing_slots,
    )]);
    let batch = kernel.kernel_batch_size();
    kernel.dispatch_batch(0, batch);
    kernel.read_full_state_blocking().to_vec()
}

#[test]
fn set_kind_changes_meal_outcome() {
    if !gpu_or_skip() {
        return;
    }
    let first = meal_outcome(FoodKind::First);
    assert_eq!(first[P_FOOD_COUNT], 1.0);
    assert_eq!(first[P_HARMFUL_MEAL_COUNT], 0.0);
    let second = meal_outcome(FoodKind::Second);
    assert_eq!(second[P_HARMFUL_MEAL_COUNT], 1.0);
    assert_eq!(second[P_FOOD_COUNT], 0.0);
}
