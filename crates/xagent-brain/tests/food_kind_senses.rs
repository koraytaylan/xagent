//! Odour by food kind. With kinds on, nostril pair 1 smells only `First`
//! items and the two extra channels right after the scent channels smell only
//! `Second` items, under one law. With kinds and the second sense off the
//! scent and the sensory stride equal the baseline.

use xagent_brain::buffers::{BrainLayout, NON_VISUAL_COUNT, SCENT_CHANNELS};
use xagent_brain::GpuKernel;
use xagent_shared::{
    BrainConfig, FoodKind, WorldConfig, NOSTRIL_FORWARD_OFFSET, NOSTRIL_SIDE_OFFSET,
    SCENT_DECAY_LENGTH, SCENT_RANGE,
};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
const FULL_METER: f32 = 100.0;
/// Ticks run after upload: one, so the vision pass fills the sensory buffer
/// before any brain output has turned the agent (the motor decision is
/// cleared before the tick).
const TICKS: u64 = 1;
/// Absolute floor for the relative comparison of odour values near zero.
const ABSOLUTE_FLOOR: f32 = 1.0e-3;
const WORLD_SEED: u64 = 5;
const RESET_SEED: u64 = 11;
/// Resets tried while looking for a changed nourishing kind.
const MAX_RESETS: u64 = 32;
const RELATIVE_TOLERANCE: f32 = 1e-6;
/// Index of the facing vector in the non-visual tail.
const FACING_SLOT: usize = 3;
/// Index of the left nostril of pair 1 in the non-visual tail.
const SCENT_SLOT: usize = NON_VISUAL_COUNT - SCENT_CHANNELS;
/// Index of the left nostril of pair 2 in the non-visual tail.
const SECOND_SLOT: usize = NON_VISUAL_COUNT;
/// Food spread over both kinds, all inside smelling range and out of reach.
const FOOD_POSITIONS: [(f32, f32, f32); 6] = [
    (-4.0, 0.35, 6.0),
    (5.0, 0.35, 4.0),
    (-6.0, 0.35, -5.0),
    (3.0, 0.35, 9.0),
    (-2.0, 0.35, 12.0),
    (8.0, 0.35, -3.0),
];
const AGENT_POSITIONS: [(f32, f32, f32); 2] = [(0.0, 1.0, 0.0), (20.0, 1.0, 20.0)];

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

fn brain_config(second: bool) -> BrainConfig {
    BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        movement_speed: 0.0,
        second_odour_sense_enabled: second,
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

/// CPU mirror of the shader's odour law over the items `keep` selects.
fn expected_odour(
    pos: (f32, f32, f32),
    facing: [f32; 3],
    strength: f32,
    keep: impl Fn(usize) -> bool,
) -> [f32; 2] {
    let right = [facing[2], -facing[0]];
    let nose = [
        pos.0 + facing[0] * NOSTRIL_FORWARD_OFFSET,
        pos.2 + facing[2] * NOSTRIL_FORWARD_OFFSET,
    ];
    let nostrils = [
        [
            nose[0] - right[0] * NOSTRIL_SIDE_OFFSET,
            nose[1] - right[1] * NOSTRIL_SIDE_OFFSET,
        ],
        [
            nose[0] + right[0] * NOSTRIL_SIDE_OFFSET,
            nose[1] + right[1] * NOSTRIL_SIDE_OFFSET,
        ],
    ];
    let mut concentration = [0.0_f32; 2];
    for (i, food) in FOOD_POSITIONS.iter().enumerate() {
        if !keep(i) {
            continue;
        }
        for (side, nostril) in nostrils.iter().enumerate() {
            let dx = food.0 - nostril[0];
            let dz = food.2 - nostril[1];
            let distance_sq = dx * dx + dz * dz;
            if distance_sq < SCENT_RANGE * SCENT_RANGE {
                concentration[side] += (-distance_sq.sqrt() / SCENT_DECAY_LENGTH).exp();
            }
        }
    }
    [
        1.0 - (-strength * concentration[0]).exp(),
        1.0 - (-strength * concentration[1]).exp(),
    ]
}

fn close(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= RELATIVE_TOLERANCE * expected.abs().max(ABSOLUTE_FLOOR)
}

/// Upload the scene and run the vision pass; returns each agent's
/// non-visual tail.
fn run(
    kernel: &mut GpuKernel,
    brain: &BrainConfig,
    positions: &[(f32, f32, f32)],
) -> Vec<Vec<f32>> {
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    let eaten = [false; FOOD_POSITIONS.len()];
    kernel.upload_world(
        &heights,
        &biomes,
        &FOOD_POSITIONS,
        &eaten,
        &[0.0; FOOD_POSITIONS.len()],
    );
    let agents: Vec<_> = positions
        .iter()
        .map(|p| {
            (
                glam::Vec3::new(p.0, p.1, p.2),
                FULL_METER,
                FULL_METER,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agents);
    let count = u32::try_from(positions.len()).expect("agent count fits u32");
    // Clear any motor decision left by an earlier run so the first physics
    // tick does not turn the agent before vision reads its senses.
    for i in 0..count {
        kernel.write_motor_decision(i, 0.0, 0.0, 0.0);
    }
    for tick in 0..TICKS {
        kernel.dispatch_batch(tick, 1);
    }
    (0..count)
        .map(|i| kernel.read_agent_telemetry_blocking(i).sensory_non_visual)
        .collect()
}

fn facing_of(tail: &[f32]) -> [f32; 3] {
    [
        tail[FACING_SLOT],
        tail[FACING_SLOT + 1],
        tail[FACING_SLOT + 2],
    ]
}

fn assert_pair(tail: &[f32], slot: usize, expected: [f32; 2], what: &str) {
    for side in 0..2 {
        assert!(
            tail[slot + side] > 0.0 || expected[side] == 0.0,
            "{what} nostril {side}: nothing smelt"
        );
        assert!(
            close(tail[slot + side], expected[side]),
            "{what} nostril {side}: GPU {} vs expected {}",
            tail[slot + side],
            expected[side]
        );
    }
}

#[test]
fn food_kind_senses_default_off_matches_baseline() {
    if !gpu_or_skip() {
        return;
    }
    let brain = brain_config(false);
    let mut kernel = GpuKernel::new(1, FOOD_POSITIONS.len(), &brain, &world_config(false));
    kernel.reset_agents_seeded(&brain, RESET_SEED);
    assert_eq!(
        kernel.brain_layout().sensory_stride,
        BrainLayout::from_config(&BrainConfig::default()).sensory_stride
    );
    let tails = run(&mut kernel, &brain, &AGENT_POSITIONS[..1]);
    let expected = expected_odour(
        AGENT_POSITIONS[0],
        facing_of(&tails[0]),
        brain.smell_strength,
        |_| true,
    );
    assert_pair(&tails[0], SCENT_SLOT, expected, "baseline pair 1");
}

#[test]
fn food_kind_senses_odour_split_by_kind() {
    if !gpu_or_skip() {
        return;
    }
    let brain = brain_config(true);
    let mut kernel = GpuKernel::new(1, FOOD_POSITIONS.len(), &brain, &world_config(true));
    kernel.reset_agents_seeded(&brain, RESET_SEED);
    let initial = kernel.nourishing_food_kind();
    let before = run(&mut kernel, &brain, &AGENT_POSITIONS[..1]);
    let facing = facing_of(&before[0]);
    let first = expected_odour(AGENT_POSITIONS[0], facing, brain.smell_strength, |i| {
        FoodKind::of_item(i) == FoodKind::First
    });
    let second = expected_odour(AGENT_POSITIONS[0], facing, brain.smell_strength, |i| {
        FoodKind::of_item(i) == FoodKind::Second
    });
    assert_pair(&before[0], SCENT_SLOT, first, "pair 1 (First only)");
    assert_pair(&before[0], SECOND_SLOT, second, "pair 2 (Second only)");

    let mut changed = false;
    for i in 0..MAX_RESETS {
        kernel.reset_agents_seeded(&brain, RESET_SEED + 1 + i);
        if kernel.nourishing_food_kind() != initial {
            changed = true;
            break;
        }
    }
    assert!(changed, "the nourishing kind never changed");
    let after = run(&mut kernel, &brain, &AGENT_POSITIONS[..1]);
    assert_eq!(before[0].len(), after[0].len(), "sensory tail length");
    for (slot, (b, a)) in before[0].iter().zip(after[0].iter()).enumerate() {
        assert_eq!(b, a, "sensory slot {slot} changed");
    }
    assert_pair(&after[0], SCENT_SLOT, first, "reseeded pair 1");
    assert_pair(&after[0], SECOND_SLOT, second, "reseeded pair 2");
}

#[test]
fn food_kind_senses_parallel_scent_matches_serial() {
    if !gpu_or_skip() {
        return;
    }
    const VAR: &str = "XAGENT_VISION_PARALLEL_SCENT";
    let brain = brain_config(true);
    let world = world_config(true);
    let mut serial = GpuKernel::new(2, FOOD_POSITIONS.len(), &brain, &world);
    serial.reset_agents_seeded(&brain, RESET_SEED);
    let serial_tails = run(&mut serial, &brain, &AGENT_POSITIONS);

    let previous = std::env::var_os(VAR);
    std::env::set_var(VAR, "1");
    let mut parallel = GpuKernel::new(2, FOOD_POSITIONS.len(), &brain, &world);
    match previous {
        Some(value) => std::env::set_var(VAR, value),
        None => std::env::remove_var(VAR),
    }
    parallel.reset_agents_seeded(&brain, RESET_SEED);
    let parallel_tails = run(&mut parallel, &brain, &AGENT_POSITIONS);

    for agent in 0..AGENT_POSITIONS.len() {
        for side in 0..2 {
            assert_eq!(
                serial_tails[agent][SCENT_SLOT + side].to_bits(),
                parallel_tails[agent][SCENT_SLOT + side].to_bits(),
                "agent {agent} pair 1 nostril {side}"
            );
            assert_eq!(
                serial_tails[agent][SECOND_SLOT + side],
                parallel_tails[agent][SECOND_SLOT + side],
                "agent {agent} pair 2 nostril {side}"
            );
        }
    }
}
