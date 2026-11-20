//! The klinotaxis turn-gain and motor-fatigue switches: defaults, packing,
//! inheritance, and the shader branches that pin each factor to `1.0`.

use xagent_brain::buffers::{
    build_config, CFG_KLINOTAXIS_TURN_GAIN_ENABLED, CFG_MOTOR_FATIGUE_ENABLED,
};
use xagent_brain::GpuKernel;
use xagent_sandbox::agent::{crossover_config, mutate_config_seeded};
use xagent_shared::{BrainConfig, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
/// Full energy and integrity at spawn.
const FULL_METER: f32 = 100.0;
/// Ticks run by the fatigue probe.
const FATIGUE_TICKS: u64 = 3000;
/// Ticks between fatigue samples.
const SAMPLE_EVERY: u64 = 25;
/// Relative tolerance on the turn ratio.
const RATIO_TOLERANCE: f32 = 1e-5;
/// Mirrors the klinotaxis factor's clamp in `brain_passes.wgsl`.
const KLINOTAXIS_MIN: f32 = 0.3;
const KLINOTAXIS_MAX: f32 = 3.0;

#[test]
fn default_modulator_flags_are_on() {
    let config = BrainConfig::default();
    assert!(config.klinotaxis_turn_gain_enabled);
    assert!(config.motor_fatigue_enabled);
}

#[test]
fn config_without_modulator_fields_defaults_on() {
    let mut value = serde_json::to_value(BrainConfig::default()).unwrap();
    let map = value.as_object_mut().unwrap();
    map.remove("klinotaxis_turn_gain_enabled");
    map.remove("motor_fatigue_enabled");
    let config: BrainConfig = serde_json::from_value(value).unwrap();
    assert!(config.klinotaxis_turn_gain_enabled);
    assert!(config.motor_fatigue_enabled);
}

#[test]
fn build_config_packs_modulator_flags() {
    let on = build_config(&BrainConfig::default());
    assert_eq!(on[CFG_KLINOTAXIS_TURN_GAIN_ENABLED], 1.0);
    assert_eq!(on[CFG_MOTOR_FATIGUE_ENABLED], 1.0);
    let off = build_config(&BrainConfig {
        klinotaxis_turn_gain_enabled: false,
        motor_fatigue_enabled: false,
        ..BrainConfig::default()
    });
    assert_eq!(off[CFG_KLINOTAXIS_TURN_GAIN_ENABLED], 0.0);
    assert_eq!(off[CFG_MOTOR_FATIGUE_ENABLED], 0.0);
}

#[test]
fn mutation_and_crossover_keep_modulator_flags() {
    let parent = BrainConfig {
        klinotaxis_turn_gain_enabled: false,
        motor_fatigue_enabled: false,
        ..BrainConfig::default()
    };
    let other = BrainConfig::default();
    for seed in 0..8 {
        let child = mutate_config_seeded(&parent, seed);
        assert!(!child.klinotaxis_turn_gain_enabled);
        assert!(!child.motor_fatigue_enabled);
    }
    let crossed = crossover_config(&parent, &other);
    assert!(!crossed.klinotaxis_turn_gain_enabled);
    assert!(!crossed.motor_fatigue_enabled);
}

/// One-agent kernel on flat ground with far food, as in the executed-action
/// credit probe.
fn probe_kernel(brain: &BrainConfig) -> GpuKernel {
    let world = WorldConfig {
        seed: 4,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(1, 1, brain, &world);
    kernel.reset_agents_seeded(brain, 11);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    kernel.upload_world(&heights, &biomes, &[(100.0, 0.0, 100.0)], &[false], &[0.0]);
    kernel.upload_agents(&[(
        glam::Vec3::new(0.0, 1.0, 0.0),
        FULL_METER,
        FULL_METER,
        brain.memory_capacity,
        brain.processing_slots,
    )]);
    kernel
}

fn stride_one(motor_fatigue_enabled: bool, klinotaxis_turn_gain_enabled: bool) -> BrainConfig {
    BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        motor_fatigue_enabled,
        klinotaxis_turn_gain_enabled,
        ..BrainConfig::default()
    }
}

/// Fatigue factors sampled over a seeded run.
fn sampled_fatigue(motor_fatigue_enabled: bool) -> Vec<f32> {
    let brain = stride_one(motor_fatigue_enabled, true);
    let mut kernel = probe_kernel(&brain);
    let mut samples = Vec::new();
    for tick in 0..FATIGUE_TICKS {
        kernel.dispatch_batch(tick, 1);
        if tick % SAMPLE_EVERY == 0 {
            samples.push(kernel.read_agent_telemetry_blocking(0).fatigue_factor);
        }
    }
    samples
}

#[test]
#[ignore = "requires a GPU adapter"]
fn motor_fatigue_off_pins_factor_to_one() {
    assert!(GpuKernel::is_available(), "a GPU adapter is required");
    let off = sampled_fatigue(false);
    assert!(!off.is_empty());
    for (i, factor) in off.iter().enumerate() {
        assert_eq!(
            *factor, 1.0,
            "sample {i}: fatigue factor with the switch off"
        );
    }
    let on = sampled_fatigue(true);
    assert!(
        on.iter().any(|factor| *factor < 1.0),
        "the default run never showed fatigue; the probe does not discriminate"
    );
}

/// Motor turn and fatigue factor after the first tick.
fn turn_after_energy_drop(klinotaxis_turn_gain_enabled: bool) -> (f32, f32) {
    let brain = stride_one(true, klinotaxis_turn_gain_enabled);
    let mut kernel = probe_kernel(&brain);
    // The first tick of life reads the full meter as a one-tick energy
    // change, which drives the gradient deviation to a clamp. Both runs start
    // from identical state, so the single tick isolates the gate.
    kernel.dispatch_batch(0, 1);
    let telemetry = kernel.read_agent_telemetry_blocking(0);
    (telemetry.motor_turn, telemetry.fatigue_factor)
}

#[test]
#[ignore = "requires a GPU adapter"]
fn klinotaxis_off_pins_turn_gain_to_one() {
    assert!(GpuKernel::is_available(), "a GPU adapter is required");
    let (turn_on, fatigue_on) = turn_after_energy_drop(true);
    let (turn_off, fatigue_off) = turn_after_energy_drop(false);
    assert_eq!(
        fatigue_on, fatigue_off,
        "fatigue must not depend on the gate"
    );
    assert!(turn_on.abs() > 1e-6, "the probe turn is zero: {turn_on}");
    let ratio = turn_off / turn_on;
    let near = |target: f32| (ratio - target).abs() <= RATIO_TOLERANCE * target.abs();
    assert!(
        near(1.0 / KLINOTAXIS_MIN) || near(1.0 / KLINOTAXIS_MAX),
        "off/on turn ratio {ratio} is not a clamp reciprocal (on {turn_on}, off {turn_off})"
    );
}
