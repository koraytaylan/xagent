//! The homeostatic gradient head must not apply a learning step on the tick
//! that crosses death. Zero is a legal prediction, so death stores an
//! out-of-range sentinel and the next brain tick skips the update.

use xagent_brain::buffers::{
    ENCODED_DIMENSION, HOMEO_PREDICTION_ABSENT, O_HOMEO, O_HOMEO_PREDICTOR_BIAS,
    O_HOMEO_PREDICTOR_WEIGHTS, O_PREV_HOMEO_PREDICTION, O_PREV_PREDICTION, PREDICTOR_DIMENSION,
    P_ALIVE, P_DEATH_COUNT, P_ENERGY, P_HOMEO_PREDICTED_GRADIENT_OUT,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Online step. Large enough that one tick moves the bias by a dyadic
/// amount, small enough that the ±0.3 bias clamp does not hide the sign.
const PROBE_LEARNING_RATE: f32 = 0.5;
/// In-range previous prediction. With a zero raw gradient the signed error
/// equals this value, so the bias steps by `-learning_rate * this`.
const PROBE_PREV_PREDICTION: f32 = 0.25;
/// Feature written into one `O_PREV_PREDICTION` lane. The weight step is
/// `learning_rate * (1/ENCODED_DIMENSION) * error * this`.
const PROBE_FEATURE: f32 = 1.0;
/// Bias seeded for the death checks. Not zero, so a reset-to-zero bug fails.
const SEEDED_BIAS: f32 = 0.125;
/// Weight seeded in lane 0 for the death checks.
const SEEDED_WEIGHT: f32 = 0.5;
/// Lane that carries [`PROBE_FEATURE`].
const PROBE_LANE: usize = 0;
/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
/// Full energy and integrity used when the tick must not be a respawn spike.
const FULL_METER: f32 = 100.0;
/// Per-dimension scale of the predictor weight step. Mirrors `TD_VECTOR_SCALE`.
const PREDICTOR_DIMENSION_SCALE: f32 = 1.0 / ENCODED_DIMENSION as f32;
/// Largest absolute value the head will store. Mirrors `MAX_HOMEOSTATIC_DELTA`.
const MAX_STORED_PREDICTION: f32 = 0.3;
/// Tolerance for the one-step bias and weight check. The expected values are
/// dyadic, so this only absorbs shader rounding.
const STEP_TOLERANCE: f32 = 1e-5;

fn predictor_brain(stride: u32) -> BrainConfig {
    BrainConfig {
        brain_tick_stride: stride,
        vision_stride: 1,
        movement_speed: 0.0,
        homeo_predictive_credit_enabled: true,
        homeo_predictor_learning_rate: PROBE_LEARNING_RATE,
        homeo_predictive_credit_beta: 0.0,
        ..BrainConfig::default()
    }
}

fn upload_flat_world(kernel: &GpuKernel) {
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    kernel.upload_world(&heights, &biomes, &[(50.0, 0.0, 50.0)], &[false], &[0.0]);
    let brain = BrainConfig::default();
    kernel.upload_agents(&[(
        glam::Vec3::new(0.0, 1.0, 0.0),
        FULL_METER,
        FULL_METER,
        brain.memory_capacity,
        brain.processing_slots,
    )]);
}

fn zero_predictor(state: &mut [f32]) {
    let weights = O_HOMEO_PREDICTOR_WEIGHTS;
    state[weights..weights + ENCODED_DIMENSION].fill(0.0);
    state[O_HOMEO_PREDICTOR_BIAS] = 0.0;
    let prev = O_PREV_PREDICTION;
    state[prev..prev + PREDICTOR_DIMENSION].fill(0.0);
}

/// Birth leaves the previous-prediction slot absent. The first brain tick
/// sees a full meter against a zero homeostatic slot, so a stored 0 would
/// be trained as a real sample and move the bias. This test seeds only the
/// bias and does not write the prediction slot.
#[test]
fn homeo_predictor_first_tick_does_not_train_birth_sentinel() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = predictor_brain(1);
    let world = WorldConfig {
        seed: 2,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(1, 1, &brain, &world);
    upload_flat_world(&kernel);

    let mut before = kernel.read_agent_state(0);
    assert_eq!(
        before.brain_state.len(),
        xagent_brain::buffers::BRAIN_STRIDE,
        "static offsets are only valid for the default vision layout"
    );
    assert_eq!(
        before.brain_state[O_PREV_HOMEO_PREDICTION], HOMEO_PREDICTION_ABSENT,
        "init stored a legal prediction of 0; the first tick would train it"
    );
    // Leave the homeostatic previous energy/integrity at their init zeros
    // and do not assign the prediction slot. Only the bias is seeded.
    before.brain_state[O_HOMEO_PREDICTOR_BIAS] = SEEDED_BIAS;
    kernel.write_agent_state(0, &before);

    kernel.dispatch_batch_masked(0, 1, 0x4);

    let after = kernel.read_agent_state(0);
    assert_eq!(
        after.brain_state[O_HOMEO_PREDICTOR_BIAS], SEEDED_BIAS,
        "first tick trained the birth sentinel as a prediction of 0"
    );
}

/// `reset_agents` is the other birth path. A planted legal 0 must be replaced
/// by the absent sentinel, and the next brain tick must not train it.
#[test]
fn homeo_predictor_first_tick_after_reset_does_not_train() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = predictor_brain(1);
    let world = WorldConfig {
        seed: 2,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(1, 1, &brain, &world);
    upload_flat_world(&kernel);

    let mut planted = kernel.read_agent_state(0);
    assert_eq!(
        planted.brain_state.len(),
        xagent_brain::buffers::BRAIN_STRIDE
    );
    planted.brain_state[O_PREV_HOMEO_PREDICTION] = 0.0;
    planted.brain_state[O_HOMEO_PREDICTOR_BIAS] = SEEDED_BIAS;
    kernel.write_agent_state(0, &planted);

    kernel.reset_agents(&brain);

    let mut before = kernel.read_agent_state(0);
    assert_eq!(
        before.brain_state[O_PREV_HOMEO_PREDICTION], HOMEO_PREDICTION_ABSENT,
        "reset_agents left a legal prediction of 0"
    );
    before.brain_state[O_HOMEO_PREDICTOR_BIAS] = SEEDED_BIAS;
    kernel.write_agent_state(0, &before);

    kernel.dispatch_batch_masked(0, 1, 0x4);

    let after = kernel.read_agent_state(0);
    assert_eq!(
        after.brain_state[O_HOMEO_PREDICTOR_BIAS], SEEDED_BIAS,
        "first tick after reset trained a stored 0"
    );
}

/// A valid previous prediction and a zero homeostatic delta must move the
/// bias and the one seeded weight. This is the control that makes the death
/// assertions able to fail: if the head never learned, "unchanged across
/// death" would pass for the wrong reason.
#[test]
fn homeo_predictor_trains_when_previous_prediction_is_in_range() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = predictor_brain(1);
    let world = WorldConfig {
        seed: 1,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(1, 1, &brain, &world);
    upload_flat_world(&kernel);

    let mut before = kernel.read_agent_state(0);
    assert_eq!(
        before.brain_state.len(),
        xagent_brain::buffers::BRAIN_STRIDE,
        "static offsets are only valid for the default vision layout"
    );
    zero_predictor(&mut before.brain_state);
    before.brain_state[O_PREV_HOMEO_PREDICTION] = PROBE_PREV_PREDICTION;
    before.brain_state[O_PREV_PREDICTION + PROBE_LANE] = PROBE_FEATURE;
    // Normalized energy and integrity already match a full meter, so the
    // raw gradient on this brain-only tick is zero and the error is the
    // previous prediction itself.
    before.brain_state[O_HOMEO + 4] = 1.0;
    before.brain_state[O_HOMEO + 5] = 1.0;
    kernel.write_agent_state(0, &before);

    // Brain pass only: physics would drain energy and can kill the agent.
    kernel.dispatch_batch_masked(0, 1, 0x4);

    let after = kernel.read_agent_state(0);
    let bias = after.brain_state[O_HOMEO_PREDICTOR_BIAS];
    let weight = after.brain_state[O_HOMEO_PREDICTOR_WEIGHTS + PROBE_LANE];
    let expected_bias = -PROBE_LEARNING_RATE * PROBE_PREV_PREDICTION;
    let expected_weight =
        -PROBE_LEARNING_RATE * PREDICTOR_DIMENSION_SCALE * PROBE_PREV_PREDICTION * PROBE_FEATURE;
    assert!(
        (bias - expected_bias).abs() < STEP_TOLERANCE,
        "bias {bias} did not step to {expected_bias}"
    );
    assert!(
        (weight - expected_weight).abs() < STEP_TOLERANCE,
        "weight {weight} did not step to {expected_weight}"
    );
    assert!(
        expected_weight.abs() > STEP_TOLERANCE,
        "probe step is too small to tell a skipped update from a real one"
    );
}

fn seed_death_state(kernel: &mut GpuKernel) -> (f32, f32) {
    let mut state = kernel.read_agent_state(0);
    assert_eq!(
        state.brain_state.len(),
        xagent_brain::buffers::BRAIN_STRIDE,
        "static offsets are only valid for the default vision layout"
    );
    zero_predictor(&mut state.brain_state);
    state.brain_state[O_HOMEO_PREDICTOR_BIAS] = SEEDED_BIAS;
    state.brain_state[O_HOMEO_PREDICTOR_WEIGHTS + PROBE_LANE] = SEEDED_WEIGHT;
    state.brain_state[O_PREV_PREDICTION + PROBE_LANE] = PROBE_FEATURE;
    // A real in-range prediction. Death must invalidate it before the brain
    // tick, otherwise this value is trained against the respawn spike.
    state.brain_state[O_PREV_HOMEO_PREDICTION] = PROBE_PREV_PREDICTION;
    kernel.write_agent_state(0, &state);
    kernel.write_agent_physics_fields(0, &[(P_ENERGY, 0.0), (P_ALIVE, 1.0)]);
    (SEEDED_BIAS, SEEDED_WEIGHT)
}

/// Fused cycle: physics kills the agent, death marks the prediction absent,
/// and the brain pass in the same cycle does not apply SGD.
#[test]
fn homeo_predictor_skips_update_on_fused_respawn_tick() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = predictor_brain(1);
    let world = WorldConfig {
        seed: 3,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(1, 1, &brain, &world);
    upload_flat_world(&kernel);
    let (bias_before, weight_before) = seed_death_state(&mut kernel);

    kernel.dispatch_ticks(0, 1);

    let phys = kernel.read_full_state_blocking();
    let deaths = phys[P_DEATH_COUNT];
    let published_prediction = phys[P_HOMEO_PREDICTED_GRADIENT_OUT];
    assert!(
        deaths >= 1.0,
        "agent did not die, so the respawn path was not exercised"
    );
    let after = kernel.read_agent_state(0);
    let bias = after.brain_state[O_HOMEO_PREDICTOR_BIAS];
    let weight = after.brain_state[O_HOMEO_PREDICTOR_WEIGHTS + PROBE_LANE];
    assert_eq!(
        bias, bias_before,
        "respawn tick changed predictor bias ({bias_before} -> {bias})"
    );
    assert_eq!(
        weight, weight_before,
        "respawn tick changed predictor weight ({weight_before} -> {weight})"
    );
    let stored = after.brain_state[O_PREV_HOMEO_PREDICTION];
    assert!(
        stored.abs() <= MAX_STORED_PREDICTION,
        "stored prediction {stored} is outside the clamp; the brain pass did not replace the sentinel"
    );
    assert_ne!(
        stored, HOMEO_PREDICTION_ABSENT,
        "brain pass left the death sentinel in place instead of storing a new prediction"
    );
    assert_eq!(
        published_prediction, 0.0,
        "respawn tick published a predictive bonus"
    );
}

/// Physics-only remainder runs `phase_death` and does not run the brain.
/// The sentinel has to be visible before the next brain tick, and the
/// forward-model features that would have made a false gradient must survive.
#[test]
fn homeo_predictor_death_remainder_marks_prediction_absent() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    // Stride 2 makes a 1-tick dispatch a physics remainder: no brain pass.
    let brain = predictor_brain(2);
    let world = WorldConfig {
        seed: 5,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(1, 1, &brain, &world);
    upload_flat_world(&kernel);
    let (bias_before, weight_before) = seed_death_state(&mut kernel);

    kernel.dispatch_ticks(0, 1);

    let phys = kernel.read_full_state_blocking();
    assert!(
        phys[P_DEATH_COUNT] >= 1.0,
        "agent did not die on the physics remainder"
    );
    let after = kernel.read_agent_state(0);
    assert_eq!(
        after.brain_state[O_PREV_HOMEO_PREDICTION], HOMEO_PREDICTION_ABSENT,
        "physics-remainder death left a usable previous prediction"
    );
    assert_eq!(
        after.brain_state[O_PREV_PREDICTION + PROBE_LANE],
        PROBE_FEATURE,
        "death cleared the forward-model features the false gradient would use"
    );
    assert_eq!(after.brain_state[O_HOMEO_PREDICTOR_BIAS], bias_before);
    assert_eq!(
        after.brain_state[O_HOMEO_PREDICTOR_WEIGHTS + PROBE_LANE],
        weight_before
    );
}
