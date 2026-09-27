//! Episodic memory with homeostatic salience.
//!
//! A pattern is stored every brain tick with no valence. When the agent's own
//! homeostatic signal jumps far outside its recent normal (a salient tick),
//! the signed, normalized size of the jump is credited back to the patterns
//! stored over the preceding window, discounted by age. Ordinary ticks leave
//! valence alone, and birth/respawn ticks are never salient; death adds no
//! label of its own. Fully valued episodes do not fade with time, and moments still
//! awaiting their outcome are never evicted in favor of valued ones.

use xagent_brain::buffers::{
    BRAIN_STRIDE, ENCODED_DIMENSION, MEMORY_CAP, O_MIN_REINF_IDX, O_PAT_ACTIVE, O_PAT_META,
    O_PAT_MOTOR, O_PAT_NORMS, O_PAT_REINF, O_PAT_RETURN, O_PAT_STATES, O_SALIENCE_LABEL,
    O_SALIENCE_MEAN, O_SALIENCE_VARIANCE, O_TICK_COUNT, O_VALUE_WEIGHTS, P_DEATH_COUNT, P_ENERGY,
    P_MAX_ENERGY,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
/// Full energy and integrity at spawn.
const FULL_METER: f32 = 100.0;
/// Ordinary brain ticks run before the probe event, so the memory holds more
/// moments than the credit window.
const WARMUP_TICKS: u64 = 20;
/// Energy removed before the probe so the meal fits under the meter's cap.
const HUNGER_DEFICIT: f32 = 50.0;
/// Energy added in one step to simulate a meal: a normalized gain of 0.3,
/// far outside the resting drain, so the label saturates at +1.
const MEAL_ENERGY: f32 = 30.0;
/// Mirrors `EPISODIC_CREDIT_WINDOW` in `common.wgsl`.
const CREDIT_WINDOW: f32 = 8.0;
/// Mirrors `EPISODIC_CREDIT_DECAY` (= TD_DISCOUNT × TD_LAMBDA) in `common.wgsl`.
const CREDIT_DECAY: f32 = 0.97 * 0.9;
/// Absorbs shader float rounding in the credited valence.
const VALENCE_TOLERANCE: f32 = 1e-4;
/// Ordinary ticks run after the meal to check the credit is not washed out.
const AFTERMATH_TICKS: u64 = 40;
/// Ticks run after the meal to check retention: several full turnovers of
/// memory, longer than an unrecalled unvalued moment lasts before its
/// reinforcement decays to zero.
const RETENTION_TICKS: u64 = 4 * MEMORY_CAP as u64;
/// Starting energy for the repeated-meal probe, low enough that every meal
/// fits under the meter's cap.
const STARVED_ENERGY: f32 = 10.0;
/// Energy of each small meal in the repeated-meal probe: far outside the
/// resting drain, so every meal is salient.
const SMALL_MEAL_ENERGY: f32 = 4.0;
/// Brain ticks between small meals: longer than the credit window, so each
/// meal credits a fresh set of moments.
const MEAL_INTERVAL: u64 = 10;
/// Enough meals to credit more moments than memory holds.
const MEAL_COUNT: u64 = 20;
/// Mirrors `TD_DISCOUNT` in `common.wgsl`: the remembered return discounts
/// the reward by γ per brain tick of age.
const RETURN_DISCOUNT: f32 = 0.97;
/// Remembered return planted on one settled moment for the replay probe.
const PLANTED_RETURN: f32 = 0.5;
/// Age (brain ticks) of the planted moment: well past the credit window.
const PLANTED_AGE: f32 = 100.0;
/// Brain ticks the replay probe runs: the planted slot comes up about
/// REPLAY_TICKS / MEMORY_CAP ≈ 16 times.
const REPLAY_TICKS: u64 = 2000;
/// After ~16 replays at CRITIC_REPLAY_RATE 0.1 and ‖key‖ = 1, the critic
/// closes ~1 − 0.95¹⁶ ≈ 56% of the gap; require well over a fifth of it.
const MIN_REPLAYED_FRACTION: f32 = 0.2;

/// Brain of the stationary probe agent. Single-tick strides make every
/// physics tick a brain tick.
fn stationary_brain() -> BrainConfig {
    BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        movement_speed: 0.0,
        ..BrainConfig::default()
    }
}

/// One stationary agent on flat, hazard-free ground with its only food item
/// far out of reach.
fn stationary_kernel() -> GpuKernel {
    stationary_kernel_with(&stationary_brain())
}

fn stationary_kernel_with(brain: &BrainConfig) -> GpuKernel {
    let world = WorldConfig {
        seed: 3,
        ..WorldConfig::default()
    };
    let kernel = GpuKernel::new(1, 1, brain, &world);
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

fn run_ticks(kernel: &mut GpuKernel, start: &mut u64, ticks: u64) {
    for _ in 0..ticks {
        kernel.dispatch_batch(*start, 1);
        *start += 1;
    }
}

/// `(age, valence)` of every active pattern, age measured in brain ticks.
fn pattern_ages_and_valences(kernel: &GpuKernel) -> Vec<(f32, f32)> {
    let state = kernel.read_agent_state(0);
    assert_eq!(
        state.brain_state.len(),
        BRAIN_STRIDE,
        "static offsets are only valid for the default vision layout"
    );
    let tick = state.brain_state[O_TICK_COUNT];
    (0..MEMORY_CAP)
        .filter(|&slot| state.patterns[O_PAT_ACTIVE + slot] >= 0.5)
        .map(|slot| {
            let created = state.patterns[O_PAT_META + slot * 3];
            let valence = state.patterns[O_PAT_MOTOR + slot * 3 + 2];
            (tick - created, valence)
        })
        .collect()
}

fn set_energy(kernel: &mut GpuKernel, energy: f32) {
    kernel.write_agent_physics_fields(0, &[(P_ENERGY, energy)]);
}

#[test]
fn salient_energy_gain_credits_only_the_preceding_window() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = stationary_kernel();
    let mut tick = 0_u64;
    // Start hungry: the deficit lands before the first brain tick, which the
    // birth guard ignores, so no moment is labeled before the meal.
    set_energy(&mut kernel, FULL_METER - HUNGER_DEFICIT);
    run_ticks(&mut kernel, &mut tick, WARMUP_TICKS);

    let before = pattern_ages_and_valences(&kernel);
    assert!(
        before.len() as f32 > CREDIT_WINDOW,
        "memory holds {} moments; the probe needs more than the credit window",
        before.len()
    );
    assert!(
        before.iter().all(|(_, valence)| *valence == 0.0),
        "moments carry valence before any salient change: {before:?}"
    );

    let energy = kernel.read_full_state_blocking()[P_ENERGY];
    set_energy(&mut kernel, energy + MEAL_ENERGY);
    run_ticks(&mut kernel, &mut tick, 1);

    let brain = kernel.read_agent_state(0).brain_state;
    assert!(
        brain[O_SALIENCE_LABEL] > 0.99,
        "a +0.3 energy step must saturate the label, got {}",
        brain[O_SALIENCE_LABEL]
    );
    let after = pattern_ages_and_valences(&kernel);
    let mut credited = 0;
    for (age, valence) in &after {
        // The brain-tick counter advances before the store pass, so the meal
        // tick's own moment sits at age 0 and the moments that preceded the
        // meal at ages 1..=WINDOW — the same ages the shader credited.
        if (1.0..=CREDIT_WINDOW).contains(age) {
            let expected = CREDIT_DECAY.powf(age - 1.0);
            assert!(
                (valence - expected).abs() < VALENCE_TOLERANCE,
                "moment {age} ticks before the meal has valence {valence}, expected {expected}"
            );
            credited += 1;
        } else {
            assert_eq!(
                *valence, 0.0,
                "moment at age {age} outside the credit window gained valence"
            );
        }
    }
    assert_eq!(
        credited, CREDIT_WINDOW as usize,
        "every moment in the credit window must be credited"
    );
}

#[test]
fn ordinary_ticks_do_not_wash_out_credited_valence() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = stationary_kernel();
    let mut tick = 0_u64;
    set_energy(&mut kernel, FULL_METER - HUNGER_DEFICIT);
    run_ticks(&mut kernel, &mut tick, WARMUP_TICKS);
    let energy = kernel.read_full_state_blocking()[P_ENERGY];
    set_energy(&mut kernel, energy + MEAL_ENERGY);
    run_ticks(&mut kernel, &mut tick, 1);

    let mut credited: Vec<f32> = pattern_ages_and_valences(&kernel)
        .into_iter()
        .filter(|(_, valence)| *valence > 0.0)
        .map(|(_, valence)| valence)
        .collect();
    credited.sort_by(f32::total_cmp);
    assert_eq!(credited.len(), CREDIT_WINDOW as usize);

    run_ticks(&mut kernel, &mut tick, AFTERMATH_TICKS);

    let mut kept: Vec<f32> = pattern_ages_and_valences(&kernel)
        .into_iter()
        .filter(|(_, valence)| *valence > 0.0)
        .map(|(_, valence)| valence)
        .collect();
    kept.sort_by(f32::total_cmp);
    assert_eq!(
        kept, credited,
        "credited moments must survive ordinary ticks unchanged (no drift, no eviction)"
    );
}

#[test]
fn first_tick_of_life_is_never_salient() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = stationary_kernel();
    let mut tick = 0_u64;

    // Birth: previous energy/integrity are zero, so the first tick reads as a
    // full-meter gain. It must not be salient and must not feed the statistics.
    run_ticks(&mut kernel, &mut tick, 1);
    let brain = kernel.read_agent_state(0).brain_state;
    assert_eq!(
        brain[O_SALIENCE_LABEL], 0.0,
        "birth tick was labeled salient"
    );
    assert_eq!(
        brain[O_SALIENCE_MEAN], 0.0,
        "birth tick fed the salience mean"
    );
    assert_eq!(
        brain[O_SALIENCE_VARIANCE], 0.0,
        "birth tick fed the salience variance"
    );

    // The next ordinary tick does feed them (the resting drain is negative).
    run_ticks(&mut kernel, &mut tick, 1);
    let brain = kernel.read_agent_state(0).brain_state;
    assert_eq!(brain[O_SALIENCE_LABEL], 0.0);
    assert!(
        brain[O_SALIENCE_MEAN] < 0.0,
        "ordinary tick did not update the salience mean ({})",
        brain[O_SALIENCE_MEAN]
    );
}

#[test]
fn starvation_death_labels_nothing_and_respawn_is_not_rewarded() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = stationary_kernel();
    let mut tick = 0_u64;
    run_ticks(&mut kernel, &mut tick, WARMUP_TICKS);

    // Starve the agent: it dies on the next tick and respawns at full energy,
    // which the first post-respawn brain tick must not read as a gain.
    set_energy(&mut kernel, 0.0);
    run_ticks(&mut kernel, &mut tick, 1);
    let physics = kernel.read_full_state_blocking();
    assert!(physics[P_DEATH_COUNT] >= 1.0, "the agent did not die");
    assert!(
        physics[P_ENERGY] > 0.5 * physics[P_MAX_ENERGY],
        "the agent did not respawn"
    );

    let brain = kernel.read_agent_state(0).brain_state;
    assert_eq!(
        brain[O_SALIENCE_LABEL], 0.0,
        "the respawn refill was labeled a salient change"
    );
    let valences = pattern_ages_and_valences(&kernel);
    assert!(
        valences.iter().all(|(_, valence)| *valence == 0.0),
        "death or respawn credited stored moments: {valences:?}"
    );
}

#[test]
fn fully_valued_episode_does_not_fade() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    // Without learning, recall adds no reinforcement, so a moment's
    // reinforcement moves only by its decay.
    let mut kernel = stationary_kernel_with(&BrainConfig {
        learning_rate: 0.0,
        ..stationary_brain()
    });
    let mut tick = 0_u64;
    set_energy(&mut kernel, FULL_METER - HUNGER_DEFICIT);
    run_ticks(&mut kernel, &mut tick, WARMUP_TICKS);
    let energy = kernel.read_full_state_blocking()[P_ENERGY];
    set_energy(&mut kernel, energy + MEAL_ENERGY);
    run_ticks(&mut kernel, &mut tick, 1);

    // The moment right before the saturating meal carries the full +1.
    let state = kernel.read_agent_state(0);
    let slot = (0..MEMORY_CAP)
        .find(|&slot| {
            state.patterns[O_PAT_ACTIVE + slot] >= 0.5
                && state.patterns[O_PAT_MOTOR + slot * 3 + 2] == 1.0
        })
        .expect("the meal must fully value the moment right before it");
    let created = state.patterns[O_PAT_META + slot * 3];
    let reinforcement = state.patterns[O_PAT_REINF + slot];

    run_ticks(&mut kernel, &mut tick, RETENTION_TICKS);

    let state = kernel.read_agent_state(0);
    assert!(
        state.patterns[O_PAT_ACTIVE + slot] >= 0.5
            && state.patterns[O_PAT_META + slot * 3] == created,
        "the fully valued episode left memory within {RETENTION_TICKS} ticks"
    );
    assert_eq!(state.patterns[O_PAT_MOTOR + slot * 3 + 2], 1.0);
    assert_eq!(
        state.patterns[O_PAT_REINF + slot],
        reinforcement,
        "the fully valued episode faded: reinforcement {} -> {}",
        reinforcement,
        state.patterns[O_PAT_REINF + slot]
    );
}

#[test]
fn memory_full_of_valued_episodes_still_credits_the_whole_window() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = stationary_kernel();
    let mut tick = 0_u64;
    set_energy(&mut kernel, STARVED_ENERGY);
    run_ticks(&mut kernel, &mut tick, WARMUP_TICKS);
    for meal in 0..MEAL_COUNT {
        let energy = kernel.read_full_state_blocking()[P_ENERGY];
        set_energy(&mut kernel, energy + SMALL_MEAL_ENERGY);
        run_ticks(&mut kernel, &mut tick, 1);
        assert!(
            kernel.read_agent_state(0).brain_state[O_SALIENCE_LABEL] > 0.0,
            "small meal {meal} was not salient"
        );
        // Meals are further apart than the credit window, so the moments at
        // ages 1..=WINDOW were stored after the previous meal and carry only
        // this meal's credit.
        let credited = pattern_ages_and_valences(&kernel)
            .iter()
            .filter(|(age, valence)| (1.0..=CREDIT_WINDOW).contains(age) && *valence > 0.0)
            .count();
        assert_eq!(
            credited, CREDIT_WINDOW as usize,
            "meal {meal}: recent moments were evicted in favor of valued episodes \
             before their outcome arrived"
        );
        run_ticks(&mut kernel, &mut tick, MEAL_INTERVAL - 1);
    }
    let valued = pattern_ages_and_valences(&kernel)
        .iter()
        .filter(|(_, valence)| *valence != 0.0)
        .count();
    assert!(
        valued >= MEMORY_CAP - MEAL_INTERVAL as usize,
        "only {valued} valued episodes; the probe must fill memory with them to test eviction"
    );
}

#[test]
fn salient_gain_is_remembered_as_the_preceding_moments_return() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = stationary_kernel();
    let mut tick = 0_u64;
    set_energy(&mut kernel, FULL_METER - HUNGER_DEFICIT);
    run_ticks(&mut kernel, &mut tick, WARMUP_TICKS);
    let energy = kernel.read_full_state_blocking()[P_ENERGY];
    set_energy(&mut kernel, energy + MEAL_ENERGY);
    run_ticks(&mut kernel, &mut tick, 1);

    let state = kernel.read_agent_state(0);
    let now = state.brain_state[O_TICK_COUNT];
    let returns: Vec<(f32, f32)> = (0..MEMORY_CAP)
        .filter(|&slot| state.patterns[O_PAT_ACTIVE + slot] >= 0.5)
        .map(|slot| {
            (
                now - state.patterns[O_PAT_META + slot * 3],
                state.patterns[O_PAT_RETURN + slot],
            )
        })
        .collect();
    let newest = returns
        .iter()
        .find(|(age, _)| *age == 1.0)
        .map(|(_, r)| *r)
        .expect("the moment right before the meal is in memory");
    assert!(
        newest > 0.0,
        "the meal left no return on the moment before it"
    );
    for (age, remembered) in &returns {
        if (1.0..=CREDIT_WINDOW).contains(age) {
            let expected = newest * RETURN_DISCOUNT.powf(age - 1.0);
            assert!(
                (remembered - expected).abs() < VALENCE_TOLERANCE,
                "moment {age} ticks before the meal remembers {remembered}, expected {expected}"
            );
        } else {
            assert_eq!(
                *remembered, 0.0,
                "moment at age {age} outside the window gained a return"
            );
        }
    }
}

#[test]
fn value_replay_moves_the_critic_toward_a_remembered_return() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = stationary_kernel();
    let mut tick = 0_u64;
    run_ticks(&mut kernel, &mut tick, WARMUP_TICKS);

    // Plant one settled, fully valued moment whose key is the first unit
    // vector, so the critic's value of it is simply value_weights[0].
    let mut state = kernel.read_agent_state(0);
    let now = state.brain_state[O_TICK_COUNT];
    let slot = (state.patterns[O_MIN_REINF_IDX] as usize + 1) % MEMORY_CAP;
    for d in 0..ENCODED_DIMENSION {
        state.patterns[O_PAT_STATES + d * MEMORY_CAP + slot] = if d == 0 { 1.0 } else { 0.0 };
        state.brain_state[O_VALUE_WEIGHTS + d] = 0.0;
    }
    state.patterns[O_PAT_NORMS + slot] = 1.0;
    state.patterns[O_PAT_REINF + slot] = 1.0;
    state.patterns[O_PAT_MOTOR + slot * 3 + 2] = 1.0;
    state.patterns[O_PAT_META + slot * 3] = now - PLANTED_AGE;
    state.patterns[O_PAT_ACTIVE + slot] = 1.0;
    state.patterns[O_PAT_RETURN + slot] = PLANTED_RETURN;
    kernel.write_agent_state(0, &state);

    run_ticks(&mut kernel, &mut tick, REPLAY_TICKS);

    let after = kernel.read_agent_state(0);
    assert_eq!(
        after.patterns[O_PAT_RETURN + slot],
        PLANTED_RETURN,
        "the planted moment was evicted or rewritten"
    );
    let value = after.brain_state[O_VALUE_WEIGHTS];
    assert!(
        value > MIN_REPLAYED_FRACTION * PLANTED_RETURN && value <= PLANTED_RETURN,
        "the critic values the remembered moment at {value}; replay should move it toward \
         {PLANTED_RETURN} (without replay it stays near 0)"
    );
}
