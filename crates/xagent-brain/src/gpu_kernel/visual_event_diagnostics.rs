//! Snapshot-only opportunities for reusing raw visual encoder inputs.
//! Published adapted features are read from the production private buffer after
//! each cycle; their raw input is the sensory buffer captured BEFORE that cycle.
//! The recurrence and encoder projections below are FP64 host diagnostics, not
//! an implemented GPU algorithm or an acceptance test for a changed trajectory.

use std::{collections::HashMap, error::Error};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::packed_store_validation::{advance, cache, prepare_kernel_with_store_suppression};
use super::rounding_validation::{check_fp32_dot, fp32_dot_reference};
use super::vision_validation::read_buffer;
use super::*;

/// Match the current packed encoder and store-suppression timing fixture.
const WIDTH: u32 = 8;
const HEIGHT: u32 = 6;
const SEEDS: [u64; 3] = [42, 314, 2026];
const WINDOW_STARTS: [u32; 2] = [256, 1_000];
/// Every window measures sixteen consecutive transitions, after a seed sample.
const WINDOW_CYCLES: u32 = 16;
const CHANNELS: [&str; 5] = ["red", "green", "blue", "alpha", "depth"];
const COLOR_COMPONENTS: usize = 4;
const RGB_COMPONENTS: usize = 3;
const DEPTH_CHANNEL: usize = CHANNELS.len() - 1;
const PHYSICS: usize = 0;
const SENSORY: usize = 7;
const BRAIN: usize = 8;
const MUTABLE_BUFFERS: usize = 13;
const WORD_BYTES: usize = size_of::<f32>();
const ALIVE_THRESHOLD: f32 = 0.5;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

#[derive(Clone, Copy, Default)]
struct ChannelCounts {
    components: u64,
    raw_changed: u64,
    adapted_changed: u64,
    adapted_changed_without_raw_change: u64,
    raw_zero: u64,
    adapted_zero: u64,
    raw_distinct_sum: u64,
    adapted_distinct_sum: u64,
    repeated_raw_groups: u64,
    repeated_raw_groups_with_distinct_adapted: u64,
    recurrence_max_abs: f64,
    recurrence_squared_sum: f64,
    adapted_squared_sum: f64,
}

impl ChannelCounts {
    fn merge(&mut self, other: Self) {
        self.components += other.components;
        self.raw_changed += other.raw_changed;
        self.adapted_changed += other.adapted_changed;
        self.adapted_changed_without_raw_change += other.adapted_changed_without_raw_change;
        self.raw_zero += other.raw_zero;
        self.adapted_zero += other.adapted_zero;
        self.raw_distinct_sum += other.raw_distinct_sum;
        self.adapted_distinct_sum += other.adapted_distinct_sum;
        self.repeated_raw_groups += other.repeated_raw_groups;
        self.repeated_raw_groups_with_distinct_adapted +=
            other.repeated_raw_groups_with_distinct_adapted;
        self.recurrence_max_abs = self.recurrence_max_abs.max(other.recurrence_max_abs);
        self.recurrence_squared_sum += other.recurrence_squared_sum;
        self.adapted_squared_sum += other.adapted_squared_sum;
    }
}

#[derive(Clone, Copy, Default)]
struct ProjectionCounts {
    rows: u64,
    signed_exceeds_budget: u64,
    envelope_exceeds_budget: u64,
    signed_max_abs: f64,
    signed_squared_sum: f64,
    envelope_max: f64,
    signed_over_budget_max: f64,
    envelope_over_budget_max: f64,
}

impl ProjectionCounts {
    fn merge(&mut self, other: Self) {
        self.rows += other.rows;
        self.signed_exceeds_budget += other.signed_exceeds_budget;
        self.envelope_exceeds_budget += other.envelope_exceeds_budget;
        self.signed_max_abs = self.signed_max_abs.max(other.signed_max_abs);
        self.signed_squared_sum += other.signed_squared_sum;
        self.envelope_max = self.envelope_max.max(other.envelope_max);
        self.signed_over_budget_max = self
            .signed_over_budget_max
            .max(other.signed_over_budget_max);
        self.envelope_over_budget_max = self
            .envelope_over_budget_max
            .max(other.envelope_over_budget_max);
    }
}

#[derive(Clone, Copy, Default)]
struct Counts {
    agent_pairs: u64,
    inactive_pairs: u64,
    transition_pairs: u64,
    rays: u64,
    raw_rgb_unchanged_rays: u64,
    raw_all_unchanged_rays: u64,
    adapted_all_unchanged_rays: u64,
    channels: [ChannelCounts; CHANNELS.len()],
    projection: ProjectionCounts,
}

impl Counts {
    fn merge(&mut self, other: Self) {
        self.agent_pairs += other.agent_pairs;
        self.inactive_pairs += other.inactive_pairs;
        self.transition_pairs += other.transition_pairs;
        self.rays += other.rays;
        self.raw_rgb_unchanged_rays += other.raw_rgb_unchanged_rays;
        self.raw_all_unchanged_rays += other.raw_all_unchanged_rays;
        self.adapted_all_unchanged_rays += other.adapted_all_unchanged_rays;
        for (total, count) in self.channels.iter_mut().zip(other.channels) {
            total.merge(count);
        }
        self.projection.merge(other.projection);
    }

    fn report(self, label: &str, rate: f32) {
        println!("VISUAL_EVENTS {label} eligible_agent_pairs={} inactive_pairs_excluded={} death_or_alive_transitions_excluded={} rays={} unchanged_raw_rgb_rays={} unchanged_raw_rgba_depth_rays={} unchanged_adapted_rgba_depth_rays={} input=before_cycle features=private_after_cycle stationary_fixture=false replay_exact_buffers={MUTABLE_BUFFERS}",
            self.agent_pairs, self.inactive_pairs, self.transition_pairs, self.rays,
            self.raw_rgb_unchanged_rays, self.raw_all_unchanged_rays, self.adapted_all_unchanged_rays);
        for (name, channel) in CHANNELS.into_iter().zip(self.channels) {
            let rms = if channel.components == 0 {
                0.0
            } else {
                (channel.recurrence_squared_sum / channel.components as f64).sqrt()
            };
            let normalized = if channel.adapted_squared_sum == 0.0 {
                if channel.recurrence_squared_sum == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                (channel.recurrence_squared_sum / channel.adapted_squared_sum).sqrt()
            };
            println!("VISUAL_EVENT_CHANNEL {label} channel={name} components={} raw_changed_bits={} adapted_changed_bits={} adapted_changed_with_raw_unchanged={} raw_zero={} adapted_zero={} raw_distinct_sum={} adapted_distinct_sum={} repeated_raw_groups={} repeated_raw_groups_with_distinct_adapted={} recurrence_max_abs={:.9e} recurrence_rms={rms:.9e} recurrence_normalized_l2={normalized:.9e} rho_f32={rate:.9e} beta_f64={:.17e} grouping=exact_bits_within_agent_and_channel",
                channel.components, channel.raw_changed, channel.adapted_changed,
                channel.adapted_changed_without_raw_change, channel.raw_zero, channel.adapted_zero,
                channel.raw_distinct_sum, channel.adapted_distinct_sum, channel.repeated_raw_groups,
                channel.repeated_raw_groups_with_distinct_adapted, channel.recurrence_max_abs,
                1.0 - f64::from(rate));
        }
        let projection = self.projection;
        let rms = if projection.rows == 0 {
            0.0
        } else {
            (projection.signed_squared_sum / projection.rows as f64).sqrt()
        };
        println!("VISUAL_EVENT_PROJECTION {label} rows={} signed_projection_exceeds_fresh_dot_budget={} absolute_envelope_exceeds_fresh_dot_budget={} signed_max_abs={:.9e} signed_rms={rms:.9e} absolute_envelope_max={:.9e} signed_over_budget_max={:.9e} envelope_over_budget_max={:.9e} weights=before_current_credit bias_and_nonvisual=current full_feature_dot_budget=nearest_gamma_2n_f64_ftz recurrence=ideal_f64_one_step matrix_update_reuse=excluded gpu_recurrence_unimplemented=true acceptance_not_asserted=true timing_not_measured=true",
            projection.rows, projection.signed_exceeds_budget, projection.envelope_exceeds_budget,
            projection.signed_max_abs, projection.envelope_max, projection.signed_over_budget_max,
            projection.envelope_over_budget_max);
    }
}

pub(super) fn bits(bytes: &[u8], index: usize) -> u32 {
    let first = index * WORD_BYTES;
    u32::from_le_bytes(bytes[first..first + WORD_BYTES].try_into().unwrap())
}

pub(super) fn number(bytes: &[u8], index: usize) -> f32 {
    let value = f32::from_bits(bits(bytes, index));
    assert!(value.is_finite());
    value
}

pub(super) fn adaptation_rate() -> f32 {
    let prefix = "const SENSORY_ADAPTATION_RATE: f32 = ";
    let common = include_str!("../shaders/kernel/common.wgsl");
    let values: Vec<_> = common
        .lines()
        .filter_map(|line| line.strip_prefix(prefix))
        .collect();
    assert_eq!(values.len(), 1);
    let rate: f32 = values[0].split(';').next().unwrap().parse().unwrap();
    assert!(rate.is_normal() && rate > 0.0 && rate < 1.0);
    let source = include_str!("../shaders/kernel/brain_passes.wgsl");
    assert!(source.contains("s_features[j] = feature - running_mean;"));
    assert!(source.contains(
        "brain_state[slot] = running_mean + SENSORY_ADAPTATION_RATE * (feature - running_mean);"
    ));
    rate
}

/// Full public snapshots plus the F published inputs for each agent.
pub(super) struct Sample {
    pub(super) before: State,
    pub(super) after: State,
    pub(super) features: Vec<f32>,
}

fn published_features(kernel: &GpuKernel, after: &State) -> TestResult<Vec<f32>> {
    let packed = cache(kernel);
    assert!(packed.is_valid());
    let bytes = read_buffer(kernel, packed.buffer(), packed.buffer().size())?;
    let agents = usize::try_from(kernel.agent_count).unwrap();
    let prefix_words =
        (kernel.layout.brain_scratch_stride * agents).div_ceil(COLOR_COMPONENTS) * COLOR_COMPONENTS;
    let matrix_words = kernel.layout.feature_count * ENCODED_DIMENSION;
    let mut features = Vec::with_capacity(agents * kernel.layout.feature_count);
    for agent in 0..agents {
        let feature_base = agent * kernel.layout.brain_scratch_stride + SCRATCH_FEATURES;
        features.extend(
            (0..kernel.layout.feature_count).map(|feature| number(&bytes, feature_base + feature)),
        );
        let first = (prefix_words + agent * matrix_words) * WORD_BYTES;
        let scalar = (agent * kernel.layout.brain_stride + O_ENC_WEIGHTS) * WORD_BYTES;
        assert_eq!(
            &bytes[first..first + matrix_words * WORD_BYTES],
            &after[BRAIN][scalar..scalar + matrix_words * WORD_BYTES],
            "public/private encoder mirror"
        );
    }
    Ok(features)
}

pub(super) fn sample(kernel: &mut GpuKernel, cycle: u32) -> TestResult<Sample> {
    let before = capture_state(kernel)?;
    let saved = checkpoint(kernel);
    advance(kernel, cycle, 1);
    let after = capture_state(kernel)?;
    let features = published_features(kernel, &after)?;
    restore(kernel, &saved);
    advance(kernel, cycle, 1);
    assert_state_equal(kernel, &after, &capture_state(kernel)?);
    Ok(Sample {
        before,
        after,
        features,
    })
}

pub(super) fn steady_agent(sample: &Sample, agent: usize) -> bool {
    let base = agent * PHYS_STRIDE;
    number(&sample.before[PHYSICS], base + P_ALIVE) >= ALIVE_THRESHOLD
        && number(&sample.after[PHYSICS], base + P_ALIVE) >= ALIVE_THRESHOLD
        && bits(&sample.before[PHYSICS], base + P_DEATH_COUNT)
            == bits(&sample.after[PHYSICS], base + P_DEATH_COUNT)
}

fn feature_index(kernel: &GpuKernel, ray: usize, channel: usize) -> usize {
    if channel == DEPTH_CHANNEL {
        kernel.layout.vision_color_count + ray
    } else {
        ray * COLOR_COMPONENTS + channel
    }
}

pub(super) fn check_adaptation(kernel: &GpuKernel, sample: &Sample, agent: usize, rate: f32) {
    let features = agent * kernel.layout.feature_count;
    let sensory = agent * kernel.layout.sensory_stride;
    let brain = agent * kernel.layout.brain_stride;
    let mean = brain + fixed_tail_base(kernel.layout.brain_stride) + O_SENSORY_MEAN
        - O_PREDICTOR_CONTEXT_WEIGHT;
    let visual_count = kernel.layout.vision_color_count + kernel.layout.vision_depth_count;
    for feature in 0..visual_count {
        let raw = number(&sample.before[SENSORY], sensory + feature);
        let old_mean = number(&sample.before[BRAIN], mean + feature);
        let adapted = sample.features[features + feature];
        let subtraction = check_fp32_dot(
            &[raw, old_mean],
            &[1.0, -1.0],
            adapted,
            "published adapted feature matches raw input minus old mean",
        );
        // A contracted mean update need not reuse the separately rounded
        // published difference. Include its already-derived subtraction error.
        let update =
            fp32_dot_reference(&[old_mean, rate], &[1.0, adapted], "adaptation mean update");
        let current_mean = f64::from(number(&sample.after[BRAIN], mean + feature));
        assert!(
            (current_mean - update.reference_f64).abs()
                <= update.forward_bound + f64::from(rate) * subtraction.forward_bound,
            "published mean violates the observed nearest-rounding/FTZ update envelope"
        );
    }
}

fn channel_observation(
    kernel: &GpuKernel,
    previous: &Sample,
    current: &Sample,
    agent: usize,
    channel: usize,
    rate: f32,
    residual: &mut [f64],
) -> ChannelCounts {
    let mut counts = ChannelCounts::default();
    let mut raw_groups: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut adapted_groups = std::collections::HashSet::new();
    let sensory = agent * kernel.layout.sensory_stride;
    let features = agent * kernel.layout.feature_count;
    for ray in 0..kernel.layout.vision_depth_count {
        let feature = feature_index(kernel, ray, channel);
        let old_raw = number(&previous.before[SENSORY], sensory + feature);
        let raw = number(&current.before[SENSORY], sensory + feature);
        let old_adapted = previous.features[features + feature];
        let adapted = current.features[features + feature];
        let raw_changed = old_raw.to_bits() != raw.to_bits();
        let adapted_changed = old_adapted.to_bits() != adapted.to_bits();
        counts.components += 1;
        counts.raw_changed += u64::from(raw_changed);
        counts.adapted_changed += u64::from(adapted_changed);
        counts.adapted_changed_without_raw_change += u64::from(!raw_changed && adapted_changed);
        counts.raw_zero += u64::from(raw == 0.0);
        counts.adapted_zero += u64::from(adapted == 0.0);
        raw_groups
            .entry(raw.to_bits())
            .or_default()
            .push(adapted.to_bits());
        adapted_groups.insert(adapted.to_bits());
        let predicted = (1.0 - f64::from(rate)) * f64::from(old_adapted)
            + (f64::from(raw) - f64::from(old_raw));
        let error = f64::from(adapted) - predicted;
        residual[feature] = error;
        counts.recurrence_max_abs = counts.recurrence_max_abs.max(error.abs());
        counts.recurrence_squared_sum += error * error;
        counts.adapted_squared_sum += f64::from(adapted) * f64::from(adapted);
    }
    counts.raw_distinct_sum = u64::try_from(raw_groups.len()).unwrap();
    counts.adapted_distinct_sum = u64::try_from(adapted_groups.len()).unwrap();
    for values in raw_groups.values().filter(|values| values.len() > 1) {
        counts.repeated_raw_groups += 1;
        counts.repeated_raw_groups_with_distinct_adapted +=
            u64::from(values.iter().any(|&value| value != values[0]));
    }
    counts
}

fn project_residual(
    kernel: &GpuKernel,
    current: &Sample,
    agent: usize,
    residual: &[f64],
) -> ProjectionCounts {
    let mut counts = ProjectionCounts::default();
    let features = kernel.layout.feature_count;
    let brain = agent * kernel.layout.brain_stride;
    let mut input = current.features[agent * features..(agent + 1) * features].to_vec();
    input.push(1.0);
    for output in 0..ENCODED_DIMENSION {
        // Encode precedes credit. After-cycle matrices would measure the wrong
        // linear operator, so every row is taken from the BEFORE snapshot.
        let mut weights: Vec<_> = (0..features)
            .map(|feature| {
                number(
                    &current.before[BRAIN],
                    brain + O_ENC_WEIGHTS + feature * ENCODED_DIMENSION + output,
                )
            })
            .collect();
        weights.push(number(
            &current.before[BRAIN],
            brain + features * ENCODED_DIMENSION + output,
        ));
        let budget = fp32_dot_reference(
            &input,
            &weights,
            "current encoder full-feature dot plus bias",
        );
        let mut signed = 0.0;
        let mut envelope = 0.0;
        for (&error, &weight) in residual.iter().zip(&weights) {
            let term = error * f64::from(weight);
            signed += term;
            envelope += term.abs();
        }
        assert!(signed.is_finite() && envelope.is_finite() && budget.forward_bound > 0.0);
        counts.rows += 1;
        counts.signed_exceeds_budget += u64::from(signed.abs() > budget.forward_bound);
        counts.envelope_exceeds_budget += u64::from(envelope > budget.forward_bound);
        counts.signed_max_abs = counts.signed_max_abs.max(signed.abs());
        counts.signed_squared_sum += signed * signed;
        counts.envelope_max = counts.envelope_max.max(envelope);
        counts.signed_over_budget_max = counts
            .signed_over_budget_max
            .max(signed.abs() / budget.forward_bound);
        counts.envelope_over_budget_max = counts
            .envelope_over_budget_max
            .max(envelope / budget.forward_bound);
    }
    counts
}

fn observe(kernel: &GpuKernel, previous: &Sample, current: &Sample, rate: f32) -> Counts {
    assert_eq!(current.before.len(), MUTABLE_BUFFERS);
    let mut counts = Counts::default();
    let visual_count = kernel.layout.vision_color_count + kernel.layout.vision_depth_count;
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        if !steady_agent(previous, agent) || !steady_agent(current, agent) {
            let base = agent * PHYS_STRIDE + P_ALIVE;
            if [
                &previous.before,
                &previous.after,
                &current.before,
                &current.after,
            ]
            .iter()
            .all(|state| number(&state[PHYSICS], base) < ALIVE_THRESHOLD)
            {
                counts.inactive_pairs += 1;
            } else {
                counts.transition_pairs += 1;
            }
            continue;
        }
        counts.agent_pairs += 1;
        check_adaptation(kernel, previous, agent, rate);
        check_adaptation(kernel, current, agent, rate);
        let tick = agent * kernel.layout.brain_stride
            + fixed_tail_base(kernel.layout.brain_stride)
            + O_TICK_COUNT
            - O_PREDICTOR_CONTEXT_WEIGHT;
        assert_eq!(
            number(&current.after[BRAIN], tick),
            number(&previous.after[BRAIN], tick) + 1.0
        );
        let mut residual = vec![0.0; visual_count];
        for channel in 0..CHANNELS.len() {
            counts.channels[channel].merge(channel_observation(
                kernel,
                previous,
                current,
                agent,
                channel,
                rate,
                &mut residual,
            ));
        }
        let sensory = agent * kernel.layout.sensory_stride;
        let features = agent * kernel.layout.feature_count;
        for ray in 0..kernel.layout.vision_depth_count {
            let raw_same = |channel| {
                let index = sensory + feature_index(kernel, ray, channel);
                bits(&previous.before[SENSORY], index) == bits(&current.before[SENSORY], index)
            };
            counts.rays += 1;
            counts.raw_rgb_unchanged_rays += u64::from((0..RGB_COMPONENTS).all(raw_same));
            counts.raw_all_unchanged_rays += u64::from((0..CHANNELS.len()).all(raw_same));
            counts.adapted_all_unchanged_rays += u64::from((0..CHANNELS.len()).all(|channel| {
                let index = features + feature_index(kernel, ray, channel);
                previous.features[index].to_bits() == current.features[index].to_bits()
            }));
        }
        counts
            .projection
            .merge(project_residual(kernel, current, agent, &residual));
    }
    assert_eq!(
        counts.agent_pairs + counts.inactive_pairs + counts.transition_pairs,
        u64::from(kernel.agent_count)
    );
    assert!(counts
        .channels
        .iter()
        .all(|channel| channel.components == counts.rays));
    assert_eq!(
        counts.projection.rows,
        counts.agent_pairs * u64::try_from(ENCODED_DIMENSION).unwrap()
    );
    counts
}

#[test]
#[ignore = "requires GPU; snapshot-only production visual-event diagnostics"]
fn raw_visual_events_and_adapted_encoder_recurrence() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel_with_store_suppression(WIDTH, HEIGHT, false, true);
    assert!(!kernel.layout.visual_cortex_enabled);
    assert!(kernel.global_credit_active());
    let world = checkpoint(&kernel);
    let rate = adaptation_rate();
    let brain = BrainConfig {
        vision_width: WIDTH,
        vision_height: HEIGHT,
        vision_stride: 1,
        ..BrainConfig::default()
    };
    let mut totals = [Counts::default(); WINDOW_STARTS.len()];
    for seed in SEEDS {
        restore(&mut kernel, &world);
        kernel.reset_agents_seeded(&brain, seed);
        let mut cycle = 0;
        for (window, start) in WINDOW_STARTS.into_iter().enumerate() {
            // Capture the raw input of the final warmup cycle as well as its
            // published output. A post-vision sensory snapshot alone cannot
            // reconstruct which raw frame produced that previous feature.
            advance(&mut kernel, cycle, start - 1 - cycle);
            let mut previous = sample(&mut kernel, start - 1)?;
            cycle = start;
            let mut counts = Counts::default();
            for _ in 0..WINDOW_CYCLES {
                let current = sample(&mut kernel, cycle)?;
                counts.merge(observe(&kernel, &previous, &current, rate));
                previous = current;
                cycle += 1;
            }
            counts.report(
                &format!("seed={seed} start_cycle={start} cycles={WINDOW_CYCLES}"),
                rate,
            );
            totals[window].merge(counts);
        }
    }
    for (start, counts) in WINDOW_STARTS.into_iter().zip(totals) {
        counts.report(
            &format!(
                "seed=all seeds={} start_cycle={start} cycles_per_seed={WINDOW_CYCLES}",
                SEEDS.len()
            ),
            rate,
        );
    }
    Ok(())
}
