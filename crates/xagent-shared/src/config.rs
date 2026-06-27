//! Configuration types and presets for brain capacity and world parameters.
//!
//! All parameters that affect the cognitive architecture's capacity constraints
//! or the world's difficulty are defined here, with named presets for common
//! experimental configurations.

use serde::{Deserialize, Serialize};

/// Configuration for the brain's capacity constraints.
///
/// Each field is tagged with its relationship to the live GPU kernel:
///
/// - **active** — directly shapes kernel behavior every tick.
/// - **locked (compile-time)** — must equal a compile-time constant baked
///   into the Rust/WGSL brain (e.g. `xagent_brain::buffers::ENCODED_DIMENSION`).
///   The value in this struct is a read-only echo; the kernel ignores any
///   other value and `build_config_for` logs a warning on mismatch.
/// - **proxy (metabolic)** — only feeds the per-tick metabolic cost formula;
///   the kernel's actual structural capacity is a compile-time constant and
///   does not change with this value.
/// - **legacy** — carried through config/UI/evolution for backwards
///   compatibility but currently has no kernel-side effect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainConfig {
    /// **Proxy (metabolic).** Scales the per-tick metabolic brain-drain cost
    /// via `metabolic_drain_per_tick` and `physics_state[P_MEMORY_CAP]`. The
    /// kernel's actual pattern-memory size is fixed at
    /// `xagent_brain::buffers::MEMORY_CAP = 128`, independent of this value.
    /// Mutated by evolution; clamped to `[1, 2048]` at breeding time.
    pub memory_capacity: usize,
    /// **Proxy (metabolic).** Scales the per-tick metabolic brain-drain cost
    /// via `metabolic_drain_per_tick` and `physics_state[P_PROCESSING_SLOTS]`.
    /// The kernel's actual recall width is fixed at
    /// `xagent_brain::buffers::RECALL_K = 16`, independent of this value.
    /// Mutated by evolution; clamped to `[1, 128]` at breeding time.
    pub processing_slots: usize,
    /// **Legacy.** Superseded by the visual-cortex config (`retina_*`,
    /// `gabor_*`); retained only for deserialization back-compat (issue #106).
    /// No kernel stage reads this field and it is no longer carried through
    /// breeding or shown in the UI editor; the `serde` default supplies it when
    /// older JSON / `xagent.db` configs omit it.
    #[serde(default = "default_visual_encoding_size")]
    pub visual_encoding_size: usize,
    /// **Locked (compile-time).** Length of the internal representation
    /// vector. Must equal `xagent_brain::buffers::ENCODED_DIMENSION` — the
    /// kernel uses that constant to size encoder weights, predictor weights,
    /// pattern rows, and workgroup arrays, none of which are resizable at
    /// runtime. The value in this struct is a read-only echo of that
    /// constant; `build_config_for` writes `ENCODED_DIMENSION` into the GPU
    /// config slot regardless and logs a warning if the config value
    /// disagrees. Not mutated by evolution and not exposed in the UI.
    #[serde(alias = "representation_dim")]
    pub representation_dimension: usize,
    /// Base learning rate for association updates.
    pub learning_rate: f32,
    /// Decay rate for unreinforced patterns per tick.
    pub decay_rate: f32,
    /// Exponent for the homeostatic distress curve. Higher = calm longer, panic harder.
    /// Heritable: mutated during breeding, clamped to [1.5, 5.0]. Default 2.0.
    #[serde(default = "default_distress_exponent")]
    pub distress_exponent: f32,
    /// Scales per-dimension variance into attenuation range. Higher = faster boredom.
    /// Heritable: mutated during breeding, clamped to [5.0, 50.0]. Default 20.0.
    #[serde(default = "default_habituation_sensitivity")]
    pub habituation_sensitivity: f32,
    /// Maximum curiosity bonus from sensory monotony. Higher = stronger exploration drive.
    /// Heritable: mutated during breeding, clamped to [0.1, 1.0]. Default 0.6.
    #[serde(default = "default_max_curiosity_bonus")]
    pub max_curiosity_bonus: f32,
    /// Minimum motor output under fatigue. Lower = harsher dampening.
    /// Heritable: mutated during breeding, clamped to [0.05, 0.4]. Default 0.1.
    #[serde(default = "default_fatigue_floor")]
    pub fatigue_floor: f32,
    /// Visual field width in pixels. Default 8. Odd × odd grids (e.g. 17×13)
    /// give the best distal-food visibility — an odd height puts a ray row on
    /// the horizon and an odd width a column straight ahead — but the default
    /// stays 8×6 until the learner can act on directional vision (see
    /// `docs/superpowers/specs/2026-06-10-learning-baseline.md`).
    #[serde(default = "default_vision_width", alias = "vision_w")]
    pub vision_width: u32,
    /// Visual field height in pixels. Default 6. See `vision_width` for the
    /// odd-grid range-visibility note.
    #[serde(default = "default_vision_height", alias = "vision_h")]
    pub vision_height: u32,
    /// Retinotopic luminance grid the visual cortex operates on. Locked per
    /// batch (compile-time `override` into the kernel, like `vision_width`), not
    /// heritable — so the brain-state stride stays uniform across the
    /// population. Optimized: reduced from 32×32 to 24×24 to cut Gabor
    /// convolution cost while maintaining probe margins.
    #[serde(default = "default_retina_width")]
    pub retina_width: usize,
    /// Retinotopic luminance grid height. Locked per batch like
    /// `retina_width`; see that field for the locked-not-heritable rationale.
    /// Optimized: reduced from 32 to 24.
    #[serde(default = "default_retina_height")]
    pub retina_height: usize,
    /// Physics ticks per brain+vision cycle. Higher = faster but less responsive.
    /// Default 10, clamped to `[1, MAX_BRAIN_TICK_STRIDE]` in the UI. Combined
    /// with `vision_stride` this sets the one-batch sensory lag — see
    /// [`BrainConfig::sensory_lag_ticks`].
    #[serde(default = "default_brain_tick_stride")]
    pub brain_tick_stride: u32,
    /// Brain cycles between global passes (grid rebuild, collisions, vision).
    /// Higher = more brain throughput, less frequent vision updates.
    /// Default 10, clamped to `[1, MAX_VISION_STRIDE]` in the UI. Combined with
    /// `brain_tick_stride` this sets the one-batch sensory lag — see
    /// [`BrainConfig::sensory_lag_ticks`].
    #[serde(default = "default_vision_stride")]
    pub vision_stride: u32,
    /// Multiplier for all energy costs (metabolic + movement). Default 0.5.
    /// Lower = agents survive longer. Higher = harsher energy pressure.
    #[serde(default = "default_metabolic_rate")]
    pub metabolic_rate: f32,
    /// Multiplier for integrity damage and regen. Default 0.5.
    /// Lower = agents take less damage. Higher = hazard zones are deadlier.
    #[serde(default = "default_integrity_scale")]
    pub integrity_scale: f32,
    /// Base movement speed (units per second). Default 20.0.
    /// Heritable: mutated during breeding, clamped to [1.0, 100.0].
    #[serde(default = "default_movement_speed")]
    pub movement_speed: f32,
    /// Exponent for speed-cost drag curve in the fused kernel's energy drain.
    /// Default 1.0 (no-op: cost is linear in speed). Values > 1.0 make drag
    /// super-linear above baseline speed. Applied to `pow(max(speed/20, 1.0), k)`.
    /// Locked per batch, not heritable.
    #[serde(default = "default_speed_cost_exponent")]
    pub speed_cost_exponent: f32,
    /// Gate flag for the Hubel-Wiesel visual cortex pass. When
    /// `false` the cortex pass is a no-op passthrough and the encoder consumes
    /// the legacy raw-vision slice, so the run is byte-identical to the
    /// pre-cortex build. The default flips to `true` only once the 0005
    /// orientation-selectivity / phase-position-invariance probes pass and the
    /// throughput regression is within budget. Locked per batch, not heritable.
    #[serde(default)]
    pub visual_cortex_enabled: bool,
    /// Gate flag for the dedicated danger percept sense. When
    /// `false` the danger bearing and distance are not packed into the sensory
    /// feature vector, so the encoder input width and the encoded state match
    /// the pre-percept build (byte-identical). The default flips to `true` only
    /// once the speed-decoupling gate passes. Locked per batch, not heritable.
    #[serde(default)]
    pub danger_percept_enabled: bool,
    /// Measurement-only ablation mask for the danger percept. When `true`, the
    /// danger distance/bearing are still detected
    /// and still feed the geometry-gated avoidance-intent counters, but the
    /// values packed into the brain's encoder feature vector are forced to the
    /// "no danger in range" sentinel (distance 1.0, bearing 0.0) — the brain is
    /// blinded to danger while the counters keep counting. This isolates the
    /// causal contribution of the percept to steering (deliberate vs incidental)
    /// in a seeded A/B where both arms keep `danger_percept_enabled = true` so
    /// the encoder width is identical. Has no effect unless
    /// `danger_percept_enabled` is also `true`. Runtime-only, not heritable,
    /// default `false` (no effect on the shipped path).
    #[serde(default)]
    pub danger_percept_blinded: bool,
    /// Gate flag for effort-rebased fitness. When `false` the
    /// composite fitness uses the legacy time-denominated formula (food per
    /// time, exploration as fraction of cells). When `true` it re-bases both
    /// axes onto effort: foraging = food/energy, exploration = min(coverage,
    /// cells/distance). The default stays `false` until the speed-decoupling
    /// gate passes. Locked per batch, not heritable.
    #[serde(default)]
    pub effort_rebased_fitness: bool,
    /// Gate flag for seeded innate instinct priors. When `false`,
    /// pattern memory initializes to all zeros (blank slate, byte-identical to
    /// pre-instinct behavior). When `true`, danger and food instinct patterns are
    /// seeded at initialization via seed_instinct_patterns(). Locked per batch,
    /// not heritable. Default `false` until the prove-or-kill A/B gate passes.
    #[serde(default)]
    pub innate_instincts_enabled: bool,
    /// Enable the homeostatic gradient predictor: a 128→1 linear head on the
    /// forward model that learns to anticipate raw_gradient. The predicted
    /// gradient provides an anticipatory credit signal (β-scaled term in the
    /// TD reward), bridging sensory latency without external targets.
    /// Weights are heritable. Zero-cost when false. (homeostatic gradient predictor head)
    #[serde(default)]
    pub homeo_predictive_credit_enabled: bool,

    /// Learning rate for the homeostatic gradient predictor's online gradient
    /// descent. Trained on every brain tick against the actual raw_gradient.
    #[serde(default = "default_homeo_predictor_learning_rate")]
    pub homeo_predictor_learning_rate: f32,

    /// Blend weight for the predicted gradient in the TD reward.
    /// reward = raw_gradient_amplified + β * prev_predicted_gradient.
    /// 0.0 = disabled; 0.3 = anticipatory credit at ~30% weight.
    #[serde(default = "default_homeo_predictive_credit_beta")]
    pub homeo_predictive_credit_beta: f32,
    /// **Heritable (visual genome).** V1 Gabor carrier wavelength λ
    /// in retina pixels for the whole simple-cell bank. The envelope σ is tied
    /// as `0.56·λ` (≈ 1-octave V1 bandwidth, Jones & Palmer 1987). Seed 5.0;
    /// mutated during breeding, clamped to
    /// `[GABOR_WAVELENGTH_MIN, GABOR_WAVELENGTH_MAX]` = `[2.0, 12.0]`. The shader
    /// re-imposes the clamp and the Gabor DC-balance invariant after reading it.
    #[serde(default = "default_gabor_wavelength")]
    pub gabor_wavelength: f32,
    /// **Heritable (visual genome).** Gabor envelope aspect ratio γ
    /// (long axis / short axis) for the whole bank; at 1.0 the envelope is
    /// isotropic. Seed 0.5; mutated during breeding, clamped to
    /// `[GABOR_ASPECT_RATIO_MIN, GABOR_ASPECT_RATIO_MAX]` = `[0.25, 1.0]`.
    #[serde(default = "default_gabor_aspect_ratio")]
    pub gabor_aspect_ratio: f32,
    /// **Heritable (visual genome).** DoG surround:center sigma ratio
    /// for the Stage-1 center-surround kernel. Seed 1.6 (Marr & Hildreth 1980
    /// edge operator); mutated during breeding, clamped to
    /// `[DOG_SURROUND_RATIO_MIN, DOG_SURROUND_RATIO_MAX]` = `[1.2, 3.0]` so a
    /// mutated value can never degenerate the kernel into a non-edge blur. The
    /// shader re-imposes the clamp and the DoG zero-sum invariant after reading.
    #[serde(default = "default_dog_surround_ratio")]
    pub dog_surround_ratio: f32,
    /// **Heritable (visual genome).** Whole-bank orientation offset in
    /// radians, added to the even `[0, π)` tiling of the Gabor bank. Seed 0.0;
    /// mutated during breeding and wrapped back into `[0, π)` (orientation is
    /// half-circle periodic for an unsigned bar), so it has no hard clamp — the
    /// wrap is the bound the shader and the mutation path both apply.
    #[serde(default = "default_orientation_offset")]
    pub orientation_offset: f32,
    /// **Heritable (innate instinct gene).** Multiplier for the danger instinct
    /// pattern's negative valence, controlling how strongly aversive the seeded danger
    /// prior is. Seed 0.8; mutated during breeding, clamped to
    /// `[INSTINCT_DANGER_STRENGTH_MIN, INSTINCT_DANGER_STRENGTH_MAX]` = `[0.1, 1.0]`.
    #[serde(default = "default_instinct_danger_strength")]
    pub instinct_danger_strength: f32,
    /// **Heritable (innate instinct gene).** Multiplier for the food instinct
    /// pattern's positive valence, controlling how strongly appetitive the seeded
    /// food/energy-gain prior is. Seed 0.8; mutated during breeding, clamped to
    /// `[INSTINCT_FOOD_STRENGTH_MIN, INSTINCT_FOOD_STRENGTH_MAX]` = `[0.1, 1.0]`.
    #[serde(default = "default_instinct_food_strength")]
    pub instinct_food_strength: f32,
    /// Profiling stage limit for the visual-cortex pass. `0` = run all stages
    /// (default, no short-circuit). `1` = retina fill only. `2` = retina + DoG
    /// center-surround. `3` = all stages (same as `0`). Non-zero values
    /// short-circuit `coop_visual_cortex` after the named stage so wall-clock
    /// timing can isolate per-component cost. Has no effect when
    /// `visual_cortex_enabled` is `false`. Runtime-only, not heritable, default
    /// `0` (no effect on the shipped path). Mirrors `CFG_CORTEX_STAGE_LIMIT` in
    /// `xagent_brain::buffers`.
    #[serde(default)]
    pub cortex_stage_limit: u32,
}

/// Upper bound (exclusive) for `orientation_offset`: π. Orientation is
/// half-circle periodic for an unsigned oriented bar, so the offset wraps modulo
/// π rather than clamping to a hard range. Single canonical source for the wrap
/// used in mutation (`agent/mod.rs`) — the shader applies the same wrap when it
/// reads the gene.
pub const ORIENTATION_OFFSET_PERIOD: f32 = std::f32::consts::PI;

/// Inclusive clamp bounds for the heritable instinct-strength scalar genes. These
/// control the magnitude of the seeded danger and food instinct patterns; they are
/// re-imposed during breeding and in the shader after seeding.
pub const INSTINCT_DANGER_STRENGTH_MIN: f32 = 0.1;
pub const INSTINCT_DANGER_STRENGTH_MAX: f32 = 1.0;
pub const INSTINCT_FOOD_STRENGTH_MIN: f32 = 0.1;
pub const INSTINCT_FOOD_STRENGTH_MAX: f32 = 1.0;

/// Inclusive clamp bounds for the heritable visual-genome scalar genes (plan
/// 0008). These mirror the WGSL `GABOR_*`/`DOG_*` clamp constants in
/// `xagent-brain` `common.wgsl` (the shader re-imposes them after reading the
/// genes); they live here as the canonical source for the mutation clamps so the
/// CPU breeding path and the GPU pass agree.
pub const GABOR_WAVELENGTH_MIN: f32 = 2.0;
/// See [`GABOR_WAVELENGTH_MIN`].
pub const GABOR_WAVELENGTH_MAX: f32 = 12.0;
/// See [`GABOR_WAVELENGTH_MIN`].
pub const GABOR_ASPECT_RATIO_MIN: f32 = 0.25;
/// See [`GABOR_WAVELENGTH_MIN`].
pub const GABOR_ASPECT_RATIO_MAX: f32 = 1.0;
/// See [`GABOR_WAVELENGTH_MIN`].
pub const DOG_SURROUND_RATIO_MIN: f32 = 1.2;
/// See [`GABOR_WAVELENGTH_MIN`].
pub const DOG_SURROUND_RATIO_MAX: f32 = 3.0;

/// Configuration for the world simulation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldConfig {
    /// World size in units (square terrain side length).
    pub world_size: f32,
    /// Energy depletion rate per tick (base metabolic cost).
    pub energy_depletion_rate: f32,
    /// Energy cost per unit of movement.
    pub movement_energy_cost: f32,
    /// Damage per tick in hazard zones.
    pub hazard_damage_rate: f32,
    /// Integrity regeneration per tick (when energy > 50%).
    pub integrity_regen_rate: f32,
    /// Energy restored per food item consumed.
    pub food_energy_value: f32,
    /// Density of food items in food-rich biomes (items per unit²).
    pub food_density: f32,
    /// Simulation ticks per second.
    pub tick_rate: f32,
    /// Random seed for world generation.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_distress_exponent() -> f32 {
    2.0
}

fn default_habituation_sensitivity() -> f32 {
    20.0
}

fn default_max_curiosity_bonus() -> f32 {
    0.6
}

fn default_fatigue_floor() -> f32 {
    0.1
}

fn default_vision_width() -> u32 {
    8
}

fn default_vision_height() -> u32 {
    6
}

/// Optimized: reduced from 32 to 24 to cut Gabor convolution cost while
/// maintaining probe margins (orientation ≥3×, phase <10%, position <15%).
/// The reduction from 32 to 24 shrinks the per-pixel work by (24/32)² = 0.5625×
/// (pixel count drops from 1024 to 576), contributing ~1.5–2× throughput speedup
/// on top of the kernel-radius and pool-size reductions. Trade-off: slightly
/// coarser spatial resolution for the cortex encoder input; the position-tolerance
/// probe confirms that 1-pixel shifts remain below the 15% tolerance on the
/// 24×24 retina. See cortex_throughput_profile_baseline for the measured throughput
/// gain.
fn default_retina_width() -> usize {
    24
}

/// Optimized: reduced from 32 to 24. See `default_retina_width`.
fn default_retina_height() -> usize {
    24
}

fn default_seed() -> u64 {
    42
}

/// Legacy back-compat default for `visual_encoding_size`. Superseded by plan
/// 0008 visual-cortex config; retained only so older saved configs that omit it
/// still deserialize (issue #106).
fn default_visual_encoding_size() -> usize {
    64
}

fn default_brain_tick_stride() -> u32 {
    10
}

fn default_vision_stride() -> u32 {
    10
}

fn default_metabolic_rate() -> f32 {
    0.5
}

fn default_integrity_scale() -> f32 {
    0.5
}

fn default_movement_speed() -> f32 {
    20.0
}

fn default_speed_cost_exponent() -> f32 {
    1.0
}

/// Seed carrier wavelength λ for the Gabor bank. Mirrors
/// `GABOR_WAVELENGTH_SEED` in the brain crate's `gabor` module / `common.wgsl`.
fn default_gabor_wavelength() -> f32 {
    5.0
}

/// Seed envelope aspect ratio γ for the Gabor bank. Mirrors
/// `GABOR_ASPECT_RATIO_SEED`.
fn default_gabor_aspect_ratio() -> f32 {
    0.5
}

/// Seed DoG surround:center sigma ratio. Mirrors
/// `DOG_SURROUND_RATIO_SEED` (Marr & Hildreth 1980).
fn default_dog_surround_ratio() -> f32 {
    1.6
}

/// Seed whole-bank orientation offset in radians. Mirrors
/// `GABOR_ORIENTATION_OFFSET_SEED`.
fn default_orientation_offset() -> f32 {
    0.0
}

fn default_instinct_danger_strength() -> f32 {
    0.8
}

fn default_instinct_food_strength() -> f32 {
    0.8
}

fn default_homeo_predictor_learning_rate() -> f32 {
    0.01
}
fn default_homeo_predictive_credit_beta() -> f32 {
    0.3
}

/// Describes an agent to be spawned into the world.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentDescriptor {
    /// Human-readable name for this agent.
    pub name: String,
    /// Brain configuration.
    pub brain: BrainConfig,
    /// Maximum energy.
    pub max_energy: f32,
    /// Maximum integrity.
    pub max_integrity: f32,
    /// Visual field resolution (width x height).
    pub visual_resolution: (u32, u32),
    /// Field of view in degrees.
    pub fov_degrees: f32,
}

/// Combined configuration for brain + world, used for JSON serialization.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FullConfig {
    #[serde(default)]
    pub brain: BrainConfig,
    #[serde(default)]
    pub world: WorldConfig,
    #[serde(default)]
    pub governor: GovernorConfig,
}

/// Configuration for the evolution governor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GovernorConfig {
    /// Number of agents per generation. The default is sized to the GPU
    /// occupancy knee (see [`default_population_size`]); configs that omit the
    /// field deserialize to that same knee.
    #[serde(default = "default_population_size")]
    pub population_size: usize,
    /// Simulation ticks per generation before evaluation.
    pub tick_budget: u64,
    /// Number of top agents whose configs survive to the next generation.
    pub elitism_count: usize,
    /// Maximum number of generations to run (0 = unlimited).
    pub max_generations: u64,
    /// Consecutive generations of fitness regression before backtracking.
    pub patience: u32,
    /// Base mutation strength (0.1 = ±10%). Scales up with failed attempts.
    #[serde(default = "default_mutation_strength")]
    pub mutation_strength: f32,
    /// How many times each unique config is evaluated per generation (noise reduction).
    #[serde(default = "default_eval_repeats")]
    pub eval_repeats: usize,
    /// Number of independent evolutionary lineages (island model).
    #[serde(default = "default_num_islands")]
    pub num_islands: usize,
    /// Generations between best-config migration across islands.
    #[serde(default = "default_migration_interval")]
    pub migration_interval: u32,
    /// Decay factor for per-island mutation momentum (0.0–1.0).
    /// Higher = longer memory of winning mutation directions.
    #[serde(default = "default_momentum_decay")]
    pub momentum_decay: f32,
}

/// Default population (agents per generation).
///
/// The GPU occupancy knee — where useful throughput (agent-ticks/sec) peaks —
/// is far higher (≈200 on the reference GPU, ~10× the raw GPU throughput of a
/// handful of agents; reproduce with `--bench-agent-sweep`). The default is
/// nonetheless kept small because every agent shares one world: at a large
/// population they compete for the world's finite food supply, which collapses
/// per-capita foraging and multiplies deaths-per-food. Enlarging the world so
/// per-agent food is preserved (`world_size ∝ √population`) removes that
/// competition — per-capita behavior then matches this default — but evaluating
/// more genomes per generation buys no measured fitness gain (the limiter is
/// learner strength, not search breadth) while costing proportionally more wall
/// time per generation. Until genomes are evaluated in independent arenas with a
/// world-size-invariant fitness, a small population is the validated,
/// fastest-per-generation choice.
fn default_population_size() -> usize {
    10
}

fn default_mutation_strength() -> f32 {
    0.1
}

fn default_eval_repeats() -> usize {
    2
}

fn default_num_islands() -> usize {
    // One lineage by default. With a small shared-world population the per-island
    // round-robin split the generation's already-thin foraging signal across
    // separate trees; concentrating it in a single lineage gives selection the
    // full population to compare each generation.
    1
}

fn default_migration_interval() -> u32 {
    5
}

fn default_momentum_decay() -> f32 {
    0.9
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            population_size: default_population_size(),
            tick_budget: 1_000_000,
            elitism_count: 3,
            max_generations: 0,
            patience: 5,
            mutation_strength: 0.1,
            eval_repeats: 2,
            num_islands: 1,
            migration_interval: 5,
            momentum_decay: 0.9,
        }
    }
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            memory_capacity: 128,
            processing_slots: 16,
            visual_encoding_size: 64,
            representation_dimension: 128,
            learning_rate: 0.05,
            decay_rate: 0.001,
            distress_exponent: 2.0,
            habituation_sensitivity: 20.0,
            max_curiosity_bonus: 0.6,
            fatigue_floor: 0.1,
            vision_width: default_vision_width(),
            vision_height: default_vision_height(),
            retina_width: default_retina_width(),
            retina_height: default_retina_height(),
            brain_tick_stride: default_brain_tick_stride(),
            vision_stride: default_vision_stride(),
            metabolic_rate: default_metabolic_rate(),
            integrity_scale: default_integrity_scale(),
            movement_speed: default_movement_speed(),
            speed_cost_exponent: default_speed_cost_exponent(),
            visual_cortex_enabled: false,
            danger_percept_enabled: false,
            danger_percept_blinded: false,
            effort_rebased_fitness: false,
            innate_instincts_enabled: false,
            homeo_predictive_credit_enabled: false,
            homeo_predictor_learning_rate: default_homeo_predictor_learning_rate(),
            homeo_predictive_credit_beta: default_homeo_predictive_credit_beta(),
            gabor_wavelength: default_gabor_wavelength(),
            gabor_aspect_ratio: default_gabor_aspect_ratio(),
            dog_surround_ratio: default_dog_surround_ratio(),
            orientation_offset: default_orientation_offset(),
            instinct_danger_strength: default_instinct_danger_strength(),
            instinct_food_strength: default_instinct_food_strength(),
            cortex_stage_limit: 0,
        }
    }
}

impl BrainConfig {
    /// Maximum `brain_tick_stride` (physics ticks per brain+vision cycle). The
    /// UI `DragValue` clamps to this; configs loaded from JSON or mutated
    /// programmatically are expected to respect it as well.
    pub const MAX_BRAIN_TICK_STRIDE: u32 = 32;

    /// Maximum `vision_stride` (brain cycles between vision passes). The UI
    /// `DragValue` clamps to this.
    pub const MAX_VISION_STRIDE: u32 = 50;

    /// Upper bound on the one-batch sensory lag, in physics ticks.
    ///
    /// # The sensory-lag invariant
    ///
    /// The brain reads vision and proprioception from `sensory_buffer`, which the
    /// global vision pass refreshes *after* each fused-kernel batch completes (see
    /// `kernel_tick.wgsl` and `GpuKernel::dispatch_batch`). One batch covers
    /// `vision_stride * brain_tick_stride` physics ticks, so the brain always acts
    /// on visual state that is exactly one batch — [`sensory_lag_ticks`] ticks —
    /// stale. The lag is intentional and constant across stride settings.
    ///
    /// # Why it is bounded
    ///
    /// Credit assignment pairs a motor command with the gradient that command
    /// produced (CONTRIBUTING.md → State Invariants: temporal alignment). The
    /// larger the lag, the more the visual evidence at decision time
    /// desynchronizes from the action's actual outcome. The credit-assignment
    /// bugs unraveled during the circling investigation (issue #13) were rooted in
    /// exactly this kind of temporal mismatch, so bounding the product acts as a
    /// tripwire: if a future change grows the strides — or makes them dynamic —
    /// past what the design was validated for, the bound trips instead of silently
    /// resurrecting those bugs.
    ///
    /// The bound is the largest lag the UI clamps permit
    /// (`MAX_BRAIN_TICK_STRIDE * MAX_VISION_STRIDE`), so every in-range config is
    /// accepted and anything larger signals a path that bypassed those clamps.
    ///
    /// See `docs/reviews/2026-04-15-gemini-31-pro.md` ("The One-Batch Sensory
    /// Lag") and issue #115.
    ///
    /// [`sensory_lag_ticks`]: BrainConfig::sensory_lag_ticks
    pub const MAX_SENSORY_LAG_TICKS: u32 = Self::MAX_BRAIN_TICK_STRIDE * Self::MAX_VISION_STRIDE;

    /// The one-batch sensory lag for this config, in physics ticks
    /// (`vision_stride * brain_tick_stride`).
    ///
    /// Saturates to [`u32::MAX`] if the product overflows (e.g. a corrupt config),
    /// so a caller comparing against
    /// [`MAX_SENSORY_LAG_TICKS`](Self::MAX_SENSORY_LAG_TICKS) still rejects it
    /// rather than wrapping to a small value. See the
    /// [`MAX_SENSORY_LAG_TICKS`](Self::MAX_SENSORY_LAG_TICKS) docs for the full
    /// invariant and the rationale for the bound.
    pub fn sensory_lag_ticks(&self) -> u32 {
        self.vision_stride.saturating_mul(self.brain_tick_stride)
    }

    /// Minimal capacity — interesting for observing constraints.
    pub fn tiny() -> Self {
        Self {
            memory_capacity: 24,
            processing_slots: 8,
            visual_encoding_size: 32,
            representation_dimension: 128,
            learning_rate: 0.08,
            decay_rate: 0.002,
            distress_exponent: 2.0,
            habituation_sensitivity: 20.0,
            max_curiosity_bonus: 0.6,
            fatigue_floor: 0.1,
            vision_width: 6,
            vision_height: 4,
            retina_width: 16,
            retina_height: 16,
            brain_tick_stride: default_brain_tick_stride(),
            vision_stride: default_vision_stride(),
            metabolic_rate: default_metabolic_rate(),
            integrity_scale: default_integrity_scale(),
            movement_speed: default_movement_speed(),
            speed_cost_exponent: default_speed_cost_exponent(),
            visual_cortex_enabled: false,
            danger_percept_enabled: false,
            danger_percept_blinded: false,
            effort_rebased_fitness: false,
            innate_instincts_enabled: false,
            homeo_predictive_credit_enabled: false,
            homeo_predictor_learning_rate: default_homeo_predictor_learning_rate(),
            homeo_predictive_credit_beta: default_homeo_predictive_credit_beta(),
            gabor_wavelength: default_gabor_wavelength(),
            gabor_aspect_ratio: default_gabor_aspect_ratio(),
            dog_surround_ratio: default_dog_surround_ratio(),
            orientation_offset: default_orientation_offset(),
            instinct_danger_strength: default_instinct_danger_strength(),
            instinct_food_strength: default_instinct_food_strength(),
            cortex_stage_limit: 0,
        }
    }

    /// More capacity — slower emergence but richer behavior.
    pub fn large() -> Self {
        Self {
            memory_capacity: 512,
            processing_slots: 32,
            visual_encoding_size: 128,
            representation_dimension: 128,
            learning_rate: 0.03,
            decay_rate: 0.0005,
            distress_exponent: 2.0,
            habituation_sensitivity: 20.0,
            max_curiosity_bonus: 0.6,
            fatigue_floor: 0.1,
            vision_width: 12,
            vision_height: 8,
            retina_width: 48,
            retina_height: 48,
            brain_tick_stride: default_brain_tick_stride(),
            vision_stride: default_vision_stride(),
            metabolic_rate: default_metabolic_rate(),
            integrity_scale: default_integrity_scale(),
            movement_speed: default_movement_speed(),
            speed_cost_exponent: default_speed_cost_exponent(),
            visual_cortex_enabled: false,
            danger_percept_enabled: false,
            danger_percept_blinded: false,
            effort_rebased_fitness: false,
            innate_instincts_enabled: false,
            homeo_predictive_credit_enabled: false,
            homeo_predictor_learning_rate: default_homeo_predictor_learning_rate(),
            homeo_predictive_credit_beta: default_homeo_predictive_credit_beta(),
            gabor_wavelength: default_gabor_wavelength(),
            gabor_aspect_ratio: default_gabor_aspect_ratio(),
            dog_surround_ratio: default_dog_surround_ratio(),
            orientation_offset: default_orientation_offset(),
            instinct_danger_strength: default_instinct_danger_strength(),
            instinct_food_strength: default_instinct_food_strength(),
            cortex_stage_limit: 0,
        }
    }
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            world_size: 256.0,
            energy_depletion_rate: 0.03,
            movement_energy_cost: 0.005,
            hazard_damage_rate: 1.0,
            integrity_regen_rate: 0.005,
            food_energy_value: 20.0,
            food_density: 0.005,
            tick_rate: 30.0,
            seed: 42,
        }
    }
}

impl WorldConfig {
    /// Lots of food, slow energy drain, mild hazards.
    pub fn easy() -> Self {
        Self {
            energy_depletion_rate: 0.015,
            movement_energy_cost: 0.002,
            hazard_damage_rate: 0.5,
            food_density: 0.005,
            food_energy_value: 30.0,
            ..Self::default()
        }
    }

    /// Scarce food, fast energy drain, deadly hazards.
    pub fn hard() -> Self {
        Self {
            energy_depletion_rate: 0.05,
            movement_energy_cost: 0.01,
            hazard_damage_rate: 2.0,
            food_density: 0.001,
            food_energy_value: 15.0,
            ..Self::default()
        }
    }
}

impl Default for AgentDescriptor {
    fn default() -> Self {
        Self {
            name: "Agent-0".into(),
            brain: BrainConfig::default(),
            max_energy: 100.0,
            max_integrity: 100.0,
            visual_resolution: (8, 6),
            fov_degrees: 90.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brain_config_tuned_defaults() {
        let config = BrainConfig::default();
        assert_eq!(config.brain_tick_stride, 10);
        assert_eq!(config.vision_stride, 10);
        assert_eq!(config.vision_width, 8);
        assert_eq!(config.vision_height, 6);
        assert!((config.metabolic_rate - 0.5).abs() < 1e-6);
        assert!((config.integrity_scale - 0.5).abs() < 1e-6);
        assert!((config.movement_speed - 20.0).abs() < 1e-6);
    }

    #[test]
    fn governor_config_tuned_defaults() {
        let config = GovernorConfig::default();
        assert_eq!(config.tick_budget, 1_000_000);
    }

    #[test]
    fn sensory_lag_is_product_of_strides() {
        let config = BrainConfig {
            brain_tick_stride: 7,
            vision_stride: 9,
            ..BrainConfig::default()
        };
        assert_eq!(config.sensory_lag_ticks(), 63);
    }

    #[test]
    fn default_sensory_lag_is_within_bound() {
        let config = BrainConfig::default();
        // Default 10 * 10 = 100 ticks, well under the bound.
        assert_eq!(config.sensory_lag_ticks(), 100);
        assert!(config.sensory_lag_ticks() <= BrainConfig::MAX_SENSORY_LAG_TICKS);
    }

    #[test]
    fn sensory_lag_bound_tracks_stride_clamps() {
        // These mirror the UI DragValue clamps in ui.rs — keep them in sync.
        assert_eq!(BrainConfig::MAX_BRAIN_TICK_STRIDE, 32);
        assert_eq!(BrainConfig::MAX_VISION_STRIDE, 50);
        assert_eq!(BrainConfig::MAX_SENSORY_LAG_TICKS, 1600);
        // A config at both clamp ceilings hits exactly the bound (inclusive).
        let config = BrainConfig {
            brain_tick_stride: BrainConfig::MAX_BRAIN_TICK_STRIDE,
            vision_stride: BrainConfig::MAX_VISION_STRIDE,
            ..BrainConfig::default()
        };
        assert_eq!(
            config.sensory_lag_ticks(),
            BrainConfig::MAX_SENSORY_LAG_TICKS
        );
    }

    #[test]
    fn legacy_config_without_visual_cortex_fields_still_loads() {
        // A config saved before the visual-cortex config carries `visual_encoding_size` but
        // none of the new `retina_*` / `gabor_*` / visual-cortex fields. The
        // `#[serde(default)]` on each new field (and on the retained legacy
        // `visual_encoding_size`) must supply the seed so the blob still loads.
        // This is the back-compat guarantee for issue #106.
        let legacy_json = r#"{
            "memory_capacity": 128,
            "processing_slots": 16,
            "visual_encoding_size": 96,
            "representation_dimension": 128,
            "learning_rate": 0.05,
            "decay_rate": 0.001
        }"#;

        let config: BrainConfig =
            serde_json::from_str(legacy_json).expect("legacy config must still deserialize");

        // The explicitly-present fields are preserved verbatim.
        assert_eq!(config.memory_capacity, 128);
        assert_eq!(config.processing_slots, 16);
        assert_eq!(config.visual_encoding_size, 96);

        // Every new visual-cortex field falls back to its seed default.
        assert_eq!(config.retina_width, default_retina_width());
        assert_eq!(config.retina_height, default_retina_height());
        assert!(!config.visual_cortex_enabled);
        assert!((config.gabor_wavelength - default_gabor_wavelength()).abs() < 1e-6);
        assert!((config.gabor_aspect_ratio - default_gabor_aspect_ratio()).abs() < 1e-6);
        assert!((config.dog_surround_ratio - default_dog_surround_ratio()).abs() < 1e-6);
        assert!((config.orientation_offset - default_orientation_offset()).abs() < 1e-6);

        // Older configs that also omit `visual_encoding_size` entirely must load
        // too — the retained serde default supplies it.
        let no_legacy_field = r#"{
            "memory_capacity": 128,
            "processing_slots": 16,
            "representation_dimension": 128,
            "learning_rate": 0.05,
            "decay_rate": 0.001
        }"#;
        let config: BrainConfig =
            serde_json::from_str(no_legacy_field).expect("config without the legacy field loads");
        assert_eq!(config.visual_encoding_size, default_visual_encoding_size());
    }

    #[test]
    fn serde_default_off_coverage() {
        // A config blob that omits `danger_percept_enabled` and
        // `effort_rebased_fitness` must deserialize both fields as `false`.
        // This is the back-compat guarantee: new flag fields must default to
        // `false` when absent from a legacy config blob, enforced by
        // `#[serde(default)]` on each field. Dropping either attribute would
        // cause deserialization to fail on a legacy blob and this test to fail.
        let minimal_json = r#"{
            "memory_capacity": 64,
            "processing_slots": 8,
            "representation_dimension": 64,
            "learning_rate": 0.01,
            "decay_rate": 0.001
        }"#;
        let config: BrainConfig = serde_json::from_str(minimal_json)
            .expect("legacy config without flag fields must load");
        assert!(
            !config.danger_percept_enabled,
            "danger_percept_enabled must default to false when the field is absent"
        );
        assert!(
            !config.effort_rebased_fitness,
            "effort_rebased_fitness must default to false when the field is absent"
        );
    }

    #[test]
    fn sensory_lag_saturates_on_overflow() {
        // A corrupt/hand-edited config must not wrap to a small lag and sneak past
        // the bound; the product saturates so the comparison still rejects it.
        let config = BrainConfig {
            brain_tick_stride: u32::MAX,
            vision_stride: 2,
            ..BrainConfig::default()
        };
        assert_eq!(config.sensory_lag_ticks(), u32::MAX);
        assert!(config.sensory_lag_ticks() > BrainConfig::MAX_SENSORY_LAG_TICKS);
    }

    #[test]
    fn speed_cost_exponent_round_trips() {
        // Test that speed_cost_exponent serializes and deserializes correctly,
        // and that the default is 1.0 (no-op).
        let config = BrainConfig::default();
        assert_eq!(config.speed_cost_exponent, 1.0);

        // Test custom values round-trip through JSON.
        let config = BrainConfig {
            speed_cost_exponent: 2.5,
            ..BrainConfig::default()
        };
        let json = serde_json::to_string(&config).expect("config serializes");
        let deserialized: BrainConfig = serde_json::from_str(&json).expect("config deserializes");
        assert_eq!(deserialized.speed_cost_exponent, 2.5);

        // Test that old configs without speed_cost_exponent deserialize with the default.
        let json_without_field = r#"{
            "memory_capacity": 128,
            "processing_slots": 16,
            "visual_encoding_size": 64,
            "representation_dimension": 128,
            "learning_rate": 0.05,
            "decay_rate": 0.001,
            "distress_exponent": 2.0,
            "habituation_sensitivity": 20.0,
            "max_curiosity_bonus": 0.6,
            "fatigue_floor": 0.1,
            "vision_width": 8,
            "vision_height": 6,
            "retina_width": 32,
            "retina_height": 32,
            "brain_tick_stride": 10,
            "vision_stride": 10,
            "metabolic_rate": 0.5,
            "integrity_scale": 0.5,
            "movement_speed": 20.0,
            "visual_cortex_enabled": false,
            "gabor_wavelength": 5.0,
            "gabor_aspect_ratio": 0.5,
            "dog_surround_ratio": 1.6,
            "orientation_offset": 0.0
        }"#;
        let deserialized: BrainConfig = serde_json::from_str(json_without_field)
            .expect("config without speed_cost_exponent loads with default");
        assert_eq!(deserialized.speed_cost_exponent, 1.0);
    }
}
