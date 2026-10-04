//! Paired, seeded full-simulation timing of opt-in GPU optimizations.
//!
//! Run with `cargo run --release -p xagent-brain --example vision_performance --
//! --ticks 10000 --agents 10 --seed 42 --width 8 --height 6`. Each arm runs in
//! its own subprocess so pipeline overrides cannot leak between measurements.
//! Add `--repeats 3` to alternate arm order and report median timings, or
//! `--execution split --vision-stride 2` to exercise another dispatch schedule.
//! The scene is synthetic: rolling terrain, hazard bands, food and colliding
//! agents. Both arms advance the same warmup before timing subsequent ticks.
//! Readbacks and pipeline construction are outside the measured interval.
//! The serial arm disables optional vision and brain transforms; the parallel
//! arm inherits selected transforms from the environment and enables ray steps.
//!
//! Matching hashes cover public readback state, not every private GPU buffer.
//! In particular, final vision depth, food flags and the full decision buffer
//! are not exposed by the public API. Use ray-level parity tests for depth.
//! `--precision fp32` reports differing hashes and checks each arm's exact
//! repeatability across multiple repetitions; numerical accuracy and behavior
//! must be checked by the separate
//! GPU validation tests, since hashes cannot measure rounding error.

use std::error::Error;
use std::process::{Command, ExitCode};
use std::time::Instant;

use rand::{Rng, SeedableRng};
use xagent_brain::{BrainExecutionMode, GpuKernel, MAX_FUSED_BATCHES};
use xagent_shared::{BrainConfig, WorldConfig};

/// A short default run suitable for a first paired measurement.
const DEFAULT_TICKS: u32 = 10_000;
/// Small population used by the performance review.
const DEFAULT_AGENTS: u32 = 10;
/// Reproducible starting point for brain and scene randomness.
const DEFAULT_SEED: u64 = 42;
/// Terrain and biome sizes expected by the kernel's world buffers.
const TERRAIN_SIDE: usize = 129;
const BIOME_SIDE: usize = 256;
/// Food rows and columns give 104 items, comparable to the default scene.
const FOOD_COLUMNS: usize = 13;
const FOOD_ROWS: usize = 8;
/// Spacing keeps initial food neighborhoods below grid capacity.
const FOOD_SPACING: f32 = 4.0;
/// Food sphere center above the terrain, matching the shader convention.
const FOOD_HEIGHT: f32 = 0.35;
/// Small placement jitter varies scenes across seeds without dense cells.
const FOOD_JITTER: f32 = 0.5;
/// Midpoint scaling for centering coordinates within a square scene.
const HALF: f32 = 0.5;
/// Agents begin close enough for collisions to exercise stale grid positions.
const AGENT_SPACING: f32 = 1.5;
/// Five agents per row place the default population near the food field.
const AGENT_COLUMNS: u32 = 5;
/// Full initial physiological meters.
const FULL_METER: f32 = 100.0;
/// Initial eye height above the terrain.
const EYE_HEIGHT: f32 = 1.0;
/// Terrain wavelengths and amplitudes provide gentle, non-flat terrain.
const TERRAIN_FREQUENCY: f32 = 0.08;
const TERRAIN_AMPLITUDE: f32 = 0.7;
/// Hazard bands occupy a quarter of each repeating biome strip.
const BIOME_BAND_WIDTH: usize = 32;
const HAZARD_BAND_WIDTH: usize = 8;
/// Numeric value of the danger biome in the shader.
const HAZARD_BIOME: u32 = 2;
/// Warm ten complete batches before starting the timer.
const WARMUP_BATCHES: u32 = 10;
/// The global shader handles one agent per invocation in a 256-wide group.
const MAX_AGENTS: u32 = 256;
/// Bound example allocations for user-supplied vision dimensions.
const MAX_RAYS: u32 = 1_024;
/// Ray coordinates divide by the dimension minus one.
const MIN_VISION_EXTENT: u32 = 2;
/// Physics, brain, patterns, sensory, decision subset and food state hashes.
const CHECKSUM_COMPONENTS: usize = 6;
/// FNV-1a parameters; floats are hashed by their little-endian bit patterns.
const HASH_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const HASH_PRIME: u64 = 0x0100_0000_01b3;

#[derive(Clone)]
struct Options {
    ticks: u32,
    agents: u32,
    seed: u64,
    width: u32,
    height: u32,
    repeats: usize,
    vision_stride: u32,
    execution: BrainExecutionMode,
    allow_rounding: bool,
    arm: Option<String>,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let brain = BrainConfig::default();
        let mut options = Self {
            ticks: DEFAULT_TICKS,
            agents: DEFAULT_AGENTS,
            seed: DEFAULT_SEED,
            width: brain.vision_width,
            height: brain.vision_height,
            repeats: 1,
            vision_stride: brain.vision_stride,
            execution: BrainExecutionMode::FusedSerial,
            allow_rounding: false,
            arm: None,
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(flag) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| format!("Missing value for {flag}"))?;
            match flag.as_str() {
                "--ticks" => options.ticks = value.parse()?,
                "--agents" => options.agents = value.parse()?,
                "--seed" => options.seed = value.parse()?,
                "--width" => options.width = value.parse()?,
                "--height" => options.height = value.parse()?,
                "--repeats" => options.repeats = value.parse()?,
                "--vision-stride" => options.vision_stride = value.parse()?,
                "--precision" => {
                    options.allow_rounding = match value.as_str() {
                        "exact" => false,
                        "fp32" => true,
                        _ => return Err("Precision must be exact or fp32".into()),
                    };
                }
                "--execution" => {
                    options.execution = match value.as_str() {
                        "fused" => BrainExecutionMode::FusedSerial,
                        "split" => BrainExecutionMode::SplitSerial,
                        "parallel-tiled" => BrainExecutionMode::ParallelTiled,
                        _ => return Err("Execution must be fused, split or parallel-tiled".into()),
                    };
                }
                "--arm" if matches!(value.as_str(), "serial" | "parallel") => {
                    options.arm = Some(value)
                }
                _ => return Err(format!("Unknown argument: {flag} {value}").into()),
            }
        }
        let rays = options
            .width
            .checked_mul(options.height)
            .ok_or("Vision dimensions overflow")?;
        if options.ticks == 0
            || options.agents == 0
            || options.agents > MAX_AGENTS
            || options.width < MIN_VISION_EXTENT
            || options.height < MIN_VISION_EXTENT
            || rays > MAX_RAYS
            || options.repeats == 0
            || options.vision_stride == 0
            || options.vision_stride > BrainConfig::MAX_VISION_STRIDE
        {
            return Err(format!(
                "Require positive ticks/repeats, 1..={MAX_AGENTS} agents, vision dimensions >= {MIN_VISION_EXTENT}, \
                 <= {MAX_RAYS} rays and vision stride 1..={}", BrainConfig::MAX_VISION_STRIDE
            ).into());
        }
        Ok(options)
    }

    fn execution_name(&self) -> &'static str {
        match self.execution {
            BrainExecutionMode::FusedSerial => "fused",
            BrainExecutionMode::SplitSerial => "split",
            BrainExecutionMode::ParallelTiled => "parallel-tiled",
        }
    }

    fn child_arguments(&self, arm: &str) -> Vec<String> {
        vec![
            "--ticks".into(),
            self.ticks.to_string(),
            "--agents".into(),
            self.agents.to_string(),
            "--seed".into(),
            self.seed.to_string(),
            "--width".into(),
            self.width.to_string(),
            "--height".into(),
            self.height.to_string(),
            "--vision-stride".into(),
            self.vision_stride.to_string(),
            "--execution".into(),
            self.execution_name().into(),
            "--arm".into(),
            arm.into(),
        ]
    }
}

struct KernelLogger;
static LOGGER: KernelLogger = KernelLogger;

impl log::Log for KernelLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info && metadata.target().starts_with("xagent_brain")
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}", record.args());
        }
    }

    fn flush(&self) {}
}

fn terrain_height(x: f32, z: f32) -> f32 {
    TERRAIN_AMPLITUDE * ((x * TERRAIN_FREQUENCY).sin() + (z * TERRAIN_FREQUENCY).cos())
}

fn initialize_scene(kernel: &mut GpuKernel, brain: &BrainConfig, world: &WorldConfig, agents: u32) {
    let mut random = rand::rngs::SmallRng::seed_from_u64(world.seed);
    let half = world.world_size * HALF;
    let terrain_step = world.world_size / (TERRAIN_SIDE - 1) as f32;
    let heights: Vec<_> = (0..TERRAIN_SIDE * TERRAIN_SIDE)
        .map(|index| {
            terrain_height(
                (index % TERRAIN_SIDE) as f32 * terrain_step - half,
                (index / TERRAIN_SIDE) as f32 * terrain_step - half,
            )
        })
        .collect();
    let biomes: Vec<_> = (0..BIOME_SIDE * BIOME_SIDE)
        .map(|index| {
            if index % BIOME_SIDE % BIOME_BAND_WIDTH < HAZARD_BAND_WIDTH {
                HAZARD_BIOME
            } else {
                0
            }
        })
        .collect();
    let food: Vec<_> = (0..FOOD_COLUMNS * FOOD_ROWS)
        .map(|index| {
            let x = ((index % FOOD_COLUMNS) as f32 - (FOOD_COLUMNS - 1) as f32 * HALF)
                * FOOD_SPACING
                + random.random_range(-FOOD_JITTER..FOOD_JITTER);
            let z = ((index / FOOD_COLUMNS) as f32 - (FOOD_ROWS - 1) as f32 * HALF) * FOOD_SPACING
                + random.random_range(-FOOD_JITTER..FOOD_JITTER);
            (x, terrain_height(x, z) + FOOD_HEIGHT, z)
        })
        .collect();
    kernel.upload_world(
        &heights,
        &biomes,
        &food,
        &vec![false; food.len()],
        &vec![0.0; food.len()],
    );
    let agent_data: Vec<_> = (0..agents)
        .map(|index| {
            let x = (index % AGENT_COLUMNS) as f32 * AGENT_SPACING;
            let z = (index / AGENT_COLUMNS) as f32 * AGENT_SPACING;
            (
                glam::Vec3::new(x, terrain_height(x, z) + EYE_HEIGHT, z),
                FULL_METER,
                FULL_METER,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.reset_agents_seeded(brain, world.seed);
    kernel.upload_agents(&agent_data);
}

fn advance_and_drain(kernel: &mut GpuKernel, start: u64, ticks: u32) {
    let chunk = kernel
        .kernel_batch_size()
        .saturating_mul(MAX_FUSED_BATCHES)
        .max(1);
    let mut completed = 0;
    while completed < ticks {
        let count = (ticks - completed).min(chunk);
        kernel.dispatch_ticks(start + u64::from(completed), count);
        kernel.poll_wait();
        completed += count;
    }
}

struct StateHash {
    value: u64,
    floats: usize,
}

impl StateHash {
    fn new() -> Self {
        Self {
            value: HASH_OFFSET,
            floats: 0,
        }
    }

    fn update(&mut self, values: &[f32]) {
        for value in values {
            for byte in value.to_bits().to_le_bytes() {
                self.value = (self.value ^ u64::from(byte)).wrapping_mul(HASH_PRIME);
            }
        }
        self.floats += values.len();
    }

    fn print(&self, name: &str) {
        println!("CHECKSUM {name} {:016x} {}", self.value, self.floats);
    }
}

fn print_state_hashes(kernel: &mut GpuKernel) -> Result<(), Box<dyn Error>> {
    let mut physics = StateHash::new();
    physics.update(kernel.read_full_state_blocking());
    physics.print("physics");
    let mut brain = StateHash::new();
    let mut patterns = StateHash::new();
    let mut senses = StateHash::new();
    let mut decision = StateHash::new();
    for agent in 0..kernel.agent_count() {
        let state = kernel.read_agent_state(agent);
        brain.update(&state.brain_state);
        patterns.update(&state.patterns);
        let telemetry = kernel.read_agent_telemetry_blocking(agent);
        senses.update(&telemetry.vision_color);
        senses.update(&telemetry.sensory_non_visual);
        decision.update(&[
            telemetry.motor_fwd,
            telemetry.motor_turn,
            telemetry.td_error,
        ]);
    }
    brain.print("brain");
    patterns.print("patterns");
    senses.print("sensory_without_depth");
    decision.print("decision_forward_turn_td_error");
    if !kernel.request_state_snapshot() {
        return Err("Final snapshot was not accepted".into());
    }
    kernel.poll_wait();
    if !kernel.try_collect_state_snapshot() {
        return Err("Final snapshot was not collected".into());
    }
    let mut food = StateHash::new();
    food.update(
        kernel
            .cached_food_state()
            .ok_or("Final food state missing")?,
    );
    food.print("food_positions_timers");
    Ok(())
}

fn run_arm(options: &Options, arm: &str) -> Result<(), Box<dyn Error>> {
    if arm == "serial" {
        std::env::set_var("XAGENT_VISION_OBJECT_QUERIES", "0");
        std::env::set_var("XAGENT_VISION_PARALLEL_SCENT", "0");
        std::env::set_var("XAGENT_BRAIN_COOPERATIVE_WHITENING", "0");
        std::env::set_var("XAGENT_BRAIN_FUSED_PREDICTOR", "0");
        std::env::set_var("XAGENT_BRAIN_DENSE_PREFETCH", "0");
        std::env::set_var("XAGENT_BRAIN_GLOBAL_CREDIT", "0");
        std::env::set_var("XAGENT_BRAIN_PREDICTOR_LANES", "4");
    }
    // Set before creating any GPU device or worker thread.
    std::env::set_var(
        "XAGENT_VISION_PARALLEL_STEPS",
        if arm == "parallel" { "1" } else { "0" },
    );
    log::set_logger(&LOGGER).map_err(|error| error.to_string())?;
    log::set_max_level(log::LevelFilter::Info);
    let brain = BrainConfig {
        vision_width: options.width,
        vision_height: options.height,
        vision_stride: options.vision_stride,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: options.seed,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(options.agents, FOOD_COLUMNS * FOOD_ROWS, &brain, &world);
    kernel.set_execution_mode(options.execution);
    initialize_scene(&mut kernel, &brain, &world, options.agents);
    let warmup = kernel
        .kernel_batch_size()
        .checked_mul(WARMUP_BATCHES)
        .ok_or("Warmup overflow")?;
    advance_and_drain(&mut kernel, 0, warmup);
    println!("SCENE synthetic agents={} seed={} vision={}x{} warmup_ticks={warmup} timed_ticks={} execution={} vision_stride={} brain_beside_vision_requested={}",
        options.agents, options.seed, options.width, options.height, options.ticks,
        options.execution_name(), options.vision_stride,
        std::env::var("XAGENT_BRAIN_BESIDE_VISION").unwrap_or_else(|_| "default".into()));
    let start = Instant::now();
    advance_and_drain(&mut kernel, u64::from(warmup), options.ticks);
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "RESULT arm={arm} elapsed_secs={seconds:.9} ticks_per_sec={:.3}",
        f64::from(options.ticks) / seconds
    );
    print_state_hashes(&mut kernel)?;
    println!("UNCOVERED final_vision_depth food_consumed_flags food_claims full_decision_buffer private_scratch");
    Ok(())
}

struct ArmResult {
    seconds: f64,
    hashes: Vec<String>,
}

fn child_result(options: &Options, arm: &str) -> Result<ArmResult, Box<dyn Error>> {
    let output = Command::new(std::env::current_exe()?)
        .args(options.child_arguments(arm))
        .env_remove("XAGENT_SKIP_GLOBAL")
        .env_remove("XAGENT_SKIP_VISION")
        .env_remove("XAGENT_SKIP_GLOBAL_VISION")
        .env_remove("XAGENT_KERNEL_PASS_LIMIT")
        .output()?;
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8(output.stdout)?;
    print!("{stdout}");
    if !output.status.success() {
        return Err(format!("{arm} arm failed: {}", output.status).into());
    }
    let result = stdout
        .lines()
        .find(|line| line.starts_with("RESULT "))
        .ok_or("Missing child timing")?;
    let seconds: f64 = result
        .split_whitespace()
        .find_map(|field| field.strip_prefix("elapsed_secs="))
        .ok_or("Missing child elapsed time")?
        .parse()?;
    let hashes = stdout
        .lines()
        .filter(|line| line.starts_with("CHECKSUM "))
        .map(str::to_owned)
        .collect();
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("Child reported an invalid elapsed time".into());
    }
    Ok(ArmResult { seconds, hashes })
}

fn run_pair(
    options: &Options,
    repetition: usize,
) -> Result<(ArmResult, ArmResult), Box<dyn Error>> {
    let arms = if repetition & 1 == 0 {
        ["serial", "parallel"]
    } else {
        ["parallel", "serial"]
    };
    println!("PAIR repetition={} first_arm={}", repetition + 1, arms[0]);
    let first = child_result(options, arms[0])?;
    let second = child_result(options, arms[1])?;
    if first.hashes.len() != CHECKSUM_COMPONENTS || second.hashes.len() != CHECKSUM_COMPONENTS {
        return Err("Missing public-state checksum component".into());
    }
    let hashes_match = first.hashes == second.hashes;
    if !hashes_match && !options.allow_rounding {
        return Err(format!(
            "Serial/parallel state hashes differ in pair {}; inspect CHECKSUM rows above",
            repetition + 1
        )
        .into());
    }
    let (serial, parallel) = if arms[0] == "serial" {
        (first, second)
    } else {
        (second, first)
    };
    println!(
        "PAIR_RESULT repetition={} public_state_hashes={} full_simulation_speedup={:.3}x",
        repetition + 1,
        if hashes_match { "match" } else { "differ" },
        serial.seconds / parallel.seconds
    );
    Ok((serial, parallel))
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() >> 1;
    if values.len() & 1 == 0 {
        (values[middle - 1] + values[middle]) * f64::from(HALF)
    } else {
        values[middle]
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    if let Some(arm) = &options.arm {
        return run_arm(&options, arm);
    }
    let mut serial_times = Vec::new();
    let mut parallel_times = Vec::new();
    let mut reference_hashes = None;
    let mut candidate_hashes = None;
    let mut all_hashes_match = true;
    for repetition in 0..options.repeats {
        let (serial, parallel) = run_pair(&options, repetition)?;
        all_hashes_match &= serial.hashes == parallel.hashes;
        if let Some(reference) = &reference_hashes {
            if &serial.hashes != reference {
                return Err("Seeded state hashes changed across repetitions".into());
            }
        } else {
            reference_hashes = Some(serial.hashes);
        }
        if let Some(reference) = &candidate_hashes {
            if &parallel.hashes != reference {
                return Err("Candidate state hashes changed across repetitions".into());
            }
        } else {
            candidate_hashes = Some(parallel.hashes);
        }
        serial_times.push(serial.seconds);
        parallel_times.push(parallel.seconds);
    }
    let serial_median = median(&mut serial_times);
    let parallel_median = median(&mut parallel_times);
    println!(
        "PAIRED repeats={} public_state_hashes={} precision={} repeatability={} serial_median_secs={serial_median:.9} \
         parallel_median_secs={parallel_median:.9} full_simulation_speedup={:.3}x synthetic_scene=true",
        options.repeats,
        if all_hashes_match { "match" } else { "differ" },
        if options.allow_rounding { "fp32_diagnostics_separate" } else { "exact_hashes" },
        if options.repeats > 1 { "per_arm_hashes_match" } else { "not_checked" },
        serial_median / parallel_median
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("vision_performance: {error}");
            ExitCode::FAILURE
        }
    }
}
