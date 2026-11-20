//! Offline pooling study runner.
//!
//! Default mode runs one seeded fresh-brain free life per seed, logging each
//! brain tick's 32 adapted column means plus the free-life harness fields to
//! `<out-dir>/seed-<n>.xpol` (and the harness tick log beside it), then fits
//! the four input arms and prints the per-seed and pooled table, the sanity
//! lines and the verdict. `--fit-only <dir>` re-fits existing logs.
//!
//! The kernel runs the parallel-tiled schedule, which writes the adapted
//! feature vector to brain scratch every brain cycle, so the column means are
//! the values the brain itself read on that tick.
//!
//! Usage:
//! `pooling_study --out-dir <dir>` or `pooling_study --fit-only <dir>`.

use std::path::{Path, PathBuf};

use rand::{Rng, SeedableRng};
use xagent_brain::buffers::{
    BrainTickProbe, PHYS_STRIDE, P_ALIVE, P_DEATH_COUNT, P_FOOD_COUNT, P_IN_DANGER_BIOME,
    P_MOTOR_TURN_OUT, P_NEAREST_DANGER_BEARING, P_NEAREST_DANGER_DISTANCE, P_NEAREST_FOOD_BEARING,
    P_NEAREST_FOOD_DISTANCE, SCRATCH_FEATURES, TD_TRACE_DECAY,
};
use xagent_brain::{BrainExecutionMode, BrainLayout, GpuKernel};
use xagent_sandbox::agent::{fresh_brain_state, Agent};
use xagent_sandbox::free_life::metrics::{
    AgentAccumulator, FreeLifeSample, MetricsConfig, FOOD_IN_VIEW_DISTANCE, VIEW_HALF_ANGLE,
};
use xagent_sandbox::free_life::tick_log::TickLogWriter;
use xagent_sandbox::pooling_study::{
    fit_log, gate_verdict, read_log, summarize, write_log, AgentFit, ArmSummary, Record, ARM_COUNT,
    CHANNEL_COUNT, COLUMN_COUNT, RAY_COLUMNS, SANITY_ARM_A, SANITY_ARM_B,
};
use xagent_sandbox::world::WorldState;
use xagent_shared::{BrainConfig, FullConfig};

/// World seeds of the study.
const SEEDS: [u64; 4] = [5, 6, 7, 8];
/// Agents per free life.
const AGENT_COUNT: usize = 10;
/// Simulation ticks per free life.
const RUN_TICKS: u64 = 300_000;
/// Arm names in fit order.
const ARM_NAMES: [&str; ARM_COUNT] = [
    "A hemifields",
    "B 32 columns",
    "C credit-trained pooling",
    "D Oja/PCA pooling",
];
/// Colour channels stored per ray in the adapted visual block (RGBA).
const COLOR_STRIDE: usize = 4;
/// Colour channels logged per column (red, green, blue); depth is the fourth.
const LOGGED_COLORS: usize = 3;
/// Absolute miss from a sanity reference that is reported as a sanity failure.
const SANITY_TOLERANCE: f64 = 0.1;
/// Index of arm A (hemifields) in the fit order.
const ARM_A_INDEX: usize = 0;
/// Index of arm B (32 columns) in the fit order.
const ARM_B_INDEX: usize = 1;
/// Index of arm C (credit-trained pooling) in the fit order.
const ARM_C_INDEX: usize = 2;
/// Index of arm D (Oja/PCA pooling) in the fit order.
const ARM_D_INDEX: usize = 3;
/// Physics flags (alive, in danger biome) are stored as 0.0 or 1.0 in `f32`;
/// a value above the midpoint decodes as set, as the free-life harness does.
const FLAG_THRESHOLD: f32 = 0.5;
/// Ticks are split into two 16-bit halves to convert them to `f32` without `as`.
const TICK_HALF_SCALE: f32 = 65_536.0;
/// Bit width of the low tick half.
const TICK_HALF_BITS: u32 = 16;
/// Mask of the low tick half.
const TICK_LOW_MASK: u64 = 0xFFFF;
/// Largest tick an `f32` holds exactly (2^24).
const MAX_EXACT_F32_TICK: u64 = 1 << 24;

type BoxError = Box<dyn std::error::Error>;

fn main() {
    if let Err(err) = run() {
        eprintln!("pooling_study: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), BoxError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [flag, dir] if flag == "--out-dir" => {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir)?;
            let config = FullConfig::default();
            check_vision(&config.brain)?;
            for seed in SEEDS {
                let records = run_seed(&config, seed, &dir)?;
                write_log(&log_path(&dir, seed), &records)?;
                println!("seed {seed}: {} records written", records.len());
            }
            fit_and_report(&dir)
        }
        [flag, dir] if flag == "--fit-only" => fit_and_report(Path::new(dir)),
        _ => Err("usage: pooling_study --out-dir <dir> | --fit-only <dir>".into()),
    }
}

fn log_path(dir: &Path, seed: u64) -> PathBuf {
    dir.join(format!("seed-{seed}.xpol"))
}

/// The column means need the raw 8-column visual block in scratch.
fn check_vision(brain: &BrainConfig) -> Result<(), BoxError> {
    let width = usize::try_from(brain.vision_width)?;
    if width != RAY_COLUMNS {
        return Err(format!("vision width {width} is not {RAY_COLUMNS}").into());
    }
    if brain.visual_cortex_enabled {
        return Err("visual cortex is enabled; the raw visual block is absent".into());
    }
    if brain.vision_stride != 1 {
        return Err(format!("vision stride {} is not 1", brain.vision_stride).into());
    }
    Ok(())
}

/// Build the world and kernel and seed fresh birth brains, as the free-life harness does.
fn start_life(config: &FullConfig, seed: u64) -> Result<GpuKernel, BoxError> {
    if !GpuKernel::is_available() {
        return Err("no GPU adapter".into());
    }
    let brain = &config.brain;
    let mut world_config = config.world.clone();
    world_config.seed = seed;
    let world = WorldState::new(world_config.clone());
    let mut kernel = GpuKernel::new(
        u32::try_from(AGENT_COUNT)?,
        world.food_items.len(),
        brain,
        &world_config,
    );
    kernel.set_execution_mode(BrainExecutionMode::ParallelTiled);
    let food_pos: Vec<(f32, f32, f32)> = world
        .food_items
        .iter()
        .map(|f| (f.position.x, f.position.y, f.position.z))
        .collect();
    let consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
    let timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
    kernel.upload_world(
        &world.terrain.heights,
        &world.biome_map.grid_as_u32(),
        &food_pos,
        &consumed,
        &timers,
    );
    let mut agents = Vec::with_capacity(AGENT_COUNT);
    for i in 0..AGENT_COUNT {
        let id = u32::try_from(i)?;
        let agent = Agent::new(id, world.safe_spawn_position(), id, brain.clone(), 0);
        agents.push((
            agent.body.body.position,
            agent.body.body.internal.max_energy,
            agent.body.body.internal.max_integrity,
            brain.memory_capacity,
            brain.processing_slots,
        ));
    }
    kernel.upload_agents(&agents);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    kernel.reset_agents_seeded(brain, rng.random());
    let state = fresh_brain_state(brain, &mut rng);
    for i in 0..AGENT_COUNT {
        let idx = u32::try_from(i)?;
        kernel.write_agent_state(idx, &state);
        kernel.write_agent_heritable_config(idx, brain);
    }
    kernel.set_brain_tick_probe(true);
    Ok(kernel)
}

fn counter_value(v: f32) -> u64 {
    if v.is_finite() && v > 0.0 {
        // Kernel counters are small non-negative integers held in f32.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let count = v.round() as u64;
        count
    } else {
        0
    }
}

/// One agent's harness sample, field for field as the free-life harness builds it.
fn build_sample(brain_tick: u64, phys: &[f32], probe: &BrainTickProbe) -> FreeLifeSample {
    FreeLifeSample {
        brain_tick,
        alive: phys[P_ALIVE] > FLAG_THRESHOLD,
        meals_total: counter_value(phys[P_FOOD_COUNT]),
        deaths_total: counter_value(phys[P_DEATH_COUNT]),
        on_hazard: phys[P_IN_DANGER_BIOME] > FLAG_THRESHOLD,
        motor_turn: phys[P_MOTOR_TURN_OUT],
        food_distance: phys[P_NEAREST_FOOD_DISTANCE],
        food_bearing: phys[P_NEAREST_FOOD_BEARING],
        hazard_distance: phys[P_NEAREST_DANGER_DISTANCE],
        hazard_bearing: phys[P_NEAREST_DANGER_BEARING],
        encoding: probe.encoding.to_vec(),
        pathway_inputs: probe.pathway_inputs,
        reward: probe.reward,
        value: probe.value,
        td_error: probe.td_error,
        forward_noise: probe.forward_noise,
        turn_noise: probe.turn_noise,
        turn_innovation: probe.turn_innovation,
        encoding_turn_output: probe.encoding_turn_output,
        scent_turn_output: probe.scent_turn_output,
        vision_turn_output: probe.vision_turn_output,
        encoding_forward_output: probe.encoding_forward_output,
    }
}

/// Mean of each column's adapted red, green, blue and depth over the rows,
/// column `c` channel `k` at `c * 4 + k`. Offsets come from `layout`.
fn column_means(
    agent_scratch: &[f32],
    layout: &BrainLayout,
) -> Result<[f32; COLUMN_COUNT], BoxError> {
    let width = usize::try_from(layout.vision_width)?;
    let height = usize::try_from(layout.vision_height)?;
    let rows = f32::from(u16::try_from(height.max(1))?);
    let mut out = [0.0; COLUMN_COUNT];
    for row in 0..height {
        for col in 0..width {
            let ray = row * width + col;
            let base = col * CHANNEL_COUNT;
            for k in 0..LOGGED_COLORS {
                out[base + k] += agent_scratch[SCRATCH_FEATURES + ray * COLOR_STRIDE + k] / rows;
            }
            out[base + LOGGED_COLORS] +=
                agent_scratch[SCRATCH_FEATURES + layout.vision_color_count + ray] / rows;
        }
    }
    Ok(out)
}

/// Harness rule: alive, within the view distance and inside the view cone.
fn food_in_view(sample: &FreeLifeSample) -> bool {
    sample.alive
        && f64::from(sample.food_distance) <= FOOD_IN_VIEW_DISTANCE
        && f64::from(sample.food_bearing).abs() < VIEW_HALF_ANGLE
}

fn index_f32(i: usize) -> f32 {
    f32::from(u16::try_from(i).unwrap_or(u16::MAX))
}

/// Exact `f32` of a tick; ticks above 2^24 are an error.
fn tick_f32(tick: u64) -> Result<f32, BoxError> {
    if tick > MAX_EXACT_F32_TICK {
        return Err(format!("tick {tick} is not exact in f32").into());
    }
    let high = u16::try_from(tick >> TICK_HALF_BITS)?;
    let low = u16::try_from(tick & TICK_LOW_MASK)?;
    Ok(f32::from(high) * TICK_HALF_SCALE + f32::from(low))
}

/// Run one free life and return its pooling records.
fn run_seed(config: &FullConfig, seed: u64, dir: &Path) -> Result<Vec<Record>, BoxError> {
    let mut kernel = start_life(config, seed)?;
    let layout = BrainLayout::from_config(&config.brain);
    let batch = kernel.kernel_batch_size();
    if batch == 0 {
        return Err("kernel batch size is zero".into());
    }
    if RUN_TICKS % u64::from(batch) != 0 {
        return Err(format!("{RUN_TICKS} ticks is not a multiple of batch {batch}").into());
    }
    let batches = RUN_TICKS / u64::from(batch);
    let rows = batches
        .checked_mul(u64::try_from(AGENT_COUNT)?)
        .ok_or("row count overflow")?;
    let mut writer = TickLogWriter::create(&dir.join(format!("seed-{seed}.ticklog")), rows)?;
    let metrics = MetricsConfig {
        trace_decay: f64::from(TD_TRACE_DECAY),
        total_brain_ticks: batches,
    };
    let mut accumulators: Vec<AgentAccumulator> = (0..AGENT_COUNT)
        .map(|_| AgentAccumulator::new(metrics))
        .collect();
    let mut deaths = [0u64; AGENT_COUNT];
    let capacity = usize::try_from(rows)?;
    let mut records = Vec::with_capacity(capacity);
    let mut probes: Vec<BrainTickProbe> = Vec::new();
    let stride = layout.brain_scratch_stride;
    for brain_tick in 0..batches {
        let tick = brain_tick * u64::from(batch);
        let _dispatched = kernel.dispatch_batch(tick, batch);
        probes.clear();
        kernel.read_brain_tick_probes_blocking(&mut probes);
        let scratch = kernel.read_brain_scratch_blocking();
        let state = kernel.read_full_state_blocking();
        if probes.len() != AGENT_COUNT
            || state.len() < AGENT_COUNT * PHYS_STRIDE
            || scratch.len() < AGENT_COUNT * stride
        {
            return Err("readback shorter than the agent count".into());
        }
        for i in 0..AGENT_COUNT {
            let phys = &state[i * PHYS_STRIDE..(i + 1) * PHYS_STRIDE];
            let sample = build_sample(brain_tick, phys, &probes[i]);
            accumulators[i].push(&sample);
            writer.write_sample(u32::try_from(i)?, &sample)?;
            let died = sample.deaths_total > deaths[i];
            deaths[i] = sample.deaths_total;
            records.push(Record {
                columns: column_means(&scratch[i * stride..(i + 1) * stride], &layout)?,
                reward: sample.reward,
                value: sample.value,
                turn_innovation: sample.turn_innovation,
                food_bearing: sample.food_bearing,
                food_in_view: if food_in_view(&sample) { 1.0 } else { 0.0 },
                death: if died { 1.0 } else { 0.0 },
                agent: index_f32(i),
                tick: tick_f32(tick)?,
            });
        }
    }
    writer.finish()?;
    report_harness(seed, accumulators);
    Ok(records)
}

/// Print the harness's whole-life meal and death totals for the seed.
fn report_harness(seed: u64, accumulators: Vec<AgentAccumulator>) {
    let (meals, deaths) = accumulators
        .into_iter()
        .map(AgentAccumulator::finish)
        .fold((0u64, 0u64), |(m, d), a| {
            (m + a.whole.meals, d + a.whole.deaths)
        });
    println!("seed {seed}: harness meals {meals}, deaths {deaths}");
}

fn print_row(label: &str, summaries: &[ArmSummary; ARM_COUNT]) {
    let cells: Vec<String> = summaries
        .iter()
        .map(|s| {
            format!(
                "{:+.3} ± {:.3} (scored {}, excluded {}, >0 {})",
                s.mean, s.standard_error, s.scored, s.excluded, s.above_zero
            )
        })
        .collect();
    println!("| {label} | {} |", cells.join(" | "));
}

fn summaries_of(fits: &[AgentFit]) -> [ArmSummary; ARM_COUNT] {
    std::array::from_fn(|arm| summarize(&fits.iter().map(|f| f.scores[arm]).collect::<Vec<_>>()))
}

fn sanity_line(arm: &str, mean: f64, reference: f64) {
    let miss = (mean - reference).abs();
    let state = if miss > SANITY_TOLERANCE {
        "SANITY FAILURE"
    } else {
        "ok"
    };
    println!("Sanity {arm}: {mean:+.3} vs reference {reference:+.3} (miss {miss:.3}) {state}");
}

fn fit_and_report(dir: &Path) -> Result<(), BoxError> {
    println!("| run | {} |", ARM_NAMES.join(" | "));
    let mut pooled = Vec::new();
    for seed in SEEDS {
        let fits = fit_log(&read_log(&log_path(dir, seed))?);
        print_row(&format!("seed {seed}"), &summaries_of(&fits));
        pooled.extend(fits);
    }
    let all = summaries_of(&pooled);
    print_row("pooled", &all);
    sanity_line("arm B", all[ARM_B_INDEX].mean, SANITY_ARM_B);
    sanity_line("arm A", all[ARM_A_INDEX].mean, SANITY_ARM_A);
    let best = if all[ARM_C_INDEX].mean >= all[ARM_D_INDEX].mean {
        ARM_C_INDEX
    } else {
        ARM_D_INDEX
    };
    println!(
        "Gate: max(C {:+.4}, D {:+.4}) = {:+.4} vs A {:+.4}; best learned arm {}",
        all[ARM_C_INDEX].mean,
        all[ARM_D_INDEX].mean,
        all[best].mean,
        all[ARM_A_INDEX].mean,
        ARM_NAMES[best]
    );
    println!(
        "Verdict: {}",
        gate_verdict(
            all[ARM_A_INDEX].mean,
            all[ARM_C_INDEX].mean,
            all[ARM_D_INDEX].mean
        )
    );
    Ok(())
}
