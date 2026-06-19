//! Headless benchmark runner for measuring raw simulation throughput.
//!
//! Runs the full tick loop via a single GPU fused kernel dispatch without
//! any UI, database, recording, or evolution overhead.

use std::time::Instant;

use xagent_brain::buffers::{PHYS_STRIDE, P_POS_X, P_POS_Y, P_POS_Z};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

use crate::world::WorldState;

/// Benchmark result returned by [`run_bench`].
pub struct BenchResult {
    pub total_ticks: u64,
    pub agent_count: usize,
    pub elapsed_secs: f64,
    pub ticks_per_sec: f64,
    /// Final agent positions for determinism validation.
    pub final_positions: Vec<[f32; 3]>,
}

/// Run a headless benchmark: `total_ticks` simulation ticks with
/// `agent_count` agents. Returns timing statistics.
pub fn run_bench(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) -> BenchResult {
    println!("[bench] Using GpuKernel ({} agents)", agent_count);

    let (mut kernel, _world) = create_kernel(&brain, &world_config, agent_count);

    let start = Instant::now();

    // Single dispatch for all ticks — one `dispatch_ticks` call, so its full
    // batches fuse into chunked submits (≤ MAX_FUSED_BATCHES per submit).
    kernel.dispatch_batch(0, total_ticks as u32);

    // Per-batch throughput probe. Captured before the
    // readback so they reflect only the dispatch path. The GPU-complete column
    // is non-zero only under `XAGENT_PROBE_GPU_WAIT=1`; `XAGENT_SKIP_GLOBAL_VISION=1`
    // skips the global+vision passes (incorrect results, measurement only).
    let probe_batches = kernel.probe_kernel_batches();
    let probe_submits = kernel.probe_submit_count();
    let probe_submit_ns = kernel.probe_submit_return_nanos();
    let probe_complete_ns = kernel.probe_gpu_complete_nanos();
    let per_batch = |total: u64| total.checked_div(probe_batches).unwrap_or(0);
    println!(
        "[BENCH-PROBE] kernel_batches={} submits={} submit_return_ns={} \
         submit_return_ns_per_batch={} gpu_complete_ns={} gpu_complete_ns_per_batch={}",
        probe_batches,
        probe_submits,
        probe_submit_ns,
        per_batch(probe_submit_ns),
        probe_complete_ns,
        per_batch(probe_complete_ns),
    );

    let state = kernel.read_full_state_blocking();

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let ticks_per_sec = total_ticks as f64 / elapsed_secs;

    let final_positions: Vec<[f32; 3]> = (0..agent_count)
        .map(|i| {
            let base = i * PHYS_STRIDE;
            [
                state[base + P_POS_X],
                state[base + P_POS_Y],
                state[base + P_POS_Z],
            ]
        })
        .collect();

    BenchResult {
        total_ticks,
        agent_count,
        elapsed_secs,
        ticks_per_sec,
        final_positions,
    }
}

/// Profile phase costs by running with different phase masks.
/// Prints a breakdown: barriers-only, physics, physics+vision, full.
pub fn run_profile(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) {
    println!(
        "[profile] {} agents, {} ticks — phase cost breakdown:",
        agent_count, total_ticks
    );

    let (mut kernel, _world) = create_kernel(&brain, &world_config, agent_count);

    // mask=0: barriers only (no compute, same barrier structure)
    let t0 = Instant::now();
    kernel.dispatch_batch_masked(0, total_ticks as u32, 0);
    let barriers_only = t0.elapsed().as_secs_f64();

    // mask=1: physics only (barriers + physics compute)
    let (mut kernel, _) = create_kernel(&brain, &world_config, agent_count);
    let t1 = Instant::now();
    kernel.dispatch_batch_masked(0, total_ticks as u32, 1);
    let physics = t1.elapsed().as_secs_f64();

    // mask=3: physics + vision
    let (mut kernel, _) = create_kernel(&brain, &world_config, agent_count);
    let t2 = Instant::now();
    kernel.dispatch_batch_masked(0, total_ticks as u32, 3);
    let phys_vision = t2.elapsed().as_secs_f64();

    // mask=7: full (physics + vision + brain)
    let (mut kernel, _) = create_kernel(&brain, &world_config, agent_count);
    let t3 = Instant::now();
    kernel.dispatch_batch_masked(0, total_ticks as u32, 7);
    let full = t3.elapsed().as_secs_f64();

    println!("  barriers only:     {:.3}s", barriers_only);
    println!(
        "  + physics:         {:.3}s  (physics compute: {:.3}s)",
        physics,
        physics - barriers_only
    );
    println!(
        "  + vision:          {:.3}s  (vision compute:  {:.3}s)",
        phys_vision,
        phys_vision - physics
    );
    println!(
        "  + brain (full):    {:.3}s  (brain compute:   {:.3}s)",
        full,
        full - phys_vision
    );
    println!("  total tps (full):  {:.0}", total_ticks as f64 / full);
}

/// A/B the fused dispatch path's GPU passes to locate the throughput ceiling.
///
/// Runs the same `total_ticks` through `dispatch_ticks` four times — full,
/// `global` skipped, `vision` skipped, both skipped — each on a fresh kernel,
/// and prints achieved tps plus the submit/batch fusion ratio per arm. A large
/// tps jump when only `global` is skipped fingers the single-workgroup `global`
/// pass as the residual ceiling; a jump only when `vision` is
/// skipped points at vision instead; little movement in either means the
/// limiter is elsewhere (CPU submit / queue back-pressure). Skipping passes
/// corrupts results — this is a timing harness only.
pub fn run_phase_ab(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) {
    println!(
        "[phase-ab] {} agents, {} ticks — fused-dispatch pass isolation",
        agent_count, total_ticks
    );

    let arms: [(&str, bool, bool); 4] = [
        ("full (baseline)", false, false),
        ("skip global", true, false),
        ("skip vision", false, true),
        ("skip global+vision", true, true),
    ];

    let mut baseline_tps = 0.0_f64;
    for (i, (label, skip_global, skip_vision)) in arms.iter().enumerate() {
        let (mut kernel, _world) = create_kernel(&brain, &world_config, agent_count);
        kernel.set_probe_pass_skips(*skip_global, *skip_vision);

        let start = Instant::now();
        kernel.dispatch_batch(0, total_ticks as u32);
        // Blocking readback forces all GPU work to complete, so the wall time
        // captures pass execution, not just submit-return.
        let _ = kernel.read_full_state_blocking();
        let secs = start.elapsed().as_secs_f64();
        let tps = total_ticks as f64 / secs;
        let batches = kernel.probe_kernel_batches();
        let submits = kernel.probe_submit_count();

        if i == 0 {
            baseline_tps = tps;
        }
        let delta = if i == 0 || baseline_tps == 0.0 {
            "—".to_string()
        } else {
            format!("{:+.0}% vs baseline", (tps / baseline_tps - 1.0) * 100.0)
        };
        println!(
            "  {label:<20} {tps:>10.0} tps  ({batches:>5} batches / {submits:>4} submits)  {delta}"
        );
    }

    println!("[phase-ab] read: a large +% on 'skip global' ALONE => the single-workgroup");
    println!("           global pass is the residual ceiling.");
}

/// Sweep agent counts to locate the GPU occupancy knee.
///
/// For each `N` in `counts`, run a fixed `total_ticks` through the single fused
/// `dispatch_batch(0, total_ticks)` path on a fresh kernel and print `N`, tps,
/// and agent-ticks/sec (`tps × N` — the useful-work metric for evolution, since
/// every agent in a generation advances in lockstep). After the sweep, flag the
/// `N` that maximizes agent-ticks/sec as the knee: below it the GPU is idle
/// (tps flat while N rises), at it useful throughput saturates, above it each
/// generation's wall time grows for no extra useful work. Read-only
/// measurement; it changes no simulation state. The default `population_size`
/// is sized to the knee this reports on the reference GPU.
pub fn run_agent_sweep(
    brain: BrainConfig,
    world_config: WorldConfig,
    total_ticks: u64,
    counts: &[usize],
) {
    println!(
        "[agent-sweep] {} ticks per N — locating the GPU occupancy knee",
        total_ticks
    );
    println!("  {:>7}  {:>14}  {:>18}", "N", "tps", "agent-ticks/sec");

    let mut knee_n = 0usize;
    let mut knee_atps = 0.0_f64;
    for &n in counts {
        if n == 0 {
            continue;
        }
        let (mut kernel, _world) = create_kernel(&brain, &world_config, n);

        let start = Instant::now();
        // Single fused dispatch for all ticks, then a blocking readback so the
        // wall time captures GPU execution, not just submit-return.
        kernel.dispatch_batch(0, total_ticks as u32);
        let _ = kernel.read_full_state_blocking();
        let secs = start.elapsed().as_secs_f64();

        let tps = total_ticks as f64 / secs;
        let agent_ticks_per_sec = tps * n as f64;
        if agent_ticks_per_sec > knee_atps {
            knee_atps = agent_ticks_per_sec;
            knee_n = n;
        }
        println!("  {n:>7}  {tps:>14.0}  {agent_ticks_per_sec:>18.0}");
    }

    println!("[agent-sweep] occupancy knee: N={knee_n} maximizes agent-ticks/sec ({knee_atps:.0})");
    println!(
        "[agent-sweep] read: tps stays flat across small N (latency-bound, GPU idle); \
         agent-ticks/sec climbs until the knee, then plateaus while per-generation \
         wall time keeps growing. Size the default population to the knee."
    );
}

/// A/B the visual-cortex pass cost: full-pipeline tps with
/// `visual_cortex_enabled` OFF vs. ON, at a fixed retina resolution and the
/// given population, measuring the throughput overhead of the visual-cortex pass.
///
/// Both arms run the same `total_ticks` on a fresh kernel and time wall-clock
/// from the first dispatch to the GPU completing the last tick (a `Maintain::Wait`
/// drain), so the measured tps reflects GPU execution — including the new
/// `coop_visual_cortex` pass when the flag is on — not just submit-return. The
/// flag and the retina dimensions are compile-time WGSL overrides baked at
/// `GpuKernel::new`, so toggling them on the `BrainConfig` per arm rebuilds the
/// pipeline correctly. The OFF arm is byte-identical to the initial fused-kernel build, so
/// its tps is the fused-kernel baseline this regression is measured against.
///
/// The dispatch is chunked into bounded windows drained with a `Maintain::Wait`
/// between chunks. The cortex-ON arm is ~100× slower; submitting all
/// `total_ticks` in one fused batch would queue minutes of GPU work and trip
/// wgpu's submission watchdog (`panic_on_timeout`). Chunking caps each
/// drain to one window's worth of work for both arms identically; the wait
/// time per chunk is real GPU execution and stays inside the timed region, so
/// the tps is faithful.
///
/// Read-only measurement; it changes no persisted state. This benchmark only
/// reports the numbers — the default-flip decision lives in
/// `visual-encoder-default-gate`.
pub fn run_visual_cortex_ab(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) {
    let retina_width = brain.retina_width;
    let retina_height = brain.retina_height;
    println!(
        "[visual-cortex-ab] {} agents, {} ticks, retina {}x{} — visual cortex on/off tps",
        agent_count, total_ticks, retina_width, retina_height
    );
    println!("  {:>22}  {:>14}  {:>22}", "arm", "tps", "vs off");

    let mut off_tps = 0.0_f64;
    for (i, enabled) in [false, true].into_iter().enumerate() {
        let mut arm_brain = brain.clone();
        arm_brain.visual_cortex_enabled = enabled;

        let (mut kernel, _world) = create_kernel(&arm_brain, &world_config, agent_count);

        // Drain in chunks so no single fused submission queues enough GPU work
        // to trip the submission watchdog. `MAX_FUSED_BATCHES` kernel batches is
        // one submit window — the unit `dispatch_batch` already chunks to.
        let chunk_ticks = kernel
            .kernel_batch_size()
            .saturating_mul(xagent_brain::MAX_FUSED_BATCHES)
            .max(1);

        let start = Instant::now();
        let mut tick: u64 = 0;
        while tick < total_ticks {
            let this_chunk = chunk_ticks.min((total_ticks - tick) as u32);
            kernel.dispatch_batch(tick, this_chunk);
            // Force the chunk to complete before queuing the next, so the queued
            // backlog never exceeds one window. The wait is real GPU execution
            // time and is intentionally inside the timed region.
            kernel
                .device()
                .poll(wgpu::Maintain::Wait)
                .panic_on_timeout();
            tick += this_chunk as u64;
        }
        let secs = start.elapsed().as_secs_f64();

        let tps = tick as f64 / secs;
        let label = if enabled {
            "cortex ON"
        } else {
            "cortex OFF (baseline)"
        };
        if i == 0 {
            off_tps = tps;
        }
        let delta = if i == 0 || off_tps == 0.0 {
            "—".to_string()
        } else {
            // Report both the retained fraction (gate metric) and the slowdown
            // factor; at this cost the fraction rounds to ~0% so a bare percent
            // would hide the magnitude.
            format!(
                "{:.2}% kept ({:.0}x slower)",
                tps / off_tps * 100.0,
                off_tps / tps.max(f64::EPSILON)
            )
        };
        println!("  {label:>22}  {tps:>14.0}  {delta:>22}");
    }

    println!(
        "[visual-cortex-ab] read: 'cortex ON' adds the DoG -> Gabor -> complex pass per agent; \
         the % kept is the fraction of the OFF (fused-kernel) tps retained. The default-flip \
         budget lives in the visual-encoder throughput documentation."
    );
}

/// Simulate the real tick loop with accumulator and per-frame dispatch —
/// no rendering. Prints DIAG lines every second and returns the result.
pub fn run_tick_loop_bench(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
    speed_multiplier: f32,
    timeout_secs: f64,
) -> BenchResult {
    const SIM_DT: f64 = 1.0 / 60.0;

    let (mut kernel, _world) = create_kernel(&brain, &world_config, agent_count);

    let start = Instant::now();
    let mut tick: u64 = 0;
    let mut accumulator: f64 = 0.0;
    let mut gpu_tick_budget: u32 = 32;
    let mut last_diag = Instant::now();
    let mut diag_ticks_since: u64 = 0;
    let mut dispatch_count: u64 = 0;

    // Simulate 120fps frame rate (8.33ms per frame)
    let frame_delta_time: f64 = 1.0 / 120.0;

    while tick < total_ticks {
        let wall_elapsed = start.elapsed().as_secs_f64();
        if wall_elapsed > timeout_secs {
            eprintln!(
                "[BENCH] TIMEOUT after {:.1}s — only {}/{} ticks ({:.0} tps)",
                wall_elapsed,
                tick,
                total_ticks,
                tick as f64 / wall_elapsed
            );
            break;
        }

        // Accumulate
        accumulator += frame_delta_time * speed_multiplier as f64;
        let remaining = (total_ticks - tick) as u32;
        let min_dispatch = kernel.brain_tick_stride();
        // Cap must allow at least min_dispatch ticks to accumulate,
        // matching the main loop logic in main.rs.
        let max_accumulator =
            SIM_DT * (speed_multiplier as f64 * 3.0).max(min_dispatch as f64 + 2.0);
        accumulator = accumulator.min(max_accumulator);
        let raw_ticks = ((accumulator / SIM_DT) as u32)
            .min(gpu_tick_budget)
            .min(500)
            .min(remaining);
        let ticks_to_run = if raw_ticks >= min_dispatch {
            raw_ticks
        } else {
            0
        };

        if ticks_to_run > 0 {
            kernel.try_collect_state_snapshot();
            kernel.dispatch_batch(tick, ticks_to_run);

            accumulator -= ticks_to_run as f64 * SIM_DT;
            gpu_tick_budget = (gpu_tick_budget + gpu_tick_budget / 4 + 1).min(64_000);
            tick += ticks_to_run as u64;
            diag_ticks_since += ticks_to_run as u64;
            dispatch_count += 1;
        }

        // DIAG every second
        if last_diag.elapsed().as_secs_f64() >= 1.0 {
            let tps = diag_ticks_since as f64 / last_diag.elapsed().as_secs_f64();
            eprintln!(
                "[BENCH-DIAG] tick={}/{} tps={:.0} budget={} dispatches={} accumulator={:.4}",
                tick, total_ticks, tps, gpu_tick_budget, dispatch_count, accumulator
            );
            diag_ticks_since = 0;
            dispatch_count = 0;
            last_diag = Instant::now();
        }
    }

    // Final readback
    let state = kernel.read_full_state_blocking();
    let elapsed_secs = start.elapsed().as_secs_f64();
    let ticks_per_sec = tick as f64 / elapsed_secs;

    eprintln!(
        "[BENCH] Done: {} ticks in {:.2}s = {:.0} tps",
        tick, elapsed_secs, ticks_per_sec
    );

    let final_positions: Vec<[f32; 3]> = (0..agent_count)
        .map(|i| {
            let base = i * PHYS_STRIDE;
            [
                state[base + P_POS_X],
                state[base + P_POS_Y],
                state[base + P_POS_Z],
            ]
        })
        .collect();

    BenchResult {
        total_ticks: tick,
        agent_count,
        elapsed_secs,
        ticks_per_sec,
        final_positions,
    }
}

fn create_kernel(
    brain: &BrainConfig,
    world_config: &WorldConfig,
    agent_count: usize,
) -> (GpuKernel, WorldState) {
    let world = WorldState::new(world_config.clone());
    let food_count = world.food_items.len();

    let kernel = GpuKernel::new(agent_count as u32, food_count, brain, world_config);

    // Upload world data
    let heights = world.terrain.heights.clone();
    let biomes = world.biome_map.grid_as_u32();
    let food_pos: Vec<(f32, f32, f32)> = world
        .food_items
        .iter()
        .map(|f| (f.position.x, f.position.y, f.position.z))
        .collect();
    let food_consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
    let food_timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
    kernel.upload_world(&heights, &biomes, &food_pos, &food_consumed, &food_timers);

    // Upload agents
    let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = (0..agent_count)
        .map(|_| {
            let pos = world.safe_spawn_position();
            (
                pos,
                100.0,
                100.0,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agent_data);

    (kernel, world)
}
