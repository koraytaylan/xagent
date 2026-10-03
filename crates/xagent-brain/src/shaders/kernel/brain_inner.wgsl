// ── Brain cycle shared by the fused kernel and the brain-and-vision entry ──
// `brain_tick_inner` runs the seven cooperative brain passes for one agent.
// The fused kernel (kernel_tick.wgsl) calls it after physics, food and danger
// detection and death/respawn; with vision_stride 1 the fused path instead
// runs it from brain_vision_tick.wgsl, beside the vision workgroups, after the
// global pass. Its push constant and the alive broadcast are declared here so
// both modules share them.

// Workgroup-uniform alive broadcast. Written by thread 0 immediately before a
// `workgroupBarrier()`, read by all threads after that barrier. Encoded as u32
// (1 = alive, 0 = dead) so no atomics are needed. See the SAFETY INVARIANT at
// the top of kernel_tick.wgsl.
var<workgroup> s_alive: u32;

// `start_tick` arrives per-batch via a push constant so that multiple
// kernel-batches can share ONE `world_config` uniform write and ONE submit
// (`vision_stride` / `brain_tick_stride` are constant across full batches, so
// the uniform no longer needs to be rewritten per batch just to carry the
// tick). The exact `u32` is strictly more precise than the former
// `WC_TICK = (tick as f32)` round-trip and matches it for every tick ≤ 2^24.
// `pass_limit` is the measurement-only per-cooperative-pass cap:
// `brain_tick_inner` runs only the first `pass_limit` of its seven
// cooperative passes so their cumulative GPU cost can be profiled pass-by-pass.
// It reuses the formerly-unused second push-constant word, so no uniform-slot
// or `WORLD_CONFIG_SIZE` change is needed. The host sets it from
// `XAGENT_KERNEL_PASS_LIMIT` (default 7 = all passes ⇒ byte-identical results;
// the determinism tests gate this). It is a push constant, hence uniform across
// the whole dispatch — see the gating in `brain_tick_inner`.
struct KernelPushConstants {
    start_tick: u32,
    pass_limit: u32,
}
var<push_constant> kpc: KernelPushConstants;

fn brain_tick_inner(agent_id: u32, tid: u32 /* KERNEL_SUBGROUP_TOPK_PARAMS */) {
    // Read the workgroup-uniform alive flag (broadcast by thread 0 before the
    // preceding workgroupBarrier()). Because `s_alive` is identical across the
    // workgroup by construction, cooperative passes with their own internal
    // barriers (`coop_recall_topk`, `coop_predict_and_act`,
    // `coop_learn_and_store`) are safe: all 256 threads either enter together
    // (hitting every internal barrier) or skip together. Inter-pass barriers
    // live outside the guards so they execute regardless of logical state.
    // See top-of-file SAFETY INVARIANT.
    let alive = s_alive != 0u;

    // Measurement-only per-pass cap: run only the first
    // `limit` cooperative passes so their cumulative GPU cost can be profiled
    // pass-by-pass (sweep `XAGENT_KERNEL_PASS_LIMIT = 0..7`; consecutive deltas
    // are the per-pass costs). `limit` is the kernel push constant, so it is
    // uniform across the whole dispatch; `alive` is the broadcast `s_alive`, so
    // `alive && (idx < limit)` is workgroup-uniform and every gated pass is
    // reached together by all 256 threads — exactly like the bare `alive` guard.
    // The barriers below stay UNCONDITIONAL, so barrier uniformity (the
    // top-of-file SAFETY INVARIANT) holds whether a pass runs or is skipped: a
    // skipped pass is skipped *with* all threads, never some. Default 7 runs all
    // passes ⇒ byte-identical to a build without this knob (the determinism
    // tests gate that). Setting it < 7 deliberately produces wrong results and
    // is never on in tests or release.
    let limit = kpc.pass_limit;

    if (alive && 0u < limit) { coop_feature_extract(agent_id, tid); }
    workgroupBarrier();

    // Visual cortex: inserted between feature extraction and encode.
    // It belongs to the early-visual stage, so it shares feature-extract's
    // profiling slot (`0u < limit`) rather than consuming a new `pass_limit`
    // index — keeping the seven counted passes (0..6) and the default limit of 7
    // unchanged. With `CFG_VISUAL_CORTEX_ENABLED` off (or in the passthrough
    // skeleton) it is a no-op, so the encoded state stays byte-identical to the
    // pre-task build. The guard is workgroup-uniform and precedes the barrier.
    if (alive && 0u < limit) { coop_visual_cortex(agent_id, tid); }
    workgroupBarrier();

    // Sensory adaptation shares the early-visual profiling slot as well.
    if (alive && 0u < limit) { coop_sensory_adapt(agent_id, tid); }
    workgroupBarrier();

    if (alive && 1u < limit) { coop_encode(agent_id, tid); }
    workgroupBarrier();

    if (alive && 2u < limit) { coop_habituate_homeo(agent_id, tid); }
    storageBarrier(); workgroupBarrier();

    if (alive && 3u < limit) { coop_recall_score(agent_id, tid); }
    workgroupBarrier();

    if (alive && 4u < limit) { coop_recall_topk(agent_id, tid /* KERNEL_SUBGROUP_TOPK_ARGS */); }
    storageBarrier(); workgroupBarrier();

    if (alive && 5u < limit) { coop_predict_and_act(agent_id, tid, false); }
    storageBarrier(); workgroupBarrier();

    if (alive && 6u < limit) { coop_learn_and_store(agent_id, tid, true); }
}
