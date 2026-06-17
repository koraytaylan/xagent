// Plan 0006: copy coop_feature_extract semantics into brain_scratch so tiled
// encode (a separate dispatch) can read features cross-workgroup.
// This shader is concatenated after common.wgsl by the host.

@compute @workgroup_size(256)
fn phase_brain_features(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let agent_id = wgid.x;
    let tid = lid.x;
    let agent_base = agent_id * BRAIN_SCRATCH_STRIDE;
    let s_base = agent_id * SENSORY_STRIDE;
    let vision_count = VISION_COLOR_COUNT + VISION_DEPTH_COUNT;

    // Mirror `coop_feature_extract`'s flag-dependent layout (plan 0008
    // wire-visual-features-into-encoder) so SCRATCH_FEATURES is sized exactly
    // like the fused `s_features`:
    //   flag OFF — copy the raw vision slice into SCRATCH_FEATURES[0 .. vision_count)
    //              and place the non-visual tail after it (legacy width).
    //   flag ON  — SCRATCH_FEATURES is the compact (VISUAL_FEATURE_COUNT +
    //              NON_VISUAL_FEATURE_COUNT) width; copying the raw vision here
    //              would overrun SCRATCH_FEATURES into SCRATCH_ENCODED, so it is
    //              skipped and the non-visual tail starts at VISUAL_FEATURE_COUNT.
    // NOTE: the split/tiled execution path has no cortex stage (the visual cortex
    // ships in the fused `FusedSerial` default per plan 0008 SCOPE), so with the
    // flag ON the leading VISUAL_FEATURE_COUNT slots remain zero here. The
    // flag-on cortex is exercised only on the fused path; this guard exists to
    // keep the split path bounded and layout-consistent, not to replicate the
    // cortex. The flag read is workgroup-uniform.
    let visual_cortex_enabled = bc_f32(CFG_VISUAL_CORTEX_ENABLED) != 0.0;
    if (!visual_cortex_enabled) {
        // All threads cooperatively copy vision color + depth into brain_scratch.
        for (var i = tid; i < vision_count; i += 256u) {
            brain_scratch[agent_base + SCRATCH_FEATURES + i] = sensory_buffer[s_base + i];
        }
    }
    let non_visual_base = select(VISUAL_FEATURE_COUNT, vision_count, !visual_cortex_enabled);

    // Non-visual features (25 values) — thread 0 only.
    if (tid == 0u) {
        var fi = non_visual_base;
        let vel_offset = vision_count;
        let vx = sensory_buffer[s_base + vel_offset];
        let vy = sensory_buffer[s_base + vel_offset + 1u];
        let vz = sensory_buffer[s_base + vel_offset + 2u];
        brain_scratch[agent_base + SCRATCH_FEATURES + fi] = sqrt(vx * vx + vy * vy + vz * vz); fi = fi + 1u;
        let fac_offset = vel_offset + 3u;
        brain_scratch[agent_base + SCRATCH_FEATURES + fi] = sensory_buffer[s_base + fac_offset]; fi = fi + 1u;
        brain_scratch[agent_base + SCRATCH_FEATURES + fi] = sensory_buffer[s_base + fac_offset + 1u]; fi = fi + 1u;
        brain_scratch[agent_base + SCRATCH_FEATURES + fi] = sensory_buffer[s_base + fac_offset + 2u]; fi = fi + 1u;
        let ang_offset = fac_offset + 3u;
        brain_scratch[agent_base + SCRATCH_FEATURES + fi] = sensory_buffer[s_base + ang_offset]; fi = fi + 1u;
        let interoception_base = agent_id * PHYS_STRIDE;
        let current_max_energy = max(physics_state[interoception_base + P_MAX_ENERGY], 1e-6);
        let current_max_integrity = max(physics_state[interoception_base + P_MAX_INTEGRITY], 1e-6);
        let current_energy = physics_state[interoception_base + P_ENERGY];
        let current_integrity = physics_state[interoception_base + P_INTEGRITY];
        brain_scratch[agent_base + SCRATCH_FEATURES + fi] = current_energy / current_max_energy; fi = fi + 1u;
        brain_scratch[agent_base + SCRATCH_FEATURES + fi] = current_integrity / current_max_integrity; fi = fi + 1u;
        brain_scratch[agent_base + SCRATCH_FEATURES + fi] = current_energy - physics_state[interoception_base + P_PREV_ENERGY]; fi = fi + 1u;
        brain_scratch[agent_base + SCRATCH_FEATURES + fi] = current_integrity - physics_state[interoception_base + P_PREV_INTEGRITY]; fi = fi + 1u;
        let touch_offset = ang_offset + 5u;
        for (var t: u32 = 0u; t < 4u; t = t + 1u) {
            let to = touch_offset + t * 4u;
            brain_scratch[agent_base + SCRATCH_FEATURES + fi] = sensory_buffer[s_base + to]; fi = fi + 1u;
            brain_scratch[agent_base + SCRATCH_FEATURES + fi] = sensory_buffer[s_base + to + 1u]; fi = fi + 1u;
            brain_scratch[agent_base + SCRATCH_FEATURES + fi] = sensory_buffer[s_base + to + 2u]; fi = fi + 1u;
            brain_scratch[agent_base + SCRATCH_FEATURES + fi] = sensory_buffer[s_base + to + 3u]; fi = fi + 1u;
        }
    }
}
