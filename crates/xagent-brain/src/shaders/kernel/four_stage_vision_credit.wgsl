// Vision reads fixed FOV/smell genes, not encoder weights. Credit reads the
// adapted feature publication and writes only its owned encoder vectors and
// scalar mirror. Vision can publish directly into sensory_buffer because all
// brain readers finished in the preceding dispatch.
override FOUR_STAGE_CREDIT_GROUPS_PER_AGENT: u32 =
    (FEATURE_COUNT * PACKED_ENCODER_OUTPUT_VECTORS + ENCODER_CREDIT_THREADS - 1u)
    / ENCODER_CREDIT_THREADS;

@compute @workgroup_size(ENCODER_CREDIT_THREADS)
fn four_stage_vision_credit(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let vision_groups = wc_u32(WC_AGENT_COUNT) * VISION_GROUPS_PER_AGENT;
    if group.x < vision_groups {
        four_stage_vision_inner(lid, group);
    } else {
        let credit_group = group.x - vision_groups;
        let agent_id = credit_group / FOUR_STAGE_CREDIT_GROUPS_PER_AGENT;
        let tile = credit_group % FOUR_STAGE_CREDIT_GROUPS_PER_AGENT;
        phase_encoder_credit(agent_id, tile * ENCODER_CREDIT_THREADS + lid.x);
    }
}
