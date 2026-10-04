// World updates and encoder-weight updates write disjoint state. Workgroup
// zero runs the original world phases; the other groups update one agent's
// encoder weights. The main dispatch has already published adapted features
// and decision credit. A dispatch boundary makes both branches' writes
// visible before vision and the next cycle.
const GLOBAL_WORLD_WORKGROUPS: u32 = 1u;
override GLOBAL_CREDIT_GROUPS_PER_AGENT: u32 =
    (FEATURE_COUNT * ENCODED_DIMENSION + ENCODER_CREDIT_THREADS - 1u)
    / ENCODER_CREDIT_THREADS;

@compute @workgroup_size(ENCODER_CREDIT_THREADS)
fn global_credit_tick(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    if wgid.x < GLOBAL_WORLD_WORKGROUPS {
        global_world_inner(lid.x);
    } else {
        let credit_group = wgid.x - GLOBAL_WORLD_WORKGROUPS;
        let agent_id = credit_group / GLOBAL_CREDIT_GROUPS_PER_AGENT;
        let tile = credit_group % GLOBAL_CREDIT_GROUPS_PER_AGENT;
        phase_encoder_credit(agent_id, tile * ENCODER_CREDIT_THREADS + lid.x);
    }
}
