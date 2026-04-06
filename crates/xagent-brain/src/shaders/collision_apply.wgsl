@group(0) @binding(0) var<storage, read_write> agent_phys: array<f32>;
@group(0) @binding(1) var<storage, read_write> scratch: array<atomic<i32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent = gid.x;
    if agent >= AGENT_COUNT_VAL { return; }

    // Read and clear scratch atomically (keeps scratch clean even for dead agents)
    let ipx = atomicExchange(&scratch[agent * 3u],      0);
    let ipy = atomicExchange(&scratch[agent * 3u + 1u], 0);
    let ipz = atomicExchange(&scratch[agent * 3u + 2u], 0);

    let b = agent * PHYS_STRIDE;

    // Skip dead agents (but scratch was already cleared above)
    if agent_phys[b + P_ALIVE] < 0.5 { return; }

    // Apply accumulated push
    agent_phys[b + P_POS_X] += f32(ipx) / COLLISION_FIXED_SCALE;
    agent_phys[b + P_POS_Y] += f32(ipy) / COLLISION_FIXED_SCALE;
    agent_phys[b + P_POS_Z] += f32(ipz) / COLLISION_FIXED_SCALE;
}
