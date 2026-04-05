@group(0) @binding(0) var<storage, read> food_state: array<f32>;
@group(0) @binding(1) var<storage, read> food_flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> food_grid: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= arrayLength(&food_state) { return; }
    let _f = food_flags[0];
    let _g = atomicLoad(&food_grid[0]);
}
