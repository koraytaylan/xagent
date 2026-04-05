@group(0) @binding(0) var<storage, read_write> food_state: array<f32>;
@group(0) @binding(1) var<storage, read_write> food_flags: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> food_grid: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> heightmap: array<f32>;
@group(0) @binding(4) var<storage, read> biome: array<u32>;
@group(0) @binding(5) var<uniform> wconfig: array<vec4<f32>, 6>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= arrayLength(&food_state) { return; }
    let _ff = atomicLoad(&food_flags[0]);
    let _fg = atomicLoad(&food_grid[0]);
    let _h = heightmap[0];
    let _b = biome[0];
    let _wc = wconfig[0];
}
