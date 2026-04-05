@group(0) @binding(0) var<storage, read_write> food_state: array<f32>;
@group(0) @binding(1) var<storage, read_write> food_flags: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> food_grid: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> heightmap: array<f32>;
@group(0) @binding(4) var<storage, read> biome: array<u32>;
@group(0) @binding(5) var<uniform> wconfig: array<vec4<f32>, 6>;

@compute @workgroup_size(1)
fn main() {}
