@group(0) @binding(0) var<storage, read> agent_phys: array<f32>;
@group(0) @binding(1) var<storage, read> heightmap: array<f32>;
@group(0) @binding(2) var<storage, read> biome: array<u32>;
@group(0) @binding(3) var<storage, read> food_state: array<f32>;
@group(0) @binding(4) var<storage, read> food_flags: array<u32>;
@group(0) @binding(5) var<storage, read> food_grid: array<u32>;
@group(0) @binding(6) var<storage, read> agent_grid: array<u32>;
@group(0) @binding(7) var<storage, read_write> sensory: array<f32>;
@group(0) @binding(8) var<uniform> wconfig: array<vec4<f32>, 6>;

@compute @workgroup_size(48)
fn main() {}
