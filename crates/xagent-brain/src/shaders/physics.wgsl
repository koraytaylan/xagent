@group(0) @binding(0) var<storage, read_write> agent_phys: array<f32>;
@group(0) @binding(1) var<storage, read> decision: array<f32>;
@group(0) @binding(2) var<storage, read> heightmap: array<f32>;
@group(0) @binding(3) var<storage, read> biome: array<u32>;
@group(0) @binding(4) var<uniform> wconfig: array<vec4<f32>, 6>;

@compute @workgroup_size(1)
fn main() {}
