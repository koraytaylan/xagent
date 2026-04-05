@group(0) @binding(0) var<storage, read> agent_phys: array<f32>;
@group(0) @binding(1) var<storage, read> agent_grid: array<u32>;
@group(0) @binding(2) var<storage, read_write> collision_scratch: array<atomic<i32>>;

@compute @workgroup_size(1)
fn main() {}
