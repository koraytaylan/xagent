@group(0) @binding(0) var<storage, read> agent_phys: array<f32>;
@group(0) @binding(1) var<storage, read_write> agent_grid: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main() {}
