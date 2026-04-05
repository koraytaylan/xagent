@group(0) @binding(0) var<storage, read> food_state: array<f32>;
@group(0) @binding(1) var<storage, read> food_flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> food_grid: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main() {}
