// Entry-point builtins are private to each invocation, so nested reduction
// helpers need no additional arguments through the cooperative brain passes.
var<private> reduction_subgroup: vec3<u32>;

// Each subgroup reduces its portion of the existing 128-value vector in
// FP32. One invocation folds subgroup totals in ascending subgroup order.
// The upper half of dense scratch is outside the original reduction input.
fn wg_reduce_dense(tid: u32) {
    var value: f32 = 0.0;
    if (tid < ENCODED_DIMENSION) {
        value = s_dense_partials[tid];
    }
    let subtotal = subgroupAdd(value);
    if (reduction_subgroup.x == 0u) {
        s_dense_partials[ENCODED_DIMENSION + reduction_subgroup.y] = subtotal;
    }
    workgroupBarrier();
    if (tid == 0u) {
        var total: f32 = 0.0;
        for (var group = 0u; group < reduction_subgroup.z; group++) {
            total += s_dense_partials[ENCODED_DIMENSION + group];
        }
        s_dense_partials[0u] = total;
    }
    workgroupBarrier();
}
