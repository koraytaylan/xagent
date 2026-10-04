// Fold the existing 128 danger slots by the original distance/cell ordering.
// Local scan expressions and the first two-phase merge stay unchanged.
fn reduce_danger_detection(tid: u32) {
    var stride = MEMORY_CAP / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            let other = tid + stride;
            if (danger_cell_precedes(s_similarities[other], shared_sort_indices[other],
                                     s_similarities[tid], shared_sort_indices[tid])) {
                s_similarities[tid] = s_similarities[other];
                shared_sort_indices[tid] = shared_sort_indices[other];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
}

// Eat ties follow the original merged-slot order, which can differ from food
// index order. Carry each winner's original slot rank in unused dense scratch.
// Food-sense distance and bearing retain their independent comparisons.
fn reduce_food_detection(tid: u32) {
    var stride = MEMORY_CAP / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            let other = tid + stride;
            let nearer = s_similarities[other] < s_similarities[tid];
            let earlier_equal = s_similarities[other] == s_similarities[tid]
                && s_dense_partials[other] < s_dense_partials[tid];
            if (nearer || earlier_equal) {
                s_similarities[tid] = s_similarities[other];
                shared_sort_indices[tid] = shared_sort_indices[other];
                s_dense_partials[tid] = s_dense_partials[other];
            }
            s_food_dist_sq[tid] = min(s_food_dist_sq[tid], s_food_dist_sq[other]);
            if (food_precedes(s_argmin_val[other], s_argmin_idx[other],
                              s_argmin_val[tid], s_argmin_idx[tid])) {
                s_argmin_val[tid] = s_argmin_val[other];
                s_argmin_idx[tid] = s_argmin_idx[other];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
}
