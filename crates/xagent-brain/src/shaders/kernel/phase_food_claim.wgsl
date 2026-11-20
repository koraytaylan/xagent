// Phase fragment: food claims, shared by the kernel and the physics-only path.
// (Not in common.wgsl: the vision module declares food_flags non-atomic.)

// Settle this agent's food claim. In the claim step each agent in reach of a
// food item records itself with atomicMin on the item's claim slot; once
// every agent has claimed (a dispatch or barrier later), the lowest-index
// claimant eats the item and the others go without. Who eats therefore
// depends only on the agents, not on which workgroup or thread ran first.
// Single-threaded per agent.
fn resolve_food_claim(agent: u32, tick: u32) {
    let b = agent * PHYS_STRIDE;
    let claim = physics_state[b + P_FOOD_CLAIM];
    if (claim < 0.5) { return; }
    physics_state[b + P_FOOD_CLAIM] = 0.0;
    let item = u32(claim) - 1u;
    if (atomicLoad(&food_flags[food_claim_slot(item)]) == agent) {
        atomicStore(&food_flags[item], 1u);
        if (food_item_is_nourishing(item)) {
            physics_state[b + P_ENERGY] += wc_f32(WC_FOOD_ENERGY);
            physics_state[b + P_FOOD_COUNT] += 1.0;
        } else {
            // A harmful meal gives no energy; it costs integrity, and death
            // at zero integrity is marked in this same scope.
            let max_integrity = physics_state[b + P_MAX_INTEGRITY];
            let loss = wc_f32(WC_FOOD_ENERGY) / max(physics_state[b + P_MAX_ENERGY], EPSILON)
                * HARMFUL_MEAL_INTEGRITY_PER_ENERGY * max_integrity;
            let integrity = clamp(physics_state[b + P_INTEGRITY] - loss, 0.0, max_integrity);
            physics_state[b + P_INTEGRITY] = integrity;
            if (integrity <= 0.0) {
                physics_state[b + P_ALIVE] = 0.0;
                physics_state[b + P_DIED_FLAG] = 1.0;
                physics_state[b + P_LAST_DEATH_TICK] = f32(tick);
            }
            physics_state[b + P_HARMFUL_MEAL_COUNT] += 1.0;
        }
    }
}

// Record this agent's claim on food item `item` (see resolve_food_claim).
fn claim_food(agent: u32, item: u32) {
    atomicMin(&food_flags[food_claim_slot(item)], agent);
    physics_state[agent * PHYS_STRIDE + P_FOOD_CLAIM] = f32(item + 1u);
}
