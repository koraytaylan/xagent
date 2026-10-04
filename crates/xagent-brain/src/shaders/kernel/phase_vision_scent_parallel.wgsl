// Parallel evaluation of food odour contributions with ordered accumulation.
// Every invocation in the selected workgroup must call vision_prepare_scent
// with the same agent_id and group_threads, including for a dead agent.
// group_threads is the actual workgroup width, and tid is the local index.
// The helper reads only storage state published by earlier dispatches. Its
// scratch and result are workgroup memory, synchronized by workgroupBarrier.

// A chunk fits the largest vision workgroup; narrower workgroups stride over
// its food items. Fixed capacity bounds workgroup memory independently of the
// number of food items in the world.
override VISION_PARALLEL_SCENT: bool = false;
const VISION_SCENT_CHUNK_SIZE: u32 = 256u;
// Range membership must be retained independently of the contribution value:
// the serial reference performs an addition only for an in-range nostril.
const VISION_SCENT_LEFT_VALID: u32 = 1u;
const VISION_SCENT_RIGHT_VALID: u32 = 2u;
// Match the non-visual senses' alive guard over the float-backed state slot.
const VISION_SCENT_ALIVE_THRESHOLD: f32 = 0.5;

var<workgroup> vision_scent_contributions: array<vec2<f32>, VISION_SCENT_CHUNK_SIZE>;
var<workgroup> vision_scent_valid: array<u32, VISION_SCENT_CHUNK_SIZE>;
var<workgroup> vision_prepared_scent: vec2<f32>;

fn vision_prepare_scent(agent_id: u32, tid: u32, group_threads: u32) {
    let base = agent_id * PHYS_STRIDE;
    let alive = !(physics_state[base + P_ALIVE] < VISION_SCENT_ALIVE_THRESHOLD);
    let pos = vec3<f32>(
        physics_state[base + P_POS_X],
        physics_state[base + P_POS_Y],
        physics_state[base + P_POS_Z],
    );
    let facing = vec3<f32>(
        physics_state[base + P_FACING_X],
        physics_state[base + P_FACING_Y],
        physics_state[base + P_FACING_Z],
    );
    let strength = clamp(
        brain_state[agent_id * BRAIN_STRIDE + O_SMELL_STRENGTH],
        SMELL_STRENGTH_MIN, SMELL_STRENGTH_MAX);
    let right = vec3<f32>(facing.z, 0.0, -facing.x);
    let nose = pos + facing * NOSTRIL_FORWARD_OFFSET;
    let left_nostril = nose - right * NOSTRIL_SIDE_OFFSET;
    let right_nostril = nose + right * NOSTRIL_SIDE_OFFSET;
    let range_sq = SCENT_RANGE * SCENT_RANGE;
    let food_count = wc_u32(WC_FOOD_COUNT);
    // Only invocation zero accumulates or publishes this private value.
    var concentration = vec2<f32>(0.0, 0.0);

    var chunk_start = 0u;
    while chunk_start < food_count {
        let chunk_count = min(food_count - chunk_start, VISION_SCENT_CHUNK_SIZE);
        for (var slot = tid; slot < chunk_count; slot += max(group_threads, 1u)) {
            let food = chunk_start + slot;
            var contribution = vec2<f32>(0.0, 0.0);
            var valid = 0u;
            if alive && food_flags[food] == 0u {
                let fbase = food * FOOD_STATE_STRIDE;
                let fx = food_state[fbase + FOOD_POSITION_X];
                let fz = food_state[fbase + FOOD_POSITION_Z];
                let left_dx = fx - left_nostril.x;
                let left_dz = fz - left_nostril.z;
                let left_sq = left_dx * left_dx + left_dz * left_dz;
                if left_sq < range_sq {
                    contribution.x = exp(-sqrt(left_sq) / SCENT_DECAY_LENGTH);
                    valid |= VISION_SCENT_LEFT_VALID;
                }
                let right_dx = fx - right_nostril.x;
                let right_dz = fz - right_nostril.z;
                let right_sq = right_dx * right_dx + right_dz * right_dz;
                if right_sq < range_sq {
                    contribution.y = exp(-sqrt(right_sq) / SCENT_DECAY_LENGTH);
                    valid |= VISION_SCENT_RIGHT_VALID;
                }
            }
            vision_scent_contributions[slot] = contribution;
            vision_scent_valid[slot] = valid;
        }
        workgroupBarrier();

        if tid == 0u {
            // Increasing slot within increasing chunk is increasing food ID.
            // Keep both the scalar addition order and the conditional skips.
            for (var slot = 0u; slot < chunk_count; slot++) {
                let valid = vision_scent_valid[slot];
                if (valid & VISION_SCENT_LEFT_VALID) != 0u {
                    concentration.x += vision_scent_contributions[slot].x;
                }
                if (valid & VISION_SCENT_RIGHT_VALID) != 0u {
                    concentration.y += vision_scent_contributions[slot].y;
                }
            }
        }
        // Every slot remains unchanged until invocation zero has read it.
        workgroupBarrier();
        chunk_start += chunk_count;
    }

    if tid == 0u {
        vision_prepared_scent = vec2<f32>(1.0, 1.0) - exp(-strength * concentration);
    }
    workgroupBarrier();
}
