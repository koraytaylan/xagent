// Test-only no-op cache. The half-gap certificate assumes round-to-nearest;
// full-state parity validates that assumption for the observed compilation.
// WGSL does not provide a universal round-to-nearest mode for ordinary ops.
const NOOP_BLOCK_WIDTH: u32 = 32u;
const NOOP_MAGNITUDE_MASK: u32 = 0x7fffffffu;
const NOOP_MANTISSA_MASK: u32 = 0x007fffffu;
const NOOP_EXPONENT_SHIFT: u32 = 23u;
const NOOP_MIN_NORMAL_BITS: u32 = 0x00800000u;
const NOOP_INFINITY_BITS: u32 = 0x7f800000u;
const NOOP_MAX_FINITE_BITS: u32 = 0x7f7fffffu;
const NOOP_MAX_WEIGHT_BITS: u32 = 0x40000000u; // +2.0, encoder clamp endpoint.
// Half a normal's spacing has exponent field e-24, or e-25 at a power of two
// because the neighbor toward zero is in the smaller binade. Tiny half gaps
// that would be subnormal are deliberately ineligible.
const NOOP_HALF_GAP_EXPONENT_OFFSET: u32 = 24u;
const NOOP_MIN_WEIGHT_EXPONENT: u32 = 25u;
// 1 + 2^-15 exceeds the short product chain's roundoff allowance by a wide
// margin. The final bit increment rounds the positive bound upward again.
const NOOP_PRODUCT_INFLATION: f32 = 1.000030517578125;
// Four factors bounded by 2^16 keep every multiplication order below 2^64,
// leaving ample normal-range headroom for the inflation and upward rounding.
// Larger inputs retain the ordinary production update without certification.
const NOOP_MAX_FACTOR: f32 = 65536.0;
const NOOP_CACHE_RECORD_WORDS: u32 = 2u; // Gap lower bound and last skip count.
override NOOP_BLOCKS_PER_AGENT: u32 = FEATURE_COUNT * ENCODED_DIMENSION / NOOP_BLOCK_WIDTH;
var<workgroup> noop_refresh: array<vec2<u32>, ENCODER_CREDIT_THREADS>;

fn noop_normal_or_zero(value: f32) -> bool {
    let magnitude = bitcast<u32>(value) & NOOP_MAGNITUDE_MASK;
    return magnitude == 0u || (magnitude >= NOOP_MIN_NORMAL_BITS && magnitude < NOOP_INFINITY_BITS);
}

fn noop_normal_positive(value: f32) -> bool {
    let bits = bitcast<u32>(value);
    return bits >= NOOP_MIN_NORMAL_BITS && bits < NOOP_INFINITY_BITS;
}

// Normal input/intermediate guards exclude flush-to-zero amplification. The
// factor cap prevents overflow before multiplication. Unsafe cases train.
// The same upper bound covers a rounded product and an FMA's exact product.
fn noop_update_bound(learning_rate: f32, credit: f32, feature: f32) -> f32 {
    let unsafe_bound = bitcast<f32>(NOOP_MAX_FINITE_BITS);
    if !noop_normal_or_zero(learning_rate) || !noop_normal_or_zero(credit)
        || !noop_normal_or_zero(feature) { return unsafe_bound; }
    if abs(learning_rate) > NOOP_MAX_FACTOR || abs(credit) > NOOP_MAX_FACTOR
        || abs(feature) > NOOP_MAX_FACTOR || !noop_normal_positive(ENCODER_CREDIT_SCALE)
        || ENCODER_CREDIT_SCALE > NOOP_MAX_FACTOR { return unsafe_bound; }
    if learning_rate == 0.0 || credit == 0.0 || feature == 0.0 { return 0.0; }
    let first = abs(learning_rate) * abs(credit);
    let second = first * ENCODER_CREDIT_SCALE;
    let product = second * abs(feature);
    if !noop_normal_positive(first) || !noop_normal_positive(second)
        || !noop_normal_positive(product) { return unsafe_bound; }
    let inflated = product * NOOP_PRODUCT_INFLATION;
    if !noop_normal_positive(inflated) { return unsafe_bound; }
    return bitcast<f32>(bitcast<u32>(inflated) + 1u);
}

fn noop_half_gap_bits(weight: f32) -> u32 {
    let magnitude = bitcast<u32>(weight) & NOOP_MAGNITUDE_MASK;
    let exponent = magnitude >> NOOP_EXPONENT_SHIFT;
    if magnitude < NOOP_MIN_NORMAL_BITS || magnitude > NOOP_MAX_WEIGHT_BITS
        || exponent <= NOOP_MIN_WEIGHT_EXPONENT { return 0u; }
    var half_exponent = exponent - NOOP_HALF_GAP_EXPONENT_OFFSET;
    if (magnitude & NOOP_MANTISSA_MASK) == 0u { half_exponent -= 1u; }
    return half_exponent << NOOP_EXPONENT_SHIFT;
}

fn noop_encoder_credit(agent_id: u32, weight: u32, tid: u32) {
    let valid = weight < FEATURE_COUNT * ENCODED_DIMENSION
        && physics_state[agent_id * PHYS_STRIDE + P_ALIVE] >= 0.5;
    let block = weight / NOOP_BLOCK_WIDTH;
    let metadata_base = wc_u32(WC_AGENT_COUNT) * BRAIN_SCRATCH_STRIDE;
    let metadata = metadata_base + (agent_id * NOOP_BLOCKS_PER_AGENT + block) * NOOP_CACHE_RECORD_WORDS;
    var next_gap = 0u;
    var skipped = 0u;
    if valid {
        let cached_gap = bitcast<u32>(brain_scratch[metadata]);
        let dim = weight % ENCODED_DIMENSION;
        let feature = weight / ENCODED_DIMENSION;
        let credit = decision_buffer[agent_id * DECISION_STRIDE + DECISION_CREDIT + dim];
        let credit_enabled = abs(credit) >= CREDIT_EPSILON;
        var keep_weight = !credit_enabled;
        if credit_enabled {
            let feature_value = brain_scratch[agent_id * BRAIN_SCRATCH_STRIDE + SCRATCH_FEATURES + feature];
            let bound = noop_update_bound(brain_config[1].x, credit, feature_value);
            keep_weight = bound < bitcast<f32>(cached_gap);
        }
        if cached_gap != 0u && keep_weight {
            // This lower bound remains valid for every retained weight.
            next_gap = cached_gap;
            skipped = select(0u, 1u, credit_enabled);
        } else {
            // The unchanged production helper owns the scalar FP32 expression.
            phase_encoder_credit(agent_id, weight);
            next_gap = noop_half_gap_bits(brain_state[agent_id * BRAIN_STRIDE + O_ENC_WEIGHTS + weight]);
        }
    }
    noop_refresh[tid] = vec2<u32>(next_gap, skipped);
    workgroupBarrier();
    let lane = tid % NOOP_BLOCK_WIDTH;
    for (var stride = NOOP_BLOCK_WIDTH / 2u; stride > 0u; stride /= 2u) {
        if lane < stride {
            let other = noop_refresh[tid + stride];
            noop_refresh[tid] = vec2<u32>(min(noop_refresh[tid].x, other.x), noop_refresh[tid].y + other.y);
        }
        workgroupBarrier();
    }
    if lane == 0u && valid {
        brain_scratch[metadata] = bitcast<f32>(noop_refresh[tid].x);
        brain_scratch[metadata + 1u] = f32(noop_refresh[tid].y);
    }
}

// The world workgroup accesses no cache or encoder weights. Credit groups
// reach every reduction barrier, including dead agents and padded lanes.
const NOOP_WORLD_WORKGROUPS: u32 = 1u;
override NOOP_GROUPS_PER_AGENT: u32 =
    (FEATURE_COUNT * ENCODED_DIMENSION + ENCODER_CREDIT_THREADS - 1u) / ENCODER_CREDIT_THREADS;
@compute @workgroup_size(ENCODER_CREDIT_THREADS)
fn encoder_noop_cache_tick(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    if wgid.x < NOOP_WORLD_WORKGROUPS {
        global_world_inner(lid.x);
    } else {
        let credit_group = wgid.x - NOOP_WORLD_WORKGROUPS;
        let agent_id = credit_group / NOOP_GROUPS_PER_AGENT;
        let tile = credit_group % NOOP_GROUPS_PER_AGENT;
        noop_encoder_credit(agent_id, tile * ENCODER_CREDIT_THREADS + lid.x, lid.x);
    }
}
