use glam::Vec3;
use xagent_shared::{MotorAction, MotorCommand};

use crate::agent::AgentBody;
use crate::world::biome::BiomeType;
use crate::world::WorldState;

/// Gravitational acceleration in units/s². Set higher than Earth's 9.8 to make
/// the simulation feel responsive — agents fall quickly, making jump timing matter.
const GRAVITY: f32 = 20.0;
/// Maximum horizontal movement speed in units/s. At this speed an agent crosses
/// the default 256-unit world in ~32 seconds — fast enough to explore but slow
/// enough that navigation decisions matter.
const MOVE_SPEED: f32 = 8.0;
/// Maximum turn rate in radians/s (~172°/s). A full 360° turn takes ~2 seconds,
/// balancing responsiveness with the need for the agent to commit to a direction.
const TURN_SPEED: f32 = 3.0;
/// Half the agent's collision height in world units. The agent's feet are at
/// `ground_height + AGENT_HALF_HEIGHT`, making the full agent 2 units tall.
const AGENT_HALF_HEIGHT: f32 = 1.0;
/// Maximum distance (in world units) at which an agent can consume a food item.
/// Slightly larger than the agent's body (2-unit cube) to provide a forgiving
/// interaction radius without requiring pixel-perfect alignment.
const FOOD_CONSUME_RADIUS: f32 = 2.5;

/// Sanitize motor commands: replace NaN/Infinity with 0.0 and clamp to [-1, 1].
fn sanitize_motor(motor: &MotorCommand) -> MotorCommand {
    MotorCommand {
        forward: if motor.forward.is_finite() { motor.forward.clamp(-1.0, 1.0) } else { 0.0 },
        strafe: if motor.strafe.is_finite() { motor.strafe.clamp(-1.0, 1.0) } else { 0.0 },
        turn: if motor.turn.is_finite() { motor.turn.clamp(-1.0, 1.0) } else { 0.0 },
        action: motor.action,
    }
}

/// Advance one simulation tick: apply motor commands, gravity, collisions, energy.
/// Returns `Some(food_index)` if the agent consumed food this tick, `None` otherwise.
pub fn step(
    agent: &mut AgentBody,
    motor: &MotorCommand,
    world: &mut WorldState,
    dt: f32,
) -> Option<usize> {
    if !agent.body.alive {
        return None;
    }

    let motor = sanitize_motor(motor);

    // Save last known good position/velocity for NaN recovery
    let last_good_position = agent.body.position;
    let last_good_velocity = agent.body.velocity;

    agent.snapshot_internals();

    // ── turning ────────────────────────────────────────────────────
    let prev_yaw = agent.yaw;
    agent.yaw += motor.turn * TURN_SPEED * dt;
    agent.angular_velocity = (agent.yaw - prev_yaw) / dt.max(1e-6);
    agent.body.facing = Vec3::new(agent.yaw.sin(), 0.0, agent.yaw.cos()).normalize();

    // ── locomotion ─────────────────────────────────────────────────
    let right = Vec3::new(agent.body.facing.z, 0.0, -agent.body.facing.x);
    let mut desired = agent.body.facing * motor.forward + right * motor.strafe;
    if desired.length_squared() > 1.0 {
        desired = desired.normalize();
    }
    agent.body.velocity.x = desired.x * MOVE_SPEED;
    agent.body.velocity.z = desired.z * MOVE_SPEED;

    // ── gravity ────────────────────────────────────────────────────
    agent.body.velocity.y -= GRAVITY * dt;

    // ── integrate position ─────────────────────────────────────────
    agent.body.position += agent.body.velocity * dt;

    // ── clamp to world bounds ──────────────────────────────────────
    let half = world.config.world_size / 2.0 - 1.0;
    agent.body.position.x = agent.body.position.x.clamp(-half, half);
    agent.body.position.z = agent.body.position.z.clamp(-half, half);

    // ── ground collision ───────────────────────────────────────────
    let ground = world.terrain.height_at(agent.body.position.x, agent.body.position.z);
    if agent.body.position.y < ground + AGENT_HALF_HEIGHT {
        agent.body.position.y = ground + AGENT_HALF_HEIGHT;
        agent.body.velocity.y = 0.0;
    }

    // ── NaN/infinity recovery ──────────────────────────────────────
    fn vec3_is_finite(v: Vec3) -> bool {
        v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
    }
    if !vec3_is_finite(agent.body.position) || !vec3_is_finite(agent.body.velocity) {
        agent.body.position = last_good_position;
        agent.body.velocity = last_good_velocity;
    }

    // ── jump ───────────────────────────────────────────────────────
    if motor.action == Some(MotorAction::Jump)
        && (agent.body.position.y - ground - AGENT_HALF_HEIGHT).abs() < 0.1
    {
        agent.body.velocity.y = 8.0;
    }

    // ── energy depletion ───────────────────────────────────────────
    let movement_mag = (motor.forward.abs() + motor.strafe.abs()).min(1.414);
    agent.body.internal.energy -= world.config.energy_depletion_rate;
    agent.body.internal.energy -= movement_mag * world.config.movement_energy_cost;

    // ── biome effects ──────────────────────────────────────────────
    let biome = world.biome_map.biome_at(agent.body.position.x, agent.body.position.z);
    if biome == BiomeType::Danger {
        agent.body.internal.integrity -= world.config.hazard_damage_rate;
    }

    // integrity regen when energy > 50 %
    if agent.body.internal.energy_signal() > 0.5
        && agent.body.internal.integrity < agent.body.internal.max_integrity
    {
        agent.body.internal.integrity = (agent.body.internal.integrity
            + world.config.integrity_regen_rate)
            .min(agent.body.internal.max_integrity);
    }

    // ── auto-consume food on contact ────────────────────────────────
    // Food is absorbed automatically when the agent is within range —
    // organisms don't "choose" to metabolize nutrients. The brain's job
    // is to MOVE TOWARD food, not to decide to eat.
    let consumed = try_consume(agent, world);

    // ── clamp & death check ────────────────────────────────────────
    agent.body.internal.energy = agent.body.internal.energy.max(0.0);
    agent.body.internal.integrity = agent.body.internal.integrity.max(0.0);
    if agent.body.internal.is_dead() {
        agent.body.alive = false;
    }

    consumed
}

/// Attempt to consume the nearest food item within FOOD_CONSUME_RADIUS.
/// Uses the spatial grid for O(1) lookup instead of scanning all food items.
/// Awards food_energy_value to the agent and marks the food as consumed
/// with a 10-second respawn timer. Returns `Some(food_index)` if food was consumed.
fn try_consume(agent: &mut AgentBody, world: &mut WorldState) -> Option<usize> {
    let pos = agent.body.position;
    let mut best: Option<(usize, f32)> = None;

    for idx in world.food_grid.query_nearby(pos.x, pos.z) {
        let food = &world.food_items[idx];
        if food.consumed {
            continue;
        }
        let d = (food.position - pos).length();
        if d < FOOD_CONSUME_RADIUS {
            if best.map_or(true, |(_, bd)| d < bd) {
                best = Some((idx, d));
            }
        }
    }

    if let Some((idx, _)) = best {
        let fx = world.food_items[idx].position.x;
        let fz = world.food_items[idx].position.z;
        world.food_items[idx].consumed = true;
        world.food_items[idx].respawn_timer = 10.0;
        world.food_grid.remove(idx, fx, fz);
        agent.body.internal.energy = (agent.body.internal.energy + world.config.food_energy_value)
            .min(agent.body.internal.max_energy);
        Some(idx)
    } else {
        None
    }
}
