# Evolvable Anti-Loop Parameters Design

**Problem:** The anti-loop mechanisms (habituation, curiosity, motor fatigue) work but their effect is subtle. The constants are one-size-fits-all — every agent has the same sensitivity to monotony, same curiosity ceiling, same fatigue threshold. There is no way for natural selection to discover that different niches favor different tolerances.

**Solution:** Promote 4 anti-loop constants to heritable `BrainConfig` fields with stronger defaults. Evolution can then tune how aggressively each lineage responds to monotony.

## Parameters

| Config field | Replaces constant | Old value | New default | Clamp range | Purpose |
|---|---|---|---|---|---|
| `habituation_sensitivity` | `SENSITIVITY` in habituation.rs | 15.0 | 20.0 | [5.0, 50.0] | How fast boredom builds (scales variance_ema into attenuation) |
| `max_curiosity_bonus` | `MAX_CURIOSITY_BONUS` in habituation.rs | 0.4 | 0.6 | [0.1, 1.0] | Ceiling on exploration boost from sensory monotony |
| `fatigue_recovery_sensitivity` | `RECOVERY_SENSITIVITY` in motor_fatigue.rs | 5.0 | 8.0 | [2.0, 20.0] | How easily fatigue lifts when motor output diversifies |
| `fatigue_floor` | `FATIGUE_FLOOR` in motor_fatigue.rs | 0.2 | 0.1 | [0.05, 0.4] | Minimum motor output under fatigue (lower = harsher) |

Constants that remain hardcoded (structural, not behavioral):
- `HABITUATION_EMA_ALPHA` (0.02) — smoothing window
- `ATTENUATION_FLOOR` (0.1) — minimum sensory attenuation
- `FATIGUE_WINDOW` (64) — ring buffer size

## Architecture

No new files, no new modules, no new telemetry. Same pattern as the existing `distress_exponent`:

| Action | File | Change |
|---|---|---|
| Modify | `crates/xagent-shared/src/config.rs` | Add 4 fields to BrainConfig with `#[serde(default)]` |
| Modify | `crates/xagent-brain/src/habituation.rs` | Replace 2 constants with struct fields, update constructor |
| Modify | `crates/xagent-brain/src/motor_fatigue.rs` | Replace 2 constants with struct fields, update constructor |
| Modify | `crates/xagent-brain/src/brain.rs` | Pass config values into constructors |
| Modify | `crates/xagent-sandbox/src/agent/mod.rs` | Mutate + crossover the 4 new fields |

### Struct changes

**SensoryHabituation:**
```rust
pub struct SensoryHabituation {
    // ...existing fields (prev_encoded, variance_ema, attenuation, habituated, curiosity_bonus, tick)...
    sensitivity: f32,
    max_curiosity_bonus: f32,
}
```
- Constructor: `new(repr_dim: usize, sensitivity: f32, max_curiosity_bonus: f32)`
- `update()` uses `self.sensitivity` instead of `SENSITIVITY`, `self.max_curiosity_bonus` instead of `MAX_CURIOSITY_BONUS`
- Remove `SENSITIVITY` and `MAX_CURIOSITY_BONUS` module-level constants

**MotorFatigue:**
```rust
pub struct MotorFatigue {
    // ...existing fields (forward_ring, turn_ring, cursor, len, fatigue_factor, motor_variance)...
    recovery_sensitivity: f32,
    fatigue_floor: f32,
}
```
- Constructor: `new(recovery_sensitivity: f32, fatigue_floor: f32)`
- `update()` uses `self.recovery_sensitivity` instead of `RECOVERY_SENSITIVITY`, `self.fatigue_floor` instead of `FATIGUE_FLOOR`
- Remove `RECOVERY_SENSITIVITY` and `FATIGUE_FLOOR` module-level constants

### Evolution

In `mutate_config_with_strength()`:
```rust
habituation_sensitivity: perturb_f(&mut rng, parent.habituation_sensitivity).clamp(5.0, 50.0),
max_curiosity_bonus: perturb_f(&mut rng, parent.max_curiosity_bonus).clamp(0.1, 1.0),
fatigue_recovery_sensitivity: perturb_f(&mut rng, parent.fatigue_recovery_sensitivity).clamp(2.0, 20.0),
fatigue_floor: perturb_f(&mut rng, parent.fatigue_floor).clamp(0.05, 0.4),
```

In `crossover_config()`: coin-flip each field from parent A or B (same pattern as all other fields).

### Brain wiring

In `Brain::new()`:
```rust
habituation: SensoryHabituation::new(
    repr_dim,
    config.habituation_sensitivity,
    config.max_curiosity_bonus,
),
motor_fatigue: MotorFatigue::new(
    config.fatigue_recovery_sensitivity,
    config.fatigue_floor,
),
```

### Backwards compatibility

All 4 new BrainConfig fields get `#[serde(default = "...")]` attributes so pre-existing JSON configs deserialize without breaking.

## Non-Goals

- No new telemetry fields (existing ones already reflect these parameters)
- No new modules or files
- No changes to the tick loop pipeline order
- No dynamic modulation based on urgency or wellbeing
- No changes to HABITUATION_EMA_ALPHA, ATTENUATION_FLOOR, or FATIGUE_WINDOW
