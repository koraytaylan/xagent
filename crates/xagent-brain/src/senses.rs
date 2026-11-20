//! Declarative description of the brain's senses.
//!
//! Each sense is a block of input channels that is gathered, centred,
//! whitened and fed to the motor routes. This module is pure data: it names
//! the senses the brain declares and the parameters of each, so that other
//! code can derive layouts and shader parameters from one table.

use crate::buffers::{
    O_VISION_PLASTICITY, SCENT_CHANNELS, SCENT_INITIAL_VARIANCE, VISION_PATHWAY_INITIAL_VARIANCE,
    VISION_PATHWAY_INPUTS,
};

/// Where a sense reads its raw inputs from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SenseInput {
    /// Channels read from the non-visual feature vector starting at `offset`.
    NonVisualFeatures {
        /// Index of the first channel in the feature vector.
        offset: usize,
    },
    /// Channels pooled from the left and right visual hemifields.
    VisualHemifields,
}

/// How a sense decorrelates its centred inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Whitening {
    /// Closed-form whitening of a two-channel sense from its 2x2 covariance.
    ClosedForm2 {
        /// Variance each channel starts with before any statistics exist.
        initial_variance: f32,
    },
    /// Iterative Jacobi eigen-decomposition refreshed on a fixed cadence.
    Jacobi {
        /// Brain ticks between whitening-matrix refreshes.
        refresh_ticks: u32,
        /// Variance each channel starts with before any statistics exist.
        initial_variance: f32,
    },
}

/// Source of the learning rate applied to a sense's weight updates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LearningRate {
    /// The actor learning rate alone.
    Actor,
    /// The actor learning rate scaled by a heritable gene stored in the
    /// brain's per-agent parameter block.
    ActorTimesGene {
        /// Index of the gene in the per-agent parameter block.
        gene_offset: usize,
    },
}

/// Motor route a sense can drive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MotorRoute {
    /// Turning, steered by the sense's turn weights.
    Turn,
    /// Forward drive.
    Forward,
    /// Critic (value head) input.
    Critic,
}

/// Whether a route's weights are inherited and mutated across generations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Heritability {
    /// Weights are inherited and perturbed by mutation.
    Mutated,
    /// Weights are learned within a life and not inherited.
    LearnedOnly,
}

/// One motor route attached to a sense.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouteDescriptor {
    /// The route driven by the sense.
    pub route: MotorRoute,
    /// Whether the route's weights are inherited.
    pub heritability: Heritability,
}

/// Full description of one sense.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SenseDescriptor {
    /// Short stable name of the sense.
    pub name: &'static str,
    /// Number of input channels.
    pub channels: usize,
    /// Where the raw inputs come from.
    pub input: SenseInput,
    /// Rate at which the running mean and covariance adapt.
    pub adaptation_rate: f32,
    /// Whitening strategy.
    pub whitening: Whitening,
    /// Learning-rate source for weight updates.
    pub learning_rate: LearningRate,
    /// Motor routes the sense drives, in declared order.
    pub routes: &'static [RouteDescriptor],
}

/// Route-mask bit set when a sense drives the turn route. The
/// `shaders/kernel/common.wgsl` constant `SENSE_ROUTE_TURN` mirrors it.
pub const SENSE_ROUTE_TURN: u32 = 1;

/// Route-mask bit set when a sense drives the forward route. The
/// `shaders/kernel/common.wgsl` constant `SENSE_ROUTE_FORWARD` mirrors it.
pub const SENSE_ROUTE_FORWARD: u32 = 2;

/// Route-mask bit set when a sense drives the critic route. The
/// `shaders/kernel/common.wgsl` constant `SENSE_ROUTE_CRITIC` mirrors it.
pub const SENSE_ROUTE_CRITIC: u32 = 4;

/// Kind code of [`SenseInput::NonVisualFeatures`]. The
/// `shaders/kernel/common.wgsl` constant `SENSE_INPUT_KIND_NON_VISUAL_FEATURES` mirrors it.
pub const SENSE_INPUT_KIND_NON_VISUAL_FEATURES: u32 = 0;

/// Kind code of [`SenseInput::VisualHemifields`]. The
/// `shaders/kernel/common.wgsl` constant `SENSE_INPUT_KIND_VISUAL_HEMIFIELDS` mirrors it.
pub const SENSE_INPUT_KIND_VISUAL_HEMIFIELDS: u32 = 1;

/// Kind code of [`Whitening::ClosedForm2`]. The
/// `shaders/kernel/common.wgsl` constant `SENSE_WHITENING_KIND_CLOSED_FORM_2` mirrors it.
pub const SENSE_WHITENING_KIND_CLOSED_FORM_2: u32 = 0;

/// Kind code of [`Whitening::Jacobi`]. The
/// `shaders/kernel/common.wgsl` constant `SENSE_WHITENING_KIND_JACOBI` mirrors it.
pub const SENSE_WHITENING_KIND_JACOBI: u32 = 1;

impl SenseInput {
    /// Shader kind code of this input source.
    pub const fn kind_code(&self) -> u32 {
        match self {
            Self::NonVisualFeatures { .. } => SENSE_INPUT_KIND_NON_VISUAL_FEATURES,
            Self::VisualHemifields => SENSE_INPUT_KIND_VISUAL_HEMIFIELDS,
        }
    }
}

impl Whitening {
    /// Shader kind code of this whitening strategy.
    pub const fn kind_code(&self) -> u32 {
        match self {
            Self::ClosedForm2 { .. } => SENSE_WHITENING_KIND_CLOSED_FORM_2,
            Self::Jacobi { .. } => SENSE_WHITENING_KIND_JACOBI,
        }
    }
}

impl LearningRate {
    /// Gene index of the learning-rate scale, or zero when the rate has none.
    pub const fn gene_offset_or_zero(&self) -> usize {
        match self {
            Self::Actor => 0,
            Self::ActorTimesGene { gene_offset } => *gene_offset,
        }
    }
}

impl SenseDescriptor {
    /// Bit mask of the routes this sense declares, built from the
    /// `SENSE_ROUTE_*` bits.
    pub const fn route_mask(&self) -> u32 {
        let mut mask = 0;
        let mut i = 0;
        while i < self.routes.len() {
            mask |= match self.routes[i].route {
                MotorRoute::Turn => SENSE_ROUTE_TURN,
                MotorRoute::Forward => SENSE_ROUTE_FORWARD,
                MotorRoute::Critic => SENSE_ROUTE_CRITIC,
            };
            i += 1;
        }
        mask
    }
}

/// Index of the first smell channel in the non-visual feature vector. The
/// `SENSE{k}_INPUT_OFFSET` default in `shaders/kernel/common.wgsl` mirrors it.
pub const SCENT_FEATURE_OFFSET: usize = 25;

/// Adaptation rate of the smell whitening statistics. The
/// `SENSE{k}_ADAPTATION_RATE` default in `shaders/kernel/common.wgsl` mirrors it.
pub const SCENT_WHITENING_RATE: f32 = 0.01;

/// Adaptation rate of the vision pathway statistics. The
/// `SENSE{k}_ADAPTATION_RATE` default in `shaders/kernel/common.wgsl` mirrors it.
pub const VISION_PATHWAY_RATE: f32 = 0.01;

/// Brain ticks between vision whitening refreshes. The
/// `SENSE{k}_REFRESH_TICKS` default in `shaders/kernel/common.wgsl` mirrors it.
pub const VISION_WHITENING_REFRESH: u32 = 20;

/// Maximum number of senses the shader parameter block has slots for.
pub const MAX_SENSE_SLOTS: usize = 4;

/// Maximum channels per sense; bounds the per-sense shader arrays.
pub const MAX_SENSE_CHANNELS: usize = 8;

/// Routes shared by both declared senses, in the order turn, forward, critic.
/// Only turn is declared: the forward and critic routes are not enabled by
/// default, so they are left out of the table.
pub const SENSE_ROUTES: &[RouteDescriptor] = &[RouteDescriptor {
    route: MotorRoute::Turn,
    heritability: Heritability::Mutated,
}];

/// The smell sense: two nostril channels whitened in closed form, learning at
/// the plain actor rate.
pub const SMELL: SenseDescriptor = SenseDescriptor {
    name: "smell",
    channels: SCENT_CHANNELS,
    input: SenseInput::NonVisualFeatures {
        offset: SCENT_FEATURE_OFFSET,
    },
    adaptation_rate: SCENT_WHITENING_RATE,
    whitening: Whitening::ClosedForm2 {
        initial_variance: SCENT_INITIAL_VARIANCE,
    },
    learning_rate: LearningRate::Actor,
    routes: SENSE_ROUTES,
};

/// The vision sense: pooled hemifield channels whitened with a periodically
/// refreshed Jacobi decomposition, learning at the actor rate scaled by the
/// heritable vision plasticity gene.
pub const VISION: SenseDescriptor = SenseDescriptor {
    name: "vision",
    channels: VISION_PATHWAY_INPUTS,
    input: SenseInput::VisualHemifields,
    adaptation_rate: VISION_PATHWAY_RATE,
    whitening: Whitening::Jacobi {
        refresh_ticks: VISION_WHITENING_REFRESH,
        initial_variance: VISION_PATHWAY_INITIAL_VARIANCE,
    },
    learning_rate: LearningRate::ActorTimesGene {
        gene_offset: O_VISION_PLASTICITY,
    },
    routes: SENSE_ROUTES,
};

/// Every sense the brain declares, in layout order.
pub const DECLARED_SENSES: &[SenseDescriptor] = &[SMELL, VISION];

/// Channel count closed-form whitening handles: its 2x2 covariance inverse
/// is written out explicitly, so it is defined for exactly two channels.
const CLOSED_FORM2_CHANNELS: usize = 2;

/// Checks the declared table against the shader limits: slot count, channel
/// bound, and that closed-form whitening is only used with two channels.
const fn declared_senses_are_valid() -> bool {
    if DECLARED_SENSES.len() > MAX_SENSE_SLOTS {
        return false;
    }
    let mut i = 0;
    while i < DECLARED_SENSES.len() {
        let sense = &DECLARED_SENSES[i];
        if sense.channels > MAX_SENSE_CHANNELS {
            return false;
        }
        if matches!(sense.whitening, Whitening::ClosedForm2 { .. })
            && sense.channels != CLOSED_FORM2_CHANNELS
        {
            return false;
        }
        i += 1;
    }
    true
}

const _: () = assert!(declared_senses_are_valid());

#[cfg(test)]
mod tests {
    use super::*;

    const COMMON_WGSL: &str = include_str!("shaders/kernel/common.wgsl");

    /// Literal text of the `override NAME: TYPE = LITERAL;` default in
    /// `common.wgsl`, with any numeric type suffix removed.
    fn wgsl_override_default(name: &str) -> String {
        let key = format!("override {name}:");
        let line = COMMON_WGSL
            .lines()
            .find(|l| l.trim_start().starts_with(&key))
            .unwrap_or_else(|| panic!("{name} has no override in common.wgsl"));
        line.split('=')
            .nth(1)
            .and_then(|v| v.split(';').next())
            .unwrap_or_else(|| panic!("{name} has no default"))
            .trim()
            .trim_end_matches(['u', 'f'])
            .to_string()
    }

    fn override_u64(name: &str) -> u64 {
        wgsl_override_default(name)
            .parse()
            .unwrap_or_else(|_| panic!("{name} default is not an integer"))
    }

    #[test]
    fn sense_specific_constants_single_source() {
        for (k, sense) in DECLARED_SENSES.iter().enumerate() {
            let field = |suffix: &str| format!("SENSE{k}_{suffix}");
            let input_offset = match sense.input {
                SenseInput::NonVisualFeatures { offset } => offset,
                SenseInput::VisualHemifields => 0,
            };
            let refresh_ticks = match sense.whitening {
                Whitening::Jacobi { refresh_ticks, .. } => refresh_ticks,
                Whitening::ClosedForm2 { .. } => 0,
            };
            let integers: [(&str, u64); 7] = [
                ("INPUT_KIND", u64::from(sense.input.kind_code())),
                (
                    "INPUT_OFFSET",
                    u64::try_from(input_offset).expect("offset fits u64"),
                ),
                (
                    "CHANNELS",
                    u64::try_from(sense.channels).expect("channels fit u64"),
                ),
                ("WHITENING_KIND", u64::from(sense.whitening.kind_code())),
                ("REFRESH_TICKS", u64::from(refresh_ticks)),
                (
                    "LEARNING_RATE_GENE",
                    u64::try_from(sense.learning_rate.gene_offset_or_zero())
                        .expect("gene offset fits u64"),
                ),
                ("ROUTE_MASK", u64::from(sense.route_mask())),
            ];
            for (suffix, expected) in integers {
                let name = field(suffix);
                assert_eq!(override_u64(&name), expected, "{name}");
            }
            let name = field("ADAPTATION_RATE");
            let rate: f32 = wgsl_override_default(&name)
                .parse()
                .unwrap_or_else(|_| panic!("{name} default is not a float"));
            assert_eq!(rate.to_bits(), sense.adaptation_rate.to_bits(), "{name}");
        }
    }

    #[test]
    fn sense_override_kind_codes() {
        assert_eq!(
            SenseInput::NonVisualFeatures { offset: 7 }.kind_code(),
            SENSE_INPUT_KIND_NON_VISUAL_FEATURES
        );
        assert_eq!(
            SenseInput::VisualHemifields.kind_code(),
            SENSE_INPUT_KIND_VISUAL_HEMIFIELDS
        );
        assert_eq!(
            Whitening::ClosedForm2 {
                initial_variance: 1.0
            }
            .kind_code(),
            SENSE_WHITENING_KIND_CLOSED_FORM_2
        );
        assert_eq!(
            Whitening::Jacobi {
                refresh_ticks: 3,
                initial_variance: 1.0
            }
            .kind_code(),
            SENSE_WHITENING_KIND_JACOBI
        );
        assert_eq!(LearningRate::Actor.gene_offset_or_zero(), 0);
        assert_eq!(
            LearningRate::ActorTimesGene { gene_offset: 9 }.gene_offset_or_zero(),
            9
        );
        assert_eq!(SMELL.route_mask(), SENSE_ROUTE_TURN);
        let all: &'static [RouteDescriptor] = &[
            RouteDescriptor {
                route: MotorRoute::Turn,
                heritability: Heritability::Mutated,
            },
            RouteDescriptor {
                route: MotorRoute::Forward,
                heritability: Heritability::LearnedOnly,
            },
            RouteDescriptor {
                route: MotorRoute::Critic,
                heritability: Heritability::LearnedOnly,
            },
        ];
        let every = SenseDescriptor {
            routes: all,
            ..SMELL
        };
        assert_eq!(
            every.route_mask(),
            SENSE_ROUTE_TURN | SENSE_ROUTE_FORWARD | SENSE_ROUTE_CRITIC
        );
        assert_eq!(
            [SENSE_ROUTE_TURN, SENSE_ROUTE_FORWARD, SENSE_ROUTE_CRITIC],
            [1, 2, 4]
        );
    }

    #[test]
    fn only_turn_routes_are_mutated() {
        for sense in DECLARED_SENSES {
            for route in sense.routes {
                match route.route {
                    MotorRoute::Turn => assert_eq!(route.heritability, Heritability::Mutated),
                    _ => assert_eq!(route.heritability, Heritability::LearnedOnly),
                }
                if route.heritability == Heritability::Mutated {
                    assert_eq!(route.route, MotorRoute::Turn);
                    assert!(sense.name == "smell" || sense.name == "vision");
                }
            }
        }
    }

    #[test]
    fn declared_routes_match_route_wiring() {
        let expected: &[RouteDescriptor] = &[RouteDescriptor {
            route: MotorRoute::Turn,
            heritability: Heritability::Mutated,
        }];
        assert_eq!(SMELL.routes, expected);
        assert_eq!(VISION.routes, expected);
    }
}
