use crate::{Gene, Genome, Organism, Vec2, WorldState};
use macroquad::rand;

#[derive(Debug, Default, PartialEq)]
pub struct Brain {
    pub outputs: Vec<NeuronNode>,
    pub edges: Vec<NeuralCon>,
}

#[derive(Debug)]
pub struct BrainBuilder {
    pub edges: Vec<NeuralCon>,
}

impl BrainBuilder {
    pub fn new(genome: &Genome) -> Option<Self> {
        let mut edges = Vec::with_capacity(genome.genes.len());
        for gene in genome.genes.iter() {
            match NeuralCon::from_gene(*gene) {
                Some(con) => edges.push(con),
                None => return None,
            }
        }

        Self::new_with_edges(edges)
    }

    /// Valid edge configuration:
    ///     - Sense    -> Action
    ///     - Sense    -> Internal
    ///     - Internal -> Action
    ///     - Internal -> Internal
    pub fn new_with_edges(edges: Vec<NeuralCon>) -> Option<Self> {
        // repeated edges
        let mut edge_comb = Vec::with_capacity(edges.len());
        for edge in edges.iter() {
            let e1 = (edge.lhs, edge.rhs);
            let e2 = (edge.rhs, edge.lhs);
            if edge_comb.contains(&e1) || edge_comb.contains(&e2) {
                return None;
            }
            edge_comb.push(e1);
        }

        for edge in edges.iter() {
            if let Neuron::Action(_) = edge.lhs {
                return None;
            } else if let Neuron::Sense(_) = edge.rhs {
                return None;
            }
        }

        Some(Self { edges })
    }

    pub fn build(self) -> Brain {
        Brain {
            edges: self.edges.clone(),
            outputs: NeuronNode::build_outputs(self.edges),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct NeuronNode {
    pub neuron: Neuron,
    pub inputs: Vec<(NeuronNode, Association)>,
}

impl NeuronNode {
    pub fn build_outputs(edges: Vec<NeuralCon>) -> Vec<NeuronNode> {
        let mut outputs = Vec::new();

        let mut output_neurons = Vec::new();
        for edge in edges.iter() {
            if let Neuron::Action(_) = edge.rhs {
                if !output_neurons.contains(&edge.rhs) {
                    output_neurons.push(edge.rhs);
                }
            }
        }

        for output in output_neurons.iter() {
            outputs.push(Self::build_output(*output, &edges));
        }

        outputs
    }

    fn build_output(neuron: Neuron, edges: &[NeuralCon]) -> NeuronNode {
        let mut inputs = Vec::new();
        for edge in edges.iter() {
            if edge.rhs == neuron {
                if edge.lhs == neuron {
                    inputs.push((
                        NeuronNode {
                            inputs: Vec::new(),
                            neuron,
                        },
                        edge.assoc,
                    ));
                } else {
                    inputs.push((Self::build_output(edge.lhs, edges), edge.assoc));
                }
            }
        }

        NeuronNode { inputs, neuron }
    }

    pub fn should_fire(&self, host: &Organism, state: &WorldState) -> bool {
        rand::gen_range(0.0, 1.0) < self.fire(host, state)
    }

    pub fn fire(&self, host: &Organism, state: &WorldState) -> f32 {
        let mut output = 0.;
        for input in self.inputs.iter() {
            output += Self::fire_node(&input.0, &input.1, host, state);
        }
        output.tanh()
    }

    fn fire_node(
        node: &NeuronNode,
        assoc: &Association,
        host: &Organism,
        state: &WorldState,
    ) -> f32 {
        match node.neuron {
            Neuron::Sense(s) => {
                assert_eq!(0, node.inputs.len());
                s.fire(host, state) * assoc.weight()
            }
            _ => {
                let mut output = 0.;
                for input in node.inputs.iter() {
                    output += Self::fire_node(&input.0, &input.1, host, state);
                }
                (output * assoc.weight()).tanh()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuralCon {
    pub assoc: Association,
    pub lhs: Neuron,
    pub rhs: Neuron,
}

impl NeuralCon {
    pub fn from_gene(gene: Gene) -> Option<Self> {
        let assoc =
            Association::new(((gene.0 & 0xFFFFFFFF) as f32) / (u32::MAX as f32) * 8.0 - 4.0);

        let lhs_type = (gene.0 >> 62) & 3;
        let lhs_variant = ((gene.0 >> (62 - 14)) & 0x3FFF) as u32;
        let lhs = match lhs_type % 3 {
            0 => Neuron::Sense(SenseNeuron::new_from_u32(lhs_variant)),
            1 => Neuron::Internal(InternalNeuron::new_from_u32(lhs_variant)),
            2 => Neuron::Action(ActionNeuron::new_from_u32(lhs_variant)),
            _ => unreachable!(),
        };

        let rhs_type = (gene.0 >> (62 - 16)) & 3;
        let rhs_variant = ((gene.0 >> (62 - 14 - 16)) & 0x3FFF) as u32;
        let rhs = match rhs_type % 3 {
            0 => Neuron::Sense(SenseNeuron::new_from_u32(rhs_variant)),
            1 => Neuron::Internal(InternalNeuron::new_from_u32(rhs_variant)),
            2 => Neuron::Action(ActionNeuron::new_from_u32(rhs_variant)),
            _ => unreachable!(),
        };

        Self::new(assoc, lhs, rhs)
    }

    pub fn new(assoc: Association, lhs: Neuron, rhs: Neuron) -> Option<Self> {
        match lhs {
            Neuron::Sense(_) | Neuron::Internal(_) => match rhs {
                Neuron::Internal(_) | Neuron::Action(_) => Some(Self { assoc, lhs, rhs }),
                _ => None,
            },
            _ => None,
        }
    }
}

/// Association weights are between -4.0 and 4.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Association {
    Positive(f32),
    Negative(f32),
}

impl Association {
    pub fn new(weight: f32) -> Self {
        assert!(!weight.is_nan(), "nan association weight");
        assert!((-4.0..4.1).contains(&weight), "invalid association weight");

        if weight.is_sign_negative() {
            Self::Negative(weight)
        } else {
            Self::Positive(weight)
        }
    }

    pub fn weight(&self) -> f32 {
        match self {
            Self::Positive(val) => *val,
            Self::Negative(val) => *val,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Neuron {
    Action(ActionNeuron),
    Internal(InternalNeuron),
    Sense(SenseNeuron),
}

macro_rules! impl_neuron_from_u32 {
    ($ty:ty) => {
        impl $ty {
            pub fn new_from_u32(variant: u32) -> Self {
                let variant = variant % std::mem::variant_count::<$ty>() as u32;
                assert!((0..std::mem::variant_count::<$ty>() as u32).contains(&variant));

                unsafe { std::mem::transmute(variant) }
            }
        }
    };
}

/// An action neuron outputs a value between -1.0 and 1.0 based on `sum(inputs).tanh()`.
///
/// If the value outputed by the neuron is positive, then this value determines the probability
/// that the neuron will `fire` in the simulation step.
///
/// Action neurons serve as the output of the network, meaning that connections are not allowed to
/// be directed outward.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionNeuron {
    MoveNorth,
    MoveSouth,
    MoveEast,
    MoveWest,
    MoveRandom,
}

impl_neuron_from_u32!(ActionNeuron);

impl ActionNeuron {
    pub fn fire(&self, host_position: &mut Vec2, state: &mut WorldState) {
        match self {
            Self::MoveRandom => {
                Self::new_from_u32(rand::gen_range(0, 4)).fire(host_position, state)
            }
            Self::MoveNorth => {
                let prev = *host_position;
                host_position.y = host_position
                    .y
                    .saturating_sub(1)
                    .clamp(0, state.height() as u32 - 1);
                let post = *host_position;
                if state.occupation.try_move(prev, post).is_err() {
                    *host_position = prev;
                }
            }
            Self::MoveSouth => {
                let prev = *host_position;
                host_position.y = host_position
                    .y
                    .saturating_add(1)
                    .clamp(0, state.height() as u32 - 1);
                let post = *host_position;
                if state.occupation.try_move(prev, post).is_err() {
                    *host_position = prev;
                }
            }
            Self::MoveEast => {
                let prev = *host_position;
                host_position.x = host_position
                    .x
                    .saturating_add(1)
                    .clamp(0, state.width() as u32 - 1);
                let post = *host_position;
                if state.occupation.try_move(prev, post).is_err() {
                    *host_position = prev;
                }
            }
            Self::MoveWest => {
                let prev = *host_position;
                host_position.x = host_position
                    .x
                    .saturating_sub(1)
                    .clamp(0, state.width() as u32 - 1);
                let post = *host_position;
                if state.occupation.try_move(prev, post).is_err() {
                    *host_position = prev;
                }
            }
        }
    }
}

/// Outputs a value between -1.0 and 1.0 based on `sum(inputs).tanh()`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InternalNeuron {
    I1,
    I2,
    I3,
    I4,
}

impl_neuron_from_u32!(InternalNeuron);

/// A sense neuron produces a value between 0.0 and 1.0 based on the environment.
///
/// Sense neurons serve as the input into the network, meaning that connections are not allowed to
/// be directed inward.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SenseNeuron {
    PosX,
    PosY,
}

impl_neuron_from_u32!(SenseNeuron);

impl SenseNeuron {
    pub fn fire(&self, host: &Organism, state: &WorldState) -> f32 {
        match self {
            Self::PosX => host.position.x as f32 / state.width() as f32,
            Self::PosY => host.position.y as f32 / state.height() as f32,
        }
        .clamp(0.0, 1.0)
    }
}

pub fn default_init_edges(num_edges: usize, max_internal_neurons: usize) -> Vec<NeuralCon> {
    assert!(max_internal_neurons > 0);
    assert!(num_edges > 0);

    let is_unique_edge = |edge: &NeuralCon, edges: &[NeuralCon]| {
        !edges.iter().any(|c| {
            (c.lhs == edge.lhs && c.rhs == edge.rhs) || (c.lhs == edge.rhs && c.rhs == edge.lhs)
        })
    };

    let unique_sense = |assoc: f32, rhs: Neuron, edges: &[NeuralCon]| {
        let mut rhs_variant = rand::gen_range(0, std::mem::variant_count::<SenseNeuron>()) as u32;
        let mut sense = SenseNeuron::new_from_u32(rhs_variant);
        while !is_unique_edge(
            &NeuralCon::new(Association::new(assoc), Neuron::Sense(sense), rhs).unwrap(),
            edges,
        ) {
            rhs_variant = rand::gen_range(0, std::mem::variant_count::<SenseNeuron>()) as u32;
            sense = SenseNeuron::new_from_u32(rhs_variant);
        }
        Neuron::Sense(sense)
    };

    let unique_internal = |assoc: f32, lhs: Neuron, edges: &[NeuralCon]| {
        let mut rhs_variant =
            rand::gen_range(0, std::mem::variant_count::<InternalNeuron>()) as u32;
        let mut internal = InternalNeuron::new_from_u32(rhs_variant);
        while !is_unique_edge(
            &NeuralCon::new(Association::new(assoc), lhs, Neuron::Internal(internal)).unwrap(),
            edges,
        ) {
            rhs_variant = rand::gen_range(0, std::mem::variant_count::<InternalNeuron>()) as u32;
            internal = InternalNeuron::new_from_u32(rhs_variant);
        }
        Neuron::Internal(internal)
    };

    let unique_action = |assoc: f32, lhs: Neuron, edges: &[NeuralCon]| {
        let mut rhs_variant = rand::gen_range(0, std::mem::variant_count::<ActionNeuron>()) as u32;
        let mut action = ActionNeuron::new_from_u32(rhs_variant);
        while !is_unique_edge(
            &NeuralCon::new(Association::new(assoc), lhs, Neuron::Action(action)).unwrap(),
            edges,
        ) {
            rhs_variant = rand::gen_range(0, std::mem::variant_count::<ActionNeuron>()) as u32;
            action = ActionNeuron::new_from_u32(rhs_variant);
        }
        Neuron::Action(action)
    };

    let mut edges = Vec::with_capacity(num_edges);

    let mut needs_conn_action = None;
    let mut needs_conn_sense = None;
    let mut num_internal = 0;
    loop {
        assert!(num_internal <= max_internal_neurons);

        if edges.len() >= num_edges {
            break;
        }

        let assoc = rand::gen_range(-4.0, 4.0);

        if edges.len() == num_edges - 1 && needs_conn_sense.is_some() && needs_conn_action.is_some()
        {
            edges.pop();
            needs_conn_sense = None;
            needs_conn_action = None;
        }

        if let Some(neuron) = needs_conn_action {
            match neuron {
                Neuron::Internal(_) => {
                    edges.push(
                        NeuralCon::new(
                            Association::new(assoc),
                            neuron,
                            unique_action(assoc, neuron, &edges),
                        )
                        .unwrap(),
                    );
                    needs_conn_action = None;
                    continue;
                }
                _ => unreachable!(),
            }
        }
        if let Some(neuron) = needs_conn_sense {
            match neuron {
                Neuron::Internal(_) => {
                    edges.push(
                        NeuralCon::new(
                            Association::new(assoc),
                            unique_sense(assoc, neuron, &edges),
                            neuron,
                        )
                        .unwrap(),
                    );
                    needs_conn_sense = None;
                    continue;
                }
                _ => unreachable!(),
            }
        }

        let lhs_variant = rand::gen_range(0, u32::MAX);
        let lhs = if num_internal < max_internal_neurons {
            match rand::gen_range(0, 2) {
                0 => Neuron::Sense(SenseNeuron::new_from_u32(lhs_variant)),
                1 => {
                    num_internal += 1;
                    Neuron::Internal(InternalNeuron::new_from_u32(lhs_variant))
                }
                _ => unreachable!(),
            }
        } else {
            Neuron::Sense(SenseNeuron::new_from_u32(lhs_variant))
        };

        let rhs = if num_internal < max_internal_neurons {
            match lhs {
                Neuron::Sense(_) => match rand::gen_range(0, 2) {
                    0 => {
                        let internal = unique_internal(assoc, lhs, &edges);
                        needs_conn_action = Some(internal);
                        internal
                    }
                    1 => unique_action(assoc, lhs, &edges),
                    _ => unreachable!(),
                },
                Neuron::Internal(_) => match rand::gen_range(0, 2) {
                    0 => {
                        let internal = unique_internal(assoc, lhs, &edges);
                        needs_conn_sense = Some(internal);
                        needs_conn_action = Some(internal);
                        internal
                    }
                    1 => {
                        needs_conn_sense = Some(lhs);
                        unique_action(assoc, lhs, &edges)
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        } else {
            let internal = unique_internal(assoc, lhs, &edges);
            needs_conn_action = Some(internal);
            internal
        };

        edges.push(NeuralCon::new(Association::new(assoc), lhs, rhs).unwrap());
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! neural_con {
        ($lhs:expr, $rhs:expr) => {
            NeuralCon::new(Association::Positive(1.), $lhs, $rhs).unwrap()
        };
    }

    #[test]
    fn brain_builder_from_edges() {
        let internal = Neuron::Internal(InternalNeuron::I1);
        let sense = Neuron::Sense(SenseNeuron::PosX);
        let action = Neuron::Action(ActionNeuron::MoveRandom);

        assert!(BrainBuilder::new_with_edges(vec![
            neural_con!(sense, internal),
            neural_con!(internal, internal),
            neural_con!(internal, action),
        ])
        .is_some());
        assert!(BrainBuilder::new_with_edges(vec![
            neural_con!(sense, action),
            neural_con!(sense, internal),
            neural_con!(internal, internal),
        ])
        .is_some());
        assert!(BrainBuilder::new_with_edges(vec![
            neural_con!(sense, action),
            neural_con!(sense, action),
        ])
        .is_none());
        assert!(BrainBuilder::new_with_edges(vec![neural_con!(internal, internal)]).is_some());
        assert!(BrainBuilder::new_with_edges(vec![neural_con!(sense, internal)]).is_some());
        assert!(BrainBuilder::new_with_edges(vec![neural_con!(internal, action)]).is_some());
        assert!(BrainBuilder::new_with_edges(vec![
            neural_con!(
                Neuron::Internal(InternalNeuron::I3),
                Neuron::Internal(InternalNeuron::I2)
            ),
            neural_con!(
                Neuron::Internal(InternalNeuron::I2),
                Neuron::Internal(InternalNeuron::I3)
            )
        ])
        .is_none());
    }

    macro_rules! neuron_node {
        ($neuron:expr) => {
            (
                NeuronNode {
                    neuron: $neuron,
                    inputs: vec![],
                },
                Association::Positive(1.),
            )
        };
        ($neuron:expr, $($input:tt)*) => {
            (
                NeuronNode {
                    neuron: $neuron,
                    inputs: vec![
                        $($input)*,
                    ],
                },
                Association::Positive(1.),
            )
        };
    }

    #[test]
    fn build_brain() {
        let internal = Neuron::Internal(InternalNeuron::I1);
        let sense = Neuron::Sense(SenseNeuron::PosX);
        let action = Neuron::Action(ActionNeuron::MoveRandom);

        assert_eq!(
            BrainBuilder::new_with_edges(vec![
                neural_con!(sense, action),
                neural_con!(internal, action),
                neural_con!(internal, internal),
                neural_con!(sense, internal),
            ])
            .unwrap()
            .build(),
            Brain {
                outputs: vec![NeuronNode {
                    neuron: action,
                    inputs: vec![
                        neuron_node!(sense),
                        neuron_node!(internal, neuron_node!(internal), neuron_node!(sense))
                    ]
                },]
            }
        );
    }

    #[test]
    fn neuron_fire() {
        let mut s = crate::WorldState::default();
        s.occupation.width = 10;
        s.occupation.height = 10;
        let h = crate::Organism::default();
        let sense1 = SenseNeuron::PosX;
        let sense2 = SenseNeuron::PosY;
        let assoc1 = 1.5;
        let assoc2 = -0.2;

        assert_eq!(
            (sense1.fire(&h, &s) * assoc1).tanh(),
            NeuronNode {
                neuron: Neuron::Action(ActionNeuron::MoveRandom),
                inputs: vec![(
                    NeuronNode {
                        neuron: Neuron::Sense(sense1),
                        inputs: vec![],
                    },
                    Association::Positive(assoc1),
                )],
            }
            .fire(&h, &s)
        );

        assert_eq!(
            (sense1.fire(&h, &s) * assoc1 + sense2.fire(&h, &s) * assoc2).tanh(),
            NeuronNode {
                neuron: Neuron::Action(ActionNeuron::MoveRandom),
                inputs: vec![
                    (
                        NeuronNode {
                            neuron: Neuron::Sense(sense1),
                            inputs: vec![],
                        },
                        Association::Positive(assoc1),
                    ),
                    (
                        NeuronNode {
                            neuron: Neuron::Sense(sense2),
                            inputs: vec![],
                        },
                        Association::Negative(assoc2),
                    )
                ],
            }
            .fire(&h, &s)
        );
    }

    #[test]
    fn test_default_init_brain() {
        rand::srand(macroquad::miniquad::date::now() as _);

        for _ in 0..1000 {
            let _ = default_init_edges(4);
        }
    }
}
