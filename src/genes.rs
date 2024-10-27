use crate::{ActionNeuron, InternalNeuron, NeuralCon, Neuron, SenseNeuron};
use std::fmt::Debug;

#[derive(Debug, Default)]
pub struct Genome {
    pub genes: Vec<Gene>,
}

impl Genome {
    pub fn new(genes: Vec<Gene>) -> Self {
        Self { genes }
    }

    pub fn from_edges(edges: &[NeuralCon]) -> Self {
        let mut genes = Vec::with_capacity(edges.len());

        for edge in edges.iter() {
            let gene = Gene::from_edge(edge);
            genes.push(gene);
        }

        Self::new(genes)
    }
}

#[derive(Clone, Copy)]
pub struct Gene(pub u64);

impl Debug for Gene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Gene")
            .field(&format_args!("{:#018x}", self.0))
            .finish()
    }
}

impl Gene {
    pub fn from_edge(edge: &NeuralCon) -> Self {
        let gen_noise = |variants: usize| {
            let noise = macroquad::rand::gen_range(0, 0x3FFF) as u32;
            let remainder = noise % variants as u32;
            if remainder == 0 {
                noise
            } else {
                noise + (variants as u32 - remainder)
            }
        };

        let (lhs, lhs_type) = match edge.lhs {
            Neuron::Sense(sense) => (
                gen_noise(std::mem::variant_count::<SenseNeuron>()) + sense as u32,
                0,
            ),
            Neuron::Internal(int) => (
                gen_noise(std::mem::variant_count::<InternalNeuron>()) + int as u32,
                1,
            ),
            Neuron::Action(act) => (
                gen_noise(std::mem::variant_count::<ActionNeuron>()) + act as u32,
                2,
            ),
        };

        let (rhs, rhs_type) = match edge.rhs {
            Neuron::Sense(sense) => (
                gen_noise(std::mem::variant_count::<SenseNeuron>()) + sense as u32,
                0,
            ),
            Neuron::Internal(int) => (
                gen_noise(std::mem::variant_count::<InternalNeuron>()) + int as u32,
                1,
            ),
            Neuron::Action(act) => (
                gen_noise(std::mem::variant_count::<ActionNeuron>()) + act as u32,
                2,
            ),
        };

        let assoc = edge.assoc.weight();

        let mut gene = 0;
        gene |= (((assoc + 4.0) / 8.0) * u32::MAX as f32) as u64;

        gene |= lhs_type << 62;
        gene |= (lhs as u64 & 0x3FFF) << (62 - 14);

        gene |= rhs_type << (62 - 16);
        gene |= (rhs as u64 & 0x3FFF) << (62 - 14 - 16);

        Self(gene)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Association, InternalNeuron, SenseNeuron};

    #[test]
    fn gene_edge_conversion() {
        for _ in 0..10_000 {
            let edge = NeuralCon::new(
                Association::Positive(macroquad::rand::gen_range(-4.0, 4.0)),
                Neuron::Sense(SenseNeuron::new_from_u32(macroquad::rand::gen_range(
                    0,
                    u32::MAX,
                ))),
                Neuron::Internal(InternalNeuron::new_from_u32(macroquad::rand::gen_range(
                    0,
                    u32::MAX,
                ))),
            )
            .unwrap();
            let gene = Gene::from_edge(&edge);
            let gene_edge = NeuralCon::from_gene(gene).unwrap();
            assert_eq!(edge.lhs, gene_edge.lhs);
            assert_eq!(edge.rhs, gene_edge.rhs);
            assert!((edge.assoc.weight() - gene_edge.assoc.weight()).abs() < 0.001);
        }
    }
}
