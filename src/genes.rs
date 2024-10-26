use std::fmt::Debug;

#[derive(Debug, Default, Clone)]
pub struct Genome {
    pub genes: Vec<Gene>,
}

impl Genome {
    pub fn new(genes: Vec<Gene>) -> Self {
        Self { genes }
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
