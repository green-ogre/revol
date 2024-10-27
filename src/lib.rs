#![feature(variant_count)]
#![feature(iter_array_chunks)]

use brain::*;
use core::f32;
use genes::*;
use graph::*;
use macroquad::prelude::*;
use miniquad::{debug, info};
use std::{cmp::Ordering, fmt::Debug, ops::Sub, process::exit};

pub mod brain;
pub mod bsp;
pub mod genes;
pub mod graph;

#[allow(clippy::type_complexity)]
pub struct World {
    dimensions: Dimensions,
    updates_per_second: Ups,
    generation: usize,
    organisms: Vec<Organism>,

    max_internal_neurons: usize,
    edges: usize,
    mutation_rate: f32,

    selection: Option<Box<dyn Fn(Vec<Organism>, &World) -> Vec<Organism>>>,
    fitness: Option<Box<dyn Fn(&Organism, &World) -> f32>>,

    state: WorldState,
}

#[derive(Debug, Clone, Copy)]
pub struct Dimensions(usize, usize);

impl Dimensions {
    pub fn new(width: usize, height: usize) -> Self {
        Self(width, height)
    }

    pub fn width(&self) -> usize {
        self.0
    }

    pub fn height(&self) -> usize {
        self.0
    }

    pub fn area(&self) -> usize {
        self.0 * self.1
    }
}

/// https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
fn default_init_position(target: usize, bounds: Dimensions) -> Vec<Vec2> {
    assert!(target < bounds.area());
    debug!("generating {} default init positions", target);

    let mut all_positions = (0..bounds.area()).collect::<Vec<_>>();
    for i in 0..target {
        let j = rand::gen_range(0, bounds.area() - 1);
        all_positions.swap(i, j);
    }

    let width = bounds.width() as u32;
    all_positions
        .iter()
        .take(target)
        .map(|i| {
            let i = *i as u32;
            let x = i % width;
            let y = i / width;

            Vec2::new(x, y)
        })
        .collect()
}

fn default_init_brain_and_genome(
    num_organisms: usize,
    num_edges: usize,
    max_internal_neurons: usize,
) -> Vec<(Brain, Genome)> {
    (0..num_organisms)
        .map(|_| {
            let edges = brain::default_init_edges(num_edges, max_internal_neurons);
            let genome = Genome::from_edges(&edges);
            let brain = BrainBuilder::new_with_edges(edges).unwrap().build();
            (brain, genome)
        })
        .collect()
}

fn default_reproduction(
    max_internal_neurons: usize,
    mutation_rate: f32,
    mother: Organism,
    father: Organism,
) -> (Vec<Organism>, usize) {
    let mut mutations = 0;
    let mut gen_org = || {
        let mut gen_brain = || {
            let mut depth = 0;
            let mut brain;
            let mut genome;
            loop {
                if depth > 500 {
                    return (None, None);
                }

                let g1 = rand::gen_range(0, 4);
                let g2 = (g1 + 1) % 4;

                let g3 = rand::gen_range(0, 4);
                let g4 = (g3 + 1) % 4;

                let mut genes = vec![
                    mother.genome.genes[g1],
                    mother.genome.genes[g2],
                    father.genome.genes[g3],
                    father.genome.genes[g4],
                ];

                if rand::gen_range(0., 1.) < mutation_rate * 4. {
                    mutations += 1;
                    genes[rand::gen_range(0, 4)].0 ^= 1 << rand::gen_range(0, 63);
                }

                genome = Some(Genome::new(genes));
                brain = BrainBuilder::new(genome.as_ref().unwrap());
                let mut internal = Vec::with_capacity(4);
                if let Some(brain) = &brain {
                    for edge in brain.edges.iter() {
                        if let Neuron::Internal(_) = edge.lhs {
                            if !internal.contains(&edge.lhs) {
                                internal.push(edge.lhs);
                            }
                        }

                        if let Neuron::Internal(_) = edge.rhs {
                            if !internal.contains(&edge.rhs) {
                                internal.push(edge.rhs);
                            }
                        }
                    }

                    if internal.len() > max_internal_neurons {
                        depth += 1;
                        continue;
                    }

                    break;
                }

                depth += 1;
            }

            (brain, genome)
        };

        if let (Some(brain), Some(genome)) = gen_brain() {
            Some(Organism::new(Vec2::default(), genome, brain.build()))
        } else {
            None
        }
    };

    let num_children = rand::gen_range(2, 5);
    let mut children = Vec::with_capacity(num_children);
    for _ in 0..num_children {
        match gen_org() {
            Some(org) => children.push(org),
            None => break,
        }
    }
    (children, mutations)
}

impl Default for World {
    fn default() -> Self {
        Self {
            dimensions: Dimensions::new(128, 128),
            updates_per_second: Ups::AsapRender,
            generation: 0,
            organisms: Vec::new(),

            edges: 4,
            mutation_rate: 1. / 1000.,
            max_internal_neurons: 1,

            selection: None,
            fitness: None,

            state: WorldState::default(),
        }
    }
}

impl World {
    const FACTOR: f32 = 8.;

    pub fn with_dimensions(dimensions: Dimensions) -> Self {
        Self {
            state: WorldState {
                occupation: CellOccupation::new(
                    dimensions.width(),
                    dimensions.height(),
                    Vec::new(),
                ),
            },
            dimensions,
            ..Default::default()
        }
    }

    /// The chance that a gene will mutate one bit.
    pub fn mutation_rate(&mut self, rate: f32) -> &mut Self {
        self.mutation_rate = rate;
        self
    }

    pub fn internal_neurons(&mut self, max_internal_neurons: usize) -> &mut Self {
        self.max_internal_neurons = max_internal_neurons;
        self
    }

    pub fn edges(&mut self, edges: usize) -> &mut Self {
        self.edges = edges;
        self
    }

    pub fn spawn(&mut self, num_organisms: usize) -> &mut Self {
        let positions = default_init_position(num_organisms, self.dimensions);
        assert_eq!(num_organisms, positions.len());

        let brains_and_genomes =
            default_init_brain_and_genome(num_organisms, self.edges, self.max_internal_neurons);
        assert_eq!(num_organisms, brains_and_genomes.len());

        for (p, (b, g)) in positions.into_iter().zip(brains_and_genomes.into_iter()) {
            self.organisms.push(Organism::new(p, g, b));
        }

        for organism in self.organisms.iter() {
            self.state.occupation.set_occupied(organism.position);
        }

        self
    }

    /// Initializes world with `num_organisms` by calling `init_organism`. If `init_organism`
    /// produces an invalid organism, the function will rerun.
    ///
    /// # Panics
    ///
    /// Panics if `init_organism` reaches re-run depth of 10,000
    pub fn spawn_with(
        &mut self,
        num_organisms: usize,
        init_organism: impl Fn(&mut World) -> Organism,
    ) -> &mut Self {
        let mut depth = 0;
        for _ in 0..num_organisms {
            loop {
                if depth >= 10_000 {
                    panic!("spawn_with maximimum depth reached");
                }

                let org = init_organism(self);
                if (org.position.x as usize) < self.dimensions.width()
                    && (org.position.y as usize) < self.dimensions.height()
                {
                    if !self.organisms.iter().any(|o| o.position == org.position) {
                        self.organisms.push(org);
                        break;
                    }
                } else {
                    println!(
                        "WARN: init_organism position {:?} exceeds world bounds of {}x{}",
                        org.position,
                        self.dimensions.width(),
                        self.dimensions.height()
                    );
                }
                depth += 1;
            }

            depth = 0;
        }

        for organism in self.organisms.iter() {
            self.state.occupation.set_occupied(organism.position);
        }

        self
    }

    pub fn with_selection(
        &mut self,
        f: impl Fn(Vec<Organism>, &World) -> Vec<Organism> + 'static,
    ) -> &mut Self {
        self.selection = Some(Box::new(f));
        self
    }

    /// Scores the fitness of an organism used to sort the survivors for reproduction.
    pub fn with_fitness_rating(
        &mut self,
        f: impl Fn(&Organism, &World) -> f32 + 'static,
    ) -> &mut Self {
        self.fitness = Some(Box::new(f));
        self
    }

    pub fn with_ups(&mut self, updates_per_second: Ups) -> &mut Self {
        self.updates_per_second = updates_per_second;
        self
    }

    pub async fn run(&mut self, generations: usize, ticks_per_generation: usize) {
        assert!(self.selection.is_some(), "no selection function provided");
        assert!(self.fitness.is_some(), "no fitness function provided");

        request_new_screen_size(
            Self::FACTOR * self.dimensions.width() as f32,
            Self::FACTOR * self.dimensions.height() as f32,
        );

        let starting_num_org = self.organisms.len();

        let mut gen_ticks = 0;
        let mut ticks = 0;
        let mut is_pressed = false;
        let mut render_fittest_graph = false;
        loop {
            if is_key_pressed(KeyCode::Escape) {
                return;
            }

            if is_key_pressed(KeyCode::G) {
                // self.render_average_edge_graph().await;
                render_fittest_graph = true;
            }

            if is_key_down(KeyCode::Space) {
                if !is_pressed {
                    match self.updates_per_second {
                        Ups::Asap => {
                            self.updates_per_second = Ups::AsapRender;
                        }
                        Ups::AsapRender => {
                            self.updates_per_second = Ups::Asap;
                        }
                        _ => {}
                    }
                    is_pressed = true;
                }
            }

            if is_key_released(KeyCode::Space) {
                match self.updates_per_second {
                    Ups::Asap => {
                        self.updates_per_second = Ups::AsapRender;
                    }
                    Ups::AsapRender => {
                        self.updates_per_second = Ups::Asap;
                    }
                    _ => {}
                }
                is_pressed = false;
            }

            if self.generation < generations {
                match self.updates_per_second {
                    Ups::Asap | Ups::AsapRender => {
                        self.step();
                        gen_ticks += 1;
                    }
                    Ups::Ups(ups) => {
                        if ticks as f32 / get_fps() as f32 >= 1. / ups as f32 {
                            self.step();
                            ticks = 0;
                            gen_ticks += 1;
                        }
                        ticks += 1;
                    }
                }

                if gen_ticks >= ticks_per_generation {
                    // let prange = 8;
                    // assert!(self.dimensions.width() % prange == 0);
                    // assert!(self.dimensions.height() % prange == 0);

                    // let mut bsp = Bsp::new(
                    //     self.dimensions.width() / prange,
                    //     self.dimensions.height() / prange,
                    //     prange,
                    //     prange,
                    // );
                    // for organism in self.organisms.drain(..) {
                    //     bsp.push_entity(organism.position, organism);
                    // }

                    let num_organisms = self.organisms.len();

                    let mut survivors = (self.selection.as_ref().unwrap())(
                        self.organisms.drain(..).collect(),
                        self,
                    );

                    let num_survivors = survivors.len();

                    if let Ups::Asap = self.updates_per_second {
                        for org in survivors.drain(..).rev() {
                            self.organisms.push(org);
                        }
                        self.render().await;
                        for org in self.organisms.drain(..).rev() {
                            survivors.push(org);
                        }
                    }

                    survivors.sort_by(|o1, o2| {
                        ((self.fitness.as_ref().unwrap())(o1, self))
                            .partial_cmp(&(self.fitness.as_ref().unwrap())(o2, self))
                            .unwrap_or(Ordering::Equal)
                    });

                    if render_fittest_graph {
                        self.render_organisms_edge_graph(&survivors).await;
                        render_fittest_graph = false;
                    }

                    let mut generation_mutations = 0;
                    let mut tmp_organisms = Vec::with_capacity(starting_num_org * 2);
                    for [mother, father] in survivors
                        .into_iter()
                        .take(starting_num_org)
                        .array_chunks::<2>()
                    {
                        let (orgs, mutations) = default_reproduction(
                            self.max_internal_neurons,
                            self.mutation_rate,
                            mother,
                            father,
                        );
                        generation_mutations += mutations;

                        for mut org in orgs.into_iter() {
                            org.position = self.random_pos(&tmp_organisms);
                            tmp_organisms.push(org);
                        }
                    }

                    let survival_rate = num_survivors as f32 / num_organisms as f32;
                    info!(
                        "Generation {}: survival rate [{:.2}%] [{}/{}] - mutations [{}] - mutation rate [{:.4}]",
                        self.generation,
                        survival_rate * 100.,
                        num_survivors,
                        num_organisms,
                        generation_mutations,
                        generation_mutations as f32 / num_survivors as f32,
                    );

                    for org in tmp_organisms.into_iter() {
                        self.organisms.push(org);
                    }

                    self.state.occupation.clear();
                    for organism in self.organisms.iter() {
                        self.state.occupation.set_occupied(organism.position);
                    }

                    self.generation += 1;
                    gen_ticks = 0;
                }
            }

            match self.updates_per_second {
                Ups::AsapRender | Ups::Ups(_) => {
                    self.render().await;
                }
                _ => {}
            }
        }
    }

    fn random_pos(&self, organisms: &[Organism]) -> Vec2 {
        loop {
            let pos = Vec2::new(
                rand::gen_range(0, self.dimensions.width() as u32),
                rand::gen_range(0, self.dimensions.height() as u32),
            );

            if !organisms.iter().any(|org| org.position == pos) {
                return pos;
            }
        }
    }

    fn step(&mut self) {
        for organism in self.organisms.iter_mut() {
            organism.update(&mut self.state);
        }
    }

    async fn render(&self) {
        clear_background(BLACK);
        self.draw_organisms();
        draw_text(&format!("fps: {}", get_fps()), 20., 20., 32., WHITE);
        draw_text(
            &format!("generation: {}", self.generation),
            180.,
            20.,
            32.,
            WHITE,
        );
        next_frame().await
    }

    fn draw_organisms(&self) {
        for organism in self.organisms.iter() {
            // let brightness = 2.;
            // let mut bits = 0;
            //
            // let gene = organism.genome.genes[0].0;
            // for i in 0..32 {
            //     bits += (gene >> i) & 1;
            // }
            // let r = bits as f32 / 32. * brightness;
            //
            // bits = 0;
            // let gene = organism.genome.genes[1].0;
            // for i in 0..32 {
            //     bits += (gene >> i) & 1;
            // }
            // let g = bits as f32 / 32. * brightness;
            //
            // bits = 0;
            // let gene = organism.genome.genes[2].0;
            // for i in 0..32 {
            //     bits += (gene >> i) & 1;
            // }
            // let b = bits as f32 / 32. * brightness;

            let r = organism.genome.genes[0].0 as f32 / u64::MAX as f32;
            let g = organism.genome.genes[1].0 as f32 / u64::MAX as f32;
            let b = organism.genome.genes[2].0 as f32 / u64::MAX as f32;

            draw_rectangle(
                organism.position.x as f32 * Self::FACTOR,
                organism.position.y as f32 * Self::FACTOR,
                Self::FACTOR,
                Self::FACTOR,
                Color { r, g, b, a: 1. },
            );
        }
    }

    async fn render_organisms_edge_graph(&self, organisms: &[Organism]) {
        render_many_force_graph_edges(
            organisms,
            &organisms.iter().map(|org| &org.brain).collect::<Vec<_>>(),
            self,
        )
        .await;
    }

    async fn render_average_edge_graph(&self) {
        let mut freq: Vec<(NeuralCon, usize)> =
            Vec::with_capacity(self.organisms.len() * self.edges);

        for org in self.organisms.iter() {
            for edge in org.brain.edges.iter() {
                if let Some((_, freq)) = freq.iter_mut().find(|(e, _)| {
                    (e.lhs == edge.lhs || e.rhs == edge.lhs)
                        || (e.lhs == edge.rhs || e.rhs == edge.rhs)
                }) {
                    *freq += 1;
                } else {
                    freq.push((*edge, 1));
                }
            }
        }

        freq.sort_by(|(_, f1), (_, f2)| f1.partial_cmp(f2).unwrap_or(Ordering::Equal));

        render_force_graph_edges(
            &freq
                .iter()
                .take(self.edges)
                .map(|(e, _)| *e)
                .collect::<Vec<_>>(),
        )
        .await;
    }
}

#[derive(Debug, Default)]
pub struct WorldState {
    pub occupation: CellOccupation,
}

impl WorldState {
    pub fn dimensions(&self) -> (usize, usize) {
        (self.occupation.width, self.occupation.height)
    }

    pub fn width(&self) -> usize {
        self.occupation.width
    }

    pub fn height(&self) -> usize {
        self.occupation.height
    }
}

#[derive(Debug, Default)]
pub struct CellOccupation {
    width: usize,
    height: usize,
    cells: Vec<bool>,
}

impl CellOccupation {
    pub fn new(width: usize, height: usize, occupations: Vec<Vec2>) -> Self {
        let mut cells = vec![false; width * height];
        for occ in occupations.iter() {
            let index = occ.x as usize + occ.y as usize * width;
            cells[index] = true;
        }

        Self {
            cells,
            width,
            height,
        }
    }

    pub fn clear(&mut self) {
        for cell in self.cells.iter_mut() {
            *cell = false;
        }
    }

    pub fn set_occupied(&mut self, location: Vec2) {
        let index = location.x as usize + location.y as usize * self.width;
        self.cells[index] = true;
    }

    pub fn is_occupied(&self, location: Vec2) -> bool {
        let index = location.x as usize + location.y as usize * self.width;
        self.cells[index]
    }

    #[allow(clippy::result_unit_err)]
    pub fn try_move(&mut self, src: Vec2, dst: Vec2) -> Result<(), ()> {
        let src = src.x as usize + src.y as usize * self.width;
        let dst = dst.x as usize + dst.y as usize * self.width;

        if !self.cells[src] {
            panic!("tried to move unoccupied cell");
        }

        if !self.cells[dst] {
            self.cells[dst] = true;
            self.cells[src] = false;
            Ok(())
        } else {
            Err(())
        }
    }
}

#[derive(Debug, Default)]
pub struct Organism {
    pub position: Vec2,
    pub genome: Genome,
    pub brain: Brain,
}

impl Organism {
    #[allow(clippy::new_without_default)]
    pub fn new(position: Vec2, genome: Genome, brain: Brain) -> Self {
        Self {
            position,
            genome,
            brain,
        }
    }

    pub fn update(&mut self, state: &mut WorldState) {
        for output in self.brain.outputs.iter() {
            if output.should_fire(self, state) {
                match output.neuron {
                    Neuron::Action(act) => act.fire(&mut self.position, state),
                    n => {
                        panic!("non action neuron in output: {:?}", n);
                    }
                }
            }
        }
    }
}

#[derive(Debug, Default)]
pub enum Ups {
    #[default]
    Asap,
    AsapRender,
    Ups(u32),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Vec2 {
    pub x: u32,
    pub y: u32,
}

impl Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Vec2 {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }

    pub fn dist2(self, other: Self) -> u32 {
        let diff = other - self;
        diff.x * diff.x + diff.y * diff.y
    }
}

pub async fn run() {
    pretty_env_logger::init();
    rand::srand(macroquad::miniquad::date::now() as _);

    World::with_dimensions(Dimensions::new(128, 128))
        .mutation_rate(1. / 1000.)
        // .mutation_rate(0.)
        .internal_neurons(1)
        .edges(4)
        .spawn(200)
        .with_selection(|mut organisms: Vec<Organism>, world: &World| {
            organisms.retain(|org| org.position.x > world.dimensions.width() as u32 / 2);
            organisms
        })
        .with_fitness_rating(|org: &Organism, world: &World| {
            org.position.x as f32 / world.dimensions.width() as f32
        })
        .with_ups(Ups::AsapRender)
        .run(usize::MAX, 200)
        .await;
}
