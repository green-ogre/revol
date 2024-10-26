#![feature(variant_count)]

use brain::*;
use bsp::Bsp;
use core::f32;
use genes::*;
use macroquad::{prelude::*, rand};
use std::{fmt::Debug, ops::Sub};

pub mod brain;
pub mod bsp;
pub mod genes;

#[allow(clippy::type_complexity)]
pub struct World {
    width: usize,
    height: usize,
    updates_per_second: Ups,
    generation: usize,
    organisms: Vec<Organism>,
    selection: Option<Box<dyn Fn(Bsp<Organism>) -> Vec<(Organism, Organism)>>>,
    reproduction: Option<Box<dyn Fn(Organism, Organism) -> Vec<Organism>>>,
    state: WorldState,
}

impl Default for World {
    fn default() -> Self {
        Self {
            width: 128,
            height: 128,
            updates_per_second: Ups::ASAP,
            generation: 0,
            organisms: Vec::new(),
            selection: None,
            reproduction: None,
            state: WorldState::default(),
        }
    }
}

impl World {
    const FACTOR: f32 = 8.;

    pub fn with_dimensions(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            state: WorldState {
                occupation: CellOccupation::new(width, height, Vec::new()),
                ..Default::default()
            },
            ..Default::default()
        }
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
                if (org.position.x as usize) < self.width && (org.position.y as usize) < self.height
                {
                    if !self.organisms.iter().any(|o| o.position == org.position) {
                        self.organisms.push(org);
                        break;
                    }
                } else {
                    println!(
                        "WARN: init_organism position {:?} exceeds world bounds of {}x{}",
                        org.position, self.width, self.height
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
        f: impl Fn(Bsp<Organism>) -> Vec<(Organism, Organism)> + 'static,
    ) -> &mut Self {
        self.selection = Some(Box::new(f));
        self
    }

    /// The positions of the children organisms are overwritten.
    pub fn with_reproduction(
        &mut self,
        f: impl Fn(Organism, Organism) -> Vec<Organism> + 'static,
    ) -> &mut Self {
        self.reproduction = Some(Box::new(f));
        self
    }

    pub fn with_ups(&mut self, updates_per_second: Ups) -> &mut Self {
        self.updates_per_second = updates_per_second;
        self
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub async fn run(&mut self, generations: usize, ticks_per_generation: usize) {
        assert!(self.selection.is_some(), "no selection function provided");
        assert!(
            self.reproduction.is_some(),
            "no reproduction function provided"
        );

        // for org in self.organisms.iter().take(1) {
        //     println!("{org:#?}");
        // }

        request_new_screen_size(
            Self::FACTOR * self.width as f32,
            Self::FACTOR * self.height as f32,
        );

        let starting_num_org = self.organisms.len();

        let mut gen_ticks = 0;
        let mut ticks = 0;
        loop {
            if is_key_down(KeyCode::Escape) {
                return;
            }

            if self.generation < generations {
                match self.updates_per_second {
                    Ups::ASAP => {
                        self.step();
                        gen_ticks += 1;
                    }
                    Ups::UPS(ups) => {
                        if ticks as f32 / get_fps() as f32 >= 1. / ups as f32 {
                            self.step();
                            ticks = 0;
                            gen_ticks += 1;
                        }
                        ticks += 1;
                    }
                }

                if gen_ticks >= ticks_per_generation {
                    let prange = 8;
                    assert!(self.width % prange == 0);
                    assert!(self.height % prange == 0);

                    let mut bsp =
                        Bsp::new(self.width / prange, self.height / prange, prange, prange);
                    for organism in self.organisms.drain(..) {
                        bsp.push_entity(organism.position, organism);
                    }
                    let couples = (self.selection.as_ref().unwrap())(bsp);

                    // for (o1, o2) in couples.into_iter() {
                    //     self.organisms.push(o1);
                    //     self.organisms.push(o2);
                    // }
                    // self.generation = usize::MAX;
                    // continue;

                    let reproduction = self.reproduction.as_ref().unwrap();

                    let mut tmp_organisms = Vec::with_capacity(self.organisms.len() * 2);
                    for couple in couples.into_iter() {
                        let orgs = reproduction(couple.0, couple.1);

                        for mut org in orgs.into_iter() {
                            org.position = self.random_pos(&tmp_organisms);
                            tmp_organisms.push(Some(org));
                        }
                    }

                    let len = tmp_organisms.len();
                    let mut indexes = (0..len).collect::<Vec<_>>();
                    for _ in 0..starting_num_org / 2 {
                        indexes.swap(rand::gen_range(0, len - 1), rand::gen_range(0, len - 1));
                    }

                    // println!(
                    //     "{:?}",
                    //     indexes.iter().take(starting_num_org).collect::<Vec<_>>()
                    // );

                    for index in indexes.iter().take(starting_num_org) {
                        self.organisms.push(tmp_organisms[*index].take().unwrap());
                    }

                    self.state.occupation.clear();
                    for organism in self.organisms.iter() {
                        self.state.occupation.set_occupied(organism.position);
                    }

                    self.generation += 1;
                    gen_ticks = 0;
                }
            }

            self.render().await;
        }
    }

    fn random_pos(&self, organisms: &[Option<Organism>]) -> Vec2 {
        loop {
            let pos = Vec2::new(
                rand::gen_range(0, self.width as u32),
                rand::gen_range(0, self.height as u32),
            );

            if !organisms
                .iter()
                .any(|org| org.as_ref().unwrap().position == pos)
            {
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
    ASAP,
    UPS(u32),
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

#[macroquad::main("Revol")]
async fn main() {
    rand::srand(macroquad::miniquad::date::now() as _);

    World::with_dimensions(128, 128)
        // TODO: stack overflow on startup sometimes
        .spawn_with(200, |world: &mut World| {
            let (width, height) = world.dimensions();
            let position = Vec2::new(
                rand::gen_range(0, width as u32),
                rand::gen_range(0, height as u32),
            );

            let gen_genes = || {
                (0..4)
                    .map(|_| {
                        // gen_range for u64 only generates top 32 bits???
                        Gene(rand::gen_range(0, u32::MAX) as u64 | rand::gen_range(0, u64::MAX))
                    })
                    .collect()
            };

            let mut genome = Genome::new(gen_genes());
            let mut brain = BrainBuilder::new(genome.clone());
            loop {
                if brain.is_some() {
                    break;
                }

                genome = Genome::new(gen_genes());
                brain = BrainBuilder::new(genome.clone());
            }
            let brain = brain.unwrap().build();

            Organism::new(position, genome, brain)
        })
        .with_selection(|mut bsp: Bsp<Organism>| {
            let mut couples = Vec::new();

            while let Some(org1) = bsp.take_first_entity() {
                if (org1.position.x as usize) < bsp.width / 2 {
                    continue;
                }

                while let Some(org2) = bsp.take_first_entity() {
                    if (org2.position.x as usize) < bsp.width / 2 {
                        continue;
                    }

                    couples.push((org1, org2));
                    break;
                }
            }

            couples
        })
        .with_reproduction(|mother: Organism, father: Organism| {
            let gen_org = || {
                let gen_genes = || {
                    let g1 = rand::gen_range(0, 3);
                    let mut g2 = rand::gen_range(0, 3);
                    while g1 == g2 {
                        g2 = rand::gen_range(0, 3);
                    }

                    let g3 = rand::gen_range(0, 3);
                    let mut g4 = rand::gen_range(0, 3);
                    while g3 == g4 {
                        g4 = rand::gen_range(0, 3);
                    }

                    let mut genes = vec![
                        mother.genome.genes[g1],
                        mother.genome.genes[g2],
                        father.genome.genes[g3],
                        father.genome.genes[g4],
                    ];

                    if rand::gen_range(0, 4) == 0 {
                        genes[rand::gen_range(0, 3)].0 ^= 1 << rand::gen_range(0, 63);
                    }

                    genes
                };

                let mut genome = Genome::new(gen_genes());
                let mut brain = BrainBuilder::new(genome.clone());
                let mut depth = 0;
                loop {
                    if brain.is_some() {
                        break;
                    }

                    genome = Genome::new(if depth > 500 {
                        // gen_genes_fallback()
                        return None;
                    } else {
                        gen_genes()
                    });

                    brain = BrainBuilder::new(genome.clone());

                    depth += 1;
                }
                let brain = brain.unwrap().build();

                Some(Organism::new(Vec2::default(), genome, brain))
            };

            let num_children = rand::gen_range(2, 6);
            let mut children = Vec::with_capacity(num_children);
            for _ in 0..num_children {
                match gen_org() {
                    Some(org) => children.push(org),
                    None => break,
                }
            }
            children
        })
        .with_ups(Ups::ASAP)
        .run(usize::MAX, 200)
        .await;
}
