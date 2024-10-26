#![allow(unused)]

use crate::Vec2;
use std::ops::Range;

#[derive(Debug, Default)]
pub struct Bsp<T> {
    pub width: usize,
    pub height: usize,
    x_partitions: usize,
    y_partitions: usize,
    partition_range_x: usize,
    partition_range_y: usize,
    partitions: Vec<Partition<T>>,
}

impl<T> Bsp<T> {
    pub fn new(
        x_partitions: usize,
        y_partitions: usize,
        partition_range_x: usize,
        partition_range_y: usize,
    ) -> Self {
        let mut partitions = Vec::with_capacity(x_partitions * y_partitions);
        for y in 0..y_partitions {
            let y_range = (partition_range_y * y) as u32;
            for x in 0..x_partitions {
                let x_range = (partition_range_x * x) as u32;
                partitions.push(Partition::new(
                    x_range..x_range + partition_range_x as u32,
                    y_range..y_range + partition_range_y as u32,
                ));
            }
        }

        Self {
            width: x_partitions * partition_range_x,
            height: y_partitions * partition_range_y,
            x_partitions,
            y_partitions,
            partition_range_x,
            partition_range_y,
            partitions,
        }
    }

    pub fn push_entity(&mut self, position: Vec2, entity: T) {
        let index = self.index(position);
        self.partitions[index].push_entity(entity, position);
    }

    pub fn get_entity(&self, position: Vec2) -> Option<&T> {
        let index = self.index(position);
        self.partitions[index].get_entity(position)
    }

    pub fn get_entity_mut(&mut self, position: Vec2) -> Option<&mut T> {
        let index = self.index(position);
        self.partitions[index].get_entity_mut(position)
    }

    pub fn take_entity(&mut self, position: Vec2) -> Option<T> {
        let index = self.index(position);
        self.partitions[index].take_entity(position)
    }

    pub fn take_first_entity(&mut self) -> Option<T> {
        if let Some(p) = self
            .partitions
            .iter()
            .flat_map(|p| p.entities.iter().map(|(_, p)| *p))
            .next()
        {
            self.take_entity(p)
        } else {
            None
        }
    }

    pub fn take_closest_entity(&mut self, position: Vec2) -> Option<T> {
        let index = self.index(position);
        if let Some(p) = self.partitions[index].closest_position(position) {
            self.take_entity(p)
        } else {
            None
        }
    }

    pub fn iter_entities(&self) -> impl Iterator<Item = &T> {
        self.partitions
            .iter()
            .flat_map(|p| p.entities.iter().map(|(e, _)| e))
    }

    pub fn iter_entities_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.partitions
            .iter_mut()
            .flat_map(|p| p.entities.iter_mut().map(|(e, _)| e))
    }

    fn index(&self, position: Vec2) -> usize {
        assert!(position.x < self.width as u32, "bsp index x out of range");
        assert!(position.y < self.height as u32, "bsp index y out of range");

        (position.x as usize / self.partition_range_x)
            + (position.y as usize / self.partition_range_y) * self.x_partitions
    }
}

#[derive(Debug)]
pub struct Partition<T> {
    x_axis: Range<u32>,
    y_axis: Range<u32>,
    entities: Vec<(T, Vec2)>,
}

impl<T> Partition<T> {
    pub fn new(x_axis: Range<u32>, y_axis: Range<u32>) -> Self {
        Self {
            entities: Vec::new(),
            x_axis,
            y_axis,
        }
    }

    pub fn push_entity(&mut self, entity: T, position: Vec2) {
        self.entities.push((entity, position));
    }

    pub fn get_entity(&self, position: Vec2) -> Option<&T> {
        self.entities
            .iter()
            .find_map(|(e, p)| if *p == position { Some(e) } else { None })
    }

    pub fn get_entity_mut(&mut self, position: Vec2) -> Option<&mut T> {
        self.entities
            .iter_mut()
            .find_map(|(e, p)| if *p == position { Some(e) } else { None })
    }

    pub fn take_entity(&mut self, position: Vec2) -> Option<T> {
        self.entities
            .iter()
            .enumerate()
            .find_map(|(i, (_, p))| if *p == position { Some(i) } else { None })
            .and_then(|i| Some(self.entities.remove(i).0))
    }

    pub fn closest_position(&self, position: Vec2) -> Option<Vec2> {
        if self.entities.len() == 0 {
            None
        } else {
            let result = self
                .entities
                .iter()
                .map(|(_, p)| (p, p.dist2(position)))
                .min_by(|&(_, d1), &(_, d2)| d1.cmp(&d2));
            result.map(|(p, _)| *p)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bsp() {
        let mut bsp: Bsp<usize> = Bsp::new(3, 2, 2, 2);
        bsp.push_entity(Vec2::new(4, 2), 69);
        assert_eq!(69, bsp.partitions[5].entities[0].0);
        assert_eq!(69, *bsp.get_entity(Vec2::new(4, 2)).unwrap());
        assert!(bsp.get_entity(Vec2::new(4, 1)).is_none());
        bsp.push_entity(Vec2::new(5, 0), 420);
        assert_eq!(420, bsp.partitions[2].entities[0].0);
        assert_eq!(420, *bsp.get_entity(Vec2::new(5, 0)).unwrap());
        assert!(bsp.get_entity(Vec2::new(4, 0)).is_none());
        // println!("{bsp:#?}");
    }
}
