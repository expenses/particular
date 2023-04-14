use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

use crate::{compute_method::WithMassive, vector::Normed};

/// A brute-force [`ComputeMethod`](super::ComputeMethod) using the CPU.
pub struct BruteForce;

impl<T, S> super::ComputeMethod<Vec<(T, S)>> for BruteForce
where
    T: Copy
        + Default
        + AddAssign
        + SubAssign
        + Sub<Output = T>
        + Mul<S, Output = T>
        + Div<S, Output = T>
        + Normed<Output = S>,
    S: Copy + Default + PartialEq + Mul<Output = S>,
{
    #[inline]
    fn compute(&mut self, storage: Vec<(T, S)>) -> Vec<T> {
        let (massive, massless) = storage
            .iter()
            .partition::<Vec<_>, _>(|(_, mu)| *mu != S::default());

        let massive_len = massive.len();

        let concat = &[massive, massless].concat()[..];
        let len = concat.len();

        let mut accelerations = vec![T::default(); len];

        for i in 0..massive_len {
            let (pos1, mu1) = concat[i];
            let mut acceleration = T::default();

            for j in (i + 1)..len {
                let (pos2, mu2) = concat[j];

                let dir = pos2 - pos1;
                let mag_2 = dir.length_squared();

                let f = dir / (mag_2 * T::sqrt(mag_2));

                acceleration += f * mu2;
                accelerations[j] -= f * mu1;
            }

            accelerations[i] += acceleration;
        }

        let (mut massive_acc, mut massless_acc) = {
            let remainder = accelerations.split_off(massive_len);

            (accelerations.into_iter(), remainder.into_iter())
        };

        storage
            .iter()
            .filter_map(|(_, mu)| {
                if *mu != S::default() {
                    massive_acc.next()
                } else {
                    massless_acc.next()
                }
            })
            .collect()
    }
}

/// A brute-force [`ComputeMethod`](super::ComputeMethod) using the CPU.
///
/// This differs from [`BruteForce`] by not iterating over the combinations of pair of particles, making it slower.
pub struct BruteForceAlt;

impl<T, S> super::ComputeMethod<WithMassive<T, S>> for BruteForceAlt
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<S, Output = T>
        + Div<S, Output = T>
        + Normed<Output = S>,
    S: Copy + Default + PartialEq + Mul<Output = S>,
{
    #[inline]
    fn compute(&mut self, storage: WithMassive<T, S>) -> Vec<T> {
        storage
            .particles
            .iter()
            .map(|&(position1, _)| {
                storage
                    .massive
                    .iter()
                    .fold(T::default(), |acceleration, &(position2, mass2)| {
                        let dir = position2 - position1;
                        let mag_2 = dir.length_squared();

                        let grav_acc = if mag_2 != S::default() {
                            dir * mass2 / (mag_2 * T::sqrt(mag_2))
                        } else {
                            dir
                        };

                        acceleration + grav_acc
                    })
            })
            .collect()
    }
}

use super::tree::{
    acceleration::TreeAcceleration,
    bbox::{BoundingBox, BoundingBoxExtend},
    Tree, TreeBuilder, TreeData,
};

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) [`ComputeMethod`](super::ComputeMethod) using the CPU.
pub struct BarnesHut<S> {
    /// Parameter ruling the accuracy and speed of the algorithm. If 0, behaves the same as [`BruteForce`].
    pub theta: S,
}

impl<T, S, O> super::ComputeMethod<WithMassive<T, S>> for BarnesHut<S>
where
    T: Copy + Default,
    S: Copy + Default + PartialEq,
    (T, S): Copy + TreeData,
    Tree<O, (T, S)>: TreeBuilder<BoundingBox<T>, (T, S)> + TreeAcceleration<T, S>,
    BoundingBox<T>: BoundingBoxExtend<Vector = T, Orthant = O>,
{
    fn compute(&mut self, storage: WithMassive<T, S>) -> Vec<T> {
        let mut tree = Tree::default();

        let bbox = BoundingBox::containing(storage.massive.iter().map(|p| p.0));
        let root = tree.build_node(storage.massive, bbox);

        storage
            .particles
            .iter()
            .map(|&(position, _)| tree.acceleration_at(position, root, self.theta))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests;
    use super::*;

    #[test]
    fn brute_force() {
        tests::acceleration_computation(BruteForce);
    }

    #[test]
    fn barnes_hut() {
        tests::acceleration_computation(BarnesHut { theta: 0.0 });
    }
}
