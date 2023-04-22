use std::ops::{AddAssign, Div, Mul, Sub, SubAssign};

use crate::vector::Normed;

/// A brute-force [`ComputeMethod`](super::ComputeMethod) using the CPU.
pub struct BruteForce;

impl<P, S, V> super::ComputeMethod<P, V, S> for BruteForce
where
    P: Copy + Sub<Output = V>,
    V: Copy
        + Default
        + AddAssign
        + SubAssign
        + Mul<S, Output = V>
        + Div<S, Output = V>
        + Normed<Output = S>,
    S: Copy + Default + PartialEq + Mul<Output = S>,
{
    #[inline]
    fn compute(&mut self, particles: &[(P, S)]) -> Vec<V> {
        let (massive, massless): (Vec<_>, Vec<_>) =
            particles.iter().partition(|(_, mu)| *mu != S::default());

        let massive_len = massive.len();

        let concat = &[massive, massless].concat()[..];
        let len = concat.len();

        let mut accelerations = vec![V::default(); len];

        for i in 0..massive_len {
            let (pos1, mu1) = concat[i];
            let mut acceleration = V::default();

            for j in (i + 1)..len {
                let (pos2, mu2) = concat[j];

                let dir = pos2 - pos1;
                let mag_2 = dir.length_squared();

                let f = dir / (mag_2 * V::sqrt(mag_2));

                acceleration += f * mu2;
                accelerations[j] -= f * mu1;
            }

            accelerations[i] += acceleration;
        }

        let (mut massive_acc, mut massless_acc) = {
            let remainder = accelerations.split_off(massive_len);

            (accelerations.into_iter(), remainder.into_iter())
        };

        particles
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

impl<T, S, O> super::ComputeMethod<T, T, S> for BarnesHut<S>
where
    T: Copy + Default,
    S: Copy + Default + PartialEq,
    (T, S): Copy + TreeData,
    Tree<O, (T, S)>: TreeBuilder<BoundingBox<T>, (T, S)> + TreeAcceleration<T, S>,
    BoundingBox<T>: BoundingBoxExtend<Vector = T, Orthant = O>,
{
    fn compute(&mut self, particles: &[(T, S)]) -> Vec<T> {
        let mut tree = Tree::default();

        let massive: Vec<_> = particles
            .iter()
            .filter(|(_, mu)| *mu != S::default())
            .copied()
            .collect();

        let bbox = BoundingBox::containing(massive.iter().map(|p| p.0));
        let root = tree.build_node(massive, bbox);

        particles
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
