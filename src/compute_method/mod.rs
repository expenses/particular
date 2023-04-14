mod tree;
#[cfg(feature = "gpu")]
mod wgpu;

#[cfg(feature = "gpu")]
/// Compute methods that use the GPU.
pub mod gpu;

#[cfg(feature = "parallel")]
/// Compute methods that use multiple CPU threads.
pub mod parallel;

/// Compute methods that use one CPU thread.
pub mod sequential;

/// Trait for algorithms computing the gravitational forces between [`Particles`](crate::particle::Particle).
///
/// To implement it, specify an internal vector representation and its scalar type.
///
/// # Example
///
/// ```
/// # use particular::prelude::*;
/// # use glam::Vec3A;
/// struct AccelerationCalculator;
///
/// impl ComputeMethod<Vec<(Vec3A, f32)>> for AccelerationCalculator {
///     fn compute(&mut self, particles: Vec<(Vec3A, f32)>) -> Vec<Vec3A> {
///     // ...
/// #       Vec::new()
///     }
/// }
/// ```
pub trait ComputeMethod<B: ParticleStorage> {
    /// Computes the acceleration of the particles.
    ///
    /// The returning vector should contain the acceleration of the particles in the same order they were input.
    fn compute(&mut self, storage: B) -> Vec<B::InternalVector>;
}

/// Storage used by [`ComputeMethod`] to access the particles.
pub trait ParticleStorage {
    /// The interal vector used for the stored particles.
    type InternalVector;

    /// The scalar type used for the stored particles.
    type Scalar;

    /// Creates an instance of a type implementing [`ParticleStorage`] from a vector of particles.
    fn new(particles: Vec<(Self::InternalVector, Self::Scalar)>) -> Self;
}

impl<T, S> ParticleStorage for Vec<(T, S)> {
    type InternalVector = T;

    type Scalar = S;

    fn new(particles: Vec<(T, S)>) -> Self {
        particles
    }
}

/// Storage for particles with a copy of the massives ones in a second vector.
pub struct WithMassive<T, S> {
    /// Particles for which the acceleration is computed.
    pub particles: Vec<(T, S)>,

    /// Particles used to compute the acceleration of the `particles`.
    pub massive: Vec<(T, S)>,
}

impl<T, S> ParticleStorage for WithMassive<T, S>
where
    T: Copy,
    S: Copy + Default + PartialEq,
{
    type InternalVector = T;

    type Scalar = S;

    fn new(particles: Vec<(T, S)>) -> Self {
        let massive: Vec<_> = particles
            .iter()
            .filter(|(_, mu)| *mu != S::default())
            .copied()
            .collect();

        Self { particles, massive }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::prelude::*;
    use glam::Vec3A;

    pub fn acceleration_computation<C, B>(mut cm: C)
    where
        B: ParticleStorage<InternalVector = Vec3A, Scalar = f32>,
        C: ComputeMethod<B>,
    {
        let massive = vec![(Vec3A::splat(0.0), 2.0), (Vec3A::splat(1.0), 3.0)];
        let massless = vec![(Vec3A::splat(5.0), 0.0)];

        let computed = cm.compute(ParticleStorage::new(
            [massive.clone(), massless.clone()].concat(),
        ));

        for (&point_mass1, computed) in massive.iter().chain(massless.iter()).zip(computed) {
            let mut acceleration = Vec3A::ZERO;

            for &point_mass2 in massive.iter() {
                let dir = point_mass2.0 - point_mass1.0;
                let mag_2 = dir.length_squared();

                if mag_2 != 0.0 {
                    acceleration += dir * point_mass2.1 / (mag_2 * mag_2.sqrt());
                }
            }

            assert!((acceleration).abs_diff_eq(computed, f32::EPSILON))
        }
    }
}
