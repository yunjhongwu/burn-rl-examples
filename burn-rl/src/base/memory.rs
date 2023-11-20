use crate::base::environment::Environment;
use crate::base::transition::Transition;
use burn::tensor::backend::Backend;
use rand::prelude::SliceRandom;

pub fn sample_memory<
    const CAP: usize,
    const SIZE: usize,
    E: Environment,
    B: Backend,
    I: Memory<E, B, CAP>,
    O: Memory<E, B, SIZE, TransitionType = <I as Memory<E, B, CAP>>::TransitionType>,
>(
    memory: &I,
) -> O {
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..memory.len()).collect();
    indices.shuffle(&mut rng);
    let mut subset = O::default();
    for index in indices.iter().take(SIZE).copied() {
        let transition = memory.get(index);
        subset.push(*transition);
    }
    subset
}

pub trait Memory<E: Environment, B: Backend, const CAP: usize>: Default {
    type TransitionType: Transition<E>;

    fn get(&self, index: usize) -> &Self::TransitionType;

    fn push(&mut self, value: Self::TransitionType);

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool;
}
