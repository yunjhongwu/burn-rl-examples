use crate::base::{Action, ElemType, State};
use burn::tensor::backend::Backend;
use burn::tensor::{BasicOps, ElementConversion, Tensor, TensorKind};
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use ringbuffer::RingBuffer;

pub(crate) fn convert_state_to_tensor<S: State, B: Backend>(state: S) -> Tensor<B, 1> {
    state.to_tensor()
}

pub(crate) fn convert_tenor_to_action<A: Action, B: Backend>(output: Tensor<B, 2>) -> A {
    unsafe {
        output
            .argmax(1)
            .to_data()
            .value
            .get_unchecked(0)
            .elem::<u32>()
            .into()
    }
}

#[allow(unused)]
pub(crate) fn sample_action_from_tensor<A: Action, B: Backend>(output: Tensor<B, 2>) -> A {
    let dist = WeightedIndex::new(
        output
            .to_data()
            .value
            .iter()
            .map(|x| x.elem::<ElemType>())
            .collect::<Vec<ElemType>>()
            .to_vec(),
    )
    .unwrap();
    let mut rng = thread_rng();
    (dist.sample(&mut rng) as u32).into()
}

pub(crate) fn stack<
    B: Backend,
    T,
    K: TensorKind<B> + BasicOps<B>,
    F: FnMut(&T) -> Tensor<B, 1, K>,
>(
    data: &impl RingBuffer<T>,
    accessor: F,
) -> Tensor<B, 2, K> {
    Tensor::cat(data.iter().map(accessor).collect::<Vec<_>>(), 0).reshape([data.len() as i32, -1])
}
