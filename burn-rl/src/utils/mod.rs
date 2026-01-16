use crate::base::{Action, ElemType, State};
use burn::module::AutodiffModule;
use burn::optim::LearningRate;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

pub(crate) fn to_state_tensor<S: State, B: Backend>(state: S) -> Tensor<B, 1> {
    state.to_tensor()
}

pub(crate) fn ref_to_state_tensor<S: State, B: Backend>(state: &S) -> Tensor<B, 1> {
    to_state_tensor(*state)
}

pub(crate) fn convert_tensor_to_action<A: Action, B: Backend>(output: Tensor<B, 2>) -> A {
    (output.argmax(1).to_data().as_slice::<i64>().unwrap()[0] as u32).into()
}

pub(crate) fn to_action_tensor<A: Action, B: Backend>(action: A) -> Tensor<B, 1, Int> {
    Tensor::<B, 1, Int>::from_ints([action.into() as i32], &Default::default())
}

pub(crate) fn ref_to_action_tensor<A: Action, B: Backend>(action: &A) -> Tensor<B, 1, Int> {
    to_action_tensor(*action)
}

pub(crate) fn to_reward_tensor<B: Backend>(reward: impl Into<ElemType> + Clone) -> Tensor<B, 1> {
    Tensor::from_floats([reward.into()], &Default::default())
}

pub(crate) fn ref_to_reward_tensor<B: Backend>(
    reward: &(impl Into<ElemType> + Clone),
) -> Tensor<B, 1> {
    to_reward_tensor(reward.clone())
}
pub(crate) fn to_not_done_tensor<B: Backend>(done: bool) -> Tensor<B, 1> {
    Tensor::from_floats([if done { 0.0 } else { 1.0 }], &Default::default())
}

pub(crate) fn ref_to_not_done_tensor<B: Backend>(done: &bool) -> Tensor<B, 1> {
    to_not_done_tensor(*done)
}
#[allow(unused)]
pub(crate) fn sample_action_from_tensor<A: Action, B: Backend>(output: Tensor<B, 2>) -> Option<A> {
    let prob = output.to_data().to_vec::<ElemType>().ok()?;

    let dist = WeightedIndex::new(prob).ok()?;

    let mut rng = thread_rng();
    Some((dist.sample(&mut rng) as u32).into())
}

pub(crate) fn get_elem<B: Backend, const D: usize>(
    i: usize,
    tensor: &Tensor<B, D>,
) -> Option<ElemType> {
    tensor.to_data().as_slice().ok()?.get(i).copied()
}

pub(crate) fn elementwise_min<B: Backend, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> Tensor<B, D> {
    let rhs_lower = rhs.clone().lower(lhs.clone());
    lhs.clone().mask_where(rhs_lower, rhs.clone())
}

pub(crate) fn update_parameters<B: AutodiffBackend, M: AutodiffModule<B>>(
    loss: Tensor<B, 1>,
    module: M,
    optimizer: &mut impl Optimizer<M, B>,
    learning_rate: LearningRate,
) -> M {
    let gradients = loss.backward();
    let gradient_params = GradientsParams::from_grads(gradients, &module);
    optimizer.step(learning_rate, module, gradient_params)
}
