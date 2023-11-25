use crate::base::{Action, ElemType, State};
use burn::module::ADModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::{ADBackend, Backend};
use burn::tensor::{ElementConversion, Int, Tensor};
use burn::LearningRate;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

pub(crate) fn to_state_tensor<S: State, B: Backend>(state: S) -> Tensor<B, 1> {
    state.to_tensor()
}

pub(crate) fn ref_to_state_tensor<S: State, B: Backend>(state: &S) -> Tensor<B, 1> {
    to_state_tensor(*state)
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

pub(crate) fn to_action_tensor<A: Action, B: Backend>(action: A) -> Tensor<B, 1, Int> {
    Tensor::<B, 1, Int>::from_ints([action.into() as i32])
}

pub(crate) fn ref_to_action_tensor<A: Action, B: Backend>(action: &A) -> Tensor<B, 1, Int> {
    to_action_tensor(*action)
}

pub(crate) fn to_reward_tensor<B: Backend>(reward: impl Into<ElemType> + Clone) -> Tensor<B, 1> {
    Tensor::from_floats([reward.into()])
}

pub(crate) fn ref_to_reward_tensor<B: Backend>(
    reward: &(impl Into<ElemType> + Clone),
) -> Tensor<B, 1> {
    to_reward_tensor(reward.clone())
}
pub(crate) fn to_not_done_tensor<B: Backend>(done: bool) -> Tensor<B, 1> {
    Tensor::from_floats([if done { 0.0 } else { 1.0 }])
}

pub(crate) fn ref_to_not_done_tensor<B: Backend>(done: &bool) -> Tensor<B, 1> {
    to_not_done_tensor(*done)
}
#[allow(unused)]
pub(crate) fn sample_action_from_tensor<A: Action, B: Backend>(output: Tensor<B, 2>) -> Option<A> {
    let prob = output
        .to_data()
        .value
        .iter()
        .map(|x| x.elem::<ElemType>())
        .collect::<Vec<_>>()
        .to_vec();

    let dist = WeightedIndex::new(prob).ok()?;

    let mut rng = thread_rng();
    Some((dist.sample(&mut rng) as u32).into())
}

pub(crate) fn get_elem<B: Backend, const D: usize>(
    i: usize,
    tensor: &Tensor<B, D>,
) -> Option<ElemType> {
    tensor.to_data().value.get(i).map(|x| x.elem::<ElemType>())
}

pub(crate) fn elementwise_min<B: Backend, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> Tensor<B, D> {
    let rhs_lower = rhs.clone().lower(lhs.clone());
    lhs.clone().mask_where(rhs_lower, rhs.clone())
}

pub(crate) fn update_parameters<B: ADBackend, M: ADModule<B>>(
    loss: Tensor<B, 1>,
    module: M,
    optimizer: &mut impl Optimizer<M, B>,
    learning_rate: LearningRate,
) -> M {
    let gradients = loss.backward();
    let gradient_params = GradientsParams::from_grads(gradients, &module);
    optimizer.step(learning_rate, module, gradient_params)
}
