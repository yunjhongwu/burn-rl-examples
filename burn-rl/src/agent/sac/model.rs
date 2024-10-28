use crate::base::{ElemType, Model};
use burn::module::{Module, Param, ParamId};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub trait SACActor<B: Backend>: Model<B, Tensor<B, 2>, Tensor<B, 2>> {}

pub trait SACCritic<B: Backend>: Model<B, Tensor<B, 2>, Tensor<B, 2>> {
    fn soft_update(this: Self, that: &Self, tau: ElemType) -> Self;
}

#[derive(Module, Debug)]
pub struct SACTemperature<B: Backend> {
    temperature: Param<Tensor<B, 2>>,
}

impl<B: Backend> Default for SACTemperature<B> {
    fn default() -> Self {
        Self {
            temperature: Param::initialized(
                ParamId::new(),
                Tensor::zeros([1, 1], &Default::default()),
            ),
        }
    }
}

impl<B: Backend> SACTemperature<B> {
    pub fn forward(&self) -> Tensor<B, 2> {
        self.temperature.val()
    }
}

pub struct SACNets<B: Backend, Actor: SACActor<B>, Critic: SACCritic<B>> {
    pub actor: Actor,
    pub critic_1: Critic,
    pub critic_1_target: Option<Critic>,

    pub critic_2: Critic,
    pub critic_2_target: Option<Critic>,
    pub temperature: SACTemperature<B>,
}

impl<B: Backend, Actor: SACActor<B>, Critic: SACCritic<B>> SACNets<B, Actor, Critic> {
    pub fn new(actor: Actor, critic_1: Critic, critic_2: Critic) -> Self {
        Self {
            actor,
            critic_1: critic_1.clone(),
            critic_1_target: Some(critic_1),
            critic_2: critic_2.clone(),
            critic_2_target: Some(critic_2),
            temperature: SACTemperature::<B>::default(),
        }
    }
}
