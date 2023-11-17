mod agent;
mod base;
mod components;
mod env;

use crate::base::{Action, ElemType, Memory, Model, State};
use crate::components::agent::Agent;
use crate::components::env::Environment;
use crate::env::cart_pole::CartPole;
use burn::backend::ndarray::NdArrayBackend;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::tensor::activation::relu;
use burn::tensor::backend::{ADBackend, Backend};
use burn::tensor::Tensor;
use burn_autodiff::ADBackendDecorator;

type DQNBackend = ADBackendDecorator<NdArrayBackend<ElemType>>;
type MyEnv = CartPole;

#[derive(Module, Debug)]
pub struct DQNModel<B: Backend> {
    linear_0: Linear<B>,
    linear_1: Linear<B>,
}

impl<B: ADBackend> DQNModel<B> {
    pub fn new(input_size: usize, dense_size: usize, output_size: usize) -> Self {
        Self {
            linear_0: LinearConfig::new(input_size, dense_size).init(),
            linear_1: LinearConfig::new(dense_size, output_size).init(),
        }
    }

    fn soft_update_tensor<const N: usize>(
        this: &Param<Tensor<B, N>>,
        that: &Param<Tensor<B, N>>,
        tau: f64,
    ) -> Param<Tensor<B, N>> {
        let other_weight = that.val();
        let self_weight = this.val();
        let new_weight = self_weight * tau + other_weight * (1.0 - tau);

        Param::from(new_weight.no_grad())
    }
}

impl<B: ADBackend> Model<B> for DQNModel<B> {
    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let layer_0_output = relu(self.linear_0.forward(input));

        relu(self.linear_1.forward(layer_0_output))
    }

    fn soft_update(&mut self, other: &Self, tau: f64) {
        self.linear_0.weight =
            Self::soft_update_tensor(&self.linear_0.weight, &other.linear_0.weight, tau);
        if let (Some(self_bias), Some(other_bias)) = (&mut self.linear_0.bias, &other.linear_0.bias)
        {
            self.linear_0.bias = Some(Self::soft_update_tensor(self_bias, other_bias, tau));
        }
        self.linear_1.weight =
            Self::soft_update_tensor(&self.linear_1.weight, &other.linear_1.weight, tau);
        if let (Some(self_bias), Some(other_bias)) = (&mut self.linear_1.bias, &other.linear_1.bias)
        {
            self.linear_1.bias = Some(Self::soft_update_tensor(self_bias, other_bias, tau));
        }
    }
}

pub fn main() {
    let eps_decay = 1000.0;
    let eps_start = 0.9;
    let eps_end = 0.05;
    let dense_size = 16;
    let reward_ewma_decay = 0.95;

    let mut env = MyEnv::new(false);
    let model = DQNModel::<DQNBackend>::new(
        <<MyEnv as Environment>::StateType as State>::size(),
        dense_size,
        <<MyEnv as Environment>::ActionType as Action>::size(),
    );

    let mut agent = agent::Dqn::<MyEnv, DQNBackend, DQNModel<DQNBackend>, false>::new(model);
    let mut state = env.state();
    let mut step = 0;

    let mut memory = Memory::<MyEnv, DQNBackend, 256>::new();
    let mut optimizer = AdamConfig::new().init::<DQNBackend, DQNModel<DQNBackend>>();
    let mut policy_net = agent.model().clone();
    let mut ewma_reward = 0.0;
    for episode in 0..1024 {
        let mut done = false;
        let mut episode_duration = 0;
        while !done {
            let eps_threshold =
                eps_end + (eps_start - eps_end) * f64::exp(-(step as f64) / eps_decay);
            let action = agent.react_with_exploration(&policy_net, state, eps_threshold);
            let mut snapshot = env.step(action);

            memory.push(
                state,
                *snapshot.state(),
                action,
                snapshot.reward(),
                snapshot.done(),
            );
            if step > memory.len() {
                policy_net = agent.train(policy_net, &memory, &mut optimizer);
            }
            if snapshot.done() {
                snapshot = env.reset();
                done = true;
            }
            state = *snapshot.state();
            step += 1;
            episode_duration += 1;
        }
        ewma_reward =
            (1.0 - reward_ewma_decay) * episode_duration as f64 + reward_ewma_decay * ewma_reward;
        println!(
            "Episode: {}, step: {}, EWMA episode duration: {}",
            episode, step, ewma_reward
        );
    }

    let mut env = MyEnv::new(true);
    let agent = agent.to_eval();
    let mut done = false;
    while !done {
        let action = agent.react(&state);
        let snapshot = env.step(action);
        state = *snapshot.state();
        done = snapshot.done();
    }
}
