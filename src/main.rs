mod agent;
mod base;
mod components;
mod env;

use crate::agent::Dqn;
use crate::base::{Action, ElemType, Memory, Model, State};
use crate::components::agent::Agent;
use crate::components::env::Environment;
use crate::env::cart_pole::CartPole;
use burn::backend::ndarray::NdArrayBackend;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamWConfig;
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
    linear_2: Linear<B>,
}

impl<B: ADBackend> DQNModel<B> {
    pub fn new(input_size: usize, dense_size: usize, output_size: usize) -> Self {
        Self {
            linear_0: LinearConfig::new(input_size, dense_size).init(),
            linear_1: LinearConfig::new(dense_size, dense_size).init(),
            linear_2: LinearConfig::new(dense_size, output_size).init(),
        }
    }

    fn soft_update_tensor<const N: usize>(
        this: &Param<Tensor<B, N>>,
        that: &Param<Tensor<B, N>>,
        tau: f64,
    ) -> Param<Tensor<B, N>> {
        let other_weight = that.val();
        let self_weight = this.val();
        let new_weight = self_weight * (1.0 - tau) + other_weight * tau;

        Param::from(new_weight.no_grad())
    }
    fn soft_update_linear(this: &mut Linear<B>, that: &Linear<B>, tau: f64) {
        this.weight = Self::soft_update_tensor(&this.weight, &that.weight, tau);
        if let (Some(self_bias), Some(other_bias)) = (&mut this.bias, &that.bias) {
            this.bias = Some(Self::soft_update_tensor(self_bias, other_bias, tau));
        }
    }
}

impl<B: ADBackend> Model<B> for DQNModel<B> {
    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let layer_0_output = relu(self.linear_0.forward(input));
        let layer_1_output = relu(self.linear_1.forward(layer_0_output));

        relu(self.linear_2.forward(layer_1_output))
    }

    fn soft_update(&mut self, other: &Self, tau: f64) {
        Self::soft_update_linear(&mut self.linear_0, &other.linear_0, tau);
        Self::soft_update_linear(&mut self.linear_1, &other.linear_1, tau);
        Self::soft_update_linear(&mut self.linear_2, &other.linear_2, tau);
    }
}

const MEMORY_SIZE: usize = 4096;
const BATCH_SIZE: usize = 128;

pub fn main() {
    let num_episodes = 256_usize;
    let eps_decay = 1000.0;
    let eps_start = 0.9;
    let eps_end = 0.05;
    let dense_size = 128_usize;

    let mut env = MyEnv::new(false);

    let model = DQNModel::<DQNBackend>::new(
        <<MyEnv as Environment>::StateType as State>::size(),
        dense_size,
        <<MyEnv as Environment>::ActionType as Action>::size(),
    );
    let mut agent = Dqn::<MyEnv, DQNBackend, DQNModel<DQNBackend>, false>::new(model);

    let mut memory = Memory::<MyEnv, DQNBackend, MEMORY_SIZE>::default();

    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Value(100.0)))
        .init::<DQNBackend, DQNModel<DQNBackend>>();

    let mut policy_net = agent.model().clone();

    let mut step = 0_usize;

    for episode in 0..num_episodes {
        let mut episode_done = false;
        let mut episode_duration = 0;
        let mut state = env.state();

        while !episode_done {
            let eps_threshold =
                eps_end + (eps_start - eps_end) * f64::exp(-(step as f64) / eps_decay);
            let action = agent.react_with_exploration(&policy_net, state, eps_threshold);
            let snapshot = env.step(action);

            memory.push(
                state,
                *snapshot.state(),
                action,
                snapshot.reward(),
                snapshot.done(),
            );

            if BATCH_SIZE < memory.len() {
                policy_net = agent.train(policy_net, memory.sample::<BATCH_SIZE>(), &mut optimizer);
            }

            step += 1;
            episode_duration += 1;

            if snapshot.done() || episode_duration >= 500 {
                env.reset();
                episode_done = true;

                println!(
                    "{{\"episode\": {}, \"duration\": {:.4}}}",
                    episode, episode_duration
                );
            } else {
                state = *snapshot.state();
            }
        }
    }

    demo_model(agent.to_eval());
}

fn demo_model(agent: Dqn<MyEnv, DQNBackend, DQNModel<DQNBackend>, true>) {
    let mut env = MyEnv::new(true);
    let mut state = env.state();
    let mut done = false;
    while !done {
        let action = agent.react(&state);
        let snapshot = env.step(action);
        state = *snapshot.state();
        done = snapshot.done();
    }
}
