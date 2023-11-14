mod agent;
mod base;
mod components;
mod demo;
mod env;

use crate::base::{Model, State};
use crate::components::agent::Agent;
use crate::components::env::Environment;
use crate::env::cart_pole::CartPole;
use burn::backend::ndarray::NdArrayBackend;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::Tensor;
use burn_autodiff::ADBackendDecorator;

type DQNBackend = ADBackendDecorator<NdArrayBackend<f32>>;

#[derive(Module, Debug, Clone)]
pub struct DQNModel {
    linear_0: Linear<DQNBackend>,
    linear_1: Linear<DQNBackend>,
}

impl DQNModel {
    pub fn new(input_size: usize, dense_size: usize, output_size: usize) -> Self {
        Self {
            linear_0: LinearConfig::new(input_size, dense_size).init(),
            linear_1: LinearConfig::new(dense_size, output_size).init(),
        }
    }
}

impl Model<DQNBackend> for DQNModel {
    fn forward<const D: usize>(&self, input: Tensor<DQNBackend, D>) -> Tensor<DQNBackend, D> {
        let layer_0_output = relu(self.linear_0.forward(input));

        relu(self.linear_1.forward(layer_0_output))
    }
}

type MyEnv = CartPole;

pub fn main() {
    let mut env = MyEnv::new();
    let model = DQNModel::new(<<MyEnv as Environment>::StateType as State>::size(), 16, 2);
    let mut agent = agent::Dqn::<MyEnv, DQNBackend, DQNModel>::new(model);
    let mut state = env.state();
    for _ in 0..100 {
        let mut snapshot = env.step(agent.react(&state));
        if snapshot.done() {
            snapshot = env.reset();
        }
        state = *snapshot.state();
        println!("{:?}", snapshot.reward());
    }
}
