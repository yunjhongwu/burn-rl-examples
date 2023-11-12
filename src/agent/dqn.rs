use crate::base::State;
use crate::components::agent::Agent;
use crate::components::env::Environment;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{log_softmax, relu};
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Tensor};
use std::marker::PhantomData;

pub struct Dqn<E: Environment, B: Backend> {
    is_eval: bool,
    model: Model<B>,
    state: PhantomData<E::StateType>,
    action: PhantomData<E::ActionType>,
}

impl<E: Environment, B: Backend> Dqn<E, B> {
    fn convert(state: &E::StateType) -> Tensor<B, 2> {
        state.data().unsqueeze()
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear_0: Linear<B>,
    linear_1: Linear<B>,
    output: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(input_size: usize) -> Self {
        Self {
            linear_0: LinearConfig::new(input_size, 16).init(),
            linear_1: LinearConfig::new(16, 8).init(),
            output: LinearConfig::new(8, 1).init(),
        }
    }
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_0.forward(input);
        let x = relu(x);
        let x = self.linear_1.forward(x);
        let x = relu(x);
        log_softmax(self.output.forward(x), 0)
    }
}

impl<E: Environment, B: Backend> Default for Dqn<E, B> {
    fn default() -> Self {
        Self {
            is_eval: false,
            model: Model::new(E::StateType::size()),
            state: PhantomData,
            action: PhantomData,
        }
    }
}

impl<E: Environment, B: Backend> Agent for Dqn<E, B> {
    type StateType = E::StateType;
    type ActionType = E::ActionType;

    fn react(&mut self, state: &Self::StateType) -> Self::ActionType {
        let output = self.model.forward(Self::convert(state));
        unsafe {
            output
                .argmax(0)
                .to_data()
                .value
                .get_unchecked(0)
                .elem::<u32>()
                .into()
        }
    }

    fn collect(&mut self, _reward: f32, _done: bool) {
        todo!()
    }

    fn reset(&mut self) {
        todo!()
    }

    fn is_eval(&self) -> bool {
        self.is_eval
    }
}
