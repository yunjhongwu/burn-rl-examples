use crate::base::{Model, State};
use crate::components::agent::Agent;
use crate::components::env::Environment;
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Tensor};
use std::marker::PhantomData;

pub struct Dqn<E: Environment, B: Backend, M: Model<B>> {
    is_eval: bool,
    model: M,
    state: PhantomData<E::StateType>,
    action: PhantomData<E::ActionType>,
    backend: PhantomData<B>,
}

impl<E: Environment, B: Backend, M: Model<B>> Dqn<E, B, M> {
    pub(crate) fn new(model: M) -> Self {
        Self {
            is_eval: false,
            model,
            state: PhantomData,
            action: PhantomData,
            backend: PhantomData,
        }
    }

    fn convert(state: &E::StateType) -> Tensor<B, 2> {
        state.data().unsqueeze()
    }
}

impl<E: Environment, B: Backend, M: Model<B>> Agent for Dqn<E, B, M> {
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
