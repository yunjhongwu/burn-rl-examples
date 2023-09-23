use crate::base::{Action, State};
use crate::components::agent::Agent;
use std::marker::PhantomData;

#[derive(Default)]
struct Dqn<S: State, A: Action> {
    is_eval: bool,
    state: PhantomData<S>,
    action: PhantomData<A>,
}

impl<S: State, A: Action> Agent for Dqn<S, A> {
    type State = S;
    type Action = A;

    fn react(&mut self, _state: &Self::State) -> Self::Action {
        todo!()
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
