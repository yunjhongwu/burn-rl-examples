use crate::base::{Action, State};
use crate::components::agent::Agent;
use std::marker::PhantomData;

#[derive(Default)]
struct Random<S: State, A: Action> {
    state: PhantomData<S>,
    action: PhantomData<A>,
}

impl<S: State, A: Action> Agent for Random<S, A> {
    type State = S;
    type Action = A;

    fn react(&mut self, _state: &Self::State) -> Self::Action {
        Self::Action::random()
    }

    fn collect(&mut self, _reward: f32, _done: bool) {}

    fn reset(&mut self) {}

    fn is_eval(&self) -> bool {
        true
    }
}
