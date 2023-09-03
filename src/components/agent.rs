use crate::base::{Action, State};

pub trait Agent {
    type State: State;
    type Action: Action;

    fn react(&mut self, state: &Self::State) -> Self::Action;
    fn collect(&mut self, reward: f32, done: bool);
    fn reset(&mut self);

    fn is_eval(&self) -> bool;
}
