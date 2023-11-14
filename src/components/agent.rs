use crate::base::{Action, State};

pub trait Agent {
    type StateType: State;
    type ActionType: Action;

    fn react(&mut self, state: &Self::StateType) -> Self::ActionType;
    fn collect(&mut self, reward: f32, done: bool);
    fn reset(&mut self);

    fn is_eval(&self) -> bool;
}
