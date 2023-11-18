use crate::base::{Action, Snapshot, State};

pub trait Environment {
    type StateType: State;
    type ActionType: Action;

    fn state(&self) -> Self::StateType;
    fn reset(&mut self) -> Snapshot<Self::StateType>;
    fn step(&mut self, action: Self::ActionType) -> Snapshot<Self::StateType>;
}
