use crate::base::{Action, State};

pub trait Agent {
    type StateType: State;
    type ActionType: Action;

    fn react(&self, state: &Self::StateType) -> Self::ActionType;
}
