use crate::base::{Action, Snapshot, State};

pub trait Environment {
    type State: State;
    type Action: Action;

    fn reset(&mut self) -> Snapshot<Self::State>;
    fn step(&mut self, action: Self::Action) -> Snapshot<Self::State>;
}
