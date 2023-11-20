use crate::base::{Action, Snapshot, State};
use std::fmt::Debug;

pub trait Environment: Debug {
    type StateType: State;
    type ActionType: Action;
    type RewardType: Debug + Copy + Clone + Into<f32>;

    const MAX_STEPS: usize = usize::MAX;

    fn state(&self) -> Self::StateType;

    fn reset(&mut self) -> Snapshot<Self::StateType>;

    fn step(&mut self, action: Self::ActionType) -> Snapshot<Self::StateType>;
}
