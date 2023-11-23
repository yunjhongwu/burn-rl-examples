use crate::base::{Action, ElemType, Snapshot, State};
use std::fmt::Debug;

pub trait Environment: Debug {
    type StateType: State;
    type ActionType: Action;
    type RewardType: Debug + Clone + Into<ElemType>;

    const MAX_STEPS: usize = usize::MAX;

    fn new(visualized: bool) -> Self;

    fn state(&self) -> Self::StateType;

    fn reset(&mut self) -> Snapshot<Self>;

    fn step(&mut self, action: Self::ActionType) -> Snapshot<Self>;
}
