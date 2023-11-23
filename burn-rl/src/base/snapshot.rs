use crate::base::Environment;
use std::fmt::Debug;

#[derive(Debug)]
pub struct Snapshot<E: Environment + ?Sized> {
    state: E::StateType,
    reward: E::RewardType,
    done: bool,
}

impl<E: Environment> Snapshot<E> {
    pub fn new(state: E::StateType, reward: E::RewardType, done: bool) -> Self {
        Self {
            state,
            reward,
            done,
        }
    }

    pub fn state(&self) -> &E::StateType {
        &self.state
    }

    pub fn reward(&self) -> &E::RewardType {
        &self.reward
    }

    pub fn done(&self) -> bool {
        self.done
    }
}
