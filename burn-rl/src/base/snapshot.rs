use crate::base::{ElemType, State};
use std::fmt::Debug;

#[derive(Debug)]
pub struct Snapshot<S: State> {
    state: S,
    reward: ElemType,
    done: bool,
}

impl<S: State> Snapshot<S> {
    pub fn new(state: S, reward: ElemType, done: bool) -> Self {
        Self {
            state,
            reward,
            done,
        }
    }

    pub fn state(&self) -> &S {
        &self.state
    }

    pub fn reward(&self) -> ElemType {
        self.reward
    }

    pub fn done(&self) -> bool {
        self.done
    }
}
