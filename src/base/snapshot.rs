use crate::base::{ElemType, State};
use std::fmt::Debug;

#[allow(unused)]
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

    #[allow(unused)]
    pub fn state(&self) -> &S {
        &self.state
    }

    #[allow(unused)]
    pub fn reward(&self) -> ElemType {
        self.reward
    }

    #[allow(unused)]
    pub fn done(&self) -> bool {
        self.done
    }
}
