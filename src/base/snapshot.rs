use crate::base::State;
use std::fmt::Debug;

#[allow(unused)]
#[derive(Debug)]
pub struct Snapshot<S: State> {
    state: S,
    reward: f64,
    done: bool,
}

impl<S: State> Snapshot<S> {
    pub fn new(state: S, reward: f64, done: bool) -> Self {
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
    pub fn reward(&self) -> f64 {
        self.reward
    }

    #[allow(unused)]
    pub fn done(&self) -> bool {
        self.done
    }
}
