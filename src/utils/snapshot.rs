use std::fmt::Debug;

#[allow(unused)]
#[derive(Debug)]
pub struct Snapshot<State: Debug> {
    state: State,
    reward: f64,
    mask: bool,
}

impl<State: Debug> Snapshot<State> {
    pub fn new(state: State, reward: f64, mask: bool) -> Self {
        Self {
            state,
            reward,
            mask,
        }
    }

    #[allow(unused)]
    pub fn state(&self) -> &State {
        &self.state
    }

    #[allow(unused)]
    pub fn reward(&self) -> f64 {
        self.reward
    }

    #[allow(unused)]
    pub fn mask(&self) -> bool {
        self.mask
    }
}
