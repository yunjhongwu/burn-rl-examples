use crate::base::{Action, State};
use rand::prelude::SliceRandom;
use std::collections::VecDeque;

#[allow(unused)]
pub struct Memory<S: State, A: Action> {
    state: VecDeque<S>,
    next_state: VecDeque<S>,
    action: VecDeque<A>,
    reward: VecDeque<f32>,
}

impl<S: State, A: Action> Memory<S, A> {
    #[allow(unused)]
    pub fn new() -> Self {
        Self {
            state: VecDeque::<S>::new(),
            next_state: VecDeque::<S>::new(),
            action: VecDeque::<A>::new(),
            reward: VecDeque::<f32>::new(),
        }
    }

    #[allow(unused)]
    pub fn push_transition(&mut self, transition: Transition<S, A>) {
        self.state.push_back(transition.state);
        self.next_state.push_back(transition.next_state);
        self.action.push_back(transition.action);
        self.reward.push_back(transition.reward);
    }

    #[allow(unused)]
    pub fn push(&mut self, state: S, next_state: S, action: A, reward: f32) {
        self.state.push_back(state);
        self.next_state.push_back(next_state);
        self.action.push_back(action);
        self.reward.push_back(reward);
    }

    #[allow(unused)]
    pub fn pop(&mut self) -> Option<Transition<S, A>> {
        let state = self.state.pop_front()?;
        let next_state = self.next_state.pop_front()?;
        let action = self.action.pop_front()?;
        let reward = self.reward.pop_front()?;
        Some(Transition {
            state,
            next_state,
            action,
            reward,
        })
    }

    #[allow(unused)]
    pub fn sample(&self, size: usize) -> Memory<S, A> {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..self.len()).collect();
        indices.shuffle(&mut rng);
        let mut memory = Memory::new();
        for index in indices.iter().take(size).copied() {
            memory.push(
                self.state[index],
                self.next_state[index],
                self.action[index],
                self.reward[index],
            );
        }
        memory
    }

    #[allow(unused)]
    pub fn len(&self) -> usize {
        self.state.len()
    }
}

#[allow(unused)]
pub struct Transition<S: State, A: Action> {
    state: S,
    next_state: S,
    action: A,
    reward: f32,
}

#[cfg(test)]
mod tests {
    use crate::base::{Action, Memory, State};

    #[derive(Debug, Copy, Clone, Default)]
    struct TestAction {
        data: i32,
    }

    impl Action for TestAction {
        fn random() -> Self {
            Self { data: 1 }
        }

        fn enumerate() -> Vec<Self> {
            vec![Self { data: 1 }]
        }
    }

    #[derive(Debug, Copy, Clone, Default)]
    struct TestState {
        data: f32,
    }

    impl State for TestState {
        type Data = f32;
        fn data(&self) -> &Self::Data {
            &self.data
        }
    }

    #[test]
    fn test_memory() {
        let mut memory = Memory::<TestState, TestAction>::new();
        for i in 0..20 {
            memory.push(
                TestState { data: i as f32 },
                TestState { data: -i as f32 },
                TestAction { data: i },
                0.1,
            );
        }
        let sample = memory.sample(5);
        assert_eq!(sample.len(), 5);
        memory.pop();
        let transition = memory.pop().unwrap();
        assert_eq!(memory.len(), 18);
        assert_eq!(transition.state.data, 1.0f32);
        assert_eq!(transition.next_state.data, -1.0f32);
        assert_eq!(transition.action.data, 1);
        assert_eq!(transition.reward, 0.1f32);
    }
}
