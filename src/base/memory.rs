use crate::components::env::Environment;
use rand::prelude::SliceRandom;
use std::collections::VecDeque;

#[allow(unused)]
pub struct Memory<E: Environment> {
    state: VecDeque<E::StateType>,
    next_state: VecDeque<E::StateType>,
    action: VecDeque<E::ActionType>,
    reward: VecDeque<f32>,
}

impl<E: Environment> Memory<E> {
    #[allow(unused)]
    pub fn new() -> Self {
        Self {
            state: VecDeque::<E::StateType>::new(),
            next_state: VecDeque::<E::StateType>::new(),
            action: VecDeque::<E::ActionType>::new(),
            reward: VecDeque::<f32>::new(),
        }
    }

    #[allow(unused)]
    pub fn push_transition(&mut self, transition: Transition<E>) {
        self.state.push_back(transition.state);
        self.next_state.push_back(transition.next_state);
        self.action.push_back(transition.action);
        self.reward.push_back(transition.reward);
    }

    #[allow(unused)]
    pub fn push(
        &mut self,
        state: E::StateType,
        next_state: E::StateType,
        action: E::ActionType,
        reward: f32,
    ) {
        self.state.push_back(state);
        self.next_state.push_back(next_state);
        self.action.push_back(action);
        self.reward.push_back(reward);
    }

    #[allow(unused)]
    pub fn pop(&mut self) -> Option<Transition<E>> {
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
    pub fn sample(&self, size: usize) -> Memory<E> {
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
pub struct Transition<E: Environment> {
    state: E::StateType,
    next_state: E::StateType,
    action: E::ActionType,
    reward: f32,
}

#[cfg(test)]
mod tests {
    use crate::base::{Action, Memory, Snapshot, State};
    use crate::components::env::Environment;

    #[derive(Debug, Copy, Clone, Default)]
    struct TestAction {
        data: i32,
    }

    impl From<u32> for TestAction {
        fn from(value: u32) -> Self {
            value.into()
        }
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

        fn size() -> usize {
            1
        }
    }

    struct TestEnv {}

    impl Environment for TestEnv {
        type StateType = TestState;
        type ActionType = TestAction;

        fn state(&self) -> Self::StateType {
            todo!()
        }

        fn reset(&mut self) -> Snapshot<Self::StateType> {
            todo!()
        }

        fn step(&mut self, _action: Self::ActionType) -> Snapshot<Self::StateType> {
            todo!()
        }
    }

    #[test]
    fn test_memory() {
        let mut memory = Memory::<TestEnv>::new();
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
