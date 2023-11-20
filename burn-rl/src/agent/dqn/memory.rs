use crate::base::{Environment, Memory, Transition};
use crate::utils::{convert_state_to_tensor, stack};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use ringbuffer::{ConstGenericRingBuffer, RingBuffer};
use std::marker::PhantomData;
use std::ops::Index;

#[derive(Debug)]
pub struct DQNTransition<E: Environment, B: Backend> {
    state: E::StateType,
    next_state: E::StateType,
    action: E::ActionType,
    reward: E::RewardType,
    done: bool,
    environment: PhantomData<E>,
    backend: PhantomData<B>,
}

impl<E: Environment, B: Backend> Transition<E> for DQNTransition<E, B> {
    fn state(&self) -> &E::StateType {
        &self.state
    }

    fn action(&self) -> &E::ActionType {
        &self.action
    }

    fn reward(&self) -> &E::RewardType {
        &self.reward
    }

    fn next_state(&self) -> &E::StateType {
        &self.next_state
    }

    fn is_done(&self) -> bool {
        self.done
    }
}

impl<E: Environment, B: Backend> Clone for DQNTransition<E, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Environment, B: Backend> Copy for DQNTransition<E, B> {}

impl<E: Environment, B: Backend> DQNTransition<E, B> {
    pub fn new(
        state: E::StateType,
        next_state: E::StateType,
        action: E::ActionType,
        reward: E::RewardType,
        done: bool,
    ) -> Self {
        Self {
            state,
            next_state,
            action,
            reward,
            done,
            environment: PhantomData,
            backend: PhantomData,
        }
    }

    fn state(&self) -> &E::StateType {
        &self.state
    }

    fn next_state(&self) -> &E::StateType {
        &self.next_state
    }

    fn action(&self) -> &E::ActionType {
        &self.action
    }

    fn reward(&self) -> &E::RewardType {
        &self.reward
    }

    fn done(&self) -> &bool {
        &self.done
    }
}

pub struct DQNMemory<E: Environment, B: Backend, const CAP: usize> {
    transitions: ConstGenericRingBuffer<DQNTransition<E, B>, CAP>,
    environment: PhantomData<E>,
    backend: PhantomData<B>,
}

impl<E: Environment, B: Backend, const CAP: usize> Default for DQNMemory<E, B, CAP> {
    fn default() -> Self {
        Self {
            transitions: ConstGenericRingBuffer::new(),
            environment: PhantomData,
            backend: PhantomData,
        }
    }
}

impl<E: Environment, B: Backend, const CAP: usize> Index<usize> for DQNMemory<E, B, CAP> {
    type Output = DQNTransition<E, B>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.transitions[index]
    }
}

impl<E: Environment, B: Backend, const CAP: usize> Memory<E, B, CAP> for DQNMemory<E, B, CAP> {
    type TransitionType = DQNTransition<E, B>;

    fn get(&self, index: usize) -> &Self::TransitionType {
        &self.transitions[index]
    }

    fn push(&mut self, transition: DQNTransition<E, B>) {
        self.transitions.push(transition);
    }

    fn len(&self) -> usize {
        self.transitions.len()
    }

    fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }
}

impl<E: Environment, B: Backend, const CAP: usize> DQNMemory<E, B, CAP> {
    pub fn next_state_batch(&self) -> Tensor<B, 2> {
        stack(&self.transitions, |transition| {
            convert_state_to_tensor(*transition.next_state())
        })
    }

    pub fn state_batch(&self) -> Tensor<B, 2> {
        stack(&self.transitions, |transition| {
            convert_state_to_tensor(*transition.state())
        })
    }

    pub fn action_batch(&self) -> Tensor<B, 2, Int> {
        stack(&self.transitions, |transition| {
            let action_index: u32 = (*transition.action()).into();
            Tensor::<B, 1, Int>::from_ints([action_index as i32])
        })
    }

    pub fn reward_batch(&self) -> Tensor<B, 2> {
        stack(&self.transitions, |transition| {
            let reward: f32 = (*transition.reward()).into();
            Tensor::from_floats([reward])
        })
    }
    pub fn not_done_batch(&self) -> Tensor<B, 2> {
        stack(&self.transitions, |transition| {
            let not_done = if *transition.done() { 0.0 } else { 1.0 };
            Tensor::from_floats([not_done])
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::agent::{DQNMemory, DQNTransition};
    use crate::base::environment::Environment;
    use crate::base::{sample_memory, Action, ElemType, Memory, Snapshot, State};
    use burn::backend::NdArrayBackend;
    use burn::tensor::backend::Backend;
    use burn::tensor::{Shape, Tensor};
    use std::fmt::Debug;

    #[derive(Debug, Copy, Clone)]
    struct TestAction {
        data: i32,
    }

    impl From<u32> for TestAction {
        fn from(value: u32) -> Self {
            value.into()
        }
    }

    impl From<TestAction> for u32 {
        fn from(action: TestAction) -> Self {
            action.data as u32
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

    #[derive(Debug, Copy, Clone)]
    struct TestState {
        data: [ElemType; 2],
    }

    impl State for TestState {
        type Data = [ElemType; 2];

        fn to_tensor<B: Backend>(&self) -> Tensor<B, 1> {
            Tensor::<B, 1>::from_floats(self.data)
        }
        fn size() -> usize {
            1
        }
    }

    #[derive(Debug)]
    struct TestEnv {}

    type TestBackend = NdArrayBackend<ElemType>;

    impl Environment for TestEnv {
        type StateType = TestState;
        type ActionType = TestAction;
        type RewardType = ElemType;

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
        let mut memory = DQNMemory::<TestEnv, TestBackend, 16>::default();
        for i in 0..20 {
            memory.push(DQNTransition::<TestEnv, TestBackend>::new(
                TestState {
                    data: [i as ElemType, (i * 2) as ElemType],
                },
                TestState {
                    data: [-i as ElemType, (i * 3) as ElemType],
                },
                TestAction { data: i },
                0.1,
                false,
            ));
        }
        let sample = sample_memory::<16, 5, _, _, _, DQNMemory<TestEnv, TestBackend, 5>>(&memory);
        assert_eq!(sample.len(), 5);

        let state_batch = sample.state_batch();
        assert_eq!(state_batch.shape(), Shape::new([5, 2]));
        let state_sample = state_batch
            .select(0, Tensor::from_ints([0, 1]))
            .to_data()
            .value;
        assert_eq!(state_sample[0] * 2.0, state_sample[1]);

        let next_state_batch = sample.next_state_batch();
        assert_eq!(next_state_batch.shape(), Shape::new([5, 2]));
        let next_state_sample = next_state_batch
            .select(0, Tensor::from_ints([0, 1]))
            .to_data()
            .value;
        assert_eq!(next_state_sample[0] * -3.0, next_state_sample[1]);

        let action_batch = sample.action_batch();
        assert_eq!(action_batch.shape(), Shape::new([5, 1]));
        assert_eq!(action_batch.to_data().value[0], state_sample[0] as i64);

        let reward_batch = sample.reward_batch();
        assert_eq!(reward_batch.shape(), Shape::new([5, 1]));
        assert_eq!(reward_batch.to_data().value[0], 0.1);

        let not_done_batch = sample.not_done_batch();
        assert_eq!(not_done_batch.shape(), Shape::new([5, 1]));
        assert_eq!(not_done_batch.to_data().value[0], 1.0);
    }
}
