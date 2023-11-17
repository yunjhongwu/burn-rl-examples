use crate::base::{ElemType, State};
use crate::components::env::Environment;
use burn::tensor::backend::Backend;
use burn::tensor::{BasicOps, Int, Tensor, TensorKind};
use rand::prelude::SliceRandom;
use ringbuffer::{ConstGenericRingBuffer, RingBuffer};
use std::marker::PhantomData;

#[allow(unused)]
pub struct Memory<E: Environment, B: Backend, const CAP: usize> {
    state: ConstGenericRingBuffer<E::StateType, CAP>,
    next_state: ConstGenericRingBuffer<E::StateType, CAP>,
    action: ConstGenericRingBuffer<E::ActionType, CAP>,
    reward: ConstGenericRingBuffer<ElemType, CAP>,
    done: ConstGenericRingBuffer<bool, CAP>,
    backend: PhantomData<B>,
}

impl<E: Environment, B: Backend, const CAP: usize> Memory<E, B, CAP> {
    #[allow(unused)]
    pub fn new() -> Self {
        Self {
            state: ConstGenericRingBuffer::new(),
            next_state: ConstGenericRingBuffer::new(),
            action: ConstGenericRingBuffer::new(),
            reward: ConstGenericRingBuffer::new(),
            done: ConstGenericRingBuffer::new(),
            backend: PhantomData,
        }
    }

    #[allow(unused)]
    pub fn push(
        &mut self,
        state: E::StateType,
        next_state: E::StateType,
        action: E::ActionType,
        reward: ElemType,
        done: bool,
    ) {
        self.state.push(state);
        self.next_state.push(next_state);
        self.action.push(action);
        self.reward.push(reward);
        self.done.push(done);
    }

    #[allow(unused)]
    pub fn sample<const SIZE: usize>(&self) -> Memory<E, B, SIZE> {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..self.len()).collect();
        indices.shuffle(&mut rng);
        let mut memory = Memory::<E, B, SIZE>::new();
        for index in indices.iter().take(SIZE).copied() {
            memory.push(
                self.state[index],
                self.next_state[index],
                self.action[index],
                self.reward[index],
                self.done[index],
            );
        }
        memory
    }

    fn stack<T, K: TensorKind<B> + BasicOps<B>, F: FnMut(&T) -> Tensor<B, 1, K>>(
        data: &impl RingBuffer<T>,
        accessor: F,
    ) -> Tensor<B, 2, K> {
        Tensor::cat(data.iter().map(accessor).collect::<Vec<_>>(), 0)
            .reshape([data.len() as i32, -1])
    }
    pub fn next_state_batch(&self) -> Tensor<B, 2> {
        Self::stack(&self.state, |state| state.data())
    }
    pub fn state_batch(&self) -> Tensor<B, 2> {
        Self::stack(&self.state, |state| state.data())
    }
    pub fn action_batch(&self) -> Tensor<B, 2, Int> {
        Self::stack(&self.action, |action| {
            Tensor::<B, 1, Int>::from_ints([(*action).into() as i32])
        })
    }
    pub fn reward_batch(&self) -> Tensor<B, 2> {
        Self::stack(&self.reward, |reward| Tensor::from_floats([*reward]))
    }
    pub fn not_done_batch(&self) -> Tensor<B, 2> {
        Self::stack(&self.done, |done| {
            Tensor::from_floats([if *done { 1.0 } else { 0.0 }])
        })
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
    reward: ElemType,
    done: bool,
}

#[cfg(test)]
mod tests {
    use crate::base::{Action, ElemType, Memory, Snapshot, State};
    use crate::components::env::Environment;
    use burn::backend::NdArrayBackend;
    use burn::tensor::backend::Backend;
    use burn::tensor::Tensor;

    #[derive(Debug, Copy, Clone, Default)]
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

    #[derive(Debug, Copy, Clone, Default)]
    struct TestState {
        data: ElemType,
    }

    impl State for TestState {
        type Data = ElemType;

        fn data<B: Backend>(&self) -> Tensor<B, 1> {
            Tensor::<B, 1>::from_floats([self.data])
        }
        fn size() -> usize {
            1
        }
    }

    struct TestEnv {}

    type TestBackend = NdArrayBackend<ElemType>;

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
        let mut memory = Memory::<TestEnv, TestBackend, 16>::new();
        for i in 0..20 {
            memory.push(
                TestState {
                    data: i as ElemType,
                },
                TestState {
                    data: -i as ElemType,
                },
                TestAction { data: i },
                0.1,
                false,
            );
        }
        let sample = memory.sample::<5>();
        assert_eq!(sample.len(), 5);
    }
}
