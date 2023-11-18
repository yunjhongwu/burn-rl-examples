use crate::base::environment::Environment;
use crate::base::{ElemType, State};
use burn::tensor::backend::Backend;
use burn::tensor::{BasicOps, Int, Tensor, TensorKind};
use rand::prelude::SliceRandom;
use ringbuffer::{ConstGenericRingBuffer, RingBuffer};
use std::marker::PhantomData;

pub struct Memory<E: Environment, B: Backend, const CAP: usize> {
    state: ConstGenericRingBuffer<Tensor<B, 1>, CAP>,
    next_state: ConstGenericRingBuffer<Tensor<B, 1>, CAP>,
    action: ConstGenericRingBuffer<u32, CAP>,
    reward: ConstGenericRingBuffer<ElemType, CAP>,
    done: ConstGenericRingBuffer<bool, CAP>,
    environment: PhantomData<E>,
    backend: PhantomData<B>,
}

impl<E: Environment, B: Backend, const CAP: usize> Default for Memory<E, B, CAP> {
    fn default() -> Self {
        Self {
            state: ConstGenericRingBuffer::new(),
            next_state: ConstGenericRingBuffer::new(),
            action: ConstGenericRingBuffer::new(),
            reward: ConstGenericRingBuffer::new(),
            done: ConstGenericRingBuffer::new(),
            environment: PhantomData,
            backend: PhantomData,
        }
    }
}

impl<E: Environment, B: Backend, const CAP: usize> Memory<E, B, CAP> {
    pub fn push(
        &mut self,
        state: E::StateType,
        next_state: E::StateType,
        action: E::ActionType,
        reward: ElemType,
        done: bool,
    ) {
        self.state.push(state.to_tensor());
        self.next_state.push(next_state.to_tensor());
        self.action.push(action.into());
        self.reward.push(reward);
        self.done.push(done);
    }

    pub fn push_tensor(
        &mut self,
        state: Tensor<B, 1>,
        next_state: Tensor<B, 1>,
        action: u32,
        reward: ElemType,
        done: bool,
    ) {
        self.state.push(state);
        self.next_state.push(next_state);
        self.action.push(action);
        self.reward.push(reward);
        self.done.push(done);
    }

    pub fn sample<const SIZE: usize>(&self) -> Memory<E, B, SIZE> {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..self.len()).collect();
        indices.shuffle(&mut rng);
        let mut memory = Memory::<E, B, SIZE>::default();
        for index in indices.iter().take(SIZE).copied() {
            memory.push_tensor(
                self.state[index].clone(),
                self.next_state[index].clone(),
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
        Self::stack(&self.next_state, |state| state.clone())
    }
    pub fn state_batch(&self) -> Tensor<B, 2> {
        Self::stack(&self.state, |state| state.clone())
    }
    pub fn action_batch(&self) -> Tensor<B, 2, Int> {
        Self::stack(&self.action, |action| {
            Tensor::<B, 1, Int>::from_ints([*action as i32])
        })
    }
    pub fn reward_batch(&self) -> Tensor<B, 2> {
        Self::stack(&self.reward, |reward| Tensor::from_floats([*reward]))
    }
    pub fn not_done_batch(&self) -> Tensor<B, 2> {
        Self::stack(&self.done, |done| {
            Tensor::from_floats([if *done { 0.0 } else { 1.0 }])
        })
    }

    pub fn len(&self) -> usize {
        self.state.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::base::environment::Environment;
    use crate::base::{Action, ElemType, Memory, Snapshot, State};
    use burn::backend::NdArrayBackend;
    use burn::tensor::backend::Backend;
    use burn::tensor::{Shape, Tensor};

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
        let mut memory = Memory::<TestEnv, TestBackend, 16>::default();
        for i in 0..20 {
            memory.push(
                TestState {
                    data: [i as ElemType, (i * 2) as ElemType],
                },
                TestState {
                    data: [-i as ElemType, (i * 3) as ElemType],
                },
                TestAction { data: i },
                0.1,
                false,
            );
        }
        let sample = memory.sample::<5>();
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
