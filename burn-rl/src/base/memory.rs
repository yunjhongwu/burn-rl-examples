use crate::base::environment::Environment;
use burn::tensor::backend::Backend;
use burn::tensor::{BasicOps, Tensor, TensorKind};
use rand::Rng;
use ringbuffer::{ConstGenericRingBuffer, RingBuffer};
use std::marker::PhantomData;

pub type MemoryIndices = Vec<usize>;

pub fn sample_indices(indices: MemoryIndices, size: usize) -> MemoryIndices {
    let mut rng = rand::thread_rng();
    let mut sample = Vec::<usize>::new();
    for _ in 0..size {
        unsafe {
            let index = rng.gen_range(0..indices.len());
            sample.push(*indices.get_unchecked(index));
        }
    }

    sample
}

pub fn get_batch<B: Backend, const CAP: usize, T, K: TensorKind<B> + BasicOps<B>>(
    data: &ConstGenericRingBuffer<T, CAP>,
    indices: &MemoryIndices,
    converter: impl Fn(&T) -> Tensor<B, 1, K>,
) -> Tensor<B, 2, K> {
    Tensor::cat(
        indices
            .iter()
            .filter_map(|i| data.get(*i))
            .map(converter)
            .collect::<Vec<_>>(),
        0,
    )
    .reshape([indices.len() as i32, -1])
}

pub struct Memory<E: Environment, B: Backend, const CAP: usize> {
    state: ConstGenericRingBuffer<E::StateType, CAP>,
    next_state: ConstGenericRingBuffer<E::StateType, CAP>,
    action: ConstGenericRingBuffer<E::ActionType, CAP>,
    reward: ConstGenericRingBuffer<E::RewardType, CAP>,
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
        reward: E::RewardType,
        done: bool,
    ) {
        self.state.push(state);
        self.next_state.push(next_state);
        self.action.push(action);
        self.reward.push(reward);
        self.done.push(done);
    }

    pub fn states(&self) -> &ConstGenericRingBuffer<E::StateType, CAP> {
        &self.state
    }

    pub fn next_states(&self) -> &ConstGenericRingBuffer<E::StateType, CAP> {
        &self.next_state
    }

    pub fn actions(&self) -> &ConstGenericRingBuffer<E::ActionType, CAP> {
        &self.action
    }

    pub fn rewards(&self) -> &ConstGenericRingBuffer<E::RewardType, CAP> {
        &self.reward
    }

    pub fn dones(&self) -> &ConstGenericRingBuffer<bool, CAP> {
        &self.done
    }

    pub fn len(&self) -> usize {
        self.state.len()
    }

    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    pub fn clear(&mut self) {
        self.state.clear();
        self.next_state.clear();
        self.action.clear();
        self.reward.clear();
        self.done.clear();
    }
}

#[cfg(test)]
mod tests {
    use crate::base::environment::Environment;
    use crate::base::{get_batch, sample_indices, Action, ElemType, Memory, Snapshot, State};
    use crate::utils::{
        ref_to_action_tensor, ref_to_not_done_tensor, ref_to_reward_tensor, ref_to_state_tensor,
    };
    use burn::backend::NdArrayBackend;
    use burn::tensor::backend::Backend;
    use burn::tensor::{Int, Shape, Tensor};

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

    #[derive(Debug, Copy, Clone)]
    struct TestEnv {}

    type TestBackend = NdArrayBackend<ElemType>;

    impl Environment for TestEnv {
        type StateType = TestState;
        type ActionType = TestAction;
        type RewardType = ElemType;

        fn new(_visualized: bool) -> Self {
            unimplemented!()
        }

        fn state(&self) -> Self::StateType {
            unimplemented!()
        }

        fn reset(&mut self) -> Snapshot<Self::StateType> {
            unimplemented!()
        }

        fn step(&mut self, _action: Self::ActionType) -> Snapshot<Self::StateType> {
            unimplemented!()
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
        let sample_indices = sample_indices((0..memory.len()).collect(), 5);

        let state_batch = get_batch(memory.states(), &sample_indices, |t| {
            t.to_tensor::<TestBackend>()
        });
        assert_eq!(state_batch.shape(), Shape::new([5, 2]));
        let state_sample = state_batch
            .select(0, Tensor::from_ints([0, 1]))
            .to_data()
            .value;
        assert_eq!(state_sample[0] * 2.0, state_sample[1]);

        let next_state_batch: Tensor<TestBackend, 2> =
            get_batch(memory.next_states(), &sample_indices, ref_to_state_tensor);
        assert_eq!(next_state_batch.shape(), Shape::new([5, 2]));
        let next_state_sample = next_state_batch
            .select(0, Tensor::from_ints([0, 1]))
            .to_data()
            .value;
        assert_eq!(next_state_sample[0] * -3.0, next_state_sample[1]);

        let action_batch: Tensor<TestBackend, 2, Int> =
            get_batch(memory.actions(), &sample_indices, ref_to_action_tensor);
        assert_eq!(action_batch.shape(), Shape::new([5, 1]));
        assert_eq!(action_batch.to_data().value[0], state_sample[0] as i64);

        let reward_batch: Tensor<TestBackend, 2> =
            get_batch(memory.rewards(), &sample_indices, ref_to_reward_tensor);
        assert_eq!(reward_batch.shape(), Shape::new([5, 1]));
        assert_eq!(reward_batch.to_data().value[0], 0.1);

        let not_done_batch: Tensor<TestBackend, 2> =
            get_batch(memory.dones(), &sample_indices, ref_to_not_done_tensor);
        assert_eq!(not_done_batch.shape(), Shape::new([5, 1]));
        assert_eq!(not_done_batch.to_data().value[0], 1.0);
    }
}
