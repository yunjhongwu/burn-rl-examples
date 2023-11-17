use crate::base::{Action, Memory, Model, State};
use crate::components::agent::Agent;
use crate::components::env::Environment;
use burn::nn::loss::{MSELoss, Reduction};
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::ADBackend;
use burn::tensor::{ElementConversion, Tensor};
use rand::random;
use std::marker::PhantomData;

const GAMMA: f64 = 0.999;
const TAU: f64 = 0.005;
const LR: f64 = 0.001;

pub struct Dqn<E: Environment, B: ADBackend, M: Model<B>, const EVAL: bool> {
    target_net: M,
    state: PhantomData<E::StateType>,
    action: PhantomData<E::ActionType>,
    backend: PhantomData<B>,
}

impl<E: Environment, B: ADBackend, M: Model<B>, const EVAL: bool> Dqn<E, B, M, EVAL> {
    pub fn new(model: M) -> Self {
        Self {
            target_net: model,
            state: PhantomData,
            action: PhantomData,
            backend: PhantomData,
        }
    }

    fn convert_state_to_tensor(state: <Self as Agent>::StateType) -> Tensor<B, 2> {
        state.to_tensor().unsqueeze()
    }

    fn convert_tenor_to_action(output: Tensor<B, 2>) -> <Self as Agent>::ActionType {
        unsafe {
            output
                .argmax(1)
                .to_data()
                .value
                .get_unchecked(0)
                .elem::<u32>()
                .into()
        }
    }

    pub fn react_with_exploration(
        &self,
        policy_net: &M,
        state: <Self as Agent>::StateType,
        eps_threshold: f64,
    ) -> <Self as Agent>::ActionType {
        if random::<f64>() > eps_threshold {
            Self::convert_tenor_to_action(policy_net.forward(Self::convert_state_to_tensor(state)))
        } else {
            Action::random()
        }
    }
}

impl<E: Environment, B: ADBackend, M: Model<B>, const EVAL: bool> Agent for Dqn<E, B, M, EVAL> {
    type StateType = E::StateType;
    type ActionType = E::ActionType;

    fn react(&self, state: &Self::StateType) -> Self::ActionType {
        Self::convert_tenor_to_action(
            self.target_net
                .forward(Self::convert_state_to_tensor(*state)),
        )
    }
}

impl<E: Environment, B: ADBackend, M: Model<B>> Dqn<E, B, M, false> {
    pub fn train<const BATCH_SIZE: usize>(
        &mut self,
        mut policy_net: M,
        sample: Memory<E, B, BATCH_SIZE>,
        optimizer: &mut (impl Optimizer<M, B> + Sized),
    ) -> M {
        let state_action_values = policy_net
            .forward(sample.state_batch())
            .gather(1, sample.action_batch());

        let next_state_values = self
            .target_net
            .forward(sample.next_state_batch())
            .max_dim(1)
            .detach();

        let not_done_batch = sample.not_done_batch();
        let reward_batch = sample.reward_batch();

        let expected_state_action_values =
            (next_state_values * not_done_batch).mul_scalar(GAMMA) + reward_batch;

        let loss = MSELoss::default().forward(
            state_action_values,
            expected_state_action_values,
            Reduction::Mean,
        );

        let gradients = loss.backward();
        let gradient_params = GradientsParams::from_grads(gradients, &policy_net);

        policy_net = optimizer.step(LR, policy_net, gradient_params);
        self.target_net.soft_update(&policy_net, TAU);

        policy_net
    }

    pub fn model(&self) -> &M {
        &self.target_net
    }
    pub fn to_eval(&self) -> Dqn<E, B, M, true> {
        Dqn::<E, B, M, true> {
            target_net: self.target_net.clone(),
            state: PhantomData,
            action: PhantomData,
            backend: PhantomData,
        }
    }
}
