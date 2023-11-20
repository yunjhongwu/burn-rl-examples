use crate::agent::{DQNMemory, DQNModel};
use crate::base::agent::Agent;
use crate::base::environment::Environment;
use crate::base::Action;
use crate::utils::{convert_state_to_tensor, convert_tenor_to_action};
use burn::module::ADModule;
use burn::nn::loss::{MSELoss, Reduction};
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::{ADBackend, Backend};
use rand::random;
use std::marker::PhantomData;

const GAMMA: f64 = 0.999;
const TAU: f64 = 0.005;
const LR: f64 = 0.001;

pub struct DQN<E: Environment, B: Backend, M: DQNModel<B>> {
    target_net: M,
    state: PhantomData<E::StateType>,
    action: PhantomData<E::ActionType>,
    backend: PhantomData<B>,
}

impl<E: Environment, B: Backend, M: DQNModel<B>> Agent<E> for DQN<E, B, M> {
    fn react(&self, state: &E::StateType) -> E::ActionType {
        convert_tenor_to_action::<E::ActionType, B>(
            self.target_net
                .forward(convert_state_to_tensor::<E::StateType, B>(*state).unsqueeze()),
        )
    }
}

impl<E: Environment, B: Backend, M: DQNModel<B>> DQN<E, B, M> {
    pub fn new(model: M) -> Self {
        Self {
            target_net: model,
            state: PhantomData,
            action: PhantomData,
            backend: PhantomData,
        }
    }

    pub fn model(&self) -> &M {
        &self.target_net
    }
}
impl<E: Environment, B: ADBackend, M: DQNModel<B>> DQN<E, B, M> {
    pub fn react_with_exploration(
        policy_net: &M,
        state: E::StateType,
        eps_threshold: f64,
    ) -> E::ActionType {
        if random::<f64>() > eps_threshold {
            convert_tenor_to_action::<E::ActionType, B>(
                policy_net.forward(convert_state_to_tensor::<E::StateType, B>(state).unsqueeze()),
            )
        } else {
            Action::random()
        }
    }
}

impl<E: Environment, B: ADBackend, M: DQNModel<B> + ADModule<B>> DQN<E, B, M> {
    pub fn train<const BATCH_SIZE: usize>(
        &mut self,
        mut policy_net: M,
        sample: DQNMemory<E, B, BATCH_SIZE>,
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
        <M as DQNModel<B>>::soft_update(&mut self.target_net, &policy_net, TAU);

        policy_net
    }

    pub fn valid(&self) -> DQN<E, B::InnerBackend, M::InnerModule>
    where
        <M as ADModule<B>>::InnerModule: DQNModel<<B as ADBackend>::InnerBackend>,
    {
        DQN::<E, B::InnerBackend, M::InnerModule>::new(self.target_net.clone().valid())
    }
}
