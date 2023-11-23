use crate::agent::DQNModel;
use crate::base::agent::Agent;
use crate::base::environment::Environment;
use crate::base::{get_batch, sample_indices, Action, ElemType, Memory};
use crate::utils::{
    convert_tenor_to_action, ref_to_action_tensor, ref_to_not_done_tensor, ref_to_reward_tensor,
    ref_to_state_tensor, to_state_tensor, update_parameters,
};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::ADModule;
use burn::nn::loss::{MSELoss, Reduction};
use burn::optim::Optimizer;
use burn::tensor::backend::{ADBackend, Backend};
use rand::random;
use std::marker::PhantomData;

pub struct DQNTrainingConfig {
    pub gamma: ElemType,
    pub tau: ElemType,
    pub learning_rate: ElemType,
    pub batch_size: usize,
    pub clip_grad: Option<GradientClippingConfig>,
}

impl Default for DQNTrainingConfig {
    fn default() -> Self {
        Self {
            gamma: 0.999,
            tau: 0.005,
            learning_rate: 0.001,
            batch_size: 32,
            clip_grad: Some(GradientClippingConfig::Value(100.0)),
        }
    }
}

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
                .forward(ref_to_state_tensor(state).unsqueeze()),
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
                policy_net.forward(to_state_tensor(state).unsqueeze()),
            )
        } else {
            Action::random()
        }
    }
}

impl<E: Environment, B: ADBackend, M: DQNModel<B> + ADModule<B>> DQN<E, B, M> {
    pub fn train<const CAP: usize>(
        &mut self,
        mut policy_net: M,
        memory: &Memory<E, B, CAP>,
        optimizer: &mut (impl Optimizer<M, B> + Sized),
        config: &DQNTrainingConfig,
    ) -> M {
        let sample_indices = sample_indices((0..memory.len()).collect(), config.batch_size);
        let state_batch = get_batch(memory.states(), &sample_indices, ref_to_state_tensor);
        let action_batch = get_batch(memory.actions(), &sample_indices, ref_to_action_tensor);
        let state_action_values = policy_net.forward(state_batch).gather(1, action_batch);

        let next_state_batch =
            get_batch(memory.next_states(), &sample_indices, ref_to_state_tensor);
        let next_state_values = self
            .target_net
            .forward(next_state_batch)
            .max_dim(1)
            .detach();

        let not_done_batch = get_batch(memory.dones(), &sample_indices, ref_to_not_done_tensor);
        let reward_batch = get_batch(memory.rewards(), &sample_indices, ref_to_reward_tensor);

        let expected_state_action_values =
            (next_state_values * not_done_batch).mul_scalar(config.gamma) + reward_batch;

        let loss = MSELoss::default().forward(
            state_action_values,
            expected_state_action_values,
            Reduction::Mean,
        );

        policy_net = update_parameters(loss, policy_net, optimizer, config.learning_rate.into());

        <M as DQNModel<B>>::soft_update(&mut self.target_net, &policy_net, config.tau);

        policy_net
    }

    pub fn valid(&self) -> DQN<E, B::InnerBackend, M::InnerModule>
    where
        <M as ADModule<B>>::InnerModule: DQNModel<<B as ADBackend>::InnerBackend>,
    {
        DQN::<E, B::InnerBackend, M::InnerModule>::new(self.target_net.clone().valid())
    }
}
