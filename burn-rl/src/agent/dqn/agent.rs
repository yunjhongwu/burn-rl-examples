use crate::agent::{DQNModel, DQNTrainingConfig};
use crate::base::agent::Agent;
use crate::base::environment::Environment;
use crate::base::{get_batch, sample_indices, Action, Memory};
use crate::utils::{
    convert_tensor_to_action, ref_to_action_tensor, ref_to_not_done_tensor, ref_to_reward_tensor,
    ref_to_state_tensor, to_state_tensor, update_parameters,
};
use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::Optimizer;
use burn::tensor::backend::{AutodiffBackend, Backend};
use rand::random;
use std::marker::PhantomData;

pub struct DQN<E: Environment, B: Backend, M: DQNModel<B>> {
    target_net: Option<M>,
    state: PhantomData<E::StateType>,
    action: PhantomData<E::ActionType>,
    backend: PhantomData<B>,
}

impl<E: Environment, B: Backend, M: DQNModel<B>> Agent<E> for DQN<E, B, M> {
    fn react(&self, state: &E::StateType) -> Option<E::ActionType> {
        Some(convert_tensor_to_action::<E::ActionType, B>(
            self.target_net
                .as_ref()?
                .infer(ref_to_state_tensor(state).unsqueeze()),
        ))
    }
}

impl<E: Environment, B: Backend, M: DQNModel<B>> DQN<E, B, M> {
    pub fn new(model: M) -> Self {
        Self {
            target_net: Some(model),
            state: PhantomData,
            action: PhantomData,
            backend: PhantomData,
        }
    }

    pub fn model(&self) -> &Option<M> {
        &self.target_net
    }
}

impl<E: Environment, B: AutodiffBackend, M: DQNModel<B>> DQN<E, B, M> {
    pub fn react_with_exploration(
        policy_net: &M,
        state: E::StateType,
        eps_threshold: f64,
    ) -> E::ActionType {
        if random::<f64>() > eps_threshold {
            convert_tensor_to_action::<E::ActionType, B>(
                policy_net.forward(to_state_tensor(state).unsqueeze()),
            )
        } else {
            Action::random()
        }
    }
}

impl<E: Environment, B: AutodiffBackend, M: DQNModel<B> + AutodiffModule<B>> DQN<E, B, M> {
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
        let target_net = self.target_net.take().unwrap();
        let next_state_values = target_net.forward(next_state_batch).max_dim(1).detach();

        let not_done_batch = get_batch(memory.dones(), &sample_indices, ref_to_not_done_tensor);
        let reward_batch = get_batch(memory.rewards(), &sample_indices, ref_to_reward_tensor);

        let expected_state_action_values =
            (next_state_values * not_done_batch).mul_scalar(config.gamma) + reward_batch;

        let loss = MseLoss.forward(
            state_action_values,
            expected_state_action_values,
            Reduction::Mean,
        );

        policy_net = update_parameters(loss, policy_net, optimizer, config.learning_rate.into());

        self.target_net = Some(<M as DQNModel<B>>::soft_update(
            target_net,
            &policy_net,
            config.tau,
        ));

        policy_net
    }

    pub fn valid(mut self) -> DQN<E, B::InnerBackend, M::InnerModule>
    where
        <M as AutodiffModule<B>>::InnerModule: DQNModel<<B as AutodiffBackend>::InnerBackend>,
    {
        DQN::<E, B::InnerBackend, M::InnerModule>::new(self.target_net.take().unwrap().valid())
    }
}
