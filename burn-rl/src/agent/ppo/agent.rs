use crate::agent::ppo::model::{PPOModel, PPOOutput};
use crate::base::{get_batch, sample_indices, Agent, ElemType, Environment, Memory, MemoryIndices};
use crate::utils::{
    elementwise_min, get_elem, ref_to_action_tensor, ref_to_not_done_tensor, ref_to_reward_tensor,
    ref_to_state_tensor, sample_action_from_tensor, to_state_tensor, update_parameters,
};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::ADModule;
use burn::nn::loss::{MSELoss, Reduction};
use burn::optim::Optimizer;
use burn::tensor::backend::{ADBackend, Backend};
use burn::tensor::Tensor;
use std::marker::PhantomData;

pub struct PPOTrainingConfig {
    gamma: ElemType,
    lambda: ElemType,
    epsilon_clip: ElemType,
    critic_weight: ElemType,
    entropy_weight: ElemType,
    learning_rate: ElemType,
    epochs: usize,
    batch_size: usize,
    pub clip_grad: Option<GradientClippingConfig>,
}

impl Default for PPOTrainingConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            epsilon_clip: 0.2,
            critic_weight: 0.5,
            entropy_weight: 0.01,
            learning_rate: 0.001,
            epochs: 8,
            batch_size: 8,
            clip_grad: Some(GradientClippingConfig::Value(100.0)),
        }
    }
}

pub struct PPO<E: Environment, B: Backend, M: PPOModel<B>> {
    model: Option<M>,
    state: PhantomData<E::StateType>,
    action: PhantomData<E::ActionType>,
    backend: PhantomData<B>,
}

impl<E: Environment, B: Backend, M: PPOModel<B>> Agent<E> for PPO<E, B, M> {
    fn react(&self, state: &E::StateType) -> E::ActionType {
        sample_action_from_tensor::<E::ActionType, B>(
            self.model
                .as_ref()
                .unwrap()
                .forward(to_state_tensor(*state).unsqueeze())
                .policies,
        )
    }
}

impl<E: Environment, B: Backend, M: PPOModel<B>> PPO<E, B, M> {
    pub fn new(model: M) -> Self {
        Self {
            model: Some(model),
            state: PhantomData,
            action: PhantomData,
            backend: PhantomData,
        }
    }

    pub fn react_with_model(state: &E::StateType, model: &M) -> E::ActionType {
        sample_action_from_tensor::<E::ActionType, _>(
            model.forward(to_state_tensor(*state).unsqueeze()).policies,
        )
    }
}

impl<E: Environment, B: Backend, M: PPOModel<B>> Default for PPO<E, B, M> {
    fn default() -> Self {
        Self {
            model: None,
            state: PhantomData,
            action: PhantomData,
            backend: PhantomData,
        }
    }
}

impl<E: Environment, B: ADBackend, M: PPOModel<B> + ADModule<B>> PPO<E, B, M> {
    pub fn train<const CAP: usize>(
        mut policy_net: M,
        memory: &Memory<E, B, CAP>,
        optimizer: &mut (impl Optimizer<M, B> + Sized),
        config: &PPOTrainingConfig,
    ) -> M {
        let memory_indices = (0..memory.len()).collect::<MemoryIndices>();
        let PPOOutput {
            policies: mut old_polices,
            values: mut old_values,
        } = policy_net.forward(get_batch(
            memory.states(),
            &memory_indices,
            ref_to_state_tensor,
        ));
        old_polices = old_polices.detach();
        old_values = old_values.detach();

        let GAEOutput {
            expected_returns,
            advantages,
        } = get_gae(
            old_values,
            get_batch(memory.rewards(), &memory_indices, ref_to_reward_tensor),
            get_batch(memory.dones(), &memory_indices, ref_to_not_done_tensor),
            config.gamma,
            config.lambda,
        );

        for _ in 0..config.epochs {
            for _ in 0..(memory.len() / config.batch_size) {
                let sample_indices = sample_indices(memory_indices.clone(), config.batch_size);

                let sample_indices_tensor = Tensor::from_ints(
                    sample_indices
                        .iter()
                        .map(|x| *x as i32)
                        .collect::<Vec<_>>()
                        .as_slice(),
                );

                let state_batch = get_batch(memory.states(), &sample_indices, ref_to_state_tensor);
                let action_batch =
                    get_batch(memory.actions(), &sample_indices, ref_to_action_tensor);
                let old_policy_batch = old_polices.clone().select(0, sample_indices_tensor.clone());
                let advantage_batch = advantages.clone().select(0, sample_indices_tensor.clone());
                let expected_return_batch = expected_returns
                    .clone()
                    .select(0, sample_indices_tensor)
                    .detach();

                let PPOOutput {
                    policies: policy_batch,
                    values: value_batch,
                } = policy_net.forward(state_batch);

                let ratios = policy_batch
                    .clone()
                    .div(old_policy_batch)
                    .gather(1, action_batch);
                let clipped_ratios = ratios
                    .clone()
                    .clamp(1.0 - config.epsilon_clip, 1.0 + config.epsilon_clip);

                let actor_loss = -elementwise_min(
                    ratios * advantage_batch.clone(),
                    clipped_ratios * advantage_batch,
                )
                .sum();
                let critic_loss =
                    MSELoss::default().forward(expected_return_batch, value_batch, Reduction::Sum);
                let policy_negative_entropy = -(policy_batch.clone().log() * policy_batch)
                    .sum_dim(1)
                    .mean();

                let loss = actor_loss
                    + critic_loss.mul_scalar(config.critic_weight)
                    + policy_negative_entropy.mul_scalar(config.entropy_weight);
                policy_net =
                    update_parameters(loss, policy_net, optimizer, config.learning_rate.into());
            }
        }
        policy_net
    }

    pub fn valid(&self, model: M) -> PPO<E, B::InnerBackend, M::InnerModule>
    where
        <M as ADModule<B>>::InnerModule: PPOModel<<B as ADBackend>::InnerBackend>,
    {
        PPO::<E, B::InnerBackend, M::InnerModule>::new(model.valid())
    }
}

pub(crate) struct GAEOutput<B: Backend> {
    expected_returns: Tensor<B, 2>,
    advantages: Tensor<B, 2>,
}

impl<B: Backend> GAEOutput<B> {
    fn new(expected_returns: Tensor<B, 2>, advantages: Tensor<B, 2>) -> Self {
        Self {
            expected_returns,
            advantages,
        }
    }
}

pub(crate) fn get_gae<B: Backend>(
    values: Tensor<B, 2>,
    rewards: Tensor<B, 2>,
    not_dones: Tensor<B, 2>,
    gamma: ElemType,
    lambda: ElemType,
) -> GAEOutput<B> {
    let mut returns = vec![0.0 as ElemType; rewards.shape().num_elements()];
    let mut advantages = returns.clone();

    let mut running_return: ElemType = 0.0;
    let mut running_advantage: ElemType = 0.0;

    for i in (0..rewards.shape().num_elements()).rev() {
        let reward = get_elem(i, &rewards).unwrap();
        let not_done = get_elem(i, &not_dones).unwrap();

        running_return = reward + gamma * running_return * not_done;
        running_advantage = reward - get_elem(i, &values).unwrap()
            + gamma
                * not_done
                * (get_elem(i + 1, &values).unwrap_or(0.0) + lambda * running_advantage);

        returns[i] = running_return;
        advantages[i] = running_advantage;
    }

    GAEOutput::new(
        Tensor::from_floats(returns.as_slice()).reshape([returns.len(), 1]),
        Tensor::from_floats(advantages.as_slice()).reshape([advantages.len(), 1]),
    )
}
