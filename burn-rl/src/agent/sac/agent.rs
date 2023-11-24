use crate::agent::sac::model::{SACNets, SACTemperature};
use crate::agent::{SACActor, SACCritic, SACOptimizer, SACTrainingConfig};
use crate::base::agent::Agent;
use crate::base::environment::Environment;
use crate::base::{get_batch, sample_indices, Action, ElemType, Memory};
use crate::utils::{
    convert_tenor_to_action, elementwise_min, ref_to_action_tensor, ref_to_not_done_tensor,
    ref_to_reward_tensor, ref_to_state_tensor, sample_action_from_tensor, to_state_tensor,
    update_parameters,
};
use burn::module::{ADModule, Module};
use burn::nn::loss::{MSELoss, Reduction};
use burn::optim::Optimizer;
use burn::tensor::backend::{ADBackend, Backend};
use rand::random;
use std::marker::PhantomData;

pub struct SAC<E: Environment, B: Backend, Actor: SACActor<B>> {
    actor: Option<Actor>,
    state: PhantomData<E::StateType>,
    action: PhantomData<E::ActionType>,
    backend: PhantomData<B>,
}

impl<E: Environment, B: Backend, Actor: SACActor<B>> Agent<E> for SAC<E, B, Actor> {
    fn react(&self, state: &E::StateType) -> E::ActionType {
        sample_action_from_tensor::<E::ActionType, B>(
            self.actor
                .as_ref()
                .unwrap()
                .forward(to_state_tensor(*state).unsqueeze()),
        )
    }
}

impl<E: Environment, B: Backend, Actor: SACActor<B>> SAC<E, B, Actor> {
    pub fn new(model: Actor) -> Self {
        Self {
            actor: Some(model),
            state: PhantomData,
            action: PhantomData,
            backend: PhantomData,
        }
    }

    pub fn react_with_model(state: &E::StateType, actor: &Actor) -> E::ActionType {
        sample_action_from_tensor::<E::ActionType, _>(
            actor.forward(to_state_tensor(*state).unsqueeze()),
        )
    }

    pub fn model(&self) -> &Option<Actor> {
        &self.actor
    }
}

impl<E: Environment, B: Backend, Actor: SACActor<B>> Default for SAC<E, B, Actor> {
    fn default() -> Self {
        Self {
            actor: None,
            state: PhantomData,
            action: PhantomData,
            backend: PhantomData,
        }
    }
}

impl<E: Environment, B: ADBackend, Actor: SACActor<B>> SAC<E, B, Actor> {
    pub fn react_with_exploration(
        policy_net: &Actor,
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

impl<E: Environment, B: ADBackend, Actor: SACActor<B> + ADModule<B>> SAC<E, B, Actor> {
    #[allow(clippy::too_many_arguments)]
    pub fn train<const CAP: usize, Critic: SACCritic<B> + ADModule<B>>(
        &mut self,
        mut nets: SACNets<B, Actor, Critic>,
        memory: &Memory<E, B, CAP>,
        optimizer: &mut SACOptimizer<
            B,
            Actor,
            Critic,
            impl Optimizer<Actor, B> + Sized,
            impl Optimizer<Critic, B> + Sized,
            impl Optimizer<SACTemperature<B>, B> + Sized,
        >,
        config: &SACTrainingConfig,
    ) -> SACNets<B, Actor, Critic> {
        let action_dim = <<E as Environment>::ActionType as Action>::size();
        let sample_indices = sample_indices((0..memory.len()).collect(), config.batch_size);
        let state_batch = get_batch(memory.states(), &sample_indices, ref_to_state_tensor);

        let action_prob = nets.actor.forward(state_batch.clone());
        let log_prob = action_prob.clone().clamp_min(config.min_probability).log();
        let q1 = nets.critic_1.forward(state_batch.clone());
        let q2 = nets.critic_2.forward(state_batch.clone());
        let q_min = elementwise_min(q1, q2);
        let log_alpha = nets.temperature.forward();
        let alpha = log_alpha.clone().exp();
        let actor_loss = (action_prob.clone() * (alpha.clone() * log_prob.clone() - q_min))
            .sum_dim(1)
            .mean();
        nets.actor = update_parameters(
            actor_loss,
            nets.actor,
            &mut optimizer.actor_optimizer,
            config.learning_rate.into(),
        );

        let entropy = (log_prob.clone() * action_prob).sum_dim(1);
        let temperature_loss = -(log_alpha.clone()
            * (entropy.clone().sub_scalar(action_dim as ElemType)).detach())
        .mean();
        nets.temperature = update_parameters(
            temperature_loss,
            nets.temperature,
            &mut optimizer.temperature_optimizer,
            config.learning_rate.into(),
        );

        let action_batch = get_batch(memory.actions(), &sample_indices, ref_to_action_tensor);
        let next_state_batch =
            get_batch(memory.next_states(), &sample_indices, ref_to_state_tensor);
        let reward_batch = get_batch(memory.rewards(), &sample_indices, ref_to_reward_tensor);
        let not_done_batch = get_batch(memory.dones(), &sample_indices, ref_to_not_done_tensor);

        let action_prob = nets
            .actor
            .clone()
            .no_grad()
            .forward(next_state_batch.clone());

        let q1_target_next = nets
            .critic_1_target
            .clone()
            .no_grad()
            .forward(next_state_batch.clone());
        let q2_target_next = nets
            .critic_2_target
            .clone()
            .no_grad()
            .forward(next_state_batch);
        let q_min_target_next = elementwise_min(q1_target_next, q2_target_next);
        let q_next = action_prob.clone() * (q_min_target_next - alpha.clone() * entropy);
        let q_target =
            reward_batch + not_done_batch.mul_scalar(config.gamma) * q_next.sum_dim(1).no_grad();

        let q1 = nets
            .critic_1
            .forward(state_batch.clone())
            .gather(1, action_batch.clone());
        let critic_1_loss = MSELoss::default().forward(q_target.clone(), q1, Reduction::Sum);
        nets.critic_1 = update_parameters(
            critic_1_loss,
            nets.critic_1,
            &mut optimizer.critic_1_optimizer,
            config.learning_rate.into(),
        );

        let q2 = nets.critic_2.forward(state_batch).gather(1, action_batch);
        let critic_2_loss = MSELoss::default().forward(q_target, q2, Reduction::Sum);
        nets.critic_2 = update_parameters(
            critic_2_loss,
            nets.critic_2,
            &mut optimizer.critic_2_optimizer,
            config.learning_rate.into(),
        );

        SACCritic::soft_update(&mut nets.critic_1_target, &nets.critic_1, config.tau);
        SACCritic::soft_update(&mut nets.critic_2_target, &nets.critic_2, config.tau);

        nets
    }

    pub fn valid(&self, actor: Actor) -> SAC<E, B::InnerBackend, Actor::InnerModule>
    where
        <Actor as ADModule<B>>::InnerModule: SACActor<<B as ADBackend>::InnerBackend>,
    {
        SAC::<E, B::InnerBackend, Actor::InnerModule>::new(actor.valid())
    }
}
