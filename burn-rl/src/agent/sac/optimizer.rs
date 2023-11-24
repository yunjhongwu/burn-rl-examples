use crate::agent::sac::model::SACTemperature;
use crate::agent::{SACActor, SACCritic};
use burn::module::ADModule;
use burn::optim::Optimizer;
use burn::tensor::backend::ADBackend;
use std::marker::PhantomData;

pub struct SACOptimizer<
    B: ADBackend,
    Actor: SACActor<B> + ADModule<B>,
    Critic: SACCritic<B> + ADModule<B>,
    OptimActor: Optimizer<Actor, B> + Sized,
    OptimCritic: Optimizer<Critic, B> + Sized,
    OptimTemperature: Optimizer<SACTemperature<B>, B> + Sized,
> {
    pub actor_optimizer: OptimActor,
    pub critic_1_optimizer: OptimCritic,
    pub critic_2_optimizer: OptimCritic,
    pub temperature_optimizer: OptimTemperature,
    actor: PhantomData<Actor>,
    critic: PhantomData<Critic>,
    backend: PhantomData<B>,
}

impl<
        B: ADBackend,
        Actor: SACActor<B> + ADModule<B>,
        Critic: SACCritic<B> + ADModule<B>,
        OptimActor: Optimizer<Actor, B> + Sized,
        OptimCritic: Optimizer<Critic, B> + Sized,
        OptimTemperature: Optimizer<SACTemperature<B>, B> + Sized,
    > SACOptimizer<B, Actor, Critic, OptimActor, OptimCritic, OptimTemperature>
{
    pub fn new(
        actor_optimizer: OptimActor,
        critic_1_optimizer: OptimCritic,
        critic_2_optimizer: OptimCritic,
        temperature_optimizer: OptimTemperature,
    ) -> Self {
        Self {
            actor_optimizer,
            critic_1_optimizer,
            critic_2_optimizer,
            temperature_optimizer,
            actor: PhantomData,
            critic: PhantomData,
            backend: PhantomData,
        }
    }
}
