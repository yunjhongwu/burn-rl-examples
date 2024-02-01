use crate::utils::soft_update_linear;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamWConfig;
use burn::tensor::activation::{relu, softmax};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use burn_rl::agent::{SACActor, SACCritic, SACNets, SACOptimizer, SACTrainingConfig, SAC};
use burn_rl::base::{Action, Agent, ElemType, Environment, Memory, Model, State};

#[derive(Module, Debug)]
pub struct Actor<B: Backend> {
    linear_0: Linear<B>,
    linear_1: Linear<B>,
    linear_2: Linear<B>,
}

impl<B: Backend> Actor<B> {
    pub fn new(input_size: usize, dense_size: usize, output_size: usize) -> Self {
        Self {
            linear_0: LinearConfig::new(input_size, dense_size).init(&Default::default()),
            linear_1: LinearConfig::new(dense_size, dense_size).init(&Default::default()),
            linear_2: LinearConfig::new(dense_size, output_size).init(&Default::default()),
        }
    }
}

impl<B: Backend> Model<B, Tensor<B, 2>, Tensor<B, 2>> for Actor<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let layer_0_output = relu(self.linear_0.forward(input));
        let layer_1_output = relu(self.linear_1.forward(layer_0_output));

        softmax(self.linear_2.forward(layer_1_output), 1)
    }

    fn infer(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(input)
    }
}

impl<B: Backend> SACActor<B> for Actor<B> {}

#[derive(Module, Debug)]
pub struct Critic<B: Backend> {
    linear_0: Linear<B>,
    linear_1: Linear<B>,
    linear_2: Linear<B>,
}

impl<B: Backend> Critic<B> {
    pub fn new(input_size: usize, dense_size: usize, output_size: usize) -> Self {
        Self {
            linear_0: LinearConfig::new(input_size, dense_size).init(&Default::default()),
            linear_1: LinearConfig::new(dense_size, dense_size).init(&Default::default()),
            linear_2: LinearConfig::new(dense_size, output_size).init(&Default::default()),
        }
    }
}

impl<B: Backend> Model<B, Tensor<B, 2>, Tensor<B, 2>> for Critic<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let layer_0_output = relu(self.linear_0.forward(input));
        let layer_1_output = relu(self.linear_1.forward(layer_0_output));

        self.linear_2.forward(layer_1_output)
    }

    fn infer(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(input)
    }
}

impl<B: Backend> SACCritic<B> for Critic<B> {
    fn soft_update(this: &mut Self, that: &Self, tau: ElemType) {
        soft_update_linear(&mut this.linear_0, &that.linear_0, tau);
        soft_update_linear(&mut this.linear_1, &that.linear_1, tau);
        soft_update_linear(&mut this.linear_2, &that.linear_2, tau);
    }
}

#[allow(unused)]
const MEMORY_SIZE: usize = 4096;
const DENSE_SIZE: usize = 32;
type MyAgent<E, B> = SAC<E, B, Actor<B>>;

#[allow(unused)]
pub fn run<E: Environment, B: AutodiffBackend>(
    num_episodes: usize,
    visualized: bool,
) -> impl Agent<E> {
    let mut env = E::new(visualized);
    let state_dim = <<E as Environment>::StateType as State>::size();
    let action_dim = <<E as Environment>::ActionType as Action>::size();

    let mut actor = Actor::<B>::new(state_dim, DENSE_SIZE, action_dim);
    let mut critic_1 = Critic::<B>::new(state_dim, DENSE_SIZE, action_dim);
    let mut critic_2 = Critic::<B>::new(state_dim, DENSE_SIZE, action_dim);
    let mut nets = SACNets::<B, Actor<B>, Critic<B>>::new(actor, critic_1, critic_2);

    let mut agent = MyAgent::default();

    let config = SACTrainingConfig::default();

    let mut memory = Memory::<E, B, MEMORY_SIZE>::default();

    let optimizer_config = AdamWConfig::new().with_grad_clipping(config.clip_grad.clone());

    let mut optimizer = SACOptimizer::new(
        optimizer_config.clone().init(),
        optimizer_config.clone().init(),
        optimizer_config.clone().init(),
        optimizer_config.init(),
    );

    let mut policy_net = agent.model().clone();

    let mut step = 0_usize;

    for episode in 0..num_episodes {
        let mut episode_done = false;
        let mut episode_reward = 0.0;
        let mut episode_duration = 0_usize;
        let mut state = env.state();

        while !episode_done {
            if let Some(action) = MyAgent::<E, _>::react_with_model(&state, &nets.actor) {
                let snapshot = env.step(action);

                episode_reward += <<E as Environment>::RewardType as Into<ElemType>>::into(
                    snapshot.reward().clone(),
                );

                memory.push(
                    state,
                    *snapshot.state(),
                    action,
                    snapshot.reward().clone(),
                    snapshot.done(),
                );

                if config.batch_size < memory.len() {
                    nets = agent.train::<MEMORY_SIZE, _>(nets, &memory, &mut optimizer, &config);
                }

                step += 1;
                episode_duration += 1;

                if snapshot.done() || episode_duration >= E::MAX_STEPS {
                    env.reset();
                    episode_done = true;

                    println!(
                        "{{\"episode\": {}, \"reward\": {:.4}, \"duration\": {}}}",
                        episode, episode_reward, episode_duration
                    );
                } else {
                    state = *snapshot.state();
                }
            }
        }
    }

    agent.valid(nets.actor)
}
