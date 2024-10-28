use crate::utils::soft_update_linear;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamWConfig;
use burn::tensor::activation::relu;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use burn_rl::agent::DQN;
use burn_rl::agent::{DQNModel, DQNTrainingConfig};
use burn_rl::base::{Action, Agent, ElemType, Environment, Memory, Model, State};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    linear_0: Linear<B>,
    linear_1: Linear<B>,
    linear_2: Linear<B>,
}

impl<B: Backend> Net<B> {
    #[allow(unused)]
    pub fn new(input_size: usize, dense_size: usize, output_size: usize) -> Self {
        Self {
            linear_0: LinearConfig::new(input_size, dense_size).init(&Default::default()),
            linear_1: LinearConfig::new(dense_size, dense_size).init(&Default::default()),
            linear_2: LinearConfig::new(dense_size, output_size).init(&Default::default()),
        }
    }

    fn consume(self) -> (Linear<B>, Linear<B>, Linear<B>) {
        (self.linear_0, self.linear_1, self.linear_2)
    }
}

impl<B: Backend> Model<B, Tensor<B, 2>, Tensor<B, 2>> for Net<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let layer_0_output = relu(self.linear_0.forward(input));
        let layer_1_output = relu(self.linear_1.forward(layer_0_output));

        relu(self.linear_2.forward(layer_1_output))
    }

    fn infer(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(input)
    }
}

impl<B: Backend> DQNModel<B> for Net<B> {
    fn soft_update(this: Self, that: &Self, tau: ElemType) -> Self {
        let (linear_0, linear_1, linear_2) = this.consume();

        Self {
            linear_0: soft_update_linear(linear_0, &that.linear_0, tau),
            linear_1: soft_update_linear(linear_1, &that.linear_1, tau),
            linear_2: soft_update_linear(linear_2, &that.linear_2, tau),
        }
    }
}

#[allow(unused)]
const MEMORY_SIZE: usize = 4096;
const DENSE_SIZE: usize = 128;
const EPS_DECAY: f64 = 1000.0;
const EPS_START: f64 = 0.9;
const EPS_END: f64 = 0.05;

type MyAgent<E, B> = DQN<E, B, Net<B>>;

#[allow(unused)]
pub fn run<E: Environment, B: AutodiffBackend>(
    num_episodes: usize,
    visualized: bool,
) -> impl Agent<E> {
    let mut env = E::new(visualized);

    let model = Net::<B>::new(
        <<E as Environment>::StateType as State>::size(),
        DENSE_SIZE,
        <<E as Environment>::ActionType as Action>::size(),
    );

    let mut agent = MyAgent::new(model);

    let config = DQNTrainingConfig::default();

    let mut memory = Memory::<E, B, MEMORY_SIZE>::default();

    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(config.clip_grad.clone())
        .init();

    let mut policy_net = agent.model().as_ref().unwrap().clone();

    let mut step = 0_usize;

    for episode in 0..num_episodes {
        let mut episode_done = false;
        let mut episode_reward: ElemType = 0.0;
        let mut episode_duration = 0_usize;
        let mut state = env.state();

        while !episode_done {
            let eps_threshold =
                EPS_END + (EPS_START - EPS_END) * f64::exp(-(step as f64) / EPS_DECAY);
            let action =
                DQN::<E, B, Net<B>>::react_with_exploration(&policy_net, state, eps_threshold);
            let snapshot = env.step(action);

            episode_reward +=
                <<E as Environment>::RewardType as Into<ElemType>>::into(snapshot.reward().clone());

            memory.push(
                state,
                *snapshot.state(),
                action,
                snapshot.reward().clone(),
                snapshot.done(),
            );

            if config.batch_size < memory.len() {
                policy_net =
                    agent.train::<MEMORY_SIZE>(policy_net, &memory, &mut optimizer, &config);
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

    agent.valid()
}
