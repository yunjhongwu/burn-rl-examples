use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::optim::AdamWConfig;
use burn::tensor::activation::{relu, softmax};
use burn::tensor::backend::{ADBackend, Backend};
use burn::tensor::Tensor;
use burn_rl::agent::{PPOModel, PPOOutput, PPOTrainingConfig, PPO};
use burn_rl::base::{Action, Agent, ElemType, Environment, Memory, Model, State};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    linear: Linear<B>,
    linear_actor: Linear<B>,
    linear_critic: Linear<B>,
}

impl<B: Backend> Net<B> {
    #[allow(unused)]
    pub fn new(input_size: usize, dense_size: usize, output_size: usize) -> Self {
        let initializer = Initializer::XavierUniform { gain: 1.0 };
        Self {
            linear: LinearConfig::new(input_size, dense_size)
                .with_initializer(initializer.clone())
                .init(),
            linear_actor: LinearConfig::new(dense_size, output_size)
                .with_initializer(initializer.clone())
                .init(),
            linear_critic: LinearConfig::new(dense_size, 1)
                .with_initializer(initializer)
                .init(),
        }
    }
}

impl<B: Backend> Model<B, Tensor<B, 2>, PPOOutput<B>, Tensor<B, 2>> for Net<B> {
    fn forward(&self, input: Tensor<B, 2>) -> PPOOutput<B> {
        let layer_0_output = relu(self.linear.forward(input));
        let policies = softmax(self.linear_actor.forward(layer_0_output.clone()), 1);
        let values = self.linear_critic.forward(layer_0_output);

        PPOOutput::<B>::new(policies, values)
    }

    fn infer(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let layer_0_output = relu(self.linear.forward(input));
        softmax(self.linear_actor.forward(layer_0_output.clone()), 1)
    }
}

impl<B: Backend> PPOModel<B> for Net<B> {}
#[allow(unused)]
const MEMORY_SIZE: usize = 512;
const DENSE_SIZE: usize = 128;

type MyAgent<E, B> = PPO<E, B, Net<B>>;

#[allow(unused)]
pub fn run<E: Environment, B: ADBackend>(num_episodes: usize, visualized: bool) -> impl Agent<E> {
    let mut env = E::new(visualized);

    let mut model = Net::<B>::new(
        <<E as Environment>::StateType as State>::size(),
        DENSE_SIZE,
        <<E as Environment>::ActionType as Action>::size(),
    );
    let agent = MyAgent::default();
    let config = PPOTrainingConfig::default();

    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(config.clip_grad.clone())
        .init();
    let mut memory = Memory::<E, B, MEMORY_SIZE>::default();
    for episode in 0..num_episodes {
        let mut episode_done = false;
        let mut episode_reward = 0.0;
        let mut episode_duration = 0_usize;

        env.reset();
        while !episode_done {
            let state = env.state();
            let action = MyAgent::<E, _>::react_with_model(&state, &model);
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

            episode_duration += 1;
            episode_done = snapshot.done() || episode_duration >= E::MAX_STEPS;
        }
        println!(
            "{{\"episode\": {}, \"reward\": {:.4}, \"duration\": {}}}",
            episode, episode_reward, episode_duration
        );

        model = MyAgent::train::<MEMORY_SIZE>(model, &memory, &mut optimizer, &config);
        memory.clear();
    }

    agent.valid(model)
}
