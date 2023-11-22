use burn::backend::NdArrayBackend;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::optim::RMSPropConfig;
use burn::tensor::activation::{relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_autodiff::ADBackendDecorator;
use burn_rl::agent::{PPOModel, PPOOutput, PPOTrainingConfig, PPO};
use burn_rl::base::{Action, Agent, ElemType, Environment, Memory, Model, State};
use burn_rl::environment::CartPole;

#[allow(unused)]
type PPOBackend = ADBackendDecorator<NdArrayBackend<ElemType>>;
#[allow(unused)]
type MyEnv = CartPole;
#[allow(unused)]
type MyAgent = PPO<MyEnv, PPOBackend, Net<PPOBackend>>;

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

impl<B: Backend> Model<B, Tensor<B, 2>, PPOOutput<B>> for Net<B> {
    fn forward(&self, input: Tensor<B, 2>) -> PPOOutput<B> {
        let layer_0_output = relu(self.linear.forward(input));
        let policies = softmax(self.linear_actor.forward(layer_0_output.clone()), 1);
        let values = self.linear_critic.forward(layer_0_output);

        PPOOutput::<B>::new(policies, values)
    }
}

impl<B: Backend> PPOModel<B> for Net<B> {
    fn params(&self) {
        println!(
            "params linear {:?}",
            self.linear.weight.val().to_data().value.to_vec()
        );
        //  println!("params linear_actor {:?}", self.linear_actor.weight.val().to_data().value.to_vec());
        // println!("params linear_critic {:?}", self.linear_critic.weight.val().to_data().value.to_vec());
    }
}

#[allow(unused)]
fn demo_model(agent: impl Agent<MyEnv>) {
    let mut env = MyEnv::new(true);
    let mut state = env.state();
    let mut done = false;
    while !done {
        let action = agent.react(&state);
        let snapshot = env.step(action);
        state = *snapshot.state();
        done = snapshot.done();
    }
}

#[allow(unused)]
const MEMORY_SIZE: usize = 512;

#[allow(unused)]
pub fn run() {
    let num_episodes = 512_usize;
    let dense_size = 64_usize;

    let mut env = MyEnv::new(false);

    let mut model = Net::<PPOBackend>::new(
        <<MyEnv as Environment>::StateType as State>::size(),
        dense_size,
        <<MyEnv as Environment>::ActionType as Action>::size(),
    );
    let agent = MyAgent::default();
    let config = PPOTrainingConfig::default();

    let mut optimizer = RMSPropConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Value(100.0)))
        .init();

    for episode in 0..num_episodes {
        let mut episode_done = false;
        let mut episode_reward = 0.0;
        let mut episode_duration = 0_usize;
        let mut memory = Memory::<MyEnv, PPOBackend, MEMORY_SIZE>::default();

        env.reset();
        while !episode_done {
            let state = env.state();
            let action = MyAgent::react_with_model(&state, &model);
            let snapshot = env.step(action);

            episode_reward += snapshot.reward();

            memory.push(
                state,
                *snapshot.state(),
                action,
                snapshot.reward(),
                snapshot.done(),
            );

            episode_duration += 1;
            episode_done = snapshot.done() || episode_duration >= MyEnv::MAX_STEPS;
        }
        println!(
            "{{\"episode\": {}, \"reward\": {:.4}, \"duration\": {}}}",
            episode, episode_reward, episode_duration
        );

        model = MyAgent::train::<MEMORY_SIZE>(model, &memory, &mut optimizer, &config);
    }

    let valid_agent = agent.valid(model);
    demo_model(valid_agent);
}
