use crate::utils::demo_model;
use burn::backend::ndarray::NdArrayBackend;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamWConfig;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_autodiff::ADBackendDecorator;
use burn_rl::agent::DQN;
use burn_rl::agent::{DQNModel, DQNTrainingConfig};
use burn_rl::base::{Action, ElemType, Environment, Memory, Model, State};
use burn_rl::environment::CartPole;

#[allow(unused)]
type DQNBackend = ADBackendDecorator<NdArrayBackend<ElemType>>;
#[allow(unused)]
type MyEnv = CartPole;
#[allow(unused)]
type MyAgent = DQN<MyEnv, DQNBackend, Net<DQNBackend>>;

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
            linear_0: LinearConfig::new(input_size, dense_size).init(),
            linear_1: LinearConfig::new(dense_size, dense_size).init(),
            linear_2: LinearConfig::new(dense_size, output_size).init(),
        }
    }

    fn soft_update_tensor<const N: usize>(
        this: &Param<Tensor<B, N>>,
        that: &Param<Tensor<B, N>>,
        tau: f64,
    ) -> Param<Tensor<B, N>> {
        let that_weight = that.val();
        let this_weight = this.val();
        let new_this_weight = this_weight * (1.0 - tau) + that_weight * tau;

        Param::from(new_this_weight.no_grad())
    }
    fn soft_update_linear(this: &mut Linear<B>, that: &Linear<B>, tau: f64) {
        this.weight = Self::soft_update_tensor(&this.weight, &that.weight, tau);
        if let (Some(this_bias), Some(that_bias)) = (&mut this.bias, &that.bias) {
            this.bias = Some(Self::soft_update_tensor(this_bias, that_bias, tau));
        }
    }
}

impl<B: Backend> Model<B, Tensor<B, 2>, Tensor<B, 2>> for Net<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let layer_0_output = relu(self.linear_0.forward(input));
        let layer_1_output = relu(self.linear_1.forward(layer_0_output));

        relu(self.linear_2.forward(layer_1_output))
    }
}

impl<B: Backend> DQNModel<B> for Net<B> {
    fn soft_update(this: &mut Self, that: &Self, tau: f64) {
        Self::soft_update_linear(&mut this.linear_0, &that.linear_0, tau);
        Self::soft_update_linear(&mut this.linear_1, &that.linear_1, tau);
        Self::soft_update_linear(&mut this.linear_2, &that.linear_2, tau);
    }
}

#[allow(unused)]
const MEMORY_SIZE: usize = 4096;

#[allow(unused)]
pub fn run(num_episodes: usize) {
    let eps_decay = 1000.0;
    let eps_start = 0.9;
    let eps_end = 0.05;
    let dense_size = 128_usize;

    let mut env = MyEnv::new(false);

    let model = Net::<DQNBackend>::new(
        <<MyEnv as Environment>::StateType as State>::size(),
        dense_size,
        <<MyEnv as Environment>::ActionType as Action>::size(),
    );
    let mut agent = MyAgent::new(model);

    let mut memory = Memory::<MyEnv, DQNBackend, MEMORY_SIZE>::default();

    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Value(100.0)))
        .init();

    let mut policy_net = agent.model().clone();

    let mut step = 0_usize;

    for episode in 0..num_episodes {
        let mut episode_done = false;
        let mut episode_reward = 0.0;
        let mut episode_duration = 0_usize;
        let mut state = env.state();

        let config = DQNTrainingConfig::default();

        while !episode_done {
            let eps_threshold =
                eps_end + (eps_start - eps_end) * f64::exp(-(step as f64) / eps_decay);
            let action = MyAgent::react_with_exploration(&policy_net, state, eps_threshold);
            let snapshot = env.step(action);

            episode_reward += snapshot.reward();

            memory.push(
                state,
                *snapshot.state(),
                action,
                snapshot.reward(),
                snapshot.done(),
            );

            if config.batch_size < memory.len() {
                policy_net =
                    agent.train::<MEMORY_SIZE>(policy_net, &memory, &mut optimizer, &config);
            }

            step += 1;
            episode_duration += 1;

            if snapshot.done() || episode_duration >= MyEnv::MAX_STEPS {
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

    let valid_agent = agent.valid();
    demo_model::<MyEnv>(valid_agent);
}
