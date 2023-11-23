use crate::utils::demo_model;
use burn::backend::NdArrayBackend;
use burn_autodiff::ADBackendDecorator;
use burn_rl::base::ElemType;
use burn_rl::environment::CartPole;

mod dqn;
mod ppo;
mod utils;

type Backend = ADBackendDecorator<NdArrayBackend<ElemType>>;
type Env = CartPole;

fn main() {
    let agent = dqn::run::<Env, Backend>(512, false);

    demo_model::<Env>(agent);
}
