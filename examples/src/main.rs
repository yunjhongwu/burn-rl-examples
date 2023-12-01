use crate::utils::demo_model;
use burn::backend::NdArray;
use burn_autodiff::Autodiff;
use burn_rl::base::ElemType;
use burn_rl::environment::CartPole;

mod dqn;
mod ppo;
mod sac;
mod utils;

type Backend = Autodiff<NdArray<ElemType>>;
type Env = CartPole;

fn main() {
    let agent = sac::run::<Env, Backend>(512, false);

    demo_model::<Env>(agent);
}
