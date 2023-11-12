mod agent;
mod base;
mod components;
mod demo;
mod env;

type ElemType = f32;

use crate::components::agent::Agent;
use crate::components::env::Environment;
use crate::env::cart_pole::CartPole;
use burn::backend::ndarray::NdArrayBackend;
use burn_autodiff::ADBackendDecorator;

type Backend = ADBackendDecorator<NdArrayBackend<ElemType>>;

pub fn main() {
    let mut env = CartPole::new();
    let mut agent = agent::dqn::Dqn::<CartPole, Backend>::default();
    let mut state = env.state();
    for _ in 0..100 {
        let mut snapshot = env.step(agent.react(&state));
        if snapshot.done() {
            snapshot = env.reset();
        }
        state = *snapshot.state();
    }
}
