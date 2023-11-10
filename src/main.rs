use crate::components::agent::Agent;
use crate::components::env::Environment;
use crate::env::cart_pole::CartPole;

mod agent;
mod base;
mod components;
mod demo;
mod env;

fn main() {
    let mut env = CartPole::new();
    let mut agent = agent::random::Random::<CartPole>::default();
    let mut state = env.state();
    for _ in 0..100 {
        let mut snapshot = env.step(agent.react(&state));
        if snapshot.done() {
            snapshot = env.reset();
        }
        state = *snapshot.state();
    }
}
