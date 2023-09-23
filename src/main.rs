use crate::demo::cart_pole_agent::RuleBasedCartPole;

mod agent;
mod base;
mod components;
mod demo;
mod env;

fn main() {
    env::cart_pole::Visualizer::<RuleBasedCartPole>::run();
}
