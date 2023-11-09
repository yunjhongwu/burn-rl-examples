use crate::components::env::Environment;
use crate::env::cart_pole::CartPole;

mod agent;
mod base;
mod components;
mod demo;
mod env;

fn main() {
    let mut env = CartPole::new();
    let snapshot = env.step(<CartPole as Environment>::Action::Left);
    println!("snapshot: {:?}", snapshot);
}
