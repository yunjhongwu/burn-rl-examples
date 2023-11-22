mod dqn;
mod ppo;
mod utils;

fn main() {
    dqn::run(512);
    // ppo::run(512);
}
