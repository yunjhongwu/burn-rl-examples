use burn_rl::base::{Agent, Environment};

#[allow(unused)]
pub fn demo_model<E: Environment>(agent: impl Agent<E>) {
    let mut env = E::new(true);
    let mut state = env.state();
    let mut done = false;
    while !done {
        let action = agent.react(&state);
        let snapshot = env.step(action);
        state = *snapshot.state();
        done = snapshot.done();
    }
}
