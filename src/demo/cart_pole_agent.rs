use crate::components::agent::Agent;
use crate::components::env::Environment;
use crate::env::cart_pole::CartPole;

#[derive(Default)]
pub struct RuleBasedCartPole {}

impl RuleBasedCartPole {}

impl Agent for RuleBasedCartPole {
    type State = <CartPole as Environment>::State;
    type Action = <CartPole as Environment>::Action;
    fn react(&mut self, state: &Self::State) -> Self::Action {
        if state[2] < 0.0 {
            Self::Action::Left
        } else {
            Self::Action::Right
        }
    }

    fn collect(&mut self, _reward: f32, _done: bool) {}

    fn reset(&mut self) {}

    fn is_eval(&self) -> bool {
        false
    }
}
