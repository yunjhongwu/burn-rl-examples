use crate::base::State;
use crate::components::agent::Agent;
use crate::components::env::Environment;
use crate::env::cart_pole::CartPole;

#[derive(Default)]
pub struct RuleBasedCartPole {}

impl RuleBasedCartPole {}

impl Agent for RuleBasedCartPole {
    type StateType = <CartPole as Environment>::StateType;
    type ActionType = <CartPole as Environment>::ActionType;
    fn react(&mut self, state: &Self::StateType) -> Self::ActionType {
        if state.data()[2] < 0.0 {
            Self::ActionType::Left
        } else {
            Self::ActionType::Right
        }
    }

    fn collect(&mut self, _reward: f32, _done: bool) {}

    fn reset(&mut self) {}

    fn is_eval(&self) -> bool {
        false
    }
}
