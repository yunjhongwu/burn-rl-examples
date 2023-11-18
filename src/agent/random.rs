use crate::base::Action;
use crate::components::agent::Agent;
use crate::components::env::Environment;

#[derive(Default)]
pub struct Random {}

impl<E: Environment> Agent<E> for Random {
    fn react(&self, _state: &E::StateType) -> E::ActionType {
        E::ActionType::random()
    }
}
