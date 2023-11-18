use crate::base::agent::Agent;
use crate::base::environment::Environment;
use crate::base::Action;

#[derive(Default)]
pub struct Random {}

impl<E: Environment> Agent<E> for Random {
    fn react(&self, _state: &E::StateType) -> E::ActionType {
        E::ActionType::random()
    }
}
