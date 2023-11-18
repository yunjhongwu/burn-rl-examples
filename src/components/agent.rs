use crate::components::env::Environment;

pub trait Agent<E: Environment> {
    fn react(&self, state: &E::StateType) -> E::ActionType;
}
