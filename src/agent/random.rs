use crate::base::Action;
use crate::components::agent::Agent;
use crate::components::env::Environment;
use std::marker::PhantomData;

pub struct Random<E: Environment> {
    state: PhantomData<E::StateType>,
    action: PhantomData<E::ActionType>,
}

impl<E: Environment> Default for Random<E> {
    fn default() -> Self {
        Self {
            state: PhantomData,
            action: PhantomData,
        }
    }
}

impl<E: Environment> Agent for Random<E> {
    type StateType = E::StateType;
    type ActionType = E::ActionType;

    fn react(&self, _state: &Self::StateType) -> Self::ActionType {
        Self::ActionType::random()
    }
}
