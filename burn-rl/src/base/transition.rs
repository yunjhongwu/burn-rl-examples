use crate::base::Environment;
use std::fmt::Debug;

pub trait Transition<E: Environment>: Debug + Copy + Clone {
    fn state(&self) -> &E::StateType;
    fn action(&self) -> &E::ActionType;
    fn reward(&self) -> &E::RewardType;
    fn next_state(&self) -> &E::StateType;
    fn is_done(&self) -> bool;
}
