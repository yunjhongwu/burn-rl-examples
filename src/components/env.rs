use crate::utils::Snapshot;
use std::fmt::Debug;

pub trait Environment<State: Debug, Action> {
    fn render(&mut self);
    fn reset(&mut self) -> Snapshot<State>;
    fn step(&mut self, action: Action) -> Snapshot<State>;
}
