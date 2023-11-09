use crate::base::Snapshot;
use crate::base::{Action, State};
use crate::components::env::Environment;
use gym_rs::core::Env;
use gym_rs::envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation};
use gym_rs::utils::renderer::RenderMode;
use rand::random;
use std::fmt::Debug;

#[derive(Debug, Copy, Clone, Default)]
pub struct CartPoleState {
    data: [f64; 4],
}

impl From<CartPoleObservation> for CartPoleState {
    fn from(observation: CartPoleObservation) -> Self {
        Self {
            data: Vec::<f64>::from(observation).try_into().unwrap(),
        }
    }
}

impl State for CartPoleState {
    type Data = [f64; 4];
    fn data(&self) -> &Self::Data {
        &self.data
    }
}

#[allow(unused)]
#[derive(Debug, Copy, Clone)]
pub enum CartPoleAction {
    Left,
    Right,
}

impl Default for CartPoleAction {
    fn default() -> Self {
        Self::Left
    }
}

impl Action for CartPoleAction {
    fn random() -> Self {
        if random::<f32>() < 0.5 {
            Self::Left
        } else {
            Self::Right
        }
    }

    fn enumerate() -> Vec<Self> {
        vec![Self::Left, Self::Right]
    }
}

#[derive(Debug)]
pub struct CartPole {
    gym_env: CartPoleEnv,
}

impl CartPole {
    #[allow(unused)]
    pub fn new() -> Self {
        Self {
            gym_env: CartPoleEnv::new(RenderMode::None),
        }
    }
}

impl Environment for CartPole {
    type State = CartPoleState;
    type Action = CartPoleAction;

    fn reset(&mut self) -> Snapshot<Self::State> {
        self.gym_env.reset(None, false, None);
        Snapshot::new(self.gym_env.state.into(), 1.0, false)
    }

    fn step(&mut self, action: CartPoleAction) -> Snapshot<CartPoleState> {
        let action = match action {
            CartPoleAction::Left => 0,
            CartPoleAction::Right => 1,
        };
        let action_reward = self.gym_env.step(action);
        Snapshot::new(
            action_reward.observation.into(),
            *action_reward.reward,
            action_reward.done,
        )
    }
}
