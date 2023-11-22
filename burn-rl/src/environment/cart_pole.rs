use crate::base::environment::Environment;
use crate::base::{Action, State};
use crate::base::{ElemType, Snapshot};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use gym_rs::core::Env;
use gym_rs::envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation};
use gym_rs::utils::renderer::RenderMode;
use std::fmt::Debug;

type StateData = [ElemType; 4];
#[derive(Debug, Copy, Clone)]
pub struct CartPoleState {
    data: StateData,
}

impl From<CartPoleObservation> for CartPoleState {
    fn from(observation: CartPoleObservation) -> Self {
        let vec = Vec::<f64>::from(observation);
        Self {
            data: [
                vec[0] as ElemType,
                vec[1] as ElemType,
                vec[2] as ElemType,
                vec[3] as ElemType,
            ],
        }
    }
}

impl State for CartPoleState {
    type Data = StateData;
    fn to_tensor<B: Backend>(&self) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(self.data)
    }

    fn size() -> usize {
        4
    }
}

#[derive(Debug, Copy, Clone)]
pub enum CartPoleAction {
    Left,
    Right,
}

impl From<u32> for CartPoleAction {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Left,
            1 => Self::Right,
            _ => panic!("Invalid action"),
        }
    }
}

impl From<CartPoleAction> for u32 {
    fn from(action: CartPoleAction) -> Self {
        match action {
            CartPoleAction::Left => 0,
            CartPoleAction::Right => 1,
        }
    }
}

impl Action for CartPoleAction {
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
    pub fn new(visualized: bool) -> Self {
        Self {
            gym_env: CartPoleEnv::new(if visualized {
                RenderMode::Human
            } else {
                RenderMode::None
            }),
        }
    }
}

impl Environment for CartPole {
    type StateType = CartPoleState;
    type ActionType = CartPoleAction;
    type RewardType = ElemType;
    const MAX_STEPS: usize = 500;

    fn state(&self) -> Self::StateType {
        self.gym_env.state.into()
    }

    fn reset(&mut self) -> Snapshot<Self::StateType> {
        self.gym_env.reset(None, false, None);
        Snapshot::new(self.gym_env.state.into(), 1.0, false)
    }

    fn step(&mut self, action: Self::ActionType) -> Snapshot<Self::StateType> {
        let action_reward = self.gym_env.step(action as usize);
        Snapshot::new(
            action_reward.observation.into(),
            *action_reward.reward as ElemType,
            action_reward.done,
        )
    }
}
