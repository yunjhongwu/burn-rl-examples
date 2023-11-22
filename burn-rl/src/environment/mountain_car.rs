use crate::base::environment::Environment;
use crate::base::{Action, State};
use crate::base::{ElemType, Snapshot};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use gym_rs::core::Env;
use gym_rs::envs::classical_control::mountain_car::{MountainCarEnv, MountainCarObservation};
use gym_rs::utils::renderer::RenderMode;
use std::fmt::Debug;

type StateData = [ElemType; 2];
#[derive(Debug, Copy, Clone)]
pub struct MountainCarState {
    data: StateData,
}

impl From<MountainCarObservation> for MountainCarState {
    fn from(observation: MountainCarObservation) -> Self {
        let vec = Vec::<f64>::from(observation);
        Self {
            data: [vec[0] as ElemType, vec[1] as ElemType],
        }
    }
}

impl State for MountainCarState {
    type Data = StateData;
    fn to_tensor<B: Backend>(&self) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(self.data)
    }

    fn size() -> usize {
        2
    }
}

#[derive(Debug, Copy, Clone)]
pub enum MountainCarAction {
    AccelerateToLeft,
    NotAccelerate,
    AccelerateToRight,
}

impl From<u32> for MountainCarAction {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::AccelerateToLeft,
            1 => Self::NotAccelerate,
            2 => Self::AccelerateToRight,
            _ => panic!("Invalid action"),
        }
    }
}

impl From<MountainCarAction> for u32 {
    fn from(action: MountainCarAction) -> Self {
        match action {
            MountainCarAction::AccelerateToLeft => 0,
            MountainCarAction::NotAccelerate => 1,
            MountainCarAction::AccelerateToRight => 2,
        }
    }
}

impl Action for MountainCarAction {
    fn enumerate() -> Vec<Self> {
        vec![
            Self::AccelerateToLeft,
            Self::NotAccelerate,
            Self::AccelerateToRight,
        ]
    }
}

#[derive(Debug)]
pub struct MountainCar {
    gym_env: MountainCarEnv,
}

impl Environment for MountainCar {
    type StateType = MountainCarState;
    type ActionType = MountainCarAction;
    type RewardType = ElemType;
    const MAX_STEPS: usize = 200;

    fn new(visualized: bool) -> Self {
        Self {
            gym_env: MountainCarEnv::new(if visualized {
                RenderMode::Human
            } else {
                RenderMode::None
            }),
        }
    }

    fn state(&self) -> Self::StateType {
        self.gym_env.state.into()
    }

    fn reset(&mut self) -> Snapshot<Self::StateType> {
        self.gym_env.reset(None, false, None);
        Snapshot::new(self.gym_env.state.into(), 0.0, false)
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
