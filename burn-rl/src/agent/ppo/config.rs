use crate::base::ElemType;
use burn::grad_clipping::GradientClippingConfig;

pub struct PPOTrainingConfig {
    pub gamma: ElemType,
    pub lambda: ElemType,
    pub epsilon_clip: ElemType,
    pub critic_weight: ElemType,
    pub entropy_weight: ElemType,
    pub learning_rate: ElemType,
    pub epochs: usize,
    pub batch_size: usize,
    pub clip_grad: Option<GradientClippingConfig>,
}

impl Default for PPOTrainingConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            epsilon_clip: 0.2,
            critic_weight: 0.5,
            entropy_weight: 0.01,
            learning_rate: 0.001,
            epochs: 8,
            batch_size: 8,
            clip_grad: Some(GradientClippingConfig::Value(100.0)),
        }
    }
}
