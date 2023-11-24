use crate::base::ElemType;
use burn::grad_clipping::GradientClippingConfig;

pub struct DQNTrainingConfig {
    pub gamma: ElemType,
    pub tau: ElemType,
    pub learning_rate: ElemType,
    pub batch_size: usize,
    pub clip_grad: Option<GradientClippingConfig>,
}

impl Default for DQNTrainingConfig {
    fn default() -> Self {
        Self {
            gamma: 0.999,
            tau: 0.005,
            learning_rate: 0.001,
            batch_size: 32,
            clip_grad: Some(GradientClippingConfig::Value(100.0)),
        }
    }
}
