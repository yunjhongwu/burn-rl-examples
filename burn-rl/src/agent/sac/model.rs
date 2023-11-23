use crate::base::{ElemType, Model};
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub trait SACActor<B: Backend>: Model<B, Tensor<B, 2>, Tensor<B, 2>> {}

pub trait SACCritic<B: Backend>: Model<B, Tensor<B, 2>, Tensor<B, 2>> {
    fn soft_update(this: &mut Self, that: &Self, tau: ElemType);
}

#[derive(Module, Debug)]
pub struct SACTemperature<B: Backend> {
    temperature: Param<Tensor<B, 2>>,
}

impl<B: Backend> Default for SACTemperature<B> {
    fn default() -> Self {
        Self {
            temperature: Param::from(Tensor::zeros([1, 1])),
        }
    }
}

impl<B: Backend> SACTemperature<B> {
    pub fn forward(&self) -> Tensor<B, 2> {
        self.temperature.val()
    }
}
