use crate::base::{ElemType, Model};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub trait DQNModel<B: Backend>: Model<B, Tensor<B, 2>, Tensor<B, 2>> {
    fn soft_update(this: Self, that: &Self, tau: ElemType) -> Self;
}
