use crate::base::Model;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub struct PPOOutput<B: Backend> {
    pub policies: Tensor<B, 2>,
    pub values: Tensor<B, 2>,
}

impl<B: Backend> PPOOutput<B> {
    pub fn new(policies: Tensor<B, 2>, values: Tensor<B, 2>) -> Self {
        Self { policies, values }
    }
}

pub trait PPOModel<B: Backend>: Model<B, Tensor<B, 2>, PPOOutput<B>, Tensor<B, 2>> {}
