use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub trait Model<B: Backend> {
    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D>;
}
