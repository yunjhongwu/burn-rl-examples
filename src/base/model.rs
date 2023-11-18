use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub trait Model<B: Backend>: Clone {
    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D>;

    fn soft_update(this: &mut Self, that: &Self, tau: f64);
}
