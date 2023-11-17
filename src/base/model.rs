use burn::module::ADModule;
use burn::tensor::backend::ADBackend;
use burn::tensor::Tensor;

pub trait Model<B: ADBackend>: Clone + ADModule<B> {
    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D>;

    fn soft_update(&mut self, other: &Self, tau: f64);
}
