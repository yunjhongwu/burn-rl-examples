use burn::tensor::backend::Backend;

pub trait Model<B: Backend, I, O>: Clone {
    fn forward(&self, input: I) -> O;
}
