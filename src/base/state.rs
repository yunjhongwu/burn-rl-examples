use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::fmt::Debug;

pub trait State: Debug + Copy + Clone + Default {
    type Data;
    fn data<B: Backend>(&self) -> Tensor<B, 1>;

    fn size() -> usize;
}
