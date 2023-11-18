use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::fmt::Debug;

pub trait State: Debug + Copy + Clone {
    type Data;
    fn to_tensor<B: Backend>(&self) -> Tensor<B, 1>;

    fn size() -> usize;
}
