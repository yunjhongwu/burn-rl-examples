use burn::nn::loss::Reduction;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::marker::PhantomData;

pub struct SmoothL1Loss<B: Backend> {
    beta: f32,
    backend: PhantomData<B>,
}

impl<B: Backend> Default for SmoothL1Loss<B> {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl<B: Backend> SmoothL1Loss<B> {
    pub fn new(beta: f32) -> Self {
        Self {
            beta,
            backend: PhantomData,
        }
    }

    pub fn forward<const D: usize>(
        &self,
        logits: Tensor<B, D>,
        targets: Tensor<B, D>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let tensor = self.forward_no_reduction(logits, targets);
        match reduction {
            Reduction::Mean | Reduction::Auto => tensor.mean(),
            Reduction::Sum => tensor.sum(),
        }
    }

    pub fn forward_no_reduction<const D: usize>(
        &self,
        logits: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let abs_diff = logits.sub(targets).abs();
        let greater_than_beta = abs_diff.clone().greater_elem(self.beta);
        let l2_loss = abs_diff.clone().powf(2.0);
        let shifted_l1_loss = abs_diff.sub_scalar(0.5 * self.beta);

        l2_loss * (greater_than_beta.clone().bool_not().float())
            + shifted_l1_loss * greater_than_beta.float()
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::SmoothL1Loss;
    use burn::nn::loss::Reduction;
    use burn::tensor::{Data, Tensor};

    type TestBackend = burn_ndarray::NdArrayBackend<f32>;

    #[test]
    fn test_smooth_l1() {
        let logits = Tensor::<TestBackend, 2>::from_data(Data::from([[1.0, 2.0], [3.0, 4.0]]));

        let targets = Tensor::<TestBackend, 2>::from_data(Data::from([[1.5, 4.0], [3.0, 1.0]]));

        let smooth_l1 = SmoothL1Loss::new(2.0);
        let loss_no_reduction = smooth_l1.forward_no_reduction(logits.clone(), targets.clone());
        let loss = smooth_l1.forward(logits.clone(), targets.clone(), Reduction::Auto);
        let loss_sum = smooth_l1.forward(logits, targets, Reduction::Sum);

        assert_eq!(
            loss_no_reduction.into_data(),
            Data::from([[0.25, 4.0], [0.0, 2.0]])
        );
        assert_eq!(loss.into_data(), Data::from([1.5625]));
        assert_eq!(loss_sum.into_data(), Data::from([6.25]));
    }
}
