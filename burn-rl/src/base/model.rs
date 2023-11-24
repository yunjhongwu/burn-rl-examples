use burn::tensor::backend::Backend;

pub trait Model<B: Backend, I, TrainingOutput, InferenceOutput = TrainingOutput>: Clone {
    fn forward(&self, input: I) -> TrainingOutput;

    fn infer(&self, input: I) -> InferenceOutput;
}
