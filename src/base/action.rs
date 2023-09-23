use std::fmt::Debug;

pub trait Action: Debug + Copy + Clone + Default {
    fn random() -> Self;
    fn enumerate() -> Vec<Self>;
}
