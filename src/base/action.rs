use std::fmt::Debug;

pub trait Action: Debug + Copy + Clone {
    fn random() -> Self;
    fn enumerate() -> Vec<Self>;
}
