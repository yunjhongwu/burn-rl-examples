use std::fmt::Debug;

pub trait Action: Debug + Copy + Clone + Default + From<u32> {
    fn random() -> Self;
    fn enumerate() -> Vec<Self>;
}
