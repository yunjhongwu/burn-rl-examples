use std::fmt::Debug;

pub trait Action: Debug + Copy + Clone + From<u32> + Into<u32> {
    fn random() -> Self;
    fn enumerate() -> Vec<Self>;

    fn size() -> usize {
        Self::enumerate().len()
    }
}
