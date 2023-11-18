use rand::{thread_rng, Rng};
use std::fmt::Debug;

pub trait Action: Debug + Copy + Clone + From<u32> + Into<u32> {
    fn random() -> Self {
        (thread_rng().gen_range(0..Self::size()) as u32).into()
    }
    fn enumerate() -> Vec<Self>;

    fn size() -> usize {
        Self::enumerate().len()
    }
}
