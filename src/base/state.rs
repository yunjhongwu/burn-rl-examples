use std::fmt::Debug;

pub trait State: Debug + Copy + Clone + Default {
    type Data;
    fn data(&self) -> &Self::Data;
}
