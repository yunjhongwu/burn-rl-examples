use std::fmt::Debug;

pub trait State: Debug + Copy + Clone {
    type Data;
    fn data(&self) -> &Self::Data;
}
