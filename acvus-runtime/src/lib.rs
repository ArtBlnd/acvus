mod storage;
mod error;

pub trait Executable {
    
}

pub struct Module<S> {
    storage: S
}